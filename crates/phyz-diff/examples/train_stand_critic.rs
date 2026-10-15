//! Push recovery over long episodes: short differentiable windows plus a
//! learned critic (the SHAC recipe).
//!
//! Backpropagating through hundreds of contact steps gives gradients that no
//! longer describe the loss (`train_stand`: exact to 150 steps, misleading
//! by 300). Here each update rolls every environment forward one window of
//! `STAND_W` steps (default 32) from wherever its episode is, and the policy
//! minimises the window's cost plus a critic's estimate of the cost-to-go
//! from the window's last state, backpropagated exactly through the window
//! and into the critic's input. The critic regresses n-step returns.
//! Episodes run `STAND_H` steps (default 600 = 0.6 s) and are scored over
//! their full length against PD.
//!
//! `cargo run --release -p phyz-diff --example train_stand_critic [ITERS]`

#[path = "train_stand.rs"]
#[allow(dead_code)]
mod stand;

use std::time::Instant;

use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::contact_step::ContactStep;
use stand::reach::{Adam, Lcg, Mlp, TORQUE};
use stand::{HIDDEN, OFFSET, SWEEPS, Stand, k1};

const GAMMA: f64 = 0.995;
const CRITIC_EPOCHS: usize = 16;
/// Polyak rate of the target critic that bootstraps the windows.
const TAU: f64 = 0.02;

fn main() {
    // episode length (also the cost scale in Stand::step_loss)
    if std::env::var_os("STAND_H").is_none() {
        unsafe { std::env::set_var("STAND_H", "600") };
    }
    let episode = stand::horizon();
    let envs_n: usize = std::env::var("STAND_ENVS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let window: usize = std::env::var("STAND_W")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(300);

    let (model, points) = k1();
    let mat = ContactMaterial {
        friction: 0.8,
        margin: 0.005,
        ..ContactMaterial::default()
    };
    let config = ContactSolverConfig::default();
    let step = ContactStep::build(&model, &points, &mat, &config, SWEEPS);
    let (ag, ao) = step.adjoint();
    let task = Stand::new(
        &model,
        step.compile(),
        ag.compile_many(&ao),
        step.nu,
        points.len(),
    );
    eprintln!("episodes of {episode} steps, windows of {window}, {envs_n} environments");

    let mut rng = Lcg(23);
    let mut pol = Mlp::new(task.n_obs(), HIDDEN, task.nu, &mut rng);
    let (w, _) = pol.offsets(2);
    let n = pol.params.len();
    pol.params[w..n].iter_mut().for_each(|p| *p = 0.0);
    let mut critic = Mlp::new(task.n_obs(), HIDDEN, 1, &mut rng);
    // a slow copy of the critic values the windows' last states, so the
    // policy chases a target that moves less than the critic it trains
    let mut target = Mlp::new(task.n_obs(), HIDDEN, 1, &mut rng);
    target.params.copy_from_slice(&critic.params);
    let mut best = (f64::INFINITY, pol.params.clone(), 0);

    let eval: Vec<Vec<f64>> = (0..32).map(|_| task.push(&mut rng)).collect();
    let zero = |_: &[f64], u: &mut [f64]| u.iter_mut().for_each(|x| *x = 0.0);
    let (base, base_drop) = task.evaluate(&eval, &zero);
    eprintln!(
        "PD over {episode} steps: loss {base:.4}, trunk drop {:.2} mm",
        base_drop * 1e3
    );

    let mut adam_p = Adam::new(pol.params.len(), 2e-4);
    let mut adam_v = Adam::new(critic.params.len(), 1e-3);
    // persistent environments: state and step within the episode
    let mut envs: Vec<(Vec<f64>, usize)> = (0..envs_n).map(|_| (task.push(&mut rng), 0)).collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let t = Instant::now();
    for it in 0..iters {
        // one window per environment, in parallel
        let chunk = envs_n.div_ceil(threads);
        let results: Vec<Window> = std::thread::scope(|sc| {
            let hs: Vec<_> = envs
                .chunks(chunk)
                .map(|c| {
                    let (task, pol, critic) = (&task, &pol, &target);
                    sc.spawn(move || {
                        c.iter()
                            .map(|(x, k)| {
                                window_grad(task, pol, critic, x, window.min(episode - k))
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let mut grad = vec![0.0; pol.params.len()];
        let mut cost = 0.0;
        let mut samples = Vec::new();
        for (r, env) in results.into_iter().zip(envs.iter_mut()) {
            grad.iter_mut()
                .zip(&r.grad)
                .for_each(|(a, b)| *a += b / envs_n as f64);
            cost += r.cost / envs_n as f64;
            samples.extend(r.samples);
            env.1 += r.steps;
            env.0 = r.last;
            if env.1 >= episode || env.0[5] < task.z0 - 0.15 {
                *env = (task.push(&mut rng), 0);
            }
        }
        // a step that blew up (a NaN anywhere in the window) teaches nothing
        // cosine decay of the policy rate: late updates refine instead of
        // chasing the critic
        adam_p.lr = 2e-4 * 0.5 * (1.0 + (std::f64::consts::PI * it as f64 / iters as f64).cos());
        if grad.iter().all(|g| g.is_finite()) {
            adam_p.step(&mut pol.params, &grad);
        }
        // critic: regress the n-step returns
        for _ in 0..CRITIC_EPOCHS {
            let mut g = vec![0.0; critic.params.len()];
            for (obs, target) in &samples {
                let c = critic.forward(obs);
                let mut gout = [2.0 * (c.out[0] - target) / samples.len() as f64];
                critic.backward(obs, &c, &mut gout, &mut g);
            }
            adam_v.step(&mut critic.params, &g);
        }
        for (t, c) in target.params.iter_mut().zip(&critic.params) {
            *t += TAU * (c - *t);
        }
        if it % 25 == 0 || it + 1 == iters {
            let (l, drop) = task.evaluate(&eval, &|o, u| act(&pol, o, u));
            if l < best.0 {
                best = (l, pol.params.clone(), it);
            }
            eprintln!(
                "iter {it:4}  window cost {cost:.4}  eval loss {l:.4} (PD {base:.4})  trunk drop {:.2} mm (PD {:.2})  {:.0}s",
                drop * 1e3,
                base_drop * 1e3,
                t.elapsed().as_secs_f64()
            );
        }
    }
    pol.params = best.1;
    // the kept policy, scored on pushes it was not selected on
    let fresh: Vec<Vec<f64>> = (0..32).map(|_| task.push(&mut rng)).collect();
    let (base, base_drop) = task.evaluate(&fresh, &zero);
    let (l, drop) = task.evaluate(&fresh, &|o, u| act(&pol, o, u));
    println!(
        "best policy (update {}), fresh pushes: push recovery over {episode}-step episodes, {iters} updates of {window}-step windows x {envs_n} envs: loss {l:.4} vs PD {base:.4} ({:+.1}%), trunk drop {:.2} vs {:.2} mm",
        best.2,
        100.0 * (l / base - 1.0),
        drop * 1e3,
        base_drop * 1e3
    );
}

/// Joint-target offsets, bounded: `OFFSET tanh(z)` (a linear head let one
/// bad update command targets the simulation cannot integrate).
fn act(pol: &Mlp, obs: &[f64], u: &mut [f64]) {
    let c = pol.forward(obs);
    for (o, a) in u.iter_mut().zip(&c.out) {
        *o = OFFSET * (a / TORQUE).tanh();
    }
}

struct Window {
    grad: Vec<f64>,
    cost: f64,
    /// `(observation, n-step return)` for the critic.
    samples: Vec<(Vec<f64>, f64)>,
    last: Vec<f64>,
    steps: usize,
}

/// The critic's value of a state: the cost-to-go estimate.
fn value(task: &Stand, critic: &Mlp, x: &[f64]) -> f64 {
    if no_critic() {
        return 0.0;
    }
    critic.forward(&task.obs(x)).out[0]
}

/// `STAND_NO_CRITIC`: plain truncated backprop (the control).
fn no_critic() -> bool {
    std::env::var_os("STAND_NO_CRITIC").is_some()
}

/// One window of `h` steps from `x0`: the policy gradient of
/// `Σ γ^t c_t + γ^h V(s_h)`, backpropagated through the steps and the
/// critic's input.
fn window_grad(task: &Stand, pol: &Mlp, critic: &Mlp, x0: &[f64], h: usize) -> Window {
    let (nq, nv, nu, nf) = (task.nq, task.nv, task.nu, task.nf);
    let (ns, ni) = (nq + nv, nq + nv + nu + nf);
    let idx = task.obs_index();
    let mut xs = Vec::with_capacity(h + 1);
    let mut caches = Vec::with_capacity(h);
    let mut costs = Vec::with_capacity(h);
    let mut x = x0.to_vec();
    let mut out = vec![0.0; ns + nf];
    for _ in 0..h {
        let obs = task.obs(&x);
        let c = pol.forward(&obs);
        for i in 0..nu {
            x[ns + i] = task.rest[ns + i] + OFFSET * (c.out[i] / TORQUE).tanh();
        }
        (task.fwd)(&x, &mut out);
        xs.push(x.clone());
        caches.push((obs, c));
        x[..ns].copy_from_slice(&out[..ns]);
        x[ns + nu..].copy_from_slice(&out[ns..]);
        costs.push(task.step_loss(&x, None));
    }
    xs.push(x.clone());
    let v_end = value(task, critic, &x);
    let disc = |t: usize| GAMMA.powi(t as i32);
    let objective: f64 = costs
        .iter()
        .enumerate()
        .map(|(t, c)| disc(t) * c)
        .sum::<f64>()
        + disc(h) * v_end;
    // n-step returns for the critic, from every state in the window
    let mut samples = Vec::with_capacity(h);
    let mut ret = v_end;
    for t in (0..h).rev() {
        ret = costs[t] + GAMMA * ret;
        samples.push((task.obs(&xs[t]), ret));
    }
    // reverse sweep; λ over the step output (q', v', f), seeded by the
    // critic's gradient at the last state
    let mut grad = vec![0.0; pol.params.len()];
    let mut lam = vec![0.0; ns + nf];
    if !no_critic() {
        let obs = task.obs(&x);
        let c = critic.forward(&obs);
        let mut scratch = vec![0.0; critic.params.len()];
        let gobs = critic.backward(&obs, &c, &mut [disc(h)], &mut scratch);
        for (k, &i) in idx.iter().enumerate() {
            lam[i] += gobs[k];
        }
    }
    let mut xin = vec![0.0; ni + ns + nf];
    let mut back = vec![0.0; ni];
    let mut gx = vec![0.0; ni];
    for t in (0..h).rev() {
        gx.iter_mut().for_each(|g| *g = 0.0);
        task.step_loss(&xs[t + 1], Some(&mut gx));
        for i in 0..ns {
            lam[i] += disc(t) * gx[i];
        }
        xin[..ni].copy_from_slice(&xs[t]);
        xin[ni..].copy_from_slice(&lam);
        (task.back)(&xin, &mut back);
        let (obs, c) = &caches[t];
        let mut gout: Vec<f64> = back[ns..ns + nu]
            .iter()
            .zip(&c.out)
            .map(|(g, o)| {
                let th = (o / TORQUE).tanh();
                g * OFFSET / TORQUE * (1.0 - th * th)
            })
            .collect();
        let gobs = pol.backward(obs, c, &mut gout, &mut grad);
        lam[..ns].copy_from_slice(&back[..ns]);
        lam[ns..].copy_from_slice(&back[ns + nu..]);
        for (k, &i) in idx.iter().enumerate() {
            lam[i] += gobs[k];
        }
    }
    Window {
        grad,
        cost: objective,
        samples,
        last: x,
        steps: h,
    }
}
