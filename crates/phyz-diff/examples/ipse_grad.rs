//! Stage A of training through the fitted K1 plant: an MLP leg policy,
//! distilled from ipse's CEM policy and then fine-tuned by backpropagating
//! through the traced contact step.
//!
//! The plant is the one that reproduces the hardware (2026-10-15): `hold`'s
//! flat gains (`IPSE_KP_LEG` etc., defaulted here to kp 90 / kd 6, upper
//! 5 / 1), the policy queried every 5 ms and applied one query later, its
//! offsets clamped by `CLAMP · tanh` and passed through the K1's standing
//! 8 Hz mode as a second-order stage (`IPSE_ACT_RES_HZ`, `IPSE_ACT_ZETA`,
//! default 8 / 0.15; 0 Hz turns it off).
//!
//! 1. **Distill.** Roll the teacher (a CEM `LinearPolicy`, e.g. s11) through
//!    random shoves in this plant, regress the MLP on its actions.
//! 2. **Fine-tune.** Batches of random shoves (sagittal both ways, lateral);
//!    the loss keeps the trunk tall, level and still, pays for action jerk,
//!    and the gradient runs back through the step adjoint, the 8 Hz stage
//!    and the hold-and-delay of each query, truncated to `GRAD_WINDOW`
//!    steps (short-horizon BPTT; no critic yet).
//! 3. **Evaluate** teacher and student: largest shove survived forward,
//!    backward, lateral, and quiet-stance peak gyro.
//!
//! `cargo run --release -p phyz-diff --example ipse_grad RIG.txt TEACHER.policy [DISTILL_ITERS] [TUNE_ITERS]`

#[path = "common/ipse_rig.rs"]
mod ipse_rig;
#[path = "train_reach.rs"]
#[allow(dead_code)]
mod reach;

use phyz_contact::ContactSolverConfig;
use phyz_diff::contact_step::ContactStep;
use reach::{Adam, Lcg, Mlp, TORQUE};
use std::time::Instant;
use tang_expr::compile::CompiledMany;

const EVERY: usize = 5;
const CLAMP: f64 = 0.15;
const HIDDEN: usize = 64;
const SETTLE: usize = 1500;
/// Steps per training episode: 0.3 s fade-in, 0.1 s quiet, the 20 ms shove,
/// then recovery.
const EPISODE: usize = 1500;
const SHOVE_AT: usize = 400;
/// Steps over which the policy's authority fades in (hold ramps its clamp).
const RAMP: usize = 300;

fn fade(k: usize) -> f64 {
    (k as f64 / RAMP as f64).min(1.0)
}
const JERK: f64 = 2.0;

fn env(k: &str, d: f64) -> f64 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

struct Plant {
    rig: ipse_rig::Rig,
    model: phyz_model::Model,
    fwd: CompiledMany,
    back: CompiledMany,
    nq: usize,
    nv: usize,
    nt: usize,
    nf: usize,
    /// Settled standing input `[q, v, targets, wrench, warm]`.
    rest: Vec<f64>,
    z0: f64,
    /// PD slot of each of the ten policy DOFs.
    slot: Vec<usize>,
    cem_dofs: Vec<usize>,
    wn: f64,
    zeta: f64,
    dt: f64,
}

impl Plant {
    fn new(path: &str) -> Self {
        for (k, v) in [
            ("IPSE_KP_LEG", "90"),
            ("IPSE_KD_LEG", "6"),
            ("IPSE_KP_UPPER", "5"),
            ("IPSE_KD_UPPER", "1"),
        ] {
            if std::env::var(k).is_err() {
                // SAFETY: single-threaded, before any thread is spawned
                unsafe { std::env::set_var(k, v) };
            }
        }
        let rig = ipse_rig::Rig::read(path);
        let model = rig.model();
        let pd = rig.pd();
        let t = Instant::now();
        let step = ContactStep::build_pd(
            &model,
            &rig.points(),
            &rig.material(),
            &ContactSolverConfig::default(),
            5,
            &pd,
        );
        let (ag, ao) = step.adjoint();
        let (fwd, back) = (step.compile(), ag.compile_many(&ao));
        eprintln!(
            "plant: step {} nodes, adjoint {} nodes, {:.1}s",
            step.live_nodes(),
            ag.live_set(&ao).len(),
            t.elapsed().as_secs_f64()
        );
        let (nq, nv, nt) = (model.nq, model.nv, step.nu - 6);
        let nf = 3 * rig.points().len();
        let cem_dofs = ipse_rig::cem_dofs(&model);
        let slot = cem_dofs
            .iter()
            .map(|d| pd.dofs.iter().position(|p| p == d).unwrap())
            .collect();
        let mut x: Vec<f64> = rig.q0.iter().chain(&rig.v0).copied().collect();
        x.extend(pd.dofs.iter().map(|&d| rig.q_target[d]));
        x.extend([0.0; 6]);
        x.extend(std::iter::repeat_n(0.0, nf));
        let mut out = vec![0.0; nq + nv + nf];
        let ns = nq + nv;
        for _ in 0..SETTLE {
            fwd(&x, &mut out);
            x[..ns].copy_from_slice(&out[..ns]);
            x[ns + nt + 6..].copy_from_slice(&out[ns..]);
        }
        let z0 = x[5];
        let dt = rig.dt;
        Self {
            rig,
            model,
            fwd,
            back,
            nq,
            nv,
            nt,
            nf,
            rest: x,
            z0,
            slot,
            cem_dofs,
            wn: std::f64::consts::TAU * env("IPSE_ACT_RES_HZ", 8.0),
            zeta: env("IPSE_ACT_ZETA", 0.15),
            dt,
        }
    }

    fn ns(&self) -> usize {
        self.nq + self.nv
    }

    fn n_obs(&self) -> usize {
        self.nq - 2 + self.nv
    }

    /// Pose error (not base x, y) and velocity.
    fn obs(&self, x: &[f64]) -> Vec<f64> {
        (0..self.nq)
            .filter(|&i| i != 3 && i != 4)
            .map(|i| x[i] - self.rest[i])
            .chain(x[self.nq..self.ns()].iter().copied())
            .collect()
    }

    fn obs_index(&self) -> Vec<usize> {
        (0..self.nq)
            .filter(|&i| i != 3 && i != 4)
            .chain(self.nq..self.ns())
            .collect()
    }

    /// Per-step loss on the state after the step (scaled per episode) and
    /// its gradient into `g` (state part).
    fn step_loss(&self, x: &[f64], g: Option<&mut [f64]>) -> f64 {
        let nq = self.nq;
        let s = 1.0 / EPISODE as f64;
        let dz = x[5] - self.z0;
        let tilt = x[0] * x[0] + x[1] * x[1];
        let vb: f64 = x[nq..nq + 3].iter().map(|v| v * v).sum();
        let vj: f64 = x[nq + 6..self.ns()].iter().map(|v| v * v).sum();
        if let Some(g) = g {
            g[5] += s * 200.0 * dz;
            g[0] += s * 20.0 * x[0];
            g[1] += s * 20.0 * x[1];
            for i in 0..3 {
                g[nq + i] += s * 2.0 * x[nq + i];
            }
            for i in nq + 6..self.ns() {
                g[i] += s * 0.02 * x[i];
            }
        }
        s * (100.0 * dz * dz + 10.0 * tilt + vb + 0.01 * vj)
    }

    fn fell(&self, x: &[f64]) -> bool {
        !x.iter().all(|v| v.is_finite()) || x[5] < 0.35
    }
}

/// A policy the plant can query: observation of the full input `x`, ten
/// clamped offsets out.
trait Act: Sync {
    fn act(&self, p: &Plant, x: &[f64]) -> [f64; 10];
}

struct Teacher(Vec<f64>);

impl Act for Teacher {
    fn act(&self, p: &Plant, x: &[f64]) -> [f64; 10] {
        let o = ipse_rig::observe_legs(
            &p.model,
            &p.rig.q_target,
            &x[..p.nq],
            &x[p.nq..p.ns()],
            &p.cem_dofs,
        );
        let mut a = [0.0; 10];
        for (o, v) in a.iter_mut().zip(ipse_rig::cem_action(&self.0, &o)) {
            *o = v.clamp(-CLAMP, CLAMP);
        }
        a
    }
}

struct Student(Mlp);

impl Act for Student {
    fn act(&self, p: &Plant, x: &[f64]) -> [f64; 10] {
        let c = self.0.forward(&p.obs(x));
        let mut a = [0.0; 10];
        for (o, v) in a.iter_mut().zip(&c.out) {
            *o = CLAMP * (v / TORQUE).tanh();
        }
        a
    }
}

/// Wrench for a shove of `imp` N·s over 20 steps starting at `at`.
fn shove(k: usize, at: usize, imp: [f64; 3], w: &mut [f64]) {
    w.iter_mut().for_each(|v| *v = 0.0);
    if (at..at + 20).contains(&k) {
        for a in 0..3 {
            w[3 + a] = imp[a] / 0.02;
        }
    }
}

/// One step of the 8 Hz stage, `(e, ev) <- f(e, ev, a)`.
fn resonate(p: &Plant, e: &mut [f64; 10], ev: &mut [f64; 10], a: &[f64; 10]) {
    if p.wn <= 0.0 {
        *e = *a;
        return;
    }
    for i in 0..10 {
        ev[i] += p.dt * (p.wn * p.wn * (a[i] - e[i]) - 2.0 * p.zeta * p.wn * ev[i]);
        e[i] += p.dt * ev[i];
    }
}

/// Roll `pol` from rest with a shove; `on_query(x, a)` sees every query.
/// Returns (survived, peak |gyro| before the shove).
fn rollout(
    p: &Plant,
    pol: &dyn Act,
    imp: [f64; 3],
    steps: usize,
    shove_at: usize,
    on_query: &mut dyn FnMut(&[f64], &[f64; 10]),
) -> (bool, f64) {
    let (ns, nt) = (p.ns(), p.nt);
    let mut x = p.rest.clone();
    let mut out = vec![0.0; ns + p.nf];
    let (mut e, mut ev, mut held, mut next) = ([0.0; 10], [0.0; 10], [0.0; 10], [0.0; 10]);
    let mut peak = 0.0f64;
    for k in 0..steps {
        if k % EVERY == 0 {
            held = next;
            next = pol.act(p, &x);
            on_query(&x, &next);
        }
        let faded = held.map(|a| a * fade(k));
        resonate(p, &mut e, &mut ev, &faded);
        for (i, &s) in p.slot.iter().enumerate() {
            x[ns + s] = p.rest[ns + s] + e[i];
        }
        shove(k, shove_at, imp, &mut x[ns + nt..ns + nt + 6]);
        (p.fwd)(&x, &mut out);
        x[..ns].copy_from_slice(&out[..ns]);
        x[ns + nt + 6..].copy_from_slice(&out[ns..]);
        if k < shove_at {
            peak = x[p.nq..p.nq + 3].iter().fold(peak, |m, v| m.max(v.abs()));
        }
        if p.fell(&x) {
            return (false, peak);
        }
    }
    (true, peak)
}

fn random_shove(rng: &mut Lcg) -> [f64; 3] {
    // sagittal ±(6..18), lateral ±(4..10)
    let u = 0.5 * (rng.next() + 1.0);
    let s = if rng.next() > 0.0 { 1.0 } else { -1.0 };
    if rng.next() > -0.4 {
        [s * (6.0 + 12.0 * u), 0.0, 0.0]
    } else {
        [0.0, s * (4.0 + 6.0 * u), 0.0]
    }
}

/// One episode's loss and its gradient into `grad`, through the step
/// adjoint, the 8 Hz stage and each query's hold-and-delay.
/// (loss, gradient, critic samples) from one worker's episodes.
type Part = (f64, Vec<f64>, Vec<(Vec<f64>, f64)>);

/// Critic input: the observation plus the fraction of the episode elapsed.
fn critic_in(p: &Plant, x: &[f64], k: usize) -> Vec<f64> {
    let mut o = p.obs(x);
    o.push(k as f64 / EPISODE as f64);
    o
}

/// One episode's loss and its gradient into `grad`, through the step
/// adjoint, the 8 Hz stage and each query's hold-and-delay. With a
/// `critic`, each truncated window ends at V(x) instead of zero (SHAC), and
/// `samples` collects (critic input, observed cost-to-go) for fitting it.
fn episode_grad(
    p: &Plant,
    pol: &Mlp,
    critic: Option<&Mlp>,
    imp: [f64; 3],
    window: usize,
    grad: &mut [f64],
    samples: &mut Vec<(Vec<f64>, f64)>,
) -> f64 {
    let (ns, nt, nf) = (p.ns(), p.nt, p.nf);
    let ni = ns + nt + 6 + nf;
    let idx = p.obs_index();
    let mut x = p.rest.clone();
    let mut out = vec![0.0; ns + nf];
    let (mut e, mut ev) = ([0.0; 10], [0.0; 10]);
    // per query: obs, mlp cache, tanh'd action
    let mut queries: Vec<(Vec<f64>, reach::Cache, [f64; 10])> = Vec::new();
    let mut xs: Vec<Vec<f64>> = Vec::with_capacity(EPISODE + 1);
    let mut evs: Vec<[f64; 10]> = Vec::with_capacity(EPISODE);
    let mut loss = 0.0;
    let mut steps = EPISODE;
    // cost incurred at each step (after it), for the critic's targets
    let mut cost: Vec<f64> = Vec::with_capacity(EPISODE);
    for k in 0..EPISODE {
        let l0 = loss;
        if k % EVERY == 0 {
            let o = p.obs(&x);
            let c = pol.forward(&o);
            let mut a = [0.0; 10];
            for (o, v) in a.iter_mut().zip(&c.out) {
                *o = CLAMP * (v / TORQUE).tanh();
            }
            if let Some((_, _, pa)) = queries.last() {
                loss += JERK / (EPISODE / EVERY) as f64
                    * a.iter().zip(pa).map(|(x, y)| (x - y).powi(2)).sum::<f64>();
            }
            queries.push((o, c, a));
        }
        // the action in force: the previous query's
        let held = if k >= EVERY {
            queries[k / EVERY - 1].2
        } else {
            [0.0; 10]
        };
        let faded = held.map(|a| a * fade(k));
        resonate(p, &mut e, &mut ev, &faded);
        evs.push(ev);
        for (i, &s) in p.slot.iter().enumerate() {
            x[ns + s] = p.rest[ns + s] + e[i];
        }
        shove(k, SHOVE_AT, imp, &mut x[ns + nt..ns + nt + 6]);
        (p.fwd)(&x, &mut out);
        xs.push(x.clone());
        x[..ns].copy_from_slice(&out[..ns]);
        x[ns + nt + 6..].copy_from_slice(&out[ns..]);
        loss += p.step_loss(&x, None);
        if p.fell(&x) {
            // the rest of the episode at the fall's cost
            loss += 1.0 * (EPISODE - k) as f64 / EPISODE as f64;
            steps = k + 1;
            cost.push(loss - l0);
            break;
        }
        cost.push(loss - l0);
    }
    xs.push(x);
    // cost-to-go from the state after step k
    let mut togo = 0.0;
    for k in (0..steps).rev() {
        if k % 10 == 9 {
            samples.push((critic_in(p, &xs[k + 1], k + 1), togo));
        }
        togo += cost[k];
    }
    let nqry = queries.len();
    // adjoints: step output (q', v', f), the stage (e, ev), each query's action
    let mut lam = vec![0.0; ns + nf];
    let (mut le, mut lev) = ([0.0; 10], [0.0; 10]);
    let mut la = vec![[0.0f64; 10]; nqry];
    // jerk terms
    for j in 1..nqry {
        let s = 2.0 * JERK / (EPISODE / EVERY) as f64;
        let (prev, cur) = la.split_at_mut(j);
        for (i, (a, b)) in cur[0].iter_mut().zip(prev[j - 1].iter_mut()).enumerate() {
            let d = queries[j].2[i] - queries[j - 1].2[i];
            *a += s * d;
            *b -= s * d;
        }
    }
    let mut xin = vec![0.0; ni + ns + nf];
    let mut back = vec![0.0; ni];
    let mut gx = vec![0.0; ns];
    let h = p.dt;
    for k in (0..steps).rev() {
        if (k + 1) % window == 0 {
            // truncate: the next window's state is treated as given, its
            // future summarized by the critic's value there
            lam.iter_mut().for_each(|v| *v = 0.0);
            le = [0.0; 10];
            lev = [0.0; 10];
            if let Some(cr) = critic
                && k + 1 < steps
            {
                let ci = critic_in(p, &xs[k + 1], k + 1);
                let c = cr.forward(&ci);
                let mut gout = vec![1.0 / TORQUE];
                let mut scratch = vec![0.0; cr.params.len()];
                let gin = cr.backward(&ci, &c, &mut gout, &mut scratch);
                for (m, &i) in idx.iter().enumerate() {
                    lam[i] += gin[m];
                }
            }
        }
        gx.iter_mut().for_each(|g| *g = 0.0);
        p.step_loss(&xs[k + 1], Some(&mut gx));
        for i in 0..ns {
            lam[i] += gx[i];
        }
        xin[..ni].copy_from_slice(&xs[k]);
        xin[ni..].copy_from_slice(&lam);
        (p.back)(&xin, &mut back);
        // targets -> e' (after this step's stage update)
        let mut le2 = le;
        for (i, &s) in p.slot.iter().enumerate() {
            le2[i] += back[ns + s];
        }
        // reverse the stage: e' = e + h ev', ev' = ev + h(wn²(a - e) - 2ζwn ev)
        let held_q = if k >= EVERY {
            Some(k / EVERY - 1)
        } else {
            None
        };
        if p.wn > 0.0 {
            let (wn, z) = (p.wn, p.zeta);
            for i in 0..10 {
                let lv = lev[i] + h * le2[i];
                le[i] = le2[i] - lv * h * wn * wn;
                lev[i] = lv * (1.0 - 2.0 * h * z * wn);
                if let Some(q) = held_q {
                    la[q][i] += lv * h * wn * wn * fade(k);
                }
            }
        } else if let Some(q) = held_q {
            for i in 0..10 {
                la[q][i] += le2[i] * fade(k);
            }
            le = [0.0; 10];
        }
        let _ = &evs;
        lam[..ns].copy_from_slice(&back[..ns]);
        lam[ns..].copy_from_slice(&back[ns + nt + 6..]);
        if k % EVERY == 0 {
            let (o, c, a) = &queries[k / EVERY];
            let mut gout: Vec<f64> = (0..10)
                .map(|i| {
                    let t = a[i] / CLAMP;
                    la[k / EVERY][i] * CLAMP * (1.0 - t * t) / TORQUE
                })
                .collect();
            let gobs = pol.backward(o, c, &mut gout, grad);
            for (m, &i) in idx.iter().enumerate() {
                lam[i] += gobs[m];
            }
        }
    }
    loss
}

/// Regress the critic (`out / TORQUE`) on cost-to-go samples.
fn fit_critic(
    cr: &mut Mlp,
    adam: &mut Adam,
    data: &[(Vec<f64>, f64)],
    iters: usize,
    rng: &mut Lcg,
) -> f64 {
    let mut last = 0.0;
    for _ in 0..iters {
        let mut g = vec![0.0; cr.params.len()];
        let mut l = 0.0;
        for _ in 0..128 {
            let (x, y) = &data[((rng.next() + 1.0) * 0.5 * (data.len() - 1) as f64) as usize];
            let c = cr.forward(x);
            let e = c.out[0] / TORQUE - y;
            l += e * e;
            let mut gout = vec![2.0 * e / TORQUE / 128.0];
            cr.backward(x, &c, &mut gout, &mut g);
        }
        adam.step(&mut cr.params, &g);
        last = l / 128.0;
    }
    last
}

fn eval(p: &Plant, pol: &dyn Act, name: &str) {
    let levels: Vec<f64> = (2..=12).map(|k| 2.0 * k as f64).collect();
    let dirs = [
        ("fwd", [1.0, 0.0]),
        ("back", [-1.0, 0.0]),
        ("lat", [0.0, 1.0]),
    ];
    let jobs: Vec<(usize, f64)> = (0..3)
        .flat_map(|d| levels.iter().map(move |&n| (d, n)))
        .collect();
    let res: Vec<(usize, f64, bool)> = std::thread::scope(|sc| {
        let hs: Vec<_> = jobs
            .iter()
            .map(|&(d, n)| {
                let imp = [n * dirs[d].1[0], n * dirs[d].1[1], 0.0];
                sc.spawn(move || (d, n, rollout(p, pol, imp, 2400, 600, &mut |_, _| {}).0))
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (_, quiet) = rollout(p, pol, [0.0; 3], 2000, 2000, &mut |_, _| {});
    // `quiet` includes the fade-in: hold's own engage
    let best = |d: usize| {
        let mut last = 0.0;
        for &n in &levels {
            if !res.iter().any(|r| r.0 == d && r.1 == n && r.2) {
                break;
            }
            last = n;
        }
        last
    };
    println!(
        "{name:>8}: survives fwd {:>2.0}  back {:>2.0}  lat {:>2.0} N·s;  quiet peak gyro {quiet:.2} rad/s",
        best(0),
        best(1),
        best(2)
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "ipse_grad RIG.txt TEACHER.policy [DISTILL_ITERS] [TUNE_ITERS]";
    let p = Plant::new(&args.next().expect(usage));
    let teacher = Teacher(ipse_rig::read_cem(&args.next().expect(usage)));
    let distill_iters: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(3000);
    let tune_iters: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(100);
    let window = env("GRAD_WINDOW", 100.0) as usize;
    let batch = env("GRAD_BATCH", 32.0) as usize;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    eval(&p, &teacher, "teacher");

    // 1. distill: (obs, teacher action) along the teacher's own shoved rollouts
    let t = Instant::now();
    let mut rng = Lcg(7);
    let shoves: Vec<[f64; 3]> = (0..96).map(|_| random_shove(&mut rng)).collect();
    let data: Vec<(Vec<f64>, [f64; 10])> = std::thread::scope(|sc| {
        let hs: Vec<_> = shoves
            .chunks(shoves.len().div_ceil(threads))
            .map(|c| {
                let (p, teacher) = (&p, &teacher);
                sc.spawn(move || {
                    let mut d = Vec::new();
                    for &imp in c {
                        rollout(p, teacher, imp, EPISODE, SHOVE_AT, &mut |x, a| {
                            d.push((p.obs(x), *a))
                        });
                    }
                    d
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let mut mlp = Mlp::new(p.n_obs(), HIDDEN, 10, &mut rng);
    let mut adam = Adam::new(mlp.params.len(), 1e-3);
    for it in 0..distill_iters {
        let mut grad = vec![0.0; mlp.params.len()];
        let mut l = 0.0;
        for _ in 0..256 {
            let (o, a) = &data[((rng.next() + 1.0) * 0.5 * (data.len() - 1) as f64) as usize];
            let c = mlp.forward(o);
            let mut gout: Vec<f64> = (0..10)
                .map(|i| {
                    let y = CLAMP * (c.out[i] / TORQUE).tanh();
                    let t = y / CLAMP;
                    l += (y - a[i]).powi(2);
                    2.0 * (y - a[i]) * CLAMP * (1.0 - t * t) / TORQUE / 256.0
                })
                .collect();
            mlp.backward(o, &c, &mut gout, &mut grad);
        }
        adam.step(&mut mlp.params, &grad);
        if it % 500 == 0 || it + 1 == distill_iters {
            eprintln!(
                "distill {it:5}: action rms err {:.4} rad",
                (l / 2560.0).sqrt()
            );
        }
    }
    eprintln!(
        "distilled on {} samples in {:.0}s",
        data.len(),
        t.elapsed().as_secs_f64()
    );
    if std::env::var("GRAD_CHECK").is_ok() {
        // the full chained gradient against central differences, one shove
        let imp = [12.0, 0.0, 0.0];
        let mut g = vec![0.0; mlp.params.len()];
        episode_grad(&p, &mlp, None, imp, window, &mut g, &mut Vec::new());
        let n = mlp.params.len();
        for k in [0, 777, n / 2, n - 300, n - 15, n - 1] {
            let p0 = mlp.params[k];
            let mut fd = Vec::new();
            for h in [1e-4, 1e-5] {
                let mut s = vec![0.0; n];
                mlp.params[k] = p0 + h;
                let lp = episode_grad(&p, &mlp, None, imp, usize::MAX, &mut s, &mut Vec::new());
                mlp.params[k] = p0 - h;
                let lm = episode_grad(&p, &mlp, None, imp, usize::MAX, &mut s, &mut Vec::new());
                fd.push((lp - lm) / (2.0 * h));
            }
            mlp.params[k] = p0;
            println!(
                "param {k:5}: backprop {:+.5e}  FD 1e-4 {:+.5e}  1e-5 {:+.5e}",
                g[k], fd[0], fd[1]
            );
        }
        return;
    }
    let mut student = Student(mlp);
    eval(&p, &student, "distill");

    // 2. fine-tune through the plant
    let mut adam = Adam::new(student.0.params.len(), env("GRAD_LR", 3e-4));
    let use_critic = env("GRAD_CRITIC", 1.0) > 0.0;
    let mut critic = Mlp::new(p.n_obs() + 1, HIDDEN, 1, &mut rng);
    let mut cadam = Adam::new(critic.params.len(), 1e-3);
    let mut replay: Vec<(Vec<f64>, f64)> = Vec::new();
    let t = Instant::now();
    if use_critic {
        // pre-train on the distilled student's own episodes
        let imps: Vec<[f64; 3]> = (0..64).map(|_| random_shove(&mut rng)).collect();
        let pol = &student.0;
        let p = &p;
        replay = std::thread::scope(|sc| {
            let hs: Vec<_> = imps
                .chunks(64usize.div_ceil(threads))
                .map(|c| {
                    sc.spawn(move || {
                        let mut smp = Vec::new();
                        let mut g = vec![0.0; pol.params.len()];
                        for &i in c {
                            episode_grad(p, pol, None, i, usize::MAX, &mut g, &mut smp);
                        }
                        smp
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let l = fit_critic(&mut critic, &mut cadam, &replay, 3000, &mut rng);
        let var = {
            let m = replay.iter().map(|s| s.1).sum::<f64>() / replay.len() as f64;
            replay.iter().map(|s| (s.1 - m).powi(2)).sum::<f64>() / replay.len() as f64
        };
        eprintln!(
            "critic pre-trained on {} samples: mse {l:.4} (target variance {var:.4}), {:.0}s",
            replay.len(),
            t.elapsed().as_secs_f64()
        );
    }
    for it in 0..tune_iters {
        let imps: Vec<[f64; 3]> = (0..batch).map(|_| random_shove(&mut rng)).collect();
        let (pol, p) = (&student.0, &p);
        let cr = use_critic.then_some(&critic);
        let parts: Vec<Part> = std::thread::scope(|sc| {
            let hs: Vec<_> = imps
                .chunks(batch.div_ceil(threads))
                .map(|c| {
                    sc.spawn(move || {
                        let mut g = vec![0.0; pol.params.len()];
                        let mut smp = Vec::new();
                        let l: f64 = c
                            .iter()
                            .map(|&i| episode_grad(p, pol, cr, i, window, &mut g, &mut smp))
                            .sum();
                        (l, g, smp)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut grad = vec![0.0; student.0.params.len()];
        let mut loss = 0.0;
        for (l, g, smp) in parts {
            replay.extend(smp);
            loss += l;
            grad.iter_mut()
                .zip(g)
                .for_each(|(a, b)| *a += b / batch as f64);
        }
        adam.step(&mut student.0.params, &grad);
        let mut closs = 0.0;
        if use_critic {
            let keep = 60_000;
            if replay.len() > keep {
                replay.drain(..replay.len() - keep);
            }
            closs = fit_critic(&mut critic, &mut cadam, &replay, 100, &mut rng);
        }
        if it % 10 == 0 || it + 1 == tune_iters {
            eprintln!("  critic mse {closs:.4}");
            eprintln!(
                "tune {it:4}: batch loss {:.4}  |g| {:.2e}  {:.0}s",
                loss / batch as f64,
                grad.iter().map(|g| g * g).sum::<f64>().sqrt(),
                t.elapsed().as_secs_f64()
            );
        }
        if (it + 1) % 50 == 0 && it + 1 != tune_iters {
            eval(p, &student, &format!("tune{}", it + 1));
        }
    }
    eval(&p, &teacher, "teacher");
    eval(&p, &student, "tuned");
    let path = format!("ipse_grad_student_{}x{}.params", distill_iters, tune_iters);
    let txt: String = student
        .0
        .params
        .iter()
        .map(|v| format!("{v:.17e}\n"))
        .collect();
    std::fs::write(&path, txt).expect("save");
    println!(
        "saved {path} ({} params, {} obs)",
        student.0.params.len(),
        p.n_obs()
    );
}
