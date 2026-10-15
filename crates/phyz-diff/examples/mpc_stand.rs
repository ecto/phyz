//! Push recovery by gradient-based model-predictive control.
//!
//! Every `MPC_REPLAN` steps (default 10) the controller optimises a sequence
//! of `MPC_H` (default 16) joint-target offsets over the PD servos: Adam on
//! the summed step cost, with the exact gradient from the traced contact
//! step's adjoint (`ContactStep::adjoint`, verified against finite
//! differences to well past this horizon). It executes the first
//! `MPC_REPLAN` offsets, then warm-starts the next solve from the shifted
//! sequence. No policy, no critic: the optimiser is the controller.
//! Scored on 600-step pushes against PD (zero offsets). The step and adjoint
//! run as native f64 code (`MPC_NATIVE=0`: interpreted).
//!
//! On one M4 Max core, defaults (horizon 16, replan 10, 3 iterations):
//! loss -85% vs PD, trunk drop 0 vs 29.5 mm, 8.1 ms per solve (inside the
//! 10 ms between replans at 1 kHz).
//!
//! `cargo run --release -p phyz-diff --example mpc_stand [ITERS_PER_SOLVE]`

#[path = "train_stand.rs"]
#[allow(dead_code)]
pub mod stand;

use std::time::Instant;

use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::contact_step::ContactStep;
use stand::reach::{Adam, Lcg};
use stand::{OFFSET, SWEEPS, Stand, k1};

pub fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    if std::env::var_os("STAND_H").is_none() {
        unsafe { std::env::set_var("STAND_H", "600") };
    }
    let episode = stand::horizon();
    let h = env("MPC_H", 16);
    let replan = env("MPC_REPLAN", 10);
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(3);
    let n_eval = env("MPC_EPISODES", 16);

    let (model, points) = k1();
    let mat = ContactMaterial {
        friction: 0.8,
        margin: 0.005,
        ..ContactMaterial::default()
    };
    let config = ContactSolverConfig::default();
    let step = ContactStep::build(&model, &points, &mat, &config, SWEEPS);
    let (ag, ao) = step.adjoint();
    // MPC_NATIVE=0 uses the interpreted closures; otherwise the step and
    // adjoint are compiled to native f64 code once (cached by source hash)
    let (fwd, back) = if std::env::var("MPC_NATIVE").as_deref() == Ok("0") {
        (step.compile(), ag.compile_many(&ao))
    } else {
        let t = Instant::now();
        let fwd = native(&step.kernel_c_f64().source, step.n_outputs());
        let back = native(
            &ag.to_c_f64(&ao, step.n_inputs() + step.n_outputs()).source,
            ao.len(),
        );
        eprintln!(
            "native step + adjoint ready in {:.1}s",
            t.elapsed().as_secs_f64()
        );
        (fwd, back)
    };
    let task = Stand::new(&model, fwd, back, step.nu, points.len());
    if std::env::var_os("MPC_BENCH").is_some() {
        // per-call cost of the native step and adjoint
        let x = task.rest.clone();
        let mut out = vec![0.0; step.n_outputs()];
        let mut xin = x.clone();
        xin.extend(std::iter::repeat_n(0.1, step.n_outputs()));
        let mut back = vec![0.0; step.n_inputs()];
        let n = 2000;
        let t = Instant::now();
        for _ in 0..n {
            (task.fwd)(std::hint::black_box(&x), &mut out);
        }
        let f_us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        let t = Instant::now();
        for _ in 0..n {
            (task.back)(std::hint::black_box(&xin), &mut back);
        }
        let b_us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        println!("native step {f_us:.1} us, adjoint {b_us:.1} us per call");
        return;
    }

    let mut rng = Lcg(29);
    let pushes: Vec<Vec<f64>> = (0..n_eval).map(|_| task.push(&mut rng)).collect();
    let zero = |_: &[f64], u: &mut [f64]| u.iter_mut().for_each(|x| *x = 0.0);
    let (base, base_drop) = task.evaluate(&pushes, &zero);
    eprintln!(
        "{n_eval} pushes, {episode} steps; PD: loss {base:.4}, trunk drop {:.2} mm",
        base_drop * 1e3
    );
    eprintln!(
        "MPC: horizon {h}, replan every {replan}, {iters} {} iterations per solve",
        std::env::var("MPC_OPT").unwrap_or_else(|_| "L-BFGS".into())
    );

    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let t = Instant::now();
    let chunk = pushes.len().div_ceil(threads);
    let results: Vec<(f64, f64, f64)> = std::thread::scope(|sc| {
        let hs: Vec<_> = pushes
            .chunks(chunk)
            .map(|c| {
                let task = &task;
                sc.spawn(move || {
                    c.iter()
                        .map(|x0| run_episode(task, x0, episode, h, replan, iters))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let n = results.len() as f64;
    let loss = results.iter().map(|r| r.0).sum::<f64>() / n;
    let drop = results.iter().map(|r| r.1).sum::<f64>() / n;
    let solve_ms = results.iter().map(|r| r.2).sum::<f64>() / n;
    println!(
        "MPC: loss {loss:.4} vs PD {base:.4} ({:+.1}%), trunk drop {:.2} vs {:.2} mm; {solve_ms:.1} ms per solve on one core ({:.0} s total)",
        100.0 * (loss / base - 1.0),
        drop * 1e3,
        base_drop * 1e3,
        t.elapsed().as_secs_f64()
    );
}

/// A tang-expr f64 C kernel (`void k0(const double*, double*, int)`) as a
/// one-environment closure: compiled with `cc -O1` into a shared library
/// in the temp directory, reused while the source is unchanged.
pub fn native(src: &str, n_out: usize) -> tang_expr::compile::CompiledMany {
    use std::hash::{Hash, Hasher};
    let mut hs = std::collections::hash_map::DefaultHasher::new();
    // MPC_CFLAGS (default -O1): part of the cache key
    let flags = std::env::var("MPC_CFLAGS").unwrap_or_else(|_| "-O1".into());
    src.hash(&mut hs);
    flags.hash(&mut hs);
    let dir = std::env::temp_dir().join("phyz-mpc-native");
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join(format!("k{:016x}", hs.finish()));
    let lib = base.with_extension(if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    });
    if !lib.exists() {
        let c = base.with_extension("c");
        std::fs::write(&c, src).unwrap();
        let ok = std::process::Command::new("cc")
            .args(flags.split_whitespace())
            .args(["-shared", "-fPIC", "-o"])
            .arg(&lib)
            .arg(&c)
            .arg("-lm")
            .status()
            .expect("run cc")
            .success();
        assert!(ok, "cc failed on {}", c.display());
    }
    let lib: &'static libloading::Library =
        Box::leak(Box::new(unsafe { libloading::Library::new(&lib) }.unwrap()));
    type K = unsafe extern "C" fn(*const f64, *mut f64, i32);
    let k: K = *unsafe { lib.get::<K>(b"k0") }.unwrap();
    Box::new(move |x: &[f64], out: &mut [f64]| {
        debug_assert!(out.len() >= n_out);
        unsafe { k(x.as_ptr(), out.as_mut_ptr(), 1) }
    })
}

/// One push episode under MPC: `(loss, trunk drop, mean ms per solve)`.
fn run_episode(
    task: &Stand,
    x0: &[f64],
    episode: usize,
    h: usize,
    replan: usize,
    iters: usize,
) -> (f64, f64, f64) {
    let (nq, nv, nu, nf) = (task.nq, task.nv, task.nu, task.nf);
    let ns = nq + nv;
    // the plan: offsets for the next h steps (unbounded internally, applied
    // through OFFSET tanh so the servo targets stay sane)
    let mut plan = vec![0.0; h * nu];
    let mut x = x0.to_vec();
    let mut out = vec![0.0; ns + nf];
    let (mut loss, mut solve_s, mut solves) = (0.0, 0.0, 0);
    let mut t = 0;
    while t < episode {
        let hh = h.min(episode - t);
        let start = Instant::now();
        solve(task, &x, &mut plan[..hh * nu], hh, iters);
        solve_s += start.elapsed().as_secs_f64();
        solves += 1;
        for k in 0..replan.min(episode - t) {
            for i in 0..nu {
                x[ns + i] = task.rest[ns + i] + bound() * plan[k * nu + i].tanh();
            }
            (task.fwd)(&x, &mut out);
            x[..ns].copy_from_slice(&out[..ns]);
            x[ns + nu..].copy_from_slice(&out[ns..]);
            loss += task.step_loss(&x, None);
        }
        t += replan;
        // warm start: shift the plan, repeat the last offsets
        let r = replan.min(h);
        plan.copy_within(r * nu.., 0);
        let last = (h - r) * nu;
        if h > r {
            let tail: Vec<f64> = plan[(h - r - 1) * nu..last].to_vec();
            for k in h - r..h {
                plan[k * nu..(k + 1) * nu].copy_from_slice(&tail);
            }
        }
    }
    (loss, task.z0 - x[5], 1e3 * solve_s / solves as f64)
}

/// Steps each planned offset is held (`MPC_KNOT`, default 1): `h` knots
/// then look `h * knot` steps ahead for the same number of unknowns.
/// The offset bound in rad (`MPC_OFFSET`, default the trainers' 0.3).
pub fn bound() -> f64 {
    std::env::var("MPC_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(OFFSET)
}

pub fn knot() -> usize {
    env("MPC_KNOT", 1).max(1)
}

/// Planner steps per knot (`MPC_PLAN_KNOT`, default `knot()`): fewer when
/// the planner's model steps are longer than the plant's (MPC_MODEL_DT_X).
pub fn plan_knot() -> usize {
    env("MPC_PLAN_KNOT", knot()).max(1)
}

/// The window's cost from `x0` under `plan` (h knots x nu raw offsets, each
/// held `knot()` steps), the last
/// step weighted 5x as a cheap terminal cost; with `grad`, also its exact
/// gradient through the contact steps (one forward and one adjoint sweep).
pub fn objective(
    task: &Stand,
    x0: &[f64],
    plan: &[f64],
    h: usize,
    grad: Option<&mut [f64]>,
) -> f64 {
    let (nq, nv, nu, nf) = (task.nq, task.nv, task.nu, task.nf);
    let (ns, ni) = (nq + nv, nq + nv + nu + nf);
    let kn = plan_knot();
    let steps = h * kn;
    let weight = |k: usize| if k + 1 == steps { 5.0 } else { 1.0 };
    let mut xs = vec![vec![0.0; ni]; steps + 1];
    let mut out = vec![0.0; ns + nf];
    let mut x = x0.to_vec();
    let mut cost = 0.0;
    for k in 0..steps {
        for i in 0..nu {
            x[ns + i] = task.rest[ns + i] + bound() * plan[(k / kn) * nu + i].tanh();
        }
        (task.fwd)(&x, &mut out);
        xs[k].copy_from_slice(&x);
        x[..ns].copy_from_slice(&out[..ns]);
        x[ns + nu..].copy_from_slice(&out[ns..]);
        cost += weight(k) * task.step_loss(&x, None);
    }
    xs[steps].copy_from_slice(&x);
    let Some(grad) = grad else { return cost };
    let mut xin = vec![0.0; ni + ns + nf];
    let mut back = vec![0.0; ni];
    let mut gx = vec![0.0; ni];
    let mut lam = vec![0.0; ns + nf];
    grad.iter_mut().for_each(|g| *g = 0.0);
    for k in (0..steps).rev() {
        gx.iter_mut().for_each(|g| *g = 0.0);
        task.step_loss(&xs[k + 1], Some(&mut gx));
        for i in 0..ns {
            lam[i] += weight(k) * gx[i];
        }
        xin[..ni].copy_from_slice(&xs[k]);
        xin[ni..].copy_from_slice(&lam);
        (task.back)(&xin, &mut back);
        for i in 0..nu {
            let j = (k / kn) * nu + i;
            let th = plan[j].tanh();
            grad[j] += back[ns + i] * bound() * (1.0 - th * th);
        }
        lam[..ns].copy_from_slice(&back[..ns]);
        lam[ns..].copy_from_slice(&back[ns + nu..]);
    }
    cost
}

/// Improve `plan` from `x0`: `iters` steps of L-BFGS with a
/// backtracking line search (the default; `MPC_OPT=adam` for Adam).
pub fn solve(task: &Stand, x0: &[f64], plan: &mut [f64], h: usize, iters: usize) {
    if std::env::var("MPC_OPT").as_deref() != Ok("adam") {
        return lbfgs(task, x0, plan, h, iters);
    }
    let mut adam = Adam::new(plan.len(), 0.05);
    let mut grad = vec![0.0; plan.len()];
    for _ in 0..iters {
        objective(task, x0, plan, h, Some(&mut grad));
        if grad.iter().all(|g| g.is_finite()) {
            adam.step(plan, &grad);
        }
    }
}

/// L-BFGS (memory 6) with Armijo backtracking: curvature from past
/// gradients, so the step length is chosen by the cost, not a tuned rate.
fn lbfgs(task: &Stand, x0: &[f64], plan: &mut [f64], h: usize, iters: usize) {
    let segments = env("MPC_SEGMENTS", 1);
    if segments > 1 {
        return multiple_shooting(task, x0, plan, h, iters, segments);
    }
    minimize(
        &|z: &[f64], g: Option<&mut [f64]>| objective(task, x0, z, h, g),
        plan,
        iters,
    );
}

/// `f(z, grad)`: a cost, and its gradient when asked.
pub type Objective<'a> = dyn Fn(&[f64], Option<&mut [f64]>) -> f64 + Sync + 'a;

/// L-BFGS on `f(z, grad)` from `z`, in place.
fn minimize(obj: &Objective<'_>, plan: &mut [f64], iters: usize) {
    let n = plan.len();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let mut g = vec![0.0; n];
    let mut f = obj(plan, Some(&mut g));
    let mut hist: std::collections::VecDeque<(Vec<f64>, Vec<f64>, f64)> = Default::default();
    for _ in 0..iters {
        if !g.iter().all(|v| v.is_finite()) {
            return;
        }
        // two-loop recursion: d = -H g
        let mut q = g.clone();
        let mut alphas = Vec::with_capacity(hist.len());
        for (sk, yk, rho) in hist.iter().rev() {
            let a = rho * dot(sk, &q);
            q.iter_mut().zip(yk).for_each(|(qi, yi)| *qi -= a * yi);
            alphas.push(a);
        }
        let gamma = hist.back().map_or_else(
            // first step: a 0.05-rad move in the steepest coordinate, like Adam's
            || 0.05 / g.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12),
            |(sk, yk, _)| dot(sk, yk) / dot(yk, yk),
        );
        q.iter_mut().for_each(|v| *v *= gamma);
        for ((sk, yk, rho), a) in hist.iter().zip(alphas.iter().rev()) {
            let b = rho * dot(yk, &q);
            q.iter_mut()
                .zip(sk)
                .for_each(|(qi, si)| *qi += (a - b) * si);
        }
        let d: Vec<f64> = q.iter().map(|v| -v).collect();
        let slope = dot(&g, &d);
        if slope >= 0.0 {
            hist.clear();
            continue;
        }
        // the full step usually passes, so it is evaluated with its gradient
        // straight away; backtracking steps are forward-only
        let mut t = 1.0;
        let mut trial = vec![0.0; n];
        let mut gn = vec![0.0; n];
        let mut accepted = None;
        // up to 14 halvings: a step that moves a shooting start into the floor
        // costs orders of magnitude, and needs to back off far
        for attempt in 0..14 {
            trial
                .iter_mut()
                .zip(plan.iter().zip(&d))
                .for_each(|(o, (p, di))| *o = p + t * di);
            let ft = obj(&trial, (attempt == 0).then_some(&mut gn[..]));
            if std::env::var_os("MPC_DEBUG").is_some() {
                eprintln!(
                    "  t {t:.4} f {ft:.6e} (need <= {:.6e})",
                    f + 1e-4 * t * slope
                );
            }
            if ft.is_finite() && ft <= f + 1e-4 * t * slope {
                accepted = Some((ft, attempt == 0));
                break;
            }
            t *= 0.5;
        }
        if std::env::var_os("MPC_DEBUG").is_some() {
            eprintln!(
                "lbfgs f {f:.6e} slope {slope:.3e} accepted {:?}",
                accepted.map(|a: (f64, bool)| a.0)
            );
        }
        let Some((fnew, have_grad)) = accepted else {
            return;
        };
        if !have_grad {
            obj(&trial, Some(&mut gn));
        }
        let sk: Vec<f64> = trial.iter().zip(plan.iter()).map(|(a, b)| a - b).collect();
        let yk: Vec<f64> = gn.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&sk, &yk);
        if sy > 1e-12 {
            hist.push_back((sk, yk, 1.0 / sy));
            if hist.len() > 6 {
                hist.pop_front();
            }
        }
        plan.copy_from_slice(&trial);
        (f, g) = (fnew, gn);
    }
}

/// Multiple shooting (`MPC_SEGMENTS` > 1), experimental and not useful yet:
/// the gradient is right (checked against central differences, MPC_CHECK)
/// but the gap penalty against contact makes the problem stiff (effective
/// curvature ~1e4 along a search direction), so steps the line search
/// accepts are too short for a few iterations to help; it holds 0/20 in
/// mpc_robust where single shooting holds 10/20.
///
/// The horizon in `segments` equal
/// pieces, each piece's start state (q, v) an unknown alongside the plan,
/// the gaps between one piece's end and the next's start penalised
/// (`MPC_RHO`, default 100). Each piece's rollout and adjoint is
/// independent given its start, so they run on separate threads: an
/// iteration costs one piece, not the horizon. Starts are initialised from
/// a single-shooting rollout of the warm-started plan (no gaps); warm
/// starts for the contact solve are taken from that rollout and held.
#[allow(clippy::needless_range_loop)] // pieces index knots, starts and gaps together
fn multiple_shooting(
    task: &Stand,
    x0: &[f64],
    plan: &mut [f64],
    h: usize,
    iters: usize,
    segments: usize,
) {
    let (nq, nv, nu, nf) = (task.nq, task.nv, task.nu, task.nf);
    let (ns, ni) = (nq + nv, nq + nv + nu + nf);
    let kn = plan_knot();
    assert_eq!(h % segments, 0, "MPC_H must be a multiple of MPC_SEGMENTS");
    let (hk, steps) = (h / segments, h / segments * kn);
    let rho: f64 = std::env::var("MPC_RHO")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100.0);
    let last = h * kn;
    // single-shooting initialisation: each piece's start input
    let mut starts: Vec<Vec<f64>> = Vec::with_capacity(segments);
    {
        let mut x = x0.to_vec();
        let mut out = vec![0.0; ns + nf];
        for k in 0..h * kn {
            if k % steps == 0 {
                starts.push(x.clone());
            }
            for i in 0..nu {
                x[ns + i] = task.rest[ns + i] + bound() * plan[(k / kn) * nu + i].tanh();
            }
            (task.fwd)(&x, &mut out);
            x[..ns].copy_from_slice(&out[..ns]);
            x[ns + nu..].copy_from_slice(&out[ns..]);
        }
    }
    let np = plan.len();
    // unknowns: the plan, then pieces 1.. start (q, v) scaled by sqrt(rho)
    // so the gap penalty has unit curvature: unscaled, the starts' gradients
    // dwarf the knots' and L-BFGS's first steps barely move the plan
    let sigma = rho.sqrt();
    let mut z = plan.to_vec();
    for st in &starts[1..] {
        z.extend(st[..ns].iter().map(|v| v * sigma));
    }
    let start = |z: &[f64], i: usize, j: usize| z[np + i * ns + j] / sigma;
    // one piece: cost from its start under its knots, with the gap to the
    // next start; gradients into its knots, its start and the next start
    let piece = |z: &[f64], i: usize, g: Option<&mut [f64]>| -> f64 {
        let mut x = starts[i].clone();
        if i > 0 {
            for j in 0..ns {
                x[j] = start(z, i - 1, j);
            }
        }
        let mut xs = vec![vec![0.0; ni]; steps + 1];
        let mut out = vec![0.0; ns + nf];
        let mut cost = 0.0;
        let weight = |k: usize| if i * steps + k + 1 == last { 5.0 } else { 1.0 };
        for k in 0..steps {
            let kk = i * hk + k / kn;
            for j in 0..nu {
                x[ns + j] = task.rest[ns + j] + bound() * z[kk * nu + j].tanh();
            }
            (task.fwd)(&x, &mut out);
            xs[k].copy_from_slice(&x);
            x[..ns].copy_from_slice(&out[..ns]);
            x[ns + nu..].copy_from_slice(&out[ns..]);
            cost += weight(k) * task.step_loss(&x, None);
        }
        let gap: Option<Vec<f64>> =
            (i + 1 < segments).then(|| (0..ns).map(|j| x[j] - start(z, i, j)).collect());
        if let Some(gp) = &gap {
            cost += 0.5 * rho * gp.iter().map(|d| d * d).sum::<f64>();
        }
        let Some(g) = g else { return cost };
        let mut lam = vec![0.0; ns + nf];
        if let Some(gp) = &gap {
            for j in 0..ns {
                lam[j] = rho * gp[j];
                // the next start pulls the other way (chain rule: s = w/sigma)
                g[np + i * ns + j] -= rho * gp[j] / sigma;
            }
        }
        let mut xin = vec![0.0; ni + ns + nf];
        let mut back = vec![0.0; ni];
        let mut gx = vec![0.0; ni];
        for k in (0..steps).rev() {
            gx.iter_mut().for_each(|v| *v = 0.0);
            task.step_loss(&xs[k + 1], Some(&mut gx));
            for j in 0..ns {
                lam[j] += weight(k) * gx[j];
            }
            xin[..ni].copy_from_slice(&xs[k]);
            xin[ni..].copy_from_slice(&lam);
            (task.back)(&xin, &mut back);
            let kk = i * hk + k / kn;
            for j in 0..nu {
                let th = z[kk * nu + j].tanh();
                g[kk * nu + j] += back[ns + j] * bound() * (1.0 - th * th);
            }
            lam[..ns].copy_from_slice(&back[..ns]);
            lam[ns..].copy_from_slice(&back[ns + nu..]);
        }
        if i > 0 {
            for j in 0..ns {
                g[np + (i - 1) * ns + j] += lam[j] / sigma;
            }
        }
        cost
    };
    let obj = |z: &[f64], g: Option<&mut [f64]>| -> f64 {
        match g {
            None => std::thread::scope(|sc| {
                let hs: Vec<_> = (0..segments)
                    .map(|i| sc.spawn(move || piece(z, i, None)))
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).sum()
            }),
            Some(g) => {
                let n = g.len();
                let parts: Vec<(f64, Vec<f64>)> = std::thread::scope(|sc| {
                    let hs: Vec<_> = (0..segments)
                        .map(|i| {
                            sc.spawn(move || {
                                let mut gi = vec![0.0; n];
                                (piece(z, i, Some(&mut gi)), gi)
                            })
                        })
                        .collect();
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                });
                g.iter_mut().for_each(|v| *v = 0.0);
                let mut f = 0.0;
                for (fi, gi) in parts {
                    f += fi;
                    g.iter_mut().zip(gi).for_each(|(a, b)| *a += b);
                }
                f
            }
        }
    };
    if std::env::var_os("MPC_CHECK").is_some() {
        // the gradient against central differences, knots and starts
        let mut zz = z.clone();
        for (k, v) in zz.iter_mut().enumerate() {
            *v += 0.01 * ((k * 7919 % 13) as f64 - 6.0) / 6.0;
        }
        let mut g = vec![0.0; zz.len()];
        obj(&zz, Some(&mut g));
        for k in [0, 7, np / 2, np - 1, np, np + 5, zz.len() - 1] {
            let hh = 1e-6;
            let (mut a, mut b) = (zz.clone(), zz.clone());
            a[k] += hh;
            b[k] -= hh;
            let fd = (obj(&a, None) - obj(&b, None)) / (2.0 * hh);
            eprintln!(
                "ms grad [{k}{}]: {:+.6e} vs FD {fd:+.6e}",
                if k < np { " knot" } else { " start" },
                g[k]
            );
        }
        std::process::exit(0);
    }
    minimize(&obj, &mut z, iters);
    plan.copy_from_slice(&z[..np]);
}
