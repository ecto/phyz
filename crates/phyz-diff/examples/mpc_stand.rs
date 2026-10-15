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

    let mut rng = Lcg(29);
    let pushes: Vec<Vec<f64>> = (0..n_eval).map(|_| task.push(&mut rng)).collect();
    let zero = |_: &[f64], u: &mut [f64]| u.iter_mut().for_each(|x| *x = 0.0);
    let (base, base_drop) = task.evaluate(&pushes, &zero);
    eprintln!(
        "{n_eval} pushes, {episode} steps; PD: loss {base:.4}, trunk drop {:.2} mm",
        base_drop * 1e3
    );
    eprintln!("MPC: horizon {h}, replan every {replan}, {iters} Adam iterations per solve");

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
    src.hash(&mut hs);
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
            .args(["-O1", "-shared", "-fPIC", "-o"])
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
                x[ns + i] = task.rest[ns + i] + OFFSET * plan[k * nu + i].tanh();
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

/// Improve `plan` (h x nu raw offsets) from state `x0` by Adam on the
/// window's cost, exact gradients through the contact steps.
pub fn solve(task: &Stand, x0: &[f64], plan: &mut [f64], h: usize, iters: usize) {
    let (nq, nv, nu, nf) = (task.nq, task.nv, task.nu, task.nf);
    let (ns, ni) = (nq + nv, nq + nv + nu + nf);
    let mut adam = Adam::new(plan.len(), 0.05);
    let mut xs = vec![vec![0.0; ni]; h + 1];
    let mut out = vec![0.0; ns + nf];
    let mut xin = vec![0.0; ni + ns + nf];
    let mut back = vec![0.0; ni];
    let mut gx = vec![0.0; ni];
    for _ in 0..iters {
        let mut x = x0.to_vec();
        for k in 0..h {
            for i in 0..nu {
                x[ns + i] = task.rest[ns + i] + OFFSET * plan[k * nu + i].tanh();
            }
            (task.fwd)(&x, &mut out);
            xs[k].copy_from_slice(&x);
            x[..ns].copy_from_slice(&out[..ns]);
            x[ns + nu..].copy_from_slice(&out[ns..]);
        }
        xs[h].copy_from_slice(&x);
        let mut grad = vec![0.0; plan.len()];
        let mut lam = vec![0.0; ns + nf];
        for k in (0..h).rev() {
            gx.iter_mut().for_each(|g| *g = 0.0);
            // the window's last state counts extra: a cheap terminal cost
            task.step_loss(&xs[k + 1], Some(&mut gx));
            let w = if k + 1 == h { 5.0 } else { 1.0 };
            for i in 0..ns {
                lam[i] += w * gx[i];
            }
            xin[..ni].copy_from_slice(&xs[k]);
            xin[ni..].copy_from_slice(&lam);
            (task.back)(&xin, &mut back);
            for i in 0..nu {
                let th = plan[k * nu + i].tanh();
                grad[k * nu + i] = back[ns + i] * OFFSET * (1.0 - th * th);
            }
            lam[..ns].copy_from_slice(&back[..ns]);
            lam[ns..].copy_from_slice(&back[ns + nu..]);
        }
        if grad.iter().all(|g| g.is_finite()) {
            adam.step(plan, &grad);
        }
    }
}
