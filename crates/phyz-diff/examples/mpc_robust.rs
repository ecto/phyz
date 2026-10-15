//! How many shoves MPC holds: strengths x1.0..x1.4 in four directions
//! (roll + sideways, each sign), 1.5 s each, success = the trunk never drops
//! more than 3 cm. PD alongside. Optimiser and budget from `mpc_stand`'s
//! knobs (`MPC_OPT`, `MPC_H`, `MPC_REPLAN`, iterations as the argument).
//!
//! `cargo run --release -p phyz-diff --example mpc_robust [ITERS]`

#[path = "mpc_stand.rs"]
#[allow(dead_code)]
mod mpc;

use mpc::stand::{SWEEPS, Stand, k1};
use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::contact_step::ContactStep;

const STEPS: usize = 1500;

fn main() {
    if std::env::var_os("STAND_H").is_none() {
        unsafe { std::env::set_var("STAND_H", "600") };
    }
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(3);
    let (h, replan) = (mpc::env("MPC_H", 16), mpc::env("MPC_REPLAN", 10));
    let (model, points) = k1();
    let mat = ContactMaterial {
        friction: 0.8,
        margin: 0.005,
        ..ContactMaterial::default()
    };
    let config = ContactSolverConfig::default();
    let step = ContactStep::build(&model, &points, &mat, &config, SWEEPS);
    let (ag, ao) = step.adjoint();
    let fwd = mpc::native(&step.kernel_c_f64().source, step.n_outputs());
    let back = mpc::native(
        &ag.to_c_f64(&ao, step.n_inputs() + step.n_outputs()).source,
        ao.len(),
    );
    let task = Stand::new(&model, fwd, back, step.nu, points.len());
    // MPC_MODEL_SWEEPS: the planner's own model with fewer PGS sweeps than
    // the plant (cheaper step and adjoint); it plans around the plant's
    // standing state
    // MPC_MODEL_DT_X: its steps that many plant steps long (pair with
    // MPC_PLAN_KNOT = MPC_KNOT / MPC_MODEL_DT_X)
    let dt_x = mpc::env("MPC_MODEL_DT_X", 1);
    let sweeps_env = std::env::var("MPC_MODEL_SWEEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let planner = (sweeps_env.is_some() || dt_x > 1).then(|| {
        let sw = sweeps_env.unwrap_or(SWEEPS);
        let mut pm = model.clone();
        pm.dt *= dt_x as f64;
        let st = ContactStep::build(&pm, &points, &mat, &config, sw);
        let (ag, ao) = st.adjoint();
        let fwd = mpc::native(&st.kernel_c_f64().source, st.n_outputs());
        let back = mpc::native(
            &ag.to_c_f64(&ao, st.n_inputs() + st.n_outputs()).source,
            ao.len(),
        );
        let mut p = Stand::new(&pm, fwd, back, st.nu, points.len());
        p.rest = task.rest.clone();
        p.z0 = task.z0;
        eprintln!(
            "planner model: {sw} sweeps, dt x{dt_x} ({} + {} nodes)",
            st.live_nodes(),
            ag.live_set(&ao).len()
        );
        p
    });
    let planner = planner.as_ref().unwrap_or(&task);

    let mut cases = Vec::new();
    for s in [1.0, 1.1, 1.2, 1.3, 1.4] {
        for (roll, side) in [(1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)] {
            let mut x = task.rest.clone();
            x[task.nq] += 1.2 * s * roll;
            x[task.nq + 4] += 0.5 * s * side;
            cases.push((s, x));
        }
    }
    // MPC_THREADS: episodes in parallel (default all cores); lower it when
    // each solve runs its own threads (MPC_SEGMENTS), so timings are honest
    let threads = std::env::var("MPC_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
    let chunk = cases.len().div_ceil(threads);
    let results: Vec<(f64, f64, f64, f64)> = std::thread::scope(|sc| {
        let hs: Vec<_> = cases
            .chunks(chunk)
            .map(|c| {
                let (task, planner) = (&task, planner);
                sc.spawn(move || {
                    c.iter()
                        .map(|(s, x0)| {
                            let (pd, _) = drop_under(task, task, x0, None);
                            let (mp, ms) = drop_under(task, planner, x0, Some((h, replan, iters)));
                            (*s, pd, mp, ms)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let held = |d: f64| d < 0.03;
    let dirs = ["+roll +side", "-roll -side", "+roll -side", "-roll +side"];
    for (k, r) in results.iter().enumerate() {
        eprintln!(
            "  x{} {:12}  drop PD {:4.0} mm  MPC {:4.0} mm",
            r.0,
            dirs[k % 4],
            r.1 * 1e3,
            r.2 * 1e3
        );
    }
    for s in [1.0, 1.1, 1.2, 1.3, 1.4] {
        let row: Vec<_> = results.iter().filter(|r| r.0 == s).collect();
        let pd = row.iter().filter(|r| held(r.1)).count();
        let mp = row.iter().filter(|r| held(r.2)).count();
        eprintln!("shove x{s}: PD holds {pd}/4, MPC holds {mp}/4");
    }
    let (pd, mp) = (
        results.iter().filter(|r| held(r.1)).count(),
        results.iter().filter(|r| held(r.2)).count(),
    );
    let ms = results.iter().map(|r| r.3).sum::<f64>() / results.len() as f64;
    println!(
        "{}: {iters} iters, horizon {h}, replan {replan}: MPC holds {mp}/{n}, PD {pd}/{n}; {ms:.1} ms per solve",
        std::env::var("MPC_OPT").unwrap_or_else(|_| "lbfgs".into()),
        n = results.len()
    );
}

/// The trunk's largest drop over the episode under PD (`None`) or MPC, and
/// the mean ms per solve.
/// `task` is the plant; `planner` is the model MPC optimises through.
fn drop_under(
    task: &Stand,
    planner: &Stand,
    x0: &[f64],
    mpc_cfg: Option<(usize, usize, usize)>,
) -> (f64, f64) {
    let (nq, nv, nu, nf) = (task.nq, task.nv, task.nu, task.nf);
    let ns = nq + nv;
    let (h, replan, iters) = mpc_cfg.unwrap_or((16, 10, 0));
    // knots: the plan is h knots of `kn` steps; a replan shifts it by the
    // knots that were executed (replan must be a multiple of kn)
    let kn = if mpc_cfg.is_some() { mpc::knot() } else { 1 };
    assert_eq!(replan % kn, 0, "MPC_REPLAN must be a multiple of MPC_KNOT");
    let shift = (replan / kn).min(h);
    let mut plan = vec![0.0; h * nu];
    let mut x = x0.to_vec();
    let mut out = vec![0.0; ns + nf];
    let (mut lowest, mut secs, mut solves) = (x[5], 0.0, 0);
    for t in 0..STEPS {
        if mpc_cfg.is_some() && t % replan == 0 {
            if t > 0 && shift < h {
                plan.copy_within(shift * nu.., 0);
                let tail: Vec<f64> = plan[(h - shift - 1) * nu..(h - shift) * nu].to_vec();
                for k in h - shift..h {
                    plan[k * nu..(k + 1) * nu].copy_from_slice(&tail);
                }
            }
            let start = std::time::Instant::now();
            mpc::solve(planner, &x, &mut plan, h, iters);
            secs += start.elapsed().as_secs_f64();
            solves += 1;
        }
        let k = (t % replan) / kn;
        for i in 0..nu {
            let off = if mpc_cfg.is_some() {
                mpc::bound() * plan[k * nu + i].tanh()
            } else {
                0.0
            };
            x[ns + i] = task.rest[ns + i] + off;
        }
        (task.fwd)(&x, &mut out);
        x[..ns].copy_from_slice(&out[..ns]);
        x[ns + nu..].copy_from_slice(&out[ns..]);
        if !x.iter().all(|v| v.is_finite()) {
            return (f64::INFINITY, 0.0);
        }
        lowest = lowest.min(x[5]);
    }
    (task.z0 - lowest, 1e3 * secs / solves.max(1) as f64)
}
