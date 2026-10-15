//! ipse's CEM stepping policy, ported and run in the traced K1 plant.
//!
//! 1. **The port.** From a CEM rollout dumped by ipse-sim's `k1_rig_dump`
//!    (state, observation and action every 10 ms), recompute the 31-entry
//!    observation and the 10 actions from each recorded state and compare.
//! 2. **The plant.** Run the policy (every 1 ms, as ipse does) in
//!    [`ContactStep::build_pd`] with ipse's shove, sagittal and lateral, and
//!    report the largest it survives next to PD's.
//!
//! `cargo run --release -p phyz-diff --example ipse_cem RIG.txt CEM.policy [CEM_DUMP.txt]`

#[path = "common/ipse_rig.rs"]
mod ipse_rig;

use phyz_contact::ContactSolverConfig;
use phyz_diff::contact_step::ContactStep;
use tang_expr::compile::CompiledMany;

fn main() {
    let mut args = std::env::args().skip(1);
    let rig = ipse_rig::Rig::read(
        &args
            .next()
            .expect("usage: ipse_cem RIG.txt CEM.policy [CEM_DUMP.txt]"),
    );
    // IPSE_POLICY_SCALE multiplies the weights (not the bias): how much
    // loop gain the plant tolerates before the policy starts to chatter
    let scale = std::env::var("IPSE_POLICY_SCALE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let mut params = ipse_rig::read_cem(&args.next().expect("CEM.policy"));
    params[..310].iter_mut().for_each(|w| *w *= scale);
    let model = rig.model();
    let dofs = ipse_rig::cem_dofs(&model);

    if let Some(dump) = args.next() {
        let traj = ipse_rig::Rig::read(&dump).traj;
        let rows = ipse_rig::cem_rows(&dump);
        let (mut wo, mut wa) = (0.0f64, 0.0f64);
        for ((t, q), (t2, v, obs, act)) in traj.iter().zip(&rows) {
            assert!((t - t2).abs() < 1e-9);
            let o = ipse_rig::observe_legs(&model, &rig.q_target, q, v, &dofs);
            let a = ipse_rig::cem_action(&params, &o);
            for (x, y) in o.iter().zip(obs) {
                wo = wo.max((x - y).abs() / (1.0 + y.abs()));
            }
            for (x, y) in a.iter().zip(act) {
                wa = wa.max((x - y).abs());
            }
        }
        println!(
            "port vs ipse over {} recorded states: observation {wo:.2e} (relative), action {wa:.2e} rad",
            rows.len()
        );
    }

    let step = ContactStep::build_pd(
        &model,
        &rig.points(),
        &rig.material(),
        &ContactSolverConfig::default(),
        5,
        &rig.pd(),
    );
    let fwd = step.compile();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let levels: Vec<f64> = (2..=12).map(|k| 2.0 * k as f64).collect();
    let jobs: Vec<(usize, f64, bool)> = (0..2)
        .flat_map(|d| {
            levels
                .iter()
                .flat_map(move |&n| [(d, n, false), (d, n, true)])
        })
        .collect();
    let chunk = jobs.len().div_ceil(threads);
    let res: Vec<(usize, f64, bool, bool)> = std::thread::scope(|sc| {
        let hs: Vec<_> = jobs
            .chunks(chunk)
            .map(|c| {
                let (rig, model, fwd, params, dofs, nu) =
                    (&rig, &model, &fwd, &params, &dofs, step.nu);
                sc.spawn(move || {
                    c.iter()
                        .map(|&(d, n, cem)| {
                            let imp = if d == 0 { [n, 0.0, 0.0] } else { [0.0, n, 0.0] };
                            (
                                d,
                                n,
                                cem,
                                survives(rig, model, fwd, nu, imp, cem.then_some((params, dofs))),
                            )
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    for (d, name) in ["sagittal", "lateral"].iter().enumerate() {
        let best = |cem: bool| {
            let mut last = 0.0;
            for &n in &levels {
                if !res
                    .iter()
                    .any(|r| r.0 == d && r.1 == n && r.2 == cem && r.3)
                {
                    break;
                }
                last = n;
            }
            last
        };
        let row: String = levels
            .iter()
            .map(|&n| {
                let ok = |cem: bool| {
                    res.iter()
                        .any(|r| r.0 == d && r.1 == n && r.2 == cem && r.3)
                };
                format!(
                    " {n:.0}:{}{}",
                    if ok(false) { "P" } else { "." },
                    if ok(true) { "C" } else { "." }
                )
            })
            .collect();
        println!(
            "{name:>9}: PD survives {:>4.0} N·s, CEM {:>4.0} N·s  |{row}",
            best(false),
            best(true)
        );
    }
}

/// Stand 0.2 s, shove (ipse's 20 ms base wrench), 2 s more; alive = trunk
/// above 0.35 m throughout. `cem`: the policy's offsets every step, else PD.
fn survives(
    rig: &ipse_rig::Rig,
    model: &phyz_model::Model,
    fwd: &CompiledMany,
    nu: usize,
    impulse: [f64; 3],
    cem: Option<(&Vec<f64>, &Vec<usize>)>,
) -> bool {
    let (nq, nv) = (model.nq, model.nv);
    let pd = rig.pd();
    let nt = nu - 6;
    let nf = 3 * rig.points().len();
    let mut x: Vec<f64> = rig.q0.iter().chain(&rig.v0).copied().collect();
    x.extend(pd.dofs.iter().map(|&d| rig.q_target[d]));
    x.extend([0.0; 6]);
    x.extend(std::iter::repeat_n(0.0, nf));
    let mut out = vec![0.0; nq + nv + nf];
    // where each CEM DOF sits among the PD targets
    let slot: Vec<usize> = cem.map_or(Vec::new(), |(_, ds)| {
        ds.iter()
            .map(|d| pd.dofs.iter().position(|p| p == d).unwrap())
            .collect()
    });
    // IPSE_CHATTER: 2 s of quiet stance under the policy before the shove,
    // reporting peak base gyro and the fraction of action sign flips
    let chatter = std::env::var("IPSE_CHATTER").is_ok();
    let (settle, shove_at) = (1500usize, if chatter { 3500 } else { 1700usize });
    let (mut peak, mut flips, mut queries) = (0.0f64, 0usize, 0usize);
    let mut prev: Vec<f64> = Vec::new();
    // the robot's loop, not the sim's: IPSE_POLICY_EVERY steps between
    // policy queries (hold runs at 50 Hz), IPSE_LATENCY steps before an
    // action lands, IPSE_GYRO_NOISE rad/s uniform on the base and joint
    // rates, IPSE_CLAMP the deploy clamp
    let env = |k: &str, d: f64| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let every = env("IPSE_POLICY_EVERY", 1.0) as usize;
    let latency = env("IPSE_LATENCY", 0.0) as usize;
    let noise = env("IPSE_GYRO_NOISE", 0.0);
    let clamp = env("IPSE_CLAMP", 0.3);
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut uniform = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let mut pending: std::collections::VecDeque<(usize, Vec<f64>)> = Default::default();
    let mut action: Vec<f64> = Vec::new();
    for k in 0..shove_at + 2000 {
        for (s, &d) in pd.dofs.iter().enumerate() {
            x[nq + nv + s] = rig.q_target[d];
        }
        if k >= settle
            && let Some((params, ds)) = cem
        {
            if (k - settle) % every == 0 {
                let mut o =
                    ipse_rig::observe_legs(model, &rig.q_target, &x[..nq], &x[nq..nq + nv], ds);
                if noise > 0.0 {
                    for i in (2..5).chain((12..o.len()).step_by(2)) {
                        o[i] += noise * uniform();
                    }
                }
                let a = ipse_rig::cem_action(params, &o);
                if chatter && k < shove_at {
                    if !prev.is_empty() {
                        flips += a
                            .iter()
                            .zip(&prev)
                            .filter(|(p, q)| p.signum() != q.signum())
                            .count();
                        queries += a.len();
                    }
                    prev = a.clone();
                }
                pending.push_back((
                    k + latency,
                    a.iter().map(|v| v.clamp(-clamp, clamp)).collect(),
                ));
            }
            while pending.front().is_some_and(|(t, _)| *t <= k) {
                action = pending.pop_front().unwrap().1;
            }
            for (i, a) in action.iter().enumerate() {
                x[nq + nv + slot[i]] += a;
            }
        }
        let w = &mut x[nq + nv + nt..nq + nv + nt + 6];
        w.iter_mut().for_each(|v| *v = 0.0);
        if (shove_at..shove_at + 20).contains(&k) {
            for a in 0..3 {
                w[3 + a] = impulse[a] / 0.02;
            }
        }
        if chatter && k == shove_at {
            println!(
                "chatter: peak gyro {peak:.2} rad/s, sign flips {:.0}%",
                100.0 * flips as f64 / queries.max(1) as f64
            );
        }
        if k >= settle && k < shove_at {
            peak = x[nq..nq + 3].iter().fold(peak, |m, v| m.max(v.abs()));
        }
        fwd(&x, &mut out);
        x[..nq + nv].copy_from_slice(&out[..nq + nv]);
        x[nq + nv + nu..].copy_from_slice(&out[nq + nv..]);
        if !x.iter().all(|v| v.is_finite()) || x[5] < 0.35 {
            return false;
        }
    }
    true
}
