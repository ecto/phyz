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
    let params = ipse_rig::read_cem(&args.next().expect("CEM.policy"));
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
    let (settle, shove_at) = (1500usize, 1700usize);
    for k in 0..shove_at + 2000 {
        for (s, &d) in pd.dofs.iter().enumerate() {
            x[nq + nv + s] = rig.q_target[d];
        }
        if k >= settle
            && let Some((params, ds)) = cem
        {
            let o = ipse_rig::observe_legs(model, &rig.q_target, &x[..nq], &x[nq..nq + nv], ds);
            for (i, a) in ipse_rig::cem_action(params, &o).into_iter().enumerate() {
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
        fwd(&x, &mut out);
        x[..nq + nv].copy_from_slice(&out[..nq + nv]);
        x[nq + nv + nu..].copy_from_slice(&out[nq + nv..]);
        if !x.iter().all(|v| v.is_finite()) || x[5] < 0.35 {
            return false;
        }
    }
    true
}
