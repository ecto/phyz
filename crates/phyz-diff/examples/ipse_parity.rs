//! The traced contact step against ipse's K1 plant, PD only, one shove.
//!
//! Reads a rig written by ipse-sim's `k1_rig_dump` example (armature, PD
//! gains, effort limits, nominal targets, initial state, foot boxes, contact
//! material, and ipse's own trajectory) and builds the same plant as a
//! [`ContactStep::build_pd`]: the PD inside the trace, the shove as a base
//! wrench exactly as ipse applies it. Then rolls it out and compares base
//! height, tilt and joints against ipse every 10 ms. ipse's contact solve is
//! the Newton convex solver to 1e-10; this one is the traced PGS
//! (`IPSE_SWEEPS`, default 5), so this measures that gap too.
//!
//! `cargo run --release -p phyz-diff --example ipse_parity RIG.txt`

#[path = "common/ipse_rig.rs"]
mod ipse_rig;

use phyz_contact::ContactSolverConfig;
use phyz_diff::contact_step::ContactStep;

fn main() {
    let path = std::env::args().nth(1).expect("usage: ipse_parity RIG.txt");
    let rig = ipse_rig::Rig::read(&path);
    let sweeps: usize = std::env::var("IPSE_SWEEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let model = rig.model();
    let config = ContactSolverConfig::default();
    let step = ContactStep::build_pd(
        &model,
        &rig.points(),
        &rig.material(),
        &config,
        sweeps,
        &rig.pd(),
    );
    eprintln!(
        "ipse rig: {} PD DOFs, {} contact points, friction {}, margin {}; step {} nodes ({sweeps} sweeps)",
        rig.pd().dofs.len(),
        rig.points().len(),
        rig.friction,
        rig.margin,
        step.live_nodes()
    );
    let fwd = step.compile();
    let (nq, nv, nu) = (model.nq, model.nv, step.nu);
    let nf = 3 * rig.points().len();
    let nt = nu - 6;
    let mut x: Vec<f64> = rig.q0.iter().chain(&rig.v0).copied().collect();
    x.extend(rig.pd().dofs.iter().map(|&d| rig.q_target[d]));
    x.extend([0.0; 6]);
    x.extend(std::iter::repeat_n(0.0, nf));
    let mut out = vec![0.0; nq + nv + nf];
    let dt = rig.dt;
    let steps = (rig.traj.last().map_or(4.0, |t| t.0) / dt).round() as usize;
    let (mut worst_z, mut worst_tilt, mut worst_joint) = (0.0f64, 0.0f64, 0.0f64);
    let mut ti = 0;
    let mut lowest = f64::INFINITY;
    let mut ipse_lowest = f64::INFINITY;
    for k in 0..steps {
        let t = k as f64 * dt;
        let wrench = &mut x[nq + nv + nt..nq + nv + nt + 6];
        wrench.iter_mut().for_each(|w| *w = 0.0);
        if let Some(f) = rig.shove_force(t) {
            wrench[3..6].copy_from_slice(&f);
        }
        fwd(&x, &mut out);
        x[..nq + nv].copy_from_slice(&out[..nq + nv]);
        x[nq + nv + nu..].copy_from_slice(&out[nq + nv..]);
        if !x.iter().all(|v| v.is_finite()) {
            println!("diverged at t = {:.3} s", (k + 1) as f64 * dt);
            return;
        }
        lowest = lowest.min(x[5]);
        if k == 2030 {
            let r = rig
                .traj
                .iter()
                .find(|(tt, _)| (tt - (k + 1) as f64 * dt).abs() < 1e-9);
            eprintln!(
                "t {:.3}: base y {:.6} vy {:.4} here; ipse base y {:?}",
                (k + 1) as f64 * dt,
                x[4],
                x[nq + 4],
                r.map(|(_, q)| q[4])
            );
        }
        // ipse records after step k when k % 10 == 0
        if k % 10 == 0
            && let Some((tt, q)) = rig.traj.get(ti)
        {
            debug_assert!((tt - (k + 1) as f64 * dt).abs() < 1e-9);
            ipse_lowest = ipse_lowest.min(q[5]);
            worst_z = worst_z.max((x[5] - q[5]).abs());
            let tilt = |w: &[f64]| (w[0] * w[0] + w[1] * w[1]).sqrt();
            worst_tilt = worst_tilt.max((tilt(&x[..3]) - tilt(&q[..3])).abs());
            for j in 6..nq {
                worst_joint = worst_joint.max((x[j] - q[j]).abs());
            }
            ti += 1;
        }
    }
    println!(
        "{} {} N·s: worst gap to ipse over {:.1} s: base z {:.3e} m, tilt {:.3e} rad, joints {:.3e} rad ({ti} samples); lowest trunk {:.1} mm here vs {:.1} mm ipse (start {:.1} mm)",
        rig.shove_dir,
        rig.shove,
        steps as f64 * dt,
        worst_z,
        worst_tilt,
        worst_joint,
        lowest * 1e3,
        ipse_lowest * 1e3,
        rig.q0[5] * 1e3
    );
}
