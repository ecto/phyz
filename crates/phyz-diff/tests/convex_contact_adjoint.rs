//! FD gates for the unified convex-contact trajectory adjoint.
//!
//! Every gradient claim is checked against a central finite difference of the
//! *same* deterministic primal rollout (`convex_rollout_objective`, which runs
//! the identical forward path). Trajectories cover the three phases that
//! matter: through an impact, through settled resting contact, and through a
//! frictional slide.

use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::{
    ConvexContactRollout, FinalStateObjective, convex_adjoint_gradient, convex_rollout_objective,
};
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder};

const HALF: f64 = 0.05;

/// A free box with collision geometry, the workhorse of the contact tests.
fn box_model(mass: f64) -> Model {
    let h = HALF;
    let ix = mass / 12.0 * (2.0 * h) * (2.0 * h) * 2.0;
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_free_body(
            "box",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                mass,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(ix, ix, ix)),
            ),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: Vec3::new(h, h, h),
    });
    model
}

/// Objective `J = q_T[i]`.
fn q_objective(i: usize) -> FinalStateObjective<'static> {
    match i {
        3 => FinalStateObjective {
            value: &|q: &[f64], _v: &[f64]| q[3],
            gradient: &|q: &[f64], v: &[f64]| {
                let mut gq = vec![0.0; q.len()];
                gq[3] = 1.0;
                (gq, vec![0.0; v.len()])
            },
        },
        5 => FinalStateObjective {
            value: &|q: &[f64], _v: &[f64]| q[5],
            gradient: &|q: &[f64], v: &[f64]| {
                let mut gq = vec![0.0; q.len()];
                gq[5] = 1.0;
                (gq, vec![0.0; v.len()])
            },
        },
        _ => unreachable!(),
    }
}

struct Scenario {
    q0: Vec<f64>,
    v0: Vec<f64>,
    steps: usize,
    tau: f64,
    obj_index: usize,
}

fn run_scenario(name: &str, sc: &Scenario, tol: f64) {
    let model = box_model(1.0);
    let material = ContactMaterial::default();
    let config = ContactSolverConfig::simulation();
    let obj = q_objective(sc.obj_index);
    let tau = sc.tau;
    let ctrl = move |_t: usize| {
        let mut u = DVec::zeros(6);
        u[3] = tau;
        u
    };
    fn make<'a>(
        m: &'a Model,
        material: &ContactMaterial,
        config: ContactSolverConfig,
        q0: &[f64],
        v0: &[f64],
        steps: usize,
        ctrl: &'a dyn Fn(usize) -> DVec,
    ) -> ConvexContactRollout<'a> {
        ConvexContactRollout {
            model: m,
            ground_height: 0.0,
            material: material.clone(),
            config,
            q0: DVec::from_slice(q0),
            v0: DVec::from_slice(v0),
            steps,
            ctrl,
        }
    }

    let rollout = make(&model, &material, config, &sc.q0, &sc.v0, sc.steps, &ctrl);
    let g = convex_adjoint_gradient(&rollout, &obj).expect("adjoint must not refuse");

    // Collect (label, adjoint, fd) per group, then assert with a denominator
    // floored at 1e-3 of the group's largest gradient: a symmetric lane whose
    // true gradient is exactly zero is judged against the group's scale, not
    // against an absolute floor a thousand times below it.
    // (label, adjoint value, FD value) per lane, grouped by channel.
    type Lane = (String, f64, f64);
    let mut groups: Vec<(String, Vec<Lane>)> = Vec::new();

    // h = 1e-7: the objective's curvature along a driven slide's normal-load
    // channel makes 1e-6 truncation-limited (measured: 3e-3 relative at 1e-6,
    // 8e-7 at 1e-7), while the primal is deterministic enough that round-off
    // at 1e-7 stays below 1e-8 absolute.
    let h = 1e-7;
    let mut rows = Vec::new();
    for i in 0..6 {
        let mut qp = sc.q0.clone();
        let mut qm = sc.q0.clone();
        qp[i] += h;
        qm[i] -= h;
        let fp = convex_rollout_objective(
            &make(&model, &material, config, &qp, &sc.v0, sc.steps, &ctrl),
            &obj,
        );
        let fm = convex_rollout_objective(
            &make(&model, &material, config, &qm, &sc.v0, sc.steps, &ctrl),
            &obj,
        );
        rows.push((format!("dJ/dq0[{i}]"), g.d_q0[i], (fp - fm) / (2.0 * h)));
    }
    groups.push(("q0".into(), rows));

    let mut rows = Vec::new();
    for i in 0..6 {
        let mut vp = sc.v0.clone();
        let mut vm = sc.v0.clone();
        vp[i] += h;
        vm[i] -= h;
        let fp = convex_rollout_objective(
            &make(&model, &material, config, &sc.q0, &vp, sc.steps, &ctrl),
            &obj,
        );
        let fm = convex_rollout_objective(
            &make(&model, &material, config, &sc.q0, &vm, sc.steps, &ctrl),
            &obj,
        );
        rows.push((format!("dJ/dv0[{i}]"), g.d_v0[i], (fp - fm) / (2.0 * h)));
    }
    groups.push(("v0".into(), rows));

    let t_probe = sc.steps / 2;
    let mut rows = Vec::new();
    for comp in [3usize, 5usize] {
        let hu = 1e-4;
        let mk_ctrl = |delta: f64| {
            move |t: usize| {
                let mut u = DVec::zeros(6);
                u[3] = tau;
                if t == t_probe {
                    u[comp] += delta;
                }
                u
            }
        };
        let cp = mk_ctrl(hu);
        let cm = mk_ctrl(-hu);
        let rp = ConvexContactRollout {
            ctrl: &cp,
            ..make(&model, &material, config, &sc.q0, &sc.v0, sc.steps, &ctrl)
        };
        let rm = ConvexContactRollout {
            ctrl: &cm,
            ..make(&model, &material, config, &sc.q0, &sc.v0, sc.steps, &ctrl)
        };
        let fd = (convex_rollout_objective(&rp, &obj) - convex_rollout_objective(&rm, &obj))
            / (2.0 * hu);
        rows.push((
            format!("dJ/du[{t_probe}][{comp}]"),
            g.d_ctrl[t_probe][comp],
            fd,
        ));
    }
    groups.push(("ctrl".into(), rows));

    let mut rows = Vec::new();
    for (k, label) in [(0usize, "mass"), (6usize, "Izz")] {
        let p0 = phyz_diff::inertia_params(&model.bodies[0].inertia);
        let hp = 1e-6 * p0[k].abs().max(1.0);
        let build = |pk: f64| {
            let mut p = p0;
            p[k] = pk;
            let mut m = model.clone();
            m.bodies[0].inertia = SpatialInertia::new(
                p[0],
                Vec3::new(p[1], p[2], p[3]),
                Mat3::new(p[4], p[7], p[8], p[7], p[5], p[9], p[8], p[9], p[6]),
            );
            m
        };
        let mp = build(p0[k] + hp);
        let mm = build(p0[k] - hp);
        let fp = convex_rollout_objective(
            &make(&mp, &material, config, &sc.q0, &sc.v0, sc.steps, &ctrl),
            &obj,
        );
        let fm = convex_rollout_objective(
            &make(&mm, &material, config, &sc.q0, &sc.v0, sc.steps, &ctrl),
            &obj,
        );
        rows.push((
            format!("dJ/d{label}"),
            g.d_inertia[0][k],
            (fp - fm) / (2.0 * hp),
        ));
    }
    groups.push(("inertia".into(), rows));

    // Gate: |adjoint - FD| <= tol * max(|FD|, group scale). Errors on a lane
    // whose true gradient is exactly zero by symmetry are judged against the
    // group's largest gradient, not against the FD noise floor.
    let mut max_rel: f64 = 0.0;
    for (gname, rows) in &groups {
        // Scale floor 1e-4: after 300 settled steps the whole v0 group decays
        // to ~1e-6 gradients whose FD is round-off noise (~1e-8); the gate
        // resolves absolute errors down to tol * 1e-4 and no further.
        let scale = rows.iter().fold(1e-4f64, |m, r| m.max(r.2.abs()));
        for (label, adj, fd) in rows {
            let rel = (adj - fd).abs() / fd.abs().max(scale);
            max_rel = max_rel.max(rel);
            assert!(
                rel <= tol,
                "{name}: {label} ({gname}) adjoint {adj} vs FD {fd} (rel {rel:.3e})"
            );
        }
    }

    eprintln!("{name}: max relative error {max_rel:.3e}");
}

/// Short drop: the trajectory ends mid/just-after impact.
#[test]
fn through_impact() {
    run_scenario(
        "through_impact",
        &Scenario {
            q0: vec![0.0, 0.0, 0.0, 0.0, 0.0, HALF + 0.01],
            v0: vec![0.0; 6],
            steps: 60,
            tau: 0.0,
            obj_index: 5,
        },
        1e-3,
    );
}

/// Long rollout: drop, impact, and a long settled-resting-contact tail.
#[test]
fn through_settled_contact() {
    run_scenario(
        "through_settled_contact",
        &Scenario {
            q0: vec![0.0, 0.0, 0.0, 0.0, 0.0, HALF + 0.01],
            v0: vec![0.0; 6],
            steps: 300,
            tau: 0.0,
            obj_index: 5,
        },
        1e-3,
    );
}

/// Frictional slide: resting on the plane with initial tangential velocity,
/// decelerated by Coulomb friction; objective is the final x position.
#[test]
fn through_slide() {
    run_scenario(
        "through_slide",
        &Scenario {
            q0: vec![0.0, 0.0, 0.0, 0.0, 0.0, HALF],
            v0: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            steps: 150,
            tau: 0.0,
            obj_index: 3,
        },
        1e-3,
    );
}

/// A driven slide (constant x force) with the settled contact carrying it.
#[test]
fn driven_slide() {
    run_scenario(
        "driven_slide",
        &Scenario {
            q0: vec![0.0, 0.0, 0.0, 0.0, 0.0, HALF],
            v0: vec![0.0; 6],
            steps: 200,
            tau: 8.0,
            obj_index: 3,
        },
        1e-3,
    );
}
