//! FD gate for the convex-contact adjoint on a *rolling* contact.
//!
//! A cylinder's support point is a material point of the body (the cap centre)
//! plus `radius · dir`, where `dir` is a rim direction that depends on the
//! *axis*, not on the spin. `Anchor::GroundCylinderRim` recomputes `dir`
//! at each perturbed configuration; a wrong `dp/dq` is a silently wrong
//! gradient rather than a failure, which is what these gates are for.
//!
//! Recomputing rather than freezing is not free and is not optional. Freezing
//! the offset — the treatment a sphere's `(0,0,−r)` gets, because for a sphere
//! it is genuinely constant — is exact for a *level* rolling wheel and wrong
//! for a banked one. Measured on `a_tilted_rolling_wheel_differentiates` at
//! 0.25 rad of tilt: frozen gave `dJ/dq0[0] = +7.8e-3` against a finite
//! difference of `−9.5e-3` (worst relative error 3.0e-2 across the lanes),
//! recomputed gives 2.9e-6.

use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::{
    ConvexContactRollout, FinalStateObjective, convex_adjoint_gradient, convex_rollout_objective,
};
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder};

const R: f64 = 0.1;
const HEIGHT: f64 = 0.04;
const MASS: f64 = 2.0;

fn wheel_model() -> Model {
    let i_axial = 0.5 * MASS * R * R;
    let i_diam = MASS * (3.0 * R * R + HEIGHT * HEIGHT) / 12.0;
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_free_body(
            "wheel",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                MASS,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(i_diam, i_diam, i_axial)),
            ),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Cylinder {
        radius: R,
        height: HEIGHT,
    });
    model
}

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

fn zero_ctrl(_t: usize) -> DVec {
    DVec::zeros(6)
}

fn rollout<'a>(
    model: &'a Model,
    material: &ContactMaterial,
    q0: &[f64],
    v0: &[f64],
    steps: usize,
    ctrl: &'a dyn Fn(usize) -> DVec,
) -> ConvexContactRollout<'a> {
    ConvexContactRollout {
        model,
        ground_height: 0.0,
        material: material.clone(),
        config: ContactSolverConfig::simulation(),
        q0: DVec::from_slice(q0),
        v0: DVec::from_slice(v0),
        steps,
        ctrl,
    }
}

/// Central-difference every `q0` and `v0` lane against the adjoint.
///
/// Returns the worst relative error, judged against the larger of the lane's
/// own FD and the group's scale, so a lane that is zero by symmetry is not
/// compared against its own round-off.
fn fd_gate(name: &str, q0: &[f64], v0: &[f64], steps: usize, obj_index: usize, tol: f64) -> f64 {
    let model = wheel_model();
    let material = ContactMaterial::default();
    let obj = q_objective(obj_index);
    let ctrl: &dyn Fn(usize) -> DVec = &zero_ctrl;

    let g = convex_adjoint_gradient(&rollout(&model, &material, q0, v0, steps, ctrl), &obj)
        .expect("adjoint must not refuse a rolling contact");

    let h = 1e-7;
    /// One lane: its label, the adjoint's value, and the finite difference.
    type Lane = (String, f64, f64);
    let mut groups: Vec<(&str, Vec<Lane>)> = Vec::new();
    for (gname, base, adj) in [("q0", q0, &g.d_q0), ("v0", v0, &g.d_v0)] {
        let mut rows = Vec::new();
        for i in 0..6 {
            let mut p = base.to_vec();
            let mut m = base.to_vec();
            p[i] += h;
            m[i] -= h;
            let (qp, vp, qm, vm) = if gname == "q0" {
                (p.as_slice(), v0, m.as_slice(), v0)
            } else {
                (q0, p.as_slice(), q0, m.as_slice())
            };
            let fp =
                convex_rollout_objective(&rollout(&model, &material, qp, vp, steps, ctrl), &obj);
            let fm =
                convex_rollout_objective(&rollout(&model, &material, qm, vm, steps, ctrl), &obj);
            rows.push((format!("dJ/d{gname}[{i}]"), adj[i], (fp - fm) / (2.0 * h)));
        }
        groups.push((gname, rows));
    }

    let mut max_rel: f64 = 0.0;
    for (gname, rows) in &groups {
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
    max_rel
}

/// A wheel rolling without slipping, differentiated through 120 steps of
/// rolling contact.
///
/// The objective is how far it got, which is the quantity a trajectory
/// optimiser would actually ask about, and it depends on the contact through
/// every step of the rollout.
#[test]
fn a_rolling_wheels_distance_differentiates() {
    let q0 = vec![std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, 0.0, R + 2e-3];
    let v0 = vec![0.0, 0.0, -12.0, 1.2, 0.0, 0.0];
    fd_gate("rolling_distance", &q0, &v0, 120, 3, 1e-3);
}

/// The same rollout, with height as the objective: the channel that carries
/// "the support point is `r` below the axle", which is exactly the thing the
/// old rim sampler got wrong.
#[test]
fn a_rolling_wheels_height_differentiates() {
    let q0 = vec![std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, 0.0, R + 2e-3];
    let v0 = vec![0.0, 0.0, -12.0, 1.2, 0.0, 0.0];
    fd_gate("rolling_height", &q0, &v0, 120, 5, 1e-3);
}

/// A banked wheel: the case that separates a recomputed rim direction from a
/// frozen one.
///
/// Tilting the axis slides the contact around the rim by `r·dû`, a channel a
/// frozen offset reports as zero. At 0.25 rad it is not a small correction —
/// it is larger than the gradient it corrects, so the frozen form came out
/// with the wrong sign (see this file's header).
#[test]
fn a_tilted_rolling_wheel_differentiates() {
    let tilt = 0.25;
    let q0 = vec![
        std::f64::consts::FRAC_PI_2 + tilt,
        0.0,
        0.0,
        0.0,
        0.0,
        R + 2e-3,
    ];
    let v0 = vec![0.0, 0.0, -8.0, 0.8, 0.0, 0.0];
    fd_gate("tilted_rolling", &q0, &v0, 120, 5, 1e-3);
}

/// A cylinder standing on its cap, where detection takes the degenerate branch
/// and reports a rim polygon.
///
/// Four contacts on one body, each with its own offset — the case where a
/// single per-shape scalar drop could not have described the manifold at all.
#[test]
fn an_upright_cylinder_on_its_rim_differentiates() {
    let q0 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * HEIGHT + 2e-3];
    let v0 = vec![0.0; 6];
    fd_gate("upright_rim", &q0, &v0, 120, 5, 1e-3);
}
