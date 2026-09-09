//! The convex-contact adjoint for a sphere rolling on a *fixed box body*,
//! against central differences. The same sphere on the ground plane matches
//! to ~5 digits; this is the body-body path (`Anchor::Pair`) with a curved
//! shape, which has no vertex to ride.

use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::{
    ConvexContactRollout, FinalStateObjective, convex_adjoint_gradient, convex_rollout_objective,
};
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder};

const R: f64 = 0.01;

fn model(tilt: f64) -> Model {
    let si = SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01);
    let m = 0.012;
    let mi = SpatialInertia::new(m, Vec3::zeros(), Mat3::identity() * (0.4 * m * R * R));
    let r = Mat3::rotation_y(tilt);
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_free_body("sphere", -1, SpatialTransform::identity(), mi)
        .add_fixed_body(
            "plate",
            -1,
            SpatialTransform::new(r.transpose(), Vec3::new(0.0, 0.0, 0.25)),
            si,
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Sphere { radius: R });
    model.bodies[1].geometry = Some(Geometry::Box {
        half_extents: Vec3::new(0.15, 0.10, 0.005),
    });
    model
}

fn objective() -> FinalStateObjective<'static> {
    FinalStateObjective {
        value: &|q: &[f64], _: &[f64]| q[3],
        gradient: &|q: &[f64], v: &[f64]| {
            let mut gq = vec![0.0; q.len()];
            gq[3] = 1.0;
            (gq, vec![0.0; v.len()])
        },
    }
}

fn check(tilt: f64, steps: usize) {
    let model = model(tilt);
    let ctrl = |_: usize| DVec::zeros(6);
    let mat = ContactMaterial {
        friction: 0.4,
        ..Default::default()
    };
    // Resting on the (possibly tilted) plate top at the origin: top is at z = 0.255 for tilt 0.
    let mut q0 = DVec::zeros(6);
    q0[5] = 0.255 + R + 0.0005;
    let make = |q0: DVec| ConvexContactRollout {
        model: &model,
        ground_height: -10.0,
        material: mat.clone(),
        config: ContactSolverConfig::gradients(),
        q0,
        v0: DVec::zeros(6),
        steps,
        ctrl: &ctrl,
    };
    let obj = objective();
    let g = convex_adjoint_gradient(&make(q0.clone()), &obj).expect("adjoint");
    // `h` large enough that solver-tolerance noise (~1e-9 on the state) does
    // not masquerade as a gradient; the objective is linear in q so the
    // truncation error is nil.
    let h = 1e-5;
    let mut worst = 0.0f64;
    for i in 3..6 {
        let (mut qp, mut qm) = (q0.clone(), q0.clone());
        qp[i] += h;
        qm[i] -= h;
        let fd = (convex_rollout_objective(&make(qp), &obj)
            - convex_rollout_objective(&make(qm), &obj))
            / (2.0 * h);
        eprintln!(
            "tilt {:.1}° steps {steps}: dJ/dq0[{i}] adjoint {:+.6e} fd {:+.6e}",
            tilt.to_degrees(),
            g.d_q0[i],
            fd
        );
        // 3% relative plus 1e-3 absolute: the relative part is the documented
        // impact-timing bias of the regularized model on the release drop, the
        // absolute part is the FD noise floor.
        worst = worst.max((g.d_q0[i] - fd).abs() / (0.03 * fd.abs() + 1e-3));
    }
    assert!(
        worst < 1.0,
        "adjoint disagrees with FD by {worst:.2e}× the tolerance"
    );
}

#[test]
fn sphere_resting_on_flat_fixed_box() {
    check(0.0, 200);
}

#[test]
fn sphere_rolling_on_tilted_fixed_box() {
    check(5f64.to_radians(), 300);
}
