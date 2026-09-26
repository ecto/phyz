//! Rough cost of the unified convex-contact adjoint on a fixed rollout.
//!
//! Prints forward-only time, gradient (forward + unified backward) time, and
//! the ratio, for a 200-step box drop-and-settle (the `diff_convex_contact`
//! gate scenario). Not a benchmark harness — a sanity probe for the cost
//! delta.

use std::time::Instant;

use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::{
    ConvexContactRollout, FinalStateObjective, convex_adjoint_gradient, convex_rollout_objective,
};
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder};

fn box_model() -> Model {
    let h = 0.05;
    let ix = 1.0 / 12.0 * (2.0 * h) * (2.0 * h) * 2.0;
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_free_body(
            "box",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                1.0,
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

fn main() {
    let model = box_model();
    let ctrl = |_t: usize| DVec::zeros(6);
    let obj = FinalStateObjective {
        value: &|q: &[f64], _v: &[f64]| q[5],
        gradient: &|q: &[f64], v: &[f64]| {
            let mut gq = vec![0.0; q.len()];
            gq[5] = 1.0;
            (gq, vec![0.0; v.len()])
        },
    };
    let mut q0 = DVec::zeros(6);
    q0[5] = 0.06;
    let rollout = ConvexContactRollout {
        model: &model,
        ground_height: 0.0,
        material: ContactMaterial::default(),
        config: ContactSolverConfig::simulation(),
        q0,
        v0: DVec::zeros(6),
        steps: 200,
        ctrl: &ctrl,
    };

    let reps = 20;
    let t0 = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(convex_rollout_objective(&rollout, &obj));
    }
    let fwd = t0.elapsed() / reps;

    let t0 = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(convex_adjoint_gradient(&rollout, &obj).unwrap());
    }
    let grad = t0.elapsed() / reps;

    println!("forward-only rollout (200 steps): {fwd:?}");
    println!("forward + unified adjoint:        {grad:?}");
    println!(
        "cost ratio (gradient / forward):  {:.1}x",
        grad.as_secs_f64() / fwd.as_secs_f64()
    );
}
