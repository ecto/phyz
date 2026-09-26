//! Inverse kinematics: point a three-link planar arm at a moving target.
//!
//! ```bash
//! cargo run --release -p phyz-examples --example ik_reach
//! ```
//!
//! Shows the three things that separate usable IK from a textbook
//! pseudo-inverse: warm starting from the previous frame's answer, damping
//! that survives the straight-arm singularity, and joint limits that hold.

use phyz::rigid::ik::{IkConfig, IkGoal, solve_ik};
use phyz::{Mat3, Model, ModelBuilder, SpatialInertia, SpatialTransform, Vec3};
use phyz_math::{DVec, SpatialTransformExt};
use phyz_rigid::forward_kinematics;

/// Three 1 m links along +X, each hinging about +Z. Reach is 3 m.
fn arm() -> Model {
    let link = |m: f64| SpatialInertia::new(m, Vec3::new(0.5, 0.0, 0.0), Mat3::identity() * 0.05);
    let offset = SpatialTransform::new(Mat3::identity(), Vec3::new(1.0, 0.0, 0.0));

    let mut model = ModelBuilder::new()
        .dt(1e-3)
        .add_revolute_body("shoulder", -1, SpatialTransform::identity(), link(1.0))
        .add_revolute_body("elbow", 0, offset, link(1.0))
        .add_revolute_body("wrist", 1, offset, link(0.5))
        .build();

    // Elbow and wrist are human-ish: they bend one way only.
    model.joints[1].limits = Some([0.0, 2.4]);
    model.joints[2].limits = Some([-2.0, 2.0]);
    model
}

/// World position of the arm's tip at configuration `q`.
fn tip(model: &Model, q: &[f64]) -> Vec3 {
    let mut state = model.default_state();
    state.q = DVec::from_slice(q);
    let (xforms, _) = forward_kinematics(model, &state);
    xforms[2].body_to_world_point(Vec3::new(1.0, 0.0, 0.0))
}

fn main() {
    let model = arm();
    let config = IkConfig::default();

    // Sweep the target around a circle that dips outside the workspace, so the
    // run covers the reachable case, the singular case, and the impossible one.
    let mut q = vec![0.2, 0.4, -0.2];
    let mut total_iterations = 0;
    let mut unreached = 0;

    println!("  step     target                      residual   iters  status");
    for step in 0..24 {
        let theta = step as f64 * std::f64::consts::TAU / 24.0;
        let radius = 2.0 + 1.4 * theta.sin();
        let target = Vec3::new(radius * theta.cos(), radius * theta.sin(), 0.0);

        // Warm start: seed with the previous frame's solution. This is what
        // keeps a redundant arm from flipping to a mirrored branch mid-sweep.
        let sol = solve_ik(
            &model,
            &q,
            &[IkGoal::position(2, Vec3::new(1.0, 0.0, 0.0), target)],
            &config,
        );
        q = sol.q.clone();
        total_iterations += sol.iterations;

        let status = if sol.converged {
            "reached"
        } else {
            unreached += 1;
            "out of reach"
        };
        println!(
            "  {step:>4}     ({:>6.3}, {:>6.3})            {:>9.2e}  {:>4}   {status}",
            target.x, target.y, sol.residual, sol.iterations
        );

        assert!(q.iter().all(|v| v.is_finite()), "IK produced non-finite q");
        assert!(
            q[1] >= -1e-9 && q[1] <= 2.4 + 1e-9,
            "elbow limit violated: {}",
            q[1]
        );
    }

    println!();
    println!(
        "  {total_iterations} iterations over 24 frames, {unreached} targets outside the workspace"
    );
    println!("  final tip: {:?}", tip(&model, &q));
}
