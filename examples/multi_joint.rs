//! Multi-joint example demonstrating revolute and prismatic joints.
//!
//! Creates a simple 2-link arm with:
//! - Link 1: revolute joint (rotation about Z axis)
//! - Link 2: prismatic joint (translation along X axis)

use tau::Simulator;
use tau_math::{GRAVITY, SpatialInertia, SpatialTransform, Vec3};
use tau_model::ModelBuilder;
use tau_rigid::total_energy;

fn main() {
    println!("Multi-joint example: revolute + prismatic");
    println!("==========================================\n");

    // Build a 2-link system: revolute joint + prismatic joint
    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(0.001)
        // Link 1: revolute joint about Z axis
        .add_revolute_body(
            "link1",
            -1, // attached to world
            SpatialTransform::identity(),
            SpatialInertia::new(
                1.0,                      // 1 kg
                Vec3::new(0.5, 0.0, 0.0), // CoM at 0.5m along X
                tau_math::Mat3::from_diagonal(&Vec3::new(0.01, 0.01, 0.01)),
            ),
        )
        // Link 2: prismatic joint along X axis
        .add_prismatic_body(
            "link2",
            0,                                                       // attached to link1
            SpatialTransform::translation(Vec3::new(1.0, 0.0, 0.0)), // 1m offset
            Vec3::new(1.0, 0.0, 0.0),                                // slide along X
            SpatialInertia::new(
                0.5,                       // 0.5 kg
                Vec3::new(0.25, 0.0, 0.0), // CoM at 0.25m along X
                tau_math::Mat3::from_diagonal(&Vec3::new(0.005, 0.005, 0.005)),
            ),
        )
        .build();

    println!("Model:");
    println!("  Bodies: {}", model.nbodies());
    println!("  DOFs: q={}, v={}", model.nq, model.nv);
    println!("  Link 1: revolute joint");
    println!("  Link 2: prismatic joint\n");

    // Create simulator
    let sim = Simulator::new();
    let mut state = model.default_state();

    // Set initial configuration
    state.q[0] = std::f64::consts::FRAC_PI_4; // 45 degrees for revolute
    state.q[1] = 0.2; // 0.2m extension for prismatic
    state.v[0] = 0.1; // small angular velocity
    state.v[1] = -0.05; // small linear velocity

    println!("Initial state:");
    println!("  q = [{:.3}, {:.3}]", state.q[0], state.q[1]);
    println!("  v = [{:.3}, {:.3}]", state.v[0], state.v[1]);

    let e0 = total_energy(&model, &state);
    println!("  Energy: {:.6} J\n", e0);

    // Simulate for 2 seconds
    println!("Simulating for 2 seconds...");
    let steps = (2.0 / model.dt) as usize;

    for _ in 0..steps {
        sim.step(&model, &mut state);
    }

    println!("\nFinal state:");
    println!("  q = [{:.3}, {:.3}]", state.q[0], state.q[1]);
    println!("  v = [{:.3}, {:.3}]", state.v[0], state.v[1]);
    println!("  time: {:.3} s", state.time);

    let e1 = total_energy(&model, &state);
    println!("  Energy: {:.6} J", e1);
    println!(
        "  Energy drift: {:.6} J ({:.2}%)",
        e1 - e0,
        100.0 * (e1 - e0) / e0
    );

    println!("\nMulti-joint example completed successfully!");
}
