//! Example: Tendon actuators in multi-body systems.
//!
//! Demonstrates cable-like actuators that span multiple bodies.

use tau::{SemiImplicitEulerSolver, Solver};
use tau_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use tau_model::ModelBuilder;
use tau_world::{Tendon, World};

fn main() {
    println!("=== Tendon Actuator Example ===\n");

    // Create a 3-link chain
    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(0.001)
        .add_revolute_body(
            "link1",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                1.0,
                Vec3::new(0.0, 0.0, -0.5),
                Mat3::from_diagonal(&Vec3::new(0.083, 0.083, 0.0)),
            ),
        )
        .add_revolute_body(
            "link2",
            0,
            SpatialTransform::translation(Vec3::new(0.0, 0.0, -1.0)),
            SpatialInertia::new(
                0.8,
                Vec3::new(0.0, 0.0, -0.5),
                Mat3::from_diagonal(&Vec3::new(0.067, 0.067, 0.0)),
            ),
        )
        .add_revolute_body(
            "link3",
            1,
            SpatialTransform::translation(Vec3::new(0.0, 0.0, -1.0)),
            SpatialInertia::new(
                0.6,
                Vec3::new(0.0, 0.0, -0.5),
                Mat3::from_diagonal(&Vec3::new(0.05, 0.05, 0.0)),
            ),
        )
        .build();

    // Create world
    let mut world = World::new(model);

    // Add tendon spanning all three bodies
    // This acts like a cable running through the links
    let tendon = Tendon::new(
        vec![0, 1, 2], // Path through bodies 0, 1, 2
        100.0,         // Stiffness (N/m)
        2.5,           // Rest length (m)
        50.0,          // Max force (N)
    )
    .with_damping(1.0); // Damping (Ns/m)

    world.add_tendon(tendon);

    println!("Created 3-link chain with tendon actuator");
    println!("  Tendon path: body 0 → body 1 → body 2");
    println!("  Stiffness: 100 N/m");
    println!("  Rest length: 2.5 m");
    println!("  Max force: 50 N\n");

    // Set initial positions to stretch the tendon
    world.state.q[0] = 0.8;
    world.state.q[1] = -0.6;
    world.state.q[2] = 0.4;

    // Compute initial tendon properties
    let initial_length = world.tendons[0].current_length(&world.state);
    let initial_stretch = initial_length - world.tendons[0].rest_length;

    println!("Initial configuration:");
    println!(
        "  Joint angles: q=[{:.3}, {:.3}, {:.3}]",
        world.state.q[0], world.state.q[1], world.state.q[2]
    );
    println!("  Tendon length: {:.3} m", initial_length);
    println!("  Stretch: {:.3} m", initial_stretch);
    println!("  Expected force: {:.1} N\n", initial_stretch.abs() * 100.0);

    // Simulate with tendon forces
    println!("Simulating for 2 seconds...");

    let solver = SemiImplicitEulerSolver;
    let steps = 2000; // 2 seconds at dt=0.001

    for step in 0..steps {
        world.step(|model, state| {
            solver.step(model, state);
        });

        // Print status every 500 steps
        if step % 500 == 0 {
            let length = world.tendons[0].current_length(&world.state);
            let stretch = length - world.tendons[0].rest_length;
            println!(
                "  t={:.2}s: length={:.3}m, stretch={:.3}m",
                world.state.time, length, stretch
            );
        }
    }

    println!("\nFinal configuration:");
    let final_length = world.tendons[0].current_length(&world.state);
    let final_stretch = final_length - world.tendons[0].rest_length;
    println!(
        "  Joint angles: q=[{:.3}, {:.3}, {:.3}]",
        world.state.q[0], world.state.q[1], world.state.q[2]
    );
    println!("  Tendon length: {:.3} m", final_length);
    println!("  Stretch: {:.3} m", final_stretch);

    // Demonstrate force computation
    println!("\n--- Tendon Force Computation ---");
    let forces = world.tendons[0].compute_forces(&world.state);
    println!("  Forces applied to {} body pairs", forces.len() / 2);
    for (i, (body_idx, force)) in forces.iter().enumerate() {
        let f_norm = force.linear().norm();
        println!("    Force {}: body {} ← {:.2} N", i, body_idx, f_norm);
    }

    // Demonstrate force saturation
    println!("\n--- Testing Force Saturation ---");
    let high_stiffness_tendon = Tendon::new(
        vec![0, 1],
        1000.0, // Very high stiffness
        1.0,    // Rest length
        10.0,   // Low max force (will saturate)
    );

    let forces = high_stiffness_tendon.compute_forces(&world.state);
    for (body_idx, force) in &forces {
        let f_norm = force.linear().norm();
        println!(
            "  Body {}: force magnitude = {:.2} N (capped at 10 N)",
            body_idx, f_norm
        );
    }

    println!("\n✓ Tendon demonstration complete!");
}
