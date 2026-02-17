//! MuJoCo Ant Benchmark
//!
//! Loads the ant quadruped from MJCF, sets initial pose with torso at z=0.75,
//! and simulates 1000 steps with zero control using step_with_contacts.
//! Reports joint positions, torso height, total sim time, and per-step timing.

use std::time::Instant;
use phyz::{ContactMaterial, Simulator};
use phyz_mjcf::MjcfLoader;

fn main() {
    println!("MuJoCo Ant Benchmark");
    println!("====================\n");

    // Load the ant model from MJCF
    let loader = MjcfLoader::from_file("models/ant.xml").expect("Failed to load ant.xml");
    let model = loader.build_model();

    println!("Model loaded:");
    println!("  Bodies: {}", model.nbodies());
    println!("  DOFs: q={}, v={}", model.nq, model.nv);
    println!("  Timestep: {} s", model.dt);
    println!(
        "  Gravity: [{:.2}, {:.2}, {:.2}]",
        model.gravity.x, model.gravity.y, model.gravity.z
    );
    println!("  Actuators: {}", model.actuators.len());
    println!();

    // Print body/joint info
    println!("Bodies and joints:");
    for (i, body) in model.bodies.iter().enumerate() {
        let joint = &model.joints[body.joint_idx];
        println!(
            "  {}: {} (joint: {:?}, ndof: {})",
            i,
            body.name,
            joint.joint_type,
            joint.ndof()
        );
    }
    println!();

    // Create simulator and state
    let sim = Simulator::new();
    let mut state = model.default_state();

    // Set initial pose: torso at z=0.75
    // Free joint: q[0..3] = position (x, y, z), q[3..6] = orientation
    state.q[2] = 0.75;

    let ground_height = 0.0;
    let material = ContactMaterial::default();
    let n_steps = 1000;

    println!("Initial state:");
    println!("  Torso height (q[2]): {:.4}", state.q[2]);
    print_joint_positions(&state.q, model.nq);
    println!();

    // Benchmark simulation
    println!("Simulating {} steps with zero control...", n_steps);
    let wall_start = Instant::now();

    for _ in 0..n_steps {
        sim.step_with_contacts(&model, &mut state, ground_height, &material);
    }

    let wall_elapsed = wall_start.elapsed();

    println!("\n--- Results ---\n");

    // Torso height
    let torso_z = state.q[2];
    println!("Torso height: {:.6} m", torso_z);

    // Joint positions
    print_joint_positions(&state.q, model.nq);

    // Simulation time
    println!("\nSimulation time: {:.4} s", state.time);

    // Benchmark
    let total_us = wall_elapsed.as_micros();
    let per_step_us = total_us as f64 / n_steps as f64;
    println!(
        "Wall time: {:.2} ms ({:.2} us/step)",
        wall_elapsed.as_secs_f64() * 1000.0,
        per_step_us
    );
    println!(
        "Realtime factor: {:.1}x",
        state.time / wall_elapsed.as_secs_f64()
    );

    println!("\nAnt benchmark completed.");
}

fn print_joint_positions(q: &phyz::phyz_math::DVec, nq: usize) {
    print!("  Joint positions q[0..{}]: [", nq);
    for i in 0..nq {
        if i > 0 {
            print!(", ");
        }
        print!("{:.4}", q[i]);
    }
    println!("]");
}
