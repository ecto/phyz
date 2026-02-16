//! Ant quadruped example loading from MJCF.
//!
//! Loads a simple ant model (4 legs, 8 joints) from MJCF XML and runs a short simulation.

use tau::Simulator;
use tau_mjcf::MjcfLoader;

fn main() {
    println!("Ant quadruped example");
    println!("====================\n");

    // Load the ant model from MJCF
    println!("Loading ant.xml...");
    let loader = MjcfLoader::from_file("models/ant.xml").expect("Failed to load ant.xml");
    let model = loader.build_model();

    println!("Model loaded:");
    println!("  Bodies: {}", model.nbodies());
    println!("  DOFs: q={}, v={}", model.nq, model.nv);
    println!("  Timestep: {} s", model.dt);
    println!(
        "  Gravity: [{:.2}, {:.2}, {:.2}]\n",
        model.gravity.x, model.gravity.y, model.gravity.z
    );

    // Print body names and joint types
    println!("Bodies and joints:");
    for (i, body) in model.bodies.iter().enumerate() {
        let joint = &model.joints[body.joint_idx];
        println!(
            "  {}: {} (joint type: {:?}, ndof: {})",
            i,
            body.name,
            joint.joint_type,
            joint.ndof()
        );
    }
    println!();

    // Create simulator and initial state
    let sim = Simulator::new();
    let mut state = model.default_state();

    // Set small initial velocities on some joints
    if state.v.len() >= 4 {
        state.v[0] = 0.1; // front-left hip
        state.v[2] = -0.1; // front-right hip
    }

    println!("Initial configuration:");
    println!(
        "  q[0..4] = [{:.3}, {:.3}, {:.3}, {:.3}]",
        state.q[0], state.q[1], state.q[2], state.q[3]
    );
    println!(
        "  v[0..4] = [{:.3}, {:.3}, {:.3}, {:.3}]",
        state.v[0], state.v[1], state.v[2], state.v[3]
    );

    // Simulate for 1 second
    println!("\nSimulating for 1 second...");
    let steps = (1.0 / model.dt) as usize;

    for i in 0..steps {
        sim.step(&model, &mut state);

        // Print state every 0.1 seconds
        if i % (steps / 10) == 0 {
            let t = state.time;
            println!(
                "  t={:.2}s: q[0]={:.3}, v[0]={:.3}",
                t, state.q[0], state.v[0]
            );
        }
    }

    println!("\nFinal configuration:");
    println!(
        "  q[0..4] = [{:.3}, {:.3}, {:.3}, {:.3}]",
        state.q[0], state.q[1], state.q[2], state.q[3]
    );
    println!(
        "  v[0..4] = [{:.3}, {:.3}, {:.3}, {:.3}]",
        state.v[0], state.v[1], state.v[2], state.v[3]
    );
    println!("  time: {:.3} s", state.time);

    println!("\nAnt example completed successfully!");
}
