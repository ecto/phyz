//! Example: Training data collection with sensors and trajectory recording.
//!
//! Demonstrates recording trajectories for machine learning training.

use phyz::{SemiImplicitEulerSolver, Solver};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::ModelBuilder;
use phyz_world::{Sensor, TrajectoryRecorder, World};

fn main() {
    println!("=== Training Data Collection Example ===\n");

    // Create a simple pendulum model
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
        .build();

    // Create world with sensors
    let mut world = World::new(model);

    // Add sensors
    world.add_sensor(Sensor::JointState { joint_idx: 0 });
    world.add_sensor(Sensor::JointState { joint_idx: 1 });
    world.add_sensor(Sensor::BodyAngularVel { body_idx: 0 });
    world.add_sensor(Sensor::Imu { body_idx: 1 });

    println!("Created world with {} sensors", world.sensors.len());

    // Set initial conditions
    world.state.q[0] = 0.8;
    world.state.q[1] = -0.5;

    // Create trajectory recorder
    let mut recorder = TrajectoryRecorder::new();

    // Run simulation and collect data
    let solver = SemiImplicitEulerSolver;
    let n_steps = 1000;

    println!(
        "Simulating {} steps ({}s)...",
        n_steps,
        n_steps as f64 * 0.001
    );

    for _ in 0..n_steps {
        // Record state before stepping
        recorder.record(&world.state);

        // Step simulation (this also records sensor data in world.sensor_history)
        world.step(|model, state| {
            solver.step(model, state);
        });
    }

    // Print statistics
    let stats = recorder.stats();
    println!("\n--- Trajectory Statistics ---");
    println!("  Steps recorded: {}", stats.nsteps);
    println!("  Position DOFs: {}", stats.nq);
    println!("  Velocity DOFs: {}", stats.nv);
    println!("  Duration: {:.3}s", stats.duration);
    println!("  Sensor readings: {}", world.sensor_history.len());

    // Show sample data
    println!("\n--- Sample Data (first 5 steps) ---");
    for i in 0..5.min(recorder.len()) {
        let q = &recorder.q_history[i];
        let v = &recorder.v_history[i];
        let t = recorder.time_history[i];
        println!(
            "  t={:.3}: q=[{:.3}, {:.3}], v=[{:.3}, {:.3}]",
            t, q[0], q[1], v[0], v[1]
        );
    }

    // Show sensor data
    if let Some(latest_sensors) = world.latest_sensor_readings() {
        println!("\n--- Latest Sensor Readings ---");
        for sensor in latest_sensors {
            println!("  Sensor {}: {:?}", sensor.sensor_id, sensor.data);
        }
    }

    // Export to JSON
    println!("\n--- Exporting Data ---");

    // Export trajectory
    if let Err(e) = recorder.to_json_file("/tmp/phyz_trajectory.json") {
        println!("  ✗ Failed to export trajectory: {}", e);
    } else {
        println!("  ✓ Exported trajectory to /tmp/phyz_trajectory.json");
    }

    // Get flat dictionary for ML frameworks
    let flat_dict = recorder.to_flat_dict();
    println!("  ✓ Flat dictionary created:");
    for (key, values) in &flat_dict {
        println!("    {}: {} values", key, values.len());
    }

    // Demonstrate multi-environment batching
    println!("\n--- Multi-Environment Batch ---");
    let mut recorders = Vec::new();

    for env_id in 0..3 {
        let model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
            .dt(0.001)
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build();

        let mut world = World::new(model);
        world.state.q[0] = (env_id as f64 + 1.0) * 0.3; // Different initial conditions

        let mut rec = TrajectoryRecorder::new();
        let solver = SemiImplicitEulerSolver;

        for _ in 0..100 {
            rec.record(&world.state);
            world.step(|model, state| {
                solver.step(model, state);
            });
        }

        recorders.push(rec);
    }

    println!(
        "  Collected data from {} parallel environments",
        recorders.len()
    );
    for (i, rec) in recorders.iter().enumerate() {
        let stats = rec.stats();
        println!(
            "    Env {}: {} steps, duration={:.3}s",
            i, stats.nsteps, stats.duration
        );
    }

    println!("\n✓ Training data collection complete!");
}
