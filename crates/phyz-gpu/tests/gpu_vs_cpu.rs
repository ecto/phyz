//! Tests comparing GPU and CPU simulation outputs.

use approx::assert_relative_eq;
use phyz_gpu::GpuSimulator;
use phyz_math::{GRAVITY, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::ModelBuilder;
use phyz_rigid::{aba, forward_kinematics};

/// Create a simple pendulum model for testing.
fn make_pendulum() -> phyz_model::Model {
    let length = 1.0;
    let mass = 1.0;

    ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001)
        .add_revolute_body(
            "pendulum",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                mass,
                Vec3::new(0.0, -length / 2.0, 0.0),
                phyz_math::Mat3::from_diagonal(&Vec3::new(
                    mass * length * length / 12.0,
                    0.0,
                    mass * length * length / 12.0,
                )),
            ),
        )
        .build()
}

#[test]
fn test_gpu_single_step_vs_cpu() {
    let model = make_pendulum();

    // Create GPU simulator with single environment
    let gpu_sim = GpuSimulator::new(model.clone(), 1).expect("Failed to create GPU simulator");

    // Initialize state
    let mut cpu_state = model.default_state();
    cpu_state.q[0] = 0.5;
    cpu_state.v[0] = 0.1;

    // Upload to GPU
    gpu_sim.load_states(&[cpu_state.clone()]);

    // CPU step
    forward_kinematics(&model, &mut cpu_state);
    let qdd_cpu = aba(&model, &cpu_state);
    cpu_state.v[0] += model.dt * qdd_cpu[0];
    cpu_state.q[0] += model.dt * cpu_state.v[0];

    // GPU step
    gpu_sim.step();
    let gpu_states = gpu_sim.readback_states();
    let gpu_state = &gpu_states[0];

    // Compare
    println!("CPU: q = {}, v = {}", cpu_state.q[0], cpu_state.v[0]);
    println!("GPU: q = {}, v = {}", gpu_state.q[0], gpu_state.v[0]);

    assert_relative_eq!(cpu_state.q[0], gpu_state.q[0], epsilon = 1e-4);
    assert_relative_eq!(cpu_state.v[0], gpu_state.v[0], epsilon = 1e-4);
}

#[test]
fn test_gpu_multiple_steps_vs_cpu() {
    let model = make_pendulum();

    // Create GPU simulator with single environment
    let gpu_sim = GpuSimulator::new(model.clone(), 1).expect("Failed to create GPU simulator");

    // Initialize state
    let mut cpu_state = model.default_state();
    cpu_state.q[0] = 1.0;
    cpu_state.v[0] = 0.0;

    // Upload to GPU
    gpu_sim.load_states(&[cpu_state.clone()]);

    // Run 100 steps
    let nsteps = 100;

    for _ in 0..nsteps {
        // CPU step
        forward_kinematics(&model, &mut cpu_state);
        let qdd_cpu = aba(&model, &cpu_state);
        cpu_state.v[0] += model.dt * qdd_cpu[0];
        cpu_state.q[0] += model.dt * cpu_state.v[0];

        // GPU step
        gpu_sim.step();
    }

    let gpu_states = gpu_sim.readback_states();
    let gpu_state = &gpu_states[0];

    println!("After {} steps:", nsteps);
    println!("CPU: q = {}, v = {}", cpu_state.q[0], cpu_state.v[0]);
    println!("GPU: q = {}, v = {}", gpu_state.q[0], gpu_state.v[0]);

    // Allow slightly larger error due to accumulated floating-point differences
    assert_relative_eq!(cpu_state.q[0], gpu_state.q[0], epsilon = 1e-3);
    assert_relative_eq!(cpu_state.v[0], gpu_state.v[0], epsilon = 1e-3);
}

#[test]
fn test_gpu_batch_consistency() {
    let model = make_pendulum();

    // Create GPU simulator with multiple environments
    let nworld = 10;
    let gpu_sim = GpuSimulator::new(model.clone(), nworld).expect("Failed to create GPU simulator");

    // Initialize with same state
    let init_state = {
        let mut s = model.default_state();
        s.q[0] = 0.3;
        s.v[0] = 0.2;
        s
    };

    let states = vec![init_state.clone(); nworld];
    gpu_sim.load_states(&states);

    // Run 50 steps
    for _ in 0..50 {
        gpu_sim.step();
    }

    let final_states = gpu_sim.readback_states();

    // All environments should have identical final states
    for i in 1..nworld {
        assert_relative_eq!(final_states[0].q[0], final_states[i].q[0], epsilon = 1e-6);
        assert_relative_eq!(final_states[0].v[0], final_states[i].v[0], epsilon = 1e-6);
    }

    println!("All {} environments evolved identically", nworld);
}
