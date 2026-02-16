//! Benchmark: 1000 parallel pendula on GPU.
//!
//! Demonstrates GPU-accelerated batch simulation with 1000 independent pendulum environments.

use std::time::Instant;
use tau_gpu::GpuSimulator;
use tau_math::{GRAVITY, SpatialInertia, SpatialTransform, Vec3};
use tau_model::ModelBuilder;

fn main() {
    println!("=== GPU Parallel Pendula Benchmark ===\n");

    // Create pendulum model
    let length = 1.0;
    let mass = 1.0;

    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001)
        .add_revolute_body(
            "pendulum",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                mass,
                Vec3::new(0.0, -length / 2.0, 0.0),
                tau_math::Mat3::from_diagonal(&Vec3::new(
                    mass * length * length / 12.0,
                    0.0,
                    mass * length * length / 12.0,
                )),
            ),
        )
        .build();

    // Number of parallel environments
    let nworld = 1000;

    println!("Creating GPU simulator with {} environments...", nworld);
    let gpu_sim = GpuSimulator::new(model.clone(), nworld).expect("Failed to create GPU simulator");

    // Initialize with random states
    let mut states = vec![model.default_state(); nworld];
    for (i, state) in states.iter_mut().enumerate() {
        // Random initial angle and velocity
        state.q[0] = (i as f64 / nworld as f64) * 2.0 * std::f64::consts::PI;
        state.v[0] = ((i % 10) as f64 - 5.0) * 0.1;
    }

    println!("Uploading initial states to GPU...");
    gpu_sim.load_states(&states);

    // Benchmark: 10000 steps
    let nsteps = 10000;
    let total_time = nsteps as f64 * model.dt;

    println!(
        "\nRunning {} steps ({:.1}s simulated time)...",
        nsteps, total_time
    );
    let start = Instant::now();

    for _ in 0..nsteps {
        gpu_sim.step();
    }

    // Wait for GPU to finish
    gpu_sim.device.poll(wgpu::Maintain::Wait);

    let elapsed = start.elapsed();
    println!("Completed in {:.3}s", elapsed.as_secs_f64());

    // Calculate throughput
    let total_steps = nworld * nsteps;
    let steps_per_sec = total_steps as f64 / elapsed.as_secs_f64();
    let speedup = total_time / elapsed.as_secs_f64();

    println!("\nPerformance:");
    println!("  Total environment-steps: {}", total_steps);
    println!("  Throughput: {:.0} env-steps/sec", steps_per_sec);
    println!("  Speedup: {:.1}× real-time", speedup);

    // Download and validate a few states
    println!("\nDownloading final states...");
    let final_states = gpu_sim.readback_states();

    println!("\nSample final states:");
    for i in [0, nworld / 4, nworld / 2, 3 * nworld / 4, nworld - 1] {
        let state = &final_states[i];
        println!(
            "  World {:4}: q = {:.4}, v = {:.4}",
            i, state.q[0], state.v[0]
        );
    }

    // Verify energy is bounded (rough check)
    let mut max_energy = 0.0;
    for state in &final_states {
        let ke = 0.5 * mass * length * length / 3.0 * state.v[0] * state.v[0];
        let pe = mass * GRAVITY * length * (1.0 - state.q[0].cos()) / 2.0;
        let total = ke + pe;
        if total > max_energy {
            max_energy = total;
        }
    }

    println!("\nMax energy across all environments: {:.4} J", max_energy);
    println!(
        "Expected max energy: {:.4} J",
        mass * GRAVITY * length / 2.0
    );
}
