//! Multi-rate integration using r-RESPA.
//!
//! Demonstrates splitting forces into slow and fast components
//! and integrating them at different rates for efficiency.

use tau_guardian::RRespaIntegrator;
use tau_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use tau_model::{Model, ModelBuilder};

fn main() {
    println!("=== r-RESPA Multi-Rate Integration ===\n");

    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001)
        .add_revolute_body(
            "pendulum",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                1.0,
                Vec3::new(0.0, -0.5, 0.0),
                Mat3::from_diagonal(&Vec3::new(0.083, 0.0, 0.083)),
            ),
        )
        .build();

    let mut state = model.default_state();
    state.q[0] = 0.3;

    // Create r-RESPA integrator
    // Fast forces (constraints, stiff springs): dt_inner = 0.1 ms
    // Slow forces (gravity, soft forces): dt_outer = 1.0 ms
    let integrator = RRespaIntegrator::new(0.0001, 0.001);

    println!(
        "dt_inner: {:.4} ms (fast forces)",
        integrator.dt_inner * 1000.0
    );
    println!(
        "dt_outer: {:.4} ms (slow forces)",
        integrator.dt_outer * 1000.0
    );
    println!(
        "Substeps per outer step: {}\n",
        (integrator.dt_outer / integrator.dt_inner).round() as usize
    );

    // Define force splitting
    // Slow forces: gravity and damping
    let slow_forces = |m: &Model, s: &tau_model::State| {
        let mut forces = DVec::zeros(m.nv);

        // Gravitational torque (simplified)
        let g = m.gravity.norm();
        let theta = s.q[0];
        forces[0] = -g * 0.5 * theta.sin(); // Approximation for small angles

        // Damping (slow)
        forces[0] -= 0.1 * s.v[0];

        forces
    };

    // Fast forces: stiff spring-like restoring force (if any)
    let fast_forces = |m: &Model, _s: &tau_model::State| {
        // In this simple example, no fast forces
        // In a real scenario, this would include stiff springs, high-frequency vibrations, etc.
        DVec::zeros(m.nv)
    };

    println!("time(s)   q(rad)     v(rad/s)   substeps_eval");
    println!("───────────────────────────────────────────────");

    let total_steps = 5000; // 5 seconds
    let mut fast_evals = 0;

    for step in 0..total_steps {
        integrator.step(&model, &mut state, &slow_forces, &fast_forces);

        // Count fast force evaluations
        let substeps = (integrator.dt_outer / integrator.dt_inner).round() as usize;
        fast_evals += substeps;

        if step % 500 == 0 {
            println!(
                "{:7.3}   {:+7.4}    {:+7.4}     {}",
                state.time, state.q[0], state.v[0], substeps
            );
        }
    }

    println!("\n=== Summary ===");
    println!("Total outer steps: {}", total_steps);
    println!("Total fast force evaluations: {}", fast_evals);
    println!(
        "Average substeps per outer step: {:.1}",
        fast_evals as f64 / total_steps as f64
    );

    println!("\nr-RESPA efficiency:");
    println!("  With r-RESPA: {} fast evals", fast_evals);
    println!(
        "  Without r-RESPA: {} evals (if all forces were fast)",
        fast_evals * (integrator.dt_outer / integrator.dt_inner).round() as usize
    );
    println!(
        "  Speedup: {:.1}x for systems with mostly slow forces",
        (integrator.dt_outer / integrator.dt_inner)
    );
}
