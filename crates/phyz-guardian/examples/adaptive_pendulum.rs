//! Adaptive time-stepping example with conservation monitoring.
//!
//! Demonstrates:
//! - PI controller adjusting dt based on error
//! - Energy conservation monitoring
//! - Graceful degradation when accuracy drops

use phyz_guardian::{
    AdaptiveTimeStep, AutoSwitchController, ConservationMonitor, ConservationState, SolverQuality,
};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::ModelBuilder;
use phyz_rigid::{aba, total_energy};

fn main() {
    println!("=== Adaptive Time-Stepping Pendulum ===\n");

    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.01) // Start with large timestep
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
    state.q[0] = 1.0; // Large initial angle for more challenging dynamics

    // Initialize conservation tracking
    let baseline = ConservationState::new(&model, &state);
    println!("Initial energy: {:.6} J", baseline.baseline_energy);
    println!("Starting dt: {:.6} s\n", model.dt);

    // Create adaptive timestepper with tight tolerance
    let mut adaptive = AdaptiveTimeStep::new(
        1e-4, // tolerance
        1e-5, // dt_min
        0.01, // dt_max
    );

    // Auto-switch controller
    let mut auto_switch = AutoSwitchController::new(1e-4, 3);

    println!("time(s)    dt(ms)   energy      error      quality    accept/reject");
    println!("──────────────────────────────────────────────────────────────────");

    let mut total_time = 0.0;
    let max_time = 5.0;
    let mut step_count = 0;

    while total_time < max_time {
        // Perform adaptive step
        let (accepted, new_dt) = adaptive.step(&mut model, &mut state, |m, s| {
            // Semi-implicit Euler step
            let qdd = aba(m, s);
            s.v += &qdd * m.dt;
            s.q += &s.v * m.dt;
            s.time += m.dt;
        });

        if accepted {
            total_time = state.time;
            step_count += 1;

            // Check conservation
            let monitor = ConservationMonitor::check(&baseline, &model, &state);
            let quality = SolverQuality::assess_from_monitor(&monitor, 1e-4);
            let _strategy = auto_switch.update(quality);

            // Print status every 0.5 seconds
            if step_count % 10 == 0 || !quality.is_acceptable() {
                println!(
                    "{:7.3}    {:5.3}    {:8.6}    {:8.2e}    {:?}    accepted",
                    total_time,
                    new_dt * 1000.0,
                    total_energy(&model, &state),
                    monitor.energy_error,
                    quality
                );
            }
        } else {
            println!(
                "  ---      {:5.3}      ---        ---        ---        rejected (dt reduced)",
                new_dt * 1000.0
            );
        }
    }

    println!("\n=== Summary ===");
    println!("Total steps: {}", step_count);
    println!("Accepted steps: {}", adaptive.accepted_steps);
    println!("Rejected steps: {}", adaptive.rejected_steps);
    println!("Final dt: {:.6} s", model.dt);

    let final_monitor = ConservationMonitor::check(&baseline, &model, &state);
    println!("\nFinal conservation errors:");
    println!("  Energy: {:.2e}", final_monitor.energy_error);
    println!("  Momentum: {:.2e}", final_monitor.momentum_error.norm());
    println!(
        "  Angular momentum: {:.2e}",
        final_monitor.angular_momentum_error.norm()
    );
}
