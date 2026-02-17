//! Conservation monitoring for chaotic double pendulum.
//!
//! Demonstrates tracking energy, momentum, and angular momentum
//! to detect when numerical integration becomes unreliable.

use phyz_guardian::{ConservationMonitor, ConservationState, SolverQuality};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::ModelBuilder;
use phyz_rigid::{aba, total_energy};

fn main() {
    println!("=== Conservation Monitor: Double Pendulum ===\n");

    let length = 1.0;
    let mass = 1.0;

    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001)
        .add_revolute_body(
            "link1",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                mass,
                Vec3::new(0.0, -length / 2.0, 0.0),
                Mat3::from_diagonal(&Vec3::new(
                    mass * length * length / 12.0,
                    0.0,
                    mass * length * length / 12.0,
                )),
            ),
        )
        .add_revolute_body(
            "link2",
            0,
            SpatialTransform::translation(Vec3::new(0.0, -length, 0.0)),
            SpatialInertia::new(
                mass,
                Vec3::new(0.0, -length / 2.0, 0.0),
                Mat3::from_diagonal(&Vec3::new(
                    mass * length * length / 12.0,
                    0.0,
                    mass * length * length / 12.0,
                )),
            ),
        )
        .build();

    let mut state = model.default_state();
    // Start with chaotic initial conditions
    state.q[0] = 2.0; // ~114 degrees
    state.q[1] = -1.5;

    // Establish baseline
    let baseline = ConservationState::new(&model, &state);
    println!("Initial conditions:");
    println!("  Energy: {:.6} J", baseline.baseline_energy);
    println!(
        "  Momentum: [{:.6}, {:.6}, {:.6}]",
        baseline.baseline_momentum.x, baseline.baseline_momentum.y, baseline.baseline_momentum.z
    );
    println!(
        "  Angular momentum: [{:.6}, {:.6}, {:.6}]\n",
        baseline.baseline_angular_momentum.x,
        baseline.baseline_angular_momentum.y,
        baseline.baseline_angular_momentum.z
    );

    println!("time(s)   energy      E_error    quality      warning");
    println!("─────────────────────────────────────────────────────────");

    let total_steps = 100_000; // 100 seconds
    let mut warning_count = 0;

    for step in 0..total_steps {
        // Semi-implicit Euler step
        let qdd = aba(&model, &state);
        state.v += &qdd * model.dt;
        state.q += &state.v * model.dt;
        state.time += model.dt;

        // Check conservation every 1000 steps
        if step % 1000 == 0 {
            let monitor = ConservationMonitor::check(&baseline, &model, &state);
            let quality = SolverQuality::assess_from_monitor(&monitor, 1e-3);

            let warning = if !quality.is_acceptable() {
                warning_count += 1;
                "⚠️  DEGRADED"
            } else {
                ""
            };

            println!(
                "{:7.3}   {:9.6}   {:8.2e}   {:?}    {}",
                state.time,
                total_energy(&model, &state),
                monitor.energy_error,
                quality,
                warning
            );

            // Demonstrate graceful degradation
            if quality.needs_intervention() {
                println!("  → Quality degraded! Consider:");
                println!("     - Reducing timestep");
                println!("     - Switching to RK4 integrator");
                println!("     - Checking for numerical instabilities");
            }
        }
    }

    println!("\n=== Summary ===");
    println!("Total steps: {}", total_steps);
    println!("Quality warnings: {}", warning_count);

    let final_monitor = ConservationMonitor::check(&baseline, &model, &state);
    println!("\nFinal conservation status:");
    println!(
        "  Energy error: {:.2e} ({})",
        final_monitor.energy_error,
        if final_monitor.energy_error < 0.01 {
            "GOOD"
        } else {
            "POOR"
        }
    );
    println!(
        "  Momentum error: {:.2e}",
        final_monitor.momentum_error.norm()
    );
    println!(
        "  Angular momentum error: {:.2e}",
        final_monitor.angular_momentum_error.norm()
    );
}
