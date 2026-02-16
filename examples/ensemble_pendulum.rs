//! Ensemble pendulum with parameter uncertainty.
//!
//! Demonstrates:
//! - Creating probabilistic state with perturbed initial conditions
//! - Sampling uncertain physical parameters (length, mass, friction)
//! - Ensemble simulation with uncertainty propagation
//! - Tracking mean ± std trajectories

use rand_distr::{Distribution as RandDist, Normal, Uniform};
use tau::{
    Model, ModelBuilder, State,
    tau_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3},
};
use tau_prob::{ProbabilisticState, ensemble_step_with_params};

/// Simple solver adapter for ensemble simulation.
struct SimpleSolver;

impl tau_prob::ensemble::EnsembleSolver for SimpleSolver {
    fn step(&self, model: &Model, state: &mut State) {
        use tau::aba;
        let dt = model.dt;
        let qdd = aba(model, state);
        state.v += &qdd * dt;
        state.q += &state.v * dt;
        state.time += dt;
    }
}

/// Create a pendulum model with custom parameters.
fn make_pendulum(length: f64, mass: f64, damping: f64) -> Model {
    // Pendulum: revolute about Z, gravity along -Y
    // CoM at [0, -L/2, 0], Inertia = mL²/12 about CoM
    let com = Vec3::new(0.0, -length / 2.0, 0.0);
    let i_about_com = mass * length * length / 12.0;

    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001)
        .add_revolute_body(
            "pendulum",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                mass,
                com,
                Mat3::from_diagonal(&Vec3::new(i_about_com, 0.0, i_about_com)),
            ),
        )
        .build();

    // Apply damping to joint
    model.joints[0].damping = damping;

    model
}

fn main() {
    println!("=== Ensemble Pendulum with Parameter Uncertainty ===\n");

    // Base parameters
    let base_length = 1.0;
    let base_mass = 1.0;
    let base_damping = 0.05;

    let base_model = make_pendulum(base_length, base_mass, base_damping);

    // Create ensemble with perturbed initial conditions
    let nsamples = 100;
    let q0 = DVec::from_vec(vec![0.1]); // 0.1 rad initial angle
    let v0 = DVec::from_vec(vec![0.0]);

    let mut prob_state =
        ProbabilisticState::perturbed_samples(&base_model, &q0, &v0, nsamples, 0.01, 0.01);

    // Sample uncertain parameters for each particle
    println!("Sampling uncertain parameters:");
    println!("  - Length:  mean={}, std=0.1", base_length);
    println!("  - Mass:    mean={}, std=0.05", base_mass);
    println!("  - Damping: uniform[0.0, 0.1]\n");

    prob_state.sample_parameters(|rng| {
        vec![
            Normal::new(base_length, 0.1).unwrap().sample(rng),
            Normal::new(base_mass, 0.05).unwrap().sample(rng),
            Uniform::new(0.0, 0.1).sample(rng),
        ]
    });

    let solver = SimpleSolver;

    // Parameter transformation function
    let apply_params = |_base: &Model, params: &[f64]| -> Model {
        let length = params[0];
        let mass = params[1];
        let damping = params[2];
        make_pendulum(length, mass, damping)
    };

    // Simulate ensemble
    let total_steps = 5000; // 5 seconds
    let print_interval = 1000;

    println!("time(s)    q_mean     q_std      v_mean     v_std      ESS");
    println!("──────────────────────────────────────────────────────────────");

    for step in 0..=total_steps {
        if step % print_interval == 0 {
            let ((q_mean, v_mean), (q_std, v_std)) = prob_state.mean_and_std();
            let ess = prob_state.q.effective_sample_size();

            println!(
                "{:6.3}     {:+7.4}    {:7.4}    {:+7.4}    {:7.4}    {:5.1}",
                prob_state.time, q_mean[0], q_std[0], v_mean[0], v_std[0], ess
            );
        }

        if step < total_steps {
            ensemble_step_with_params(&base_model, &mut prob_state, &solver, apply_params);
        }
    }

    println!("\n=== Summary ===");
    let ((q_final_mean, v_final_mean), (q_final_std, v_final_std)) = prob_state.mean_and_std();
    println!(
        "Final position: {:.4} ± {:.4} rad",
        q_final_mean[0], q_final_std[0]
    );
    println!(
        "Final velocity: {:.4} ± {:.4} rad/s",
        v_final_mean[0], v_final_std[0]
    );
    println!("\nObservations:");
    println!("  - Standard deviation grows over time (uncertainty propagates)");
    println!("  - Damping causes trajectories to converge to equilibrium");
    println!("  - ESS remains high (no resampling needed for this simple case)");
}
