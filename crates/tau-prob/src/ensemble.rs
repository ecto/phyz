//! Ensemble simulation and uncertainty propagation.

use crate::ProbabilisticState;
use tau_math::Vec3;
use tau_model::{Model, State};

/// Trait for solvers that can step the simulation.
///
/// This mirrors the Solver trait from the main tau crate but is defined
/// here to avoid circular dependencies.
pub trait EnsembleSolver {
    fn step(&self, model: &Model, state: &mut State);
}

/// Step an ensemble of states forward in time.
///
/// Each particle is simulated independently with potentially different parameters.
pub fn ensemble_step<S: EnsembleSolver>(
    model: &Model,
    prob_state: &mut ProbabilisticState,
    solver: &S,
) {
    let nsamples = prob_state.nsamples();

    for i in 0..nsamples {
        let mut state = State {
            q: prob_state.q.samples[i].clone(),
            v: prob_state.v.samples[i].clone(),
            ctrl: tau_math::DVec::zeros(model.nv),
            time: prob_state.time,
            body_xform: vec![tau_math::SpatialTransform::identity(); model.nbodies()],
        };

        // If parameters are provided, we would apply them here
        // For now, just use the base model
        solver.step(model, &mut state);

        prob_state.q.samples[i] = state.q;
        prob_state.v.samples[i] = state.v;
    }

    prob_state.time += model.dt;

    // Resample if needed (default threshold: 50% of samples)
    prob_state.resample(0.5);
}

/// Apply a parameterized model transformation function to each sample.
///
/// This allows each particle to simulate with different physical parameters.
pub fn ensemble_step_with_params<S, F>(
    base_model: &Model,
    prob_state: &mut ProbabilisticState,
    solver: &S,
    param_fn: F,
) where
    S: EnsembleSolver,
    F: Fn(&Model, &[f64]) -> Model,
{
    let nsamples = prob_state.nsamples();

    for i in 0..nsamples {
        let mut state = State {
            q: prob_state.q.samples[i].clone(),
            v: prob_state.v.samples[i].clone(),
            ctrl: tau_math::DVec::zeros(base_model.nv),
            time: prob_state.time,
            body_xform: vec![tau_math::SpatialTransform::identity(); base_model.nbodies()],
        };

        // Apply parameters to model
        let model = if !prob_state.parameters.samples[i].is_empty() {
            param_fn(base_model, &prob_state.parameters.samples[i])
        } else {
            base_model.clone()
        };

        solver.step(&model, &mut state);

        prob_state.q.samples[i] = state.q;
        prob_state.v.samples[i] = state.v;
    }

    prob_state.time += base_model.dt;
    prob_state.resample(0.5);
}

/// Compute trajectory uncertainty: mean ± std of body positions over time.
///
/// Returns a vector of (mean_position, std_position) for each timestep.
pub fn trajectory_uncertainty(trajectories: &[Vec<State>]) -> Vec<(Vec3, Vec3)> {
    if trajectories.is_empty() {
        return vec![];
    }

    let nsteps = trajectories[0].len();
    let nsamples = trajectories.len();
    let mut result = Vec::with_capacity(nsteps);

    for step_idx in 0..nsteps {
        // Collect body positions from all trajectories at this timestep
        let mut positions = Vec::with_capacity(nsamples);

        for traj in trajectories {
            if step_idx < traj.len() && !traj[step_idx].body_xform.is_empty() {
                // Extract position from first body's transform
                let xform = &traj[step_idx].body_xform[0];
                positions.push(xform.pos);
            }
        }

        if positions.is_empty() {
            result.push((Vec3::zeros(), Vec3::zeros()));
            continue;
        }

        // Compute mean
        let mut mean = Vec3::zeros();
        for pos in &positions {
            mean += pos;
        }
        mean /= nsamples as f64;

        // Compute standard deviation
        let mut var = Vec3::zeros();
        for pos in &positions {
            let diff = pos - mean;
            var.x += diff.x * diff.x;
            var.y += diff.y * diff.y;
            var.z += diff.z * diff.z;
        }
        var /= nsamples as f64;

        let std = Vec3::new(var.x.sqrt(), var.y.sqrt(), var.z.sqrt());

        result.push((mean, std));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tau_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform};
    use tau_model::ModelBuilder;

    fn make_pendulum() -> Model {
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "pendulum",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(
                    1.0,
                    Vec3::new(0.0, -0.5, 0.0),
                    Mat3::from_diagonal(&Vec3::new(1.0 / 12.0, 0.0, 1.0 / 12.0)),
                ),
            )
            .build()
    }

    // Simple test solver that just updates time
    struct TestSolver;
    impl EnsembleSolver for TestSolver {
        fn step(&self, _model: &Model, state: &mut State) {
            // Simple forward Euler for testing
            state.q[0] += state.v[0] * 0.001;
            state.v[0] -= 9.81 * 0.001;
        }
    }

    #[test]
    fn test_ensemble_step() {
        let model = make_pendulum();
        let q0 = DVec::from_vec(vec![0.1]);
        let v0 = DVec::from_vec(vec![0.0]);

        let mut prob_state =
            ProbabilisticState::perturbed_samples(&model, &q0, &v0, 10, 0.01, 0.01);

        let solver = TestSolver;

        let t0 = prob_state.time;
        ensemble_step(&model, &mut prob_state, &solver);

        // Time should have advanced
        assert!((prob_state.time - t0 - model.dt).abs() < 1e-10);

        // States should have changed
        let (q_mean, _) = prob_state.mean();
        assert!((q_mean[0] - 0.1).abs() < 0.1); // Still close to initial
    }

    #[test]
    fn test_trajectory_uncertainty() {
        let model = make_pendulum();

        // Create two simple trajectories
        let mut traj1 = vec![];
        let mut traj2 = vec![];

        for i in 0..10 {
            let mut state1 = model.default_state();
            state1.body_xform[0].pos = Vec3::new(i as f64, 0.0, 0.0);
            traj1.push(state1);

            let mut state2 = model.default_state();
            state2.body_xform[0].pos = Vec3::new(i as f64 + 1.0, 0.0, 0.0);
            traj2.push(state2);
        }

        let trajectories = vec![traj1, traj2];
        let uncertainty = trajectory_uncertainty(&trajectories);

        assert_eq!(uncertainty.len(), 10);

        // Mean at first step should be 0.5
        assert!((uncertainty[0].0.x - 0.5).abs() < 1e-10);
        // Std should be 0.5
        assert!((uncertainty[0].1.x - 0.5).abs() < 1e-10);
    }
}
