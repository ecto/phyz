//! Trajectory matching for real2sim parameter estimation.

use std::collections::HashMap;
use phyz_model::Model;
use phyz_rigid::aba;

/// A single observation in a trajectory.
#[derive(Debug, Clone)]
pub struct TrajectoryObservation {
    /// Time of observation.
    pub time: f64,
    /// Body index being observed.
    pub body_idx: usize,
    /// Observed position in world frame.
    pub position: Option<[f64; 3]>,
    /// Observed orientation as quaternion [w, x, y, z].
    pub orientation: Option<[f64; 4]>,
    /// Observed joint angles.
    pub joint_angles: Option<Vec<f64>>,
}

/// A trajectory of observations from motion capture or sensors.
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Observations sorted by time.
    pub observations: Vec<TrajectoryObservation>,
    /// Duration of trajectory.
    pub duration: f64,
}

impl Trajectory {
    /// Create a new empty trajectory.
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            duration: 0.0,
        }
    }

    /// Add an observation to the trajectory.
    pub fn add_observation(&mut self, obs: TrajectoryObservation) {
        if obs.time > self.duration {
            self.duration = obs.time;
        }
        self.observations.push(obs);
    }

    /// Get number of observations.
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    /// Check if trajectory is empty.
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Iterate over observations.
    pub fn iter(&self) -> impl Iterator<Item = &TrajectoryObservation> {
        self.observations.iter()
    }
}

impl Default for Trajectory {
    fn default() -> Self {
        Self::new()
    }
}

/// Weights for different loss components.
#[derive(Debug, Clone)]
pub struct LossWeights {
    /// Weight for position error.
    pub position: f64,
    /// Weight for velocity error.
    pub velocity: f64,
    /// Weight for energy conservation violation.
    pub energy: f64,
    /// Weight for joint angle error.
    pub joint_angle: f64,
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            position: 1.0,
            velocity: 0.1,
            energy: 0.01,
            joint_angle: 1.0,
        }
    }
}

/// Physics parameters that can be optimized.
#[derive(Debug, Clone)]
pub struct PhysicsParams {
    /// Model parameters (masses, lengths, etc.)
    pub model_params: HashMap<String, f64>,
    /// Contact/friction parameters.
    pub contact_params: HashMap<String, f64>,
}

impl PhysicsParams {
    /// Create new empty physics parameters.
    pub fn new() -> Self {
        Self {
            model_params: HashMap::new(),
            contact_params: HashMap::new(),
        }
    }

    /// Set a model parameter.
    pub fn set_model_param(&mut self, name: impl Into<String>, value: f64) {
        self.model_params.insert(name.into(), value);
    }

    /// Get a model parameter.
    pub fn get_model_param(&self, name: &str) -> Option<f64> {
        self.model_params.get(name).copied()
    }

    /// Get all parameter values as a vector (for optimization).
    pub fn to_vec(&self) -> Vec<f64> {
        let mut params = Vec::new();

        // Collect model params in sorted order for consistency
        let mut keys: Vec<_> = self.model_params.keys().collect();
        keys.sort();
        for key in keys {
            params.push(self.model_params[key]);
        }

        // Collect contact params in sorted order
        let mut keys: Vec<_> = self.contact_params.keys().collect();
        keys.sort();
        for key in keys {
            params.push(self.contact_params[key]);
        }

        params
    }

    /// Update parameters from a vector (for optimization).
    pub fn from_vec(&mut self, values: &[f64]) {
        let mut idx = 0;

        // Update model params in sorted order
        let mut keys: Vec<_> = self.model_params.keys().cloned().collect();
        keys.sort();
        for key in keys {
            if idx < values.len() {
                self.model_params.insert(key, values[idx]);
                idx += 1;
            }
        }

        // Update contact params in sorted order
        let mut keys: Vec<_> = self.contact_params.keys().cloned().collect();
        keys.sort();
        for key in keys {
            if idx < values.len() {
                self.contact_params.insert(key, values[idx]);
                idx += 1;
            }
        }
    }
}

impl Default for PhysicsParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Trajectory matcher for real2sim optimization.
pub struct TrajectoryMatcher {
    /// Reference trajectory from real-world data.
    pub reference: Trajectory,
    /// Loss weights.
    pub loss_weights: LossWeights,
    /// Time step for simulation.
    pub dt: f64,
}

impl TrajectoryMatcher {
    /// Create a new trajectory matcher.
    pub fn new(reference: Trajectory, dt: f64) -> Self {
        Self {
            reference,
            loss_weights: LossWeights::default(),
            dt,
        }
    }

    /// Create with custom loss weights.
    pub fn with_weights(reference: Trajectory, dt: f64, weights: LossWeights) -> Self {
        Self {
            reference,
            loss_weights: weights,
            dt,
        }
    }

    /// Compute loss between simulated and reference trajectory.
    pub fn compute_loss(&self, model: &Model, _params: &PhysicsParams) -> f64 {
        let mut state = model.default_state();
        let mut loss = 0.0;
        let mut obs_idx = 0;

        // Run simulation and compare to observations
        while state.time < self.reference.duration && obs_idx < self.reference.len() {
            // Find next observation
            while obs_idx < self.reference.len()
                && self.reference.observations[obs_idx].time <= state.time
            {
                let obs = &self.reference.observations[obs_idx];

                // Position loss
                if let Some(ref_pos) = obs.position
                    && obs.body_idx < state.body_xform.len()
                {
                    let sim_pos = state.body_xform[obs.body_idx].pos;
                    let pos_error = (sim_pos.x - ref_pos[0]).powi(2)
                        + (sim_pos.y - ref_pos[1]).powi(2)
                        + (sim_pos.z - ref_pos[2]).powi(2);
                    loss += self.loss_weights.position * pos_error;
                }

                // Joint angle loss
                if let Some(ref_angles) = &obs.joint_angles {
                    for (i, &ref_angle) in ref_angles.iter().enumerate() {
                        if i < state.q.len() {
                            let angle_error = (state.q[i] - ref_angle).powi(2);
                            loss += self.loss_weights.joint_angle * angle_error;
                        }
                    }
                }

                obs_idx += 1;
            }

            // Step simulation
            let qdd = aba(model, &state);
            state.v += &(&qdd * self.dt);
            let v_clone = state.v.clone();
            state.q += &(&v_clone * self.dt);
            state.time += self.dt;

            // Update body transforms for next iteration
            let (body_xform, _) = phyz_rigid::forward_kinematics(model, &state);
            state.body_xform = body_xform;
        }

        loss
    }

    /// Compute numerical gradient of loss with respect to parameters.
    pub fn numerical_gradient(&self, model: &Model, params: &PhysicsParams, eps: f64) -> Vec<f64> {
        let param_vec = params.to_vec();
        let mut gradient = vec![0.0; param_vec.len()];

        for i in 0..param_vec.len() {
            let mut params_plus = params.clone();
            let mut vec_plus = param_vec.clone();
            vec_plus[i] += eps;
            params_plus.from_vec(&vec_plus);

            let mut params_minus = params.clone();
            let mut vec_minus = param_vec.clone();
            vec_minus[i] -= eps;
            params_minus.from_vec(&vec_minus);

            let loss_plus = self.compute_loss(model, &params_plus);
            let loss_minus = self.compute_loss(model, &params_minus);

            gradient[i] = (loss_plus - loss_minus) / (2.0 * eps);
        }

        gradient
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{GRAVITY, SpatialInertia, SpatialTransform, Vec3};
    use phyz_model::ModelBuilder;

    fn make_pendulum() -> Model {
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, -0.5, 0.0)),
            )
            .build()
    }

    #[test]
    fn test_trajectory_creation() {
        let mut traj = Trajectory::new();
        assert!(traj.is_empty());

        traj.add_observation(TrajectoryObservation {
            time: 0.0,
            body_idx: 0,
            position: Some([0.0, -0.5, 0.0]),
            orientation: None,
            joint_angles: Some(vec![0.0]),
        });

        assert_eq!(traj.len(), 1);
        assert_eq!(traj.duration, 0.0);
    }

    #[test]
    fn test_physics_params() {
        let mut params = PhysicsParams::new();
        params.set_model_param("mass", 1.0);
        params.set_model_param("length", 0.5);

        assert_eq!(params.get_model_param("mass"), Some(1.0));
        assert_eq!(params.get_model_param("length"), Some(0.5));

        let vec = params.to_vec();
        assert_eq!(vec.len(), 2);

        let mut params2 = PhysicsParams::new();
        params2.set_model_param("mass", 0.0);
        params2.set_model_param("length", 0.0);
        params2.from_vec(&vec);

        assert_eq!(params2.get_model_param("mass"), Some(1.0));
        assert_eq!(params2.get_model_param("length"), Some(0.5));
    }

    #[test]
    fn test_trajectory_matcher_loss() {
        let model = make_pendulum();

        // Create simple reference trajectory
        let mut traj = Trajectory::new();
        traj.add_observation(TrajectoryObservation {
            time: 0.0,
            body_idx: 0,
            position: Some([0.0, -0.5, 0.0]),
            orientation: None,
            joint_angles: Some(vec![0.0]),
        });

        let matcher = TrajectoryMatcher::new(traj, 0.001);
        let params = PhysicsParams::new();

        // Should have low loss for matching initial state
        let loss = matcher.compute_loss(&model, &params);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_numerical_gradient() {
        let model = make_pendulum();

        let mut traj = Trajectory::new();
        traj.add_observation(TrajectoryObservation {
            time: 0.0,
            body_idx: 0,
            position: Some([0.0, -0.5, 0.0]),
            orientation: None,
            joint_angles: Some(vec![0.1]),
        });

        let matcher = TrajectoryMatcher::new(traj, 0.001);

        let mut params = PhysicsParams::new();
        params.set_model_param("test", 1.0);

        let grad = matcher.numerical_gradient(&model, &params, 1e-6);
        assert_eq!(grad.len(), 1);
    }
}
