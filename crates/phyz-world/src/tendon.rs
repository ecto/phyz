//! Tendon model for cable-like actuators spanning multiple bodies.

use phyz_math::{SpatialVec, Vec3};
use phyz_model::{Model, State};

/// A tendon is a spring-damper system that follows a path through multiple bodies.
///
/// It generates forces along the path based on the current length vs. rest length.
#[derive(Debug, Clone)]
pub struct Tendon {
    /// Sequence of body indices defining the tendon path.
    pub path: Vec<usize>,
    /// Spring stiffness (N/m or Nm/rad).
    pub stiffness: f64,
    /// Rest length (m).
    pub rest_length: f64,
    /// Maximum force the tendon can exert (N).
    pub max_force: f64,
    /// Damping coefficient (Ns/m).
    pub damping: f64,
}

impl Tendon {
    /// Create a new tendon with the given parameters.
    pub fn new(path: Vec<usize>, stiffness: f64, rest_length: f64, max_force: f64) -> Self {
        Self {
            path,
            stiffness,
            rest_length,
            max_force,
            damping: 0.0,
        }
    }

    /// Create a new tendon with damping.
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping;
        self
    }

    /// Compute the current length of the tendon.
    ///
    /// Sums the Euclidean distances between consecutive bodies in the path.
    pub fn current_length(&self, state: &State) -> f64 {
        let mut length = 0.0;

        for i in 0..self.path.len().saturating_sub(1) {
            let body_a = self.path[i];
            let body_b = self.path[i + 1];

            if body_a < state.body_xform.len() && body_b < state.body_xform.len() {
                let pos_a = state.body_xform[body_a].pos;
                let pos_b = state.body_xform[body_b].pos;
                length += (pos_b - pos_a).norm();
            }
        }

        length
    }

    /// Compute the rate of change of tendon length.
    ///
    /// This requires velocities, which we approximate from body transforms.
    /// For now, return 0.0 (proper implementation requires forward kinematics with velocities).
    pub fn current_length_rate(&self, _state: &State) -> f64 {
        0.0 // Placeholder
    }

    /// Compute forces applied to each body in the tendon path.
    ///
    /// Returns a vector of (body_idx, force) pairs.
    /// Force is a SpatialVec (6D force/torque in world frame).
    pub fn compute_forces(&self, state: &State) -> Vec<(usize, SpatialVec)> {
        let current_len = self.current_length(state);
        let stretch = current_len - self.rest_length;

        // Spring force: F = -k * (L - L0) - b * dL/dt
        let length_rate = self.current_length_rate(state);
        let mut force_magnitude = -self.stiffness * stretch - self.damping * length_rate;

        // Saturate force
        if force_magnitude.abs() > self.max_force {
            force_magnitude = force_magnitude.signum() * self.max_force;
        }

        // Distribute force along tendon path
        let mut forces = Vec::new();

        for i in 0..self.path.len().saturating_sub(1) {
            let body_a = self.path[i];
            let body_b = self.path[i + 1];

            if body_a < state.body_xform.len() && body_b < state.body_xform.len() {
                let pos_a = state.body_xform[body_a].pos;
                let pos_b = state.body_xform[body_b].pos;

                let direction = (pos_b - pos_a).normalize();
                let force_vec = direction * force_magnitude;

                // Apply force to body_a (pulling toward body_b)
                forces.push((body_a, SpatialVec::new(Vec3::zeros(), force_vec)));

                // Apply opposite force to body_b
                forces.push((body_b, SpatialVec::new(Vec3::zeros(), -force_vec)));
            }
        }

        forces
    }

    /// Check if tendon is valid for the given model.
    pub fn is_valid(&self, model: &Model) -> bool {
        if self.path.len() < 2 {
            return false;
        }

        for &body_idx in &self.path {
            if body_idx >= model.nbodies() {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{SpatialInertia, SpatialTransform, Vec3};
    use phyz_model::ModelBuilder;

    #[test]
    fn test_tendon_length() {
        // Create a simple 2-body chain
        let model = ModelBuilder::new()
            .add_revolute_body(
                "link1",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .add_revolute_body(
                "link2",
                0,
                SpatialTransform::translation(Vec3::new(0.0, 0.0, -1.0)),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build();

        let mut state = model.default_state();

        // Set body transforms manually for testing
        state.body_xform[0] = SpatialTransform::translation(Vec3::new(0.0, 0.0, 0.0));
        state.body_xform[1] = SpatialTransform::translation(Vec3::new(0.0, 0.0, -1.0));

        let tendon = Tendon::new(vec![0, 1], 100.0, 0.5, 10.0);
        let length = tendon.current_length(&state);

        // Distance between (0,0,0) and (0,0,-1) is 1.0
        assert!((length - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tendon_forces() {
        let model = ModelBuilder::new()
            .add_revolute_body(
                "link1",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .add_revolute_body(
                "link2",
                0,
                SpatialTransform::translation(Vec3::new(0.0, 0.0, -1.0)),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build();

        let mut state = model.default_state();
        state.body_xform[0] = SpatialTransform::translation(Vec3::new(0.0, 0.0, 0.0));
        state.body_xform[1] = SpatialTransform::translation(Vec3::new(0.0, 0.0, -1.5));

        // Tendon with rest length 1.0, current length 1.5 -> stretched by 0.5
        let tendon = Tendon::new(vec![0, 1], 100.0, 1.0, 1000.0);
        let forces = tendon.compute_forces(&state);

        // Should have 2 forces (one per body)
        assert_eq!(forces.len(), 2);

        // Force magnitude should be k * stretch = 100 * 0.5 = 50
        let force_mag = forces[0].1.linear().norm();
        assert!((force_mag - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_tendon_force_saturation() {
        let model = ModelBuilder::new()
            .add_revolute_body(
                "link1",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .add_revolute_body(
                "link2",
                0,
                SpatialTransform::translation(Vec3::new(0.0, 0.0, -1.0)),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build();

        let mut state = model.default_state();
        state.body_xform[0] = SpatialTransform::translation(Vec3::new(0.0, 0.0, 0.0));
        state.body_xform[1] = SpatialTransform::translation(Vec3::new(0.0, 0.0, -2.0));

        // Large stretch (1.0) with high stiffness (100) = 100N force
        // But max_force is 10N, so should saturate
        let tendon = Tendon::new(vec![0, 1], 100.0, 1.0, 10.0);
        let forces = tendon.compute_forces(&state);

        let force_mag = forces[0].1.linear().norm();
        assert!((force_mag - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_tendon_validation() {
        let model = ModelBuilder::new()
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build();

        let valid_tendon = Tendon::new(vec![0], 100.0, 1.0, 10.0);
        assert!(!valid_tendon.is_valid(&model)); // Single body path is invalid

        let invalid_tendon = Tendon::new(vec![0, 99], 100.0, 1.0, 10.0);
        assert!(!invalid_tendon.is_valid(&model)); // Body 99 doesn't exist
    }
}
