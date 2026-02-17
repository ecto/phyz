//! Conservation law monitoring for physics simulations.
//!
//! Tracks energy, momentum, and angular momentum conservation to detect
//! numerical drift and instabilities.

use phyz_math::Vec3;
use phyz_model::{Model, State};
use phyz_rigid::{forward_kinematics, total_energy};

/// Baseline conservation quantities to track drift.
#[derive(Debug, Clone)]
pub struct ConservationState {
    pub baseline_energy: f64,
    pub baseline_momentum: Vec3,
    pub baseline_angular_momentum: Vec3,
}

impl ConservationState {
    /// Initialize conservation state from current model and state.
    pub fn new(model: &Model, state: &State) -> Self {
        Self {
            baseline_energy: total_energy(model, state),
            baseline_momentum: total_momentum(model, state),
            baseline_angular_momentum: total_angular_momentum(model, state),
        }
    }
}

/// Conservation law errors at current timestep.
#[derive(Debug, Clone)]
pub struct ConservationMonitor {
    /// Relative energy error: |E - E₀| / |E₀|
    pub energy_error: f64,
    /// Absolute momentum error: |p - p₀|
    pub momentum_error: Vec3,
    /// Absolute angular momentum error: |L - L₀|
    pub angular_momentum_error: Vec3,
}

impl ConservationMonitor {
    /// Check conservation laws against baseline.
    pub fn check(baseline: &ConservationState, model: &Model, state: &State) -> Self {
        let energy = total_energy(model, state);
        let momentum = total_momentum(model, state);
        let ang_momentum = total_angular_momentum(model, state);

        // Relative energy error (avoid division by zero)
        let energy_error = if baseline.baseline_energy.abs() > 1e-12 {
            (energy - baseline.baseline_energy).abs() / baseline.baseline_energy.abs()
        } else {
            (energy - baseline.baseline_energy).abs()
        };

        Self {
            energy_error,
            momentum_error: momentum - baseline.baseline_momentum,
            angular_momentum_error: ang_momentum - baseline.baseline_angular_momentum,
        }
    }

    /// Check if any conservation law is violated beyond tolerance.
    pub fn is_violated(&self, energy_tol: f64, momentum_tol: f64, ang_momentum_tol: f64) -> bool {
        self.energy_error > energy_tol
            || self.momentum_error.norm() > momentum_tol
            || self.angular_momentum_error.norm() > ang_momentum_tol
    }

    /// Maximum relative error across all conservation laws (normalized).
    pub fn max_relative_error(&self) -> f64 {
        let mom_rel = self.momentum_error.norm() / (1.0 + self.momentum_error.norm());
        let ang_rel =
            self.angular_momentum_error.norm() / (1.0 + self.angular_momentum_error.norm());
        self.energy_error.max(mom_rel).max(ang_rel)
    }
}

/// Compute total linear momentum: sum_i m_i * v_i
pub fn total_momentum(model: &Model, state: &State) -> Vec3 {
    let (_xforms, vels) = forward_kinematics(model, state);
    let mut momentum = Vec3::zeros();

    for (i, body) in model.bodies.iter().enumerate() {
        let mass = body.inertia.mass;
        let vel_linear = vels[i].linear();
        momentum += mass * vel_linear;
    }

    momentum
}

/// Compute total angular momentum about world origin: sum_i (r_i × m_i*v_i + I_i*ω_i)
pub fn total_angular_momentum(model: &Model, state: &State) -> Vec3 {
    let (xforms, vels) = forward_kinematics(model, state);
    let mut ang_momentum = Vec3::zeros();

    for (i, body) in model.bodies.iter().enumerate() {
        let mass = body.inertia.mass;
        let xf = &xforms[i];
        let vel = &vels[i];

        // Transform CoM to world frame
        let com_local = body.inertia.com;
        let com_world = xf.rot.transpose() * com_local + xf.pos;

        // Linear part: r × (m * v)
        let vel_linear = vel.linear();
        ang_momentum += com_world.cross(&(mass * vel_linear));

        // Angular part: I * ω (need to transform inertia to world frame)
        let omega = vel.angular();
        let inertia_local = body.inertia.inertia;
        let inertia_world = xf.rot.transpose() * inertia_local * xf.rot;
        ang_momentum += inertia_world * omega;
    }

    ang_momentum
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform};
    use phyz_model::ModelBuilder;
    use phyz_rigid::aba;

    #[test]
    fn test_conservation_monitor_pendulum() {
        // Single pendulum conserves energy
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
        state.q[0] = 0.1;

        let baseline = ConservationState::new(&model, &state);

        // Step forward with semi-implicit Euler
        for _ in 0..1000 {
            let qdd = aba(&model, &state);
            state.v += &qdd * model.dt;
            state.q += &state.v * model.dt;
        }

        let monitor = ConservationMonitor::check(&baseline, &model, &state);

        // Energy should be conserved to ~1% for semi-implicit Euler
        assert!(
            monitor.energy_error < 0.01,
            "Energy error too large: {}",
            monitor.energy_error
        );
    }

    #[test]
    fn test_momentum_conservation() {
        // Pendulum in zero gravity should conserve momentum
        let model = ModelBuilder::new()
            .gravity(Vec3::zeros())
            .dt(0.01)
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
        // Give it some initial velocity
        state.v[0] = 1.0; // angular velocity

        let baseline = ConservationState::new(&model, &state);

        // Step forward
        for _ in 0..100 {
            let qdd = aba(&model, &state);
            state.v += &qdd * model.dt;
            state.q += &state.v * model.dt;
        }

        let monitor = ConservationMonitor::check(&baseline, &model, &state);

        // Momentum should be conserved (no external forces in zero gravity)
        assert!(
            monitor.momentum_error.norm() < 1e-8,
            "Momentum error too large: {}",
            monitor.momentum_error.norm()
        );
    }
}
