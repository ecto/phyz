//! Adaptive time-stepping controllers and error estimation.
//!
//! Implements:
//! - PI controller for adaptive dt
//! - Embedded RK methods for error estimation
//! - Shadow Hamiltonian monitoring

use phyz_math::DVec;
use phyz_model::{Model, State};

/// PI controller for adaptive time-stepping.
///
/// Adjusts dt based on current and previous error estimates:
/// dt_new = dt_old * (tol/error)^α * (tol/error_prev)^β
///
/// Typical values: α = 0.4, β = 0.2 (PID without derivative term).
#[derive(Debug, Clone)]
pub struct PiController {
    /// Target tolerance for local error
    pub tolerance: f64,
    /// Proportional gain (typically 0.4)
    pub alpha: f64,
    /// Integral gain (typically 0.2)
    pub beta: f64,
    /// Minimum allowed timestep
    pub dt_min: f64,
    /// Maximum allowed timestep
    pub dt_max: f64,
    /// Previous error estimate (for integral term)
    prev_error: f64,
}

impl PiController {
    /// Create a new PI controller with given tolerance.
    pub fn new(tolerance: f64, dt_min: f64, dt_max: f64) -> Self {
        Self {
            tolerance,
            alpha: 0.4,
            beta: 0.2,
            dt_min,
            dt_max,
            prev_error: tolerance, // Initialize to tolerance
        }
    }

    /// Compute new timestep based on current error.
    ///
    /// Returns (new_dt, accept_step).
    pub fn adjust(&mut self, current_dt: f64, error: f64) -> (f64, bool) {
        // Safety factor to be conservative
        let safety = 0.9;

        // Proportional term
        let p_term = if error > 1e-14 {
            (self.tolerance / error).powf(self.alpha)
        } else {
            2.0 // If error is tiny, allow aggressive growth
        };

        // Integral term (based on previous error)
        let i_term = if self.prev_error > 1e-14 {
            (self.tolerance / self.prev_error).powf(self.beta)
        } else {
            1.0
        };

        let factor = safety * p_term * i_term;

        // Clamp factor to avoid wild swings
        let factor = factor.clamp(0.1, 5.0);

        let new_dt = (current_dt * factor).clamp(self.dt_min, self.dt_max);

        // Accept step if error is below tolerance
        let accept = error <= self.tolerance;

        self.prev_error = error;

        (new_dt, accept)
    }

    /// Reset controller state (e.g., after manual dt change).
    pub fn reset(&mut self) {
        self.prev_error = self.tolerance;
    }
}

/// Embedded Runge-Kutta error estimator.
///
/// Uses two RK methods of different orders to estimate local truncation error.
/// Example: RK4 (order 4) and RK5 (order 5) for error estimation.
pub struct EmbeddedRkError;

impl EmbeddedRkError {
    /// Estimate error by comparing RK4 and RK5 solutions.
    ///
    /// Returns maximum relative error over all state components.
    pub fn estimate(
        model: &Model,
        state: &State,
        derivatives_fn: impl Fn(&Model, &State) -> (DVec, DVec),
    ) -> f64 {
        let dt = model.dt;

        // RK4 solution
        let state_rk4 = Self::rk4_step(model, state, dt, &derivatives_fn);

        // RK5 solution (Cash-Karp coefficients)
        let state_rk5 = Self::rk5_step(model, state, dt, &derivatives_fn);

        // Compute error: max relative difference
        let mut max_error = 0.0_f64;

        for i in 0..state.q.len() {
            let err = (state_rk4.q[i] - state_rk5.q[i]).abs();
            let scale = state_rk4.q[i].abs().max(1e-6);
            max_error = max_error.max(err / scale);
        }

        for i in 0..state.v.len() {
            let err = (state_rk4.v[i] - state_rk5.v[i]).abs();
            let scale = state_rk4.v[i].abs().max(1e-6);
            max_error = max_error.max(err / scale);
        }

        max_error
    }

    fn rk4_step(
        model: &Model,
        state: &State,
        dt: f64,
        derivatives_fn: &impl Fn(&Model, &State) -> (DVec, DVec),
    ) -> State {
        let mut result = state.clone();

        // k1
        let (dq1, dv1) = derivatives_fn(model, state);

        // k2
        let mut s2 = state.clone();
        s2.q += &dq1 * (dt / 2.0);
        s2.v += &dv1 * (dt / 2.0);
        let (dq2, dv2) = derivatives_fn(model, &s2);

        // k3
        let mut s3 = state.clone();
        s3.q += &dq2 * (dt / 2.0);
        s3.v += &dv2 * (dt / 2.0);
        let (dq3, dv3) = derivatives_fn(model, &s3);

        // k4
        let mut s4 = state.clone();
        s4.q += &dq3 * dt;
        s4.v += &dv3 * dt;
        let (dq4, dv4) = derivatives_fn(model, &s4);

        // Combine
        result.q += &(&dq1 + &dq2 * 2.0 + &dq3 * 2.0 + &dq4) * (dt / 6.0);
        result.v += &(&dv1 + &dv2 * 2.0 + &dv3 * 2.0 + &dv4) * (dt / 6.0);

        result
    }

    fn rk5_step(
        model: &Model,
        state: &State,
        dt: f64,
        derivatives_fn: &impl Fn(&Model, &State) -> (DVec, DVec),
    ) -> State {
        let mut result = state.clone();

        // Cash-Karp RK5 coefficients (simplified version)
        // k1
        let (dq1, dv1) = derivatives_fn(model, state);

        // k2
        let mut s2 = state.clone();
        s2.q += &dq1 * (dt / 5.0);
        s2.v += &dv1 * (dt / 5.0);
        let (dq2, dv2) = derivatives_fn(model, &s2);

        // k3
        let mut s3 = state.clone();
        s3.q += &(&dq1 * (3.0 / 40.0) + &dq2 * (9.0 / 40.0)) * dt;
        s3.v += &(&dv1 * (3.0 / 40.0) + &dv2 * (9.0 / 40.0)) * dt;
        let (dq3, dv3) = derivatives_fn(model, &s3);

        // k4
        let mut s4 = state.clone();
        s4.q += &(&dq1 * (3.0 / 10.0) - &dq2 * (9.0 / 10.0) + &dq3 * (6.0 / 5.0)) * dt;
        s4.v += &(&dv1 * (3.0 / 10.0) - &dv2 * (9.0 / 10.0) + &dv3 * (6.0 / 5.0)) * dt;
        let (dq4, dv4) = derivatives_fn(model, &s4);

        // k5
        let mut s5 = state.clone();
        s5.q += &(&dq1 * (-11.0 / 54.0) + &dq2 * (5.0 / 2.0) - &dq3 * (70.0 / 27.0)
            + &dq4 * (35.0 / 27.0))
            * dt;
        s5.v += &(&dv1 * (-11.0 / 54.0) + &dv2 * (5.0 / 2.0) - &dv3 * (70.0 / 27.0)
            + &dv4 * (35.0 / 27.0))
            * dt;
        let (dq5, dv5) = derivatives_fn(model, &s5);

        // 5th order solution
        result.q += &(&dq1 * (37.0 / 378.0)
            + &dq3 * (250.0 / 621.0)
            + &dq4 * (125.0 / 594.0)
            + &dq5 * (512.0 / 1771.0))
            * dt;
        result.v += &(&dv1 * (37.0 / 378.0)
            + &dv3 * (250.0 / 621.0)
            + &dv4 * (125.0 / 594.0)
            + &dv5 * (512.0 / 1771.0))
            * dt;

        result
    }
}

/// Adaptive time-stepping manager.
///
/// Combines error estimation with PI controller to automatically adjust dt.
pub struct AdaptiveTimeStep {
    pub controller: PiController,
    /// Number of rejected steps
    pub rejected_steps: usize,
    /// Number of accepted steps
    pub accepted_steps: usize,
}

impl AdaptiveTimeStep {
    /// Create a new adaptive timestepper.
    pub fn new(tolerance: f64, dt_min: f64, dt_max: f64) -> Self {
        Self {
            controller: PiController::new(tolerance, dt_min, dt_max),
            rejected_steps: 0,
            accepted_steps: 0,
        }
    }

    /// Perform one adaptive step.
    ///
    /// Returns (accepted, new_dt).
    pub fn step<F>(&mut self, model: &mut Model, state: &mut State, step_fn: F) -> (bool, f64)
    where
        F: Fn(&Model, &mut State),
    {
        // Estimate error using embedded RK
        let error = EmbeddedRkError::estimate(model, state, |m, s| {
            use phyz_rigid::aba;
            let qdd = aba(m, s);
            (s.v.clone(), qdd)
        });

        // Adjust timestep based on error
        let (new_dt, accept) = self.controller.adjust(model.dt, error);

        if accept {
            // Accept step and advance
            step_fn(model, state);
            self.accepted_steps += 1;
        } else {
            // Reject step, reduce dt and try again
            self.rejected_steps += 1;
        }

        // Update model's dt for next step
        model.dt = new_dt;

        (accept, new_dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pi_controller() {
        let mut controller = PiController::new(1e-4, 1e-6, 1e-2);

        // Error below tolerance should increase dt
        let (new_dt, accept) = controller.adjust(0.001, 1e-5);
        assert!(accept);
        assert!(new_dt > 0.001);

        // Error above tolerance should decrease dt
        let (new_dt, accept) = controller.adjust(0.001, 1e-3);
        assert!(!accept);
        assert!(new_dt < 0.001);
    }

    #[test]
    fn test_pi_controller_clamping() {
        let mut controller = PiController::new(1e-4, 1e-5, 1e-3);

        // Should clamp to dt_max
        let (new_dt, _) = controller.adjust(1e-3, 1e-10);
        assert!((new_dt - 1e-3).abs() < 1e-10);

        // Should clamp to dt_min
        let (new_dt, _) = controller.adjust(1e-5, 1.0);
        assert!((new_dt - 1e-5).abs() < 1e-10);
    }
}
