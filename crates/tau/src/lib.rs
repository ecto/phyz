//! tau — Multi-physics differentiable simulation engine.
//!
//! This is the umbrella crate that provides the `Simulator` and re-exports
//! core types from sub-crates.

pub use tau_diff::{self, StepJacobians};
pub use tau_math::{self, Vec3};
pub use tau_mjcf::{self};
pub use tau_model::{self, Model, ModelBuilder, State};
pub use tau_rigid::{self, aba, crba, forward_kinematics, rnea};

/// Pluggable solver trait.
///
/// Implementations define how to advance the simulation by one timestep.
pub trait Solver {
    /// Advance state by dt. Reads from `state` and writes the result back.
    fn step(&self, model: &Model, state: &mut State);

    /// Advance state by dt and return step Jacobians.
    fn step_with_jacobians(&self, model: &Model, state: &mut State) -> StepJacobians;
}

/// Semi-implicit Euler integrator using Featherstone ABA.
pub struct SemiImplicitEulerSolver;

impl Solver for SemiImplicitEulerSolver {
    fn step(&self, model: &Model, state: &mut State) {
        let dt = model.dt;
        let qdd = aba(model, state);

        // Semi-implicit Euler: update velocity first, then position
        state.v += &qdd * dt;
        state.q += &state.v * dt;
        state.time += dt;
    }

    fn step_with_jacobians(&self, model: &Model, state: &mut State) -> StepJacobians {
        let jac = tau_diff::analytical_step_jacobians(model, state);
        self.step(model, state);
        jac
    }
}

/// 4th-order Runge-Kutta integrator.
///
/// Much better energy conservation than semi-implicit Euler for systems
/// with configuration-dependent mass matrices (e.g., double pendulum).
pub struct Rk4Solver;

impl Rk4Solver {
    /// Evaluate derivatives: given (q, v, ctrl) → (dq/dt, dv/dt) = (v, ABA(q,v,ctrl)).
    fn derivatives(model: &Model, state: &State) -> (tau_math::DVec, tau_math::DVec) {
        let qdd = aba(model, state);
        (state.v.clone(), qdd)
    }
}

impl Solver for Rk4Solver {
    fn step(&self, model: &Model, state: &mut State) {
        let dt = model.dt;

        // k1
        let (dq1, dv1) = Self::derivatives(model, state);

        // k2
        let mut s2 = state.clone();
        s2.q += &dq1 * (dt / 2.0);
        s2.v += &dv1 * (dt / 2.0);
        let (dq2, dv2) = Self::derivatives(model, &s2);

        // k3
        let mut s3 = state.clone();
        s3.q += &dq2 * (dt / 2.0);
        s3.v += &dv2 * (dt / 2.0);
        let (dq3, dv3) = Self::derivatives(model, &s3);

        // k4
        let mut s4 = state.clone();
        s4.q += &dq3 * dt;
        s4.v += &dv3 * dt;
        let (dq4, dv4) = Self::derivatives(model, &s4);

        // Combine
        state.q += &(&dq1 + &dq2 * 2.0 + &dq3 * 2.0 + &dq4) * (dt / 6.0);
        state.v += &(&dv1 + &dv2 * 2.0 + &dv3 * 2.0 + &dv4) * (dt / 6.0);
        state.time += dt;
    }

    fn step_with_jacobians(&self, model: &Model, state: &mut State) -> StepJacobians {
        // Use finite-diff Jacobians for RK4 (analytical would need chain rule through 4 stages)
        let jac = tau_diff::finite_diff_jacobians(model, state, 1e-7);
        self.step(model, state);
        jac
    }
}

/// Main simulation driver.
pub struct Simulator {
    solver: Box<dyn Solver>,
}

impl Simulator {
    /// Create a simulator with the default semi-implicit Euler solver.
    pub fn new() -> Self {
        Self {
            solver: Box::new(SemiImplicitEulerSolver),
        }
    }

    /// Create a simulator with the RK4 solver.
    pub fn rk4() -> Self {
        Self {
            solver: Box::new(Rk4Solver),
        }
    }

    /// Create a simulator with a custom solver.
    pub fn with_solver(solver: Box<dyn Solver>) -> Self {
        Self { solver }
    }

    /// Advance simulation by one timestep.
    pub fn step(&self, model: &Model, state: &mut State) {
        self.solver.step(model, state);
    }

    /// Advance simulation by one timestep and return Jacobians.
    pub fn step_with_jacobians(&self, model: &Model, state: &mut State) -> StepJacobians {
        self.solver.step_with_jacobians(model, state)
    }

    /// Run simulation for `n` steps.
    pub fn simulate(&self, model: &Model, state: &mut State, n: usize) {
        for _ in 0..n {
            self.step(model, state);
        }
    }
}

impl Default for Simulator {
    fn default() -> Self {
        Self::new()
    }
}
