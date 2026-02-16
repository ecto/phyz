//! tau — Multi-physics differentiable simulation engine.
//!
//! This is the umbrella crate that provides the `Simulator` and re-exports
//! core types from sub-crates.

pub use tau_collision::{self};
pub use tau_contact::{self, ContactMaterial};
pub use tau_diff::{self, StepJacobians};
pub use tau_math::{self, Vec3};
pub use tau_mjcf::{self};
pub use tau_model::{self, Actuator, Geometry, Model, ModelBuilder, State};
pub use tau_particle::{self, Material, MpmSolver, Particle, material};
pub use tau_prob::{self};
pub use tau_rigid::{self, aba, aba_with_external_forces, crba, forward_kinematics, rnea};
pub use tau_world::{
    self, Sensor, SensorOutput, Tendon, TrajectoryRecorder, World, WorldGenerator,
};

pub use tau_compile::{self};
pub use tau_coupling::{self};
pub use tau_em::{self};
pub use tau_format::{self};
pub use tau_gpu::{self};
pub use tau_gravity::{self};
pub use tau_guardian::{self};
pub use tau_lbm::{self};
pub use tau_md::{self};
pub use tau_qft::{self};
pub use tau_real2sim::{self};

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

        // Update body transforms via FK
        let (xforms, _) = forward_kinematics(model, state);
        state.body_xform = xforms;
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

        // Update body transforms via FK
        let (xforms, _) = forward_kinematics(model, state);
        state.body_xform = xforms;
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

    /// Advance simulation with contact detection and resolution.
    ///
    /// 1. Runs FK to get body transforms and velocities
    /// 2. Detects ground contacts (and body-body contacts if geometries provided)
    /// 3. Computes contact forces
    /// 4. Runs ABA with contact forces as external forces
    /// 5. Integrates and updates FK
    pub fn step_with_contacts(
        &self,
        model: &Model,
        state: &mut State,
        ground_height: f64,
        material: &ContactMaterial,
    ) {
        let dt = model.dt;

        // Run FK to get current transforms and velocities
        let (xforms, velocities) = forward_kinematics(model, state);
        state.body_xform = xforms;

        // Collect body geometries
        let geometries: Vec<Option<tau_model::Geometry>> = model
            .bodies
            .iter()
            .map(|b| b.geometry.clone())
            .collect();

        // Find ground contacts
        let mut contacts =
            tau_contact::find_ground_contacts(state, &geometries, ground_height);

        // Find body-body contacts
        let body_contacts = tau_contact::find_contacts(model, state, &geometries);
        contacts.extend(body_contacts);

        if contacts.is_empty() {
            // No contacts — standard step
            let qdd = aba(model, state);
            state.v += &qdd * dt;
            state.q += &state.v * dt;
        } else {
            // Compute contact spatial forces per body
            let materials = vec![material.clone()];
            let spatial_forces =
                tau_contact::contact_forces(&contacts, state, &materials, Some(&velocities));

            // Run ABA with external forces
            let qdd = aba_with_external_forces(model, state, Some(&spatial_forces));
            state.v += &qdd * dt;
            state.q += &state.v * dt;
        }

        state.time += dt;

        // Update body transforms
        let (xforms, _) = forward_kinematics(model, state);
        state.body_xform = xforms;
    }
}

impl Default for Simulator {
    fn default() -> Self {
        Self::new()
    }
}
