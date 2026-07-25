//! Simulation state — mutable per-timestep data.

use crate::math::{DVec, SpatialTransform};

/// Mutable simulation state.
#[derive(Debug, Clone)]
pub struct State {
    /// Generalized positions (joint angles/displacements).
    pub q: DVec,
    /// Generalized velocities.
    pub v: DVec,
    /// Control inputs.
    ///
    /// If the model has actuators, entry `i` is the raw control of actuator `i`
    /// (clamped to its `ctrlrange` and scaled by its `gear` before reaching the
    /// dynamics — see [`Model::actuator_forces`]). If the model has no
    /// actuators, entries are raw per-DOF generalized forces.
    ///
    /// [`Model::actuator_forces`]: crate::Model::actuator_forces
    pub ctrl: DVec,
    /// Simulation time.
    pub time: f64,

    // Cached quantities (filled by forward kinematics / dynamics)
    /// Body-to-world transforms for each body.
    pub body_xform: Vec<SpatialTransform>,
    /// External generalized forces (e.g., from contacts).
    pub qfrc_external: DVec,
}

impl State {
    /// Create a zero-initialized state for `nq` position DOFs and `nv` velocity DOFs.
    pub fn new(nq: usize, nv: usize, nbodies: usize) -> Self {
        Self::new_with_nu(nq, nv, 0, nbodies)
    }

    /// Create a zero-initialized state, sizing `ctrl` for `nu` actuators.
    ///
    /// `ctrl` is sized `max(nv, nu)` so that both conventions (per-DOF forces
    /// for unactuated models, per-actuator controls otherwise) fit.
    pub fn new_with_nu(nq: usize, nv: usize, nu: usize, nbodies: usize) -> Self {
        Self {
            q: DVec::zeros(nq),
            v: DVec::zeros(nv),
            ctrl: DVec::zeros(nv.max(nu)),
            time: 0.0,
            body_xform: vec![SpatialTransform::identity(); nbodies],
            qfrc_external: DVec::zeros(nv),
        }
    }

    /// Compute total kinetic energy given mass matrix.
    /// KE = 0.5 * v^T * M * v
    pub fn kinetic_energy(&self, mass_matrix: &crate::math::DMat) -> f64 {
        0.5 * self.v.dot(&(mass_matrix * &self.v))
    }
}
