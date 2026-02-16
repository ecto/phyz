//! Simulation state — mutable per-timestep data.

use tau_math::{DVec, SpatialTransform};

/// Mutable simulation state.
#[derive(Debug, Clone)]
pub struct State {
    /// Generalized positions (joint angles/displacements).
    pub q: DVec,
    /// Generalized velocities.
    pub v: DVec,
    /// Control inputs / applied torques.
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
        Self {
            q: DVec::zeros(nq),
            v: DVec::zeros(nv),
            ctrl: DVec::zeros(nv),
            time: 0.0,
            body_xform: vec![SpatialTransform::identity(); nbodies],
            qfrc_external: DVec::zeros(nv),
        }
    }

    /// Compute total kinetic energy given mass matrix.
    /// KE = 0.5 * v^T * M * v
    pub fn kinetic_energy(&self, mass_matrix: &tau_math::DMat) -> f64 {
        0.5 * self.v.dot(&(mass_matrix * &self.v))
    }
}
