//! Energy computation for rigid body systems.

use tau_model::{Model, State};

/// Compute kinetic energy: 0.5 * v^T * M(q) * v
pub fn kinetic_energy(model: &Model, state: &State) -> f64 {
    let m = crate::crba(model, state);
    0.5 * state.v.dot(&(&m * &state.v))
}

/// Compute gravitational potential energy.
///
/// PE = -sum_i m_i * g^T * x_com_i
pub fn potential_energy(model: &Model, state: &State) -> f64 {
    let (xforms, _) = crate::forward_kinematics(model, state);
    let mut pe = 0.0;

    for (i, body) in model.bodies.iter().enumerate() {
        let mass = body.inertia.mass;
        let com_local = body.inertia.com;

        // Transform CoM to world frame
        let xf = &xforms[i];
        let com_world = xf.rot.transpose() * com_local + xf.pos;

        // PE = -m * g · r
        pe -= mass * model.gravity.dot(&com_world);
    }

    pe
}

/// Total mechanical energy (kinetic + potential).
pub fn total_energy(model: &Model, state: &State) -> f64 {
    kinetic_energy(model, state) + potential_energy(model, state)
}
