//! Generalized forces entering forward dynamics: actuators + passive joint terms.
//!
//! `Model.actuators` describes controls in *actuator space*; the dynamics need
//! forces in *DOF space*. This module performs that mapping (clamp to
//! `ctrlrange`, scale by `gear`, scatter to the joint's DOF) and adds the
//! passive per-joint terms (damping, spring, dry friction, soft joint limits).
//!
//! Joint limits are penalty forces for now, matching the penalty contact layer.
//! See [`crate::model::Joint::limit_force`] for the rationale and the migration
//! path to a unilateral constraint.

use crate::math::DVec;
use crate::model::{Model, State};

/// Total generalized force applied to each velocity DOF.
///
/// `qfrc = actuator_forces(ctrl) + passive_forces(q, v)`.
pub fn generalized_forces(model: &Model, state: &State) -> DVec {
    let mut qfrc = model.actuator_forces(&state.ctrl, &state.q, &state.v);
    add_passive_forces(model, state, &mut qfrc);
    qfrc
}

/// Add damping, spring, dry friction and soft joint-limit forces into `qfrc`.
pub fn add_passive_forces(model: &Model, state: &State, qfrc: &mut DVec) {
    for (j, joint) in model.joints.iter().enumerate() {
        let ndof = joint.ndof();
        if ndof == 0 {
            continue;
        }
        let q_idx = model.q_offsets[j];
        let v_idx = model.v_offsets[j];
        for k in 0..ndof {
            let q = state.q[q_idx + k];
            let qd = state.v[v_idx + k];
            qfrc[v_idx + k] += joint.passive_force(k, q, qd);
        }
    }
}

/// Armature (rotor) inertia for each velocity DOF.
///
/// Added to the diagonal of the articulated-body inertia `D = S^T I^A S`, which
/// is where MuJoCo's `armature` enters the equations of motion.
pub fn armature_diag(model: &Model) -> DVec {
    let mut a = DVec::zeros(model.nv);
    for (j, joint) in model.joints.iter().enumerate() {
        let v_idx = model.v_offsets[j];
        for k in 0..joint.ndof() {
            a[v_idx + k] = joint.armature;
        }
    }
    a
}
