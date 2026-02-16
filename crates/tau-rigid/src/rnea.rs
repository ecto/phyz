//! Recursive Newton-Euler Algorithm (RNEA) — inverse dynamics.
//!
//! Given (q, v, qdd), compute the required joint torques tau.

use tau_math::{DVec, SpatialTransform, SpatialVec, Vec3};
use tau_model::{Model, State};

/// Run RNEA: compute inverse dynamics torques.
///
/// Given state (q, v) and desired accelerations `qdd`, returns the torques needed.
pub fn rnea(model: &Model, state: &State, qdd: &DVec) -> DVec {
    let nb = model.nbodies();
    let mut tau = DVec::zeros(model.nv);

    let mut x_tree = vec![SpatialTransform::identity(); nb];
    let mut vel = vec![SpatialVec::zero(); nb];
    let mut acc = vec![SpatialVec::zero(); nb];

    let a0 = SpatialVec::new(Vec3::zeros(), -model.gravity);

    // ── Forward pass: velocities and accelerations ──
    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let q_idx = model.q_offsets[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];

        let q = state.q[q_idx];
        let qd = state.v[v_idx];
        let qdd_i = qdd[v_idx];

        let x_joint = joint.joint_transform(q);
        x_tree[i] = x_joint.compose(&joint.parent_to_joint);

        let s_i = joint.motion_subspace();

        if body.parent < 0 {
            vel[i] = s_i * qd;
            acc[i] = x_tree[i].apply_motion(&a0) + s_i * qdd_i;
        } else {
            let pi = body.parent as usize;
            let v_parent = x_tree[i].apply_motion(&vel[pi]);
            vel[i] = v_parent + s_i * qd;

            let a_parent = x_tree[i].apply_motion(&acc[pi]);
            acc[i] = a_parent + vel[i].cross_motion(&(s_i * qd)) + s_i * qdd_i;
        }
    }

    // ── Backward pass: forces and torques ──
    let mut forces = vec![SpatialVec::zero(); nb];
    for i in 0..nb {
        let body = &model.bodies[i];
        let ia = body.inertia.to_matrix();
        forces[i] = ia.mul_vec(&acc[i]) + vel[i].cross_force(&ia.mul_vec(&vel[i]));
    }

    for i in (0..nb).rev() {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let s_i = joint.motion_subspace();

        tau[v_idx] = s_i.dot(&forces[i]);

        if body.parent >= 0 {
            let pi = body.parent as usize;
            forces[pi] = forces[pi] + x_tree[i].inv_apply_force(&forces[i]);
        }
    }

    tau
}
