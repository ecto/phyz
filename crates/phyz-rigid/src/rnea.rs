//! Recursive Newton-Euler Algorithm (RNEA) — inverse dynamics.
//!
//! Given (q, v, qdd), compute the required joint torques tau.

use phyz_math::{DVec, SpatialTransform, SpatialVec, Vec3};
use phyz_model::{JointType, Model, State};

/// Compute joint velocity contribution S * qd for any joint type.
fn joint_velocity(joint: &phyz_model::Joint, qd: &[f64]) -> SpatialVec {
    match joint.joint_type {
        JointType::Revolute | JointType::Hinge => {
            SpatialVec::new(joint.axis * qd[0], Vec3::zeros())
        }
        JointType::Prismatic | JointType::Slide => {
            SpatialVec::new(Vec3::zeros(), joint.axis * qd[0])
        }
        JointType::Spherical | JointType::Ball => {
            SpatialVec::new(Vec3::new(qd[0], qd[1], qd[2]), Vec3::zeros())
        }
        JointType::Free => SpatialVec::new(
            Vec3::new(qd[0], qd[1], qd[2]),
            Vec3::new(qd[3], qd[4], qd[5]),
        ),
        JointType::Fixed => SpatialVec::zero(),
    }
}

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
        let ndof = joint.ndof();

        let x_joint = if ndof == 0 {
            SpatialTransform::identity()
        } else {
            let q_slice = &state.q.as_slice()[q_idx..q_idx + ndof];
            joint.joint_transform_slice(q_slice)
        };
        x_tree[i] = x_joint.compose(&joint.parent_to_joint);

        let v_joint = joint_velocity(joint, &state.v.as_slice()[v_idx..v_idx + ndof]);
        let a_joint = joint_velocity(joint, &qdd.as_slice()[v_idx..v_idx + ndof]);

        if body.parent < 0 {
            vel[i] = v_joint;
            acc[i] = x_tree[i].apply_motion(&a0) + a_joint;
        } else {
            let pi = body.parent as usize;
            let v_parent = x_tree[i].apply_motion(&vel[pi]);
            vel[i] = v_parent + v_joint;

            let a_parent = x_tree[i].apply_motion(&acc[pi]);
            acc[i] = a_parent + vel[i].cross_motion(&v_joint) + a_joint;
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
        let ndof = joint.ndof();

        if ndof == 1 {
            let s_i = joint.motion_subspace();
            tau[v_idx] = s_i.dot(&forces[i]);
        } else if ndof > 1 {
            let s_mat = joint.motion_subspace_matrix(); // 6 x ndof
            let f_vec = nalgebra::DVector::from_column_slice(forces[i].data.as_slice());
            let phyz_vec = s_mat.transpose() * &f_vec; // ndof x 1
            for k in 0..ndof {
                tau[v_idx + k] = phyz_vec[k];
            }
        }

        if body.parent >= 0 {
            let pi = body.parent as usize;
            forces[pi] = forces[pi] + x_tree[i].inv_apply_force(&forces[i]);
        }
    }

    tau
}
