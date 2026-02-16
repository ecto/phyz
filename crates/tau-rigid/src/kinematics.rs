//! Forward kinematics — compute body transforms and velocities.

use tau_math::{SpatialTransform, SpatialVec};
use tau_model::{Model, State};

/// Compute forward kinematics: body transforms and spatial velocities.
///
/// Returns (world_to_body transforms, velocities in body frame).
/// The world_to_body transforms can be inverted to get body positions in world frame.
pub fn forward_kinematics(
    model: &Model,
    state: &State,
) -> (Vec<SpatialTransform>, Vec<SpatialVec>) {
    let nb = model.nbodies();
    let mut x_world_to_body = vec![SpatialTransform::identity(); nb];
    let mut velocities = vec![SpatialVec::zero(); nb];

    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let q_idx = model.q_offsets[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];

        let q = state.q[q_idx];
        let qd = state.v[v_idx];

        // x_tree: parent → body (Plücker transform)
        let x_joint = joint.joint_transform(q);
        let x_tree = x_joint.compose(&joint.parent_to_joint);

        if body.parent < 0 {
            // Root body: world → body = x_tree (since parent IS world)
            x_world_to_body[i] = x_tree;
            velocities[i] = joint.motion_subspace() * qd;
        } else {
            let pi = body.parent as usize;
            // world → body_i = (parent → body_i) composed after (world → parent)
            // compose(x_tree, x_world_to_body[pi]) = "x_world_to_body[pi] then x_tree"
            // = world → parent → body_i ✓
            x_world_to_body[i] = x_tree.compose(&x_world_to_body[pi]);

            // Velocity: v_i = X_tree * v_parent + S_i * qd_i
            let v_parent = x_tree.apply_motion(&velocities[pi]);
            velocities[i] = v_parent + joint.motion_subspace() * qd;
        }
    }

    (x_world_to_body, velocities)
}
