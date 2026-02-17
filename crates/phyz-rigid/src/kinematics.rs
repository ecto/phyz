//! Forward kinematics — compute body transforms and velocities.

use phyz_math::{SpatialTransform, SpatialVec, Vec3};
use phyz_model::{JointType, Model, State};

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
        let ndof = joint.ndof();

        // x_tree: parent → body (Plücker transform)
        let x_joint = if ndof == 0 {
            SpatialTransform::identity()
        } else {
            let q_slice = &state.q.as_slice()[q_idx..q_idx + ndof];
            joint.joint_transform_slice(q_slice)
        };
        let x_tree = x_joint.compose(&joint.parent_to_joint);

        // Compute velocity contribution from this joint
        let v_joint = joint_velocity(joint, &state.v.as_slice()[v_idx..v_idx + ndof]);

        if body.parent < 0 {
            x_world_to_body[i] = x_tree;
            velocities[i] = v_joint;
        } else {
            let pi = body.parent as usize;
            x_world_to_body[i] = x_tree.compose(&x_world_to_body[pi]);

            // Velocity: v_i = X_tree * v_parent + S_i * qd_i
            let v_parent = x_tree.apply_motion(&velocities[pi]);
            velocities[i] = v_parent + v_joint;
        }
    }

    (x_world_to_body, velocities)
}

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
            // 3 DOF angular velocity
            SpatialVec::new(Vec3::new(qd[0], qd[1], qd[2]), Vec3::zeros())
        }
        JointType::Free => {
            // 6 DOF: [wx, wy, wz, vx, vy, vz]
            SpatialVec::new(
                Vec3::new(qd[0], qd[1], qd[2]),
                Vec3::new(qd[3], qd[4], qd[5]),
            )
        }
        JointType::Fixed => SpatialVec::zero(),
    }
}
