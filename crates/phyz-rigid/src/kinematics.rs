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

/// Body transforms, velocities, and accelerations from a full kinematics pass.
#[derive(Debug, Clone)]
pub struct BodyKinematics {
    /// World→body Plücker transforms. `pos` is the body origin in world
    /// coordinates; `rot` maps world coordinates into the body frame.
    pub xforms: Vec<SpatialTransform>,
    /// Spatial velocities in each body's own frame.
    pub velocities: Vec<SpatialVec>,
    /// Spatial accelerations in each body's own frame, **excluding** gravity
    /// (so a body in free fall reads a downward acceleration, not zero).
    pub accelerations: Vec<SpatialVec>,
}

impl BodyKinematics {
    /// Classical linear acceleration of body `i`'s origin, in the body frame.
    ///
    /// Featherstone's spatial acceleration is not the time derivative of a
    /// point's velocity; converting to the acceleration an accelerometer bolted
    /// to the body origin would see needs the `ω × v` term.
    pub fn classical_linear_accel(&self, i: usize) -> Vec3 {
        let v = &self.velocities[i];
        let a = &self.accelerations[i];
        a.linear + v.angular.cross(v.linear)
    }
}

/// Forward kinematics extended with accelerations, for a given `qdd`.
///
/// Gravity is deliberately not folded into the base acceleration here (unlike
/// [`crate::rnea()`], which uses that trick), so the results are true kinematic
/// accelerations in an inertial frame.
pub fn forward_kinematics_acc(
    model: &Model,
    state: &State,
    qdd: &phyz_math::DVec,
) -> BodyKinematics {
    let nb = model.nbodies();
    let mut xforms = vec![SpatialTransform::identity(); nb];
    let mut velocities = vec![SpatialVec::zero(); nb];
    let mut accelerations = vec![SpatialVec::zero(); nb];

    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let q_idx = model.q_offsets[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let ndof = joint.ndof();

        let x_joint = if ndof == 0 {
            SpatialTransform::identity()
        } else {
            joint.joint_transform_slice(&state.q.as_slice()[q_idx..q_idx + ndof])
        };
        let x_tree = x_joint.compose(&joint.parent_to_joint);

        let v_joint = joint_velocity(joint, &state.v.as_slice()[v_idx..v_idx + ndof]);
        let a_joint = joint_velocity(joint, &qdd.as_slice()[v_idx..v_idx + ndof]);

        if body.parent < 0 {
            xforms[i] = x_tree;
            velocities[i] = v_joint;
            accelerations[i] = a_joint;
        } else {
            let pi = body.parent as usize;
            xforms[i] = x_tree.compose(&xforms[pi]);
            velocities[i] = x_tree.apply_motion(&velocities[pi]) + v_joint;
            accelerations[i] = x_tree.apply_motion(&accelerations[pi])
                + velocities[i].cross_motion(&v_joint)
                + a_joint;
        }
    }

    BodyKinematics {
        xforms,
        velocities,
        accelerations,
    }
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
