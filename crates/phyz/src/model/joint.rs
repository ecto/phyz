//! Joint types and definitions.

use crate::math::{DMat, Mat3, Quat, SpatialTransform, SpatialVec, Vec3};

/// Joint type enumeration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointType {
    /// Single rotational DOF about an axis.
    Revolute,
    /// Single translational DOF along an axis.
    Prismatic,
    /// 3 DOF spherical joint (ball joint) using quaternions.
    Spherical,
    /// 6 DOF free joint (3 translation + 3 rotation).
    Free,
    /// 0 DOF fixed joint (rigid attachment).
    Fixed,
    /// Alias for Revolute (MuJoCo compatibility).
    Hinge,
    /// Alias for Prismatic (MuJoCo compatibility).
    Slide,
    /// Alias for Spherical (MuJoCo compatibility).
    Ball,
}

/// A joint connecting two bodies.
#[derive(Debug, Clone)]
pub struct Joint {
    /// Joint type.
    pub joint_type: JointType,
    /// Transform from parent body frame to joint frame (constant).
    ///
    /// `pos` is the joint origin expressed in parent-body coordinates and
    /// `rot` is the coordinate transform *parent → joint*, i.e. the joint's
    /// orientation in the parent frame is `rot.transpose()`.
    pub parent_to_joint: SpatialTransform,
    /// Joint axis, expressed in the joint frame (for revolute: typically Z).
    pub axis: Vec3,
    /// Viscous damping coefficient (torque or force per unit velocity).
    pub damping: f64,
    /// Static (Coulomb) friction: constant opposing torque/force magnitude.
    pub friction: f64,
    /// Joint position limits [lower, upper] (None = unlimited).
    pub limits: Option<[f64; 2]>,
    /// Actuation effort limit (torque or force), if the source declares one.
    pub effort_limit: Option<f64>,
    /// Velocity limit, if the source declares one.
    pub velocity_limit: Option<f64>,
    /// Name of the joint in the source file (empty if unnamed).
    pub name: String,
}

impl Joint {
    /// Create a joint of the given type with default dynamics and no limits.
    fn bare(joint_type: JointType, parent_to_joint: SpatialTransform, axis: Vec3) -> Self {
        Self {
            joint_type,
            parent_to_joint,
            axis,
            damping: 0.0,
            friction: 0.0,
            limits: None,
            effort_limit: None,
            velocity_limit: None,
            name: String::new(),
        }
    }

    /// Create a revolute joint with the given parent-to-joint transform.
    pub fn revolute(parent_to_joint: SpatialTransform) -> Self {
        // revolute about Z
        Self::bare(
            JointType::Revolute,
            parent_to_joint,
            Vec3::new(0.0, 0.0, 1.0),
        )
    }

    /// Create a prismatic joint with the given parent-to-joint transform and axis.
    pub fn prismatic(parent_to_joint: SpatialTransform, axis: Vec3) -> Self {
        Self::bare(JointType::Prismatic, parent_to_joint, axis)
    }

    /// Create a spherical (ball) joint with the given parent-to-joint transform.
    pub fn spherical(parent_to_joint: SpatialTransform) -> Self {
        // axis unused for spherical
        Self::bare(JointType::Spherical, parent_to_joint, Vec3::zeros())
    }

    /// Create a free joint with the given parent-to-joint transform.
    pub fn free(parent_to_joint: SpatialTransform) -> Self {
        // axis unused for free
        Self::bare(JointType::Free, parent_to_joint, Vec3::zeros())
    }

    /// Create a fixed joint (rigid attachment).
    pub fn fixed(parent_to_joint: SpatialTransform) -> Self {
        Self::bare(JointType::Fixed, parent_to_joint, Vec3::zeros())
    }

    /// Set the joint name (builder style).
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Number of degrees of freedom for this joint type.
    pub fn ndof(&self) -> usize {
        match self.joint_type {
            JointType::Revolute | JointType::Hinge => 1,
            JointType::Prismatic | JointType::Slide => 1,
            JointType::Spherical | JointType::Ball => 3,
            JointType::Free => 6,
            JointType::Fixed => 0,
        }
    }

    /// Compute the joint transform for the given joint position(s).
    ///
    /// Returns the Plücker transform from predecessor to successor frame.
    /// `q` slice should have length >= ndof().
    pub fn joint_transform_slice(&self, q: &[f64]) -> SpatialTransform {
        match self.joint_type {
            JointType::Revolute | JointType::Hinge => {
                // Passive rotation: negate angle for coordinate transform
                let angle = q[0];
                let (s, c) = (-angle).sin_cos();
                let a = &self.axis;
                let ax = crate::math::skew(a);
                let rot = Mat3::identity() + ax * s + ax * ax * (1.0 - c);
                SpatialTransform::new(rot, Vec3::zeros())
            }
            JointType::Prismatic | JointType::Slide => {
                let distance = q[0];
                let pos = self.axis * distance;
                SpatialTransform::new(Mat3::identity(), pos)
            }
            JointType::Spherical | JointType::Ball => {
                // q = [qw, qx, qy, qz] (quaternion components)
                // But we store as exponential coordinates in integration
                // For now, interpret as axis-angle representation (3 DOF)
                let wx = q[0];
                let wy = q[1];
                let wz = q[2];
                let w = Vec3::new(wx, wy, wz);
                let quat = Quat::exp(&w);
                let rot = quat.to_matrix();
                SpatialTransform::new(rot, Vec3::zeros())
            }
            JointType::Free => {
                // q = [x, y, z, wx, wy, wz] (position + exponential coordinates)
                let pos = Vec3::new(q[0], q[1], q[2]);
                let w = Vec3::new(q[3], q[4], q[5]);
                let quat = Quat::exp(&w);
                let rot = quat.to_matrix();
                SpatialTransform::new(rot, pos)
            }
            JointType::Fixed => {
                // No DOF, return identity
                SpatialTransform::identity()
            }
        }
    }

    /// Compute the joint transform for a single-DOF joint (backward compat).
    /// For multi-DOF joints, use joint_transform_slice instead.
    pub fn joint_transform(&self, q: f64) -> SpatialTransform {
        self.joint_transform_slice(&[q])
    }

    /// Motion subspace matrix S for this joint.
    /// Returns a matrix of size 6 × ndof.
    /// For single-DOF joints, returns a 6×1 column vector.
    pub fn motion_subspace_matrix(&self) -> DMat {
        match self.joint_type {
            JointType::Revolute | JointType::Hinge => {
                let s = SpatialVec::new(self.axis, Vec3::zeros());
                let arr = s.as_array();
                DMat::from_row_slice(6, 1, &arr)
            }
            JointType::Prismatic | JointType::Slide => {
                let s = SpatialVec::new(Vec3::zeros(), self.axis);
                let arr = s.as_array();
                DMat::from_row_slice(6, 1, &arr)
            }
            JointType::Spherical | JointType::Ball => {
                // 3 DOF: angular velocity in body frame
                // S = [I_3×3; 0_3×3] (angular part is identity, linear is zero)
                DMat::from_row_slice(
                    6,
                    3,
                    &[
                        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                    ],
                )
            }
            JointType::Free => {
                // 6 DOF: [angular; linear] velocity
                // S = I_6×6
                DMat::identity(6)
            }
            JointType::Fixed => {
                // 0 DOF: empty 6×0 matrix
                DMat::zeros(6, 0)
            }
        }
    }

    /// Motion subspace for single-DOF joints (backward compat).
    /// For multi-DOF joints, use motion_subspace_matrix instead.
    /// Fixed joints return a zero vector.
    pub fn motion_subspace(&self) -> SpatialVec {
        match self.joint_type {
            JointType::Revolute | JointType::Hinge => SpatialVec::new(self.axis, Vec3::zeros()),
            JointType::Prismatic | JointType::Slide => SpatialVec::new(Vec3::zeros(), self.axis),
            JointType::Fixed => SpatialVec::zero(), // 0 DOF
            _ => panic!(
                "motion_subspace() only valid for single-DOF joints; use motion_subspace_matrix() for multi-DOF joints"
            ),
        }
    }
}
