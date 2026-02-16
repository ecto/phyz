//! Joint types and definitions.

use tau_math::{Mat3, SpatialTransform, SpatialVec, Vec3};

/// Joint type enumeration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointType {
    /// Single rotational DOF about local Z axis.
    Revolute,
    // Future: Prismatic, Spherical, Free, Fixed
}

/// A joint connecting two bodies.
#[derive(Debug, Clone)]
pub struct Joint {
    /// Joint type.
    pub joint_type: JointType,
    /// Transform from parent body frame to joint frame (constant).
    pub parent_to_joint: SpatialTransform,
    /// Joint axis in local frame (for revolute: typically Z).
    pub axis: Vec3,
    /// Damping coefficient.
    pub damping: f64,
    /// Joint position limits [lower, upper] (None = unlimited).
    pub limits: Option<[f64; 2]>,
}

impl Joint {
    /// Create a revolute joint with the given parent-to-joint transform.
    pub fn revolute(parent_to_joint: SpatialTransform) -> Self {
        Self {
            joint_type: JointType::Revolute,
            parent_to_joint,
            axis: Vec3::new(0.0, 0.0, 1.0), // revolute about Z
            damping: 0.0,
            limits: None,
        }
    }

    /// Number of degrees of freedom for this joint type.
    pub fn ndof(&self) -> usize {
        match self.joint_type {
            JointType::Revolute => 1,
        }
    }

    /// Compute the joint transform for the given joint position.
    ///
    /// Returns the Plücker transform from predecessor to successor frame.
    /// For revolute: the successor frame has rotated by +q about the joint axis,
    /// so the coordinate transform (parent→body) uses R(-q) = R(q)^T.
    pub fn joint_transform(&self, q: f64) -> SpatialTransform {
        match self.joint_type {
            JointType::Revolute => {
                // Passive rotation: negate angle for coordinate transform
                let (s, c) = (-q).sin_cos();
                let a = &self.axis;
                let ax = tau_math::skew(a);
                let rot = Mat3::identity() + ax * s + ax * ax * (1.0 - c);
                SpatialTransform::new(rot, Vec3::zeros())
            }
        }
    }

    /// Motion subspace matrix S (6D) for this joint.
    /// For revolute about Z: S = [0 0 1 0 0 0]^T
    pub fn motion_subspace(&self) -> SpatialVec {
        match self.joint_type {
            JointType::Revolute => SpatialVec::new(self.axis, Vec3::zeros()),
        }
    }
}
