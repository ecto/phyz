//! Spatial algebra and math primitives for phyz physics engine.
//!
//! Implements 6D spatial vectors, Plücker transforms, and spatial inertia
//! following Featherstone's conventions.

pub mod quaternion;
pub mod spatial;

pub use quaternion::Quat;
pub use spatial::{SpatialInertia, SpatialMat, SpatialTransform, SpatialVec};

use nalgebra as na;

/// 3D vector alias.
pub type Vec3 = na::Vector3<f64>;
/// 3x3 matrix alias.
pub type Mat3 = na::Matrix3<f64>;
/// 4x4 matrix alias.
pub type Mat4 = na::Matrix4<f64>;
/// 6D vector alias.
pub type Vec6 = na::Vector6<f64>;
/// 6x6 matrix alias.
pub type Mat6 = na::Matrix6<f64>;
/// Dynamic vector.
pub type DVec = na::DVector<f64>;
/// Dynamic matrix.
pub type DMat = na::DMatrix<f64>;

/// Cross-product matrix: [v]× such that [v]× w = v × w.
#[inline]
pub fn skew(v: &Vec3) -> Mat3 {
    Mat3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Standard gravity (m/s²).
pub const GRAVITY: f64 = 9.81;
