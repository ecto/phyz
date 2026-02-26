//! Spatial algebra and math primitives for phyz physics engine.
//!
//! Implements 6D spatial vectors, Plucker transforms, and spatial inertia
//! following Featherstone's conventions.

pub mod quaternion;
pub mod spatial;

pub use quaternion::Quat;
pub use spatial::{SpatialInertia, SpatialMat, SpatialTransform, SpatialVec};

/// 3D vector alias.
pub type Vec3 = tang::Vec3<f64>;
/// 3x3 matrix alias.
pub type Mat3 = tang::Mat3<f64>;
/// 4x4 matrix alias.
pub type Mat4 = tang::Mat4<f64>;
/// Dynamic vector.
pub type DVec = tang_la::DVec<f64>;
/// Dynamic matrix.
pub type DMat = tang_la::DMat<f64>;

/// Cross-product matrix: [v]x such that [v]x w = v x w.
#[inline]
pub fn skew(v: &Vec3) -> Mat3 {
    Mat3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Standard gravity (m/s^2).
pub const GRAVITY: f64 = 9.81;
