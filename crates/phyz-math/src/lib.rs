//! Spatial algebra and math primitives for phyz physics engine.
//!
//! Implements 6D spatial vectors, Plucker transforms, and spatial inertia
//! following Featherstone's conventions.

#![warn(missing_docs)]

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

pub mod spatial;

pub use spatial::{SpatialInertia, SpatialMat, SpatialTransform, SpatialTransformExt, SpatialVec};

/// Unit quaternion for 3D rotations (scalar `w`, vector `v`).
pub type Quat = tang::Quat<f64>;

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

pub use tang::{GRAVITY, skew};
