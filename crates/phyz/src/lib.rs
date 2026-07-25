//! phyz — the differentiable rigid-body core.
//!
//! This crate is self-contained: spatial algebra ([`math`]), articulated
//! models ([`model`]), Featherstone dynamics ([`rigid`]), collision
//! ([`collision`]), contact ([`contact`]), and a reverse-mode differentiable
//! rollout ([`diff`]). It does **not** re-export the rest of the phyz
//! workspace — `phyz-gpu`, `phyz-particle`, `phyz-lbm` and friends are
//! separate crates you add explicitly.
//!
//! # Example
//!
//! ```
//! use phyz::{Mat3, ModelBuilder, SpatialInertia, SpatialTransform, Vec3};
//! use phyz::rigid::aba;
//!
//! let model = ModelBuilder::new()
//!     .gravity(Vec3::new(0.0, -9.81, 0.0))
//!     .dt(0.002)
//!     .add_revolute_body(
//!         "pendulum",
//!         -1,
//!         SpatialTransform::identity(),
//!         SpatialInertia::new(
//!             1.0,
//!             Vec3::new(0.0, -0.5, 0.0),
//!             Mat3::from_diagonal(&Vec3::new(0.083, 0.0, 0.083)),
//!         ),
//!     )
//!     .build();
//!
//! let mut state = model.default_state();
//! state.q[0] = 0.3;
//!
//! let qdd = aba(&model, &state);
//! assert_eq!(qdd.len(), model.nv);
//! ```
//!
//! For gradients through a whole trajectory, see [`diff`].

#![warn(missing_docs)]

// Compile every ```rust block in the crate README and the repository README
// as doc-tests, so the documented API cannot drift from the real one.
// `cfg(doctest)` keeps both out of the rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct CrateReadmeDocTests;

#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
pub struct RepoReadmeDocTests;

pub mod collision;
pub mod contact;
pub mod diff;
pub use phyz_math as math;
pub mod model;
pub mod rigid;

// Re-export core types at crate root for convenience
pub use collision::{
    AABB, Collision, epa_penetration, epa_penetration_rot, gjk_distance, gjk_distance_rot,
    sweep_and_prune,
};
pub use contact::{ContactMaterial, contact_forces, find_contacts, find_ground_contacts};
pub use math::{DMat, DVec, GRAVITY, Mat3, Mat4, Vec3, skew};
pub use math::{Quat, SpatialInertia, SpatialMat, SpatialTransform, SpatialVec};
pub use model::{Actuator, Body, Geometry, Joint, JointType, Model, ModelBuilder, State};
pub use rigid::{aba, aba_with_external_forces, crba, forward_kinematics, rnea};
