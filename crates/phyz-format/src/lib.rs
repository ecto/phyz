//! Universal multi-domain physics format for tau.
//!
//! The .phyz JSON format is a unified specification for multi-domain physics simulations.
//! It supports:
//! - Multiple physics domains (rigid body, particles, EM, quantum, etc.)
//! - Cross-domain coupling definitions
//! - Parameter specifications with uncertainty
//! - Import from MJCF (MuJoCo) and URDF (ROS) descriptions
//!
//! USD and SDF import are *not* implemented; [`from_usd`] and [`from_sdf`]
//! return [`TauFormatError::UnsupportedImportFormat`] rather than a stub.

#![warn(missing_docs)]

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

pub mod coupling;
pub mod domain;
pub mod error;
pub mod importers;
pub mod schema;

pub use coupling::{Coupling, CouplingType, ForceTransfer};
pub use domain::{Domain, DomainType, RigidBodyDomain};
pub use error::{Result, TauFormatError};
pub use importers::{from_mjcf, from_sdf, from_urdf, from_urdf_model, from_usd};
pub use schema::{PhyzSpec, WorldConfig, export_phyz, load_phyz_model};
