//! Universal multi-domain physics format for tau.
//!
//! The .phyz JSON format is a unified specification for multi-domain physics simulations.
//! It supports:
//! - Multiple physics domains (rigid body, particles, EM, quantum, etc.)
//! - Cross-domain coupling definitions
//! - Parameter specifications with uncertainty
//! - Import/export from common formats (MJCF, URDF, USD, SDF)

pub mod coupling;
pub mod domain;
pub mod error;
pub mod importers;
pub mod schema;

pub use coupling::{Coupling, CouplingType, ForceTransfer};
pub use domain::{Domain, DomainType, RigidBodyDomain};
pub use error::{Result, TauFormatError};
pub use importers::{from_mjcf, from_urdf};
pub use schema::{PhyzSpec, WorldConfig, export_phyz, load_phyz_model};
