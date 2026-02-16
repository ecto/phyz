//! Multi-scale coupling for different physics solvers.
//!
//! Couples different solvers (QM→MM, EM→rigid, particle→rigid) via:
//! - Handshake regions: overlapping spatial domains for force exchange
//! - Subcycling: run fast solvers at higher frequency
//! - Force transfer: direct coupling, flux transfer, or potential barriers
//!
//! # Example
//!
//! ```
//! use tau_coupling::{Coupling, ForceTransfer, BoundingBox, SolverType};
//! use tau_math::Vec3;
//!
//! // Define overlap region between electromagnetic and rigid body solvers
//! let coupling = Coupling {
//!     solver_a: SolverType::Electromagnetic,
//!     solver_b: SolverType::RigidBody,
//!     overlap_region: BoundingBox {
//!         min: Vec3::new(-1.0, -1.0, -1.0),
//!         max: Vec3::new(1.0, 1.0, 1.0),
//!     },
//!     force_transfer: ForceTransfer::Direct { damping: 0.1 },
//! };
//!
//! // Compute Lorentz force on charged body
//! let force = tau_coupling::lorentz_force(
//!     1e-6,  // charge (C)
//!     Vec3::new(0.0, 0.0, 0.0),
//!     Vec3::new(1.0, 0.0, 0.0),  // velocity
//!     &Vec3::new(0.0, 0.0, 1e3), // E field
//!     &Vec3::new(0.0, 1.0, 0.0), // B field
//! );
//! ```

pub mod boundary;
pub mod coupling;
pub mod lorentz;
pub mod subcycling;

pub use boundary::BoundingBox;
pub use coupling::{Coupling, ForceTransfer, SolverType};
pub use lorentz::{lorentz_force, magnetic_torque};
pub use subcycling::{SubcyclingSchedule, TimeScale};
