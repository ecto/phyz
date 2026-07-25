//! Featherstone articulated rigid body dynamics.
//!
//! Implements:
//! - Forward kinematics
//! - Articulated Body Algorithm (ABA) for forward dynamics
//! - Recursive Newton-Euler Algorithm (RNEA) for inverse dynamics
//! - Composite Rigid Body Algorithm (CRBA) for mass matrix
//! - Semi-implicit Euler integration

pub mod aba;
pub mod actuation;
pub mod crba;
pub mod energy;
pub mod kinematics;
pub mod rnea;

pub use aba::{aba, aba_with_external_forces};
pub use crba::crba;
pub use energy::{kinetic_energy, potential_energy, total_energy};
pub use kinematics::{BodyKinematics, forward_kinematics, forward_kinematics_acc};
pub use rnea::{body_wrenches, rnea};
