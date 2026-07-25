//! Featherstone articulated rigid body dynamics.
//!
//! Implements:
//! - Forward kinematics
//! - Articulated Body Algorithm (ABA) for forward dynamics
//! - Recursive Newton-Euler Algorithm (RNEA) for inverse dynamics
//! - Composite Rigid Body Algorithm (CRBA) for mass matrix
//! - Semi-implicit Euler integration

pub mod aba;
pub mod crba;
pub mod energy;
pub mod integrate;
pub mod kinematics;
pub mod rnea;

pub use aba::{Dynamics, aba, aba_dynamics, aba_with_external_forces};
pub use crba::crba;
pub use energy::{kinetic_energy, potential_energy, total_energy};
pub use integrate::semi_implicit_euler;
pub use kinematics::forward_kinematics;
pub use rnea::{rnea, rnea_with_wrenches};
