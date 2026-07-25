//! Featherstone articulated rigid body dynamics.
//!
//! Implements:
//! - Forward kinematics
//! - Articulated Body Algorithm (ABA) for forward dynamics
//! - Recursive Newton-Euler Algorithm (RNEA) for inverse dynamics
//! - Composite Rigid Body Algorithm (CRBA) for mass matrix
//! - Semi-implicit Euler integration

#![warn(missing_docs)]

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

pub mod aba;
pub mod actuation;
pub mod crba;
pub mod energy;
pub mod integrate;
pub mod jacobian;
pub mod kinematics;
pub mod rnea;

pub use aba::{aba, aba_with_external_forces};
pub use crba::crba;
pub use energy::{kinetic_energy, potential_energy, total_energy};
pub use integrate::semi_implicit_euler;
pub use jacobian::{point_jacobian, relative_point_jacobian};
pub use kinematics::{BodyKinematics, forward_kinematics, forward_kinematics_acc};
pub use rnea::{body_wrenches, rnea};
