//! Model and state types for phyz physics engine.
//!
//! `Model` is the static description of a physical system (topology, masses, joint types).
//! `State` is the mutable simulation state (positions, velocities, forces).

#![warn(missing_docs)]

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

pub mod body;
pub mod heightfield;
pub mod joint;
pub mod material;
pub mod model;
pub mod state;
pub mod validate;

pub use body::{Body, GeomInstance, Geometry};
pub use heightfield::Heightfield;
pub use joint::{Joint, JointType};
pub use material::{ContactMaterial, SolImp, SolRef};
pub use model::{Actuator, Model, ModelBuilder};
pub use state::State;
pub use validate::Issue;
