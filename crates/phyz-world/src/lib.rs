//! Procedural world generation for phyz physics engine.
//!
//! This module provides utilities for generating random articulated systems,
//! environments with obstacles, and other procedural physics scenarios.

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

mod generator;
pub mod sensor;
pub mod tendon;
pub mod trajectory;
pub mod world;

pub use generator::WorldGenerator;
pub use sensor::{Sensor, SensorOutput};
pub use tendon::Tendon;
pub use trajectory::TrajectoryRecorder;
pub use world::World;
