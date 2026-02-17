//! Procedural world generation for phyz physics engine.
//!
//! This module provides utilities for generating random articulated systems,
//! environments with obstacles, and other procedural physics scenarios.

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
