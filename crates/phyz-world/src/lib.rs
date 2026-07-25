//! Procedural world generation for phyz physics engine.
//!
//! This module provides utilities for generating random articulated systems,
//! environments with obstacles, and other procedural physics scenarios.
//!
//! It also provides [`sensor`], whose sensors read real simulation quantities:
//! rangefinders and contact sensors query `phyz-collision` against the
//! kinematic bodies plus a [`Scene`] of static obstacles, and the inertial
//! sensors run forward kinematics and dynamics rather than reporting zeros.

mod generator;
pub mod scene;
pub mod sensor;
pub mod tendon;
pub mod trajectory;
pub mod world;

pub use generator::WorldGenerator;
pub use scene::{Obstacle, PlacedShape, Scene, ShapeOwner};
pub use sensor::{RangeHit, Sensor, SensorContext, SensorOutput};
pub use tendon::Tendon;
pub use trajectory::TrajectoryRecorder;
pub use world::World;
