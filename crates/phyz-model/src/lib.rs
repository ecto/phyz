//! Model and state types for phyz physics engine.
//!
//! `Model` is the static description of a physical system (topology, masses, joint types).
//! `State` is the mutable simulation state (positions, velocities, forces).

pub mod body;
pub mod joint;
pub mod model;
pub mod state;
pub mod validate;

pub use body::{Body, GeomOffset, Geometry};
pub use joint::{Joint, JointType};
pub use model::{Actuator, Model, ModelBuilder};
pub use state::State;
pub use validate::Issue;
