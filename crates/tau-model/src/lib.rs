//! Model and state types for tau physics engine.
//!
//! `Model` is the static description of a physical system (topology, masses, joint types).
//! `State` is the mutable simulation state (positions, velocities, forces).

pub mod body;
pub mod joint;
pub mod model;
pub mod state;

pub use body::Body;
pub use joint::{Joint, JointType};
pub use model::{Model, ModelBuilder};
pub use state::State;
