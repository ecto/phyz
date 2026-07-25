//! Inverse problems and parameter estimation for phyz physics engine.
//!
//! Provides tools for matching simulation to real-world observations (real2sim):
//! - Trajectory matching against motion capture or sensor data
//! - Parameter estimation via gradient descent
//! - Observation adapters for different sensor types
//! - Loss functions for trajectory comparison

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

pub mod observation;
pub mod optimizer;
pub mod trajectory_matcher;

pub use observation::{EndEffectorPoseObserver, JointAngleObserver, ObservationAdapter};
pub use optimizer::{GradientDescentOptimizer, Optimizer, OptimizerConfig};
pub use trajectory_matcher::{
    LossWeights, PhysicsParams, Trajectory, TrajectoryMatcher, TrajectoryObservation,
};
