//! Inverse problems and parameter estimation for tau physics engine.
//!
//! Provides tools for matching simulation to real-world observations (real2sim):
//! - Trajectory matching against motion capture or sensor data
//! - Parameter estimation via gradient descent
//! - Observation adapters for different sensor types
//! - Loss functions for trajectory comparison

pub mod observation;
pub mod optimizer;
pub mod trajectory_matcher;

pub use observation::{EndEffectorPoseObserver, JointAngleObserver, ObservationAdapter};
pub use optimizer::{GradientDescentOptimizer, Optimizer, OptimizerConfig};
pub use trajectory_matcher::{
    LossWeights, PhysicsParams, Trajectory, TrajectoryMatcher, TrajectoryObservation,
};
