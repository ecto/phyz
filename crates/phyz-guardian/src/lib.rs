//! Self-correcting simulation with adaptive time-stepping and conservation monitoring.
//!
//! This crate provides:
//! - Conservation law monitoring (energy, momentum, angular momentum)
//! - Adaptive time-stepping with PI controller
//! - Shadow Hamiltonian tracking for symplectic integrators
//! - Solver quality assessment and graceful degradation
//! - Multi-rate integration (r-RESPA)

#![warn(missing_docs)]

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

pub mod conservation;
pub mod degradation;
pub mod multi_rate;
pub mod time_step;

pub use conservation::{
    ConservationMonitor, ConservationState, total_angular_momentum, total_momentum,
};
pub use degradation::{AutoSwitchController, DegradationStrategy, SolverQuality, SolverSuggestion};
pub use multi_rate::{RRespaIntegrator, split_forces_gravity, split_forces_stiffness};
pub use time_step::{AdaptiveTimeStep, EmbeddedRkError, PiController};
