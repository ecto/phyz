//! Probabilistic simulation and uncertainty quantification for phyz.
//!
//! Provides:
//! - Distribution-wrapped state for ensemble simulation
//! - Uncertainty propagation through physics
//! - Ensemble methods (particle filters, SVGD)
//! - Randomized smoothing for contacts

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

pub mod distribution;
pub mod ensemble;
pub mod state;
pub mod svgd;

pub use distribution::Distribution;
pub use ensemble::{
    EnsembleSolver, ensemble_step, ensemble_step_with_params, trajectory_uncertainty,
};
pub use state::ProbabilisticState;
pub use svgd::svgd_step;
