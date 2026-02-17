//! Probabilistic simulation and uncertainty quantification for phyz.
//!
//! Provides:
//! - Distribution-wrapped state for ensemble simulation
//! - Uncertainty propagation through physics
//! - Ensemble methods (particle filters, SVGD)
//! - Randomized smoothing for contacts

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
