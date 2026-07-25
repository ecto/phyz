//! Lattice Boltzmann Method (LBM) for emergent fluid dynamics.
//!
//! Implements D2Q9 (2D) and D3Q19 (3D) lattice Boltzmann models. At macroscopic
//! scales, LBM recovers the incompressible Navier–Stokes equations.
//!
//! # What is available
//!
//! - **Collision**: BGK, TRT and (in 2D) full MRT — see [`collision`]. TRT with
//!   `Λ = 3/16` is the default because it removes the viscosity-dependent
//!   bounce-back wall error that makes plain BGK unusable for wall-bounded flow.
//! - **Boundaries**: declared once as [`Boundaries`] and applied automatically
//!   by `step()` — periodic, no-slip, moving wall, velocity inlet (Zou–He),
//!   pressure outlet, symmetry.
//! - **Forcing**: uniform body force via Guo's second-order scheme, correct
//!   under BGK, TRT and MRT.
//! - **Turbulence**: Smagorinsky LES, computed from the non-equilibrium stress
//!   already available during collision.
//! - **Validation**: closed-form and published reference solutions in
//!   [`analytic`], with the benchmark suite in `tests/validation.rs`.
//!
//! # Example
//!
//! ```
//! use phyz_lbm::{boundary, LatticeBoltzmann2D};
//!
//! // Lid-driven cavity flow: walls and lid declared once, applied every step.
//! let mut lbm = LatticeBoltzmann2D::new(64, 64, 0.1)
//!     .with_boundaries(boundary::cavity_2d([0.1, 0.0]));
//! lbm.initialize_uniform(1.0, [0.0, 0.0]);
//! lbm.run(1000);
//!
//! let u = lbm.velocity(32, 32);
//! println!("Center velocity: [{:.4}, {:.4}]", u[0], u[1]);
//! ```

// Lattice kernels index several parallel arrays (E, W, OPP, f, f^eq, the source
// term) by the same direction number. Iterating one of them and indexing the
// rest is strictly less readable than the shared index, so the lint is off here.
#![allow(clippy::needless_range_loop)]

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

pub mod analytic;
pub mod boundary;
pub mod collision;
pub mod d2q9;
pub mod d3q19;
pub mod equation_free;

pub use boundary::{Boundaries, Boundary, Side};
pub use collision::{CollisionModel, MAGIC_BOUNCE_BACK, MAGIC_STABILITY, Turbulence};
pub use d2q9::LatticeBoltzmann2D;
pub use d3q19::LatticeBoltzmann3D;
pub use equation_free::{CoarseProjector, EquationFreeWrapper, FineSolver, effective_information};

/// Lattice sound speed: c_s = 1/sqrt(3)
pub const C_S: f64 = 0.577350269189626;

/// Lattice sound speed squared
pub const C_S_SQ: f64 = 1.0 / 3.0;
