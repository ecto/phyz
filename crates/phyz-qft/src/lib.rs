//! Lattice gauge theory with Hybrid Monte Carlo sampling.
//!
//! Implements lattice QCD-like simulations with:
//! - U(1), SU(2), and SU(3) gauge groups
//! - Wilson action for plaquette interactions
//! - Hybrid Monte Carlo (HMC) for sampling
//! - Observables: plaquette expectation, Wilson loops, Polyakov loops
//!
//! # Example
//!
//! ```
//! use phyz_qft::{Lattice, U1, HmcParams};
//!
//! // Create U(1) lattice
//! let mut lattice = Lattice::<U1>::new(4, 4, 4, 4, 1.0);
//!
//! // Configure HMC
//! let params = HmcParams {
//!     n_md_steps: 10,
//!     dt: 0.1,
//!     ..Default::default()
//! };
//!
//! // Thermalize
//! for _ in 0..100 {
//!     lattice.hmc_step(&params);
//! }
//!
//! // Measure plaquette
//! let plaq = lattice.measure_plaquette();
//! println!("⟨Tr U_plaq⟩ = {:.4}", plaq);
//! ```

pub mod group;
pub mod hmc;
pub mod lattice;
pub mod observables;

pub use group::{Group, SU2, SU3, U1};
pub use hmc::{HmcParams, HmcState};
pub use lattice::Lattice;
pub use observables::{Observables, PolyakovLoop, WilsonLoop};
