//! Concrete SU(3) lattice gauge theory: heatbath/overrelaxation sampling,
//! smoothing, and gluon field observables.
//!
//! Separate from the generic [`crate::Lattice`] because the SU(3) algorithms
//! need matrix sums (staples, clover leaves) that a group-only trait can't
//! express, and because a flat concrete type is much faster.

mod baryon;
mod export;
mod fields;
mod lattice;
mod mat;
mod rng;

pub use baryon::{FluxAccumulator, staircase};
pub use export::FieldFile;
pub use fields::{FieldStrength, PLANES};
pub use lattice::Su3Lattice;
pub use mat::{C64, Su3};
