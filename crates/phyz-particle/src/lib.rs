//! Material Point Method (MPM) for particles.
//!
//! Implements elastic, plastic, granular, and fluid materials using the
//! Material Point Method with APIC (Affine Particle-In-Cell) transfer.

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

pub mod material;
pub mod mpm;
pub mod particle;

pub use material::Material;
pub use mpm::MpmSolver;
pub use particle::Particle;
