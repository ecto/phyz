//! Material Point Method (MPM) for particles.
//!
//! Implements elastic, plastic, granular, and fluid materials using the
//! Material Point Method with APIC (Affine Particle-In-Cell) transfer.

pub mod material;
pub mod mpm;
pub mod particle;

pub use material::Material;
pub use mpm::MpmSolver;
pub use particle::Particle;
