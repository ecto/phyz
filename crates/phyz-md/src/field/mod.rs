//! Structure-of-arrays MD engine: whole-system potentials, integrators, and
//! minimization over plain slices.
//!
//! Unlike the particle/[`crate::system`] API (AoS `Particle`s with a pairwise
//! [`crate::forcefield::ForceField`]), this module exposes the MD numerics as
//! free-standing operations over caller-owned arrays (`&[[f64; 3]]` positions,
//! `&[f64]` masses, …). That makes it embeddable in hosts that keep their own
//! system representation (e.g. vcad's `vcad-kernel-atoms`, which binds an IR
//! `MoleculeSystem` and delegates force/integration numerics here).
//!
//! Contents:
//! - [`cell`] — triclinic lattice cells and minimum-image displacement
//! - [`potentials`] — Lennard-Jones (per-species + Lorentz-Berthelot mixing,
//!   optional energy shift), direct Coulomb, harmonic bonds and angles
//! - [`verlet`] — velocity-Verlet step with optional Berendsen thermostat
//! - [`fire`](mod@fire) — FIRE energy minimization
//! - [`units`] — the Å / eV / amu / fs / e / K unit constants shared by all of
//!   the above
//!
//! All quantities follow the "metal-like" unit convention documented in
//! [`units`].

pub mod cell;
pub mod fire;
pub mod potentials;
pub mod units;
pub mod verlet;

pub use cell::{Lattice, min_image};
pub use fire::{FireOptions, FireResult, fire};
pub use potentials::{Coulomb, HarmonicAngles, HarmonicBonds, LennardJones};
pub use verlet::{Berendsen, verlet_step};
