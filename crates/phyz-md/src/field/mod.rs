//! Structure-of-arrays MD engine: whole-system potentials, neighbor lists,
//! integrators, and minimization over plain slices.
//!
//! This is *the* MD engine in the crate. [`crate::system::MdSystem`] is a
//! stateful driver on top of it; both operate on the same SoA arrays
//! (`&[[f64; 3]]` positions, `&[f64]` masses, …), which also makes the numerics
//! embeddable in hosts that keep their own system representation (e.g. vcad's
//! `vcad-kernel-atoms`, which binds an IR `MoleculeSystem` and delegates
//! force/integration numerics here).
//!
//! Contents:
//! - [`cell`] — triclinic lattice cells, wrapping, and minimum-image
//!   displacement
//! - [`neighbor`] — cell-list-backed Verlet neighbor lists with a skin and
//!   displacement-triggered rebuilds
//! - [`potentials`] — Lennard-Jones (per-species + Lorentz-Berthelot mixing,
//!   optional energy shift), cutoff Coulomb, harmonic bonds and angles
//! - [`dihedral`] — periodic proper torsions and harmonic impropers
//! - [`ewald`] — Ewald summation and particle mesh Ewald for periodic
//!   electrostatics
//! - [`virial`] — energy/force/virial bundles and the pressure they produce
//! - [`verlet`] — velocity-Verlet stepping with Berendsen and Nosé-Hoover
//!   thermostats and a Berendsen barostat
//! - [`fire`] — FIRE energy minimization
//! - [`units`] — the Å / eV / amu / fs / e / K unit constants shared by all of
//!   the above
//! - [`special`], [`fft`] — the `erf`/`erfc` and radix-2 FFT support the above
//!   need
//!
//! All quantities follow the "metal-like" unit convention documented in
//! [`units`].

pub mod cell;
pub mod dihedral;
pub mod ewald;
pub mod fft;
pub mod fire;
pub mod neighbor;
pub mod potentials;
pub mod special;
pub mod units;
pub mod verlet;
pub mod virial;

pub use cell::{Lattice, min_image};
pub use dihedral::{DihedralTerm, HarmonicImpropers, ImproperTerm, PeriodicDihedrals};
pub use ewald::{Ewald, Pme};
pub use fire::{FireOptions, FireResult, fire};
pub use neighbor::NeighborList;
pub use potentials::{Coulomb, HarmonicAngles, HarmonicBonds, LennardJones};
pub use verlet::{Barostat, Berendsen, NoseHoover, kinetic_energy, verlet_step};
pub use virial::{Contribution, pressure, pressure_tensor};
