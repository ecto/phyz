//! Molecular dynamics with interatomic potentials.
//!
//! A single structure-of-arrays engine ([`field`]) provides the numerics, and
//! [`MdSystem`] is a stateful driver over it:
//!
//! - Lennard-Jones with per-species Lorentz-Berthelot mixing
//! - Long-range electrostatics by Ewald summation or particle mesh Ewald,
//!   validated against the NaCl Madelung constant
//! - Bonded terms: harmonic bonds and angles, periodic proper torsions, and
//!   harmonic impropers — enough to express an AMBER/CHARMM/OPLS-style force
//!   field
//! - Cell-list Verlet neighbor lists with a skin and displacement-triggered
//!   rebuilds, O(N) per step
//! - Virial and pressure, so NPT is possible at all
//! - Velocity Verlet (NVE) and BAOAB Langevin (NVT) integration, plus
//!   Berendsen and Nosé-Hoover thermostats and a Berendsen barostat
//! - A per-system seeded PRNG, so trajectories are reproducible by choice
//! - FIRE energy minimization
//!
//! # Units
//!
//! Å, eV, amu, fs, e, K. See [`field::units`] — in particular
//! [`field::units::FORCE_TO_ACCEL`], which relates a force in eV/Å on a mass in
//! amu to an acceleration in Å/fs².
//!
//! # Example
//!
//! ```
//! use phyz_md::{LennardJones, MdSystem, Particle};
//! use phyz_math::Vec3;
//!
//! // Create argon fluid
//! let mut system = MdSystem::lennard_jones(LennardJones::argon(), 1.0); // 1 fs
//!
//! // Add particles
//! for i in 0..10 {
//!     system.add_particle(Particle::new(
//!         Vec3::new(i as f64 * 3.4, 0.0, 0.0),
//!         Vec3::zeros(),
//!         39.948, // argon mass (amu)
//!         0,
//!     ));
//! }
//!
//! // Set periodic boundaries
//! system.set_box_size(Vec3::new(34.0, 34.0, 34.0));
//!
//! // Initialize velocities at 300K
//! let k_b = phyz_md::field::units::KB_EV_PER_K;
//! system.initialize_velocities(300.0, k_b);
//! system.compute_forces();
//!
//! // Run simulation
//! for _ in 0..100 {
//!     system.step();
//! }
//! let temp = system.temperature(k_b);
//! let pressure = system.pressure_gpa();
//! ```
//!
//! # Migration from the pre-consolidation API
//!
//! The crate previously shipped two disjoint stacks: an array-of-structs
//! `ForceField`/`NeighborList` path behind `MdSystem`, and this SoA `field`
//! module. They shared no code, and the exported path was the weaker of the
//! two — it could not use the cell lists that already existed. They are now one
//! engine, which changes the following:
//!
//! | Before | Now |
//! |---|---|
//! | `MdSystem::new(Arc<dyn ForceField>, dt)` | `MdSystem::lennard_jones(lj, dt)`, or `MdSystem::new(dt)` plus `set_lennard_jones` / `set_pme` / … |
//! | `forcefield::{ForceField, LennardJones, Coulomb, HarmonicBond}` | [`field::potentials`] equivalents; the `ForceField` trait is gone |
//! | `system.particles: Vec<Particle>` | [`MdSystem::positions`] & co. (SoA); [`MdSystem::particle`] reads one atom back |
//! | `neighbor::NeighborList` (O(N²)) | [`field::neighbor::NeighborList`] (cell lists, O(N)) |
//! | `Bond { i, j, potential }` | [`system::Bond`] `{ i, j, k, r0 }` |
//!
//! The Langevin thermostat's API is unchanged: `set_thermostat`,
//! `clear_thermostat`, `with_seed`/`set_seed`, and the [`Integrator`] enum all
//! behave as before — only the state they act on became structure-of-arrays.
//!
//! Accelerations now go through [`field::units::FORCE_TO_ACCEL`] rather than
//! `a = f/m` on raw amu, so timesteps are in real femtoseconds. Trajectories
//! from the old code will not reproduce numerically; they were not in a
//! self-consistent unit system.

pub mod analysis;
pub mod field;
pub mod particle;
pub mod system;

pub use analysis::Rdf;
pub use field::cell::{Lattice, min_image};
pub use field::dihedral::{DihedralTerm, HarmonicImpropers, ImproperTerm, PeriodicDihedrals};
pub use field::ewald::{Ewald, Pme};
pub use field::neighbor::NeighborList;
pub use field::potentials::{Coulomb, HarmonicAngles, HarmonicBonds, LennardJones};
pub use field::verlet::{Barostat, Berendsen, NoseHoover};
pub use field::virial::Contribution;
pub use particle::Particle;
pub use system::{Bond, Electrostatics, Integrator, MdSystem, Thermostat};
