//! Molecular dynamics with interatomic potentials.
//!
//! Implements Velocity Verlet integration for MD simulation with:
//! - Lennard-Jones and Coulomb potentials
//! - Harmonic bonded interactions
//! - Neighbor lists for efficient force computation
//! - Periodic boundary conditions
//! - Langevin thermostat
//!
//! # Example
//!
//! ```
//! use phyz_md::{MdSystem, LennardJones, Particle};
//! use phyz_math::Vec3;
//! use std::sync::Arc;
//!
//! // Create argon fluid
//! let lj = Arc::new(LennardJones::argon());
//! let mut system = MdSystem::new(lj, 0.001); // 1 fs timestep
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
//! system.set_box_size(Vec3::new(34.0, 10.0, 10.0));
//!
//! // Initialize velocities at 300K
//! let k_b = 8.617e-5; // eV/K
//! system.initialize_velocities(300.0, k_b);
//!
//! // Run simulation
//! for _ in 0..1000 {
//!     system.step();
//!
//!     if system.step % 100 == 0 {
//!         let ke = system.kinetic_energy();
//!         let pe = system.potential_energy();
//!         let temp = system.temperature(k_b);
//!         println!("Step {}: T={:.1}K, E={:.4} eV", system.step, temp, ke + pe);
//!     }
//! }
//! ```

pub mod forcefield;
pub mod neighbor;
pub mod particle;
pub mod system;

pub use forcefield::{Coulomb, ForceField, HarmonicBond, LennardJones};
pub use neighbor::{NeighborList, minimum_image};
pub use particle::Particle;
pub use system::{Bond, MdSystem, Thermostat};
