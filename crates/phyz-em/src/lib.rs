//! Electromagnetic field simulation using FDTD (Finite-Difference Time-Domain).
//!
//! Implements Yee grid FDTD solver for Maxwell's equations with:
//! - Electric (E) and magnetic (H) field evolution
//! - PML, periodic, and PEC boundary conditions
//! - Point dipoles, plane waves, and current loop sources
//! - Energy conservation and Poynting flux observables
//!
//! # Example
//!
//! ```
//! use phyz_em::{YeeGrid, FdtdSolver, Source, BoundaryCondition};
//! use phyz_math::Vec3;
//!
//! // Create grid
//! let dx = 1e-9; // 1 nm spacing
//! let c = 3e8;
//! let dt = dx / (c * 3_f64.sqrt() * 1.1); // CFL-stable timestep
//! let grid = YeeGrid::new(32, 32, 32, dx, dt);
//!
//! let mut solver = FdtdSolver::new(grid);
//!
//! // Add point dipole source
//! solver.add_source(Source::PointDipole {
//!     pos: Vec3::new(16.0 * dx, 16.0 * dx, 16.0 * dx),
//!     freq: 1e9, // 1 GHz
//!     amplitude: 1.0,
//!     direction: Vec3::new(1.0, 0.0, 0.0),
//! });
//!
//! // Set PML boundary
//! solver.set_boundary(BoundaryCondition::Pml {
//!     order: 2,
//!     sigma_max: 1.0,
//! });
//!
//! // Add probe
//! solver.add_probe(Vec3::new(20.0 * dx, 16.0 * dx, 16.0 * dx));
//!
//! // Run simulation
//! for _ in 0..100 {
//!     solver.step();
//!     if solver.step % 10 == 0 {
//!         let energy = solver.total_energy();
//!         println!("Step {}: Energy = {:.3e} J", solver.step, energy);
//!     }
//! }
//! ```

pub mod boundary;
pub mod fdtd;
pub mod grid;
pub mod solver;
pub mod source;

pub use boundary::BoundaryCondition;
pub use grid::{Array3D, YeeGrid};
pub use solver::FdtdSolver;
pub use source::{Probe, Source};
