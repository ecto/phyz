//! Electromagnetic field simulation using FDTD (Finite-Difference Time-Domain).
//!
//! Yee-grid solver for Maxwell's equations with:
//! - Electric (E) and magnetic (H) field evolution on rectangular cells
//! - A true **CPML** absorbing boundary ([`cpml`]), plus periodic and PEC walls
//! - **Dispersive media** — Drude, Lorentz and Debye poles via ADE
//!   ([`dispersion`])
//! - **TFSF** plane-wave injection ([`tfsf`]) for illuminating scatterers
//! - Point dipoles, plane waves, and current loop sources
//! - Energy, Poynting flux, and spectral analysis observables ([`analysis`])
//!
//! # Example
//!
//! ```
//! use phyz_em::{YeeGrid, FdtdSolver, Source, BoundaryCondition, CpmlConfig};
//! use phyz_math::Vec3;
//!
//! // Create grid
//! let dx = 1e-9; // 1 nm spacing
//! let c = 299_792_458.0;
//! let dt = dx / (c * 3_f64.sqrt()) * 0.99; // CFL-stable timestep
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
//! // Absorb outgoing waves with a 10-cell CPML on every face.
//! solver.set_boundary(BoundaryCondition::Cpml(CpmlConfig::with_thickness(10)));
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
//!
//! # Simulating a metal
//!
//! ```
//! use phyz_em::{YeeGrid, DispersiveMaterial};
//!
//! let dx = 1e-9;
//! let dt = dx / (299_792_458.0 * 3_f64.sqrt()) * 0.99;
//! let mut grid = YeeGrid::new(32, 32, 32, dx, dt);
//!
//! // Drude silver: ε_inf ≈ 3.7, ω_p ≈ 1.4e16 rad/s, γ ≈ 3.2e13 rad/s
//! let silver = grid.add_material(DispersiveMaterial::drude(3.7, 1.4e16, 3.2e13));
//! grid.assign_material_halfspace_z(silver, 20);
//! grid.build_dispersion();
//!
//! // The analytic response is available for validation:
//! let omega = 2.0 * std::f64::consts::PI * 3e8 / 600e-9; // 600 nm
//! let r = grid.material(silver).fresnel_normal(omega);
//! assert!(r.norm() > 0.9); // silver is a good mirror in the visible
//! ```

pub mod analysis;
pub mod boundary;
pub mod cpml;
pub mod dispersion;
pub mod fdtd;
pub mod grid;
pub mod scattering;
pub mod solver;
pub mod source;
pub mod tfsf;

pub use analysis::{db, gaussian_pulse, reflection_db, ricker_pulse};
pub use boundary::BoundaryCondition;
pub use cpml::{Cpml, CpmlConfig};
pub use dispersion::{C64, DispersiveMaterial, Pole};
pub use grid::{Array3D, YeeGrid};
pub use scattering::{CrossSectionMonitor, rayleigh_cross_section};
pub use solver::FdtdSolver;
pub use source::{Probe, Source};
pub use tfsf::{Aux1D, Tfsf};
