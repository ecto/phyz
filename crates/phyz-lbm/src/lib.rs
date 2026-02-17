//! Lattice Boltzmann Method (LBM) for emergent fluid dynamics.
//!
//! Implements D2Q9 (2D) and D3Q19 (3D) lattice Boltzmann models with BGK collision operator.
//! At macroscopic scales, LBM recovers incompressible Navier-Stokes equations.
//!
//! # Example
//!
//! ```
//! use phyz_lbm::LatticeBoltzmann2D;
//! use phyz_math::Vec3;
//!
//! // Lid-driven cavity flow
//! let mut lbm = LatticeBoltzmann2D::new(64, 64, 0.1);  // 64x64 grid, nu=0.1
//! lbm.initialize_uniform(1.0, [0.0, 0.0]);
//!
//! // Set boundary conditions
//! for x in 0..64 {
//!     lbm.set_velocity_bc(x, 63, [0.1, 0.0]);  // lid velocity
//!     lbm.set_no_slip_bc(x, 0);                 // bottom wall
//! }
//!
//! // Simulate
//! for _ in 0..1000 {
//!     lbm.collide_and_stream();
//! }
//!
//! let u = lbm.velocity(32, 32);
//! println!("Center velocity: [{:.4}, {:.4}]", u[0], u[1]);
//! ```

pub mod d2q9;
pub mod d3q19;
pub mod equation_free;

pub use d2q9::LatticeBoltzmann2D;
pub use d3q19::LatticeBoltzmann3D;
pub use equation_free::{CoarseProjector, EquationFreeWrapper, FineSolver, effective_information};

/// Lattice sound speed: c_s = 1/sqrt(3)
pub const C_S: f64 = 0.577350269189626;

/// Lattice sound speed squared
pub const C_S_SQ: f64 = 1.0 / 3.0;
