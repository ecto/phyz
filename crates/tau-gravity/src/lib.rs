//! Layered gravity solver from constant g to post-Newtonian corrections.
//!
//! Implements 5 gravity layers:
//! 1. Constant gravity (uniform g)
//! 2. Poisson solver (local density fields)
//! 3. N-body with Barnes-Hut tree (O(N log N) far-field)
//! 4. Post-Newtonian corrections (1PN + 2.5PN)
//! 5. Numerical GR (BSSN, future work)
//!
//! # Example
//!
//! ```
//! use tau_gravity::{GravityParticle, PostNewtonianSolver, GravitySolver};
//! use tau_math::Vec3;
//!
//! // Solar system: Mercury perihelion precession
//! let m_sun = 1.989e30;  // kg
//! let m_mercury = 3.285e23;
//!
//! let mut particles = vec![
//!     GravityParticle::new(Vec3::zeros(), Vec3::zeros(), m_sun),
//!     GravityParticle::new(
//!         Vec3::new(57.9e9, 0.0, 0.0),  // 57.9M km
//!         Vec3::new(0.0, 47.4e3, 0.0),  // 47.4 km/s
//!         m_mercury,
//!     ),
//! ];
//!
//! let mut solver = PostNewtonianSolver::new(2.5); // 2.5PN order
//! solver.compute_forces(&mut particles);
//! ```

pub mod constant;
pub mod nbody;
pub mod particle;
pub mod pn;
pub mod poisson;

pub use constant::ConstantGravity;
pub use nbody::{BarnesHutTree, NBodySolver};
pub use particle::GravityParticle;
pub use pn::{PostNewtonianSolver, orbital_elements, perihelion_precession_rate};
pub use poisson::PoissonSolver;

/// Trait for all gravity solvers.
pub trait GravitySolver {
    /// Compute gravitational forces on all particles.
    fn compute_forces(&mut self, particles: &mut [GravityParticle]);

    /// Compute gravitational potential energy of the system.
    fn potential_energy(&self, particles: &[GravityParticle]) -> f64;
}

/// Gravitational constant (m³ kg⁻¹ s⁻²).
pub const G: f64 = 6.67430e-11;

/// Speed of light (m/s).
pub const C: f64 = 299_792_458.0;

/// Standard Earth gravity (m/s²).
pub const G_EARTH: f64 = 9.80665;
