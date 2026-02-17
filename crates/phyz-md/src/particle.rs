//! Particle representation for molecular dynamics.

use phyz_math::Vec3;

/// A single particle in the MD simulation.
#[derive(Clone, Debug)]
pub struct Particle {
    /// Position (Å or nm).
    pub x: Vec3,
    /// Velocity (Å/fs or nm/ps).
    pub v: Vec3,
    /// Force accumulator (eV/Å or similar).
    pub f: Vec3,
    /// Mass (amu or kg).
    pub mass: f64,
    /// Atom type for force field lookup.
    pub atom_type: u32,
}

impl Particle {
    /// Create a new particle.
    pub fn new(x: Vec3, v: Vec3, mass: f64, atom_type: u32) -> Self {
        Self {
            x,
            v,
            f: Vec3::zeros(),
            mass,
            atom_type,
        }
    }

    /// Reset force accumulator.
    pub fn reset_force(&mut self) {
        self.f = Vec3::zeros();
    }

    /// Add force to accumulator.
    pub fn add_force(&mut self, f: Vec3) {
        self.f += f;
    }

    /// Kinetic energy: 0.5 * m * v^2.
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.v.norm_squared()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_creation() {
        let p = Particle::new(Vec3::new(1.0, 2.0, 3.0), Vec3::zeros(), 1.0, 0);
        assert_eq!(p.x, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(p.mass, 1.0);
    }

    #[test]
    fn test_kinetic_energy() {
        let p = Particle::new(Vec3::zeros(), Vec3::new(1.0, 0.0, 0.0), 2.0, 0);
        assert!((p.kinetic_energy() - 1.0).abs() < 1e-10);
    }
}
