//! Particle representation for molecular dynamics.
//!
//! [`MdSystem`](crate::system::MdSystem) stores its state as parallel arrays;
//! `Particle` is the per-atom *view* used to add atoms to a system and to read
//! them back out. It is not the system's storage.

use phyz_math::Vec3;

/// A single particle in the MD simulation.
#[derive(Clone, Debug, PartialEq)]
pub struct Particle {
    /// Position (Å).
    pub x: Vec3,
    /// Velocity (Å/fs).
    pub v: Vec3,
    /// Force (eV/Å).
    pub f: Vec3,
    /// Mass (amu).
    pub mass: f64,
    /// Charge in units of the elementary charge.
    pub charge: f64,
    /// Atom type for force field lookup.
    pub atom_type: u32,
}

impl Particle {
    /// Create a new, uncharged particle.
    pub fn new(x: Vec3, v: Vec3, mass: f64, atom_type: u32) -> Self {
        Self {
            x,
            v,
            f: Vec3::zeros(),
            mass,
            charge: 0.0,
            atom_type,
        }
    }

    /// Set the particle's charge (e).
    pub fn with_charge(mut self, charge: f64) -> Self {
        self.charge = charge;
        self
    }

    /// Reset force accumulator.
    pub fn reset_force(&mut self) {
        self.f = Vec3::zeros();
    }

    /// Add force to accumulator.
    pub fn add_force(&mut self, f: Vec3) {
        self.f += f;
    }

    /// Kinetic energy in eV: `½ m v² / FORCE_TO_ACCEL`.
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.v.norm_squared() / crate::field::units::FORCE_TO_ACCEL
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
        assert_eq!(p.charge, 0.0);
    }

    #[test]
    fn test_kinetic_energy() {
        let p = Particle::new(Vec3::zeros(), Vec3::new(1.0, 0.0, 0.0), 2.0, 0);
        let want = 1.0 / crate::field::units::FORCE_TO_ACCEL;
        assert!((p.kinetic_energy() - want).abs() < 1e-10);
    }

    #[test]
    fn charge_is_carried_through() {
        let p = Particle::new(Vec3::zeros(), Vec3::zeros(), 1.0, 0).with_charge(-0.834);
        assert_eq!(p.charge, -0.834);
    }
}
