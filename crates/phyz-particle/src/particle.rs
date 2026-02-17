//! Particle state representation for MPM.

use phyz_math::{Mat3, Vec3};

use crate::material::Material;

/// A material point particle.
#[derive(Debug, Clone)]
pub struct Particle {
    /// Position in world frame (m).
    pub x: Vec3,
    /// Velocity (m/s).
    pub v: Vec3,
    /// Mass (kg).
    pub mass: f64,
    /// Reference volume (m³).
    pub volume: f64,
    /// Deformation gradient F.
    pub f: Mat3,
    /// Affine velocity field C (for APIC).
    pub c: Mat3,
    /// Determinant of F (volume change ratio).
    pub j: f64,
    /// Material constitutive model.
    pub material: Material,
}

impl Particle {
    /// Create a new particle.
    pub fn new(x: Vec3, v: Vec3, mass: f64, volume: f64, material: Material) -> Self {
        Self {
            x,
            v,
            mass,
            volume,
            f: Mat3::identity(),
            c: Mat3::zeros(),
            j: 1.0,
            material,
        }
    }

    /// Update deformation gradient from velocity gradient.
    pub fn update_deformation(&mut self, grad_v: &Mat3, dt: f64) {
        // F_new = (I + dt * ∇v) * F_old
        self.f = (Mat3::identity() + grad_v * dt) * self.f;
        self.j = self.f.determinant();

        // Prevent degenerate deformation
        if !self.j.is_finite() || self.j < 0.01 {
            self.f = Mat3::identity();
            self.j = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;

    #[test]
    fn test_particle_creation() {
        let mat = Material::Elastic { e: 1e6, nu: 0.3 };
        let p = Particle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
            0.1,
            mat,
        );

        assert_eq!(p.mass, 1.0);
        assert_eq!(p.j, 1.0);
        assert_eq!(p.f, Mat3::identity());
    }

    #[test]
    fn test_deformation_update() {
        let mat = Material::Elastic { e: 1e6, nu: 0.3 };
        let mut p = Particle::new(Vec3::zeros(), Vec3::zeros(), 1.0, 0.1, mat);

        // Apply uniform compression
        let grad_v = Mat3::identity() * -0.1; // 10% shrink per second
        p.update_deformation(&grad_v, 0.1);

        // After 0.1s, should be (1 - 0.01) = 0.99 scale factor
        assert!((p.j - 0.99_f64.powi(3)).abs() < 1e-6);
    }
}
