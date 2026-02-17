//! Constant uniform gravity solver (Layer 1).

use crate::{GravityParticle, GravitySolver};
use phyz_math::Vec3;

/// Constant uniform gravity field.
#[derive(Debug, Clone)]
pub struct ConstantGravity {
    /// Gravitational acceleration vector (m/s²).
    pub g: Vec3,
}

impl ConstantGravity {
    /// Create constant gravity pointing down (Earth-like).
    pub fn new(g: Vec3) -> Self {
        Self { g }
    }

    /// Standard Earth gravity (-9.81 m/s² in z).
    pub fn earth() -> Self {
        Self {
            g: Vec3::new(0.0, 0.0, -crate::G_EARTH),
        }
    }
}

impl GravitySolver for ConstantGravity {
    fn compute_forces(&mut self, particles: &mut [GravityParticle]) {
        for p in particles.iter_mut() {
            p.reset_force();
            p.add_force(self.g * p.m);
        }
    }

    fn potential_energy(&self, particles: &[GravityParticle]) -> f64 {
        // U = -m * g · x (assuming g points down, so U = m*|g|*h)
        particles.iter().map(|p| -p.m * self.g.dot(&p.x)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_gravity() {
        let mut solver = ConstantGravity::earth();
        let mut particles = vec![GravityParticle::new(
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::zeros(),
            1.0,
        )];

        solver.compute_forces(&mut particles);

        // Force should be -9.81 N in z
        assert!((particles[0].f.z + crate::G_EARTH).abs() < 1e-10);
        assert!(particles[0].f.x.abs() < 1e-10);
        assert!(particles[0].f.y.abs() < 1e-10);
    }

    #[test]
    fn test_potential_energy() {
        let solver = ConstantGravity::earth();
        let particles = vec![GravityParticle::new(
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::zeros(),
            2.0,
        )];

        let u = solver.potential_energy(&particles);
        // U = m*g*h = 2.0 * 9.80665 * 10.0 = 196.133
        assert!((u - 196.133).abs() < 1e-3);
    }
}
