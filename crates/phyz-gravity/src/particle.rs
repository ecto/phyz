//! Gravity particle representation.

use phyz_math::Vec3;

/// A massive particle subject to gravitational forces.
#[derive(Clone, Debug)]
pub struct GravityParticle {
    /// Position (m).
    pub x: Vec3,
    /// Velocity (m/s).
    pub v: Vec3,
    /// Force accumulator (N).
    pub f: Vec3,
    /// Mass (kg).
    pub m: f64,
}

impl GravityParticle {
    /// Create a new gravity particle.
    pub fn new(x: Vec3, v: Vec3, m: f64) -> Self {
        Self {
            x,
            v,
            f: Vec3::zeros(),
            m,
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

    /// Integrate using velocity Verlet (kick-drift-kick).
    pub fn velocity_verlet_step(&mut self, dt: f64) {
        // Half-step velocity update
        self.v += self.f / self.m * (dt / 2.0);

        // Full-step position update
        self.x += self.v * dt;
    }

    /// Complete velocity Verlet after force recomputation.
    pub fn velocity_verlet_complete(&mut self, dt: f64) {
        // Second half-step velocity update
        self.v += self.f / self.m * (dt / 2.0);
    }

    /// Kinetic energy: 0.5 * m * v².
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.m * self.v.norm_squared()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_creation() {
        let p = GravityParticle::new(Vec3::new(1.0, 2.0, 3.0), Vec3::zeros(), 1.0);
        assert_eq!(p.x, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(p.m, 1.0);
    }

    #[test]
    fn test_kinetic_energy() {
        let p = GravityParticle::new(Vec3::zeros(), Vec3::new(1.0, 0.0, 0.0), 2.0);
        assert!((p.kinetic_energy() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_velocity_verlet() {
        let mut p = GravityParticle::new(Vec3::zeros(), Vec3::new(1.0, 0.0, 0.0), 1.0);
        p.f = Vec3::new(1.0, 0.0, 0.0); // 1 N force

        let dt = 0.1;
        p.velocity_verlet_step(dt);

        // x = v0*t + 0.5*a*t^2 = 1.0*0.1 + 0.5*1.0*0.01 = 0.105
        assert!((p.x.x - 0.105).abs() < 1e-10);
    }
}
