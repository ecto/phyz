//! Poisson solver for density-dependent gravity (Layer 2).
//!
//! Solves Poisson equation on a grid:
//!   ∇² Φ = 4π G ρ
//!   g = -∇ Φ
//!
//! Currently implements naive finite-difference Jacobi iteration.
//! Future: FFT for periodic boundaries, multigrid for non-periodic.

use crate::{G, GravityParticle, GravitySolver};
use phyz_math::Vec3;

/// Poisson solver for local gravity fields.
#[derive(Debug, Clone)]
pub struct PoissonSolver {
    /// Grid resolution (cells per dimension).
    pub resolution: usize,
    /// Grid bounds (min, max).
    pub bounds: (Vec3, Vec3),
    /// Density grid (kg/m³).
    pub density: Vec<f64>,
    /// Potential grid Φ (m²/s²).
    pub potential: Vec<f64>,
    /// Max Jacobi iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
}

impl PoissonSolver {
    /// Create a new Poisson solver.
    pub fn new(resolution: usize, bounds: (Vec3, Vec3)) -> Self {
        let n = resolution * resolution * resolution;
        Self {
            resolution,
            bounds,
            density: vec![0.0; n],
            potential: vec![0.0; n],
            max_iter: 100,
            tolerance: 1e-6,
        }
    }

    /// Convert 3D index to linear index.
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + j * self.resolution + k * self.resolution * self.resolution
    }

    /// Grid spacing.
    fn dx(&self) -> f64 {
        (self.bounds.1.x - self.bounds.0.x) / self.resolution as f64
    }

    /// Deposit particle mass onto density grid (cloud-in-cell).
    pub fn deposit_density(&mut self, particles: &[GravityParticle]) {
        self.density.fill(0.0);

        let dx = self.dx();
        let min = self.bounds.0;

        for p in particles {
            // Grid coordinates
            let gx = ((p.x.x - min.x) / dx).floor() as isize;
            let gy = ((p.x.y - min.y) / dx).floor() as isize;
            let gz = ((p.x.z - min.z) / dx).floor() as isize;

            // Deposit to nearest grid point (simple NGP scheme)
            if gx >= 0
                && gx < self.resolution as isize
                && gy >= 0
                && gy < self.resolution as isize
                && gz >= 0
                && gz < self.resolution as isize
            {
                let idx = self.idx(gx as usize, gy as usize, gz as usize);
                self.density[idx] += p.m / (dx * dx * dx);
            }
        }
    }

    /// Solve Poisson equation using Jacobi iteration.
    pub fn solve_poisson(&mut self) {
        let n = self.resolution;
        let dx = self.dx();
        let dx2 = dx * dx;

        let mut new_potential = self.potential.clone();

        for _iter in 0..self.max_iter {
            let mut max_delta: f64 = 0.0;

            for i in 1..n - 1 {
                for j in 1..n - 1 {
                    for k in 1..n - 1 {
                        let idx = self.idx(i, j, k);

                        // ∇² Φ = 4π G ρ
                        let rhs = 4.0 * std::f64::consts::PI * G * self.density[idx];

                        // Jacobi update: Φ_new = (sum of neighbors + dx² * rhs) / 6
                        new_potential[idx] = (self.potential[self.idx(i + 1, j, k)]
                            + self.potential[self.idx(i - 1, j, k)]
                            + self.potential[self.idx(i, j + 1, k)]
                            + self.potential[self.idx(i, j - 1, k)]
                            + self.potential[self.idx(i, j, k + 1)]
                            + self.potential[self.idx(i, j, k - 1)]
                            + dx2 * rhs)
                            / 6.0;

                        let delta = (new_potential[idx] - self.potential[idx]).abs();
                        max_delta = max_delta.max(delta);
                    }
                }
            }

            std::mem::swap(&mut self.potential, &mut new_potential);

            if max_delta < self.tolerance {
                break;
            }
        }
    }

    /// Compute gravitational force at a position via finite-difference gradient.
    pub fn force_at(&self, x: Vec3) -> Vec3 {
        let dx = self.dx();
        let min = self.bounds.0;

        // Grid coordinates
        let gx = ((x.x - min.x) / dx).floor() as isize;
        let gy = ((x.y - min.y) / dx).floor() as isize;
        let gz = ((x.z - min.z) / dx).floor() as isize;

        if gx < 1
            || gx >= (self.resolution - 1) as isize
            || gy < 1
            || gy >= (self.resolution - 1) as isize
            || gz < 1
            || gz >= (self.resolution - 1) as isize
        {
            return Vec3::zeros();
        }

        let i = gx as usize;
        let j = gy as usize;
        let k = gz as usize;

        // Central difference: g = -∇Φ
        let grad_x = (self.potential[self.idx(i + 1, j, k)]
            - self.potential[self.idx(i - 1, j, k)])
            / (2.0 * dx);
        let grad_y = (self.potential[self.idx(i, j + 1, k)]
            - self.potential[self.idx(i, j - 1, k)])
            / (2.0 * dx);
        let grad_z = (self.potential[self.idx(i, j, k + 1)]
            - self.potential[self.idx(i, j, k - 1)])
            / (2.0 * dx);

        Vec3::new(-grad_x, -grad_y, -grad_z)
    }
}

impl GravitySolver for PoissonSolver {
    fn compute_forces(&mut self, particles: &mut [GravityParticle]) {
        // Deposit density
        self.deposit_density(particles);

        // Solve Poisson equation
        self.solve_poisson();

        // Compute forces
        for p in particles.iter_mut() {
            p.reset_force();
            let g = self.force_at(p.x);
            p.add_force(g * p.m);
        }
    }

    fn potential_energy(&self, particles: &[GravityParticle]) -> f64 {
        // U = Σ m * Φ(x)
        let dx = self.dx();
        let min = self.bounds.0;

        particles
            .iter()
            .map(|p| {
                let gx = ((p.x.x - min.x) / dx).floor() as isize;
                let gy = ((p.x.y - min.y) / dx).floor() as isize;
                let gz = ((p.x.z - min.z) / dx).floor() as isize;

                if gx >= 0
                    && gx < self.resolution as isize
                    && gy >= 0
                    && gy < self.resolution as isize
                    && gz >= 0
                    && gz < self.resolution as isize
                {
                    let idx = self.idx(gx as usize, gy as usize, gz as usize);
                    p.m * self.potential[idx]
                } else {
                    0.0
                }
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_solver_creation() {
        let solver =
            PoissonSolver::new(10, (Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0)));
        assert_eq!(solver.resolution, 10);
        assert_eq!(solver.density.len(), 1000);
    }

    #[test]
    fn test_density_deposition() {
        let mut solver =
            PoissonSolver::new(10, (Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0)));

        let particles = vec![GravityParticle::new(Vec3::zeros(), Vec3::zeros(), 1000.0)];

        solver.deposit_density(&particles);

        // Check that density is non-zero somewhere
        let total_density: f64 = solver.density.iter().sum();
        assert!(total_density > 0.0);
    }
}
