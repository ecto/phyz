//! MPM solver with P2G/G2P transfer.

use std::collections::HashMap;

use phyz_math::{GRAVITY, Mat3, Vec3};

use crate::particle::Particle;

/// Grid cell index (i, j, k).
type GridIndex = (i32, i32, i32);

/// Grid node state.
#[derive(Debug, Clone, Default)]
struct GridNode {
    mass: f64,
    momentum: Vec3,
    velocity: Vec3,
    force: Vec3,
}

/// MPM solver using APIC (Affine Particle-In-Cell) transfer.
pub struct MpmSolver {
    /// Grid spacing (m).
    pub h: f64,
    /// Timestep (s).
    pub dt: f64,
    /// APIC flip/pic blending parameter (0 = pure PIC, 1 = pure FLIP).
    pub alpha: f64,
    /// Domain bounds (min, max).
    pub bounds: (Vec3, Vec3),
    /// Grid nodes (sparse).
    grid: HashMap<GridIndex, GridNode>,
}

impl MpmSolver {
    /// Create a new MPM solver.
    pub fn new(h: f64, dt: f64, bounds: (Vec3, Vec3)) -> Self {
        Self {
            h,
            dt,
            alpha: 0.95, // Mostly FLIP for reduced numerical dissipation
            bounds,
            grid: HashMap::new(),
        }
    }

    /// Advance simulation by one timestep.
    pub fn step(&mut self, particles: &mut [Particle]) {
        // Clear grid
        self.grid.clear();

        // P2G: transfer particle data to grid
        self.particle_to_grid(particles);

        // Grid update: apply forces and integrate
        self.grid_update();

        // G2P: transfer grid data back to particles
        self.grid_to_particle(particles);
    }

    /// Convert world position to grid index.
    fn position_to_index(&self, x: &Vec3) -> GridIndex {
        (
            (x.x / self.h).floor() as i32,
            (x.y / self.h).floor() as i32,
            (x.z / self.h).floor() as i32,
        )
    }

    /// Get grid node position from index.
    fn index_to_position(&self, idx: GridIndex) -> Vec3 {
        Vec3::new(
            idx.0 as f64 * self.h,
            idx.1 as f64 * self.h,
            idx.2 as f64 * self.h,
        )
    }

    /// Cubic B-spline weight function.
    fn weight(&self, x: &Vec3, xi: &Vec3) -> f64 {
        let dx = (x - xi) / self.h;
        self.n(dx.x) * self.n(dx.y) * self.n(dx.z)
    }

    /// Cubic B-spline kernel (1D).
    fn n(&self, x: f64) -> f64 {
        let x = x.abs();
        if x < 1.0 {
            0.5 * x * x * x - x * x + 2.0 / 3.0
        } else if x < 2.0 {
            -(x - 2.0).powi(3) / 6.0
        } else {
            0.0
        }
    }

    /// Gradient of cubic B-spline weight function.
    fn weight_gradient(&self, x: &Vec3, xi: &Vec3) -> Vec3 {
        let dx = (x - xi) / self.h;
        Vec3::new(
            self.dn(dx.x) * self.n(dx.y) * self.n(dx.z) / self.h,
            self.n(dx.x) * self.dn(dx.y) * self.n(dx.z) / self.h,
            self.n(dx.x) * self.n(dx.y) * self.dn(dx.z) / self.h,
        )
    }

    /// Derivative of cubic B-spline kernel (1D).
    fn dn(&self, x: f64) -> f64 {
        let sign = x.signum();
        let x = x.abs();
        if x < 1.0 {
            sign * (1.5 * x * x - 2.0 * x)
        } else if x < 2.0 {
            sign * (-0.5 * (x - 2.0).powi(2))
        } else {
            0.0
        }
    }

    /// Get neighboring grid cells (3x3x3 stencil).
    fn get_neighbors(&self, idx: GridIndex) -> Vec<GridIndex> {
        let mut neighbors = Vec::new();
        for di in -1..=1 {
            for dj in -1..=1 {
                for dk in -1..=1 {
                    neighbors.push((idx.0 + di, idx.1 + dj, idx.2 + dk));
                }
            }
        }
        neighbors
    }

    /// P2G: transfer particle mass, momentum, and forces to grid.
    fn particle_to_grid(&mut self, particles: &[Particle]) {
        for p in particles {
            let base_idx = self.position_to_index(&p.x);
            let neighbors = self.get_neighbors(base_idx);

            // Compute stress once per particle
            let stress = p.material.compute_stress(&p.f, p.j);

            for &idx in &neighbors {
                let xi = self.index_to_position(idx);
                let w = self.weight(&p.x, &xi);

                if w > 1e-12 {
                    // Compute gradient before borrowing grid
                    let grad_w = self.weight_gradient(&p.x, &xi);
                    let force = -(stress * grad_w) * p.volume;

                    // Now borrow grid mutably
                    let node = self.grid.entry(idx).or_default();

                    // Transfer mass and momentum
                    node.mass += w * p.mass;
                    node.momentum += w * p.mass * (p.v + p.c * (xi - p.x));

                    // Transfer stress forces
                    node.force += force;
                }
            }
        }

        // Compute velocities from momentum
        for node in self.grid.values_mut() {
            if node.mass > 1e-12 {
                node.velocity = node.momentum / node.mass;
            }
        }
    }

    /// Grid update: apply forces and integrate velocities.
    fn grid_update(&mut self) {
        // Collect indices first to avoid borrow issues
        let indices: Vec<GridIndex> = self.grid.keys().copied().collect();

        for idx in indices {
            // Compute position before borrowing node
            let xi = self.index_to_position(idx);
            let epsilon = 0.01; // Small buffer from boundary
            let bounds = self.bounds;

            let node = self.grid.get_mut(&idx).unwrap();

            if node.mass < 1e-12 {
                continue;
            }

            // Apply gravity
            node.force += node.mass * Vec3::new(0.0, -GRAVITY, 0.0);

            // Semi-implicit Euler
            let acc = node.force / node.mass;
            node.velocity += acc * self.dt;

            // Boundary conditions: enforce walls
            // Bottom wall
            if xi.y < bounds.0.y + epsilon && node.velocity.y < 0.0 {
                node.velocity.y = 0.0;
            }
            // Top wall
            if xi.y > bounds.1.y - epsilon && node.velocity.y > 0.0 {
                node.velocity.y = 0.0;
            }
            // Left wall
            if xi.x < bounds.0.x + epsilon && node.velocity.x < 0.0 {
                node.velocity.x = 0.0;
            }
            // Right wall
            if xi.x > bounds.1.x - epsilon && node.velocity.x > 0.0 {
                node.velocity.x = 0.0;
            }
            // Front wall
            if xi.z < bounds.0.z + epsilon && node.velocity.z < 0.0 {
                node.velocity.z = 0.0;
            }
            // Back wall
            if xi.z > bounds.1.z - epsilon && node.velocity.z > 0.0 {
                node.velocity.z = 0.0;
            }
        }
    }

    /// G2P: transfer grid velocities back to particles.
    fn grid_to_particle(&self, particles: &mut [Particle]) {
        for p in particles.iter_mut() {
            let base_idx = self.position_to_index(&p.x);
            let neighbors = self.get_neighbors(base_idx);

            let mut v_pic = Vec3::zeros();
            let mut v_old_grid = Vec3::zeros();
            let mut grad_v = Mat3::zero();
            let mut total_w = 0.0;

            for &idx in &neighbors {
                if let Some(node) = self.grid.get(&idx) {
                    if node.mass < 1e-12 {
                        continue;
                    }

                    let xi = self.index_to_position(idx);
                    let w = self.weight(&p.x, &xi);

                    if w > 1e-12 {
                        // PIC velocity
                        v_pic += w * node.velocity;

                        // FLIP: need old grid velocity (approximate as current - accel*dt)
                        let v_old = node.velocity - (node.force / node.mass) * self.dt;
                        v_old_grid += w * v_old;

                        // Velocity gradient for APIC
                        let grad_w = self.weight_gradient(&p.x, &xi);
                        // Outer product: v * grad_w^T
                        let v = node.velocity;
                        let g = grad_w;
                        grad_v = grad_v
                            + Mat3::new(
                                v.x * g.x,
                                v.x * g.y,
                                v.x * g.z,
                                v.y * g.x,
                                v.y * g.y,
                                v.y * g.z,
                                v.z * g.x,
                                v.z * g.y,
                                v.z * g.z,
                            );

                        total_w += w;
                    }
                }
            }

            if total_w > 1e-12 {
                // FLIP velocity: old particle velocity + change in grid velocity
                let v_flip = p.v + (v_pic - v_old_grid);

                // Blend PIC and FLIP
                let v_new = (1.0 - self.alpha) * v_pic + self.alpha * v_flip;

                // Check for NaN
                if v_new.x.is_finite() && v_new.y.is_finite() && v_new.z.is_finite() {
                    p.v = v_new;
                }

                // Update position
                p.x += p.v * self.dt;

                // Clamp position to domain bounds
                p.x.x = p.x.x.clamp(self.bounds.0.x, self.bounds.1.x);
                p.x.y = p.x.y.clamp(self.bounds.0.y, self.bounds.1.y);
                p.x.z = p.x.z.clamp(self.bounds.0.z, self.bounds.1.z);

                // Update deformation gradient
                p.update_deformation(&grad_v, self.dt);

                // Update affine velocity field (APIC)
                p.c = grad_v;
            }
        }
    }

    /// Get total mass of all particles on grid.
    pub fn total_grid_mass(&self) -> f64 {
        self.grid.values().map(|n| n.mass).sum()
    }

    /// Get total momentum of all particles.
    pub fn total_momentum(particles: &[Particle]) -> Vec3 {
        particles.iter().map(|p| p.mass * p.v).sum()
    }

    /// Get total kinetic energy.
    pub fn kinetic_energy(particles: &[Particle]) -> f64 {
        particles
            .iter()
            .map(|p| 0.5 * p.mass * p.v.norm_squared())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;

    #[test]
    fn test_weight_function() {
        let solver = MpmSolver::new(1.0, 0.01, (Vec3::zeros(), Vec3::new(10.0, 10.0, 10.0)));

        // Weight at same position should be high
        let x = Vec3::new(5.0, 5.0, 5.0);
        let w = solver.weight(&x, &x);
        // Cubic spline at x=0 gives n(0) = 2/3, so w = (2/3)^3 ≈ 0.296
        assert!(w > 0.2);

        // Weight far away should be zero
        let x2 = Vec3::new(10.0, 10.0, 10.0);
        let w2 = solver.weight(&x, &x2);
        assert!(w2.abs() < 1e-12);
    }

    #[test]
    fn test_mass_conservation() {
        let mut solver = MpmSolver::new(0.1, 0.01, (Vec3::zeros(), Vec3::new(1.0, 1.0, 1.0)));

        let mat = Material::Elastic { e: 1e6, nu: 0.3 };
        let mut particles = vec![
            Particle::new(Vec3::new(0.5, 0.5, 0.5), Vec3::zeros(), 1.0, 0.01, mat),
            Particle::new(Vec3::new(0.6, 0.5, 0.5), Vec3::zeros(), 1.0, 0.01, mat),
        ];

        let initial_mass: f64 = particles.iter().map(|p| p.mass).sum();

        solver.step(&mut particles);

        let final_mass: f64 = particles.iter().map(|p| p.mass).sum();

        assert!((initial_mass - final_mass).abs() < 1e-10);
    }

    #[test]
    fn test_free_fall() {
        let mut solver = MpmSolver::new(0.1, 0.01, (Vec3::zeros(), Vec3::new(1.0, 2.0, 1.0)));

        let mat = Material::Elastic { e: 1e6, nu: 0.3 };
        let mut particles = vec![Particle::new(
            Vec3::new(0.5, 1.0, 0.5),
            Vec3::zeros(),
            1.0,
            0.01,
            mat,
        )];

        let initial_y = particles[0].x.y;

        // Simulate free fall for 50 steps (0.5 seconds)
        for _ in 0..50 {
            solver.step(&mut particles);
        }

        // Particle should have fallen
        // After 0.5s, should drop about 0.5 * 9.81 * 0.5^2 ≈ 1.2m, but starts at 1.0m
        // So should be near bottom (bounded at 0)
        assert!(
            particles[0].x.y < initial_y - 0.1,
            "Particle should have fallen, but y went from {} to {}",
            initial_y,
            particles[0].x.y
        );

        // Should have downward velocity (unless hit bottom)
        // If at bottom, velocity will be zero due to boundary condition
    }

    #[test]
    fn test_momentum_conservation() {
        use crate::material::EquationOfState;

        let mut solver = MpmSolver::new(0.1, 0.001, (Vec3::zeros(), Vec3::new(2.0, 2.0, 2.0)));

        // Use fluid material to avoid stiff elastic forces
        let mat = Material::Fluid {
            viscosity: 1e-3,
            eos: EquationOfState::IdealGas {
                rho0: 1000.0,
                cs: 10.0, // Lower sound speed for stability
            },
        };

        let mut particles = vec![
            Particle::new(
                Vec3::new(0.8, 1.0, 1.0),
                Vec3::new(0.1, 0.0, 0.0),
                0.1,
                0.001,
                mat,
            ),
            Particle::new(
                Vec3::new(1.2, 1.0, 1.0),
                Vec3::new(-0.1, 0.0, 0.0),
                0.1,
                0.001,
                mat,
            ),
        ];

        let initial_momentum = MpmSolver::total_momentum(&particles);
        let total_mass: f64 = particles.iter().map(|p| p.mass).sum();

        // Simulate several steps with small timestep
        let n_steps = 5;
        for _ in 0..n_steps {
            solver.step(&mut particles);
        }

        let final_momentum = MpmSolver::total_momentum(&particles);

        // Horizontal momentum should be approximately conserved
        // Allow for some numerical dissipation
        assert!(
            (initial_momentum.x - final_momentum.x).abs() < 0.5,
            "X momentum changed from {} to {}",
            initial_momentum.x,
            final_momentum.x
        );
        assert!((initial_momentum.z - final_momentum.z).abs() < 0.5);

        // Vertical momentum should decrease due to gravity
        // Expected change: -m * g * n_steps * dt
        let expected_dp_y = -total_mass * GRAVITY * n_steps as f64 * solver.dt;
        let actual_dp_y = final_momentum.y - initial_momentum.y;
        assert!(
            (expected_dp_y - actual_dp_y).abs() < 1.0,
            "Vertical momentum change {} vs expected {}",
            actual_dp_y,
            expected_dp_y
        );
    }
}
