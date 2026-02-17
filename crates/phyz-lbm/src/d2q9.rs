//! D2Q9 Lattice Boltzmann solver for 2D incompressible flow.
//!
//! Nine velocity directions on 2D square lattice:
//! ```text
//!   6   2   5
//!    \  |  /
//!   3 - 0 - 1
//!    /  |  \
//!   7   4   8
//! ```

/// D2Q9 Lattice Boltzmann solver for 2D flow.
pub struct LatticeBoltzmann2D {
    /// Grid width
    pub nx: usize,
    /// Grid height
    pub ny: usize,
    /// Kinematic viscosity
    pub nu: f64,
    /// Relaxation time τ = 3ν + 0.5
    pub tau: f64,
    /// Distribution functions f_i at each grid point
    /// Shape: [nx, ny, 9]
    pub f: Vec<f64>,
    /// Temporary buffer for streaming step
    f_temp: Vec<f64>,
}

/// D2Q9 discrete velocities: [vx, vy]
const E: [[i32; 2]; 9] = [
    [0, 0],   // 0: rest
    [1, 0],   // 1: east
    [0, 1],   // 2: north
    [-1, 0],  // 3: west
    [0, -1],  // 4: south
    [1, 1],   // 5: northeast
    [-1, 1],  // 6: northwest
    [-1, -1], // 7: southwest
    [1, -1],  // 8: southeast
];

/// D2Q9 weights
const W: [f64; 9] = [
    4.0 / 9.0, // 0: rest
    1.0 / 9.0, // 1-4: cardinal
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0, // 5-8: diagonal
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

/// Opposite direction indices for bounce-back
const OPP: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

impl LatticeBoltzmann2D {
    /// Create new 2D LBM solver.
    ///
    /// # Arguments
    /// * `nx` - Grid width
    /// * `ny` - Grid height
    /// * `nu` - Kinematic viscosity
    pub fn new(nx: usize, ny: usize, nu: f64) -> Self {
        let tau = 3.0 * nu + 0.5;
        let size = nx * ny * 9;
        Self {
            nx,
            ny,
            nu,
            tau,
            f: vec![0.0; size],
            f_temp: vec![0.0; size],
        }
    }

    /// Initialize with uniform density and velocity.
    pub fn initialize_uniform(&mut self, rho: f64, u: [f64; 2]) {
        for y in 0..self.ny {
            for x in 0..self.nx {
                for i in 0..9 {
                    let feq = self.equilibrium(i, rho, u);
                    self.set_f(x, y, i, feq);
                }
            }
        }
    }

    /// Compute equilibrium distribution f_i^eq.
    ///
    /// f_i^eq = w_i ρ [1 + 3(e_i·u) + 9/2(e_i·u)² - 3/2(u·u)]
    #[inline]
    fn equilibrium(&self, i: usize, rho: f64, u: [f64; 2]) -> f64 {
        let ex = E[i][0] as f64;
        let ey = E[i][1] as f64;
        let eu = ex * u[0] + ey * u[1];
        let uu = u[0] * u[0] + u[1] * u[1];
        W[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uu)
    }

    /// Get distribution function at (x, y, i).
    #[inline]
    fn get_f(&self, x: usize, y: usize, i: usize) -> f64 {
        self.f[x + y * self.nx + i * self.nx * self.ny]
    }

    /// Set distribution function at (x, y, i).
    #[inline]
    fn set_f(&mut self, x: usize, y: usize, i: usize, val: f64) {
        self.f[x + y * self.nx + i * self.nx * self.ny] = val;
    }

    /// Get temporary distribution function at (x, y, i).
    #[inline]
    fn get_f_temp(&self, x: usize, y: usize, i: usize) -> f64 {
        self.f_temp[x + y * self.nx + i * self.nx * self.ny]
    }

    /// Set temporary distribution function at (x, y, i).
    #[inline]
    fn set_f_temp(&mut self, x: usize, y: usize, i: usize, val: f64) {
        self.f_temp[x + y * self.nx + i * self.nx * self.ny] = val;
    }

    /// Compute macroscopic density at (x, y).
    pub fn density(&self, x: usize, y: usize) -> f64 {
        (0..9).map(|i| self.get_f(x, y, i)).sum()
    }

    /// Compute macroscopic velocity at (x, y).
    pub fn velocity(&self, x: usize, y: usize) -> [f64; 2] {
        let rho = self.density(x, y);
        if rho < 1e-12 {
            return [0.0, 0.0];
        }
        let mut u = [0.0, 0.0];
        for (i, e) in E.iter().enumerate() {
            let f = self.get_f(x, y, i);
            u[0] += f * e[0] as f64;
            u[1] += f * e[1] as f64;
        }
        u[0] /= rho;
        u[1] /= rho;
        u
    }

    /// Maximum velocity magnitude in domain.
    pub fn max_velocity(&self) -> f64 {
        let mut umax: f64 = 0.0;
        for y in 0..self.ny {
            for x in 0..self.nx {
                let u = self.velocity(x, y);
                let umag = (u[0] * u[0] + u[1] * u[1]).sqrt();
                umax = umax.max(umag);
            }
        }
        umax
    }

    /// Set velocity boundary condition at (x, y).
    pub fn set_velocity_bc(&mut self, x: usize, y: usize, u: [f64; 2]) {
        let rho = self.density(x, y);
        for i in 0..9 {
            let feq = self.equilibrium(i, rho, u);
            self.set_f(x, y, i, feq);
        }
    }

    /// Set no-slip boundary condition (bounce-back) at (x, y).
    pub fn set_no_slip_bc(&mut self, x: usize, y: usize) {
        // Bounce-back: swap directions
        let f_new: [f64; 9] = std::array::from_fn(|i| self.get_f(x, y, OPP[i]));
        for (i, &val) in f_new.iter().enumerate() {
            self.set_f(x, y, i, val);
        }
    }

    /// Perform collision step (BGK operator).
    fn collide(&mut self) {
        for y in 0..self.ny {
            for x in 0..self.nx {
                let rho = self.density(x, y);
                let u = self.velocity(x, y);

                for i in 0..9 {
                    let f = self.get_f(x, y, i);
                    let feq = self.equilibrium(i, rho, u);
                    let f_new = f - (f - feq) / self.tau;
                    self.set_f_temp(x, y, i, f_new);
                }
            }
        }
    }

    /// Perform streaming step.
    fn stream(&mut self) {
        for y in 0..self.ny {
            for x in 0..self.nx {
                for (i, e) in E.iter().enumerate() {
                    let ex = e[0];
                    let ey = e[1];
                    let xp = (x as i32 + ex).rem_euclid(self.nx as i32) as usize;
                    let yp = (y as i32 + ey).rem_euclid(self.ny as i32) as usize;

                    let f = self.get_f_temp(x, y, i);
                    self.set_f(xp, yp, i, f);
                }
            }
        }
    }

    /// Perform one full LBM step: collision + streaming.
    pub fn collide_and_stream(&mut self) {
        self.collide();
        self.stream();
    }

    /// Compute total kinetic energy.
    pub fn kinetic_energy(&self) -> f64 {
        let mut ke = 0.0;
        for y in 0..self.ny {
            for x in 0..self.nx {
                let rho = self.density(x, y);
                let u = self.velocity(x, y);
                let umag_sq = u[0] * u[0] + u[1] * u[1];
                ke += 0.5 * rho * umag_sq;
            }
        }
        ke
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equilibrium_rest() {
        let lbm = LatticeBoltzmann2D::new(10, 10, 0.1);
        let feq = lbm.equilibrium(0, 1.0, [0.0, 0.0]);
        assert!((feq - W[0]).abs() < 1e-12);
    }

    #[test]
    fn test_density_conservation() {
        let mut lbm = LatticeBoltzmann2D::new(32, 32, 0.1);
        lbm.initialize_uniform(1.0, [0.01, 0.0]);

        let mut rho0 = 0.0;
        for y in 0..lbm.ny {
            for x in 0..lbm.nx {
                rho0 += lbm.density(x, y);
            }
        }

        for _ in 0..100 {
            lbm.collide_and_stream();
        }

        let mut rho1 = 0.0;
        for y in 0..lbm.ny {
            for x in 0..lbm.nx {
                rho1 += lbm.density(x, y);
            }
        }

        // Density should be conserved
        assert!(
            (rho1 - rho0).abs() / rho0 < 1e-10,
            "Density not conserved: {} vs {}",
            rho0,
            rho1
        );
    }

    #[test]
    fn test_viscous_dissipation() {
        let mut lbm = LatticeBoltzmann2D::new(32, 32, 0.1);

        // Initialize with non-uniform velocity field (shear flow)
        for y in 0..lbm.ny {
            for x in 0..lbm.nx {
                let u_x = if y < lbm.ny / 2 { 0.1 } else { -0.1 };
                let rho = 1.0;
                for i in 0..9 {
                    let feq = lbm.equilibrium(i, rho, [u_x, 0.0]);
                    lbm.set_f(x, y, i, feq);
                }
            }
        }

        let ke0 = lbm.kinetic_energy();

        for _ in 0..1000 {
            lbm.collide_and_stream();
        }

        let ke1 = lbm.kinetic_energy();

        // Kinetic energy should decrease due to viscous dissipation (shear layer smoothing)
        assert!(ke1 < ke0 * 0.9, "KE should decrease: {} -> {}", ke0, ke1);
    }
}
