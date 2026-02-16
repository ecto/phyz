//! D3Q19 Lattice Boltzmann solver for 3D incompressible flow.
//!
//! Nineteen velocity directions on 3D cubic lattice:
//! - 1 rest (0)
//! - 6 face-centered (±x, ±y, ±z)
//! - 12 edge-centered (±x±y, ±y±z, ±z±x)

/// D3Q19 Lattice Boltzmann solver for 3D flow.
pub struct LatticeBoltzmann3D {
    /// Grid size in x
    pub nx: usize,
    /// Grid size in y
    pub ny: usize,
    /// Grid size in z
    pub nz: usize,
    /// Kinematic viscosity
    pub nu: f64,
    /// Relaxation time τ = 3ν + 0.5
    pub tau: f64,
    /// Distribution functions f_i at each grid point
    /// Shape: [nx, ny, nz, 19]
    pub f: Vec<f64>,
    /// Temporary buffer for streaming
    f_temp: Vec<f64>,
}

/// D3Q19 discrete velocities: [vx, vy, vz]
const E: [[i32; 3]; 19] = [
    [0, 0, 0], // 0: rest
    [1, 0, 0], // 1-6: face
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 0], // 7-18: edge
    [-1, -1, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [1, 0, 1],
    [-1, 0, -1],
    [1, 0, -1],
    [-1, 0, 1],
    [0, 1, 1],
    [0, -1, -1],
    [0, 1, -1],
    [0, -1, 1],
];

/// D3Q19 weights
const W: [f64; 19] = [
    1.0 / 3.0,  // 0: rest
    1.0 / 18.0, // 1-6: face
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 36.0, // 7-18: edge
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

/// Opposite direction indices for bounce-back
const OPP: [usize; 19] = [
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17,
];

impl LatticeBoltzmann3D {
    /// Create new 3D LBM solver.
    ///
    /// # Arguments
    /// * `nx`, `ny`, `nz` - Grid dimensions
    /// * `nu` - Kinematic viscosity
    pub fn new(nx: usize, ny: usize, nz: usize, nu: f64) -> Self {
        let tau = 3.0 * nu + 0.5;
        let size = nx * ny * nz * 19;
        Self {
            nx,
            ny,
            nz,
            nu,
            tau,
            f: vec![0.0; size],
            f_temp: vec![0.0; size],
        }
    }

    /// Initialize with uniform density and velocity.
    pub fn initialize_uniform(&mut self, rho: f64, u: [f64; 3]) {
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    for i in 0..19 {
                        let feq = self.equilibrium(i, rho, u);
                        self.set_f(x, y, z, i, feq);
                    }
                }
            }
        }
    }

    /// Compute equilibrium distribution f_i^eq.
    #[inline]
    fn equilibrium(&self, i: usize, rho: f64, u: [f64; 3]) -> f64 {
        let ex = E[i][0] as f64;
        let ey = E[i][1] as f64;
        let ez = E[i][2] as f64;
        let eu = ex * u[0] + ey * u[1] + ez * u[2];
        let uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
        W[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uu)
    }

    /// Get distribution function at (x, y, z, i).
    #[inline]
    fn get_f(&self, x: usize, y: usize, z: usize, i: usize) -> f64 {
        let idx = x + y * self.nx + z * self.nx * self.ny + i * self.nx * self.ny * self.nz;
        self.f[idx]
    }

    /// Set distribution function at (x, y, z, i).
    #[inline]
    fn set_f(&mut self, x: usize, y: usize, z: usize, i: usize, val: f64) {
        let idx = x + y * self.nx + z * self.nx * self.ny + i * self.nx * self.ny * self.nz;
        self.f[idx] = val;
    }

    /// Get temporary distribution function at (x, y, z, i).
    #[inline]
    fn get_f_temp(&self, x: usize, y: usize, z: usize, i: usize) -> f64 {
        let idx = x + y * self.nx + z * self.nx * self.ny + i * self.nx * self.ny * self.nz;
        self.f_temp[idx]
    }

    /// Set temporary distribution function at (x, y, z, i).
    #[inline]
    fn set_f_temp(&mut self, x: usize, y: usize, z: usize, i: usize, val: f64) {
        let idx = x + y * self.nx + z * self.nx * self.ny + i * self.nx * self.ny * self.nz;
        self.f_temp[idx] = val;
    }

    /// Compute macroscopic density at (x, y, z).
    pub fn density(&self, x: usize, y: usize, z: usize) -> f64 {
        (0..19).map(|i| self.get_f(x, y, z, i)).sum()
    }

    /// Compute macroscopic velocity at (x, y, z).
    pub fn velocity(&self, x: usize, y: usize, z: usize) -> [f64; 3] {
        let rho = self.density(x, y, z);
        if rho < 1e-12 {
            return [0.0, 0.0, 0.0];
        }
        let mut u = [0.0, 0.0, 0.0];
        for (i, e) in E.iter().enumerate() {
            let f = self.get_f(x, y, z, i);
            u[0] += f * e[0] as f64;
            u[1] += f * e[1] as f64;
            u[2] += f * e[2] as f64;
        }
        u[0] /= rho;
        u[1] /= rho;
        u[2] /= rho;
        u
    }

    /// Maximum velocity magnitude in domain.
    pub fn max_velocity(&self) -> f64 {
        let mut umax: f64 = 0.0;
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let u = self.velocity(x, y, z);
                    let umag = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                    umax = umax.max(umag);
                }
            }
        }
        umax
    }

    /// Set velocity boundary condition at (x, y, z).
    pub fn set_velocity_bc(&mut self, x: usize, y: usize, z: usize, u: [f64; 3]) {
        let rho = self.density(x, y, z);
        for i in 0..19 {
            let feq = self.equilibrium(i, rho, u);
            self.set_f(x, y, z, i, feq);
        }
    }

    /// Set no-slip boundary condition (bounce-back) at (x, y, z).
    pub fn set_no_slip_bc(&mut self, x: usize, y: usize, z: usize) {
        let f_new: [f64; 19] = std::array::from_fn(|i| self.get_f(x, y, z, OPP[i]));
        for (i, &val) in f_new.iter().enumerate() {
            self.set_f(x, y, z, i, val);
        }
    }

    /// Perform collision step (BGK operator).
    fn collide(&mut self) {
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let rho = self.density(x, y, z);
                    let u = self.velocity(x, y, z);

                    for i in 0..19 {
                        let f = self.get_f(x, y, z, i);
                        let feq = self.equilibrium(i, rho, u);
                        let f_new = f - (f - feq) / self.tau;
                        self.set_f_temp(x, y, z, i, f_new);
                    }
                }
            }
        }
    }

    /// Perform streaming step.
    fn stream(&mut self) {
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    for (i, e) in E.iter().enumerate() {
                        let ex = e[0];
                        let ey = e[1];
                        let ez = e[2];
                        let xp = (x as i32 + ex).rem_euclid(self.nx as i32) as usize;
                        let yp = (y as i32 + ey).rem_euclid(self.ny as i32) as usize;
                        let zp = (z as i32 + ez).rem_euclid(self.nz as i32) as usize;

                        let f = self.get_f_temp(x, y, z, i);
                        self.set_f(xp, yp, zp, i, f);
                    }
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
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let rho = self.density(x, y, z);
                    let u = self.velocity(x, y, z);
                    let umag_sq = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
                    ke += 0.5 * rho * umag_sq;
                }
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
        let lbm = LatticeBoltzmann3D::new(10, 10, 10, 0.1);
        let feq = lbm.equilibrium(0, 1.0, [0.0, 0.0, 0.0]);
        assert!((feq - W[0]).abs() < 1e-12);
    }

    #[test]
    fn test_density_conservation() {
        let mut lbm = LatticeBoltzmann3D::new(16, 16, 16, 0.1);
        lbm.initialize_uniform(1.0, [0.01, 0.0, 0.0]);

        let mut rho0 = 0.0;
        for z in 0..lbm.nz {
            for y in 0..lbm.ny {
                for x in 0..lbm.nx {
                    rho0 += lbm.density(x, y, z);
                }
            }
        }

        for _ in 0..100 {
            lbm.collide_and_stream();
        }

        let mut rho1 = 0.0;
        for z in 0..lbm.nz {
            for y in 0..lbm.ny {
                for x in 0..lbm.nx {
                    rho1 += lbm.density(x, y, z);
                }
            }
        }

        assert!((rho1 - rho0).abs() / rho0 < 1e-10, "Density not conserved");
    }
}
