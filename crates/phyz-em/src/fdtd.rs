//! FDTD update equations for Maxwell's equations.

use crate::grid::YeeGrid;

/// FDTD update functions for Maxwell's equations on Yee grid.
///
/// Implements the staggered-grid finite-difference time-domain method
/// for solving Maxwell's curl equations:
///   ∇ × E = -∂B/∂t = -μ₀μᵣ ∂H/∂t
///   ∇ × H = ∂D/∂t + J = ε₀εᵣ ∂E/∂t + σE
impl YeeGrid {
    /// Update H-field (magnetic field) using ∇ × E.
    ///
    /// H^(n+1/2) = H^(n-1/2) - (dt/μ) ∇ × E^n
    pub fn update_h_field(&mut self) {
        let dt_over_dx = self.dt / self.dx;

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let mu = self.mu0 * self.mu_r.get(i, j, k);
                    let coef = dt_over_dx / mu;

                    // Hx update: ∂Ez/∂y - ∂Ey/∂z
                    if i < self.nx - 1 && j < self.ny - 1 && k < self.nz - 1 {
                        let curl_x = (self.ez.get(i, j + 1, k) - self.ez.get(i, j, k))
                            - (self.ey.get(i, j, k + 1) - self.ey.get(i, j, k));
                        let hx = self.hx.get(i, j, k) - coef * curl_x;
                        self.hx.set(i, j, k, hx);
                    }

                    // Hy update: ∂Ex/∂z - ∂Ez/∂x
                    if i < self.nx - 1 && j < self.ny - 1 && k < self.nz - 1 {
                        let curl_y = (self.ex.get(i, j, k + 1) - self.ex.get(i, j, k))
                            - (self.ez.get(i + 1, j, k) - self.ez.get(i, j, k));
                        let hy = self.hy.get(i, j, k) - coef * curl_y;
                        self.hy.set(i, j, k, hy);
                    }

                    // Hz update: ∂Ey/∂x - ∂Ex/∂y
                    if i < self.nx - 1 && j < self.ny - 1 && k < self.nz - 1 {
                        let curl_z = (self.ey.get(i + 1, j, k) - self.ey.get(i, j, k))
                            - (self.ex.get(i, j + 1, k) - self.ex.get(i, j, k));
                        let hz = self.hz.get(i, j, k) - coef * curl_z;
                        self.hz.set(i, j, k, hz);
                    }
                }
            }
        }
    }

    /// Update E-field (electric field) using ∇ × H with conductivity loss.
    ///
    /// E^(n+1) = ((1 - σΔt/2ε)/(1 + σΔt/2ε)) E^n + (Δt/ε)/(1 + σΔt/2ε) ∇ × H^(n+1/2)
    pub fn update_e_field(&mut self) {
        let dt_over_dx = self.dt / self.dx;

        for i in 1..self.nx {
            for j in 1..self.ny {
                for k in 1..self.nz {
                    let eps = self.eps0 * self.eps_r.get(i, j, k);
                    let sigma = self.sigma.get(i, j, k);

                    // Conductivity loss factor (implicit Crank-Nicolson for stability)
                    let sigma_term = sigma * self.dt / (2.0 * eps);
                    let loss_factor = (1.0 - sigma_term) / (1.0 + sigma_term);
                    let curl_coef = (dt_over_dx / eps) / (1.0 + sigma_term);

                    // Ex update: ∂Hz/∂y - ∂Hy/∂z
                    let curl_x = (self.hz.get(i, j, k) - self.hz.get(i, j - 1, k))
                        - (self.hy.get(i, j, k) - self.hy.get(i, j, k - 1));
                    let ex = loss_factor * self.ex.get(i, j, k) + curl_coef * curl_x;
                    self.ex.set(i, j, k, ex);

                    // Ey update: ∂Hx/∂z - ∂Hz/∂x
                    let curl_y = (self.hx.get(i, j, k) - self.hx.get(i, j, k - 1))
                        - (self.hz.get(i, j, k) - self.hz.get(i - 1, j, k));
                    let ey = loss_factor * self.ey.get(i, j, k) + curl_coef * curl_y;
                    self.ey.set(i, j, k, ey);

                    // Ez update: ∂Hy/∂x - ∂Hx/∂y
                    let curl_z = (self.hy.get(i, j, k) - self.hy.get(i - 1, j, k))
                        - (self.hx.get(i, j, k) - self.hx.get(i, j - 1, k));
                    let ez = loss_factor * self.ez.get(i, j, k) + curl_coef * curl_z;
                    self.ez.set(i, j, k, ez);
                }
            }
        }
    }

    /// Compute Courant–Friedrichs–Lewy (CFL) number.
    ///
    /// For stability, CFL ≤ 1/√3 in 3D.
    pub fn cfl_number(&self) -> f64 {
        self.c0 * self.dt / self.dx
    }

    /// Check if the simulation is stable according to CFL condition.
    pub fn is_stable(&self) -> bool {
        self.cfl_number() <= 1.0 / 3_f64.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfl_stability() {
        // Stable timestep
        let dx = 1e-9; // 1 nm
        let c = 3e8;
        let dt = dx / (c * 3_f64.sqrt() * 1.01); // Just under CFL limit
        let grid = YeeGrid::new(10, 10, 10, dx, dt);
        assert!(grid.is_stable());

        // Unstable timestep
        let dt_unstable = dx / (c * 3_f64.sqrt() * 0.99); // Over CFL limit
        let grid_unstable = YeeGrid::new(10, 10, 10, dx, dt_unstable);
        assert!(!grid_unstable.is_stable());
    }

    #[test]
    fn test_field_updates_no_crash() {
        let mut grid = YeeGrid::new(16, 16, 16, 1e-9, 1e-18);

        // Set a test field
        grid.ex.set(8, 8, 8, 1.0);

        // Run updates
        grid.update_h_field();
        grid.update_e_field();

        // Should not crash, fields should have changed
        let energy = grid.total_energy();
        assert!(energy > 0.0);
    }

    /// A single spatial eigenmode must oscillate at exactly the frequency the
    /// analytic Yee dispersion relation predicts.
    ///
    /// For one eigenmode the leapfrog collapses to the three-term recurrence
    /// `E^{n+1} = 2cos(ωΔt)E^n − E^{n−1}`, so the measured `ω` matches the
    /// analytic root to round-off when the update coefficients are correct.
    ///
    /// The full benchmark — dispersion vs the continuum with a second-order
    /// convergence study, a rectangular-cavity resonance, and a measured
    /// absorbing-boundary reflection coefficient — lives in `phyz-validate`.
    #[test]
    fn test_yee_dispersion_relation() {
        let nz = 65;
        let dx = 1e-3;
        let (nx, ny) = (4, 4);
        let dt = 0.5 * dx / (299_792_458.0 * 3_f64.sqrt());

        let mut grid = YeeGrid::new(nx, ny, nz, dx, dt);
        let c = grid.c0;
        let kz = std::f64::consts::PI / ((nz - 1) as f64 * dx);

        // Ex = sin(k z), H = 0: an exact standing eigenmode of the PEC cavity.
        for k in 0..nz {
            let v = (kz * k as f64 * dx).sin();
            for i in 0..nx {
                for j in 0..ny {
                    grid.ex.set(i, j, k, v);
                }
            }
        }

        let (i0, j0, k0) = (nx / 2, ny / 2, (nz - 1) / 2);
        let mut series = vec![grid.ex.get(i0, j0, k0)];
        for _ in 0..400 {
            grid.update_h_field();
            grid.update_e_field();
            grid.apply_boundary(crate::BoundaryCondition::PerfectConductor);
            // Re-impose transverse uniformity: this is a 1-D problem run on the
            // 3-D grid, whose update loops leave the outermost slab untouched.
            for k in 0..nz {
                let (ex, hy) = (grid.ex.get(i0, j0, k), grid.hy.get(i0, j0, k));
                for i in 0..nx {
                    for j in 0..ny {
                        grid.ex.set(i, j, k, ex);
                        grid.hy.set(i, j, k, hy);
                    }
                }
            }
            series.push(grid.ex.get(i0, j0, k0));
        }

        // cos(ωΔt) by least squares over the record.
        let mut num = 0.0;
        let mut den = 0.0;
        for n in 1..series.len() - 1 {
            num += series[n] * (series[n + 1] + series[n - 1]);
            den += 2.0 * series[n] * series[n];
        }
        let omega = (num / den).clamp(-1.0, 1.0).acos() / dt;

        // sin(ωΔt/2) = (cΔt/Δx) sin(kΔx/2)
        let expected = 2.0 / dt * ((c * dt / dx) * (0.5 * kz * dx).sin()).asin();
        let rel = (omega - expected).abs() / expected;
        assert!(
            rel < 1e-9,
            "ω = {omega}, Yee dispersion predicts {expected} (rel err {rel:.3e})"
        );

        // ...and that discrete frequency must be within the O((kΔx)²) grid
        // dispersion error of the continuum ω = ck.
        let continuum = c * kz;
        let dispersion_error = (omega - continuum).abs() / continuum;
        assert!(
            dispersion_error < 1e-3,
            "grid dispersion error = {dispersion_error:.3e} at kΔx = {:.4}",
            kz * dx
        );
    }
}
