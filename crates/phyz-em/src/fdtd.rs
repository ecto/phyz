//! FDTD update equations for Maxwell's equations.
//!
//! Implements the staggered-grid finite-difference time-domain method for
//! Maxwell's curl equations,
//!
//! ```text
//!   ∇ × E = −μ ∂H/∂t
//!   ∇ × H = ∂D/∂t + σE
//! ```
//!
//! with three optional extensions folded into the same loops:
//!
//! * **CPML** — each spatial derivative is replaced by `(1/κ)∂/∂w + ψ`, where ψ
//!   is a recursively updated convolution variable (see [`crate::cpml`]).
//! * **Dispersive media** — `∂D/∂t = ε₀ε_inf ∂E/∂t + Σ_p ∂P_p/∂t`, with the
//!   polarizations advanced by ADE recursions (see [`crate::dispersion`]).
//! * **Periodic axes** — neighbour lookups wrap; non-periodic axes are
//!   terminated by a PEC wall at both outer field planes.

use crate::grid::YeeGrid;

/// Neighbour index one step *down* an axis, wrapping if the axis is periodic.
#[inline]
fn prev_idx(idx: usize, n: usize, periodic: bool) -> Option<usize> {
    if idx > 0 {
        Some(idx - 1)
    } else if periodic {
        Some(n - 1)
    } else {
        None
    }
}

/// Neighbour index one step *up* an axis, wrapping if the axis is periodic.
#[inline]
fn next_idx(idx: usize, n: usize, periodic: bool) -> Option<usize> {
    if idx + 1 < n {
        Some(idx + 1)
    } else if periodic {
        Some(0)
    } else {
        None
    }
}

impl YeeGrid {
    /// Update H-field (magnetic field) using ∇ × E.
    ///
    /// H^(n+1/2) = H^(n-1/2) − (Δt/μ) ∇ × E^n
    pub fn update_h_field(&mut self) {
        use crate::cpml::Psi;

        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let (dx, dy, dz) = (self.dx, self.dy, self.dz);
        let (dt, mu0) = (self.dt, self.mu0);
        let [px, py, pz] = self.periodic;

        // Move the CPML out of `self` so the loop can freely touch field arrays.
        let mut cpml = self.cpml.take();

        for k in 0..nz {
            let kp = next_idx(k, nz, pz);
            for j in 0..ny {
                let jp = next_idx(j, ny, py);
                for i in 0..nx {
                    let ip = next_idx(i, nx, px);

                    let mu = mu0 * self.mu_r.get(i, j, k);
                    let coef = dt / mu;

                    // Hx: ∂Ez/∂y − ∂Ey/∂z
                    if let (Some(jp), Some(kp)) = (jp, kp) {
                        let d_y = (self.ez.get(i, jp, k) - self.ez.get(i, j, k)) / dy;
                        let d_z = (self.ey.get(i, j, kp) - self.ey.get(i, j, k)) / dz;
                        let mut curl = d_y - d_z;
                        if let Some(c) = cpml.as_mut() {
                            if c.axis_active(1) {
                                curl += c.advance(Psi::Hxy, 1, j, false, i, j, k, d_y);
                                curl -= (1.0 - c.kinv_h(1, j)) * d_y;
                            }
                            if c.axis_active(2) {
                                curl -= c.advance(Psi::Hxz, 2, k, false, i, j, k, d_z);
                                curl += (1.0 - c.kinv_h(2, k)) * d_z;
                            }
                        }
                        let hx = self.hx.get(i, j, k) - coef * curl;
                        self.hx.set(i, j, k, hx);
                    }

                    // Hy: ∂Ex/∂z − ∂Ez/∂x
                    if let (Some(kp), Some(ip)) = (kp, ip) {
                        let d_z = (self.ex.get(i, j, kp) - self.ex.get(i, j, k)) / dz;
                        let d_x = (self.ez.get(ip, j, k) - self.ez.get(i, j, k)) / dx;
                        let mut curl = d_z - d_x;
                        if let Some(c) = cpml.as_mut() {
                            if c.axis_active(2) {
                                curl += c.advance(Psi::Hyz, 2, k, false, i, j, k, d_z);
                                curl -= (1.0 - c.kinv_h(2, k)) * d_z;
                            }
                            if c.axis_active(0) {
                                curl -= c.advance(Psi::Hyx, 0, i, false, i, j, k, d_x);
                                curl += (1.0 - c.kinv_h(0, i)) * d_x;
                            }
                        }
                        let hy = self.hy.get(i, j, k) - coef * curl;
                        self.hy.set(i, j, k, hy);
                    }

                    // Hz: ∂Ey/∂x − ∂Ex/∂y
                    if let (Some(ip), Some(jp)) = (ip, jp) {
                        let d_x = (self.ey.get(ip, j, k) - self.ey.get(i, j, k)) / dx;
                        let d_y = (self.ex.get(i, jp, k) - self.ex.get(i, j, k)) / dy;
                        let mut curl = d_x - d_y;
                        if let Some(c) = cpml.as_mut() {
                            if c.axis_active(0) {
                                curl += c.advance(Psi::Hzx, 0, i, false, i, j, k, d_x);
                                curl -= (1.0 - c.kinv_h(0, i)) * d_x;
                            }
                            if c.axis_active(1) {
                                curl -= c.advance(Psi::Hzy, 1, j, false, i, j, k, d_y);
                                curl += (1.0 - c.kinv_h(1, j)) * d_y;
                            }
                        }
                        let hz = self.hz.get(i, j, k) - coef * curl;
                        self.hz.set(i, j, k, hz);
                    }
                }
            }
        }

        self.cpml = cpml;
    }

    /// Update E-field (electric field) from ∇ × H, with conduction loss and
    /// dispersive polarization currents.
    ///
    /// ```text
    ///   ε₀ε_inf (E^{n+1} − E^n)/Δt = ∇×H^{n+1/2} − ΣΔP/Δt − σ(E^{n+1}+E^n)/2
    /// ```
    pub fn update_e_field(&mut self) {
        use crate::cpml::Psi;

        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let (dx, dy, dz) = (self.dx, self.dy, self.dz);
        let (dt, eps0) = (self.dt, self.eps0);
        let [px, py, pz] = self.periodic;

        // Advance the ADE polarizations first: they depend on E^n only.
        if let Some(disp) = self.dispersion.as_mut() {
            disp.advance(&self.mat_id, [&self.ex, &self.ey, &self.ez], nx, ny, nz);
        }

        let mut cpml = self.cpml.take();
        let disp = self.dispersion.take();

        for k in 0..nz {
            let km = prev_idx(k, nz, pz);
            let z_wall = !pz && (k == 0 || k == nz - 1);
            for j in 0..ny {
                let jm = prev_idx(j, ny, py);
                let y_wall = !py && (j == 0 || j == ny - 1);
                for i in 0..nx {
                    let im = prev_idx(i, nx, px);
                    let x_wall = !px && (i == 0 || i == nx - 1);

                    let eps = eps0 * self.eps_r.get(i, j, k);
                    let sigma = self.sigma.get(i, j, k);
                    let ca = eps / dt - sigma / 2.0;
                    let cb = eps / dt + sigma / 2.0;

                    // Ex is tangential to the y and z outer walls.
                    if y_wall || z_wall {
                        self.ex.set(i, j, k, 0.0);
                    } else if let (Some(jm), Some(km)) = (jm, km) {
                        let d_y = (self.hz.get(i, j, k) - self.hz.get(i, jm, k)) / dy;
                        let d_z = (self.hy.get(i, j, k) - self.hy.get(i, j, km)) / dz;
                        let mut curl = d_y - d_z;
                        if let Some(c) = cpml.as_mut() {
                            if c.axis_active(1) {
                                curl += c.advance(Psi::Exy, 1, j, true, i, j, k, d_y);
                                curl -= (1.0 - c.kinv_e(1, j)) * d_y;
                            }
                            if c.axis_active(2) {
                                curl -= c.advance(Psi::Exz, 2, k, true, i, j, k, d_z);
                                curl += (1.0 - c.kinv_e(2, k)) * d_z;
                            }
                        }
                        let dp = disp.as_ref().map_or(0.0, |d| d.delta_p(0, i, j, k));
                        let ex = (ca * self.ex.get(i, j, k) + curl - dp / dt) / cb;
                        self.ex.set(i, j, k, ex);
                    }

                    // Ey is tangential to the x and z outer walls.
                    if x_wall || z_wall {
                        self.ey.set(i, j, k, 0.0);
                    } else if let (Some(km), Some(im)) = (km, im) {
                        let d_z = (self.hx.get(i, j, k) - self.hx.get(i, j, km)) / dz;
                        let d_x = (self.hz.get(i, j, k) - self.hz.get(im, j, k)) / dx;
                        let mut curl = d_z - d_x;
                        if let Some(c) = cpml.as_mut() {
                            if c.axis_active(2) {
                                curl += c.advance(Psi::Eyz, 2, k, true, i, j, k, d_z);
                                curl -= (1.0 - c.kinv_e(2, k)) * d_z;
                            }
                            if c.axis_active(0) {
                                curl -= c.advance(Psi::Eyx, 0, i, true, i, j, k, d_x);
                                curl += (1.0 - c.kinv_e(0, i)) * d_x;
                            }
                        }
                        let dp = disp.as_ref().map_or(0.0, |d| d.delta_p(1, i, j, k));
                        let ey = (ca * self.ey.get(i, j, k) + curl - dp / dt) / cb;
                        self.ey.set(i, j, k, ey);
                    }

                    // Ez is tangential to the x and y outer walls.
                    if x_wall || y_wall {
                        self.ez.set(i, j, k, 0.0);
                    } else if let (Some(im), Some(jm)) = (im, jm) {
                        let d_x = (self.hy.get(i, j, k) - self.hy.get(im, j, k)) / dx;
                        let d_y = (self.hx.get(i, j, k) - self.hx.get(i, jm, k)) / dy;
                        let mut curl = d_x - d_y;
                        if let Some(c) = cpml.as_mut() {
                            if c.axis_active(0) {
                                curl += c.advance(Psi::Ezx, 0, i, true, i, j, k, d_x);
                                curl -= (1.0 - c.kinv_e(0, i)) * d_x;
                            }
                            if c.axis_active(1) {
                                curl -= c.advance(Psi::Ezy, 1, j, true, i, j, k, d_y);
                                curl += (1.0 - c.kinv_e(1, j)) * d_y;
                            }
                        }
                        let dp = disp.as_ref().map_or(0.0, |d| d.delta_p(2, i, j, k));
                        let ez = (ca * self.ez.get(i, j, k) + curl - dp / dt) / cb;
                        self.ez.set(i, j, k, ez);
                    }
                }
            }
        }

        self.cpml = cpml;
        self.dispersion = disp;
    }

    /// Courant number, generalized to rectangular cells.
    ///
    /// `S = c Δt √(1/Δx² + 1/Δy² + 1/Δz²)`; stability requires `S ≤ 1`.
    pub fn cfl_number(&self) -> f64 {
        let inv = 1.0 / (self.dx * self.dx) + 1.0 / (self.dy * self.dy) + 1.0 / (self.dz * self.dz);
        self.c0 * self.dt * inv.sqrt()
    }

    /// Largest stable timestep for this grid (the Courant limit).
    pub fn max_stable_dt(&self) -> f64 {
        let inv = 1.0 / (self.dx * self.dx) + 1.0 / (self.dy * self.dy) + 1.0 / (self.dz * self.dz);
        1.0 / (self.c0 * inv.sqrt())
    }

    /// Check if the simulation satisfies the CFL stability condition.
    pub fn is_stable(&self) -> bool {
        self.cfl_number() <= 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{amplitude_at, db};
    use crate::boundary::BoundaryCondition;
    use crate::cpml::CpmlConfig;

    #[test]
    fn test_cfl_stability() {
        let dx = 1e-9; // 1 nm
        let c = 299_792_458.0;
        // Cubic cells: the limit is dx / (c√3).
        let dt = dx / (c * 3_f64.sqrt()) * 0.99;
        let grid = YeeGrid::new(10, 10, 10, dx, dt);
        assert!(grid.is_stable());
        assert!(
            (grid.max_stable_dt() - dx / (c * 3_f64.sqrt())).abs() / grid.max_stable_dt() < 1e-6
        );

        let dt_unstable = dx / (c * 3_f64.sqrt()) * 1.01;
        let grid_unstable = YeeGrid::new(10, 10, 10, dx, dt_unstable);
        assert!(!grid_unstable.is_stable());
    }

    #[test]
    fn test_rectangular_cfl() {
        let c = 299_792_458.0;
        let (dx, dy, dz) = (1e-9_f64, 2e-9_f64, 4e-9_f64);
        let inv = (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz)).sqrt();
        let dt = 0.99 / (c * inv);
        let grid = YeeGrid::new_rect(8, 8, 8, dx, dy, dz, dt);
        assert!(grid.is_stable());
        assert!((grid.cfl_number() - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_field_updates_no_crash() {
        let mut grid = YeeGrid::new(16, 16, 16, 1e-9, 1e-18);
        grid.ex.set(8, 8, 8, 1.0);
        grid.update_h_field();
        grid.update_e_field();
        assert!(grid.total_energy() > 0.0);
    }

    /// A plane wave launched along a periodic-transverse waveguide must travel
    /// at very nearly `c`, with the deficit set by the analytic 1D numerical
    /// dispersion relation.
    ///
    /// The Yee scheme's exact 1D relation is
    /// `sin(ωΔt/2)/(cΔt) = sin(k Δz/2)/Δz`,
    /// so the numerical phase velocity is measurably below `c` and the test
    /// checks the measured value against that prediction, not against `c`.
    #[test]
    fn numerical_dispersion_relation_matches_theory() {
        let nz = 400;
        let dz = 1e-9;
        let c = 299_792_458.0;
        let dt = 0.5 * dz / c;

        // Points per wavelength: 20 → a dispersion deficit of order 1e-3,
        // large enough to distinguish from `c` yet small enough to be clean.
        let ppw = 20.0;
        let lambda = ppw * dz;
        let freq = c / lambda;
        let omega = 2.0 * std::f64::consts::PI * freq;

        let mut grid = YeeGrid::new_rect(1, 1, nz, dz, dz, dz, dt);
        grid.periodic = [true, true, false];
        grid.set_boundary(BoundaryCondition::Cpml(
            CpmlConfig::with_thickness(10).on_axes([false, false, true]),
        ));

        let k_src = 30;
        let (k_a, k_b) = (120, 320);
        let n_steps = 3000;
        let mut probe_a = Vec::with_capacity(n_steps);
        let mut probe_b = Vec::with_capacity(n_steps);

        for n in 0..n_steps {
            grid.update_h_field();
            grid.update_e_field();
            let t = n as f64 * dt;
            // Ramped sinusoid: avoids a broadband turn-on transient.
            let ramp = (t * freq / 8.0).min(1.0);
            let v = ramp * (omega * t).sin();
            grid.ex.add(0, 0, k_src, v);
            probe_a.push(grid.ex.get(0, 0, k_a));
            probe_b.push(grid.ex.get(0, 0, k_b));
        }

        // Use the last half of the record: steady state.
        let tail = n_steps / 2;
        let (amp_a, ph_a) = amplitude_at(&probe_a[tail..], omega, dt);
        let (amp_b, ph_b) = amplitude_at(&probe_b[tail..], omega, dt);
        assert!(amp_a > 1e-3 && amp_b > 1e-3, "no steady-state signal");

        // Phase accumulated over the probe separation gives k.
        let mut dphi = ph_a - ph_b;
        let n_cells = (k_b - k_a) as f64;
        // Unwrap using the continuum estimate as a guide.
        let k_expected = omega / c;
        let target = k_expected * n_cells * dz;
        let two_pi = 2.0 * std::f64::consts::PI;
        dphi += two_pi * ((target - dphi) / two_pi).round();
        let k_measured = dphi / (n_cells * dz);

        // Analytic Yee 1D dispersion relation, solved for k.
        let k_theory = 2.0 / dz * ((dz / (c * dt)) * (omega * dt / 2.0).sin()).asin();

        let rel_err = (k_measured - k_theory).abs() / k_theory;
        assert!(
            rel_err < 2e-3,
            "k_measured = {k_measured:.6e}, k_theory = {k_theory:.6e} (rel err {rel_err:.2e}); \
             continuum k = {k_expected:.6e}"
        );

        // Sanity: the numerical wave really is slower than light, and by the
        // predicted amount rather than an arbitrary one.
        assert!(k_theory > k_expected, "expected sub-luminal phase velocity");

        // The wave should not decay across the (vacuum) probe span.
        let decay_db = db(amp_b / amp_a);
        assert!(
            decay_db.abs() < 0.5,
            "unexpected amplitude change across vacuum: {decay_db:.2} dB"
        );
    }

    /// With periodic transverse boundaries a transversely uniform plane wave
    /// must stay exactly uniform — this is what the old x-only periodic
    /// implementation could not do.
    #[test]
    fn periodic_boundaries_preserve_transverse_uniformity() {
        let dz = 1e-9;
        let c = 299_792_458.0;
        let dt = 0.5 * dz / c;
        let mut grid = YeeGrid::new_rect(6, 5, 60, dz, dz, dz, dt);
        grid.periodic = [true, true, false];
        grid.set_boundary(BoundaryCondition::Cpml(
            CpmlConfig::with_thickness(8).on_axes([false, false, true]),
        ));

        for n in 0..300 {
            grid.update_h_field();
            grid.update_e_field();
            let t = n as f64 * dt;
            let v = (-((t - 20.0 * dt) / (6.0 * dt)).powi(2)).exp();
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    grid.ex.add(i, j, 20, v);
                }
            }
        }

        // Every transverse column must agree to round-off.
        let mut max_dev: f64 = 0.0;
        let mut scale: f64 = 0.0;
        for k in 0..grid.nz {
            let ref_v = grid.ex.get(0, 0, k);
            scale = scale.max(ref_v.abs());
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    max_dev = max_dev.max((grid.ex.get(i, j, k) - ref_v).abs());
                }
            }
        }
        assert!(scale > 1e-6, "no field present to compare");
        assert!(
            max_dev / scale < 1e-12,
            "transverse non-uniformity {:.3e} (relative)",
            max_dev / scale
        );
    }
}
