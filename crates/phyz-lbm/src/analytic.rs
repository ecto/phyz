//! Analytic and published reference solutions for validating the solvers.
//!
//! These are deliberately **public** and free of any dependency on the solver
//! types, so the workspace-wide analytic validation suite can call them
//! directly instead of re-deriving the same formulas. Add new closed-form
//! solutions here rather than inlining them in a test.

use std::f64::consts::PI;

/// Steady plane Poiseuille flow driven by a uniform body force.
///
/// Walls at `wall_lo` and `wall_hi` (no-slip), body force `g` per unit mass
/// along the channel, kinematic viscosity `nu`:
///
/// ```text
/// u(y) = g/(2ν) (y - y_lo)(y_hi - y),   u_max = g H² / (8ν)
/// ```
///
/// With halfway bounce-back the walls sit half a lattice spacing outside the
/// first and last fluid nodes, i.e. `wall_lo = -0.5`, `wall_hi = ny - 0.5`.
pub fn poiseuille_force_driven(y: f64, wall_lo: f64, wall_hi: f64, g: f64, nu: f64) -> f64 {
    g / (2.0 * nu) * (y - wall_lo) * (wall_hi - y)
}

/// Peak velocity of [`poiseuille_force_driven`] for a channel of width `h`.
pub fn poiseuille_peak(h: f64, g: f64, nu: f64) -> f64 {
    g * h * h / (8.0 * nu)
}

/// Body force that produces a given peak velocity — the inverse of
/// [`poiseuille_peak`]. Handy for holding the Mach number fixed while sweeping
/// viscosity.
pub fn poiseuille_force_for_peak(h: f64, u_peak: f64, nu: f64) -> f64 {
    8.0 * nu * u_peak / (h * h)
}

/// Amplitude decay factor of a 2D Taylor–Green vortex at time `t`.
///
/// The velocity decays as `exp(-ν(k_x² + k_y²) t)`; kinetic energy, being
/// quadratic, decays at twice that rate.
pub fn taylor_green_decay(t: f64, nu: f64, kx: f64, ky: f64) -> f64 {
    (-nu * (kx * kx + ky * ky) * t).exp()
}

/// Analytic 2D Taylor–Green vortex velocity field.
///
/// ```text
/// u_x = -u₀ cos(k_x x) sin(k_y y) e^{-ν(k_x²+k_y²)t}
/// u_y =  u₀ (k_x/k_y) sin(k_x x) cos(k_y y) e^{-ν(k_x²+k_y²)t}
/// ```
///
/// The `k_x/k_y` factor keeps the field divergence-free for anisotropic wave
/// numbers; for `k_x = k_y` it is the familiar symmetric pair.
pub fn taylor_green_velocity(
    x: f64,
    y: f64,
    t: f64,
    u0: f64,
    kx: f64,
    ky: f64,
    nu: f64,
) -> [f64; 2] {
    let decay = taylor_green_decay(t, nu, kx, ky);
    [
        -u0 * (kx * x).cos() * (ky * y).sin() * decay,
        u0 * (kx / ky) * (kx * x).sin() * (ky * y).cos() * decay,
    ]
}

/// Density field consistent with the Taylor–Green pressure, for initialisation.
///
/// `p = -ρ₀ u₀²/4 [ (k_y/k_x) cos(2k_x x) + (k_x/k_y) cos(2k_y y) ] e^{-2ν k² t}`,
/// converted to density through the isothermal equation of state `p = ρ c_s²`.
#[allow(clippy::too_many_arguments)] // every argument is a distinct physical parameter
pub fn taylor_green_density(
    x: f64,
    y: f64,
    t: f64,
    rho0: f64,
    u0: f64,
    kx: f64,
    ky: f64,
    nu: f64,
) -> f64 {
    let decay = taylor_green_decay(t, nu, kx, ky);
    let p = -0.25
        * rho0
        * u0
        * u0
        * decay
        * decay
        * ((ky / kx) * (2.0 * kx * x).cos() + (kx / ky) * (2.0 * ky * y).cos());
    rho0 + p * 3.0 // c_s² = 1/3
}

/// Wave number for `n` full periods across `n_cells` lattice sites.
pub fn wavenumber(n_cells: usize, periods: usize) -> f64 {
    2.0 * PI * periods as f64 / n_cells as f64
}

/// Reference data for the lid-driven cavity, from
/// Ghia, Ghia & Shin, *J. Comput. Phys.* 48 (1982) 387–411, Tables I and II.
///
/// Coordinates are normalised to the unit cavity; velocities are normalised by
/// the lid speed. These are the de-facto standard against which cavity solvers
/// are checked.
pub mod ghia {
    /// `y/L` stations for the `u` profile on the vertical centreline.
    pub const Y: [f64; 17] = [
        0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000, 0.6172, 0.7344,
        0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000,
    ];

    /// `u/U_lid` on the vertical centreline at Re = 100.
    pub const U_RE100: [f64; 17] = [
        0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581,
        -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000,
    ];

    /// `u/U_lid` on the vertical centreline at Re = 400.
    pub const U_RE400: [f64; 17] = [
        0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119, -0.11477,
        0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 0.68439, 0.75837, 1.00000,
    ];

    /// `x/L` stations for the `v` profile on the horizontal centreline.
    pub const X: [f64; 17] = [
        0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5000, 0.8047, 0.8594,
        0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000,
    ];

    /// `v/U_lid` on the horizontal centreline at Re = 100.
    pub const V_RE100: [f64; 17] = [
        0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533,
        -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.00000,
    ];
}

/// Least-squares slope of `log(error)` against `log(1/n)` — the observed order
/// of accuracy under grid refinement.
pub fn convergence_order(cells: &[usize], errors: &[f64]) -> f64 {
    assert_eq!(cells.len(), errors.len());
    assert!(cells.len() >= 2, "need at least two grids");
    let xs: Vec<f64> = cells.iter().map(|&n| (n as f64).ln()).collect();
    let ys: Vec<f64> = errors.iter().map(|&e| e.ln()).collect();
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let num: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    -num / den
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poiseuille_peak_matches_profile_centre() {
        let (h, g, nu) = (32.0, 1e-5, 0.05);
        let centre = poiseuille_force_driven(-0.5 + h / 2.0, -0.5, -0.5 + h, g, nu);
        assert!((centre - poiseuille_peak(h, g, nu)).abs() < 1e-15);
    }

    #[test]
    fn poiseuille_force_inverts_peak() {
        let g = poiseuille_force_for_peak(32.0, 0.05, 0.05);
        assert!((poiseuille_peak(32.0, g, 0.05) - 0.05).abs() < 1e-15);
    }

    #[test]
    fn poiseuille_vanishes_at_walls() {
        assert!(poiseuille_force_driven(-0.5, -0.5, 31.5, 1e-5, 0.05).abs() < 1e-18);
        assert!(poiseuille_force_driven(31.5, -0.5, 31.5, 1e-5, 0.05).abs() < 1e-18);
    }

    #[test]
    fn taylor_green_is_divergence_free() {
        let (u0, nu) = (0.04, 0.02);
        let (kx, ky) = (wavenumber(32, 1), wavenumber(32, 2));
        let h = 1e-5;
        for &(x, y) in &[(3.0, 5.0), (11.5, 2.25), (0.0, 0.0)] {
            let dudx = (taylor_green_velocity(x + h, y, 0.0, u0, kx, ky, nu)[0]
                - taylor_green_velocity(x - h, y, 0.0, u0, kx, ky, nu)[0])
                / (2.0 * h);
            let dvdy = (taylor_green_velocity(x, y + h, 0.0, u0, kx, ky, nu)[1]
                - taylor_green_velocity(x, y - h, 0.0, u0, kx, ky, nu)[1])
                / (2.0 * h);
            assert!((dudx + dvdy).abs() < 1e-6, "div = {}", dudx + dvdy);
        }
    }

    #[test]
    fn convergence_order_recovers_a_known_slope() {
        let cells = [16usize, 32, 64];
        let errors: Vec<f64> = cells.iter().map(|&n| 3.0 * (n as f64).powi(-2)).collect();
        assert!((convergence_order(&cells, &errors) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn ghia_tables_are_well_formed() {
        // Monotone stations, endpoints pinned by the boundary conditions.
        assert!(ghia::Y.windows(2).all(|w| w[0] < w[1]));
        assert!(ghia::X.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(ghia::U_RE100[0], 0.0);
        assert_eq!(ghia::U_RE100[16], 1.0); // the lid itself
        assert_eq!(ghia::V_RE100[0], 0.0);
        assert_eq!(ghia::V_RE100[16], 0.0);
    }
}
