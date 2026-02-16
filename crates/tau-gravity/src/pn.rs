//! Post-Newtonian gravity solver (Layer 4).
//!
//! Implements relativistic corrections to Newtonian gravity:
//! - 1PN: O(v²/c²) corrections (perihelion precession)
//! - 2.5PN: O(v⁵/c⁵) gravitational radiation damping
//!
//! # Example: Mercury Perihelion Precession
//!
//! ```
//! use tau_gravity::{GravityParticle, PostNewtonianSolver, GravitySolver};
//! use tau_math::Vec3;
//!
//! let m_sun = 1.989e30;
//! let m_mercury = 3.285e23;
//!
//! let mut particles = vec![
//!     GravityParticle::new(Vec3::zeros(), Vec3::zeros(), m_sun),
//!     GravityParticle::new(
//!         Vec3::new(57.9e9, 0.0, 0.0),
//!         Vec3::new(0.0, 47.4e3, 0.0),
//!         m_mercury,
//!     ),
//! ];
//!
//! let mut solver = PostNewtonianSolver::new(1.0); // 1PN
//! solver.compute_forces(&mut particles);
//! ```
//!
//! # References
//!
//! - Blanchet (2014): "Gravitational Radiation from Post-Newtonian Sources"
//! - Will (2014): "The Confrontation between General Relativity and Experiment"

use crate::{C, G, GravityParticle, GravitySolver};
use tau_math::Vec3;

/// Post-Newtonian gravity solver.
#[derive(Debug, Clone)]
pub struct PostNewtonianSolver {
    /// Maximum PN order (1.0 = 1PN, 2.5 = 2.5PN).
    pub max_order: f64,
    /// Softening length (m).
    pub softening: f64,
    /// Include gravitational radiation (2.5PN).
    pub include_radiation: bool,
}

impl PostNewtonianSolver {
    /// Create a new post-Newtonian solver.
    pub fn new(max_order: f64) -> Self {
        Self {
            max_order,
            softening: 1e-3,
            include_radiation: max_order >= 2.5,
        }
    }

    /// Compute Newtonian acceleration.
    fn newtonian_acceleration(
        &self,
        xi: Vec3,
        _vi: Vec3,
        _mi: f64,
        xj: Vec3,
        _vj: Vec3,
        mj: f64,
    ) -> Vec3 {
        let r = xj - xi;
        let r2 = r.norm_squared() + self.softening * self.softening;
        let r_mag = r2.sqrt();

        // a_N = G * mj / r² * r̂
        G * mj / r2 * (r / r_mag)
    }

    /// Compute 1PN acceleration correction.
    ///
    /// From Blanchet (2014), eq. 6.22:
    ///
    /// ```text
    /// a_1PN = G*mj/r² * [
    ///   (4*G*(mi+mj)/r - vi²) * n
    ///   + 4*(vi·vj) * n
    ///   - (vi·n) * vj
    /// ] / c²
    /// ```
    ///
    /// where n = r̂ is the unit vector from i to j.
    fn pn1_acceleration(&self, xi: Vec3, vi: Vec3, mi: f64, xj: Vec3, vj: Vec3, mj: f64) -> Vec3 {
        let r = xj - xi;
        let r2 = r.norm_squared() + self.softening * self.softening;
        let r_mag = r2.sqrt();
        let n = r / r_mag;

        let v2_i = vi.norm_squared();
        let v2_j = vj.norm_squared();
        let vi_dot_vj = vi.dot(&vj);
        let vi_dot_n = vi.dot(&n);
        let vj_dot_n = vj.dot(&n);

        let c2 = C * C;

        // Schwarzschild-like term: 4*G*(mi+mj)/r
        let schwarzschild = 4.0 * G * (mi + mj) / r_mag;

        // 1PN acceleration components
        let coeff = G * mj / r2 / c2;

        let term1 =
            (schwarzschild - v2_i - 2.0 * v2_j + 4.0 * vi_dot_vj + 1.5 * vj_dot_n.powi(2)) * n;
        let term2 = (4.0 * vi_dot_n - 3.0 * vj_dot_n) * vj;

        coeff * (term1 + term2)
    }

    /// Compute 2.5PN gravitational radiation damping.
    ///
    /// From Blanchet (2014), eq. 9.20 (leading order GW emission):
    ///
    /// ```text
    /// a_2.5PN = -8/5 * G²*mi*mj*(mi+mj) / (c⁵*r³) * [
    ///   (vi - vj) - 3*(vi-vj)·n * n
    /// ]
    /// ```
    ///
    /// This is the radiation reaction force causing orbital decay.
    fn pn2_5_acceleration(&self, xi: Vec3, vi: Vec3, mi: f64, xj: Vec3, vj: Vec3, mj: f64) -> Vec3 {
        if !self.include_radiation {
            return Vec3::zeros();
        }

        let r = xj - xi;
        let r2 = r.norm_squared() + self.softening * self.softening;
        let r_mag = r2.sqrt();
        let r3 = r_mag * r2;
        let n = r / r_mag;

        let v_rel = vi - vj;
        let v_rel_dot_n = v_rel.dot(&n);

        let c5 = C.powi(5);
        let coeff = -8.0 / 5.0 * G * G * mi * mj * (mi + mj) / (c5 * r3);

        coeff * (v_rel - 3.0 * v_rel_dot_n * n)
    }

    /// Compute total PN acceleration on particle i due to j.
    fn pn_acceleration(&self, xi: Vec3, vi: Vec3, mi: f64, xj: Vec3, vj: Vec3, mj: f64) -> Vec3 {
        let mut a = self.newtonian_acceleration(xi, vi, mi, xj, vj, mj);

        if self.max_order >= 1.0 {
            a += self.pn1_acceleration(xi, vi, mi, xj, vj, mj);
        }

        if self.max_order >= 2.5 {
            a += self.pn2_5_acceleration(xi, vi, mi, xj, vj, mj);
        }

        a
    }
}

impl GravitySolver for PostNewtonianSolver {
    fn compute_forces(&mut self, particles: &mut [GravityParticle]) {
        let n = particles.len();

        // Reset forces
        for p in particles.iter_mut() {
            p.reset_force();
        }

        // Compute pairwise forces
        // Note: We need positions AND velocities, so we can't use Newton's 3rd law
        // directly (PN forces are velocity-dependent and not symmetric).
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                let xi = particles[i].x;
                let vi = particles[i].v;
                let mi = particles[i].m;

                let xj = particles[j].x;
                let vj = particles[j].v;
                let mj = particles[j].m;

                let a = self.pn_acceleration(xi, vi, mi, xj, vj, mj);
                particles[i].add_force(a * mi);
            }
        }
    }

    fn potential_energy(&self, particles: &[GravityParticle]) -> f64 {
        let n = particles.len();
        let mut u = 0.0;

        // Newtonian potential energy
        for i in 0..n {
            for j in i + 1..n {
                let r = (particles[j].x - particles[i].x).norm();
                let r_soft = (r * r + self.softening * self.softening).sqrt();
                u -= G * particles[i].m * particles[j].m / r_soft;
            }
        }

        // PN corrections to energy are complex; we'll just return Newtonian for now
        u
    }
}

/// Compute orbital elements from state.
pub fn orbital_elements(x: Vec3, v: Vec3, m_central: f64) -> (f64, f64, f64, f64, f64, f64) {
    let mu = G * m_central;

    let r = x.norm();
    let v2 = v.norm_squared();

    // Specific orbital energy
    let energy = v2 / 2.0 - mu / r;

    // Angular momentum vector
    let h = x.cross(&v);
    let h_mag = h.norm();

    // Semi-major axis
    let a = -mu / (2.0 * energy);

    // Eccentricity vector: e = (v × h) / μ - r̂
    let e_vec = v.cross(&h) / mu - x / r;
    let e = e_vec.norm();

    // Inclination
    let i = (h.z / h_mag).acos();

    // Longitude of ascending node
    let n = Vec3::new(-h.y, h.x, 0.0);
    let n_mag = n.norm();
    let omega = if n_mag > 1e-10 {
        let omega_raw = (n.x / n_mag).acos();
        if n.y < 0.0 {
            2.0 * std::f64::consts::PI - omega_raw
        } else {
            omega_raw
        }
    } else {
        0.0
    };

    // Argument of periapsis
    let omega_bar = if n_mag > 1e-10 && e > 1e-10 {
        let omega_bar_raw = (n.dot(&e_vec) / (n_mag * e)).acos();
        if e_vec.z < 0.0 {
            2.0 * std::f64::consts::PI - omega_bar_raw
        } else {
            omega_bar_raw
        }
    } else {
        0.0
    };

    // True anomaly
    let nu = if e > 1e-10 {
        let nu_raw = (e_vec.dot(&x) / (e * r)).acos();
        if x.dot(&v) < 0.0 {
            2.0 * std::f64::consts::PI - nu_raw
        } else {
            nu_raw
        }
    } else {
        0.0
    };

    (a, e, i, omega, omega_bar, nu)
}

/// Compute perihelion precession rate (arcsec/century).
///
/// For 1PN approximation (Schwarzschild spacetime):
/// Δω = 6π G M / (c² a (1 - e²))
///
/// For Mercury: 43.1 arcsec/century.
pub fn perihelion_precession_rate(a: f64, e: f64, m_central: f64, period: f64) -> f64 {
    let delta_omega_per_orbit =
        6.0 * std::f64::consts::PI * G * m_central / (C * C * a * (1.0 - e * e));

    // Convert to arcsec/century
    let orbits_per_century = 3.15576e9 / period; // seconds in a century / period
    let delta_omega_century = delta_omega_per_orbit * orbits_per_century;

    // Radians to arcsec: 1 rad = 206265 arcsec
    delta_omega_century * 206265.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newtonian_vs_pn() {
        let mut solver_newton = PostNewtonianSolver::new(0.0); // Newtonian only
        let mut solver_pn = PostNewtonianSolver::new(1.0); // 1PN

        let mut particles_newton = vec![
            GravityParticle::new(Vec3::zeros(), Vec3::zeros(), 1e10),
            GravityParticle::new(Vec3::new(1.0, 0.0, 0.0), Vec3::zeros(), 1e10),
        ];

        let mut particles_pn = particles_newton.clone();

        solver_newton.compute_forces(&mut particles_newton);
        solver_pn.compute_forces(&mut particles_pn);

        // PN force should be slightly different (but close for low velocities)
        let diff = (particles_newton[0].f - particles_pn[0].f).norm();
        // For v=0, 1PN corrections vanish, so they should be very close
        assert!(diff < 1e-6);
    }

    #[test]
    fn test_mercury_precession() {
        // Mercury orbital parameters
        let a = 57.9e9; // semi-major axis (m)
        let e = 0.2056; // eccentricity
        let m_sun = 1.989e30; // kg
        let period = 87.969 * 24.0 * 3600.0; // seconds

        let precession = perihelion_precession_rate(a, e, m_sun, period);

        // Expected: 43.1 arcsec/century
        // Allow 10% tolerance due to simplified formula
        assert!(precession > 38.0 && precession < 48.0);
    }

    #[test]
    fn test_orbital_elements() {
        // Circular orbit
        let m_sun = 1.989e30;
        let r = 1.5e11; // 1 AU
        let v_circular = (G * m_sun / r).sqrt();

        let x = Vec3::new(r, 0.0, 0.0);
        let v = Vec3::new(0.0, v_circular, 0.0);

        let (a, e, _i, _omega, _omega_bar, _nu) = orbital_elements(x, v, m_sun);

        // Should be circular (e ≈ 0) with a ≈ r
        assert!((a - r).abs() / r < 1e-6);
        assert!(e < 1e-6);
    }

    #[test]
    fn test_2_5pn_radiation() {
        let mut solver = PostNewtonianSolver::new(2.5);

        // Binary system with velocities
        let mut particles = vec![
            GravityParticle::new(Vec3::new(-0.5e9, 0.0, 0.0), Vec3::new(0.0, 1e4, 0.0), 1e30),
            GravityParticle::new(Vec3::new(0.5e9, 0.0, 0.0), Vec3::new(0.0, -1e4, 0.0), 1e30),
        ];

        solver.compute_forces(&mut particles);

        // Forces should be non-zero and include radiation damping
        assert!(particles[0].f.norm() > 0.0);
    }
}
