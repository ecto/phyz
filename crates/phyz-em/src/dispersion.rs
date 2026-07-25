//! Frequency-dispersive materials via the auxiliary differential equation
//! (ADE) method.
//!
//! A material is described by a high-frequency permittivity `eps_inf`, an
//! optional static conductivity, and a sum of poles:
//!
//! ```text
//!   ε(ω) = ε_inf + Σ_p χ_p(ω) + i σ / (ε₀ ω)          [e^{-iωt} convention]
//! ```
//!
//! Each pole carries its own polarization density `P_p`, advanced in time
//! alongside the fields. Second-order (Lorentz / Drude) poles obey
//!
//! ```text
//!   ∂²P/∂t² + γ ∂P/∂t + ω₀² P = ε₀ A E
//! ```
//!
//! with `A = Δε ω₀²` for Lorentz and `A = ω_p²`, `ω₀ = 0` for Drude — one code
//! path serves both. Debye poles are first order:
//!
//! ```text
//!   ∂P/∂t + P/τ = ε₀ Δε E / τ
//! ```
//!
//! Central differencing gives an explicit three-level recursion
//! `P^{n+1} = a₁ P^n + a₂ P^{n-1} + a₃ E^n` which depends only on *past* field
//! values, so the polarization can be advanced in a pass before the E update
//! and its time derivative fed into Ampère's law:
//!
//! ```text
//!   ε₀ε_inf (E^{n+1} − E^n)/Δt = ∇×H − Σ_p (P_p^{n+1} − P_p^n)/Δt − σ(E^{n+1}+E^n)/2
//! ```

use crate::grid::{Array3D, EPS0};

/// A minimal complex number, enough for permittivity/index arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Complex conjugate.
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    pub fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }

    /// Principal square root.
    pub fn sqrt(self) -> Self {
        let r = self.norm();
        if r == 0.0 {
            return Self::new(0.0, 0.0);
        }
        let re = ((r + self.re) / 2.0).sqrt();
        let im = ((r - self.re) / 2.0).sqrt() * if self.im < 0.0 { -1.0 } else { 1.0 };
        Self::new(re, im)
    }
}

impl std::ops::Add for C64 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
}

impl std::ops::Sub for C64 {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
}

impl std::ops::Mul for C64 {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl std::ops::Div for C64 {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Self::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
}

/// A single dispersive pole.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pole {
    /// Lorentz oscillator: χ(ω) = Δε ω₀² / (ω₀² − ω² − iγω).
    Lorentz {
        /// Oscillator strength Δε (dimensionless).
        delta_eps: f64,
        /// Resonance angular frequency ω₀ (rad/s).
        omega0: f64,
        /// Damping rate γ (rad/s).
        gamma: f64,
    },
    /// Drude free-carrier term: χ(ω) = −ω_p² / (ω² + iγω).
    Drude {
        /// Plasma angular frequency ω_p (rad/s).
        omega_p: f64,
        /// Collision rate γ (rad/s).
        gamma: f64,
    },
    /// Debye relaxation: χ(ω) = Δε / (1 − iωτ).
    Debye {
        /// Relaxation strength Δε.
        delta_eps: f64,
        /// Relaxation time τ (s).
        tau: f64,
    },
}

impl Pole {
    /// Complex susceptibility at angular frequency `omega` (e^{-iωt}).
    pub fn susceptibility(&self, omega: f64) -> C64 {
        match *self {
            Pole::Lorentz {
                delta_eps,
                omega0,
                gamma,
            } => {
                let num = C64::new(delta_eps * omega0 * omega0, 0.0);
                let den = C64::new(omega0 * omega0 - omega * omega, -gamma * omega);
                num / den
            }
            Pole::Drude { omega_p, gamma } => {
                let num = C64::new(-omega_p * omega_p, 0.0);
                let den = C64::new(omega * omega, gamma * omega);
                if den.norm() == 0.0 {
                    C64::new(f64::NEG_INFINITY, 0.0)
                } else {
                    num / den
                }
            }
            Pole::Debye { delta_eps, tau } => {
                let num = C64::new(delta_eps, 0.0);
                let den = C64::new(1.0, -omega * tau);
                num / den
            }
        }
    }

    /// ADE recursion coefficients `(a1, a2, a3)` for
    /// `P^{n+1} = a1 P^n + a2 P^{n-1} + a3 E^n`.
    pub fn ade_coeffs(&self, dt: f64) -> (f64, f64, f64) {
        match *self {
            Pole::Lorentz {
                delta_eps,
                omega0,
                gamma,
            } => second_order_coeffs(delta_eps * omega0 * omega0, omega0, gamma, dt),
            Pole::Drude { omega_p, gamma } => {
                second_order_coeffs(omega_p * omega_p, 0.0, gamma, dt)
            }
            Pole::Debye { delta_eps, tau } => {
                // Forward-Euler on a first-order relaxation.
                (1.0 - dt / tau, 0.0, EPS0 * delta_eps * dt / tau)
            }
        }
    }
}

/// Coefficients for ∂²P/∂t² + γ ∂P/∂t + ω₀² P = ε₀ A E, centered in time.
fn second_order_coeffs(a: f64, omega0: f64, gamma: f64, dt: f64) -> (f64, f64, f64) {
    let denom = 1.0 + gamma * dt / 2.0;
    let a1 = (2.0 - omega0 * omega0 * dt * dt) / denom;
    let a2 = -(1.0 - gamma * dt / 2.0) / denom;
    let a3 = EPS0 * a * dt * dt / denom;
    (a1, a2, a3)
}

/// A dispersive material: instantaneous response plus a set of poles.
#[derive(Debug, Clone, PartialEq)]
pub struct DispersiveMaterial {
    /// High-frequency (instantaneous) relative permittivity.
    pub eps_inf: f64,
    /// Static conductivity (S/m).
    pub sigma: f64,
    /// Relative permeability (non-dispersive).
    pub mu_r: f64,
    /// Dispersive poles.
    pub poles: Vec<Pole>,
}

impl DispersiveMaterial {
    /// Vacuum (the reserved material id 0).
    pub fn vacuum() -> Self {
        Self {
            eps_inf: 1.0,
            sigma: 0.0,
            mu_r: 1.0,
            poles: Vec::new(),
        }
    }

    /// A material with the given `eps_inf` and no poles.
    pub fn non_dispersive(eps_inf: f64) -> Self {
        Self {
            eps_inf,
            sigma: 0.0,
            mu_r: 1.0,
            poles: Vec::new(),
        }
    }

    /// A single-pole Drude metal.
    pub fn drude(eps_inf: f64, omega_p: f64, gamma: f64) -> Self {
        Self {
            eps_inf,
            sigma: 0.0,
            mu_r: 1.0,
            poles: vec![Pole::Drude { omega_p, gamma }],
        }
    }

    /// A single-pole Lorentz medium.
    pub fn lorentz(eps_inf: f64, delta_eps: f64, omega0: f64, gamma: f64) -> Self {
        Self {
            eps_inf,
            sigma: 0.0,
            mu_r: 1.0,
            poles: vec![Pole::Lorentz {
                delta_eps,
                omega0,
                gamma,
            }],
        }
    }

    /// A single-pole Debye medium.
    pub fn debye(eps_inf: f64, delta_eps: f64, tau: f64) -> Self {
        Self {
            eps_inf,
            sigma: 0.0,
            mu_r: 1.0,
            poles: vec![Pole::Debye { delta_eps, tau }],
        }
    }

    /// Add a pole, builder style.
    pub fn with_pole(mut self, pole: Pole) -> Self {
        self.poles.push(pole);
        self
    }

    /// Set the static conductivity, builder style.
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// Complex relative permittivity at angular frequency `omega`.
    pub fn permittivity(&self, omega: f64) -> C64 {
        let mut eps = C64::new(self.eps_inf, 0.0);
        for p in &self.poles {
            eps = eps + p.susceptibility(omega);
        }
        if self.sigma != 0.0 && omega != 0.0 {
            eps = eps + C64::new(0.0, self.sigma / (EPS0 * omega));
        }
        eps
    }

    /// Complex refractive index n = √(ε μ) at angular frequency `omega`.
    pub fn refractive_index(&self, omega: f64) -> C64 {
        self.permittivity(omega).scale(self.mu_r).sqrt()
    }

    /// Fresnel reflection coefficient for normal incidence from vacuum onto a
    /// half-space of this material.
    ///
    /// r = (n₁ − n₂) / (n₁ + n₂) with n₁ = 1.
    pub fn fresnel_normal(&self, omega: f64) -> C64 {
        let n2 = self.refractive_index(omega);
        let one = C64::new(1.0, 0.0);
        (one - n2) / (one + n2)
    }

    fn is_dispersive(&self) -> bool {
        !self.poles.is_empty()
    }
}

/// ADE auxiliary state: polarization densities for every pole slot.
///
/// Slot `p` holds the polarization of the `p`-th pole of whichever material
/// occupies a cell; materials with fewer poles simply leave the higher slots
/// at zero (their coefficients are zero, so they stay there).
pub struct DispersiveState {
    /// Number of pole slots (max over all materials).
    pub n_poles: usize,
    /// P at time level n, `[slot][component]`.
    p: Vec<[Array3D; 3]>,
    /// P at time level n−1.
    p_prev: Vec<[Array3D; 3]>,
    /// Σ_p (P^{n+1} − P^n) per component, refreshed each step.
    dp: [Array3D; 3],
    /// Per-material, per-pole ADE coefficients, flattened `[mat][pole]`.
    coeffs: Vec<Vec<(f64, f64, f64)>>,
}

impl DispersiveState {
    /// Allocate ADE state for the given material library, or `None` if no
    /// material is actually dispersive.
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dt: f64,
        materials: &[DispersiveMaterial],
    ) -> Option<Self> {
        if !materials.iter().any(|m| m.is_dispersive()) {
            return None;
        }
        let n_poles = materials.iter().map(|m| m.poles.len()).max().unwrap_or(0);
        let coeffs = materials
            .iter()
            .map(|m| m.poles.iter().map(|p| p.ade_coeffs(dt)).collect())
            .collect();

        let triple = || {
            [
                Array3D::zeros(nx, ny, nz),
                Array3D::zeros(nx, ny, nz),
                Array3D::zeros(nx, ny, nz),
            ]
        };

        Some(Self {
            n_poles,
            p: (0..n_poles).map(|_| triple()).collect(),
            p_prev: (0..n_poles).map(|_| triple()).collect(),
            dp: triple(),
            coeffs,
        })
    }

    /// Σ_p ΔP for component `c` (0=x, 1=y, 2=z) at a cell.
    #[inline]
    pub fn delta_p(&self, c: usize, i: usize, j: usize, k: usize) -> f64 {
        self.dp[c].get(i, j, k)
    }

    /// Advance all polarizations from level n to n+1 using the current E field.
    ///
    /// Must be called *before* the E-field update of the same step.
    pub fn advance(
        &mut self,
        mat_id: &[u32],
        e: [&Array3D; 3],
        nx: usize,
        ny: usize,
        nz: usize,
    ) {
        for d in self.dp.iter_mut() {
            d.clear();
        }

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let id = mat_id[i + j * nx + k * nx * ny] as usize;
                    if id == 0 {
                        continue;
                    }
                    let mat_coeffs = &self.coeffs[id];
                    if mat_coeffs.is_empty() {
                        continue;
                    }

                    #[allow(clippy::needless_range_loop)]
                    for c in 0..3 {
                        let e_val = e[c].get(i, j, k);
                        let mut sum = 0.0;
                        for (slot, &(a1, a2, a3)) in mat_coeffs.iter().enumerate() {
                            let p_n = self.p[slot][c].get(i, j, k);
                            let p_m = self.p_prev[slot][c].get(i, j, k);
                            let p_new = a1 * p_n + a2 * p_m + a3 * e_val;
                            self.p_prev[slot][c].set(i, j, k, p_n);
                            self.p[slot][c].set(i, j, k, p_new);
                            sum += p_new - p_n;
                        }
                        self.dp[c].set(i, j, k, sum);
                    }
                }
            }
        }
    }

    /// Zero all polarization memory.
    pub fn reset(&mut self) {
        for slot in 0..self.n_poles {
            for c in 0..3 {
                self.p[slot][c].clear();
                self.p_prev[slot][c].clear();
            }
        }
        for d in self.dp.iter_mut() {
            d.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_sqrt_matches_square() {
        for &(re, im) in &[(4.0, 0.0), (-1.0, 0.0), (3.0, 4.0), (2.0, -5.0)] {
            let z = C64::new(re, im);
            let s = z.sqrt();
            let back = s * s;
            assert!((back.re - re).abs() < 1e-9, "{re}+{im}i");
            assert!((back.im - im).abs() < 1e-9, "{re}+{im}i");
        }
    }

    #[test]
    fn drude_permittivity_is_negative_below_plasma_frequency() {
        let wp = 1.0e16;
        let m = DispersiveMaterial::drude(1.0, wp, 0.0);
        let eps_lo = m.permittivity(0.5 * wp);
        let eps_hi = m.permittivity(2.0 * wp);
        // ε = 1 − ωp²/ω²: −3 at ω = ωp/2, +0.75 at ω = 2ωp.
        assert!((eps_lo.re - (-3.0)).abs() < 1e-9);
        assert!((eps_hi.re - 0.75).abs() < 1e-9);
    }

    #[test]
    fn lorentz_static_limit_is_eps_inf_plus_delta() {
        let m = DispersiveMaterial::lorentz(2.0, 3.0, 1e15, 1e13);
        let eps = m.permittivity(1e9); // ω ≪ ω₀
        assert!((eps.re - 5.0).abs() < 1e-6, "got {}", eps.re);
    }

    #[test]
    fn lossless_dielectric_fresnel_matches_index_formula() {
        let m = DispersiveMaterial::non_dispersive(4.0);
        let r = m.fresnel_normal(1e15);
        // n = 2 → r = (1−2)/(1+2) = −1/3
        assert!((r.re - (-1.0 / 3.0)).abs() < 1e-12);
        assert!(r.im.abs() < 1e-12);
    }

    #[test]
    fn debye_limits() {
        let m = DispersiveMaterial::debye(2.0, 3.0, 1e-12);
        let lo = m.permittivity(1e6); // ωτ ≪ 1
        let hi = m.permittivity(1e18); // ωτ ≫ 1
        assert!((lo.re - 5.0).abs() < 1e-6);
        assert!((hi.re - 2.0).abs() < 1e-6);
    }

    #[test]
    fn ade_recursion_reproduces_lorentz_susceptibility() {
        // Drive a single Lorentz pole with a sinusoid and compare the
        // steady-state P/E ratio against the analytic χ(ω).
        let omega0 = 1.0e15;
        let gamma = 1.0e13;
        let delta_eps = 2.0;
        let pole = Pole::Lorentz {
            delta_eps,
            omega0,
            gamma,
        };
        let omega = 0.7e15;
        let dt = 2.0 * std::f64::consts::PI / omega / 400.0;
        let (a1, a2, a3) = pole.ade_coeffs(dt);

        let mut p = 0.0;
        let mut p_prev = 0.0;
        let n_steps = 200_000;
        // Track the last full period to extract amplitude and phase.
        let period_steps = (2.0 * std::f64::consts::PI / (omega * dt)).round() as usize;
        let mut acc_cos = 0.0;
        let mut acc_sin = 0.0;
        for n in 0..n_steps {
            let t = n as f64 * dt;
            let e = (omega * t).sin();
            let p_new = a1 * p + a2 * p_prev + a3 * e;
            p_prev = p;
            p = p_new;
            if n >= n_steps - period_steps {
                acc_cos += p * (omega * t).cos();
                acc_sin += p * (omega * t).sin();
            }
        }
        let norm = 2.0 / period_steps as f64;
        // For E = sin(ωt) under e^{-iωt}, P/ε₀ = Re[χ] sin(ωt) − Im[χ] cos(ωt).
        let chi_re = acc_sin * norm / EPS0;
        let chi_im = -acc_cos * norm / EPS0;
        let chi = pole.susceptibility(omega);

        assert!(
            (chi_re - chi.re).abs() < 0.02 * chi.norm(),
            "Re: ADE {chi_re} vs analytic {}",
            chi.re
        );
        assert!(
            (chi_im - chi.im).abs() < 0.02 * chi.norm(),
            "Im: ADE {chi_im} vs analytic {}",
            chi.im
        );
    }
}
