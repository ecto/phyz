//! Collision operators and sub-grid turbulence closure.
//!
//! # Why TRT is the default
//!
//! Single-relaxation-time BGK relaxes every non-equilibrium moment at the same
//! rate `1/τ`. Two consequences matter in practice:
//!
//! 1. The kinetic (ghost) moments, which carry no hydrodynamics, relax at the
//!    same rate as the stress. At low viscosity (`τ → 1/2`) they are barely
//!    damped and the scheme goes unstable well before the physics does.
//! 2. The effective location of a bounce-back wall drifts with `τ`. For plane
//!    Poiseuille flow the wall sits exactly halfway between nodes only when
//!    `(τ - 1/2)² = 3/16`; at any other viscosity the measured profile is a
//!    parabola of the *wrong width*. The discretisation error is therefore a
//!    function of viscosity, which is indefensible — refining the physics
//!    changes the geometry.
//!
//! TRT ([`CollisionModel::Trt`]) splits every population into its symmetric
//! (even) and antisymmetric (odd) parts and gives them independent rates `ω⁺`
//! and `ω⁻`. Viscosity is set by `ω⁺` alone, so `ω⁻` is free; fixing the *magic
//! parameter*
//!
//! ```text
//! Λ = (1/ω⁺ - 1/2)(1/ω⁻ - 1/2)
//! ```
//!
//! at [`MAGIC_BOUNCE_BACK`] (`3/16`) pins the bounce-back wall exactly halfway
//! between nodes **for every viscosity**. That single change removes the
//! viscosity-dependent boundary error entirely — it is the large majority of
//! what MRT buys, at essentially the cost of BGK (one extra pass over opposite
//! pairs, no matrix transform). That is why it is the default here.
//!
//! [`CollisionModel::Mrt`] is provided for the cases TRT does not cover:
//! independent control of the bulk viscosity and of the two ghost modes, which
//! matters for acoustics and for the last bit of high-Reynolds stability. It
//! costs two 9×9 transforms per node. Its `s_q` is chosen to reproduce
//! `Λ = 3/16`, so it inherits TRT's boundary behaviour.

/// Magic parameter that places a bounce-back wall exactly halfway between
/// nodes, independent of viscosity. The correct default for wall-bounded flow.
pub const MAGIC_BOUNCE_BACK: f64 = 3.0 / 16.0;

/// Magic parameter with the best stability margin (Ginzburg & d'Humières).
/// Prefer this over [`MAGIC_BOUNCE_BACK`] only for unbounded/periodic flow.
pub const MAGIC_STABILITY: f64 = 1.0 / 4.0;

/// Magic parameter that makes the *advection* error vanish.
pub const MAGIC_ADVECTION: f64 = 1.0 / 6.0;

/// Choice of collision operator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CollisionModel {
    /// Single relaxation time. Simple and fast; unstable at low viscosity and
    /// has a viscosity-dependent wall position. Kept for reference and for
    /// regression tests that demonstrate the defect.
    Bgk,
    /// Two relaxation times, parameterised by the magic number
    /// `Λ = (1/ω⁺ - 1/2)(1/ω⁻ - 1/2)`. See the module docs.
    Trt {
        /// Magic parameter Λ. Use [`MAGIC_BOUNCE_BACK`] for wall-bounded flow.
        magic: f64,
    },
    /// Full multiple-relaxation-time collision in moment space.
    ///
    /// Implemented for D2Q9. The D3Q19 solver falls back to TRT with
    /// [`MAGIC_BOUNCE_BACK`], which is equivalent for wall accuracy.
    Mrt,
}

impl Default for CollisionModel {
    fn default() -> Self {
        CollisionModel::Trt {
            magic: MAGIC_BOUNCE_BACK,
        }
    }
}

/// Sub-grid turbulence closure applied on top of the collision operator.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Turbulence {
    /// Direct simulation — no model. Only valid if the grid resolves the
    /// dissipation scale.
    #[default]
    None,
    /// Smagorinsky LES. The eddy viscosity is computed locally from the
    /// second moment of the non-equilibrium distribution, which is already
    /// available during collision — no finite differences, no extra sweep.
    Smagorinsky {
        /// Smagorinsky constant `C_s` times the filter width (Δ = 1 lattice
        /// unit), so this is just `C_s`. Typical range 0.1–0.2.
        cs: f64,
    },
}

/// Solve `Λ = (1/ω⁺ - 1/2)(1/ω⁻ - 1/2)` for `ω⁻`.
#[inline]
pub fn trt_omega_minus(omega_plus: f64, magic: f64) -> f64 {
    let lambda_plus = 1.0 / omega_plus - 0.5;
    1.0 / (magic / lambda_plus + 0.5)
}

/// Smagorinsky-corrected relaxation time.
///
/// The filtered strain rate is recovered from the non-equilibrium momentum
/// flux `Q_αβ = Σ_i e_iα e_iβ (f_i - f_i^eq)` via `S_αβ ≈ -Q_αβ / (2 ρ c_s² τ)`,
/// which turns `ν_t = (C_s Δ)² |S̄|` into a quadratic for the total `τ`:
///
/// ```text
/// τ_total = ½ ( τ₀ + sqrt(τ₀² + 18 C_s² |Q| / ρ) ),   |Q| = sqrt(2 Q_αβ Q_αβ)
/// ```
///
/// (Hou et al., 1996.) Reduces to `τ₀` when the flow is locally resolved.
#[inline]
pub fn smagorinsky_tau(tau0: f64, rho: f64, q_norm: f64, cs: f64) -> f64 {
    if rho <= 1e-12 {
        return tau0;
    }
    0.5 * (tau0 + (tau0 * tau0 + 18.0 * cs * cs * q_norm / rho).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trt_reduces_to_bgk_at_matching_magic() {
        // BGK is the special case ω⁻ = ω⁺, i.e. Λ = (1/ω - 1/2)².
        let tau = 0.8;
        let omega = 1.0 / tau;
        let magic = (tau - 0.5) * (tau - 0.5);
        let om = trt_omega_minus(omega, magic);
        assert!((om - omega).abs() < 1e-12, "{om} vs {omega}");
    }

    #[test]
    fn magic_bounce_back_is_viscosity_independent() {
        // Λ stays 3/16 by construction whatever ω⁺ is — that is the whole point.
        for tau in [0.51, 0.6, 0.9, 1.5, 3.0] {
            let op = 1.0 / tau;
            let om = trt_omega_minus(op, MAGIC_BOUNCE_BACK);
            let lambda = (1.0 / op - 0.5) * (1.0 / om - 0.5);
            assert!((lambda - MAGIC_BOUNCE_BACK).abs() < 1e-12);
            assert!(om > 0.0 && om < 2.0, "ω⁻ out of stable range: {om}");
        }
    }

    #[test]
    fn smagorinsky_is_inert_on_resolved_flow() {
        assert!((smagorinsky_tau(0.8, 1.0, 0.0, 0.1) - 0.8).abs() < 1e-14);
        // and strictly increases viscosity when strain is present
        assert!(smagorinsky_tau(0.8, 1.0, 0.5, 0.16) > 0.8);
    }
}
