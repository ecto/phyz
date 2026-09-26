//! Reproducible floating-point primitives.
//!
//! `+`, `-`, `*`, `/` and `sqrt` are correctly rounded by IEEE-754, so every
//! conforming target computes them to the same bits. The transcendental
//! functions are not: `sin`, `cos`, `atan2`, `pow` and `tanh` come from the
//! platform's libm, and Apple's, glibc's and musl's implementations disagree
//! in the last ulp for many inputs. That is enough. A 1-ulp difference in one
//! `sin` inside `integrate_configuration` is amplified by a contact-rich
//! rollout until the trajectories are visibly different, which is exactly the
//! failure mode where you cannot tell chaos from a bug.
//!
//! So the rollout path does not call the platform's libm. It calls this
//! module, which forwards to the [`libm`] crate — a pure-Rust port of musl's
//! libm that computes the same bits from the same inputs on every target,
//! because it is the same Rust source compiled with the same IEEE semantics
//! (Rust never contracts to FMA or reassociates FP without explicit opt-in).
//!
//! # The `portable-fp` feature
//!
//! On by default. Turning it off routes every function here straight to `std`,
//! which is faster (the platform libm is usually vectorized and tuned) and
//! gives up cross-platform bit equality. Do that only if you have measured the
//! difference and do not need to compare numbers across machines.
//!
//! # What this does *not* fix
//!
//! Nothing here makes a chaotic rollout stop being chaotic. It makes the
//! *same* rollout on two machines take the same path through that chaos. See
//! `docs/determinism.md` for the full contract.

/// Sine.
#[inline]
pub fn sin(x: f64) -> f64 {
    #[cfg(feature = "portable-fp")]
    {
        libm::sin(x)
    }
    #[cfg(not(feature = "portable-fp"))]
    {
        x.sin()
    }
}

/// Cosine.
#[inline]
pub fn cos(x: f64) -> f64 {
    #[cfg(feature = "portable-fp")]
    {
        libm::cos(x)
    }
    #[cfg(not(feature = "portable-fp"))]
    {
        x.cos()
    }
}

/// `(sin(x), cos(x))`, matching [`f64::sin_cos`]'s tuple order.
#[inline]
pub fn sin_cos(x: f64) -> (f64, f64) {
    #[cfg(feature = "portable-fp")]
    {
        libm::sincos(x)
    }
    #[cfg(not(feature = "portable-fp"))]
    {
        x.sin_cos()
    }
}

/// Two-argument arctangent.
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    #[cfg(feature = "portable-fp")]
    {
        libm::atan2(y, x)
    }
    #[cfg(not(feature = "portable-fp"))]
    {
        y.atan2(x)
    }
}

/// Arccosine.
#[inline]
pub fn acos(x: f64) -> f64 {
    #[cfg(feature = "portable-fp")]
    {
        libm::acos(x)
    }
    #[cfg(not(feature = "portable-fp"))]
    {
        x.acos()
    }
}

/// `x^y` for real `y`.
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    #[cfg(feature = "portable-fp")]
    {
        libm::pow(x, y)
    }
    #[cfg(not(feature = "portable-fp"))]
    {
        x.powf(y)
    }
}

/// Hyperbolic tangent.
#[inline]
pub fn tanh(x: f64) -> f64 {
    #[cfg(feature = "portable-fp")]
    {
        libm::tanh(x)
    }
    #[cfg(not(feature = "portable-fp"))]
    {
        x.tanh()
    }
}

/// Natural exponential.
#[inline]
pub fn exp(x: f64) -> f64 {
    #[cfg(feature = "portable-fp")]
    {
        libm::exp(x)
    }
    #[cfg(not(feature = "portable-fp"))]
    {
        x.exp()
    }
}

/// Natural logarithm.
#[inline]
pub fn ln(x: f64) -> f64 {
    #[cfg(feature = "portable-fp")]
    {
        libm::log(x)
    }
    #[cfg(not(feature = "portable-fp"))]
    {
        x.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The portable path must still be *correct*, not merely consistent: musl
    /// and the platform libm agree to within an ulp or so, and a bug in the
    /// wiring (a swapped `sincos` tuple, say) would show up as a gross
    /// mismatch rather than a last-digit one.
    #[test]
    fn agrees_with_std_to_a_few_ulp() {
        let xs: [f64; 12] = [
            -3.0,
            -1.25,
            -0.5,
            -1e-9,
            0.0,
            1e-9,
            0.5,
            1.0,
            1.25,
            2.0,
            3.0,
            std::f64::consts::TAU,
        ];
        for x in xs {
            let tol = 8.0 * f64::EPSILON * (1.0 + x.abs());
            assert!((sin(x) - x.sin()).abs() <= tol, "sin({x})");
            assert!((cos(x) - x.cos()).abs() <= tol, "cos({x})");
            assert!((tanh(x) - x.tanh()).abs() <= tol, "tanh({x})");
            let (s, c) = sin_cos(x);
            assert_eq!((s, c), (sin(x), cos(x)), "sin_cos({x}) tuple order");
            assert!((atan2(x, 1.5) - x.atan2(1.5)).abs() <= tol, "atan2({x})");
            if x > 0.0 {
                assert!((pow(x, 1.5) - x.powf(1.5)).abs() <= 8.0 * tol, "pow({x})");
                assert!((ln(x) - x.ln()).abs() <= 8.0 * tol, "ln({x})");
            }
        }
        assert!((acos(0.25) - 0.25f64.acos()).abs() <= 8.0 * f64::EPSILON);
    }
}
