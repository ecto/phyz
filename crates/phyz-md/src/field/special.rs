//! Special functions needed by the Ewald/PME electrostatics.
//!
//! `std` has no error function, and the accuracy of the Ewald real-space sum is
//! bounded directly by the accuracy of `erfc` — a 1e-7 approximation would put
//! a floor under the Madelung-constant validation. These implementations reach
//! ~1e-15 relative accuracy by splitting at |x| = 2: a stable exponential series
//! below, a Lentz-evaluated continued fraction above.

use std::f64::consts::PI;

/// Error function `erf(x) = 2/√π ∫₀ˣ e^{-t²} dt`.
pub fn erf(x: f64) -> f64 {
    if x.abs() >= 2.0 {
        return x.signum() * (1.0 - erfc_cf(x.abs()));
    }
    erf_series(x)
}

/// Complementary error function `erfc(x) = 1 - erf(x)`.
///
/// Evaluated directly (not as `1 - erf`) for large `x`, so the exponentially
/// small tail keeps full relative precision — which is exactly the regime the
/// Ewald real-space cutoff lives in.
pub fn erfc(x: f64) -> f64 {
    if x >= 2.0 {
        return erfc_cf(x);
    }
    if x <= -2.0 {
        return 2.0 - erfc_cf(-x);
    }
    1.0 - erf_series(x)
}

/// `erf` via the everywhere-positive series
/// `erf(x) = 2x/√π · e^{-x²} · Σₙ (2x²)ⁿ / (1·3···(2n+1))`.
///
/// Every term is positive, so there is no cancellation; for |x| ≤ 2 it
/// converges in a few dozen terms.
fn erf_series(x: f64) -> f64 {
    let x2 = x * x;
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 1..200 {
        term *= 2.0 * x2 / (2.0 * n as f64 + 1.0);
        sum += term;
        if term <= f64::EPSILON * sum {
            break;
        }
    }
    2.0 * x / PI.sqrt() * (-x2).exp() * sum
}

/// `erfc(x)` for `x ≥ 2` via the continued fraction
/// `erfc(x) = e^{-x²}/√π · 1/(x + ½/(x + 1/(x + 3/2/(x + …))))`,
/// evaluated with the modified Lentz algorithm.
fn erfc_cf(x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut f = TINY;
    let mut c = f;
    let mut d = 0.0f64;
    for i in 0..300 {
        // Continued fraction terms: b₀ = x, and for i ≥ 1 a_i = i/2, b_i = x.
        let a = if i == 0 { 1.0 } else { 0.5 * i as f64 };
        let b = x;
        d = b + a * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + a / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < f64::EPSILON {
            break;
        }
    }
    (-x * x).exp() / PI.sqrt() * f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        // Reference values (Abramowitz & Stegun table / exact limits).
        for &(x, want) in &[
            (0.0, 0.0),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (3.0, 0.999_977_909_503_001_4),
        ] {
            assert!((erf(x) - want).abs() < 1e-14, "erf({x}) = {}", erf(x));
            assert!((erf(-x) + want).abs() < 1e-14);
        }
    }

    #[test]
    fn erfc_keeps_relative_accuracy_in_the_tail() {
        // erfc(5) ≈ 1.5374597944280351e-12 — a `1 - erf` implementation would
        // lose every significant digit here.
        let want = 1.537_459_794_428_035_1e-12;
        assert!((erfc(5.0) - want).abs() < 1e-26);
        assert!((erfc(5.0) / want - 1.0).abs() < 1e-12);
    }

    #[test]
    fn erf_and_erfc_are_complementary() {
        for i in 0..80 {
            let x = -4.0 + i as f64 * 0.1;
            assert!((erf(x) + erfc(x) - 1.0).abs() < 1e-14, "x = {x}");
        }
    }

    #[test]
    fn erf_derivative_matches_finite_difference() {
        // d/dx erf(x) = 2/√π e^{-x²}
        let h = 1e-6;
        for i in 0..30 {
            let x = 0.1 + i as f64 * 0.1;
            let fd = (erf(x + h) - erf(x - h)) / (2.0 * h);
            let exact = 2.0 / PI.sqrt() * (-x * x).exp();
            assert!((fd - exact).abs() < 1e-8, "x = {x}");
        }
    }
}
