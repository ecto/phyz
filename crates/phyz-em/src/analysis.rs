//! Signal-analysis helpers used to turn FDTD time series into the quantitative
//! numbers a validation test can assert on (amplitudes, phases, reflection
//! coefficients in dB).

use crate::dispersion::C64;

/// Ratio expressed in decibels: `20 log10(ratio)`.
///
/// Returns −∞ for a ratio of zero, which is the right answer for a perfectly
/// absorbing boundary and prints readably in assertion messages.
pub fn db(ratio: f64) -> f64 {
    if ratio <= 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * ratio.log10()
    }
}

/// Single-frequency DFT (Goertzel-style projection) of a uniformly sampled
/// real signal.
///
/// Returns the complex phasor `X` such that the signal is approximately
/// `Re[X] cos(ωt) + Im[X] sin(ωt)`, i.e. amplitude `|X|` and phase
/// `atan2(Im, Re)` in the convention `s(t) ≈ |X| cos(ωt + phase)`.
pub fn phasor_at(signal: &[f64], omega: f64, dt: f64) -> C64 {
    let n = signal.len();
    if n == 0 {
        return C64::new(0.0, 0.0);
    }
    let mut acc_c = 0.0;
    let mut acc_s = 0.0;
    for (i, &s) in signal.iter().enumerate() {
        let t = i as f64 * dt;
        acc_c += s * (omega * t).cos();
        acc_s += s * (omega * t).sin();
    }
    let norm = 2.0 / n as f64;
    C64::new(acc_c * norm, -acc_s * norm)
}

/// Amplitude and phase of `signal` at angular frequency `omega`.
///
/// `s(t) ≈ amplitude · cos(ω t + phase)`.
pub fn amplitude_at(signal: &[f64], omega: f64, dt: f64) -> (f64, f64) {
    let x = phasor_at(signal, omega, dt);
    (x.norm(), x.im.atan2(x.re))
}

/// Naive DFT of a real signal onto a set of angular frequencies.
///
/// Used by broadband (Gaussian-pulse) validations, where one run yields the
/// response at many frequencies at once.
pub fn spectrum(signal: &[f64], omegas: &[f64], dt: f64) -> Vec<C64> {
    omegas.iter().map(|&w| phasor_at(signal, w, dt)).collect()
}

/// Peak absolute value of a slice.
pub fn peak_abs(signal: &[f64]) -> f64 {
    signal.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

/// Broadband reflection coefficient magnitude, in dB, per frequency.
///
/// `reflected` and `incident` are time series recorded at the same point;
/// the ratio of their spectra is the reflection coefficient.
pub fn reflection_db(reflected: &[f64], incident: &[f64], omegas: &[f64], dt: f64) -> Vec<f64> {
    let r = spectrum(reflected, omegas, dt);
    let i = spectrum(incident, omegas, dt);
    r.iter()
        .zip(i.iter())
        .map(|(rr, ii)| {
            if ii.norm() == 0.0 {
                f64::NAN
            } else {
                db(rr.norm() / ii.norm())
            }
        })
        .collect()
}

/// Complex reflection coefficient per frequency (magnitude *and* phase).
pub fn reflection_coefficient(
    reflected: &[f64],
    incident: &[f64],
    omegas: &[f64],
    dt: f64,
) -> Vec<C64> {
    let r = spectrum(reflected, omegas, dt);
    let i = spectrum(incident, omegas, dt);
    r.iter()
        .zip(i.iter())
        .map(|(rr, ii)| {
            if ii.norm() == 0.0 {
                C64::new(f64::NAN, f64::NAN)
            } else {
                *rr / *ii
            }
        })
        .collect()
}

/// A Gaussian-modulated sinusoid, the standard broadband FDTD excitation.
///
/// `t0` is the pulse centre and `spread` its 1/e half-width.
pub fn gaussian_pulse(t: f64, t0: f64, spread: f64) -> f64 {
    let x = (t - t0) / spread;
    (-x * x).exp()
}

/// A differentiated Gaussian: zero DC content, which keeps a static charge
/// from accumulating when injected as a soft source.
pub fn ricker_pulse(t: f64, t0: f64, spread: f64) -> f64 {
    let x = (t - t0) / spread;
    -2.0 * x * (-x * x).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phasor_recovers_amplitude_and_phase() {
        let omega = 2.0 * std::f64::consts::PI * 1e9;
        let dt = 1.0 / (1e9 * 200.0);
        // Exactly 10 periods so the projection is orthogonal.
        let n = 2000;
        let phase = 0.7;
        let amp = 3.5;
        let sig: Vec<f64> = (0..n)
            .map(|i| amp * (omega * i as f64 * dt + phase).cos())
            .collect();

        let (a, p) = amplitude_at(&sig, omega, dt);
        assert!((a - amp).abs() < 1e-6, "amplitude {a}");
        assert!((p - phase).abs() < 1e-6, "phase {p}");
    }

    #[test]
    fn db_conversion() {
        assert!((db(1.0)).abs() < 1e-12);
        assert!((db(0.1) + 20.0).abs() < 1e-12);
        assert!((db(0.001) + 60.0).abs() < 1e-12);
        assert_eq!(db(0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn reflection_db_of_known_ratio() {
        let omega = 2.0 * std::f64::consts::PI * 1e9;
        let dt = 1.0 / (1e9 * 200.0);
        let n = 2000;
        let inc: Vec<f64> = (0..n).map(|i| (omega * i as f64 * dt).cos()).collect();
        let refl: Vec<f64> = inc.iter().map(|v| 0.01 * v).collect();
        let r = reflection_db(&refl, &inc, &[omega], dt);
        assert!((r[0] + 40.0).abs() < 1e-6, "got {} dB", r[0]);
    }
}
