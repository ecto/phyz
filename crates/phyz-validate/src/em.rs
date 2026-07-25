//! FDTD (`phyz-em`) validation against closed-form electromagnetics.
//!
//! 1. **Numerical dispersion** — the measured oscillation frequency of a single
//!    Yee eigenmode versus the analytic Yee dispersion relation
//!    `sin²(ωΔt/2)/(cΔt)² = Σ_a sin²(k_a Δx/2)/Δx²`, and the deviation of that
//!    frequency from the continuum `ω = ck`, which must vanish as `Δx²`.
//! 2. **Rectangular cavity resonance** — the TM₁₁₀ mode of a PEC box against
//!    `f = (c/2)√((m/Lx)² + (n/Ly)² + (p/Lz)²)`.
//! 3. **Absorbing-boundary reflection** — the measured broadband reflection
//!    coefficient of both boundary options the crate offers.
//!
//! The 1-D and 2-D cases are run on the 3-D solver with the uniform directions
//! set periodic, which the update loops handle natively — no field surgery by
//! the harness.

use crate::report::{Convergence, ErrorKind, Status, Suite, Validation};
use phyz_em::analysis::peak_abs;
use phyz_em::{BoundaryCondition, CpmlConfig, YeeGrid, gaussian_pulse, reflection_db};

const CRATE: &str = "phyz-em";

/// Extract `ω` from a probe time series of a single undamped eigenmode.
///
/// For one spatial eigenmode the Yee leapfrog collapses exactly to the
/// three-term recurrence `E^{n+1} = 2cos(ωΔt) E^n − E^{n−1}`, independent of the
/// initial phase. Least squares over the whole record recovers `cos(ωΔt)` to
/// round-off when the update coefficients are correct.
fn fit_omega(series: &[f64], dt: f64) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for n in 1..series.len() - 1 {
        num += series[n] * (series[n + 1] + series[n - 1]);
        den += 2.0 * series[n] * series[n];
    }
    if den == 0.0 {
        return f64::NAN;
    }
    (num / den).clamp(-1.0, 1.0).acos() / dt
}

/// Analytic Yee dispersion relation, solved for `ω`.
///
/// `sin²(ωΔt/2) = (cΔt)² Σ_a sin²(k_a Δx/2) / Δx²`
fn yee_omega(k: [f64; 3], dx: f64, dt: f64, c: f64) -> f64 {
    let s: f64 = k
        .iter()
        .map(|&ka| (0.5 * ka * dx).sin().powi(2) / (dx * dx))
        .sum();
    let rhs = (c * dt) * (c * dt) * s;
    2.0 / dt * rhs.sqrt().clamp(-1.0, 1.0).asin()
}

/// A 1-D PEC-cavity standing mode: `(ω_measured, ω_yee, ω_continuum)`.
///
/// `Ex = sin(k z)` with `H = 0` is an exact eigenvector of the discrete operator
/// with Dirichlet walls, so the probe records a single pure mode.
fn cavity_1d(nz: usize, mode: usize, courant: f64, steps: usize) -> (f64, f64, f64) {
    let dx = 1e-3;
    let dt = courant * dx / (299_792_458.0 * 3_f64.sqrt());

    let mut g = YeeGrid::new(4, 4, nz, dx, dt);
    g.set_boundary(BoundaryCondition::PerfectConductor);
    // The transverse directions are uniform; making them periodic means the
    // update loops see no transverse gradient at all.
    g.set_periodic([true, true, false]);

    let c = g.c0;
    let l = (nz - 1) as f64 * dx;
    let kz = mode as f64 * std::f64::consts::PI / l;

    for k in 0..nz {
        let v = (kz * k as f64 * dx).sin();
        for i in 0..g.nx {
            for j in 0..g.ny {
                g.ex.set(i, j, k, v);
            }
        }
    }

    let k_probe = ((nz - 1) as f64 / (2.0 * mode as f64)).round() as usize;
    let mut series = Vec::with_capacity(steps + 1);
    series.push(g.ex.get(0, 0, k_probe));
    for _ in 0..steps {
        g.update_h_field();
        g.update_e_field();
        series.push(g.ex.get(0, 0, k_probe));
    }

    let omega = fit_omega(&series, dt);
    (omega, yee_omega([0.0, 0.0, kz], dx, dt, c), c * kz)
}

/// A 2-D TM₁₁₀ PEC-cavity mode: `(ω_measured, ω_yee, ω_continuum)`.
///
/// `Ez = sin(k_x x) sin(k_y y)`, uniform in z, so this exercises both the x- and
/// y-curl updates and the PEC walls on four faces.
fn cavity_tm110(n: usize, courant: f64, steps: usize) -> (f64, f64, f64) {
    let dx = 1e-3;
    let dt = courant * dx / (299_792_458.0 * 3_f64.sqrt());

    let mut g = YeeGrid::new(n, n, 4, dx, dt);
    g.set_boundary(BoundaryCondition::PerfectConductor);
    g.set_periodic([false, false, true]);

    let c = g.c0;
    let l = (n - 1) as f64 * dx;
    let kx = std::f64::consts::PI / l;
    let ky = kx;

    for i in 0..n {
        for j in 0..n {
            let v = (kx * i as f64 * dx).sin() * (ky * j as f64 * dx).sin();
            for k in 0..g.nz {
                g.ez.set(i, j, k, v);
            }
        }
    }

    let (i0, j0) = (n / 2, n / 2);
    let mut series = Vec::with_capacity(steps + 1);
    series.push(g.ez.get(i0, j0, 0));
    for _ in 0..steps {
        g.update_h_field();
        g.update_e_field();
        series.push(g.ez.get(i0, j0, 0));
    }

    let omega = fit_omega(&series, dt);
    (
        omega,
        yee_omega([kx, ky, 0.0], dx, dt, c),
        c * (kx * kx + ky * ky).sqrt(),
    )
}

/// Launch a broadband Gaussian pulse down a 1-D grid and record `Ex` at
/// `k_probe`, terminating the `+z` end with `bc`.
fn pulse_probe(
    nz: usize,
    k_src: usize,
    k_probe: usize,
    steps: usize,
    bc: BoundaryCondition,
) -> (Vec<f64>, f64) {
    let dz = 1e-3;
    let c = 299_792_458.0;
    let dt = 0.5 * dz / c;

    let mut g = YeeGrid::new(1, 1, nz, dz, dt);
    g.set_boundary(bc);
    g.set_periodic([true, true, false]);

    let (t0, spread) = (40.0 * dt, 12.0 * dt);
    let mut probe = Vec::with_capacity(steps);
    for n in 0..steps {
        g.update_h_field();
        g.update_e_field();
        g.ex.add(0, 0, k_src, gaussian_pulse(n as f64 * dt, t0, spread));
        probe.push(g.ex.get(0, 0, k_probe));
    }
    (probe, dt)
}

/// Worst-case broadband reflection, in dB, of the `+z` termination.
///
/// The reference is the *same* boundary condition in a domain long enough that
/// the far wall is never reached within the record. Both runs therefore share
/// their `−z` treatment and their source, and the only difference is whether the
/// `+z` wall was close enough to send anything back — so the difference of the
/// two probes is exactly what that wall reflected.
///
/// Using a different boundary for the reference would fold the `−z` behaviour
/// into the measurement: a CPML absorbs the backward-going half of the source
/// where a PEC wall returns it, and that difference alone swamps the reflection
/// being measured.
fn reflection(bc: BoundaryCondition) -> f64 {
    let nz_test = 300;
    let nz_ref = 4000;
    let k_src = 20;
    let k_probe = 60;
    // At cΔt/Δz = 1/2 the pulse advances half a cell per step, so the record
    // must cover the source→wall→probe round trip plus the pulse's own width.
    let steps = 2 * ((nz_test - k_src) + (nz_test - k_probe)) + 400;

    let (reference, dt) = pulse_probe(nz_ref, k_src, k_probe, steps, bc);
    let (test, _) = pulse_probe(nz_test, k_src, k_probe, steps, bc);

    if peak_abs(&reference) <= 0.0 {
        return f64::NAN;
    }

    // Isolate the *first* return. The probe sees the incident pulse, then the
    // `+z` reflection, and later that reflection bouncing off the `−z` wall and
    // passing a second time. Integrating the whole record mixes those and can
    // report more energy returning than left, which says nothing about the one
    // wall under test. Window each event to its own arrival instead.
    //
    // At half a cell per step, arrival times in steps are twice the path length
    // in cells, offset by the source's own delay `t0`.
    let t0_steps = 40;
    let half_window = 200;
    let t_incident = 2 * (k_probe - k_src) + t0_steps;
    let t_reflected = 2 * ((nz_test - 1 - k_src) + (nz_test - 1 - k_probe)) + t0_steps;

    let window = |signal: &[f64], centre: usize| -> Vec<f64> {
        let lo = centre.saturating_sub(half_window);
        let hi = (centre + half_window).min(signal.len());
        let mut out = vec![0.0; signal.len()];
        out[lo..hi].copy_from_slice(&signal[lo..hi]);
        out
    };

    let reflected = window(
        &test
            .iter()
            .zip(reference.iter())
            .map(|(t, r)| t - r)
            .collect::<Vec<f64>>(),
        t_reflected,
    );
    let reference = window(&reference, t_incident);

    // Only report the band the pulse actually carries. A reflection coefficient
    // is a ratio of spectra, so at frequencies where the incident spectrum is
    // near zero the ratio is dominated by round-off and can exceed 0 dB — which
    // says nothing about the boundary. Keep the frequencies carrying at least
    // 5% of the peak incident amplitude.
    let c = 299_792_458.0;
    let omega_max = c / (6.0 * 1e-3);
    let candidates: Vec<f64> = (1..=80).map(|i| omega_max * i as f64 / 80.0).collect();
    let incident = phyz_em::analysis::spectrum(&reference, &candidates, dt);
    let peak = incident.iter().fold(0.0_f64, |m, z| m.max(z.norm()));
    let omegas: Vec<f64> = candidates
        .iter()
        .zip(incident.iter())
        .filter(|(_, z)| z.norm() >= 0.05 * peak)
        .map(|(w, _)| *w)
        .collect();
    if omegas.is_empty() {
        return f64::NAN;
    }

    reflection_db(&reflected, &reference, &omegas, dt)
        .into_iter()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Fraction of the peak field energy still in the grid long after the pulse
/// should have left through the `+z` wall.
///
/// This is well conditioned for *any* boundary, good or bad, where a reflection
/// coefficient in dB is not: the ratio-of-spectra measurement above needs the
/// returning wave to be a single identifiable arrival, which stops being true
/// once a boundary reflects most of what hits it and the domain reverberates.
fn residual_energy_fraction(bc: BoundaryCondition) -> f64 {
    let nz = 300;
    let dz = 1e-3;
    let c = 299_792_458.0;
    let dt = 0.5 * dz / c;

    let mut g = YeeGrid::new(1, 1, nz, dz, dt);
    g.set_boundary(bc);
    g.set_periodic([true, true, false]);

    let (t0, spread) = (40.0 * dt, 12.0 * dt);
    let mut peak = 0.0_f64;
    // Long enough for several transits, so a good boundary has drained the grid.
    for n in 0..4000 {
        g.update_h_field();
        g.update_e_field();
        g.ex.add(0, 0, 150, gaussian_pulse(n as f64 * dt, t0, spread));
        if n < 200 {
            peak = peak.max(g.total_energy());
        }
    }
    if peak <= 0.0 {
        return f64::NAN;
    }
    g.total_energy() / peak
}

/// Record a one-sided "must be at or below" criterion.
fn at_most(mut v: Validation, limit: f64) -> Validation {
    v.error = (v.measured - limit).max(0.0);
    v.status = if v.error <= 0.0 {
        Status::Pass
    } else {
        Status::Fail
    };
    v
}

/// Run every FDTD validation.
pub fn run() -> Suite {
    let mut suite = Suite::new("Electromagnetics — FDTD on a Yee grid (`phyz-em`)");

    // ---- 1. Numerical dispersion vs the analytic Yee relation ----------------
    let (w_meas, w_yee, _) = cavity_1d(65, 1, 0.5, 400);
    suite.push(
        Validation::new(
            "em.yee_dispersion.m1",
            "Numerical dispersion, 1-D PEC cavity mode m=1",
            CRATE,
            "Yee (1966); Taflove & Hagness §4.3, eq. 4.14 — \
             sin²(ωΔt/2)/(cΔt)² = Σ_a sin²(k_aΔx/2)/Δx²",
            "ω of the m=1 standing mode (rad/s)",
            w_meas,
            w_yee,
            ErrorKind::Relative,
            1e-9,
        )
        .note(
            "For one spatial eigenmode the leapfrog is exactly \
             E^{n+1} = 2cos(ωΔt)E^n − E^{n−1}, so a correct implementation matches the \
             analytic Yee root to round-off. This directly tests the update coefficients \
             Δt/(μΔx) and Δt/(εΔx).",
        ),
    );

    let (w3, y3, _) = cavity_1d(33, 3, 0.5, 400);
    suite.push(Validation::new(
        "em.yee_dispersion.m3",
        "Numerical dispersion, 1-D PEC cavity mode m=3 (coarse, kΔx large)",
        CRATE,
        "Yee (1966); Taflove & Hagness §4.3, eq. 4.14",
        "ω of the m=3 standing mode (rad/s)",
        w3,
        y3,
        ErrorKind::Relative,
        1e-9,
    ));

    let mut samples = Vec::new();
    for &n in &[17_usize, 33, 65, 129] {
        let (m, _, cont) = cavity_1d(n, 1, 0.5, 400);
        samples.push((1.0 / (n - 1) as f64, (m - cont).abs() / cont));
    }
    let coarsest = samples.first().unwrap().1;
    let finest = samples.last().unwrap().1;
    // Three halvings at second order shrink the error 64×; allow 1.5× slack.
    let bound = 1.5 * coarsest / 64.0;
    suite.push(
        Validation::new(
            "em.dispersion_convergence",
            "Phase-velocity error vanishes as Δx² under refinement",
            CRATE,
            "Second-order accuracy of the Yee scheme; ω/ck − 1 = −(kΔx)²(1 − S²)/24 + O(Δx⁴)",
            "|ω_num − ck| / ck at the finest grid (Δx = L/128)",
            finest,
            0.0,
            ErrorKind::Absolute,
            bound,
        )
        .with_convergence(Convergence::fit("Δx/L", samples, 2.0, 0.15))
        .note(format!(
            "Tolerance is 1.5 × (error at Δx = L/16) / 64 = {bound:.3e}, derived from the \
             measured coarse grid rather than chosen after the fact."
        )),
    );

    // ---- 2. Rectangular cavity resonance ------------------------------------
    let (w_meas, w_yee, w_cont) = cavity_tm110(41, 0.5, 600);
    let two_pi = 2.0 * std::f64::consts::PI;
    suite.push(
        Validation::new(
            "em.cavity_tm110.discrete",
            "TM₁₁₀ square-cavity resonance vs the discrete Yee root",
            CRATE,
            "Yee dispersion relation with k = (π/L, π/L, 0)",
            "ω of the TM₁₁₀ mode (rad/s)",
            w_meas,
            w_yee,
            ErrorKind::Relative,
            1e-9,
        )
        .note("Exercises the x- and y-curl updates and the PEC boundary on four walls."),
    );
    suite.push(
        Validation::new(
            "em.cavity_tm110.physical",
            "TM₁₁₀ square-cavity resonance vs closed-form f = (c/2)√((m/Lx)²+(n/Ly)²)",
            CRATE,
            "Pozar, *Microwave Engineering* 4e, §6.3 — rectangular cavity resonant frequency",
            "resonant frequency (Hz), L = 40 mm, 41×41 cells",
            w_meas / two_pi,
            w_cont / two_pi,
            ErrorKind::Relative,
            2e-3,
        )
        .note("Residual error is the Yee grid-dispersion error at kΔx = π/40."),
    );

    let mut samples = Vec::new();
    for &n in &[11_usize, 21, 41, 81] {
        let (m, _, cont) = cavity_tm110(n, 0.5, 600);
        samples.push((1.0 / (n - 1) as f64, (m - cont).abs() / cont));
    }
    let coarsest = samples.first().unwrap().1;
    let finest = samples.last().unwrap().1;
    let bound = 1.5 * coarsest / 64.0;
    suite.push(
        Validation::new(
            "em.cavity_convergence",
            "Cavity resonance error vanishes as Δx² under refinement",
            CRATE,
            "Second-order accuracy of the Yee scheme",
            "|f_num − f_exact| / f_exact at 81×81",
            finest,
            0.0,
            ErrorKind::Absolute,
            bound,
        )
        .with_convergence(Convergence::fit("Δx/L", samples, 2.0, 0.15)),
    );

    // ---- 3. Absorbing-boundary reflection -----------------------------------
    let cpml_db = reflection(BoundaryCondition::Cpml(
        CpmlConfig::with_thickness(10).on_axes([false, false, true]),
    ));
    suite.push(
        at_most(
            Validation::new(
                "em.cpml_reflection",
                "Reflection coefficient of the CPML absorbing boundary",
                CRATE,
                "Roden & Gedney (2000); Taflove & Hagness ch. 7 — a correctly implemented \
                 10-cell CPML reaches R < −60 dB for a normally incident broadband pulse",
                "worst broadband reflection R (dB), 10-cell CPML",
                cpml_db,
                -60.0,
                ErrorKind::Absolute,
                0.0,
            ),
            -60.0,
        )
        .note(
            "One-sided criterion: any value at or below −60 dB passes. Measured by \
             differencing the probe against the same excitation in a 4000-cell domain, so \
             what remains is exactly what the boundary sent back.",
        ),
    );

    // A reflection coefficient in dB is only meaningful for a boundary that
    // returns an identifiable single arrival. The cheap absorber reflects most
    // of what reaches it and the domain reverberates, so characterise it by how
    // much energy it fails to drain instead — the same quantity for both
    // boundaries, well conditioned for both.
    let cpml_residual = residual_energy_fraction(BoundaryCondition::Cpml(
        CpmlConfig::with_thickness(10).on_axes([false, false, true]),
    ));
    let lossy_residual = residual_energy_fraction(BoundaryCondition::LossyAbsorber {
        thickness: 16,
        order: 3,
        sigma_max: 1.0,
    });

    // State the criterion in the form the comparison supports — CPML must leave
    // orders of magnitude less energy than the cheap layer — rather than
    // inventing an absolute floor, which would encode the source tail and the
    // round-off floor of this particular setup rather than anything physical.
    let limit = lossy_residual * 1e-3;
    suite.push(
        at_most(
            Validation::new(
                "em.cpml_drains_the_grid",
                "CPML drains a radiating pulse out of the grid",
                CRATE,
                "An absorbing boundary removes outgoing energy; after several transits an \
                 open domain should retain a negligible fraction of the peak field energy, \
                 and orders of magnitude less than an impedance-mismatched lossy layer",
                "residual field energy / peak, after 4000 steps on a 300-cell grid",
                cpml_residual,
                limit,
                ErrorKind::Absolute,
                0.0,
            ),
            limit,
        )
        .note(format!(
            "The cheap `LossyAbsorber` retains {lossy_residual:.3e} of the peak on the same \
             problem — {:.0}× more energy left ringing in the domain. Criterion: CPML must \
             leave at most 1/1000 of that, i.e. {limit:.3e}.",
            lossy_residual / cpml_residual.max(f64::MIN_POSITIVE)
        )),
    );

    suite.push(
        Validation::new(
            "em.lossy_absorber_residual",
            "Residual energy left by the cheap graded-conductivity absorber",
            CRATE,
            "`BoundaryCondition::LossyAbsorber` adds electric loss σ without the matching \
             magnetic loss σ* = σμ/ε, so it is impedance-mismatched at every frequency and \
             is documented as not being a PML",
            "residual field energy / peak, after 4000 steps on a 300-cell grid",
            lossy_residual,
            0.0,
            ErrorKind::Absolute,
            0.0,
        )
        .diagnostic()
        .note(
            "Reported, not failed: this boundary is offered as the cheap option and the \
             crate documents its limits. The measurement quantifies the gap so a caller can \
             decide; use `BoundaryCondition::Cpml` when reflections matter.",
        ),
    );

    suite
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omega_fit_is_exact_for_a_pure_cosine() {
        let dt = 1e-3;
        let w = 137.0;
        let s: Vec<f64> = (0..500)
            .map(|n| 0.7 * (w * n as f64 * dt + 0.3).cos())
            .collect();
        assert!((fit_omega(&s, dt) - w).abs() / w < 1e-10);
    }
}
