//! FDTD (`phyz-em`) validation against closed-form electromagnetics.
//!
//! Three benchmarks:
//!
//! 1. **Numerical dispersion** — the measured oscillation frequency of a single
//!    Yee eigenmode versus the analytic Yee dispersion relation
//!    `sin²(ωΔt/2)/(cΔt)² = Σ_a sin²(k_a Δx/2)/Δx²`, and the deviation of that
//!    frequency from the continuum `ω = ck` (which must vanish as `Δx²`).
//! 2. **Rectangular cavity resonance** — the TM₁₁₀ mode of a PEC box against
//!    `f = (c/2)√((m/Lx)² + (n/Ly)² + (p/Lz)²)`.
//! 3. **Absorbing-boundary reflection** — the measured reflection coefficient of
//!    the layer that `phyz_em::PmlLayer` builds.
//!
//! ## Harness note: symmetry projection
//!
//! `YeeGrid`'s update loops skip the highest index in each direction and read
//! out-of-range cells as zero, so the outermost slab is not a usable boundary.
//! Benchmarks 1 and 3 are 1-D problems and benchmark 2 is 2-D; the harness runs
//! them on the 3-D solver and, after each step, re-broadcasts the interior
//! column/slab across the directions the exact solution is uniform in. That
//! enforces the intended symmetry rather than substituting for physics: the
//! curl updates along the non-uniform directions are untouched.

use crate::report::{Convergence, ErrorKind, Suite, Validation};
use phyz_em::{BoundaryCondition, PmlLayer, YeeGrid};

const CRATE: &str = "phyz-em";

/// Broadcast the column at `(i0, j0)` across all `(i, j)` for every field.
fn project_xy_uniform(g: &mut YeeGrid, i0: usize, j0: usize) {
    for k in 0..g.nz {
        let vals = [
            g.ex.get(i0, j0, k),
            g.ey.get(i0, j0, k),
            g.ez.get(i0, j0, k),
            g.hx.get(i0, j0, k),
            g.hy.get(i0, j0, k),
            g.hz.get(i0, j0, k),
        ];
        for i in 0..g.nx {
            for j in 0..g.ny {
                g.ex.set(i, j, k, vals[0]);
                g.ey.set(i, j, k, vals[1]);
                g.ez.set(i, j, k, vals[2]);
                g.hx.set(i, j, k, vals[3]);
                g.hy.set(i, j, k, vals[4]);
                g.hz.set(i, j, k, vals[5]);
            }
        }
    }
}

/// Broadcast the slab at `k0` across all `k` for every field.
fn project_z_uniform(g: &mut YeeGrid, k0: usize) {
    for i in 0..g.nx {
        for j in 0..g.ny {
            let vals = [
                g.ex.get(i, j, k0),
                g.ey.get(i, j, k0),
                g.ez.get(i, j, k0),
                g.hx.get(i, j, k0),
                g.hy.get(i, j, k0),
                g.hz.get(i, j, k0),
            ];
            for k in 0..g.nz {
                g.ex.set(i, j, k, vals[0]);
                g.ey.set(i, j, k, vals[1]);
                g.ez.set(i, j, k, vals[2]);
                g.hx.set(i, j, k, vals[3]);
                g.hy.set(i, j, k, vals[4]);
                g.hz.set(i, j, k, vals[5]);
            }
        }
    }
}

/// Extract `ω` from a probe time series of a single undamped eigenmode.
///
/// For one spatial eigenmode the Yee leapfrog collapses exactly to the
/// three-term recurrence `E^{n+1} = 2cos(ωΔt) E^n − E^{n−1}`, independent of the
/// initial phase. Least-squares over the whole record gives `cos(ωΔt)` to
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

/// Analytic Yee dispersion relation: solve for `ω` given the wavevector.
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

/// Run a 1-D PEC-cavity standing mode and return `(ω_measured, ω_yee, ω_continuum)`.
fn cavity_1d(nz: usize, mode: usize, courant: f64, steps: usize) -> (f64, f64, f64) {
    let dx = 1e-3;
    let (nx, ny) = (4, 4);
    let c_nominal = 299_792_458.0;
    let dt = courant * dx / (c_nominal * 3_f64.sqrt());

    let mut g = YeeGrid::new(nx, ny, nz, dx, dt);
    let c = g.c0;
    let l = (nz - 1) as f64 * dx;
    let kz = mode as f64 * std::f64::consts::PI / l;

    for k in 0..nz {
        let v = (kz * k as f64 * dx).sin();
        for i in 0..nx {
            for j in 0..ny {
                g.ex.set(i, j, k, v);
            }
        }
    }

    let k_probe = ((nz - 1) as f64 / (2.0 * mode as f64)).round() as usize;
    let (i0, j0) = (nx / 2, ny / 2);
    let mut series = Vec::with_capacity(steps + 1);
    series.push(g.ex.get(i0, j0, k_probe));

    for _ in 0..steps {
        g.update_h_field();
        g.update_e_field();
        g.apply_boundary(BoundaryCondition::PerfectConductor);
        project_xy_uniform(&mut g, i0, j0);
        series.push(g.ex.get(i0, j0, k_probe));
    }

    let omega = fit_omega(&series, dt);
    (omega, yee_omega([0.0, 0.0, kz], dx, dt, c), c * kz)
}

/// Run a 2-D TM₁₁₀ PEC-cavity mode and return `(ω_measured, ω_yee, ω_continuum)`.
fn cavity_tm110(n: usize, courant: f64, steps: usize) -> (f64, f64, f64) {
    let dx = 1e-3;
    let nz = 4;
    let c_nominal = 299_792_458.0;
    let dt = courant * dx / (c_nominal * 3_f64.sqrt());

    let mut g = YeeGrid::new(n, n, nz, dx, dt);
    let c = g.c0;
    let l = (n - 1) as f64 * dx;
    let kx = std::f64::consts::PI / l;
    let ky = std::f64::consts::PI / l;

    for i in 0..n {
        for j in 0..n {
            let v = (kx * i as f64 * dx).sin() * (ky * j as f64 * dx).sin();
            for k in 0..nz {
                g.ez.set(i, j, k, v);
            }
        }
    }

    let (i0, j0, k0) = (n / 2, n / 2, 1);
    let mut series = Vec::with_capacity(steps + 1);
    series.push(g.ez.get(i0, j0, k0));

    for _ in 0..steps {
        g.update_h_field();
        g.update_e_field();
        g.apply_boundary(BoundaryCondition::PerfectConductor);
        project_z_uniform(&mut g, k0);
        series.push(g.ez.get(i0, j0, k0));
    }

    let omega = fit_omega(&series, dt);
    (
        omega,
        yee_omega([kx, ky, 0.0], dx, dt, c),
        c * (kx * kx + ky * ky).sqrt(),
    )
}

/// Propagate a Gaussian pulse in a 1-D reduced grid and record `Ex` at `k_probe`.
///
/// `pml` gives `(thickness, order, sigma_max)` applied to the `+z` end only, via
/// the crate's own [`PmlLayer`] grading. `None` runs a plain (reference) grid.
fn pulse_run(
    nz: usize,
    k0: f64,
    width: f64,
    k_probe: usize,
    steps: usize,
    pml: Option<(usize, usize, f64)>,
) -> Vec<f64> {
    let dx = 1e-3;
    let (nx, ny) = (3, 3);
    let c_nominal = 299_792_458.0;
    let courant = 0.5; // 1-D stability limit is cΔt/Δx ≤ 1
    let dt = courant * dx / c_nominal;

    let mut g = YeeGrid::new(nx, ny, nz, dx, dt);
    let eta = (g.mu0 / g.eps0).sqrt();
    let s = g.c0 * dt / dx;

    // Right-travelling Gaussian: Hy is offset by half a cell in space and half a
    // step in time so the pair is a (nearly) pure +z mode.
    for k in 0..nz {
        let zk = k as f64;
        let e = (-((zk - k0) / width).powi(2)).exp();
        let h = (-((zk + 0.5 - 0.5 * s - k0) / width).powi(2)).exp() / eta;
        for i in 0..nx {
            for j in 0..ny {
                g.ex.set(i, j, k, e);
                g.hy.set(i, j, k, h);
            }
        }
    }

    if let Some((thickness, order, sigma_max)) = pml {
        let layer = PmlLayer::new(thickness, order, sigma_max);
        for k in 0..nz {
            let d = nz - 1 - k;
            if d < thickness {
                let sigma = layer.get_sigma(d);
                for i in 0..nx {
                    for j in 0..ny {
                        g.sigma.set(i, j, k, sigma);
                    }
                }
            }
        }
    }

    let (i0, j0) = (nx / 2, ny / 2);
    let mut probe = Vec::with_capacity(steps + 1);
    probe.push(g.ex.get(i0, j0, k_probe));
    for _ in 0..steps {
        g.update_h_field();
        g.update_e_field();
        project_xy_uniform(&mut g, i0, j0);
        probe.push(g.ex.get(i0, j0, k_probe));
    }
    probe
}

/// Measure the reflection coefficient of the `+z` absorbing layer, in dB.
fn measure_reflection(sigma_max: f64, thickness: usize, order: usize) -> f64 {
    let nz_test = 400;
    let nz_ref = 2000; // long enough that nothing returns to the probe
    let k0 = 40.0; // pulse launch site
    let width = 12.0;
    let k_probe = 120; // downstream of the source, upstream of the layer
    // 0.5 cell/step; the pulse must reach the layer and any reflection return.
    let steps =
        2 * ((nz_test - thickness) as f64 - k0 + (nz_test as f64 - k_probe as f64)) as usize + 100;

    let reference = pulse_run(nz_ref, k0, width, k_probe, steps, None);
    let test = pulse_run(
        nz_test,
        k0,
        width,
        k_probe,
        steps,
        Some((thickness, order, sigma_max)),
    );

    let incident = reference.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    // Only look after the incident pulse has cleared the probe.
    let start = (2.0 * (k_probe as f64 - k0) + 8.0 * width) as usize;
    let reflected = reference
        .iter()
        .zip(test.iter())
        .skip(start.min(steps))
        .fold(0.0_f64, |a, (&r, &t)| a.max((t - r).abs()));

    if incident <= 0.0 || reflected <= 0.0 {
        return f64::NEG_INFINITY;
    }
    20.0 * (reflected / incident).log10()
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
             dt/(μΔx) and dt/(εΔx).",
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

    // Deviation from the continuum must vanish as Δx² at fixed Courant number
    // and fixed physical wavelength.
    let mut samples = Vec::new();
    let mut finest = 0.0;
    for &n in &[17_usize, 33, 65, 129] {
        let (m, _, cont) = cavity_1d(n, 1, 0.5, 400);
        let h = 1.0 / (n - 1) as f64; // Δx / L
        let err = (m - cont).abs() / cont;
        samples.push((h, err));
        finest = err;
    }
    let conv = Convergence::fit("Δx/L", samples, 2.0, 0.15);
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
            1e-4,
        )
        .with_convergence(conv),
    );

    // ---- 2. Rectangular cavity resonance ------------------------------------
    let (w_meas, w_yee, w_cont) = cavity_tm110(41, 0.5, 600);
    let f_meas = w_meas / (2.0 * std::f64::consts::PI);
    let f_cont = w_cont / (2.0 * std::f64::consts::PI);
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
            f_meas,
            f_cont,
            ErrorKind::Relative,
            2e-3,
        )
        .note("Residual error is the Yee grid-dispersion error at kΔx = π/40."),
    );

    let mut samples = Vec::new();
    let mut finest = 0.0;
    for &n in &[11_usize, 21, 41, 81] {
        let (m, _, cont) = cavity_tm110(n, 0.5, 600);
        let h = 1.0 / (n - 1) as f64;
        let err = (m - cont).abs() / cont;
        samples.push((h, err));
        finest = err;
    }
    let conv = Convergence::fit("Δx/L", samples, 2.0, 0.15);
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
            1e-3,
        )
        .with_convergence(conv),
    );

    // ---- 3. Absorbing-boundary reflection -----------------------------------
    // Sweep σ_max and report the *best* reflection the layer can achieve.
    let thickness = 16;
    let order = 3;
    let mut best = (f64::INFINITY, 0.0);
    let mut sweep_notes = Vec::new();
    for e in -2..=6 {
        let sigma_max = 10_f64.powi(e);
        let r_db = measure_reflection(sigma_max, thickness, order);
        sweep_notes.push(format!("σ_max = 1e{e:<2} S/m → R = {r_db:7.2} dB"));
        if r_db < best.0 {
            best = (r_db, sigma_max);
        }
    }

    let mut v = Validation::new(
        "em.absorbing_boundary_reflection",
        "Reflection coefficient of the `PmlLayer` absorbing boundary",
        CRATE,
        "Berenger (1994); Taflove & Hagness ch. 7 — a correctly implemented \
         10–16 cell CPML reaches R < −60 dB (typically −80 dB) for a normally \
         incident broadband pulse",
        "reflection coefficient R (dB), 16-cell layer, best σ_max over 1e−2…1e6 S/m",
        best.0,
        -60.0,
        ErrorKind::Absolute,
        0.0,
    );
    // A *smaller* (more negative) R than the target is a pass, so evaluate the
    // one-sided criterion explicitly rather than through |measured − expected|.
    v.error = (best.0 - (-60.0)).max(0.0);
    v.status = if v.error <= 0.0 {
        crate::report::Status::Pass
    } else {
        crate::report::Status::Fail
    };
    suite.push(
        v.note(format!("best σ_max = {:.3e} S/m", best.1))
            .note(format!("σ_max sweep: {}", sweep_notes.join("; ")))
            .note(
                "`phyz_em::PmlLayer` (crates/phyz-em/src/boundary.rs:28-67) builds only a \
                 graded *electric* conductivity profile σ(d). `update_h_field` \
                 (crates/phyz-em/src/fdtd.rs:15-50) has no magnetic-loss term, so the \
                 matching condition σ*/μ = σ/ε is never imposed: the layer is a lossy \
                 dielectric slab, not a perfectly matched layer. Its wave impedance \
                 η = √(μ/(ε + iσ/ω)) differs from the vacuum impedance at every \
                 frequency, and the impedance jump at the layer's front face reflects \
                 regardless of how the profile is graded.",
            )
            .note(
                "`apply_pml_boundary` (boundary.rs:163-205) additionally *sums* the σ \
                 contributions of all six faces, so corner cells receive up to 6σ_max, \
                 and thickness is derived from `nx` alone (`8.min(self.nx / 4)`) even for \
                 anisotropic grids. This benchmark bypasses that routine and applies the \
                 graded profile to the +z face only, which is the most favourable case \
                 for the layer.",
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
        let s: Vec<f64> = (0..500).map(|n| 0.7 * (w * n as f64 * dt + 0.3).cos()).collect();
        assert!((fit_omega(&s, dt) - w).abs() / w < 1e-10);
    }
}
