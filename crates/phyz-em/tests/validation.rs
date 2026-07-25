//! Quantitative validation of the FDTD solver against analytic results.
//!
//! Scope: this file validates the *electromagnetics* — boundary reflection,
//! material dispersion, plane-wave injection — by measuring numbers and
//! comparing them to closed-form theory. It deliberately does not restate
//! generic numerical-analysis checks (convergence order, energy drift for
//! generic integrators); those belong to the workspace-wide analytic
//! validation suite. Anything here is specific to Maxwell's equations on a Yee
//! grid and would not be meaningful outside `phyz-em`.
//!
//! Each test prints the measured quantity so a run doubles as a report.

use phyz_em::analysis::{db, gaussian_pulse, reflection_db, ricker_pulse, spectrum};
use phyz_em::boundary::BoundaryCondition;
use phyz_em::cpml::CpmlConfig;
use phyz_em::dispersion::DispersiveMaterial;
use phyz_em::grid::YeeGrid;
use phyz_em::tfsf::Tfsf;

const C: f64 = 299_792_458.0;

/// Compare a measured reflection magnitude to theory.
///
/// A pure relative tolerance is the wrong metric where `|r|` is near zero — a
/// 0.001 absolute discrepancy on `|r| = 0.02` is 7% relative but physically
/// negligible — so an absolute floor is applied alongside it.
fn assert_reflection_close(measured: f64, analytic: f64, rel_tol: f64, context: &str) {
    let allowed = (rel_tol * analytic).max(0.005);
    assert!(
        (measured - analytic).abs() <= allowed,
        "{context}: |r| = {measured:.4} vs analytic {analytic:.4} \
         (off by {:.4}, allowed {allowed:.4})",
        (measured - analytic).abs()
    );
}

/// A transversely-uniform (effectively 1D) z-directed waveguide.
///
/// `nx = ny = 1` with periodic transverse boundaries makes the 3D Yee update
/// reduce exactly to the 1D scheme, which is what lets these tests reach
/// reflection floors far below what a 3D run with transverse discretization
/// error would allow.
fn line_grid(nz: usize, dz: f64, dt: f64) -> YeeGrid {
    let mut grid = YeeGrid::new_rect(1, 1, nz, dz, dz, dz, dt);
    grid.set_periodic([true, true, false]);
    grid
}

/// Run a soft-source pulse on a z-line and record `Ex` at `k_probe`.
#[allow(clippy::too_many_arguments)]
fn run_soft_source(
    nz: usize,
    dz: f64,
    dt: f64,
    boundary: Option<BoundaryCondition>,
    k_src: usize,
    k_probe: usize,
    n_steps: usize,
    spread: f64,
    t0: f64,
) -> Vec<f64> {
    let mut grid = line_grid(nz, dz, dt);
    if let Some(bc) = boundary {
        grid.set_boundary(bc);
    }
    let mut rec = Vec::with_capacity(n_steps);
    for n in 0..n_steps {
        grid.update_h_field();
        grid.update_e_field();
        let t = n as f64 * dt;
        grid.ex.add(0, 0, k_src, ricker_pulse(t, t0, spread));
        rec.push(grid.ex.get(0, 0, k_probe));
    }
    rec
}

/// **Item 1 — the headline measurement.**
///
/// Reflection off the absorbing boundary, measured the only way that means
/// anything: run the same excitation in a domain large enough that no
/// boundary reflection can reach the probe within the record window, subtract
/// that reference from the short-domain record, and take the spectral ratio.
///
/// A graded-conductivity absorber (the previous `Pml` variant) is not
/// impedance matched and reflects at the −10 dB level. A genuine CPML should
/// be below −60 dB. Both are measured here, and the test asserts on the
/// difference as well as the absolute CPML figure.
#[test]
fn boundary_reflection_coefficient_in_db() {
    let dz = 1e-9;
    let dt = 0.5 * dz / C; // Courant number 0.5

    // Geometry. The short domain puts the probe 90 cells from the inner
    // surface of the low-z layer; the reference reproduces the same
    // source-probe separation with ~800 extra cells of run-off at both ends.
    let nz_short = 400;
    let k_src_short = 200;
    let k_probe_short = 100;

    let nz_ref = 1600;
    let k_src_ref = 1000;
    let k_probe_ref = 900;

    // Record long enough for the low-z reflection to return (≈560 steps) but
    // not long enough for the high-z one (≈960 steps), so the number below is
    // unambiguously the low-z boundary's reflection.
    let n_steps = 800;

    // Ricker pulse centred near 20 cells per wavelength.
    let spread = 12.0 * dt;
    let t0 = 5.0 * spread;

    let reference = run_soft_source(
        nz_ref,
        dz,
        dt,
        None,
        k_src_ref,
        k_probe_ref,
        n_steps,
        spread,
        t0,
    );

    let measure = |bc: BoundaryCondition, label: &str| -> f64 {
        let total = run_soft_source(
            nz_short,
            dz,
            dt,
            Some(bc),
            k_src_short,
            k_probe_short,
            n_steps,
            spread,
            t0,
        );
        let reflected: Vec<f64> = total
            .iter()
            .zip(reference.iter())
            .map(|(a, b)| a - b)
            .collect();

        // Evaluate across the band where the pulse actually carries energy:
        // 10 to 40 cells per wavelength.
        let ppw: Vec<f64> = (10..=40).map(|p| p as f64).collect();
        let omegas: Vec<f64> = ppw
            .iter()
            .map(|p| 2.0 * std::f64::consts::PI * C / (p * dz))
            .collect();
        let inc_mag: Vec<f64> = spectrum(&reference, &omegas, dt)
            .iter()
            .map(|c| c.norm())
            .collect();
        let r_db = reflection_db(&reflected, &reference, &omegas, dt);

        // Report, and take the worst reflection over the well-excited part of
        // the band (where the incident spectrum is within 20 dB of its peak).
        let peak_inc = inc_mag.iter().cloned().fold(0.0_f64, f64::max);
        let mut worst = f64::NEG_INFINITY;
        println!("\n--- {label} ---");
        for (idx, p) in ppw.iter().enumerate() {
            if inc_mag[idx] < peak_inc * 0.1 {
                continue;
            }
            println!("  {p:>4.0} cells/λ : {:>7.1} dB", r_db[idx]);
            worst = worst.max(r_db[idx]);
        }
        println!("  worst in band: {worst:.1} dB");
        worst
    };

    let cpml = measure(
        BoundaryCondition::Cpml(CpmlConfig::with_thickness(10).on_axes([false, false, true])),
        "CPML, 10 cells",
    );
    // σ_max = 3e6 S/m is near this absorber's optimum for a 10-cell layer on
    // a 1 nm grid (swept: it degrades in both directions), so the comparison
    // below is against the cheap absorber at its best, not a strawman.
    let lossy = measure(
        BoundaryCondition::LossyAbsorber {
            thickness: 10,
            order: 2,
            sigma_max: 3.0e6,
        },
        "graded-conductivity absorber, 10 cells, near-optimal σ (not a PML)",
    );

    println!(
        "\nCPML improves on the cheap absorber by {:.1} dB\n",
        lossy - cpml
    );

    assert!(
        cpml < -60.0,
        "CPML reflection {cpml:.1} dB is worse than the -60 dB target"
    );
    // The unmatched layer cannot do better than roughly -10 dB no matter how
    // it is tuned; if it ever measures much better, the test is not actually
    // exercising it.
    assert!(
        lossy > -20.0,
        "the cheap absorber measured {lossy:.1} dB — suspiciously good; is it \
         actually running the unmatched layer?"
    );
    assert!(
        lossy - cpml > 45.0,
        "CPML should beat the unmatched absorber by tens of dB, got {:.1} dB",
        lossy - cpml
    );
}

/// A thicker CPML must absorb better. Monotone improvement with thickness is
/// the signature of a matched layer; an unmatched absorber saturates.
#[test]
fn cpml_reflection_improves_with_thickness() {
    let dz = 1e-9;
    let dt = 0.5 * dz / C;
    let n_steps = 800;
    let spread = 12.0 * dt;
    let t0 = 5.0 * spread;

    let reference = run_soft_source(1600, dz, dt, None, 1000, 900, n_steps, spread, t0);
    let omegas: Vec<f64> = (12..=30)
        .map(|p| 2.0 * std::f64::consts::PI * C / (p as f64 * dz))
        .collect();

    let worst_for = |thickness: usize| -> f64 {
        let total = run_soft_source(
            400,
            dz,
            dt,
            Some(BoundaryCondition::Cpml(
                CpmlConfig::with_thickness(thickness).on_axes([false, false, true]),
            )),
            200,
            100,
            n_steps,
            spread,
            t0,
        );
        let refl: Vec<f64> = total.iter().zip(&reference).map(|(a, b)| a - b).collect();
        reflection_db(&refl, &reference, &omegas, dt)
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max)
    };

    let r6 = worst_for(6);
    let r10 = worst_for(10);
    let r16 = worst_for(16);
    println!("CPML reflection: 6 cells {r6:.1} dB, 10 cells {r10:.1} dB, 16 cells {r16:.1} dB");

    assert!(
        r10 < r6,
        "10-cell CPML ({r10:.1} dB) should beat 6-cell ({r6:.1} dB)"
    );
    assert!(
        r16 < r10,
        "16-cell CPML ({r16:.1} dB) should beat 10-cell ({r10:.1} dB)"
    );
}

/// Shared driver for half-space reflection: illuminate a half-space through a
/// one-way TFSF injector and read the reflected wave out of the
/// scattered-field region.
///
/// Returns `(omegas, measured |r|)`.
#[allow(clippy::too_many_arguments)]
fn measure_halfspace_reflection(
    material: DispersiveMaterial,
    dz: f64,
    dt: f64,
    nz: usize,
    k_interface: usize,
    n_steps: usize,
    spread: f64,
    omegas: &[f64],
) -> Vec<f64> {
    let mut grid = line_grid(nz, dz, dt);
    grid.set_boundary(BoundaryCondition::Cpml(
        CpmlConfig::with_thickness(12).on_axes([false, false, true]),
    ));

    let id = grid.add_material(material);
    grid.assign_material_halfspace_z(id, k_interface);
    grid.build_dispersion();

    // Inject at k = 30; everything below is pure scattered field.
    let k_inject = 30;
    let k_probe = 20;
    let mut tfsf = Tfsf::injector_z(&grid, k_inject);

    let t0 = 5.0 * spread;
    let src = move |t: f64| gaussian_pulse(t, t0, spread);

    let mut reflected = Vec::with_capacity(n_steps);
    let mut incident = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        tfsf.step(&mut grid, &src);
        reflected.push(grid.ex.get(0, 0, k_probe));
        // The incident field at the *same* plane, so propagation phase between
        // probe and interface cancels in the magnitude ratio.
        incident.push(tfsf.ex_inc(k_probe));
    }

    let r = spectrum(&reflected, omegas, dt);
    let i = spectrum(&incident, omegas, dt);
    r.iter()
        .zip(i.iter())
        .map(|(a, b)| a.norm() / b.norm())
        .collect()
}

/// **Item 2, part 1** — reflection off a non-dispersive dielectric half-space
/// must match the Fresnel coefficient `r = (1 − n)/(1 + n)`.
///
/// This is the baseline: it exercises the TFSF injector, the material
/// assignment path and the spectral analysis without any ADE machinery, so a
/// failure in the dispersive tests below can be localised.
#[test]
fn halfspace_reflection_matches_fresnel_nondispersive() {
    let dz = 1e-9;
    let dt = 0.5 * dz / C;
    let nz = 500;
    let spread = 20.0 * dt;

    for &eps in &[2.25_f64, 4.0, 9.0] {
        // The interface error is O((kΔ)²) in the *dense* medium, so the
        // sampling band has to scale with the index: 20 cells per wavelength
        // inside the material, whatever that costs in vacuum cells.
        let p_min = (20.0 * eps.sqrt()).ceil() as usize;
        let ppw: Vec<f64> = (p_min..=p_min + 30).step_by(6).map(|p| p as f64).collect();
        let omegas: Vec<f64> = ppw
            .iter()
            .map(|p| 2.0 * std::f64::consts::PI * C / (p * dz))
            .collect();

        let mat = DispersiveMaterial::non_dispersive(eps);
        let analytic = (1.0 - eps.sqrt()).abs() / (1.0 + eps.sqrt());
        let measured = measure_halfspace_reflection(mat, dz, dt, nz, 250, 1600, spread, &omegas);

        println!("\nε_r = {eps}  (analytic |r| = {analytic:.4})");
        let mut errors = Vec::new();
        for (idx, p) in ppw.iter().enumerate() {
            let err = (measured[idx] - analytic).abs() / analytic;
            errors.push(err);
            println!(
                "  {p:>4.0} cells/λ : |r| = {:.4}   error {:.2}%",
                measured[idx],
                err * 100.0
            );
            assert_reflection_close(
                measured[idx],
                analytic,
                0.025,
                &format!("ε_r = {eps} at {p} cells/λ"),
            );
        }

        // What is left is discretization error, not a modelling error: it must
        // shrink as the grid resolves the wave better.
        assert!(
            errors[errors.len() - 1] < errors[0],
            "ε_r = {eps}: error does not decrease with resolution \
             ({:.3}% at {} cells/λ vs {:.3}% at {} cells/λ) — that points at a \
             bug, not discretization",
            errors[0] * 100.0,
            ppw[0],
            errors[errors.len() - 1] * 100.0,
            ppw[ppw.len() - 1]
        );
    }
}

/// **Item 2, part 2** — a Drude metal.
///
/// Below the plasma frequency ε is negative, the wave is evanescent and the
/// half-space is a near-perfect mirror; above it the metal turns transparent.
/// Reproducing that crossover, and the analytic `|r|` on both sides of it, is
/// the real test of the ADE implementation.
#[test]
fn halfspace_reflection_matches_drude_metal() {
    // Choose the grid so that the *pole* is well resolved: ω_p Δt ≈ 0.05.
    let dz = 15e-9;
    let dt = 0.5 * dz / C;
    let omega_p = 2.0e15;
    let gamma = 1.0e14;
    assert!(
        omega_p * dt < 0.1,
        "plasma pole under-resolved: ω_p Δt = {}",
        omega_p * dt
    );

    let material = DispersiveMaterial::drude(1.0, omega_p, gamma);

    // Sample across the plasma frequency: ε goes from −3 to +0.75.
    let omegas: Vec<f64> = vec![0.6, 0.8, 1.0, 1.4, 1.8, 2.4]
        .into_iter()
        .map(|f: f64| f * omega_p)
        .collect();

    // A Gaussian of width `spread` has spectrum exp(-(ω·spread/2)²), so the
    // band it usefully covers is ω ≲ 4/spread. Sampling out to 2.4 ω_p
    // therefore needs spread ≈ 1.2/ω_p — the earlier 6/ω_p left the top of
    // the band with no incident energy at all, making the ratio pure noise.
    let spread = 1.2 / omega_p;
    let nz = 600;
    let measured =
        measure_halfspace_reflection(material.clone(), dz, dt, nz, 300, 3000, spread, &omegas);

    println!("\nDrude metal: ω_p = {omega_p:.2e} rad/s, γ = {gamma:.2e} rad/s");
    for (idx, &w) in omegas.iter().enumerate() {
        let analytic = material.fresnel_normal(w).norm();
        let eps = material.permittivity(w);
        let err = (measured[idx] - analytic).abs() / analytic;
        println!(
            "  ω/ω_p = {:.2}  ε = {:+.3}{:+.3}i   |r| measured {:.4} vs analytic {:.4}  ({:.2}%)",
            w / omega_p,
            eps.re,
            eps.im,
            measured[idx],
            analytic,
            err * 100.0
        );
        assert_reflection_close(
            measured[idx],
            analytic,
            0.03,
            &format!("Drude at ω/ω_p = {:.2}", w / omega_p),
        );
    }

    // The physics, stated as an assertion: below ω_p it is a mirror, above it
    // is not.
    let below = material.fresnel_normal(0.6 * omega_p).norm();
    let above = material.fresnel_normal(2.4 * omega_p).norm();
    assert!(
        below > 0.8 && above < 0.3,
        "Drude crossover is not being reproduced"
    );
}

/// **Item 2, part 3** — a Lorentz oscillator.
///
/// Reflection must track the resonant lineshape: low on the low-frequency
/// side, peaking near ω₀ where ε is large, then falling through the
/// reststrahlen region.
#[test]
fn halfspace_reflection_matches_lorentz_oscillator() {
    let dz = 15e-9;
    let dt = 0.5 * dz / C;
    let omega0 = 1.5e15;
    let gamma = 2.0e14;
    let delta_eps = 3.0;
    assert!(
        omega0 * dt < 0.1,
        "resonance under-resolved: ω₀ Δt = {}",
        omega0 * dt
    );

    let material = DispersiveMaterial::lorentz(1.5, delta_eps, omega0, gamma);

    let omegas: Vec<f64> = vec![0.4, 0.7, 1.0, 1.3, 1.8, 2.5]
        .into_iter()
        .map(|f: f64| f * omega0)
        .collect();

    // See the Drude test: spread ≈ 1.2/ω₀ keeps the whole sampled band excited.
    let spread = 1.2 / omega0;
    let measured =
        measure_halfspace_reflection(material.clone(), dz, dt, 600, 300, 3000, spread, &omegas);

    println!("\nLorentz: ω₀ = {omega0:.2e} rad/s, γ = {gamma:.2e}, Δε = {delta_eps}");
    for (idx, &w) in omegas.iter().enumerate() {
        let analytic = material.fresnel_normal(w).norm();
        let eps = material.permittivity(w);
        let err = (measured[idx] - analytic).abs() / analytic;
        println!(
            "  ω/ω₀ = {:.2}  ε = {:+.3}{:+.3}i   |r| measured {:.4} vs analytic {:.4}  ({:.2}%)",
            w / omega0,
            eps.re,
            eps.im,
            measured[idx],
            analytic,
            err * 100.0
        );
        assert_reflection_close(
            measured[idx],
            analytic,
            0.03,
            &format!("Lorentz at ω/ω₀ = {:.2}", w / omega0),
        );
    }
}

/// A Debye relaxor, for completeness — the first-order pole path.
#[test]
fn halfspace_reflection_matches_debye_relaxor() {
    let dz = 15e-9;
    let dt = 0.5 * dz / C;
    // τ must be short enough that the sampled wavelengths (λ = 2πc·τ/ωτ) fit
    // in the domain: at ωτ = 0.3, λ ≈ 3 µm ≈ 200 cells, which they do.
    let tau = 5.0e-16;
    assert!(
        dt / tau < 0.1,
        "relaxation under-resolved: Δt/τ = {}",
        dt / tau
    );

    let material = DispersiveMaterial::debye(2.0, 4.0, tau);
    let omegas: Vec<f64> = vec![0.3, 0.6, 1.0, 1.6, 2.5]
        .into_iter()
        .map(|f: f64| f / tau)
        .collect();

    let spread = 1.2 * tau;
    let measured =
        measure_halfspace_reflection(material.clone(), dz, dt, 600, 300, 3000, spread, &omegas);

    println!("\nDebye: τ = {tau:.2e} s");
    for (idx, &w) in omegas.iter().enumerate() {
        let analytic = material.fresnel_normal(w).norm();
        let err = (measured[idx] - analytic).abs() / analytic;
        println!(
            "  ωτ = {:.2}   |r| measured {:.4} vs analytic {:.4}  ({:.2}%)",
            w * tau,
            measured[idx],
            analytic,
            err * 100.0
        );
        assert_reflection_close(
            measured[idx],
            analytic,
            0.04,
            &format!("Debye at ωτ = {:.2}", w * tau),
        );
    }
}

/// Energy conservation at a lossless interface: `|r|² + |t|²·(n₂/n₁) = 1`.
///
/// An independent check on the half-space machinery that does not reuse the
/// Fresnel formula — it only assumes energy is conserved.
#[test]
fn lossless_interface_conserves_power() {
    let dz = 1e-9;
    let dt = 0.5 * dz / C;
    let eps: f64 = 4.0;
    let n = eps.sqrt();

    let mut grid = line_grid(500, dz, dt);
    grid.set_boundary(BoundaryCondition::Cpml(
        CpmlConfig::with_thickness(12).on_axes([false, false, true]),
    ));
    let id = grid.add_material(DispersiveMaterial::non_dispersive(eps));
    grid.assign_material_halfspace_z(id, 250);
    grid.build_dispersion();

    let mut tfsf = Tfsf::injector_z(&grid, 30);
    let spread = 20.0 * dt;
    let t0 = 5.0 * spread;
    let src = move |t: f64| gaussian_pulse(t, t0, spread);

    let (k_refl, k_trans) = (20, 300);
    let mut reflected = Vec::new();
    let mut transmitted = Vec::new();
    let mut incident = Vec::new();
    for _ in 0..2000 {
        tfsf.step(&mut grid, &src);
        reflected.push(grid.ex.get(0, 0, k_refl));
        transmitted.push(grid.ex.get(0, 0, k_trans));
        incident.push(tfsf.ex_inc(k_refl));
    }

    let ppw: Vec<f64> = (30..=60).step_by(10).map(|p| p as f64).collect();
    let omegas: Vec<f64> = ppw
        .iter()
        .map(|p| 2.0 * std::f64::consts::PI * C / (p * dz))
        .collect();
    let sr = spectrum(&reflected, &omegas, dt);
    let st = spectrum(&transmitted, &omegas, dt);
    let si = spectrum(&incident, &omegas, dt);

    println!("\nPower balance at an ε_r = {eps} interface:");
    for (idx, p) in ppw.iter().enumerate() {
        let r = sr[idx].norm() / si[idx].norm();
        let t = st[idx].norm() / si[idx].norm();
        // Time-averaged power ∝ n |E|² for a non-magnetic medium.
        let total = r * r + n * t * t;
        println!(
            "  {p:>4.0} cells/λ : |r|² = {:.4}, n|t|² = {:.4}, sum = {:.4}",
            r * r,
            n * t * t,
            total
        );
        assert!(
            (total - 1.0).abs() < 0.03,
            "power balance at {p} cells/λ is {total:.4}, not 1"
        );
    }
}

/// **Item 3** — TFSF injection quality across the spectrum.
///
/// The single-number peak-leakage check lives in the `tfsf` unit tests; this
/// one resolves leakage per frequency, which is what determines the smallest
/// scattering cross-section the injector can resolve.
#[test]
fn tfsf_leakage_spectrum() {
    let dz = 1e-9;
    let dt = 0.5 * dz / C;
    let nz = 300;

    let mut grid = line_grid(nz, dz, dt);
    grid.set_boundary(BoundaryCondition::Cpml(
        CpmlConfig::with_thickness(12).on_axes([false, false, true]),
    ));

    let mut tfsf = Tfsf::slab_z(&grid, 60, 240);
    let spread = 12.0 * dt;
    let t0 = 5.0 * spread;
    let src = move |t: f64| ricker_pulse(t, t0, spread);

    let mut sf = Vec::new();
    let mut tf = Vec::new();
    for _ in 0..1200 {
        tfsf.step(&mut grid, &src);
        sf.push(grid.ex.get(0, 0, 40));
        tf.push(grid.ex.get(0, 0, 150));
    }

    let ppw: Vec<f64> = (10..=40).step_by(5).map(|p| p as f64).collect();
    let omegas: Vec<f64> = ppw
        .iter()
        .map(|p| 2.0 * std::f64::consts::PI * C / (p * dz))
        .collect();
    let leak = reflection_db(&sf, &tf, &omegas, dt);

    println!("\nTFSF leakage into the scattered-field region:");
    let mut worst = f64::NEG_INFINITY;
    for (idx, p) in ppw.iter().enumerate() {
        println!("  {p:>4.0} cells/λ : {:>7.1} dB", leak[idx]);
        worst = worst.max(leak[idx]);
    }
    println!("  worst: {worst:.1} dB");
    assert!(
        worst < -60.0,
        "TFSF leakage {worst:.1} dB exceeds the -60 dB target"
    );
}

/// The scattered field from an empty TFSF box in 3D is the leakage floor for
/// any scattering computation. Measured with all six faces closed, which is
/// the geometry a Mie calculation would use.
#[test]
fn tfsf_box_leakage_in_three_dimensions() {
    let d = 4e-9;
    let dt = d / (C * 3_f64.sqrt()) * 0.99;
    let n = 44;

    let mut grid = YeeGrid::new(n, n, n, d, dt);
    grid.set_boundary(BoundaryCondition::Cpml(CpmlConfig::with_thickness(10)));

    let mut tfsf = Tfsf::box_region(&grid, 14, 29, 14, 29, 14, 29);
    let spread = 12.0 * dt;
    let t0 = 5.0 * spread;
    let src = move |t: f64| ricker_pulse(t, t0, spread);

    let mut peak_inside: f64 = 0.0;
    let mut peak_outside: f64 = 0.0;
    for _ in 0..500 {
        tfsf.step(&mut grid, &src);
        peak_inside = peak_inside.max(grid.ex.get(22, 22, 22).abs());
        // Sample the scattered-field region on each side of the box.
        for &(i, j, k) in &[(22, 22, 6), (22, 22, 37), (6, 22, 22), (22, 6, 22)] {
            peak_outside = peak_outside.max(grid.ex.get(i, j, k).abs());
        }
    }

    assert!(peak_inside > 1e-3, "no incident wave inside the box");
    let leak = db(peak_outside / peak_inside);
    println!("\n3D TFSF box leakage: {leak:.1} dB");
    // With the half-cell index ranges right, the closed box leaks at round-off:
    // there is no approximation left, only floating-point cancellation.
    assert!(leak < -100.0, "3D TFSF box leakage {leak:.1} dB");
}

/// **Item 3, quantitative** — a finite dielectric slab.
///
/// A slab has an exact analytic reflection (the Airy / Fabry–Pérot formula),
/// including the interference fringes from multiple internal round trips:
///
/// ```text
///   r = r₀₁ (1 − e^{2iβ}) / (1 − r₀₁² e^{2iβ}),   β = n k₀ d
/// ```
///
/// Reproducing the *fringe positions* as well as the magnitudes tests far more
/// than a half-space does: it requires the injected wave, both interfaces, and
/// the in-medium phase velocity all to be right at once. A half-space test
/// passes even if the optical thickness is systematically wrong.
#[test]
fn slab_reflection_matches_fabry_perot() {
    use phyz_em::dispersion::C64;

    let dz = 1e-9;
    let dt = 0.5 * dz / C;
    let eps: f64 = 4.0;
    let n = eps.sqrt();

    let k_front = 250;
    let cells = 40;
    let thickness = cells as f64 * dz;

    let mut grid = line_grid(500, dz, dt);
    grid.set_boundary(BoundaryCondition::Cpml(
        CpmlConfig::with_thickness(12).on_axes([false, false, true]),
    ));
    let id = grid.add_material(DispersiveMaterial::non_dispersive(eps));
    grid.assign_material_box(id, (0, 1), (0, 1), (k_front, k_front + cells));
    grid.build_dispersion();

    let mut tfsf = Tfsf::injector_z(&grid, 30);
    let spread = 16.0 * dt;
    let t0 = 5.0 * spread;
    let src = move |t: f64| gaussian_pulse(t, t0, spread);

    let k_probe = 20;
    let mut reflected = Vec::new();
    let mut incident = Vec::new();
    // Long enough for many internal round trips to leak back out, which is
    // what builds the fringe structure.
    for _ in 0..4000 {
        tfsf.step(&mut grid, &src);
        reflected.push(grid.ex.get(0, 0, k_probe));
        incident.push(tfsf.ex_inc(k_probe));
    }

    // Sample across a full fringe period. One round trip is 2 n d, so the free
    // spectral range in vacuum wavelength around λ = 4 n d / m is broad here.
    // From 50 cells/λ in vacuum — 25 inside the ε = 4 slab — up to 140.
    let ppw: Vec<f64> = (50..=140).step_by(10).map(|p| p as f64).collect();
    let omegas: Vec<f64> = ppw
        .iter()
        .map(|p| 2.0 * std::f64::consts::PI * C / (p * dz))
        .collect();
    let sr = spectrum(&reflected, &omegas, dt);
    let si = spectrum(&incident, &omegas, dt);

    println!("\nFabry-Perot slab: ε_r = {eps}, {cells} cells thick");
    let mut seen_peak = false;
    let mut seen_null = false;
    for (idx, p) in ppw.iter().enumerate() {
        let k0 = 2.0 * std::f64::consts::PI / (p * dz);
        let beta = n * k0 * thickness;
        let r01 = (1.0 - n) / (1.0 + n);
        let e2 = C64::new((2.0 * beta).cos(), (2.0 * beta).sin());
        let one = C64::new(1.0, 0.0);
        let num = C64::new(r01, 0.0) * (one - e2);
        let den = one - C64::new(r01 * r01, 0.0) * e2;
        let analytic = (num / den).norm();

        let measured = sr[idx].norm() / si[idx].norm();
        println!(
            "  {p:>4.0} cells/λ : |r| measured {:.4} vs analytic {:.4}",
            measured, analytic
        );
        if analytic > 0.5 {
            seen_peak = true;
        }
        if analytic < 0.05 {
            seen_null = true;
        }
        // Near a fringe null the relative metric is meaningless (the analytic
        // value is exactly zero), so the constraint there is absolute: 0.02 on
        // a curve that spans 0 to 0.6 still pins the fringe positions tightly.
        let allowed = (0.05 * analytic).max(0.02);
        assert!(
            (measured - analytic).abs() <= allowed,
            "slab at {p} cells/λ: |r| = {measured:.4} vs analytic {analytic:.4}              (off by {:.4}, allowed {allowed:.4})",
            (measured - analytic).abs()
        );
    }

    // Confirm the sampled band really does span the interesting structure, so
    // that a pass means the fringes were reproduced, not just a flat curve.
    assert!(
        seen_peak && seen_null,
        "the sampled band did not cover both a fringe peak and a fringe null"
    );
}

/// **Item 3, Mie** — scattering from a dielectric sphere, in the Rayleigh
/// limit where the Mie series reduces to a closed form.
///
/// A TFSF box illuminates the sphere; a closed flux surface drawn in the
/// scattered-field region between the TFSF box and the CPML collects the
/// scattered power. The measured cross-section is compared to
/// `σ = (8/3)π k⁴ a⁶ |(m²−1)/(m²+2)|²`, and — the sharper check — to the `k⁴`
/// scaling that defines Rayleigh scattering.
///
/// Ignored by default: a converged 3D run takes tens of seconds in release and
/// minutes in a debug build. Run it with
/// `cargo test --release -p phyz-em -- --ignored --nocapture`.
#[test]
#[ignore = "3D scattering run; too slow for the default suite"]
fn sphere_scattering_matches_rayleigh() {
    use phyz_em::scattering::{CrossSectionMonitor, rayleigh_cross_section};

    let d = 5e-9;
    let dt = d / (C * 3_f64.sqrt()) * 0.99;
    let n = 72;
    let eps: f64 = 4.0;
    let m = eps.sqrt();

    let mut grid = YeeGrid::new(n, n, n, d, dt);
    grid.set_boundary(BoundaryCondition::Cpml(CpmlConfig::with_thickness(10)));

    let radius = 6.0 * d;
    let centre = grid.index_to_position(n / 2, n / 2, n / 2);
    let id = grid.add_material(DispersiveMaterial::non_dispersive(eps));
    grid.assign_material_sphere(id, centre, radius);
    grid.build_dispersion();

    // TFSF box well inside the CPML; flux surface between the two.
    let mut tfsf = Tfsf::box_region(&grid, 18, 53, 18, 53, 18, 53);
    let mut monitor_omegas = Vec::new();
    // Size parameters ka = 0.25 … 0.5: firmly Rayleigh, and each wavelength
    // (≈ 75–150 cells) still fits the domain.
    let kas = [0.25_f64, 0.30, 0.35, 0.40, 0.45, 0.50];
    for &ka in &kas {
        monitor_omegas.push(ka / radius * C);
    }
    let mut monitor =
        CrossSectionMonitor::new(&grid, 14, 57, 14, 57, 14, 57, monitor_omegas.clone());

    let spread = 30.0 * dt;
    let t0 = 5.0 * spread;
    let src = move |t: f64| gaussian_pulse(t, t0, spread);

    let n_steps = 4000;
    let mut inc = vec![0.0_f64; monitor_omegas.len()];
    let mut inc_series = Vec::with_capacity(n_steps);
    for step in 0..n_steps {
        tfsf.step(&mut grid, &src);
        monitor.record(&grid, step as f64 * dt);
        inc_series.push(tfsf.ex_inc(n / 2));
    }
    for (fi, &w) in monitor_omegas.iter().enumerate() {
        inc[fi] = spectrum(&inc_series, &[w], dt)[0].norm();
    }

    let eta0 = (4.0 * std::f64::consts::PI * 1e-7 / 8.854187817e-12_f64).sqrt();
    let measured = monitor.cross_section(&inc, eta0);

    println!("\nRayleigh scattering from an ε_r = {eps} sphere, a = {radius:.2e} m");
    println!("  (radius = 6 cells; staircasing dominates the error)");
    let geometric = std::f64::consts::PI * radius * radius;
    let mut ratios = Vec::new();
    for (fi, &ka) in kas.iter().enumerate() {
        let k = ka / radius;
        let analytic = rayleigh_cross_section(k, radius, m);
        let ratio = measured[fi] / analytic;
        ratios.push(ratio);
        println!(
            "  ka = {ka:.2} : σ/σ_geom measured {:.4e}, analytic {:.4e}, ratio {:.3}",
            measured[fi] / geometric,
            analytic / geometric,
            ratio
        );
    }

    // 1. The k⁴ scaling. This is the signature of Rayleigh scattering and it
    //    is insensitive to the constant staircasing bias, so it is the strong
    //    claim: the ratio to theory must be flat across the band.
    let lo = ratios[0];
    let hi = ratios[ratios.len() - 1];
    println!("  ratio-to-theory across the band: {lo:.3} → {hi:.3}");
    assert!(
        (hi / lo - 1.0).abs() < 0.15,
        "cross-section does not follow the k⁴ law: ratio drifts {lo:.3} → {hi:.3}"
    );

    // 2. The absolute magnitude, to the accuracy a 6-cell staircased sphere
    //    can support.
    for (fi, &ka) in kas.iter().enumerate() {
        assert!(
            (ratios[fi] - 1.0).abs() < 0.35,
            "ka = {ka:.2}: measured cross-section is {:.2}× theory",
            ratios[fi]
        );
    }
}
