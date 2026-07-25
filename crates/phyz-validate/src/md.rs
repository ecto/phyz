//! Molecular dynamics (`phyz-md`) validation.
//!
//! 1. **Symplectic energy behaviour** — for velocity Verlet the total-energy
//!    error must be a *bounded* O(Δt²) oscillation, not a secular drift. Both
//!    the amplitude scaling and the secular slope are measured.
//! 2. **Start-up consistency** — that the first step is a valid Verlet step,
//!    established by the convergence order of its energy error.
//! 3. **Dense-fluid structure** — the radial distribution function of the
//!    Lennard-Jones fluid at Verlet's canonical state point, against published
//!    values.
//!
//! `phyz-md` works in Å, eV, amu, fs. The integrator benchmarks use reduced
//! Lennard-Jones parameters (ε = σ = m = 1) so their numbers are comparable to
//! the literature; the corresponding reduced time unit is `τ = σ√(m/ε)`, which
//! in this unit system is `1/√FORCE_TO_ACCEL ≈ 10.18 fs`.

use crate::report::{Convergence, ErrorKind, Suite, Validation};
use phyz_math::Vec3;
use phyz_md::field::units::FORCE_TO_ACCEL;
use phyz_md::{Lattice, LennardJones, MdSystem, Particle, Rdf};

const CRATE: &str = "phyz-md";

/// The Lennard-Jones reduced time unit `σ√(m/ε)` for ε = σ = m = 1, in fs.
fn tau() -> f64 {
    1.0 / FORCE_TO_ACCEL.sqrt()
}

/// Reduced-unit Lennard-Jones with the conventional 2.5σ cutoff.
fn reduced_lj() -> LennardJones {
    LennardJones::monatomic(1.0, 1.0, 2.5)
}

/// Two particles bound in the Lennard-Jones well, oscillating radially.
///
/// Released from rest at 1.5σ: `V(1.5σ) < 0` with zero kinetic energy, so the
/// pair is genuinely bound and oscillates between 1.5σ and ≈1.0σ, always well
/// inside the cutoff. That isolates the integrator from force-truncation and
/// neighbour-list effects.
fn lj_dimer(dt: f64) -> MdSystem {
    let mut system = MdSystem::lennard_jones(reduced_lj(), dt);
    system.add_particle(Particle::new(
        Vec3::new(-0.75, 0.0, 0.0),
        Vec3::zeros(),
        1.0,
        0,
    ));
    system.add_particle(Particle::new(
        Vec3::new(0.75, 0.0, 0.0),
        Vec3::zeros(),
        1.0,
        0,
    ));
    system.compute_forces();
    system
}

/// Run for a fixed physical time; return `(bounded amplitude, secular slope)`
/// of the relative total-energy error, the slope per reduced time unit.
fn energy_error(system: &mut MdSystem, dt: f64, t_end: f64) -> (f64, f64) {
    let steps = (t_end / dt).round() as usize;
    let e0 = system.total_energy();
    let scale = e0.abs().max(1e-12);

    let mut amplitude: f64 = 0.0;
    let (mut sx, mut sy, mut sxx, mut sxy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for s in 0..steps {
        system.step();
        let err = (system.total_energy() - e0) / scale;
        amplitude = amplitude.max(err.abs());
        let t = (s + 1) as f64 * dt / tau();
        sx += t;
        sy += err;
        sxx += t * t;
        sxy += t * err;
        n += 1.0;
    }
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    (amplitude, slope)
}

/// Equilibrated Lennard-Jones argon at a reduced state point.
struct FluidResult {
    g: Vec<f64>,
    r: Vec<f64>,
    u_per_particle: f64,
    temperature: f64,
    pressure: f64,
    sigma: f64,
    epsilon: f64,
}

/// Melt an FCC lattice of argon and sample `g(r)`, `⟨U⟩/N`, `⟨T⟩` and `⟨P⟩`.
///
/// Run in the crate's own argon parameterisation and converted to reduced units
/// only for reporting, so nothing depends on the harness inventing a unit system.
fn lj_fluid(rho_star: f64, t_star: f64, n_cells: usize) -> FluidResult {
    let (epsilon, sigma) = (0.0103, 3.4);
    let k_b = phyz_md::field::units::KB_EV_PER_K;
    let mass = 39.948;

    // The Lennard-Jones time unit τ = σ√(m/ε). Because this unit system writes
    // a = FORCE_TO_ACCEL·F/m, the mass that enters τ is m/FORCE_TO_ACCEL, so
    // τ = σ√(m / (FORCE_TO_ACCEL·ε)) — about 2.16 ps for argon. Use the standard
    // Lennard-Jones timestep of 0.005 τ.
    let tau_ar = sigma * (mass / (FORCE_TO_ACCEL * epsilon)).sqrt();
    let dt = 0.005 * tau_ar;

    let mut system = MdSystem::with_seed(dt, 0x5EED_1234);
    system.set_lennard_jones(LennardJones::argon());

    let n_particles = 4 * n_cells * n_cells * n_cells;
    let l = (n_particles as f64 * sigma * sigma * sigma / rho_star).cbrt();
    let a = l / n_cells as f64;
    let basis = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ];
    for i in 0..n_cells {
        for j in 0..n_cells {
            for k in 0..n_cells {
                for (bx, by, bz) in basis {
                    system.add_particle(Particle::new(
                        Vec3::new(
                            (i as f64 + bx) * a,
                            (j as f64 + by) * a,
                            (k as f64 + bz) * a,
                        ),
                        Vec3::zeros(),
                        mass,
                        0,
                    ));
                }
            }
        }
    }
    system.set_box_size(Vec3::new(l, l, l));

    let temperature = t_star * epsilon / k_b;
    system.initialize_velocities(temperature, k_b);
    system.compute_forces();

    // Melt under a Langevin thermostat, then relax in NVE.
    system.set_thermostat(temperature, 0.01, k_b);
    for _ in 0..20_000 {
        system.step();
    }
    system.clear_thermostat();
    let t_now = system.temperature(k_b);
    let scale = (temperature / t_now).sqrt();
    for v in system.velocities.iter_mut() {
        for c in v.iter_mut() {
            *c *= scale;
        }
    }
    for _ in 0..5_000 {
        system.step();
    }

    // Production.
    let r_max = 0.5 * l;
    let bins = 250;
    let mut rdf = Rdf::new(bins, r_max);
    let cell = Lattice::cubic(l);
    let (mut u_sum, mut t_sum, mut p_sum, mut samples) = (0.0, 0.0, 0.0, 0.0);
    for s in 0..20_000_usize {
        system.step();
        if s.is_multiple_of(10) {
            rdf.accumulate(&system.positions, &cell);
            u_sum += system.compute_potential_energy() / n_particles as f64;
            t_sum += system.temperature(k_b);
            p_sum += system.pressure();
            samples += 1.0;
        }
    }
    let g = rdf.finish().to_vec();
    let dr = r_max / bins as f64;
    let r = (0..bins).map(|b| (b as f64 + 0.5) * dr / sigma).collect();

    FluidResult {
        g,
        r,
        u_per_particle: u_sum / samples / epsilon,
        temperature: t_sum / samples * k_b / epsilon,
        // P* = P σ³/ε; `pressure()` is in eV/Å³.
        pressure: p_sum / samples * sigma * sigma * sigma / epsilon,
        sigma,
        epsilon,
    }
}

/// Locate the first maximum of `g(r)` and the following minimum, in reduced `r`.
fn peak_and_trough(r: &[f64], g: &[f64]) -> ((f64, f64), (f64, f64)) {
    let start = r.iter().position(|&x| x > 0.9).unwrap_or(0);
    let mut imax = start;
    for i in start..g.len() {
        if r[i] > 1.4 {
            break;
        }
        if g[i] > g[imax] {
            imax = i;
        }
    }
    let mut imin = imax;
    for i in imax..g.len() {
        if r[i] > 1.9 {
            break;
        }
        if g[i] < g[imin] {
            imin = i;
        }
    }
    ((r[imax], g[imax]), (r[imin], g[imin]))
}

/// Run every molecular-dynamics validation.
pub fn run() -> Suite {
    let mut suite = Suite::new("Molecular dynamics — Lennard-Jones (`phyz-md`)");

    // ---- 1. Symplectic energy behaviour on a clean two-body problem ---------
    let t_end = 40.0 * tau();
    let mut samples = Vec::new();
    let mut slopes = Vec::new();
    for &dt_star in &[0.008_f64, 0.004, 0.002, 0.001] {
        let dt = dt_star * tau();
        let mut s = lj_dimer(dt);
        let (amp, slope) = energy_error(&mut s, dt, t_end);
        samples.push((dt_star, amp));
        slopes.push((dt_star, slope.abs()));
    }
    let finest = samples.last().unwrap().1;
    suite.push(
        Validation::new(
            "md.verlet_energy_order",
            "Velocity Verlet: bounded energy error scales as Δt²",
            CRATE,
            "Hairer, Lubich & Wanner, *Geometric Numerical Integration* — a symplectic \
             second-order integrator has an O(Δt²) *bounded* energy error, with no secular term",
            "peak |ΔE|/|E| over 40 reduced time units, LJ dimer, Δt = 0.001 τ",
            finest,
            0.0,
            ErrorKind::Absolute,
            1e-4,
        )
        .with_convergence(Convergence::fit("Δt/τ", samples.clone(), 2.0, 0.3))
        .note(
            "The dimer never approaches the 2.5σ cutoff, so this isolates the integrator from \
             force-truncation and neighbour-list artefacts.",
        ),
    );

    let slope_finest = slopes.last().unwrap().1;
    suite.push(
        Validation::new(
            "md.verlet_secular_drift",
            "Velocity Verlet: no secular energy drift on the LJ dimer",
            CRATE,
            "Symplectic integrators conserve a shadow Hamiltonian exactly, so d⟨E⟩/dt = 0",
            "|d(ΔE/E)/dt| per reduced time unit, Δt = 0.001 τ",
            slope_finest,
            0.0,
            ErrorKind::Absolute,
            1e-6,
        )
        .note(format!(
            "secular slopes across Δt: {}",
            slopes
                .iter()
                .map(|(d, s)| format!("Δt={d:.4}τ → {s:.3e}"))
                .collect::<Vec<_>>()
                .join(", ")
        )),
    );

    // ---- 2. Start-up consistency -------------------------------------------
    // A start-up that drifts with a = 0 drops half of the first kick — an
    // O(Δt) velocity error, which shows up here as convergence order 1. A sound
    // start-up shows the ordinary O(Δt³) local truncation error of one step.
    let mut startup = Vec::new();
    for &dt_star in &[0.008_f64, 0.004, 0.002, 0.001] {
        let dt = dt_star * tau();
        let mut s = MdSystem::lennard_jones(reduced_lj(), dt);
        // A non-zero initial velocity along the force direction is essential:
        // the dropped half-kick perturbs v by ½a(0)Δt, so the energy error is
        // m·v·Δv = O(Δt) only when v(0)·a(0) ≠ 0. Starting from rest would hide
        // the defect behind an O(Δt²) leading term.
        s.add_particle(Particle::new(
            Vec3::new(-0.75, 0.0, 0.0),
            Vec3::new(0.15, 0.0, 0.0),
            1.0,
            0,
        ));
        s.add_particle(Particle::new(
            Vec3::new(0.75, 0.0, 0.0),
            Vec3::new(-0.15, 0.0, 0.0),
            1.0,
            0,
        ));
        // Measure E(0) without disturbing `s`, then take one step through the
        // crate's own start-up path.
        let mut probe = MdSystem::lennard_jones(reduced_lj(), dt);
        probe.add_particle(s.particle(0));
        probe.add_particle(s.particle(1));
        probe.compute_forces();
        let e0 = probe.total_energy();

        s.step();
        startup.push((dt_star, (s.total_energy() - e0).abs() / e0.abs()));
    }
    let finest = startup.last().unwrap().1;
    suite.push(
        Validation::new(
            "md.startup_consistency",
            "First integration step is a valid velocity-Verlet step",
            CRATE,
            "A correct velocity-Verlet start-up evaluates a(0) before the first drift, so the \
             local truncation error of step 1 is O(Δt³) like every other step",
            "|ΔE|/|E| across the first step alone, Δt = 0.001 τ, v(0)·a(0) ≠ 0",
            finest,
            0.0,
            ErrorKind::Absolute,
            1e-6,
        )
        .with_convergence(Convergence::fit("Δt/τ", startup, 3.0, 0.3))
        .note(
            "`MdSystem::step` primes the force accumulator when `self.step == 0`, so it holds \
             F(x(0)) before the first drift. An earlier revision did not, and this benchmark \
             measured order 0.993 against it.",
        ),
    );

    // ---- 3. Dense-fluid structure at Verlet's state point -------------------
    let (rho_star, t_star) = (0.8442, 0.722);
    let fluid = lj_fluid(rho_star, t_star, 4); // 256 particles
    let ((r_peak, g_peak), (r_min, g_min)) = peak_and_trough(&fluid.r, &fluid.g);
    let core = fluid
        .r
        .iter()
        .zip(fluid.g.iter())
        .filter(|(r, _)| **r < 0.8)
        .fold(0.0_f64, |a, (_, g)| a.max(*g));

    suite.push(
        Validation::new(
            "md.rdf.core_exclusion",
            "LJ fluid g(r): excluded volume inside the repulsive core",
            CRATE,
            "Verlet, *Phys. Rev.* 165 (1968) 201 — g(r) ≈ 0 for r* < 0.85 at ρ* = 0.8442",
            "max g(r) for r* < 0.8",
            core,
            0.0,
            ErrorKind::Absolute,
            0.02,
        )
        .note(format!(
            "production average T* = {:.4} (target {t_star}); ε = {} eV, σ = {} Å",
            fluid.temperature, fluid.epsilon, fluid.sigma
        )),
    );
    suite.push(Validation::new(
        "md.rdf.first_peak_position",
        "LJ fluid g(r): first-peak position",
        CRATE,
        "Verlet (1968); Hansen & McDonald, *Theory of Simple Liquids* Fig. 4.2 — \
         first peak at r* ≈ 1.09 for ρ* = 0.8442, T* = 0.722",
        "r* of the first maximum of g(r)",
        r_peak,
        1.09,
        ErrorKind::Absolute,
        0.04,
    ));
    suite.push(Validation::new(
        "md.rdf.first_peak_height",
        "LJ fluid g(r): first-peak height",
        CRATE,
        "Verlet (1968) — g(r_max) ≈ 3.0 for ρ* = 0.8442, T* = 0.722",
        "g(r) at the first maximum",
        g_peak,
        3.0,
        ErrorKind::Relative,
        0.12,
    ));
    suite.push(Validation::new(
        "md.rdf.first_minimum",
        "LJ fluid g(r): first-minimum position",
        CRATE,
        "Verlet (1968) — first minimum at r* ≈ 1.55",
        "r* of the first minimum of g(r)",
        r_min,
        1.55,
        ErrorKind::Absolute,
        0.06,
    ));
    suite.push(
        Validation::new(
            "md.rdf.first_minimum_depth",
            "LJ fluid g(r): depth of the first minimum",
            CRATE,
            "Verlet (1968) — g ≈ 0.60 at the first minimum",
            "g(r) at the first minimum",
            g_min,
            0.60,
            ErrorKind::Relative,
            0.2,
        )
        .note(format!("measured minimum at r* = {r_min:.3}")),
    );

    // Thermodynamics is reported rather than asserted: the published value moves
    // by more than this run's statistical error depending on whether the
    // potential is truncated, truncated-and-shifted, or tail-corrected.
    // `LennardJones` here shifts at the cutoff.
    suite.push(
        Validation::new(
            "md.thermo.energy",
            "LJ fluid excess energy at ρ* = 0.8442, T* = 0.722",
            CRATE,
            "Verlet, *Phys. Rev.* 159 (1967) 98, Table II gives U*/N ≈ −5.7 for the \
             *truncated* potential; Johnson, Zollweg & Gubbins, *Mol. Phys.* 78 (1993) 591 \
             give ≈ −5.4 truncated-and-shifted at 2.5σ, which is what this crate implements",
            "⟨U⟩/N in reduced units",
            fluid.u_per_particle,
            -5.4,
            ErrorKind::Relative,
            0.05,
        )
        .diagnostic()
        .note(
            "Reported: the truncation convention shifts the reference by ~5%, more than the \
             statistical error here, so a tight pass/fail claim would be about the \
             convention rather than the solver.",
        ),
    );
    suite.push(
        Validation::new(
            "md.thermo.pressure",
            "LJ fluid virial pressure at ρ* = 0.8442, T* = 0.722",
            CRATE,
            "Verlet (1967) Table II — P* ≈ 0.3 at this state point, with the same \
             truncation-convention caveat as the energy",
            "⟨P⟩ in reduced units",
            fluid.pressure,
            0.3,
            ErrorKind::Absolute,
            0.35,
        )
        .diagnostic(),
    );

    suite
}
