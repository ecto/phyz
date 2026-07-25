//! Molecular dynamics (`phyz-md`) validation.
//!
//! All benchmarks run in Lennard-Jones reduced units (ε = σ = m = k_B = 1) so
//! the numbers are directly comparable to the published LJ literature.
//!
//! 1. **Symplectic energy-drift scaling** — for velocity Verlet the total-energy
//!    error must be a *bounded* O(Δt²) oscillation, not a secular drift. Both the
//!    amplitude scaling and the secular slope are measured.
//! 2. **Start-up consistency** — whether the first step is a valid Verlet step.
//! 3. **Radial distribution function** — structure of the LJ fluid at Verlet's
//!    canonical state point ρ* = 0.8442, T* = 0.722, against published values.

use crate::report::{Convergence, ErrorKind, Suite, Validation};
use phyz_math::Vec3;
use phyz_md::{LennardJones, MdSystem, Particle};
use std::sync::Arc;

const CRATE: &str = "phyz-md";

/// Reduced-unit Lennard-Jones with the conventional 2.5σ cutoff.
fn lj() -> Arc<LennardJones> {
    Arc::new(LennardJones::new(1.0, 1.0, 2.5))
}

/// Fill `system` with an FCC lattice of `4·n³` particles at reduced density `rho`.
fn fcc(system: &mut MdSystem, n: usize, rho: f64) -> f64 {
    let n_particles = 4 * n * n * n;
    let l = (n_particles as f64 / rho).cbrt();
    let a = l / n as f64;
    let basis = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for (bx, by, bz) in basis {
                    system.add_particle(Particle::new(
                        Vec3::new(
                            (i as f64 + bx) * a,
                            (j as f64 + by) * a,
                            (k as f64 + bz) * a,
                        ),
                        Vec3::zeros(),
                        1.0,
                        0,
                    ));
                }
            }
        }
    }
    system.set_box_size(Vec3::new(l, l, l));
    l
}

/// Compute and store the forces the system's own `compute_forces` would produce.
///
/// `MdSystem::step` reads `particle.f` as `a(t)`, but nothing initialises it, so
/// the harness primes it explicitly for the integrator benchmarks. See the
/// `md.startup_consistency` entry for what happens when it is not primed.
fn prime_forces(system: &mut MdSystem) {
    system
        .neighbor_list
        .build(&system.particles, system.box_size);
    for p in system.particles.iter_mut() {
        p.reset_force();
    }
    let pairs = system.neighbor_list.pairs.clone();
    for (i, j) in pairs {
        let mut r = system.particles[j].x - system.particles[i].x;
        if let Some(b) = system.box_size {
            r = phyz_md::minimum_image(r, b);
        }
        let (f, _) = system.force_field.compute_force(
            r,
            system.particles[i].atom_type,
            system.particles[j].atom_type,
        );
        system.particles[i].add_force(f);
        system.particles[j].add_force(-f);
    }
}

fn total_energy(system: &MdSystem) -> f64 {
    system.kinetic_energy() + system.potential_energy()
}

/// Two particles bound in the Lennard-Jones well, oscillating radially.
///
/// The pair never approaches the cutoff, so this isolates the integrator from
/// the force-truncation and neighbour-list effects present in a dense fluid.
fn lj_dimer(dt: f64) -> MdSystem {
    let mut system = MdSystem::new(lj(), dt);
    system.rebuild_frequency = 1;
    // Released from rest at 1.5σ: V(1.5σ) < 0 with zero kinetic energy, so the
    // pair is genuinely bound and oscillates between 1.5σ and ~1.0σ, always
    // well inside the 2.5σ cutoff.
    let r0 = 1.5;
    system.add_particle(Particle::new(
        Vec3::new(-0.5 * r0, 0.0, 0.0),
        Vec3::zeros(),
        1.0,
        0,
    ));
    system.add_particle(Particle::new(
        Vec3::new(0.5 * r0, 0.0, 0.0),
        Vec3::zeros(),
        1.0,
        0,
    ));
    prime_forces(&mut system);
    system
}

/// Run for a fixed physical time and return `(bounded amplitude, secular slope)`
/// of the relative total-energy error.
fn energy_error(system: &mut MdSystem, dt: f64, t_end: f64) -> (f64, f64) {
    let steps = (t_end / dt).round() as usize;
    let e0 = total_energy(system);
    let scale = e0.abs().max(1e-12);

    let mut amplitude: f64 = 0.0;
    let (mut sx, mut sy, mut sxx, mut sxy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for s in 0..steps {
        system.step();
        let err = (total_energy(system) - e0) / scale;
        amplitude = amplitude.max(err.abs());
        let t = (s + 1) as f64 * dt;
        sx += t;
        sy += err;
        sxx += t * t;
        sxy += t * err;
        n += 1.0;
    }
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    (amplitude, slope)
}

/// Radial distribution function of the current configuration.
fn rdf(system: &MdSystem, n_bins: usize, r_max: f64, hist: &mut [f64]) {
    let dr = r_max / n_bins as f64;
    let b = system.box_size.expect("rdf needs a periodic box");
    let n = system.particles.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = phyz_md::minimum_image(system.particles[j].x - system.particles[i].x, b);
            let r = d.norm();
            if r < r_max {
                hist[(r / dr) as usize] += 2.0;
            }
        }
    }
}

/// Normalise an accumulated pair histogram into g(r); returns `(r_centres, g)`.
fn normalize_rdf(
    hist: &[f64],
    n_frames: f64,
    n_particles: usize,
    volume: f64,
    r_max: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_bins = hist.len();
    let dr = r_max / n_bins as f64;
    let rho = n_particles as f64 / volume;
    let mut r = Vec::with_capacity(n_bins);
    let mut g = Vec::with_capacity(n_bins);
    for (k, &h) in hist.iter().enumerate() {
        let r_lo = k as f64 * dr;
        let r_hi = r_lo + dr;
        let shell = 4.0 / 3.0 * std::f64::consts::PI * (r_hi.powi(3) - r_lo.powi(3));
        let ideal = shell * rho * n_particles as f64;
        r.push(0.5 * (r_lo + r_hi));
        g.push(h / (n_frames * ideal));
    }
    (r, g)
}

/// Virial pressure `P = ρk_BT + W/(3V)` with `W = −Σ r_ij·F_i` in the crate's
/// `r_ij = x_j − x_i` convention.
fn pressure(system: &MdSystem, temperature: f64) -> f64 {
    let b = system.box_size.expect("pressure needs a periodic box");
    let v = b.x * b.y * b.z;
    let n = system.particles.len() as f64;
    let mut w = 0.0;
    for &(i, j) in &system.neighbor_list.pairs {
        let r = phyz_md::minimum_image(system.particles[j].x - system.particles[i].x, b);
        let (f, _) = system.force_field.compute_force(
            r,
            system.particles[i].atom_type,
            system.particles[j].atom_type,
        );
        w -= r.dot(&f);
    }
    n / v * temperature + w / (3.0 * v)
}

/// Equilibrate a dense LJ fluid and accumulate g(r), ⟨U⟩/N, ⟨T⟩ and ⟨P⟩.
struct FluidResult {
    r: Vec<f64>,
    g: Vec<f64>,
    u_per_particle: f64,
    temperature: f64,
    pressure: f64,
}

fn lj_fluid(rho: f64, t_target: f64, n_cells: usize) -> FluidResult {
    let dt = 0.002;
    let mut system = MdSystem::new(lj(), dt);
    let l = fcc(&mut system, n_cells, rho);
    let n_particles = system.particles.len();
    system.initialize_velocities(t_target, 1.0);
    prime_forces(&mut system);

    // Melt and thermostat.
    system.set_thermostat(t_target, 1.0, 1.0);
    for _ in 0..15_000 {
        system.step();
    }
    system.thermostat = None;

    // Rescale once to the target temperature, then relax in NVE.
    let t_now = system.temperature(1.0);
    let scale = (t_target / t_now).sqrt();
    for p in system.particles.iter_mut() {
        p.v *= scale;
    }
    for _ in 0..5_000 {
        system.step();
    }

    // Production.
    let n_bins = 200;
    let r_max = 0.5 * l;
    let mut hist = vec![0.0; n_bins];
    let mut frames = 0.0;
    let mut u_sum = 0.0;
    let mut t_sum = 0.0;
    let mut p_sum = 0.0;
    let mut samples = 0.0;
    for s in 0..20_000_usize {
        system.step();
        if s.is_multiple_of(10) {
            rdf(&system, n_bins, r_max, &mut hist);
            frames += 1.0;
            u_sum += system.potential_energy() / n_particles as f64;
            let t = system.temperature(1.0);
            t_sum += t;
            p_sum += pressure(&system, t);
            samples += 1.0;
        }
    }

    let (r, g) = normalize_rdf(&hist, frames, n_particles, l * l * l, r_max);
    FluidResult {
        r,
        g,
        u_per_particle: u_sum / samples,
        temperature: t_sum / samples,
        pressure: p_sum / samples,
    }
}

/// Locate the first maximum and the following minimum of g(r) above `r_min`.
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
    let t_end = 40.0; // reduced time units, ~50 oscillation periods
    let mut samples = Vec::new();
    let mut slopes = Vec::new();
    for &dt in &[0.008_f64, 0.004, 0.002, 0.001] {
        let mut s = lj_dimer(dt);
        let (amp, slope) = energy_error(&mut s, dt, t_end);
        samples.push((dt, amp));
        slopes.push((dt, slope.abs()));
    }
    let finest = samples.last().unwrap().1;
    suite.push(
        Validation::new(
            "md.verlet_energy_order",
            "Velocity Verlet: bounded energy error scales as Δt²",
            CRATE,
            "Hairer, Lubich & Wanner, *Geometric Numerical Integration* — a symplectic \
             second-order integrator has an O(Δt²) *bounded* energy error, with no secular term",
            "peak |ΔE|/|E| over 40 reduced time units, LJ dimer, Δt = 0.001",
            finest,
            0.0,
            ErrorKind::Absolute,
            1e-4,
        )
        .with_convergence(Convergence::fit("Δt", samples.clone(), 2.0, 0.3))
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
            "|d(ΔE/E)/dt| per reduced time unit, Δt = 0.001",
            slope_finest,
            0.0,
            ErrorKind::Absolute,
            1e-6,
        )
        .note(format!(
            "secular slopes across Δt: {}",
            slopes
                .iter()
                .map(|(d, s)| format!("Δt={d:.4} → {s:.3e}"))
                .collect::<Vec<_>>()
                .join(", ")
        )),
    );

    // ---- 2. Start-up consistency -------------------------------------------
    // `MdSystem::step` reads `particle.f` as a(t) but never initialises it, so
    // the first step applies only half of the first kick.
    let mut startup = Vec::new();
    for &dt in &[0.008_f64, 0.004, 0.002, 0.001] {
        let mut s = MdSystem::new(lj(), dt);
        s.rebuild_frequency = 1;
        // A non-zero initial velocity along the force direction is essential:
        // the dropped half-kick perturbs v by ½a(0)Δt, and the resulting energy
        // error is m·v·Δv = O(Δt) only when v(0)·a(0) ≠ 0. Starting from rest
        // would hide the defect behind an O(Δt²) leading term.
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
        // Deliberately do NOT prime the forces — this is the crate's own start-up path.
        s.neighbor_list.build(&s.particles, s.box_size);
        let e0 = total_energy(&s);
        s.step();
        let e1 = total_energy(&s);
        startup.push((dt, (e1 - e0).abs() / e0.abs()));
    }
    let finest = startup.last().unwrap().1;
    suite.push(
        Validation::new(
            "md.startup_consistency",
            "First integration step is a valid velocity-Verlet step",
            CRATE,
            "A correct velocity-Verlet start-up evaluates a(0) before the first drift, so the \
             energy error of step 1 is O(Δt²) like every other step",
"|ΔE|/|E| across the first step alone, Δt = 0.001, v(0)·a(0) ≠ 0",
            finest,
            0.0,
            ErrorKind::Absolute,
            1e-6,
        )
        .with_convergence(Convergence::fit("Δt", startup, 2.0, 0.3))
        .note(
            "`MdSystem::step` (crates/phyz-md/src/system.rs:210-263) reads `particle.f` as a(t), \
             but forces are only ever written at the *end* of a step and nothing computes them at \
             construction time. On step 0 the stored force is zero, so the drift uses a = 0 and \
             the first half-kick is dropped entirely — a one-off O(Δt) velocity error. The \
             measured convergence order below distinguishes the two cases: order 1 confirms the \
             dropped kick, order 2 would mean the start-up is sound. Every benchmark above \
             primes the forces explicitly to work around this.",
        ),
    );

    // ---- 3. Dense-fluid structure at Verlet's state point -------------------
    let rho = 0.8442;
    let t_target = 0.722;
    let fluid = lj_fluid(rho, t_target, 4); // 256 particles
    let ((r_peak, g_peak), (r_min, g_min)) = peak_and_trough(&fluid.r, &fluid.g);

    // g(r) must vanish inside the repulsive core.
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
            "production average T* = {:.4} (target {t_target})",
            fluid.temperature
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
        "LJ fluid g(r): first-minimum position and depth",
        CRATE,
        "Verlet (1968) — first minimum at r* ≈ 1.55 with g ≈ 0.60",
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

    suite.push(
        Validation::new(
            "md.thermo.energy",
            "LJ fluid excess energy at ρ* = 0.8442, T* = 0.722",
            CRATE,
            "Verlet, *Phys. Rev.* 159 (1967) 98, Table II; Johnson, Zollweg & Gubbins, \
             *Mol. Phys.* 78 (1993) 591 — U*/N ≈ −5.7 with a 2.5σ truncation (no tail correction)",
            "⟨U⟩/N in reduced units",
            fluid.u_per_particle,
            -5.7,
            ErrorKind::Relative,
            0.05,
        )
        .note("Reported without a long-range tail correction, matching the crate's hard truncation."),
    );
    suite.push(
        Validation::new(
            "md.thermo.pressure",
            "LJ fluid virial pressure at ρ* = 0.8442, T* = 0.722",
            CRATE,
            "Verlet (1967) Table II — P*V/Nk_BT ≈ 0.5, i.e. P* ≈ 0.3 at this state point \
             with a 2.5σ truncation",
            "⟨P⟩ in reduced units",
            fluid.pressure,
            0.3,
            ErrorKind::Absolute,
            0.35,
        )
        .diagnostic()
        .note(
            "Reported as a diagnostic: the truncation convention (shifted vs unshifted, tail \
             correction) moves the reference value by more than the statistical error of this \
             run, so a tight pass/fail claim would not be meaningful.",
        ),
    );

    suite
}
