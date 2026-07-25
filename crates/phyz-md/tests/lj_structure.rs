//! Validation of the Lennard-Jones fluid against published structural data.
//!
//! The radial distribution function of the LJ fluid at ρ* = 0.8442, T* = 0.728
//! is one of the most reproduced results in liquid-state physics (Verlet,
//! *Phys. Rev.* **165**, 201 (1968); Rahman's 1964 argon study is at
//! essentially the same state point, and it is the standard benchmark case in
//! Allen & Tildesley and in Frenkel & Smit). The published curve has:
//!
//! - a hard core: `g(r) ≈ 0` below `0.85 σ`
//! - a first peak at `r ≈ 1.09 σ` of height `≈ 3.0`
//! - a first minimum near `1.55 σ` at `g ≈ 0.6`
//! - a second peak near `2.0 σ` at `g ≈ 1.2`
//! - `g → 1` beyond `≈ 2.5 σ`
//!
//! Reproducing all five simultaneously exercises the potential, the neighbor
//! list, the minimum-image convention, and the integrator at once — any one of
//! them being wrong distorts the curve.

use phyz_math::Vec3;
use phyz_md::field::units::KB_EV_PER_K;
use phyz_md::{Berendsen, LennardJones, MdSystem, Particle, Rdf};

const SIGMA: f64 = 3.4;
const EPSILON: f64 = 0.0103;
const MASS_AR: f64 = 39.948;

/// The Verlet reference state point in reduced units.
const RHO_STAR: f64 = 0.8442;
const T_STAR: f64 = 0.728;

/// Build an FCC configuration of `4 n_side³` atoms at reduced density
/// `RHO_STAR`.
fn fcc_system(n_side: usize, dt: f64) -> MdSystem {
    let n = 4 * n_side * n_side * n_side;
    let rho = RHO_STAR / SIGMA.powi(3);
    let l = (n as f64 / rho).cbrt();
    let a = l / n_side as f64;

    // Seeded so the melt/equilibrate/sample run is reproducible.
    let mut system = MdSystem::with_seed(dt, 0x5EED_1EE7);
    system.set_lennard_jones(LennardJones::monatomic(EPSILON, SIGMA, 2.5 * SIGMA));
    // A skin that keeps at least three cells per axis in this box, so the
    // binned neighbor build is what gets exercised.
    system.neighbor_list.skin = 0.6;

    for ix in 0..n_side {
        for iy in 0..n_side {
            for iz in 0..n_side {
                let origin = Vec3::new(ix as f64, iy as f64, iz as f64) * a;
                for basis in [
                    Vec3::zeros(),
                    Vec3::new(0.5, 0.5, 0.0) * a,
                    Vec3::new(0.5, 0.0, 0.5) * a,
                    Vec3::new(0.0, 0.5, 0.5) * a,
                ] {
                    system.add_particle(Particle::new(origin + basis, Vec3::zeros(), MASS_AR, 0));
                }
            }
        }
    }
    system.set_box_size(Vec3::new(l, l, l));
    system
}

#[test]
fn lennard_jones_rdf_matches_published_reference() {
    let t_melt = 3.0 * EPSILON / KB_EV_PER_K; // T* = 3.0, well into the fluid
    let t_target = T_STAR * EPSILON / KB_EV_PER_K; // ≈ 87 K

    let dt = 8.0; // fs, ≈ 0.004 LJ time units for argon
    let mut system = fcc_system(5, dt); // 500 atoms
    let l = system.volume().cbrt();
    assert!(
        2.5 * SIGMA < 0.5 * l,
        "cutoff must fit the minimum-image convention"
    );

    system.initialize_velocities(t_melt, KB_EV_PER_K);
    system.compute_forces();
    assert!(
        !system.neighbor_list.used_fallback(),
        "expected the cell-list path for {} atoms in a {l:.1} Å box",
        system.len()
    );

    // Melt the lattice, then cool to the reference state point.
    system.berendsen = Some(Berendsen {
        target_k: t_melt,
        tau_fs: 100.0,
    });
    for _ in 0..1500 {
        system.step();
    }
    system.berendsen = Some(Berendsen {
        target_k: t_target,
        tau_fs: 200.0,
    });
    for _ in 0..3000 {
        system.step();
    }

    // Sample g(r) over the production run.
    let r_max = 0.5 * l * 0.99;
    let mut rdf = Rdf::new(150, r_max);
    for s in 0..3000 {
        system.step();
        if s % 10 == 0 {
            rdf.accumulate(&system.positions, system.cell.as_ref().unwrap());
        }
    }
    rdf.finish();

    let t_final = system.temperature(KB_EV_PER_K) / (EPSILON / KB_EV_PER_K);
    assert!(
        (t_final - T_STAR).abs() < 0.25,
        "sampled at T* = {t_final:.3}, wanted {T_STAR}"
    );

    let at = |r_over_sigma: f64| -> f64 {
        let target = r_over_sigma * SIGMA;
        let (mut best, mut best_d) = (0.0, f64::INFINITY);
        for (i, &r) in rdf.r.iter().enumerate() {
            if (r - target).abs() < best_d {
                best_d = (r - target).abs();
                best = rdf.g[i];
            }
        }
        best
    };

    // Excluded volume: nothing inside 0.85 σ.
    for (i, &r) in rdf.r.iter().enumerate() {
        if r < 0.85 * SIGMA {
            assert!(
                rdf.g[i] < 0.05,
                "g({:.2} σ) = {:.3}, expected an empty core",
                r / SIGMA,
                rdf.g[i]
            );
        }
    }

    // First peak: position and height.
    let (mut peak_r, mut peak_g) = (0.0, 0.0);
    for (i, &r) in rdf.r.iter().enumerate() {
        if r < 1.5 * SIGMA && rdf.g[i] > peak_g {
            peak_g = rdf.g[i];
            peak_r = r;
        }
    }
    let peak_r_star = peak_r / SIGMA;
    assert!(
        (1.03..=1.16).contains(&peak_r_star),
        "first peak at r = {peak_r_star:.3} σ, reference ≈ 1.09 σ"
    );
    assert!(
        (2.4..=3.7).contains(&peak_g),
        "first peak height {peak_g:.2}, reference ≈ 3.0"
    );

    // First minimum, second peak, and the approach to unity.
    let g_min = at(1.55);
    assert!(
        (0.4..=0.9).contains(&g_min),
        "first minimum g(1.55 σ) = {g_min:.2}, reference ≈ 0.6"
    );
    let g_second = at(2.0);
    assert!(
        (1.0..=1.45).contains(&g_second),
        "second peak g(2.0 σ) = {g_second:.2}, reference ≈ 1.2"
    );
    let g_tail = at(3.5);
    assert!(
        (g_tail - 1.0).abs() < 0.15,
        "g(3.5 σ) = {g_tail:.3}, should have decayed to 1"
    );
}

/// Pressure from the virial must agree with `−dE/dV` measured by finite
/// difference on a real many-atom configuration, not just on the analytic
/// pieces checked in unit tests.
#[test]
fn virial_pressure_matches_numerical_volume_derivative() {
    let mut system = fcc_system(3, 4.0); // 108 atoms
    system.initialize_velocities(80.0, KB_EV_PER_K);
    system.compute_forces();
    // Nudge off the perfect lattice so the configuration is generic.
    for _ in 0..200 {
        system.step();
    }
    system.compute_forces();

    let v0 = system.volume();
    let virial_p = (system.virial[0][0] + system.virial[1][1] + system.virial[2][2]) / (3.0 * v0);

    // Uniformly scale positions and cell, and difference the potential energy.
    let energy_at = |lambda: f64| -> f64 {
        let mut s = fcc_system(3, 4.0);
        s.positions = system
            .positions
            .iter()
            .map(|p| [p[0] * lambda, p[1] * lambda, p[2] * lambda])
            .collect();
        let l = system.volume().cbrt() * lambda;
        s.set_box_size(Vec3::new(l, l, l));
        s.neighbor_list.build(&s.positions, s.cell.as_ref());
        s.compute_potential_energy()
    };
    let h: f64 = 1e-5;
    let vp = v0 * (1.0 + h).powi(3);
    let vm = v0 * (1.0 - h).powi(3);
    let de_dv = (energy_at(1.0 + h) - energy_at(1.0 - h)) / (vp - vm);

    assert!(
        (virial_p + de_dv).abs() < 1e-3 * virial_p.abs().max(1e-8),
        "virial pressure {virial_p:.6e} vs −dE/dV {:.6e}",
        -de_dv
    );
}
