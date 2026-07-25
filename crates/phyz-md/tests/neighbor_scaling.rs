//! Empirical confirmation that force evaluation is O(N), not O(N²).
//!
//! The deterministic version of this check lives in
//! `field::neighbor::tests::neighbor_build_is_linear_in_atom_count`, which
//! counts candidate distance evaluations. This one measures wall-clock time
//! end to end, because that is what the claim actually means for a user.
//!
//! Timing is noisy, so the assertions are deliberately loose: across an 8×
//! increase in atom count, an O(N) implementation takes ~8× longer in total and
//! an O(N²) one ~64×. Anything under 24× is unambiguously not quadratic.

use std::time::{Duration, Instant};

use phyz_math::Vec3;
use phyz_md::{LennardJones, MdSystem, Particle};

/// A simple-cubic argon system of `n_side³` atoms at 4 Å spacing.
fn cubic_system(n_side: usize) -> MdSystem {
    let spacing = 4.0;
    let mut system = MdSystem::lennard_jones(LennardJones::monatomic(0.0103, 3.4, 5.0), 1.0);
    system.neighbor_list.skin = 0.5;
    for i in 0..n_side {
        for j in 0..n_side {
            for k in 0..n_side {
                // Break the perfect symmetry so bins fill unevenly, as they do
                // in a real fluid.
                let jitter = ((i * 7 + j * 13 + k * 29) % 11) as f64 * 0.02;
                system.add_particle(Particle::new(
                    Vec3::new(
                        i as f64 * spacing + jitter,
                        j as f64 * spacing - jitter,
                        k as f64 * spacing + 0.5 * jitter,
                    ),
                    Vec3::zeros(),
                    39.948,
                    0,
                ));
            }
        }
    }
    let l = n_side as f64 * spacing;
    system.set_box_size(Vec3::new(l, l, l));
    system
}

/// Time `steps` full force evaluations (neighbor rebuild included), taking the
/// best of several trials to suppress scheduler noise.
fn time_force_evaluations(n_side: usize, steps: usize, trials: usize) -> (usize, Duration) {
    let mut system = cubic_system(n_side);
    system.compute_forces();
    assert!(
        !system.neighbor_list.used_fallback(),
        "cell lists must be active at n_side = {n_side}"
    );
    // Force a rebuild every step: this measures the neighbor build *and* the
    // force loop, which is the pair that has to be linear.
    system.rebuild_frequency = 1;

    let mut best = Duration::MAX;
    for _ in 0..trials {
        let t0 = Instant::now();
        for _ in 0..steps {
            system.compute_forces();
        }
        best = best.min(t0.elapsed());
    }
    (system.len(), best)
}

#[test]
fn force_evaluation_scales_linearly_with_atom_count() {
    // 216 → 1728 atoms, an 8× increase at constant density.
    let small = time_force_evaluations(6, 6, 3);
    let large = time_force_evaluations(12, 6, 3);

    let n_ratio = large.0 as f64 / small.0 as f64;
    let t_ratio = large.1.as_secs_f64() / small.1.as_secs_f64();
    assert!(
        (n_ratio - 8.0).abs() < 0.01,
        "expected an 8× atom-count increase, got {n_ratio}"
    );

    assert!(
        t_ratio < 24.0,
        "force evaluation took {t_ratio:.1}× longer for {n_ratio:.0}× the atoms \
         ({:?} → {:?}); O(N) predicts ≈8×, O(N²) ≈64×",
        small.1,
        large.1
    );
}

/// The same claim across four sizes: time per atom must stay bounded rather
/// than growing with N.
#[test]
fn time_per_atom_stays_bounded() {
    let mut per_atom = Vec::new();
    for &n_side in &[6usize, 8, 10, 12] {
        let (n, elapsed) = time_force_evaluations(n_side, 4, 3);
        per_atom.push((n, elapsed.as_secs_f64() / n as f64));
    }
    let first = per_atom[0].1;
    let last = per_atom[per_atom.len() - 1].1;
    assert!(
        last / first < 3.0,
        "time per atom grew {:.1}× from {} to {} atoms — {per_atom:?}",
        last / first,
        per_atom[0].0,
        per_atom[per_atom.len() - 1].0
    );
}
