//! Validation of the XPBD solver against closed forms.
//!
//! Every test here compares against something derivable on paper, and prints
//! the measured number so a failure says *how far off* rather than just "no".

use phyz_math::Vec3;
use phyz_xpbd::{Constraint, ParticleSystem, XpbdSolver, tet_volume};

const G: f64 = 9.81;

/// A mass on a compliant distance constraint, run to static rest.
///
/// Returns the stretch beyond rest length.
fn hanging_mass_stretch(compliance: f64, mass: f64, substeps: usize, iterations: usize) -> f64 {
    let mut p = ParticleSystem::new();
    let anchor = p.add_pinned(Vec3::zeros());
    let bob = p.add(Vec3::new(0.0, -1.0, 0.0), mass);
    let mut cs = vec![Constraint::distance(anchor, bob, 1.0, compliance)];
    let solver = XpbdSolver {
        dt: 1.0 / 60.0,
        substeps,
        iterations,
        gravity: Vec3::new(0.0, -G, 0.0),
        // Strong damping: we want the static solution, not a bouncing spring.
        damping: 40.0,
    };
    for _ in 0..2000 {
        solver.step(&mut p, &mut cs);
    }
    (p.positions[bob] - p.positions[anchor]).norm() - 1.0
}

/// A single distance constraint's static stretch is `α · f`, exactly.
///
/// This is the defining property of compliance: it is the inverse stiffness of
/// a real spring, so at equilibrium `f = k·x = x/α`. A PBD "stiffness in
/// [0,1]" has no such closed form.
#[test]
fn distance_constraint_matches_hookes_law() {
    for &(alpha, m) in &[(1.0e-3, 1.0), (1.0e-2, 1.0), (1.0e-3, 5.0), (1.0e-4, 2.0)] {
        let measured = hanging_mass_stretch(alpha, m, 10, 4);
        let expected = alpha * m * G;
        let rel = (measured - expected).abs() / expected;
        assert!(
            rel < 1.0e-6,
            "α = {alpha:e}, m = {m}: stretch {measured:.12e}, expected {expected:.12e}, rel {rel:.3e}"
        );
    }
}

/// **XPBD's central claim over PBD**: the converged answer does not depend on
/// how many constraint iterations you run.
///
/// PBD fails this outright — each pass removes a fixed fraction of the
/// remaining error, so more passes mean a stiffer material. XPBD's multiplier
/// feedback makes the fixed point a property of `α` alone.
///
/// # Why this uses a chain and not a single constraint
///
/// On a **single** constraint this property is true for a reason that has
/// nothing to do with XPBD: one Gauss-Seidel sweep already solves a
/// one-constraint system exactly, so every later iteration is a no-op and any
/// iteration count returns bit-identical numbers. A PBD implementation with a
/// bug in exactly the place this test exists to guard would sail through it.
/// The test would report a perfect `0.0` spread and mean nothing — the same
/// trivial-agreement trap as comparing two engines on a model that does not
/// move.
///
/// Coupling is what makes iteration count bite: on a 20-link chain each sweep
/// propagates the multiplier one link further, so the *rate* of convergence
/// depends on the iteration count even though the *fixed point* must not.
/// Measured worst-case relative deviation from the 32-iteration answer:
///
/// ```text
/// iterations   1        2        3        4        8, 16, 32
/// deviation    2.6e-3   1.4e-5   7.1e-8   3.7e-10  bit-identical
/// ```
///
/// Each extra sweep cuts the remaining error by roughly 200×, and from eight
/// sweeps on the result stops changing in the last bit. That is the shape the
/// claim predicts: iterations buy convergence speed, `α` alone sets where it
/// converges to.
#[test]
fn converged_result_is_independent_of_iteration_count() {
    let reference = chain_depths(32);

    // The fixed point: once converged, more sweeps change nothing at all.
    for &iters in &[8, 16, 32] {
        let d = chain_depths(iters);
        for (k, (a, b)) in d.iter().zip(&reference).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "iterations = {iters}, mass {k}: {a:.17e} vs reference {b:.17e}"
            );
        }
    }

    // The approach to it: monotone, and fast. Anything that fails here is
    // either not converging or converging to an iteration-dependent answer,
    // and the printed sequence says which.
    let mut prev = f64::INFINITY;
    for &iters in &[1, 2, 3, 4] {
        let d = chain_depths(iters);
        let worst = d
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs() / b)
            .fold(0.0f64, f64::max);
        assert!(
            worst < prev,
            "iterations = {iters}: deviation {worst:.3e} did not improve on {prev:.3e}"
        );
        prev = worst;
    }
    assert!(
        prev < 1.0e-8,
        "four sweeps should be within 1e-8 of converged; got {prev:.3e}"
    );
}

/// Static depths of every mass in the 20-link hanging chain, at a given
/// constraint-iteration count. Shared by the iteration-independence test and
/// the closed-form equilibrium test so both describe the same system.
fn chain_depths(iterations: usize) -> Vec<f64> {
    let (n, m, alpha, rest) = (20usize, 0.5, 1.0e-3, 0.1);
    let mut p = ParticleSystem::new();
    let pin = p.add_pinned(Vec3::zeros());
    let mut idx = vec![pin];
    for k in 1..=n {
        idx.push(p.add(Vec3::new(0.0, -rest * k as f64, 0.0), m));
    }
    let mut cs: Vec<Constraint> = (0..n)
        .map(|j| Constraint::distance(idx[j], idx[j + 1], rest, alpha))
        .collect();
    let solver = XpbdSolver {
        dt: 1.0 / 60.0,
        substeps: 10,
        iterations,
        gravity: Vec3::new(0.0, -G, 0.0),
        damping: 40.0,
    };
    for _ in 0..4000 {
        solver.step(&mut p, &mut cs);
    }
    idx.iter().skip(1).map(|&i| -p.positions[i].y).collect()
}

/// The single-constraint case, kept for what it *does* establish: the solver
/// reaches a stable answer and the stretch is the Hooke's-law one at every
/// iteration count. It cannot establish iteration independence — see
/// `converged_result_is_independent_of_iteration_count` for why.
#[test]
fn single_constraint_stretch_is_stable_across_iteration_counts() {
    let alpha = 2.0e-3;
    let expected = alpha * 1.0 * G;
    for &iters in &[1, 2, 4, 8, 16, 32] {
        let s = hanging_mass_stretch(alpha, 1.0, 10, iters);
        let rel = (s - expected).abs() / expected;
        assert!(
            rel < 1.0e-6,
            "iterations = {iters}: stretch {s:.15e}, Hooke {expected:.15e} (rel {rel:.3e})"
        );
    }
}

/// The same, across substep counts. Substepping changes the integration error,
/// so this is a looser claim than the iteration one — but the static
/// equilibrium is still set by `α` and must not drift with `substeps`.
#[test]
fn converged_result_is_stable_across_substep_counts() {
    let alpha = 2.0e-3;
    let reference = hanging_mass_stretch(alpha, 1.0, 4, 1);
    for &subs in &[4, 8, 10, 20, 40] {
        let s = hanging_mass_stretch(alpha, 1.0, subs, 1);
        let rel = (s - reference).abs() / reference;
        assert!(
            rel < 1.0e-9,
            "substeps = {subs}: stretch {s:.15e} vs reference {reference:.15e} (rel {rel:.3e})"
        );
    }
}

/// A chain of `n` equal masses hanging from a pin settles to the analytic
/// static equilibrium.
///
/// Closed form: link `j` (1-based from the top) carries the weight of the
/// `n − j + 1` masses below it, so it stretches by `α·(n − j + 1)·m·g`. The
/// depth of mass `k` below the pin is the sum of the link lengths above it.
#[test]
fn hanging_chain_reaches_static_equilibrium() {
    let n = 20usize;
    let m = 0.5;
    let alpha = 1.0e-3;
    let rest = 0.1;

    let mut p = ParticleSystem::new();
    let pin = p.add_pinned(Vec3::zeros());
    let mut idx = vec![pin];
    for k in 1..=n {
        idx.push(p.add(Vec3::new(0.0, -rest * k as f64, 0.0), m));
    }
    let mut cs: Vec<Constraint> = (0..n)
        .map(|j| Constraint::distance(idx[j], idx[j + 1], rest, alpha))
        .collect();

    let solver = XpbdSolver {
        dt: 1.0 / 60.0,
        substeps: 10,
        iterations: 20,
        gravity: Vec3::new(0.0, -G, 0.0),
        damping: 40.0,
    };
    for _ in 0..4000 {
        solver.step(&mut p, &mut cs);
    }

    let mut depth = 0.0;
    let mut worst: f64 = 0.0;
    for (k, &node) in idx.iter().enumerate().skip(1) {
        // Link k spans mass k-1 → k and carries n − k + 1 masses.
        depth += rest + alpha * (n - k + 1) as f64 * m * G;
        let measured = -p.positions[node].y;
        worst = worst.max((measured - depth).abs());
        assert!(
            (measured - depth).abs() < 1.0e-5,
            "mass {k}: depth {measured:.12} m, analytic {depth:.12} m"
        );
    }
    assert!(worst < 1.0e-5, "worst node depth error {worst:.3e} m");
}

/// The chain converges towards the analytic equilibrium as the projection
/// budget grows — the honest statement of the Gauss–Seidel limitation, with
/// numbers.
#[test]
fn hanging_chain_error_falls_with_projection_budget() {
    let n = 20usize;
    let m = 0.5;
    let alpha = 1.0e-3;
    let rest = 0.1;

    let run = |iterations: usize| -> f64 {
        let mut p = ParticleSystem::new();
        let pin = p.add_pinned(Vec3::zeros());
        let mut idx = vec![pin];
        for k in 1..=n {
            idx.push(p.add(Vec3::new(0.0, -rest * k as f64, 0.0), m));
        }
        let mut cs: Vec<Constraint> = (0..n)
            .map(|j| Constraint::distance(idx[j], idx[j + 1], rest, alpha))
            .collect();
        let solver = XpbdSolver {
            dt: 1.0 / 60.0,
            substeps: 10,
            iterations,
            gravity: Vec3::new(0.0, -G, 0.0),
            damping: 40.0,
        };
        for _ in 0..4000 {
            solver.step(&mut p, &mut cs);
        }
        let mut depth = 0.0;
        let mut worst: f64 = 0.0;
        for (k, &node) in idx.iter().enumerate().skip(1) {
            depth += rest + alpha * (n - k + 1) as f64 * m * G;
            worst = worst.max((-p.positions[node].y - depth).abs());
        }
        worst
    };

    let e1 = run(1);
    let e20 = run(20);
    println!(
        "20-link chain worst node depth error: 1 iteration {e1:.3e} m, 20 iterations {e20:.3e} m"
    );
    assert!(
        e20 < e1 * 1.0e-2,
        "budget did not help: 1 iteration {e1:.3e} m, 20 iterations {e20:.3e} m"
    );
}

/// A tetrahedron under load preserves its volume to exactly the residual its
/// compliance predicts.
///
/// This is not a "small enough" test. At static equilibrium XPBD's multiplier
/// satisfies `C = −α·λ`, and `λ` is fixed by force balance on the one free
/// vertex: `λ = m g / |∇V_d|`, with `∇V_d = ½·(base area)·n̂ / 3` — here
/// `|∇V_d| = 1/6` for a unit right-triangle base. So the residual volume error
/// is `α·m·g/|∇V_d|`, and the test checks that number, not a tolerance.
#[test]
fn tetrahedron_preserves_volume_under_load() {
    let mut p = ParticleSystem::new();
    // Three pinned base vertices and one loaded apex: gravity pulls the apex
    // down and the volume constraint must resist the collapse.
    let a = p.add_pinned(Vec3::new(0.0, 0.0, 0.0));
    let b = p.add_pinned(Vec3::new(1.0, 0.0, 0.0));
    let c = p.add_pinned(Vec3::new(0.0, 0.0, 1.0));
    let d = p.add(Vec3::new(0.0, 1.0, 0.0), 10.0);

    let v0 = tet_volume(
        p.positions[a],
        p.positions[b],
        p.positions[c],
        p.positions[d],
    );
    let mut cs = vec![Constraint::volume(a, b, c, d, v0, 1.0e-9)];
    let solver = XpbdSolver {
        dt: 1.0 / 60.0,
        substeps: 10,
        iterations: 5,
        gravity: Vec3::new(0.0, -G, 0.0),
        damping: 40.0,
    };
    for _ in 0..1200 {
        solver.step(&mut p, &mut cs);
    }
    let v = tet_volume(
        p.positions[a],
        p.positions[b],
        p.positions[c],
        p.positions[d],
    );
    let alpha = 1.0e-9;
    let grad_norm = 1.0 / 6.0;
    let predicted = alpha * 10.0 * G / grad_norm; // |C| at equilibrium, m³
    let measured = (v - v0).abs();
    let rel = (measured - predicted).abs() / predicted;
    assert!(
        rel < 1.0e-3,
        "volume residual {measured:.12e} m³, compliance predicts {predicted:.12e} m³ \
         (rel {rel:.3e}); V = {v:.15e}, V₀ = {v0:.15e}"
    );
}

/// A softer tet visibly loses volume, and by the amount compliance predicts.
///
/// This is the negative control for the test above: if the volume constraint
/// were simply rigid, compliance would not matter, and it does.
#[test]
fn volume_loss_scales_with_compliance() {
    let run = |alpha: f64| -> f64 {
        let mut p = ParticleSystem::new();
        let a = p.add_pinned(Vec3::new(0.0, 0.0, 0.0));
        let b = p.add_pinned(Vec3::new(1.0, 0.0, 0.0));
        let c = p.add_pinned(Vec3::new(0.0, 0.0, 1.0));
        let d = p.add(Vec3::new(0.0, 1.0, 0.0), 10.0);
        let v0 = tet_volume(
            p.positions[a],
            p.positions[b],
            p.positions[c],
            p.positions[d],
        );
        let mut cs = vec![Constraint::volume(a, b, c, d, v0, alpha)];
        let solver = XpbdSolver {
            dt: 1.0 / 60.0,
            substeps: 10,
            iterations: 5,
            gravity: Vec3::new(0.0, -G, 0.0),
            damping: 40.0,
        };
        for _ in 0..1200 {
            solver.step(&mut p, &mut cs);
        }
        let v = tet_volume(
            p.positions[a],
            p.positions[b],
            p.positions[c],
            p.positions[d],
        );
        (v0 - v) / v0
    };
    let stiff = run(1.0e-9);
    let soft = run(1.0e-5);
    assert!(
        soft > stiff * 100.0,
        "compliance had no effect: stiff loss {stiff:.3e}, soft loss {soft:.3e}"
    );
}

/// Bending drives a folded triangle pair back to its flat rest angle.
///
/// Note the rest angle is `π`, not `0`. `C` measures the angle between the two
/// triangle normals as built from the *shared edge first*, so a flat pair whose
/// triangles wind oppositely about that edge — the usual case for a mesh where
/// the two apexes are on opposite sides — has antiparallel normals. Getting
/// this backwards is the classic way to end up with cloth that creases itself
/// flat, so it is worth stating rather than leaving the caller to discover.
#[test]
fn bending_restores_the_rest_dihedral_angle() {
    // Shared edge along X; apexes at ±Z, with the second apex folded up.
    let mut p = ParticleSystem::new();
    let a = p.add_pinned(Vec3::new(-0.5, 0.0, 0.0));
    let b = p.add_pinned(Vec3::new(0.5, 0.0, 0.0));
    let c = p.add(Vec3::new(0.0, 0.0, -1.0), 1.0);
    let d = p.add(Vec3::new(0.0, 0.6, 0.8), 1.0);

    let mut cs = vec![
        // Hold the triangles rigid so only the dihedral can change.
        Constraint::distance(a, c, (p.positions[a] - p.positions[c]).norm(), 0.0),
        Constraint::distance(b, c, (p.positions[b] - p.positions[c]).norm(), 0.0),
        Constraint::distance(a, d, (p.positions[a] - p.positions[d]).norm(), 0.0),
        Constraint::distance(b, d, (p.positions[b] - p.positions[d]).norm(), 0.0),
        Constraint::bending(a, b, c, d, std::f64::consts::PI, 1.0e-8),
    ];
    let solver = XpbdSolver {
        dt: 1.0 / 120.0,
        substeps: 10,
        iterations: 8,
        gravity: Vec3::zeros(), // isolate bending from gravity
        damping: 20.0,
    };
    for _ in 0..3000 {
        solver.step(&mut p, &mut cs);
    }

    let n1 = (p.positions[b] - p.positions[a])
        .cross(p.positions[c] - p.positions[a])
        .normalize();
    let n2 = (p.positions[b] - p.positions[a])
        .cross(p.positions[d] - p.positions[a])
        .normalize();
    let angle = n1.dot(n2).clamp(-1.0, 1.0).acos();
    let err = (angle - std::f64::consts::PI).abs();
    assert!(
        err < 1.0e-3,
        "dihedral angle {angle:.9} rad, rest π (error {err:.3e} rad); started at 2.4981 rad"
    );
}

/// A compliant attachment holds a mass at `α·m·g` below its anchor.
#[test]
fn attachment_holds_at_the_compliance_predicted_offset() {
    let alpha = 5.0e-4;
    let m = 2.0;
    let mut p = ParticleSystem::new();
    let bob = p.add(Vec3::new(0.0, 0.0, 0.0), m);
    let mut cs = vec![Constraint::attachment(bob, Vec3::zeros(), alpha)];
    let solver = XpbdSolver {
        dt: 1.0 / 60.0,
        substeps: 10,
        iterations: 4,
        gravity: Vec3::new(0.0, -G, 0.0),
        damping: 40.0,
    };
    for _ in 0..2000 {
        solver.step(&mut p, &mut cs);
    }
    let sag = -p.positions[bob].y;
    let expected = alpha * m * G;
    let rel = (sag - expected).abs() / expected;
    assert!(rel < 1.0e-6, "sag {sag:.12e}, expected {expected:.12e}");
}

/// Two identical runs produce bit-identical state.
///
/// Determinism is a product guarantee across phyz, so it is asserted on bits,
/// not on a tolerance — a tolerance would pass even if the solver had picked
/// up a nondeterministic reduction.
#[test]
fn two_runs_are_bit_identical() {
    fn run() -> (Vec<Vec3>, Vec<Vec3>) {
        let mut p = ParticleSystem::new();
        let pin = p.add_pinned(Vec3::zeros());
        let mut idx = vec![pin];
        for k in 1..=12 {
            // Deliberately irregular positions so the trajectory is not
            // symmetric enough to hide an ordering difference.
            let f = k as f64;
            idx.push(p.add(Vec3::new(0.03 * f, -0.1 * f, 0.017 * f * f), 0.3 + 0.01 * f));
        }
        let mut cs: Vec<Constraint> = (0..12)
            .map(|j| Constraint::distance(idx[j], idx[j + 1], 0.1, 1.0e-4))
            .collect();
        for j in 0..10 {
            cs.push(Constraint::bending(
                idx[j],
                idx[j + 1],
                idx[j + 2],
                idx[j + 3.min(12 - j)],
                0.0,
                1.0e-6,
            ));
        }
        let solver = XpbdSolver {
            dt: 1.0 / 60.0,
            substeps: 8,
            iterations: 3,
            gravity: Vec3::new(0.3, -G, -0.2),
            damping: 0.5,
        };
        for _ in 0..500 {
            solver.step(&mut p, &mut cs);
        }
        (p.positions.clone(), p.velocities.clone())
    }

    let (x1, v1) = run();
    let (x2, v2) = run();
    let bits = |v: Vec3| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()];
    for i in 0..x1.len() {
        assert_eq!(
            bits(x1[i]),
            bits(x2[i]),
            "position {i} differs: {:?} vs {:?}",
            x1[i],
            x2[i]
        );
        assert_eq!(bits(v1[i]), bits(v2[i]), "velocity {i} differs");
    }
    assert!(
        x1.iter().all(|p| p.norm().is_finite()),
        "run went non-finite"
    );
}

/// A pinned particle never moves and never reports a velocity.
#[test]
fn pinned_particles_do_not_move() {
    let mut p = ParticleSystem::new();
    let pin = p.add_pinned(Vec3::new(1.0, 2.0, 3.0));
    let bob = p.add(Vec3::new(1.0, 1.0, 3.0), 1.0);
    let mut cs = vec![Constraint::distance(pin, bob, 1.0, 0.0)];
    let solver = XpbdSolver::default();
    for _ in 0..100 {
        solver.step(&mut p, &mut cs);
    }
    assert_eq!(p.positions[pin], Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(p.velocities[pin], Vec3::zeros());
}

/// A rigid (`α = 0`) chain in free fall conserves its link lengths.
#[test]
fn rigid_constraints_hold_length_in_free_fall() {
    let mut p = ParticleSystem::new();
    let mut idx = Vec::new();
    for k in 0..8 {
        idx.push(p.add(Vec3::new(0.1 * k as f64, 0.0, 0.0), 1.0));
    }
    let mut cs: Vec<Constraint> = (0..7)
        .map(|j| Constraint::distance(idx[j], idx[j + 1], 0.1, 0.0))
        .collect();
    let solver = XpbdSolver::default();
    for _ in 0..300 {
        solver.step(&mut p, &mut cs);
    }
    for j in 0..7 {
        let len = (p.positions[idx[j + 1]] - p.positions[idx[j]]).norm();
        assert!((len - 0.1).abs() < 1.0e-12, "link {j} length {len:.15}");
    }
}

/// Reports the measured numbers quoted in the crate docs. Not an assertion of
/// quality — a guard that the documented figures stay true.
#[test]
fn documented_numbers_are_current() {
    let alpha = 2.0e-3;
    let r1 = hanging_mass_stretch(alpha, 1.0, 10, 1);
    let r32 = hanging_mass_stretch(alpha, 1.0, 10, 32);
    let spread = (r32 - r1).abs() / r1;
    let hooke = (hanging_mass_stretch(1.0e-3, 1.0, 10, 4) - 1.0e-3 * G).abs() / (1.0e-3 * G);
    println!("iteration-count spread (1 vs 32): {spread:.3e}");
    println!("hooke relative error: {hooke:.3e}");
    assert!(spread < 1.0e-12 && hooke < 1.0e-6);
}
