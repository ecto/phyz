//! Does the solver-level differential match a finite difference of the solve?
//!
//! [`contact_solve_differential`] re-executes the solve carrying a parameter
//! differential alongside the primal, so what it returns is the derivative of
//! *the iterate the solver produced* — converged or not. That is a much
//! stronger claim than the implicit-function-theorem gradient next door makes,
//! and it is falsifiable in the most direct way there is: perturb the problem,
//! re-solve, and difference.
//!
//! These tests deliberately run the solve **short of convergence** as well as
//! to it. An IFT gradient is only correct in the first case; if this one is
//! correct in both, it is differentiating the algorithm rather than a fixed
//! point it hoped the algorithm reached.
//!
//! Everything here is cold-started, which is what makes the comparison exact:
//! a warm start's seed is itself a function of the parameters, and this API
//! takes `d_initial` from the caller rather than inventing it.

use phyz_contact::{
    ContactProblem, ContactRow, ContactSolverConfig, contact_solve_differential,
    solve_contacts_warm,
};
use phyz_math::Vec3;

/// A deterministic, strictly diagonally dominant (hence SPD) Delassus operator
/// with genuine off-diagonal coupling, plus a free velocity that drives every
/// contact into load. Nothing random: the numbers have to be reproducible for a
/// failure to be debuggable.
fn problem(n: usize, mu: f64) -> ContactProblem {
    let dim = 3 * n;
    let mut delassus = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let (fi, fj) = (i as f64, j as f64);
            // Symmetric by construction, and the diagonal dominates.
            delassus[i * dim + j] = if i == j {
                2.0 + 0.35 * fi
            } else {
                0.12 * ((fi * 0.7).sin() * (fj * 0.7).sin())
            };
        }
    }
    let free_velocity: Vec<f64> = (0..dim)
        .map(|i| {
            let f = i as f64;
            if i % 3 == 0 {
                // Approaching: drives a positive normal impulse.
                -0.4 - 0.05 * f
            } else {
                0.15 * (f * 1.3).cos()
            }
        })
        .collect();
    ContactProblem {
        n,
        delassus,
        free_velocity,
        rows: (0..n)
            .map(|c| ContactRow {
                mu,
                bias: 0.01 * (c as f64 + 1.0),
                ..Default::default()
            })
            .collect(),
        bodies: (0..n).map(|c| (c, usize::MAX)).collect(),
    }
}

/// One direction in parameter space: a symmetric `dA` and a `d(b - e_n bias)`.
///
/// Symmetry matters — an asymmetric `dA` is not a direction the Delassus
/// operator can actually move in, and testing along one would validate
/// arithmetic the physics never exercises.
fn direction(n: usize) -> (Vec<f64>, Vec<f64>) {
    let dim = 3 * n;
    let mut d_apr = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let v = 0.05 * ((i * 31 + j * 17) as f64 * 0.37).sin();
            d_apr[i * dim + j] = v;
            d_apr[j * dim + i] = v;
        }
    }
    let dc = (0..dim)
        .map(|i| 0.08 * ((i * 13) as f64 * 0.29).cos())
        .collect();
    (d_apr, dc)
}

/// Apply `eps` along the direction. The impedance is left at `1.0` so the
/// regularizer is the constant config floor and `d(A + R) = dA` exactly —
/// keeping this test about the solve rather than about `regularization_diag`.
fn perturbed(base: &ContactProblem, d_apr: &[f64], dc: &[f64], eps: f64) -> ContactProblem {
    let mut p = base.clone();
    for (a, da) in p.delassus.iter_mut().zip(d_apr) {
        *a += eps * da;
    }
    // `dc` is `d(b - e_n bias)`; put all of it on `b` and none on `bias`, which
    // is one valid split and the one with no extra bookkeeping.
    for (b, db) in p.free_velocity.iter_mut().zip(dc) {
        *b += eps * db;
    }
    p
}

/// Central difference of the solved impulses along the direction.
fn fd(
    base: &ContactProblem,
    cfg: &ContactSolverConfig,
    d_apr: &[f64],
    dc: &[f64],
    eps: f64,
) -> Vec<Vec3> {
    let up = solve_contacts_warm(&perturbed(base, d_apr, dc, eps), cfg, &[]);
    let dn = solve_contacts_warm(&perturbed(base, d_apr, dc, -eps), cfg, &[]);
    up.impulses
        .iter()
        .zip(&dn.impulses)
        .map(|(a, b)| (*a - *b) * (1.0 / (2.0 * eps)))
        .collect()
}

/// Worst relative disagreement between two impulse differentials, scaled by the
/// largest component so a near-zero row cannot dominate the ratio.
fn worst_rel(a: &[Vec3], b: &[Vec3]) -> f64 {
    let scale = a.iter().chain(b).fold(1e-12f64, |m, v| {
        m.max(v.x.abs()).max(v.y.abs()).max(v.z.abs())
    });
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| {
        let d = *x - *y;
        m.max(d.x.abs().max(d.y.abs()).max(d.z.abs()))
    }) / scale
}

fn check(cfg: ContactSolverConfig, n: usize, mu: f64, tol: f64, label: &str) {
    let p = problem(n, mu);
    let (d_apr, dc) = direction(n);
    let (sol, df) = contact_solve_differential(&p, &cfg, &[], &[], &d_apr, &dc);

    // The re-execution must reproduce the primal exactly, or the branches the
    // derivative followed are not the branches the solve took.
    let plain = solve_contacts_warm(&p, &cfg, &[]);
    assert_eq!(
        sol.iterations, plain.iterations,
        "{label}: differentiated run took a different number of iterations"
    );
    for (a, b) in sol.impulses.iter().zip(&plain.impulses) {
        assert_eq!(a.x, b.x, "{label}: primal diverged under differentiation");
        assert_eq!(a.y, b.y, "{label}: primal diverged under differentiation");
        assert_eq!(a.z, b.z, "{label}: primal diverged under differentiation");
    }

    // Richardson: the central difference is O(h^2), so the best agreement sits
    // at a middling h. Take the best of a small sweep rather than betting on one.
    let best = [1e-4, 1e-5, 1e-6]
        .iter()
        .map(|&h| worst_rel(&df, &fd(&p, &cfg, &d_apr, &dc, h)))
        .fold(f64::INFINITY, f64::min);
    assert!(
        best < tol,
        "{label}: solver-level differential disagrees with finite differences \
         by {best:.3e} (converged {}, {} iterations, residual {:.3e})",
        sol.converged,
        sol.iterations,
        sol.residual
    );
}

/// **Converged.** Where the solve reaches a KKT point the solver-level answer
/// must agree with the finite difference — and with the IFT gradient, which is
/// also correct here. This is the easy case and the one that would catch a
/// plain sign error.
#[test]
fn it_matches_finite_differences_at_a_converged_solve() {
    let cfg = ContactSolverConfig {
        tolerance: 1e-14,
        max_iterations: 4000,
        ..ContactSolverConfig::gradients()
    };
    check(cfg, 5, 0.7, 1e-9, "converged, sticking and sliding mixed");
}

/// **Truncated — the case the implicit function theorem gets wrong.** Two
/// sweeps, nowhere near a fixed point. The iterate is still a perfectly
/// well-defined function of the parameters and the finite difference measures
/// it; an IFT gradient has no anchor here at all.
#[test]
fn it_matches_finite_differences_at_a_deliberately_truncated_solve() {
    let cfg = ContactSolverConfig {
        tolerance: 0.0,
        max_iterations: 2,
        newton: false,
        ..ContactSolverConfig::gradients()
    };
    check(cfg, 5, 0.7, 1e-9, "two sweeps, unconverged");
}

/// **Truncated with the Newton stage live.** The dense KKT solve and its line
/// search are on the path, so this exercises `newton_step_diff` — including the
/// slip-direction rotation in the sliding pin rows, which is the term most
/// likely to be silently dropped.
#[test]
fn it_matches_finite_differences_through_the_newton_stage() {
    let cfg = ContactSolverConfig {
        tolerance: 0.0,
        max_iterations: 40,
        ..ContactSolverConfig::gradients()
    };
    check(cfg, 5, 0.7, 1e-9, "40 iterations with Newton, unconverged");
}

/// **Everything sliding.** A low `mu` puts every contact on the cone boundary,
/// so the disc clamp's derivative — the `s (I - t_hat t_hat^T)` projector — is
/// on every tangential row rather than a couple of them.
#[test]
fn it_matches_finite_differences_when_every_contact_slides() {
    let cfg = ContactSolverConfig {
        tolerance: 1e-14,
        max_iterations: 4000,
        ..ContactSolverConfig::gradients()
    };
    check(cfg, 4, 0.02, 1e-9, "all sliding");
}

/// **A separating contact in the set.** `f_n = max(0, .)` pins one contact's
/// whole row, and the derivative has to be pinned with it. A high bias on the
/// others keeps the problem non-trivial around it.
#[test]
fn it_matches_finite_differences_with_a_separating_contact() {
    let cfg = ContactSolverConfig {
        tolerance: 1e-14,
        max_iterations: 4000,
        ..ContactSolverConfig::gradients()
    };
    let mut p = problem(4, 0.6);
    // Drive contact 2 firmly apart: its normal impulse is zero at the solution.
    p.free_velocity[6] = 3.0;
    let (d_apr, dc) = direction(4);
    let (sol, df) = contact_solve_differential(&p, &cfg, &[], &[], &d_apr, &dc);
    assert!(
        sol.impulses[2].x == 0.0,
        "the test needs contact 2 separating; it carries {:e}",
        sol.impulses[2].x
    );
    let best = [1e-4, 1e-5, 1e-6]
        .iter()
        .map(|&h| worst_rel(&df, &fd(&p, &cfg, &d_apr, &dc, h)))
        .fold(f64::INFINITY, f64::min);
    assert!(best < 1e-9, "separating contact: disagreement {best:.3e}");
}
