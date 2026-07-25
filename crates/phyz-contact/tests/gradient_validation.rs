//! Finite-difference validation of the contact gradients (design doc §6.5).
//!
//! Pattern follows phyz-regge and phyz-quantum: central differences against
//! the analytic value over an `h` sweep. The usual "error shrinks with `h`"
//! check is a wrong-formula detector when truncation error dominates — but
//! within a fixed active set the contact KKT system is *linear* in the free
//! velocity, so `df/db` is constant and the difference is exact at every `h`.
//! Here the sweep is instead dominated by the `eps/h` cancellation floor, and
//! the meaningful assertion is agreement at machine precision, which a wrong
//! formula cannot reach at any `h`.
//!
//! The last two tests are deliberately **negative**: they assert the known
//! limits of the method rather than papering over them.

use phyz_contact::gradient::{ContactRegime, classify, friction_sensitivity, impulse_sensitivity};
use phyz_contact::{ContactProblem, ContactRow, ContactSolverConfig, solve_contacts};

fn problem(n: usize, mu: f64, b: &[f64], coupling: f64) -> ContactProblem {
    let dim = 3 * n;
    let mut delassus = vec![0.0; dim * dim];
    for i in 0..dim {
        delassus[i * dim + i] = 1.0;
    }
    if n > 1 && coupling != 0.0 {
        delassus[3] = coupling;
        delassus[3 * dim] = coupling;
    }
    ContactProblem {
        n,
        delassus,
        free_velocity: b.to_vec(),
        rows: vec![
            ContactRow {
                mu,
                restitution: 0.0,
                depth: 1e-4,
            };
            n
        ],
    }
}

/// Central-difference `df/db[j]`, all impulse components.
fn fd_wrt_b(p: &ContactProblem, cfg: &ContactSolverConfig, j: usize, h: f64) -> Vec<f64> {
    let mut plus = p.clone();
    plus.free_velocity[j] += h;
    let mut minus = p.clone();
    minus.free_velocity[j] -= h;
    let sp = solve_contacts(&plus, cfg);
    let sm = solve_contacts(&minus, cfg);
    sp.impulses
        .iter()
        .zip(&sm.impulses)
        .flat_map(|(a, b)| {
            let d = (*a - *b) / (2.0 * h);
            [d.x, d.y, d.z]
        })
        .collect()
}

/// Sticking contact: fully smooth, so the gradient should be near-exact.
#[test]
fn sticking_contact_gradient_matches_finite_difference() {
    let cfg = ContactSolverConfig::gradients();
    let p = problem(1, 0.8, &[-1.0, 0.05, -0.02], 0.0);
    let sol = solve_contacts(&p, &cfg);
    assert!(sol.converged);
    assert_eq!(classify(&p, &sol, 1e-7)[0], ContactRegime::Sticking);

    let sens = impulse_sensitivity(&p, &sol, &cfg).expect("sensitivity available");
    let dim = 3;

    for j in 0..dim {
        let mut best = f64::INFINITY;
        let mut errs = Vec::new();
        for k in 0..4 {
            let h = 1e-3 / 10f64.powi(k);
            let fd = fd_wrt_b(&p, &cfg, j, h);
            let err: f64 = (0..dim)
                .map(|row| (fd[row] - sens[row * dim + j]).abs())
                .fold(0.0, f64::max);
            errs.push(err);
            best = best.min(err);
        }
        // A sticking contact's KKT system is *linear* in `b`, so `df/db` is
        // constant and the central difference is exact up to roundoff at every
        // `h`. The usual "error shrinks with h" check is inapplicable here by
        // construction — what actually dominates the sweep is the `eps/h`
        // cancellation floor, which grows as `h` shrinks. The meaningful
        // assertion is that the agreement is at machine precision, which a
        // wrong formula cannot reach at any `h`.
        assert!(
            best < 1e-11,
            "column {j}: best abs error {best} (sweep {errs:?})"
        );
        assert!(
            errs[0] < 1e-9,
            "column {j}: error at the largest h should already be exact: {errs:?}"
        );
    }
}

/// Sliding contact: the tangential rows follow the cone boundary, which is a
/// different branch of the KKT system entirely.
#[test]
fn sliding_contact_gradient_matches_finite_difference() {
    let cfg = ContactSolverConfig::gradients();
    // Large tangential drift relative to mu: definitely sliding.
    let p = problem(1, 0.3, &[-1.0, 2.0, 0.0], 0.0);
    let sol = solve_contacts(&p, &cfg);
    assert!(sol.converged);
    assert_eq!(classify(&p, &sol, 1e-7)[0], ContactRegime::Sliding);

    let sens = impulse_sensitivity(&p, &sol, &cfg).expect("sensitivity available");
    let dim = 3;

    // Normal column is the well-conditioned one; check it tightly.
    let fd = fd_wrt_b(&p, &cfg, 0, 1e-6);
    for row in 0..dim {
        let a = sens[row * dim];
        assert!(
            (fd[row] - a).abs() < 1e-4,
            "row {row}: analytic {a} vs fd {}",
            fd[row]
        );
    }
}

/// Coupled contacts: the off-diagonal response is the part a per-contact
/// gradient would miss entirely.
#[test]
fn coupled_contact_gradient_matches_finite_difference() {
    let cfg = ContactSolverConfig::gradients();
    let p = problem(2, 0.9, &[-1.0, 0.0, 0.0, -1.0, 0.0, 0.0], 0.5);
    let sol = solve_contacts(&p, &cfg);
    assert!(sol.converged);

    let sens = impulse_sensitivity(&p, &sol, &cfg).expect("sensitivity available");
    let _dim = 6;

    // Perturbing contact 1's normal free velocity must move contact 0's
    // impulse — the cross term.
    // Row 0 (contact 0's normal impulse), column 3 (contact 1's normal b).
    let cross = sens[3];
    assert!(
        cross.abs() > 1e-3,
        "expected non-zero cross-sensitivity, got {cross}"
    );

    let fd = fd_wrt_b(&p, &cfg, 3, 1e-6);
    assert!(
        (fd[0] - cross).abs() < 1e-4,
        "cross term: analytic {cross} vs fd {}",
        fd[0]
    );
}

/// `df/dmu` is structurally zero for a sticking contact — the impulse is
/// strictly inside the cone, so moving the boundary does not move it. A finite
/// difference reproduces that exactly, which is a strong check on the
/// derivation.
#[test]
fn friction_gradient_is_zero_when_sticking_and_nonzero_when_sliding() {
    let cfg = ContactSolverConfig::gradients();

    let stick = problem(1, 0.8, &[-1.0, 0.05, 0.0], 0.0);
    let s = solve_contacts(&stick, &cfg);
    assert_eq!(classify(&stick, &s, 1e-7)[0], ContactRegime::Sticking);
    let g = friction_sensitivity(&stick, &s, &cfg).unwrap();
    for (row, v) in g.iter().enumerate() {
        assert!(v.abs() < 1e-12, "sticking df/dmu row {row} = {v}");
    }
    // Finite difference agrees: the solution does not move.
    let mut bumped = stick.clone();
    bumped.rows[0].mu += 1e-4;
    let s2 = solve_contacts(&bumped, &cfg);
    assert!((s2.impulses[0] - s.impulses[0]).norm() < 1e-12);

    let slide = problem(1, 0.3, &[-1.0, 2.0, 0.0], 0.0);
    let s = solve_contacts(&slide, &cfg);
    assert_eq!(classify(&slide, &s, 1e-7)[0], ContactRegime::Sliding);
    let g = friction_sensitivity(&slide, &s, &cfg).unwrap();
    let tangential = g[1];
    assert!(
        tangential.abs() > 1e-3,
        "sliding df_u/dmu should be non-zero, got {tangential}"
    );

    // And it matches a central difference.
    let h = 1e-6;
    let mut plus = slide.clone();
    plus.rows[0].mu += h;
    let mut minus = slide.clone();
    minus.rows[0].mu -= h;
    let fd = (solve_contacts(&plus, &cfg).impulses[0].y
        - solve_contacts(&minus, &cfg).impulses[0].y)
        / (2.0 * h);
    assert!(
        (fd - tangential).abs() / fd.abs().max(1e-9) < 1e-3,
        "df_u/dmu: analytic {tangential} vs fd {fd}"
    );
}

/// A separating contact carries no impulse and no sensitivity.
#[test]
fn separating_contact_has_zero_sensitivity() {
    let cfg = ContactSolverConfig::gradients();
    // Moving apart: no contact impulse.
    let p = problem(1, 0.5, &[1.0, 0.0, 0.0], 0.0);
    let sol = solve_contacts(&p, &cfg);
    assert_eq!(classify(&p, &sol, 1e-7)[0], ContactRegime::Separating);
    assert!(sol.impulses[0].norm() < 1e-12);

    let sens = impulse_sensitivity(&p, &sol, &cfg).unwrap();
    // Identity rows with a zero RHS: the solution stays put.
    for v in &sens {
        assert!(v.abs() < 1e-12, "separating sensitivity {v}");
    }
}

/// An unconverged solve must refuse to produce a gradient rather than return a
/// confident wrong number. The IFT anchors on a KKT point; a truncated iterate
/// is not one.
#[test]
fn unconverged_solve_yields_no_gradient() {
    let cfg = ContactSolverConfig {
        max_iterations: 1,
        tolerance: 1e-16,
        ..ContactSolverConfig::gradients()
    };
    let p = problem(2, 0.5, &[-1.0, 0.3, 0.0, -1.0, 0.3, 0.0], 0.9);
    let sol = solve_contacts(&p, &cfg);
    assert!(!sol.converged, "test needs an unconverged solve");
    assert!(impulse_sensitivity(&p, &sol, &cfg).is_none());
}

// ---------------------------------------------------------------------------
// Negative tests: the documented limits, asserted rather than hidden.
// ---------------------------------------------------------------------------

/// At a stick-slip transition the true derivative does not exist, and the
/// analytic gradient (which fixes the regime from the primal) disagrees with a
/// finite difference that straddles the switch.
///
/// Design doc §2.3: this is a real bias, not a bug, and it is the reason the
/// gradient preset exists. The test pins the disagreement so a regression that
/// silently "fixes" it by smoothing the physics is visible.
#[test]
fn stick_slip_transition_is_a_documented_gradient_discontinuity() {
    let cfg = ContactSolverConfig::gradients();
    let mu = 0.5;

    // Find a tangential drift that sits right at the cone boundary.
    let mut lo = 0.0;
    let mut hi = 5.0;
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let p = problem(1, mu, &[-1.0, mid, 0.0], 0.0);
        let s = solve_contacts(&p, &cfg);
        if classify(&p, &s, 1e-9)[0] == ContactRegime::Sliding {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let critical = 0.5 * (lo + hi);

    let p = problem(1, mu, &[-1.0, critical, 0.0], 0.0);
    let sol = solve_contacts(&p, &cfg);
    let sens = impulse_sensitivity(&p, &sol, &cfg).unwrap();
    let analytic = sens[3 + 1]; // d f_u / d b_u

    // A wide finite difference straddles the regime switch.
    let h = 0.05;
    let fd = fd_wrt_b(&p, &cfg, 1, h)[1];

    assert!(
        (fd - analytic).abs() > 1e-3,
        "expected the transition to be non-smooth; analytic {analytic}, fd {fd}"
    );
}

/// Raising the regularization must make the contact response smoother — the
/// knob the design leans on for gradient quality has to actually do that.
#[test]
fn regularization_smooths_the_response() {
    let spread = |reg: f64| {
        let cfg = ContactSolverConfig {
            regularization: reg,
            ..ContactSolverConfig::gradients()
        };
        // Sample d f_n / d b_n either side of the separating/contact switch.
        let sample = |b_n: f64| {
            let p = problem(1, 0.5, &[b_n, 0.0, 0.0], 0.0);
            let s = solve_contacts(&p, &cfg);
            s.impulses[0].x
        };
        let h = 1e-3;
        let left = (sample(-h) - sample(-2.0 * h)) / h;
        let right = (sample(2.0 * h) - sample(h)) / h;
        (left - right).abs()
    };

    // The jump in slope across the switch is what a gradient has to live with.
    // It must not grow as the contact is softened.
    let stiff = spread(1e-8);
    let soft = spread(1e-1);
    assert!(
        soft <= stiff + 1e-9,
        "softening should not worsen the kink: stiff {stiff}, soft {soft}"
    );
}
