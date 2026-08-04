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
                ..Default::default()
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

/// `df/ddepth` — the sensitivity channel that position stabilization added.
///
/// Penetration now enters the solve through the solref bias, so a trajectory
/// adjoint that only contracted through `df/db` would report that penetration
/// has no effect on the impulses. It has exactly one, and this pins it against
/// a central difference taken through the *whole* pipeline: perturb the depth,
/// rebuild the rows from the material, re-solve.
///
/// The depths here are past `solimp.width`, where the impedance is pinned at
/// `dmax`. That is the regime the analytic form claims to cover — the
/// deliberately excluded second-order path is the impedance's own depth
/// dependence, which only exists inside the sigmoid's width.
#[test]
fn depth_gradient_matches_finite_difference() {
    use phyz_contact::ContactMaterial;
    use phyz_contact::gradient::depth_sensitivity;

    let cfg = ContactSolverConfig::gradients();
    let dt = 1e-3;
    let material = ContactMaterial {
        friction: 0.9,
        ..ContactMaterial::default()
    };
    let depth = 5e-3; // Well past solimp.width, so `d` is pinned at dmax.

    let build = |d: f64| {
        let mut p = problem(1, material.friction, &[-0.4, 0.02, 0.0], 0.0);
        p.rows[0] = ContactRow::from_material(&material, d, dt, 0.0);
        p
    };

    let p = build(depth);
    let sol = solve_contacts(&p, &cfg);
    assert!(sol.converged);
    assert_eq!(
        classify(&p, &sol, 1e-7)[0],
        ContactRegime::Sticking,
        "this fixture is meant to exercise the sticking branch"
    );

    let sens = depth_sensitivity(&p, &sol, &cfg, &material.solref, dt).expect("converged");

    for h in [1e-5, 1e-6, 1e-7] {
        let plus = solve_contacts(&build(depth + h), &cfg);
        let minus = solve_contacts(&build(depth - h), &cfg);
        let fd = (plus.impulses[0] - minus.impulses[0]) / (2.0 * h);
        for (k, want) in [fd.x, fd.y, fd.z].iter().enumerate() {
            let got = sens[k * p.n];
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                "h={h}: d f[{k}]/d depth analytic {got} vs finite difference {want}"
            );
        }
    }

    // Deeper penetration must push harder, not less.
    assert!(
        sens[0] > 0.0,
        "d f_n / d depth = {} should be positive",
        sens[0]
    );
}

/// A separating contact carries no impulse, so its depth cannot matter either.
#[test]
fn depth_gradient_is_zero_when_separating() {
    use phyz_contact::ContactMaterial;
    use phyz_contact::gradient::depth_sensitivity;

    let cfg = ContactSolverConfig::simulation();
    let material = ContactMaterial::default();
    let dt = 1e-3;
    // Moving apart fast enough that no bias this small closes the gap.
    let mut p = problem(1, 0.5, &[2.0, 0.0, 0.0], 0.0);
    p.rows[0] = ContactRow::from_material(&material, 1e-5, dt, 0.0);
    let sol = solve_contacts(&p, &cfg);
    assert_eq!(classify(&p, &sol, 1e-7)[0], ContactRegime::Separating);

    let sens = depth_sensitivity(&p, &sol, &cfg, &material.solref, dt).expect("converged");
    for (k, v) in sens.iter().enumerate() {
        assert_eq!(*v, 0.0, "separating contact has a depth sensitivity at {k}");
    }
}

/// The depth sensitivity **inside the contact margin band**, where the
/// stabilization bias is identically zero and the impedance is the only channel
/// from depth to force.
///
/// This is the region the previous form of `depth_sensitivity` reported as
/// exactly zero: it bailed on `depth <= 0.0`, which is every contact in the
/// band. The force across the band is not flat — it ramps from most of body
/// weight to nothing over `margin` — so "zero" was wrong by the entire
/// derivative, not by the documented few percent.
///
/// It matters more than a normal gap in coverage because the band is precisely
/// where contact activation is differentiable. A contact-timing gradient for
/// system identification lives here and nowhere else.
#[test]
fn depth_gradient_inside_the_margin_matches_finite_difference() {
    use phyz_contact::ContactMaterial;
    use phyz_contact::gradient::depth_sensitivity;

    let cfg = ContactSolverConfig::gradients();
    let dt = 1e-3;
    let material = ContactMaterial {
        friction: 0.9,
        ..ContactMaterial::default()
    };
    let margin = material.margin;
    assert!(margin > 0.0);

    let build = |d: f64| {
        let mut p = problem(1, material.friction, &[-0.4, 0.02, 0.0], 0.0);
        p.rows[0] = ContactRow::from_material(&material, d, dt, 0.0);
        p
    };

    let mut worst = 0.0f64;
    // Fractions of the way into the band, from just separated to most of the
    // way out. Stopping short of the very edge is deliberate and documented
    // below.
    for frac in [0.05, 0.15, 0.3, 0.5, 0.65, 0.8] {
        let depth = -frac * margin;
        let p = build(depth);
        let sol = solve_contacts(&p, &cfg);
        assert!(sol.converged);
        assert_ne!(
            classify(&p, &sol, 1e-7)[0],
            ContactRegime::Separating,
            "gap {frac} of the band should still carry load"
        );

        let sens = depth_sensitivity(&p, &sol, &cfg, &material.solref, dt).expect("converged");
        assert!(
            sens[0] > 0.0,
            "at gap {frac} of the band, closing the gap must increase the \
             normal impulse; got d f_n / d depth = {}",
            sens[0]
        );

        for h in [1e-7, 1e-8] {
            let plus = solve_contacts(&build(depth + h), &cfg);
            let minus = solve_contacts(&build(depth - h), &cfg);
            assert!(plus.converged && minus.converged);
            let fd = (plus.impulses[0] - minus.impulses[0]) / (2.0 * h);
            for (k, want) in [fd.x, fd.y, fd.z].iter().enumerate() {
                let got = sens[k * p.n];
                let rel = (got - want).abs() / want.abs().max(1e-9);
                worst = worst.max(rel);
                assert!(
                    rel < 1e-5,
                    "gap {frac} of band, h={h}: d f[{k}]/d depth analytic {got} \
                     vs finite difference {want} (rel {rel:.3e})"
                );
            }
        }
    }
    // Printed so the PR can quote a number rather than a tolerance.
    println!("worst relative FD error inside the margin band: {worst:.3e}");
}

/// Straddling `depth = 0`: the sensitivity has a **hinge** there, and this
/// test pins it to its exact analytic cause rather than papering over it.
///
/// The impedance channel is `C^1` across zero — both one-sided derivatives of
/// `impedance_at` are zero there — so it contributes nothing to the jump. The
/// *bias* channel is what hinges: `bias = d * erp * max(depth, 0) / dt`, whose
/// slope is `0` below zero and `d * erp / dt` above it. That is a `max()`, so
/// the bias is `C^0` but not `C^1`, and it has been that way since position
/// stabilization landed — it is not something the contact margin introduced.
///
/// This is a kink, not a cliff, and the distinction is the whole point of the
/// margin work: the *force* stays continuous across zero (a hinge in the
/// derivative), whereas the zero-margin contact-set cutoff was a discontinuity
/// in the force itself. An optimizer handles the former the way it handles any
/// ReLU; the latter it cannot see coming at all.
///
/// So the assertion is not "no jump". It is that each one-sided derivative is
/// individually correct, and that their difference is exactly the bias hinge.
#[test]
fn depth_gradient_has_a_documented_hinge_at_zero_depth() {
    use phyz_contact::ContactMaterial;
    use phyz_contact::gradient::depth_sensitivity;

    let cfg = ContactSolverConfig::gradients();
    let dt = 1e-3;
    let material = ContactMaterial {
        friction: 0.9,
        ..ContactMaterial::default()
    };
    let build = |d: f64| {
        let mut p = problem(1, material.friction, &[-0.4, 0.02, 0.0], 0.0);
        p.rows[0] = ContactRow::from_material(&material, d, dt, 0.0);
        p
    };
    let sens_at = |depth: f64| {
        let p = build(depth);
        let sol = solve_contacts(&p, &cfg);
        assert!(sol.converged);
        depth_sensitivity(&p, &sol, &cfg, &material.solref, dt).expect("converged")[0]
    };

    // Each side matches a finite difference taken *entirely within that side*.
    // A difference straddling the hinge would match neither, and a one-sided
    // difference is only O(h) — the smoothstep's curvature near the band top is
    // large enough that the truncation error would swamp the comparison. A
    // central difference at 1% into each regime is O(h^2) and stays put.
    let fd_at = |depth: f64, h: f64| {
        let plus = solve_contacts(&build(depth + h), &cfg);
        let minus = solve_contacts(&build(depth - h), &cfg);
        assert!(plus.converged && minus.converged);
        (plus.impulses[0].x - minus.impulses[0].x) / (2.0 * h)
    };
    let off = 1e-5; // 1% of the margin, and 1% of solimp.width.
    let h = 1e-8; // Comfortably inside `off`, so neither side crosses zero.
    for (side, depth) in [("below", -off), ("above", off)] {
        let analytic = sens_at(depth);
        let fd = fd_at(depth, h);
        let rel = (analytic - fd).abs() / fd.abs().max(1e-9);
        assert!(
            rel < 1e-5,
            "{side} zero depth: analytic {analytic} vs finite difference {fd} \
             (rel {rel:.3e})"
        );
    }

    // The jump itself, measured as close to the hinge as floating point allows.
    // There the impedance channel has vanished on both sides (it is C^1 with
    // zero slope at the join), so whatever is left is the bias switching on.
    let eps = 1e-9;
    let below = sens_at(-eps);
    let above = sens_at(eps);
    assert!(
        above > below,
        "crossing into penetration must strengthen the response: {below} -> {above}"
    );

    // And that jump is exactly `df_n/db_n * (-d * erp / dt)` — the bias term
    // and nothing else. Pinning the discontinuity to a closed form is what
    // makes it a documented property rather than an unexplained artifact.
    let p = build(eps);
    let sol = solve_contacts(&p, &cfg);
    assert!(sol.converged);
    let db = impulse_sensitivity(&p, &sol, &cfg).expect("converged");
    let erp = material.solref.error_reduction(dt);
    let predicted = -db[0] * p.rows[0].impedance * erp / dt;
    let observed = above - below;
    assert!(
        (observed - predicted).abs() / predicted.abs() < 1e-3,
        "the hinge must be exactly the bias term switching on: \
         observed {observed} vs predicted {predicted}"
    );

    // And the force itself is continuous across zero — the hinge is in the
    // slope only. This is precisely what the margin bought, and it is the
    // difference between a kink an optimizer can walk through and the cliff
    // this PR removed.
    let force_at = |depth: f64| {
        let sol = solve_contacts(&build(depth), &cfg);
        assert!(sol.converged);
        sol.impulses[0].x
    };
    let f_below = force_at(-eps);
    let f_above = force_at(eps);
    assert!(
        (f_above - f_below).abs() < 1e-6 * f_above.abs().max(1e-9),
        "force must be continuous across zero depth: {f_below} vs {f_above}"
    );
}

/// On the penetrating side, inside `solimp.width`, the impedance term is the
/// `~5%` correction the old form knowingly dropped. It is included now, so FD
/// must agree here too — the existing `depth_gradient_matches_finite_difference`
/// sits past `width`, where the sigmoid is pinned and the term is exactly zero,
/// so it could never have caught this.
#[test]
fn depth_gradient_inside_the_solimp_sigmoid_matches_finite_difference() {
    use phyz_contact::ContactMaterial;
    use phyz_contact::gradient::depth_sensitivity;

    let cfg = ContactSolverConfig::gradients();
    let dt = 1e-3;
    let material = ContactMaterial {
        friction: 0.9,
        ..ContactMaterial::default()
    };
    let width = material.solimp.width;

    let build = |d: f64| {
        let mut p = problem(1, material.friction, &[-0.4, 0.02, 0.0], 0.0);
        p.rows[0] = ContactRow::from_material(&material, d, dt, 0.0);
        p
    };

    let mut worst = 0.0f64;
    for frac in [0.2, 0.4, 0.6, 0.8] {
        let depth = frac * width;
        let p = build(depth);
        let sol = solve_contacts(&p, &cfg);
        assert!(sol.converged);
        let sens = depth_sensitivity(&p, &sol, &cfg, &material.solref, dt).expect("converged");

        for h in [1e-7, 1e-8] {
            let plus = solve_contacts(&build(depth + h), &cfg);
            let minus = solve_contacts(&build(depth - h), &cfg);
            let fd = (plus.impulses[0] - minus.impulses[0]) / (2.0 * h);
            for (k, want) in [fd.x, fd.y, fd.z].iter().enumerate() {
                let got = sens[k * p.n];
                let rel = (got - want).abs() / want.abs().max(1e-9);
                worst = worst.max(rel);
                assert!(
                    rel < 1e-5,
                    "depth {frac}*width, h={h}: d f[{k}]/d depth analytic {got} \
                     vs finite difference {want} (rel {rel:.3e})"
                );
            }
        }
    }
    println!("worst relative FD error inside the solimp sigmoid: {worst:.3e}");
}

/// Depth sensitivity across *coupled* contacts, with one of them inside the
/// margin band.
///
/// The single-contact tests would pass even if the impedance perturbation were
/// wired only into its own contact's rows. It is not: `dF/ddepth_c` is
/// supported on contact `c`'s rows but propagates to every other contact
/// through `A`, exactly as the friction sensitivity does. Perturbing the depth
/// of a barely-touching corner has to move the impulse on the corner across
/// from it, which is the whole reason the Delassus operator is here.
#[test]
fn coupled_depth_gradient_in_the_margin_matches_finite_difference() {
    use phyz_contact::ContactMaterial;
    use phyz_contact::gradient::depth_sensitivity;

    let cfg = ContactSolverConfig::gradients();
    let dt = 1e-3;
    let material = ContactMaterial {
        friction: 0.9,
        ..ContactMaterial::default()
    };
    let margin = material.margin;

    // Contact 0 sits inside the margin band; contact 1 is genuinely
    // penetrating. They are coupled through the off-diagonal of A.
    let other_depth = 2e-4;
    let build = |d0: f64| {
        let mut p = problem(
            2,
            material.friction,
            &[-0.4, 0.02, 0.0, -0.5, 0.0, 0.0],
            0.35,
        );
        p.rows[0] = ContactRow::from_material(&material, d0, dt, 0.0);
        p.rows[1] = ContactRow::from_material(&material, other_depth, dt, 0.0);
        p
    };

    let depth0 = -0.3 * margin;
    let p = build(depth0);
    let sol = solve_contacts(&p, &cfg);
    assert!(sol.converged);
    let sens = depth_sensitivity(&p, &sol, &cfg, &material.solref, dt).expect("converged");

    let mut worst = 0.0f64;
    for h in [1e-7, 1e-8] {
        let plus = solve_contacts(&build(depth0 + h), &cfg);
        let minus = solve_contacts(&build(depth0 - h), &cfg);
        assert!(plus.converged && minus.converged);
        // Column 0 of df/ddepth: the response of *both* contacts to contact
        // zero's depth.
        for c in 0..2 {
            let fd = (plus.impulses[c] - minus.impulses[c]) / (2.0 * h);
            for (k, want) in [fd.x, fd.y, fd.z].iter().enumerate() {
                let got = sens[(3 * c + k) * p.n];
                let rel = (got - want).abs() / want.abs().max(1e-9);
                worst = worst.max(rel);
                assert!(
                    rel < 1e-5,
                    "h={h}: d f{c}[{k}]/d depth0 analytic {got} vs finite \
                     difference {want} (rel {rel:.3e})"
                );
            }
        }
    }

    // The cross term is real, not incidentally zero.
    assert!(
        sens[3 * p.n].abs() > 1e-6,
        "contact 1 must respond to contact 0's depth through A; got {}",
        sens[3 * p.n]
    );
    println!("worst relative FD error, coupled, contact 0 in band: {worst:.3e}");
}
