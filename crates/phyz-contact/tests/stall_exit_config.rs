//! Does [`ContactSolverConfig::no_stall_exit`] actually reach the solver?
//!
//! The stagnation exit used to be reachable only through
//! `PHYZ_NO_STALL_EXIT`, which is latched in a `OnceLock`: a process gets one
//! answer, so two solves in it could not disagree about whether to spend the
//! whole iteration budget. That is a real constraint rather than a stylistic
//! one — ipse's `shac_gradient_path.rs` holds two findings that need opposite
//! settings (the exit makes the jump window discontinuous at the shipped
//! default; the second contact solve is as differentiable as the first, which
//! is only measurable with the exit off), and as a variable those two claims
//! could not share a test binary.
//!
//! The variable itself cannot be exercised from a test — latching it would
//! poison every other test in the binary — so what is pinned here is the half
//! that is both testable and load-bearing: on a problem the staged solver
//! stalls out of, the *field* must change where the solve stops. A field that
//! were accepted and then ignored would report the identical stalled
//! residual, which is exactly what this catches.

use phyz_contact::{ContactProblem, ContactRow, ContactSolverConfig, solve_contacts};

/// A rank-deficient manifold: `n` contacts whose Delassus rows come from three
/// generators, so `A` is singular and the null-space directions decay at the
/// regularizer's rate — the case the stagnation exit exists to abandon.
fn redundant_problem(n: usize, mu: f64) -> ContactProblem {
    let dim = 3 * n;
    let gens: Vec<Vec<f64>> = (0..3)
        .map(|g| {
            (0..dim)
                .map(|i| ((i * 7 + g * 13) as f64 * 0.41).sin())
                .collect()
        })
        .collect();
    let mut delassus = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut acc = 0.0;
            for g in &gens {
                acc += g[i] * g[j];
            }
            delassus[i * dim + j] = acc + if i == j { 0.02 } else { 0.0 };
        }
    }
    let free_velocity: Vec<f64> = (0..dim)
        .map(|i| {
            if i % 3 == 0 {
                -0.5
            } else {
                0.2 * ((i * 5) as f64 * 0.9).sin()
            }
        })
        .collect();
    ContactProblem {
        n,
        delassus,
        free_velocity,
        rows: (0..n)
            .map(|_| ContactRow {
                mu,
                bias: 0.005,
                ..Default::default()
            })
            .collect(),
        bodies: (0..n).map(|_| (0, usize::MAX)).collect(),
    }
}

#[test]
fn the_no_stall_exit_field_spends_the_budget() {
    let p = redundant_problem(8, 0.6);
    let tight = ContactSolverConfig {
        tolerance: 1e-14,
        max_iterations: 600,
        ..ContactSolverConfig::gradients()
    };
    let stalled = solve_contacts(&p, &tight);
    let full = solve_contacts(
        &p,
        &ContactSolverConfig {
            no_stall_exit: true,
            ..tight
        },
    );

    assert!(
        !stalled.converged && !full.converged,
        "the fixture must be one neither setting can converge, or this \
         compares two converged answers and proves nothing ({} / {})",
        stalled.converged,
        full.converged
    );
    assert!(
        full.iterations > stalled.iterations,
        "no_stall_exit must spend more of the budget: {} iterations against \
         the stalling default's {}",
        full.iterations,
        stalled.iterations
    );
    assert!(
        full.residual <= stalled.residual,
        "spending the budget must not make the residual worse: {:e} against \
         {:e}",
        full.residual,
        stalled.residual
    );
}

/// The default presets leave the exit **on**, so nothing that did not ask for
/// the field changes behaviour.
#[test]
fn every_preset_ships_with_the_stagnation_exit_on() {
    for (name, cfg) in [
        ("simulation", ContactSolverConfig::simulation()),
        ("gradients", ContactSolverConfig::gradients()),
        ("mujoco_compat", ContactSolverConfig::mujoco_compat()),
        ("gpu_equivalent", ContactSolverConfig::gpu_equivalent()),
        ("default", ContactSolverConfig::default()),
    ] {
        assert!(
            !cfg.no_stall_exit,
            "{name} must not silently spend the whole iteration budget"
        );
    }
}
