//! Convergence budget: iterations-to-tolerance versus contact count.
//!
//! Redundancy is what makes the contact solve expensive. A determined support
//! polygon (three or four contacts) constrains six body DoF with an `A` that
//! has full rank on the active set; eight or thirty-two coplanar contacts do
//! not, and `A` is singular by a margin that grows with the count. Anything
//! that iterates on `A` alone therefore has its rate set by the *regularizer*
//! in those directions, which is deliberately tiny (see
//! `convex::regularization_diag`), so a pure projected-Gauss-Seidel sweep needs
//! O(1e5) iterations to remove a null-space impulse.
//!
//! This file exists so that cost cannot silently regress. It is a budget, not a
//! benchmark harness: it asserts an iteration ceiling per contact count and
//! prints the measured table under `--nocapture`.

use phyz_collision::Collision;
use phyz_contact::{ContactMaterial, ContactSolverConfig, assemble, solve_contacts};
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Model, ModelBuilder, State};

const MASS: f64 = 10.0;
const DT: f64 = 2e-3;

fn slab() -> (Model, State) {
    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(DT)
        .add_free_body(
            "slab",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                MASS,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(0.3, 0.3, 0.5)),
            ),
        )
        .build();
    let mut state = model.default_state();
    state.body_xform[0] = SpatialTransform::identity();
    (model, state)
}

/// `n` coplanar contacts on a ring, exactly as `redundant_contacts.rs` builds
/// them — symmetric for every `n`, so the correct answer is always "an even
/// share of the load and no friction".
fn ring(n: usize) -> Vec<Collision> {
    (0..n)
        .map(|k| {
            let t = k as f64 * std::f64::consts::TAU / n as f64;
            Collision {
                body_i: 0,
                body_j: Collision::WORLD,
                contact_point: Vec3::new(0.1 * t.cos(), 0.1 * t.sin(), -0.05),
                contact_normal: Vec3::z(),
                penetration_depth: 5e-5,
            }
        })
        .collect()
}

/// Iterations the solve reported, and whether it reached tolerance.
fn sweeps_to_tolerance(n: usize, warm: bool) -> (usize, bool, f64) {
    let (model, state) = slab();
    let contacts = ring(n);
    let mut free_qd = DVec::zeros(model.nv);
    free_qd[5] = -GRAVITY * DT;
    let cfg = ContactSolverConfig::simulation();
    let assembly = assemble(
        &model,
        &state,
        &contacts,
        &[ContactMaterial::default()],
        &free_qd,
        DT,
        &cfg,
    );
    let cold = solve_contacts(&assembly.problem, &cfg);
    if !warm {
        return (cold.iterations, cold.converged, cold.residual);
    }
    // A stepping loop re-solves an almost identical problem every step; the
    // warm figure is the one that governs throughput in practice.
    let hot = phyz_contact::solve_contacts_warm(&assembly.problem, &cfg, &cold.impulses);
    (hot.iterations, hot.converged, hot.residual)
}

/// The budget. Numbers are generous against the measured table so that ordinary
/// numerical drift does not fail the build, but tight enough that a return to
/// pure-PGS behaviour (hundreds to thousands of sweeps, or the cap) does.
#[test]
fn convergence_budget_by_contact_count() {
    // (contacts, iteration ceiling — the same for both start conditions)
    let budget = [(2, 40), (4, 40), (8, 40), (16, 40), (32, 40)];

    println!("\n contacts | cold iters | warm iters | residual");
    println!("----------+------------+------------+----------");
    let mut failures = Vec::new();
    for (n, ceiling) in budget {
        let (cold, cold_ok, res) = sweeps_to_tolerance(n, false);
        let (warm, warm_ok, _) = sweeps_to_tolerance(n, true);
        println!(" {n:>8} | {cold:>10} | {warm:>10} | {res:>8.2e}");
        if !cold_ok || !warm_ok {
            failures.push(format!(
                "n={n} did not converge (cold={cold_ok}, warm={warm_ok})"
            ));
        }
        if cold > ceiling {
            failures.push(format!(
                "n={n} cold start took {cold} iters, budget {ceiling}"
            ));
        }
        if warm > ceiling {
            failures.push(format!(
                "n={n} warm start took {warm} iters, budget {ceiling}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Redundancy must not cost asymptotically more than determinacy. Four contacts
/// is the determined reference; thirty-two is eight times as redundant and must
/// stay within a small constant factor of it.
///
/// This is the property a pure PGS sweep does not have: its rate is set by the
/// conditioning of `A` on the null space, which degrades with the count, so the
/// ratio grew without bound.
#[test]
fn redundancy_does_not_blow_up_the_iteration_count() {
    let (four, _, _) = sweeps_to_tolerance(4, false);
    let (thirty_two, _, _) = sweeps_to_tolerance(32, false);
    assert!(
        thirty_two <= 4 * four.max(1),
        "32 coplanar contacts took {thirty_two} iterations against {four} for 4"
    );
}
