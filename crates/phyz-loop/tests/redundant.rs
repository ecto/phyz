//! Redundant and over-constrained loops — the case naive formulations blow up
//! on.
//!
//! Three regimes, all real:
//!
//! 1. **Redundant and consistent.** The same closure stated twice. `J M^-1 J^T`
//!    is exactly singular; there is a whole affine family of multipliers giving
//!    the same acceleration. A direct KKT factorization has nothing to return.
//! 2. **Redundant and inconsistent.** The same thing, but the two copies
//!    disagree by a hair — which is what a *consistent* redundant set becomes
//!    after one timestep of drift, so this is the normal case, not the exotic
//!    one. There is now no exact solution at all.
//! 3. **Over-constrained.** A weld where the mechanism only admits a pin: the
//!    mobility drops to zero and the linkage becomes a structure.
//!
//! The claim being tested is *graceful degradation*, and specifically not
//! "correctness": in regime 2 there is no correct answer, so what is checked is
//! that the answer stays bounded, close to the well-posed nearby problem, and
//! reports a residual that reflects the inconsistency.

mod common;

use common::{FOUR_BAR, four_bar_closure, four_bar_model, four_bar_state};
use phyz_loop::{
    Anchor, LoopConstraint, LoopConstraintSet, LoopSolverConfig, assemble, constraint_rank,
    mobility, solve,
};
use phyz_math::Vec3;

fn duplicated(offset: f64) -> LoopConstraintSet {
    let mut set = four_bar_closure();
    // Second copy of the same pin. `offset = 0` duplicates it exactly;
    // any non-zero offset makes the pair unsatisfiable.
    set.push(LoopConstraint::point(
        "duplicate pin",
        Anchor::body(1, Vec3::new(FOUR_BAR.coupler, 0.0, 0.0)),
        Anchor::body(2, Vec3::new(FOUR_BAR.rocker + offset, 0.0, 0.0)),
    ));
    set
}

fn spinning_state(
    model: &phyz_model::Model,
    set: &LoopConstraintSet,
    cfg: &LoopSolverConfig,
) -> phyz_model::State {
    let mut state = four_bar_state(model);
    state.v[0] = 4.0;
    state.v = phyz_loop::project_velocity(model, &state, set, cfg);
    state
}

#[test]
fn duplicated_closure_is_rank_deficient() {
    let model = four_bar_model(1e-3);
    let state = four_bar_state(&model);
    let set = duplicated(0.0);
    let sys = assemble(&model, &state, &set);
    let rank = constraint_rank(&sys.jacobian, 1e-9);
    println!(
        "duplicated four-bar closure: rows={} rank={rank} mobility={}",
        sys.nrows(),
        mobility(model.nv, &sys.jacobian, 1e-9)
    );
    assert_eq!(sys.nrows(), 6);
    assert_eq!(rank, 2, "six rows, two of them independent");
}

/// Regime 1: stating the constraint twice must not change the physics.
#[test]
fn consistent_redundancy_reproduces_the_single_constraint_solution() {
    let model = four_bar_model(1e-3);
    let cfg = LoopSolverConfig::for_model(&model);
    let single = four_bar_closure();
    let state = spinning_state(&model, &single, &cfg);

    let a1 = solve(&model, &state, &single, &cfg);
    let a2 = solve(&model, &state, &duplicated(0.0), &cfg);
    assert!(
        a1.converged && a2.converged,
        "{} {}",
        a1.residual,
        a2.residual
    );

    let diff = (0..model.nv).fold(0.0_f64, |m, i| m.max((a1.qdd[i] - a2.qdd[i]).abs()));
    println!(
        "single vs duplicated: max |a difference| = {diff:.3e} rad/s^2, \
         iterations {} vs {}, |mu|_inf {:.3e} vs {:.3e}",
        a1.iterations,
        a2.iterations,
        inf(&a1.multipliers),
        inf(&a2.multipliers)
    );
    assert!(
        diff < 1e-6,
        "redundancy changed the accelerations by {diff:e}"
    );
}

/// Regime 2, the important one: an *inconsistent* redundant set must degrade,
/// not explode.
///
/// There is no solution here, so there is no correct answer to check against.
/// What is checked instead:
///
/// - the **accelerations** stay finite and close to the well-posed nearby
///   problem. This is the part that matters, and it survives even a large
///   multiplier, because the redundant multipliers very nearly cancel in
///   `J^T mu`;
/// - the solver reports `converged: false`. It can never report anything else
///   here: the residual floor of an unsatisfiable set is the inconsistency
///   itself, about `beta^2 * 1e-6 = 1e-2 m/s^2` at these gains, and no choice
///   of `epsilon` moves it. A solver that called this converged would be lying;
/// - `epsilon` is the knob that bounds the multiplier. Sized to the problem
///   (`1e-4`) it caps `|mu|` at roughly `residual / epsilon`; at the default
///   `1e-9`, sized for full-rank sets, the cap is far above anything reachable
///   inside the iteration budget, so it is the *budget* doing the bounding;
/// - with `epsilon = 0` there is no fixed point at all — ten times the
///   iterations gives ten times the multiplier.
#[test]
fn inconsistent_redundancy_degrades_gracefully() {
    let model = four_bar_model(1e-3);
    let cfg = LoopSolverConfig::for_model(&model);
    let set = duplicated(1e-6); // 1 micron of disagreement
    let state = spinning_state(&model, &four_bar_closure(), &cfg);

    let reference = solve(&model, &state, &four_bar_closure(), &cfg);
    let da = |s: &phyz_loop::LoopSolution| {
        (0..model.nv).fold(0.0_f64, |m, i| m.max((s.qdd[i] - reference.qdd[i]).abs()))
    };
    let report = |name: &str, s: &phyz_loop::LoopSolution| {
        println!(
            "  {name:<22} |mu|_inf = {:9.3e}  |a - a_ref|_inf = {:9.3e}  residual = {:9.3e}  iters = {:4}  converged = {}",
            inf(&s.multipliers),
            da(s),
            s.residual,
            s.iterations,
            s.converged
        );
    };

    let default = solve(&model, &state, &set, &cfg);
    let compliant = solve(
        &model,
        &state,
        &set,
        &LoopSolverConfig {
            epsilon: 1e-4,
            ..cfg
        },
    );
    let unguarded = solve(
        &model,
        &state,
        &set,
        &LoopSolverConfig {
            epsilon: 0.0,
            ..cfg
        },
    );
    let unguarded_long = solve(
        &model,
        &state,
        &set,
        &LoopSolverConfig {
            epsilon: 0.0,
            max_iterations: cfg.max_iterations * 10,
            ..cfg
        },
    );

    println!("inconsistent redundant set, the two copies 1e-6 m apart:");
    report("default eps=1e-9", &default);
    report("compliant eps=1e-4", &compliant);
    report("eps=0, 200 iters", &unguarded);
    report("eps=0, 2000 iters", &unguarded_long);

    // The physics stays right under the default, and the solver is honest
    // about not having converged.
    assert!(default.qdd.iter().all(|x| x.is_finite()));
    assert!(
        da(&default) < 1e-2,
        "accelerations diverged from the nearby well-posed problem by {:e}",
        da(&default)
    );
    assert!(
        !default.converged,
        "an unsatisfiable set must not report convergence"
    );

    // No epsilon makes an unsatisfiable set satisfiable, and none of these
    // report otherwise.
    assert!(!compliant.converged && !unguarded.converged);
    // But a compliance sized to the problem caps the multiplier, and the
    // physics is unchanged by the capping.
    assert!(
        inf(&compliant.multipliers) < 0.2 * inf(&default.multipliers),
        "epsilon = 1e-4 did not bound the multiplier: {:e} vs default {:e}",
        inf(&compliant.multipliers),
        inf(&default.multipliers)
    );
    assert!(da(&compliant) < 1e-2);

    // And without the dual proximal term there is no fixed point at all: the
    // multiplier is proportional to however long you iterate.
    let growth = inf(&unguarded_long.multipliers) / inf(&unguarded.multipliers);
    println!("  eps=0 multiplier growth over 10x the iterations: {growth:.1}x");
    assert!(
        growth > 5.0,
        "epsilon = 0 was expected to run away, grew only {growth}x"
    );
}

/// Regime 3: welding the four-bar's pin turns a 1-DOF linkage into a 0-DOF
/// structure. Nothing should move, and the solve should say so rather than
/// failing.
#[test]
fn welding_the_pin_makes_a_structure() {
    let model = four_bar_model(1e-3);
    let state = four_bar_state(&model);
    let cfg = LoopSolverConfig::for_model(&model);

    let mut set = LoopConstraintSet::new();
    set.push(LoopConstraint::weld(
        &model,
        &state,
        "welded pin",
        Anchor::body(1, Vec3::new(FOUR_BAR.coupler, 0.0, 0.0)),
        Anchor::body(2, Vec3::new(FOUR_BAR.rocker, 0.0, 0.0)),
    ));

    let sys = assemble(&model, &state, &set);
    let rank = constraint_rank(&sys.jacobian, 1e-9);
    let dof = mobility(model.nv, &sys.jacobian, 1e-9);
    println!(
        "welded four-bar: rows={} rank={rank} mobility={dof}, initial |c|_inf = {:.3e}",
        sys.nrows(),
        sys.position_residual_inf()
    );
    assert_eq!(sys.nrows(), 6, "a weld contributes 6 rows");
    assert_eq!(dof, 0, "a welded four-bar is a structure");

    let sol = solve(&model, &state, &set, &cfg);
    let a = inf(&sol.qdd);
    println!(
        "welded four-bar: |a|_inf = {a:.3e} rad/s^2, residual = {:.3e}, iters = {}, converged = {}",
        sol.residual, sol.iterations, sol.converged
    );
    assert!(sol.converged, "residual {:e}", sol.residual);
    assert!(a < 1e-6, "a structure accelerated at {a:e}");
}

/// A weld captures the relative orientation it was built at, so it must read
/// zero residual at that state and non-zero once a body has turned.
#[test]
fn weld_residual_responds_to_relative_rotation() {
    let model = four_bar_model(1e-3);
    let state = four_bar_state(&model);
    let mut set = LoopConstraintSet::new();
    set.push(LoopConstraint::weld(
        &model,
        &state,
        "welded pin",
        Anchor::body(1, Vec3::new(FOUR_BAR.coupler, 0.0, 0.0)),
        Anchor::body(2, Vec3::new(FOUR_BAR.rocker, 0.0, 0.0)),
    ));
    let at_capture = assemble(&model, &state, &set).position_residual_inf();

    // Turn the rocker by 0.01 rad about +Z and nothing else. The rotational
    // rows must report that, with the sign fixed by the A-minus-B convention:
    // B (the rocker) turned by +0.01, so `c = -0.01` on the Z row.
    let mut turned = state.clone();
    turned.q[2] += 0.01;
    let sys = assemble(&model, &turned, &set);
    let z_row = sys.position_error[5];
    println!(
        "weld: |c| at capture {at_capture:.3e}, c_z after +0.01 rad on the rocker = {z_row:.6e}"
    );
    assert!(at_capture < 1e-15);
    assert!(
        (z_row + 0.01_f64.sin()).abs() < 1e-9,
        "weld rotational row has the wrong sign or scale: {z_row}"
    );
}

fn inf(v: &phyz_math::DVec) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}
