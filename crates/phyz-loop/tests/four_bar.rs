//! Four-bar linkage: the canonical closed chain, and the thing `phyz-rigid`
//! cannot represent at all.
//!
//! Claims pinned here, each against a closed form or a measured number that is
//! printed rather than merely asserted:
//!
//! - mobility 1, agreeing with the planar Grübler count;
//! - the constraint residual stays bounded over a 10-second rollout;
//! - turning stabilization off makes it *not* stay bounded, which is what
//!   makes the previous claim mean something.

mod common;

use common::{FOUR_BAR, four_bar_closure, four_bar_model, four_bar_state};
use phyz_loop::{
    LoopSolverConfig, MobilitySpace, Stabilization, assemble, constraint_rank, grubler, mobility,
    solve, step,
};

#[test]
fn four_bar_mobility_matches_grubler() {
    let model = four_bar_model(1e-3);
    let state = four_bar_state(&model);
    let set = four_bar_closure();
    let sys = assemble(&model, &state, &set);

    // Grübler for a planar four-bar: 4 links *counting the ground*, 4 revolute
    // joints (3 in the tree, 1 cut), one DOF each.
    let g = grubler(MobilitySpace::Planar, 4, &[1, 1, 1, 1]);
    assert_eq!(g, 1, "Grübler mobility");

    // Numerical: nv - rank(J). The 3-row point constraint has an identically
    // zero Z row on a planar mechanism, so its rank is 2, not 3. A formulation
    // that assumed full row rank would have divided by zero here.
    let rank = constraint_rank(&sys.jacobian, 1e-9);
    let m = mobility(model.nv, &sys.jacobian, 1e-9);
    println!(
        "four-bar: nv={} rank(J)={} mobility={} grubler={}",
        model.nv, rank, m, g
    );
    assert_eq!(rank, 2, "constraint rank");
    assert_eq!(m as i64, g, "numerical mobility must agree with Grübler");
}

#[test]
fn four_bar_is_assembled_to_machine_precision() {
    let model = four_bar_model(1e-3);
    let state = four_bar_state(&model);
    let sys = assemble(&model, &state, &four_bar_closure());
    let r = sys.position_residual_inf();
    println!("four-bar initial |c|_inf = {r:.3e} m");
    assert!(r < 1e-15, "test fixture is not actually closed: {r:e}");
}

/// The headline claim: a long rollout of a closed chain whose constraint does
/// not run away.
#[test]
fn four_bar_residual_stays_bounded_over_ten_seconds() {
    let dt = 1e-3;
    let model = four_bar_model(dt);
    let mut state = four_bar_state(&model);
    let set = four_bar_closure();
    let cfg = LoopSolverConfig::for_model(&model);

    // Spin it. The centripetal term in `Jdot v` is proportional to omega^2, so
    // a stationary start would not exercise it at all. The projection is not
    // optional: "crank at 4 rad/s, everything else zero" is a state in which
    // the loop is being torn open at 8 m/s.
    state.v[0] = 4.0;
    state.v = phyz_loop::project_velocity(&model, &state, &set, &cfg);

    let steps = (10.0 / dt) as usize;
    let mut max_pos = 0.0_f64;
    let mut max_vel = 0.0_f64;
    let mut max_iters = 0usize;
    let mut total_iters = 0usize;
    let mut nonconverged = 0usize;

    for _ in 0..steps {
        let sol = step(&model, &mut state, &set, &cfg);
        max_pos = max_pos.max(inf(&sol.system.position_error));
        max_vel = max_vel.max(inf(&sol.system.velocity_error));
        max_iters = max_iters.max(sol.iterations);
        total_iters += sol.iterations;
        if !sol.converged {
            nonconverged += 1;
        }
    }

    let final_sys = assemble(&model, &state, &set);
    println!(
        "four-bar 10 s @ dt={dt}: max |c|_inf = {max_pos:.3e} m, max |Jv|_inf = {max_vel:.3e} m/s, \
         final |c|_inf = {:.3e} m, iterations mean {:.1} max {max_iters}, non-converged steps {nonconverged}/{steps}",
        final_sys.position_residual_inf(),
        total_iters as f64 / steps as f64,
    );

    assert_eq!(nonconverged, 0, "solver failed to converge on some steps");
    assert!(max_pos < 2e-3, "position drift {max_pos:e} m exceeded 2 mm");
    assert!(max_vel < 1e-1, "velocity residual {max_vel:e} m/s");
}

/// Without Baumgarte the same rollout drifts, by orders of magnitude more.
/// This is the control that makes the bounded-drift claim above non-vacuous —
/// and it is also the honest statement that the acceleration-level constraint
/// on its own does *not* hold position.
#[test]
fn unstabilized_four_bar_drifts_much_further() {
    let dt = 1e-3;
    let model = four_bar_model(dt);
    let set = four_bar_closure();

    let run = |st: Stabilization| {
        let mut state = four_bar_state(&model);
        let cfg = LoopSolverConfig {
            stabilization: st,
            ..LoopSolverConfig::for_model(&model)
        };
        state.v[0] = 4.0;
        state.v = phyz_loop::project_velocity(&model, &state, &set, &cfg);
        let mut worst = 0.0_f64;
        for _ in 0..2_000 {
            let sol = step(&model, &mut state, &set, &cfg);
            worst = worst.max(inf(&sol.system.position_error));
        }
        worst
    };

    let stabilized = run(LoopSolverConfig::for_model(&model).stabilization);
    let raw = run(Stabilization::NONE);
    println!("2 s drift: stabilized {stabilized:.3e} m, unstabilized {raw:.3e} m");
    assert!(
        raw > 20.0 * stabilized,
        "unstabilized drift {raw:e} was not meaningfully worse than stabilized {stabilized:e}"
    );
}

/// Gauss's principle in its most checkable form: the constrained acceleration
/// must satisfy the KKT stationarity `M (a - a_free) = J^T mu`.
#[test]
fn solution_satisfies_kkt_stationarity() {
    let model = four_bar_model(1e-3);
    let mut state = four_bar_state(&model);
    state.v[0] = 2.0;
    state.v[1] = -0.5;
    let set = four_bar_closure();
    let sol = solve(&model, &state, &set, &LoopSolverConfig::for_model(&model));
    assert!(sol.converged, "residual {:e}", sol.residual);

    let a_free = phyz_rigid::aba(&model, &state);
    let mass = phyz_rigid::crba(&model, &state);
    let da = phyz_math::DVec::from_fn(model.nv, |i| sol.qdd[i] - a_free[i]);
    let lhs = mass.mul_vec(&da);
    let rhs = sol.system.jacobian.transpose().mul_vec(&sol.multipliers);
    let err = (0..model.nv).fold(0.0_f64, |m, i| m.max((lhs[i] - rhs[i]).abs()));
    println!("KKT stationarity |M(a-a_free) - J^T mu|_inf = {err:.3e}");
    // The proximal term `sigma (a^k - a)` is exactly the slack in this
    // identity, and it vanishes as the iteration converges.
    // The returned `mu` is the *proximally regularized* multiplier, so the
    // identity holds to O(rho * epsilon * |J^T mu|) — about 1e-4 here — not to
    // machine precision. Tightening `epsilon` tightens this; it also removes
    // the redundancy guard, which is the trade the module docs describe.
    assert!(err < 1e-3, "KKT stationarity violated by {err:e}");
}

#[test]
fn geometry_constants_are_consistent() {
    // The fixture claims `coupler = |(2,2) - (0,1)|`. If that ever stops being
    // true the "assembled to machine precision" test would be measuring the
    // wrong thing.
    let expect =
        ((FOUR_BAR.ground - 0.0f64).powi(2) + (FOUR_BAR.rocker - FOUR_BAR.crank).powi(2)).sqrt();
    assert!((FOUR_BAR.coupler - expect).abs() < 1e-15);
}

/// The residual left over is the *integrator's*, not the constraint's: it
/// falls with `dt` rather than sitting at a floor. Measured, 10 s of the same
/// spinning four-bar:
///
/// | dt | max \|c\|_inf | max \|J v\|_inf |
/// |---|---|---|
/// | 1e-3 | 8.7e-4 m | 3.7e-2 m/s |
/// | 2e-4 | 3.8e-5 m | 9.3e-3 m/s |
///
/// A constraint formulation with a sign or frame error would show a floor here
/// instead, because the error would be a modelling error and modelling errors
/// do not care about `dt`.
#[test]
fn drift_falls_with_timestep() {
    let run = |dt: f64| {
        let model = four_bar_model(dt);
        let set = four_bar_closure();
        let cfg = LoopSolverConfig::for_model(&model);
        let mut state = four_bar_state(&model);
        state.v[0] = 4.0;
        state.v = phyz_loop::project_velocity(&model, &state, &set, &cfg);
        let steps = (2.0 / dt) as usize;
        let mut worst = 0.0_f64;
        for _ in 0..steps {
            let sol = step(&model, &mut state, &set, &cfg);
            worst = worst.max(inf(&sol.system.position_error));
        }
        worst
    };
    let coarse = run(1e-3);
    let fine = run(2e-4);
    println!(
        "2 s drift: dt=1e-3 {coarse:.3e} m, dt=2e-4 {fine:.3e} m, ratio {:.1}",
        coarse / fine
    );
    assert!(
        fine < 0.25 * coarse,
        "drift did not fall with dt: {coarse:e} -> {fine:e}"
    );
}

fn inf(v: &phyz_math::DVec) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}
