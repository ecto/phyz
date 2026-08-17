//! Determinism is a product guarantee, so it gets an assertion rather than a
//! paragraph.
//!
//! Bit-identical, not "within tolerance". A tolerance-based check would pass
//! on a solver that iterated over a `HashMap` and happened to get the same
//! answer to 12 digits this run.

mod common;

use common::{
    four_bar_closure, four_bar_model, four_bar_state, slider_crank_closure, slider_crank_model,
    slider_crank_state,
};
use phyz_loop::{LoopSolverConfig, project_velocity, step};

/// A full rollout's `(q, v)` and the solver's own diagnostics, as raw bits.
fn rollout_bits(mechanism: usize, steps: usize) -> Vec<u64> {
    let dt = 1e-3;
    let (model, set, mut state) = if mechanism == 0 {
        let model = four_bar_model(dt);
        let set = four_bar_closure();
        let state = four_bar_state(&model);
        (model, set, state)
    } else {
        let model = slider_crank_model(dt);
        let set = slider_crank_closure();
        let state = slider_crank_state(&model, 0.7);
        (model, set, state)
    };

    let cfg = LoopSolverConfig::for_model(&model);
    state.v[0] = 3.0;
    state.v = project_velocity(&model, &state, &set, &cfg);

    let mut out = Vec::with_capacity(steps * (2 * model.nv + 2));
    for _ in 0..steps {
        let sol = step(&model, &mut state, &set, &cfg);
        for i in 0..model.nv {
            out.push(state.q[i].to_bits());
            out.push(state.v[i].to_bits());
        }
        out.push(sol.residual.to_bits());
        out.push(sol.iterations as u64);
    }
    out
}

#[test]
fn rollouts_are_bit_identical_across_runs() {
    for mechanism in 0..2 {
        let a = rollout_bits(mechanism, 500);
        let b = rollout_bits(mechanism, 500);
        assert_eq!(a.len(), b.len());
        let first_diff = a.iter().zip(b.iter()).position(|(x, y)| x != y);
        assert_eq!(
            first_diff, None,
            "mechanism {mechanism}: rollouts diverged at word {first_diff:?}"
        );
    }
    println!("two independent 500-step rollouts of both mechanisms agreed in every bit");
}

/// Solving the same state twice must give the same bits too — the solve itself
/// carries no hidden state (no warm start, no cached factorization keyed on
/// anything).
#[test]
fn repeated_solves_of_one_state_are_bit_identical() {
    let model = four_bar_model(1e-3);
    let set = four_bar_closure();
    let cfg = LoopSolverConfig::for_model(&model);
    let mut state = four_bar_state(&model);
    state.v[0] = 2.5;
    state.v = project_velocity(&model, &state, &set, &cfg);

    let a = phyz_loop::solve(&model, &state, &set, &cfg);
    let b = phyz_loop::solve(&model, &state, &set, &cfg);
    for i in 0..model.nv {
        assert_eq!(a.qdd[i].to_bits(), b.qdd[i].to_bits(), "qdd[{i}]");
    }
    for i in 0..a.multipliers.len() {
        assert_eq!(
            a.multipliers[i].to_bits(),
            b.multipliers[i].to_bits(),
            "mu[{i}]"
        );
    }
    assert_eq!(a.iterations, b.iterations);
    assert_eq!(a.residual.to_bits(), b.residual.to_bits());
}
