//! Slider-crank against its closed-form kinematics.
//!
//! ```text
//! x(theta)     = r cos(theta) + sqrt(l^2 - r^2 sin^2(theta))
//! xdot(theta)  = dx/dtheta * thetadot
//! ```
//!
//! This mechanism is the useful complement to the four-bar: it is
//! **asymmetric**, it mixes a prismatic DOF with revolute ones, and its
//! closure relates a rotation to a translation. A sign flip in the constraint
//! rows, or a swapped A/B convention, survives a symmetric four-bar and dies
//! here.

mod common;

use common::{
    CRANK_R, ROD_L, slider_crank_closure, slider_crank_model, slider_crank_state, slider_x,
    slider_xdot,
};
use phyz_loop::{
    LoopSolverConfig, MobilitySpace, assemble, constraint_rank, grubler, mobility, step,
};

#[test]
fn slider_crank_mobility_is_one() {
    let model = slider_crank_model(1e-3);
    let state = slider_crank_state(&model, 1.0);
    let sys = assemble(&model, &state, &slider_crank_closure());
    let g = grubler(MobilitySpace::Planar, 4, &[1, 1, 1, 1]);
    let m = mobility(model.nv, &sys.jacobian, 1e-9);
    println!(
        "slider-crank: nv={} rank(J)={} mobility={m} grubler={g}",
        model.nv,
        constraint_rank(&sys.jacobian, 1e-9)
    );
    assert_eq!(m as i64, g);
}

#[test]
fn fixture_is_assembled_at_every_crank_angle() {
    let model = slider_crank_model(1e-3);
    let set = slider_crank_closure();
    let mut worst = 0.0_f64;
    for k in 0..36 {
        let theta = k as f64 * std::f64::consts::TAU / 36.0;
        let state = slider_crank_state(&model, theta);
        worst = worst.max(assemble(&model, &state, &set).position_residual_inf());
    }
    println!("slider-crank fixture max |c|_inf over a full turn = {worst:.3e} m");
    assert!(worst < 1e-14);
}

/// The load-bearing test: simulate, and check the *simulated* slider position
/// against the closed form evaluated at the *simulated* crank angle. Any
/// consistent pair `(theta, x)` on the analytic curve passes; the dynamics can
/// do whatever they like.
#[test]
fn simulated_slider_tracks_closed_form_kinematics() {
    let dt = 1e-3;
    let model = slider_crank_model(dt);
    let set = slider_crank_closure();
    let cfg = LoopSolverConfig::for_model(&model);

    let mut state = slider_crank_state(&model, 0.7);
    state.v[0] = 6.0; // spin the crank, then make the rest consistent
    state.v = phyz_loop::project_velocity(&model, &state, &set, &cfg);

    let steps = 5_000;
    let mut worst_x = 0.0_f64;
    let mut worst_xdot = 0.0_f64;
    let mut worst_speed = 0.0_f64;
    let mut nonconverged = 0usize;

    for _ in 0..steps {
        let sol = step(&model, &mut state, &set, &cfg);
        if !sol.converged {
            nonconverged += 1;
        }
        let theta = state.q[0];
        worst_x = worst_x.max((state.q[2] - slider_x(theta)).abs());
        worst_xdot = worst_xdot.max((state.v[2] - slider_xdot(theta, state.v[0])).abs());
        worst_speed = worst_speed.max(state.v[2].abs());
    }

    println!(
        "slider-crank 5 s @ dt={dt}: max |x - x_analytic| = {worst_x:.3e} m, \
         max |xdot - xdot_analytic| = {worst_xdot:.3e} m/s (peak |xdot| = {worst_speed:.2} m/s), \
         non-converged {nonconverged}/{steps}"
    );
    assert_eq!(nonconverged, 0);
    assert!(worst_x < 2e-3, "slider position error {worst_x:e} m");
    assert!(
        worst_xdot < 5e-2,
        "slider velocity error {worst_xdot:e} m/s"
    );
}

/// A sanity check on the fixture's own algebra, independent of the solver:
/// the analytic `xdot` must be the derivative of the analytic `x`.
#[test]
fn analytic_velocity_is_the_derivative_of_analytic_position() {
    let h = 1e-6;
    let mut worst = 0.0_f64;
    for k in 1..36 {
        let theta = k as f64 * std::f64::consts::TAU / 36.0;
        let fd = (slider_x(theta + h) - slider_x(theta - h)) / (2.0 * h);
        worst = worst.max((fd - slider_xdot(theta, 1.0)).abs());
    }
    println!("analytic dx/dtheta vs finite difference: {worst:.3e} (r={CRANK_R}, l={ROD_L})");
    assert!(worst < 1e-8);
}
