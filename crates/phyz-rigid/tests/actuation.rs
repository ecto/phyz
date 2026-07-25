//! Joint limits, actuator gear/ctrlrange, armature, springs and dry friction.

use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Actuator, Model, ModelBuilder};
use phyz_rigid::aba;

/// Single link of length 1, mass 1, rotating about +Z, gravity along -Y.
/// Joint angle q measures rotation from the "hanging down" configuration.
fn pendulum() -> Model {
    let (length, mass) = (1.0, 1.0);
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001)
        .add_revolute_body(
            "link",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                mass,
                Vec3::new(0.0, -length / 2.0, 0.0),
                Mat3::from_diagonal(&Vec3::new(
                    mass * length * length / 12.0,
                    0.0,
                    mass * length * length / 12.0,
                )),
            ),
        )
        .build()
}

fn motor(model: &mut Model, gear: f64, ctrl_range: Option<[f64; 2]>) {
    let mut act = Actuator::motor("m", "link", 0);
    act.gear = gear;
    act.ctrl_range = ctrl_range;
    model.actuators.push(act);
}

/// Semi-implicit Euler for the duration, returning (q, qd) history extremes.
fn simulate(model: &Model, state: &mut phyz_model::State, dt: f64, steps: usize) {
    for _ in 0..steps {
        let qdd = aba(model, state);
        for i in 0..model.nv {
            state.v[i] += dt * qdd[i];
            state.q[i] += dt * state.v[i];
        }
        state.time += dt;
    }
}

#[test]
fn joint_limit_stops_swing() {
    let mut model = pendulum();
    // Limit well before the pendulum would otherwise swing.
    model.joints[0].limits = Some([-0.2, 0.5]);
    model.joints[0].limit_stiffness = 5000.0;
    model.joints[0].limit_damping = 100.0;

    let mut state = model.default_state();
    state.q[0] = 0.5; // start at the upper limit
    state.v[0] = 2.0; // driving further into it

    simulate(&model, &mut state, 1e-4, 20_000);

    assert!(
        state.q[0] <= 0.5 + 0.05,
        "tunneled/overshot upper limit: q = {}",
        state.q[0]
    );
    assert!(
        state.q[0] >= -0.2 - 0.05,
        "tunneled through lower limit: q = {}",
        state.q[0]
    );
}

#[test]
fn joint_limit_no_tunneling_at_large_dt() {
    // The gated damping term makes the limit dissipative, so even a coarse
    // timestep cannot pump energy at the boundary.
    let mut model = pendulum();
    model.joints[0].limits = Some([-0.3, 0.3]);
    model.joints[0].limit_stiffness = 500.0;
    model.joints[0].limit_damping = 50.0;

    let mut state = model.default_state();
    state.v[0] = 3.0;

    let dt = 5e-3;
    let mut max_q: f64 = 0.0;
    for _ in 0..4000 {
        simulate(&model, &mut state, dt, 1);
        max_q = max_q.max(state.q[0].abs());
        assert!(state.q[0].is_finite(), "diverged");
    }

    assert!(max_q < 1.0, "escaped the limit band: max |q| = {}", max_q);
    // Energy is removed at the limit, so it settles rather than ringing forever.
    assert!(
        state.v[0].abs() < 3.0,
        "limit added energy: final qd = {}",
        state.v[0]
    );
}

#[test]
fn joint_limit_settles_without_oscillation() {
    let mut model = pendulum();
    model.joints[0].limits = Some([-0.4, 0.4]);
    // Defaults must be usable out of the box for a unit-scale link.
    let mut state = model.default_state();
    state.q[0] = 0.4;

    // Gravity pushes it toward -q, so hold it at the limit with a torque.
    state.ctrl[0] = 20.0;
    simulate(&model, &mut state, 1e-4, 30_000);

    assert!(state.q[0].is_finite());
    assert!(
        state.q[0] > 0.4 && state.q[0] < 0.4 + 0.1,
        "resting penetration out of range: q = {}",
        state.q[0]
    );
    assert!(
        state.v[0].abs() < 1e-2,
        "still oscillating at the limit: qd = {}",
        state.v[0]
    );
}

#[test]
fn unlimited_joint_swings_freely() {
    let model = pendulum();
    let mut state = model.default_state();
    state.q[0] = 1.0;
    simulate(&model, &mut state, 1e-4, 5_000);
    assert!(
        state.q[0] < 1.0,
        "no limits configured, pendulum should fall: q = {}",
        state.q[0]
    );
}

#[test]
fn gear_scales_joint_torque() {
    let mut m1 = pendulum();
    motor(&mut m1, 1.0, None);
    let mut m5 = pendulum();
    motor(&mut m5, 5.0, None);

    let mut s1 = m1.default_state();
    s1.ctrl[0] = 0.7;
    let mut s5 = m5.default_state();
    s5.ctrl[0] = 0.7;

    // Subtract the (identical) gravity-only acceleration to isolate the torque.
    let s0 = m1.default_state();
    let a0 = aba(&m1, &s0)[0];

    let a1 = aba(&m1, &s1)[0] - a0;
    let a5 = aba(&m5, &s5)[0] - a0;

    assert!(a1.abs() > 1e-6, "gear=1 produced no torque");
    assert!(
        (a5 - 5.0 * a1).abs() < 1e-9,
        "gear=5 gave {} vs 5x{} = {}",
        a5,
        a1,
        5.0 * a1
    );
}

#[test]
fn ctrl_is_clamped_to_ctrl_range() {
    let mut model = pendulum();
    motor(&mut model, 2.0, Some([-1.0, 1.0]));

    let mut over = model.default_state();
    over.ctrl[0] = 50.0;
    let mut at_limit = model.default_state();
    at_limit.ctrl[0] = 1.0;
    let mut under = model.default_state();
    under.ctrl[0] = -50.0;
    let mut at_lower = model.default_state();
    at_lower.ctrl[0] = -1.0;

    assert!((aba(&model, &over)[0] - aba(&model, &at_limit)[0]).abs() < 1e-12);
    assert!((aba(&model, &under)[0] - aba(&model, &at_lower)[0]).abs() < 1e-12);
    // And an in-range value is not clamped.
    let mut mid = model.default_state();
    mid.ctrl[0] = 0.5;
    assert!((aba(&model, &mid)[0] - aba(&model, &at_limit)[0]).abs() > 1e-6);
}

#[test]
fn actuator_force_matches_raw_torque() {
    // gear * clamp(ctrl) must equal the equivalent unactuated per-DOF torque.
    let mut geared = pendulum();
    motor(&mut geared, 3.0, Some([-2.0, 2.0]));
    let raw = pendulum(); // no actuators: ctrl is a raw per-DOF torque

    let mut sg = geared.default_state();
    sg.ctrl[0] = 5.0; // clamps to 2.0 -> 6.0 N·m
    let mut sr = raw.default_state();
    sr.ctrl[0] = 6.0;

    assert!((aba(&geared, &sg)[0] - aba(&raw, &sr)[0]).abs() < 1e-12);
}

#[test]
fn armature_reduces_acceleration() {
    let model = pendulum();
    let mut with_rotor = pendulum();
    with_rotor.joints[0].armature = 1.0;

    let mut s = model.default_state();
    s.q[0] = 0.5;
    let mut sr = with_rotor.default_state();
    sr.q[0] = 0.5;

    let a = aba(&model, &s)[0].abs();
    let ar = aba(&with_rotor, &sr)[0].abs();
    assert!(ar < a, "armature should add inertia: {} vs {}", ar, a);
}

#[test]
fn spring_pulls_toward_springref() {
    let mut model = pendulum();
    model.gravity = Vec3::zeros();
    model.joints[0].stiffness = 10.0;
    model.joints[0].spring_ref = 0.3;

    let mut s = model.default_state();
    s.q[0] = 0.0;
    assert!(aba(&model, &s)[0] > 0.0, "spring should pull q up to 0.3");

    s.q[0] = 0.6;
    assert!(aba(&model, &s)[0] < 0.0, "spring should pull q down to 0.3");

    s.q[0] = 0.3;
    assert!(aba(&model, &s)[0].abs() < 1e-9, "at rest at springref");
}

#[test]
fn dry_friction_opposes_motion() {
    let mut model = pendulum();
    model.gravity = Vec3::zeros();
    model.joints[0].friction_loss = 0.5;

    let mut s = model.default_state();
    s.v[0] = 1.0;
    assert!(aba(&model, &s)[0] < 0.0);

    s.v[0] = -1.0;
    assert!(aba(&model, &s)[0] > 0.0);

    s.v[0] = 0.0;
    assert!(aba(&model, &s)[0].abs() < 1e-9, "no friction at rest");
}
