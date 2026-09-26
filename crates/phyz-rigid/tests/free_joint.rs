//! Regression tests for the free joint's DOF ordering and for CRBA/ABA
//! agreeing about armature.
//!
//! # The bug these lock down
//!
//! A free joint's `q` used to be laid out `[x, y, z, wx, wy, wz]` — translation
//! first — while its motion subspace is the 6×6 identity in `SpatialVec`'s
//! `[angular; linear]` order, so `v` and `qdd` were angular first. Every
//! integration site in the workspace then did a flat `q += v·dt`, which put
//! gravity's `qdd[5]` into `q[5]`: the yaw exponential coordinate. A single
//! unit-mass free body released under gravity stayed at `z = 1.0` forever and
//! spun up at 9.81 rad/s².
//!
//! `q` is now `[wx, wy, wz, x, y, z]`, and the configuration update goes
//! through [`phyz_rigid::integrate_configuration`].

use phyz_math::{DVec, Mat3, Quat, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Joint, Model, ModelBuilder, State};
use phyz_rigid::{
    aba, crba, forward_kinematics, integrate_configuration, rnea, semi_implicit_euler,
};

const G: f64 = 9.81;

/// One unit-mass free body, one metre up, nothing else.
fn free_body() -> (Model, State) {
    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -G))
        .dt(0.001)
        .add_body(
            "b",
            -1,
            Joint::free(SpatialTransform::identity()),
            SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity()),
        )
        .build();
    let mut state = model.default_state();
    // q = [wx, wy, wz, x, y, z] — z is slot 5.
    state.q[5] = 1.0;
    let (x, _) = forward_kinematics(&model, &state);
    state.body_xform = x;
    (model, state)
}

#[test]
fn gravity_lands_in_the_last_velocity_dof() {
    let (model, state) = free_body();
    let qdd = aba(&model, &state);
    for i in 0..5 {
        assert!(qdd[i].abs() < 1e-12, "qdd[{i}] = {} should be zero", qdd[i]);
    }
    assert!((qdd[5] + G).abs() < 1e-9, "qdd[5] = {}", qdd[5]);
}

#[test]
fn free_body_falls_at_g_and_does_not_yaw() {
    let (model, mut state) = free_body();
    let t = 0.2;
    let steps = (t / model.dt).round() as usize;
    for _ in 0..steps {
        let qdd = aba(&model, &state);
        semi_implicit_euler(&model, &mut state, qdd.as_slice(), model.dt);
    }
    let (x, _) = forward_kinematics(&model, &state);

    // Semi-implicit Euler over-integrates a constant acceleration by exactly
    // one half-step, so the analytic target is `½g(t² + t·dt)`.
    let expected = 1.0 - 0.5 * G * (t * t + t * model.dt);
    assert!(
        (x[0].pos.z - expected).abs() < 1e-9,
        "z = {}, expected {expected}",
        x[0].pos.z
    );
    // Nothing may leak into the rotational coordinates.
    for i in 0..3 {
        assert!(
            state.q[i].abs() < 1e-14,
            "q[{i}] = {} — the body yawed",
            state.q[i]
        );
    }
    assert!(state.q[3].abs() < 1e-14 && state.q[4].abs() < 1e-14);
    assert!((state.v[5] + G * t).abs() < 1e-9, "vz = {}", state.v[5]);
}

/// A torque-free axisymmetric body spun about a principal axis keeps spinning
/// about it at constant rate, and the exponential-map integrator must track the
/// angle exactly (not just to first order): after `t` the rotation is
/// `exp(ω·t)`.
#[test]
fn free_body_with_initial_spin_precesses_correctly() {
    let model = ModelBuilder::new()
        .gravity(Vec3::zeros())
        .dt(0.001)
        .add_body(
            "b",
            -1,
            Joint::free(SpatialTransform::identity()),
            SpatialInertia::new(
                1.0,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(1.0, 1.0, 2.0)),
            ),
        )
        .build();
    let mut state = model.default_state();
    let omega = 3.0;
    state.v[2] = omega; // spin about the symmetry (z) axis

    let t = 1.0;
    let steps = (t / model.dt).round() as usize;
    for _ in 0..steps {
        let qdd = aba(&model, &state);
        semi_implicit_euler(&model, &mut state, qdd.as_slice(), model.dt);
    }

    // Angular velocity is unchanged (no torque, spin about a principal axis).
    assert!(state.v[0].abs() < 1e-9 && state.v[1].abs() < 1e-9);
    assert!((state.v[2] - omega).abs() < 1e-9);

    // The exponential-map step is exact for constant ω: q = [0, 0, ω·t] mod 2π.
    let expected_angle = omega * t;
    let w = Vec3::new(state.q[0], state.q[1], state.q[2]);
    let angle = Quat::exp(&w).log().z;
    let wrapped = ((expected_angle + std::f64::consts::PI) % (2.0 * std::f64::consts::PI))
        - std::f64::consts::PI;
    assert!(
        w.x.abs() < 1e-9 && w.y.abs() < 1e-9,
        "spin axis drifted: {w:?}"
    );
    assert!(
        (angle - wrapped).abs() < 1e-9,
        "rotation angle {angle}, expected {wrapped}"
    );

    // A torque-free spin translates nowhere.
    for i in 3..6 {
        assert!(state.q[i].abs() < 1e-12, "q[{i}] = {}", state.q[i]);
    }
}

/// A free body given a body-frame linear velocity while rotated must translate
/// along the *world* direction that velocity maps to — the flat `q += v·dt`
/// would translate along the parent axes regardless of orientation.
#[test]
fn free_joint_linear_velocity_is_body_frame() {
    let model = ModelBuilder::new()
        .gravity(Vec3::zeros())
        .dt(0.01)
        .add_body(
            "b",
            -1,
            Joint::free(SpatialTransform::identity()),
            SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity()),
        )
        .build();
    let mut state = model.default_state();
    // Yaw 90° about z, then drive body-frame +x.
    state.q[2] = std::f64::consts::FRAC_PI_2;
    state.v[3] = 1.0;

    let v = state.v.clone();
    integrate_configuration(&model, state.q.as_mut_slice(), v.as_slice(), 1.0);

    // Body +x maps to world +y under a +90° yaw.
    assert!(state.q[3].abs() < 1e-12, "x = {}", state.q[3]);
    assert!((state.q[4] - 1.0).abs() < 1e-12, "y = {}", state.q[4]);
    assert!(state.q[5].abs() < 1e-12, "z = {}", state.q[5]);
}

/// `crba` must include armature, because the contact solver builds its Delassus
/// operator from that matrix while `aba` integrates with the armature applied.
#[test]
fn crba_and_aba_agree_with_armature() {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -G))
        .dt(0.002)
        .add_revolute_body(
            "l1",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(1.3, Vec3::new(0.0, 0.0, -0.4), Mat3::identity() * 0.2),
        )
        .add_revolute_body(
            "l2",
            0,
            SpatialTransform::new(Mat3::identity(), Vec3::new(0.0, 0.0, -0.8)),
            SpatialInertia::new(0.7, Vec3::new(0.0, 0.0, -0.3), Mat3::identity() * 0.1),
        )
        .build();
    model.joints[0].armature = 0.37;
    model.joints[1].armature = 0.11;

    let mut state = model.default_state();
    state.q[0] = 0.3;
    state.q[1] = -0.7;
    state.v[0] = 0.9;
    state.v[1] = -1.4;

    let qdd = aba(&model, &state);
    let m = crba(&model, &state);
    // rnea with qdd = 0 is the bias force (Coriolis + centrifugal + gravity),
    // and it is armature-free by construction: armature multiplies qdd only.
    let bias = rnea(&model, &state, &DVec::zeros(model.nv));
    let m_qdd = &m * &qdd;

    for i in 0..model.nv {
        assert!(
            (m_qdd[i] + bias[i]).abs() < 1e-9,
            "row {i}: M·qdd = {}, -bias = {}",
            m_qdd[i],
            -bias[i]
        );
    }

    // And the armature is actually there: without it `M` would be lighter.
    let mut bare = model.clone();
    bare.joints[0].armature = 0.0;
    bare.joints[1].armature = 0.0;
    let m_bare = crba(&bare, &state);
    assert!((m[(0, 0)] - m_bare[(0, 0)] - 0.37).abs() < 1e-12);
    assert!((m[(1, 1)] - m_bare[(1, 1)] - 0.11).abs() < 1e-12);
}

/// Same check with a floating base, so the multi-DOF branch of CRBA is on the
/// hook for the armature block too.
#[test]
fn crba_and_aba_agree_with_armature_on_a_floating_base() {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -G))
        .dt(0.002)
        .add_body(
            "torso",
            -1,
            Joint::free(SpatialTransform::identity()),
            SpatialInertia::new(
                4.0,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(0.3, 0.4, 0.5)),
            ),
        )
        .add_revolute_body(
            "arm",
            0,
            SpatialTransform::new(Mat3::identity(), Vec3::new(0.1, 0.0, 0.0)),
            SpatialInertia::new(0.5, Vec3::new(0.0, 0.0, -0.2), Mat3::identity() * 0.05),
        )
        .build();
    model.joints[0].armature = 0.05; // all six root DOFs
    model.joints[1].armature = 0.2;

    let mut state = model.default_state();
    state.q[0] = 0.2;
    state.q[2] = -0.1;
    state.q[5] = 1.0;
    state.q[6] = 0.4;
    for i in 0..model.nv {
        state.v[i] = 0.1 * (i as f64 + 1.0);
    }

    let qdd = aba(&model, &state);
    let m = crba(&model, &state);
    let bias = rnea(&model, &state, &DVec::zeros(model.nv));
    let m_qdd = &m * &qdd;
    for i in 0..model.nv {
        assert!(
            (m_qdd[i] + bias[i]).abs() < 1e-8,
            "row {i}: M·qdd = {}, -bias = {}",
            m_qdd[i],
            -bias[i]
        );
    }
}
