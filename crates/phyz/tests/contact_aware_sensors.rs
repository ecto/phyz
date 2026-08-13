//! Inertial sensors must see the contacts the solver resolved.
//!
//! `phyz_world::SensorContext` cannot resolve contacts — `phyz-world` does not
//! depend on `phyz-contact`. So the acceleration driving [`Sensor::Imu`],
//! [`Sensor::BodyAccel`] and [`Sensor::ForceTorque`] has to come from the
//! caller that already solved them. These tests pin both halves: what the
//! free-flight acceleration reports (wrong under contact) and what the realized
//! acceleration reports (right).
//!
//! The headline: a body at rest on the ground reads ≈ +9.81 m/s² of specific
//! force along its up axis. A body in free fall reads ≈ 0. Anything that cannot
//! tell those apart does not have a vestibular sense.

use phyz::Simulator;
use phyz_contact::ContactMaterial;
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder, State};
use phyz_world::{Scene, Sensor, SensorContext};

const MASS: f64 = 1.0;
const HALF: f64 = 0.5;
const G: f64 = 9.81;

/// A single free-floating cube. Free-joint `q` is `[rx, ry, rz, x, y, z]`.
fn cube() -> Model {
    let ix = MASS * (2.0 * (2.0 * HALF) * (2.0 * HALF)) / 12.0;
    let mut model = ModelBuilder::new()
        .add_free_body(
            "cube",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                MASS,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(ix, ix, ix)),
            ),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: Vec3::new(HALF, HALF, HALF),
    });
    model
}

/// Drop the cube onto the plane at `z = 0` and let it settle.
fn settled_on_ground(model: &Model, sim: &Simulator, material: &ContactMaterial) -> State {
    let mut state = model.default_state();
    state.q[5] = HALF + 0.02;
    for _ in 0..4000 {
        sim.step_with_contacts(model, &mut state, 0.0, material);
    }
    assert!(
        state.v[5].abs() < 1e-3,
        "cube never settled: vz = {}",
        state.v[5]
    );
    state
}

fn imu(ctx: &SensorContext<'_>) -> Vec<f64> {
    Sensor::Imu { body_idx: 0 }.read(ctx, 0).data
}

/// The headline. Resting on the ground → +g up. Free fall → 0.
#[test]
fn resting_body_reads_plus_g_and_falling_body_reads_zero() {
    let model = cube();
    let sim = Simulator::new();
    let material = ContactMaterial::default();
    let scene = Scene::empty();

    // --- At rest on the ground, with the acceleration the solver realized.
    let state = settled_on_ground(&model, &sim, &material);
    let qdd = sim.contact_acceleration(&model, &state, 0.0, &material);
    let resting = imu(&SensorContext::with_acceleration(
        &model, &state, &scene, &qdd,
    ));
    assert!(
        (resting[2] - G).abs() < 0.05,
        "resting specific force z = {} (want ≈ +{G})",
        resting[2]
    );
    assert!(
        resting[0].abs() < 0.05 && resting[1].abs() < 0.05,
        "resting specific force should be purely vertical, got {resting:?}"
    );

    // --- In free fall, high above the ground. No contacts to find, so the
    // contact-aware acceleration degenerates to the free one and the sensor
    // reads the weightlessness it should.
    let mut falling = model.default_state();
    falling.q[5] = 50.0;
    let qdd = sim.contact_acceleration(&model, &falling, 0.0, &material);
    let free_fall = imu(&SensorContext::with_acceleration(
        &model, &falling, &scene, &qdd,
    ));
    assert!(
        free_fall.iter().take(3).all(|c| c.abs() < 1e-9),
        "free fall should be weightless, got {free_fall:?}"
    );
}

/// The bug and the fix, side by side on one state. This is the regression
/// guard: if `with_acceleration` ever silently degrades to the free
/// acceleration, the two readings collapse together and this fails.
#[test]
fn free_acceleration_context_is_contact_blind() {
    let model = cube();
    let sim = Simulator::new();
    let material = ContactMaterial::default();
    let scene = Scene::empty();
    let state = settled_on_ground(&model, &sim, &material);

    // Before: `SensorContext::free_flight` (what `new` used to do silently).
    // The free acceleration has the resting cube falling at g, so `a − g`
    // cancels and the accelerometer reports weightlessness.
    let blind = imu(&SensorContext::free_flight(&model, &state, &scene));
    assert!(
        blind[2].abs() < 0.5,
        "free-acceleration IMU should read ≈0 on a resting body, got {}",
        blind[2]
    );

    // After: the realized acceleration.
    let qdd = sim.contact_acceleration(&model, &state, 0.0, &material);
    let aware = imu(&SensorContext::with_acceleration(
        &model, &state, &scene, &qdd,
    ));
    assert!(
        (aware[2] - G).abs() < 0.05,
        "contact-aware IMU should read ≈+{G}, got {}",
        aware[2]
    );

    // The whole point: these are different answers to the same question.
    assert!(
        (aware[2] - blind[2]).abs() > 9.0,
        "contact-aware and contact-blind readings must differ: {} vs {}",
        aware[2],
        blind[2]
    );
}

/// `BodyAccel` is a kinematic acceleration, so the resting cube must read ≈0
/// — it is not moving — while the contact-blind context has it falling at g.
#[test]
fn body_accel_sees_the_contact() {
    let model = cube();
    let sim = Simulator::new();
    let material = ContactMaterial::default();
    let scene = Scene::empty();
    let state = settled_on_ground(&model, &sim, &material);

    let blind = Sensor::BodyAccel { body_idx: 0 }
        .read(&SensorContext::free_flight(&model, &state, &scene), 0)
        .data;
    assert!(
        (blind[2] + G).abs() < 0.5,
        "free-acceleration BodyAccel should read ≈−{G}, got {}",
        blind[2]
    );

    let qdd = sim.contact_acceleration(&model, &state, 0.0, &material);
    let aware = Sensor::BodyAccel { body_idx: 0 }
        .read(
            &SensorContext::with_acceleration(&model, &state, &scene, &qdd),
            0,
        )
        .data;
    assert!(
        aware[2].abs() < 0.05,
        "a resting cube is not accelerating, got {}",
        aware[2]
    );
}

/// The reaction wrench at the root joint must carry the cube's weight, not the
/// zero of an unresisted free fall.
#[test]
fn force_torque_reports_the_contact_load() {
    let model = cube();
    let sim = Simulator::new();
    let material = ContactMaterial::default();
    let scene = Scene::empty();
    let state = settled_on_ground(&model, &sim, &material);

    let read = |ctx: &SensorContext<'_>| Sensor::ForceTorque { body_idx: 0 }.read(ctx, 0).data;

    // Contact-blind: the body is in free fall, so no net force is transmitted.
    let blind = read(&SensorContext::free_flight(&model, &state, &scene));
    assert!(
        blind[5].abs() < 0.5,
        "free-acceleration ForceTorque should see no support force, got {}",
        blind[5]
    );

    // Contact-aware: the ground pushes up with m·g.
    let qdd = sim.contact_acceleration(&model, &state, 0.0, &material);
    let aware = read(&SensorContext::with_acceleration(
        &model, &state, &scene, &qdd,
    ));
    assert!(
        (aware[5] - MASS * G).abs() < 0.05 * MASS * G,
        "ForceTorque should carry the weight m·g = {}, got {}",
        MASS * G,
        aware[5]
    );
}

/// The acceleration `step_with_contacts` returns is the one the step realized,
/// so it reconstructs the same sensor reading as `contact_acceleration` at the
/// pre-step state.
#[test]
fn step_with_contacts_returns_the_realized_acceleration() {
    let model = cube();
    let sim = Simulator::new();
    let material = ContactMaterial::default();
    let scene = Scene::empty();

    let mut state = settled_on_ground(&model, &sim, &material);
    let before = state.clone();
    let stepped_qdd = sim.step_with_contacts(&model, &mut state, 0.0, &material);

    let f = imu(&SensorContext::with_acceleration(
        &model,
        &before,
        &scene,
        &stepped_qdd,
    ));
    assert!(
        (f[2] - G).abs() < 0.05,
        "IMU from the step's own qdd should read ≈+{G}, got {}",
        f[2]
    );
}

/// A mismatched acceleration is a bug, not a reading. Fail loudly.
#[test]
#[should_panic(expected = "qdd has 3 entries but the model has 6 DOF")]
fn wrong_length_acceleration_panics() {
    let model = cube();
    let state = model.default_state();
    let scene = Scene::empty();
    let qdd = phyz_math::DVec::zeros(3);
    let _ = SensorContext::with_acceleration(&model, &state, &scene, &qdd);
}
