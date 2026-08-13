//! The body-attached contact plane, on device.
//!
//! This is the deck of a skateboard: other bodies stand on a moving body's
//! top face, and the reaction lands on that body. It is deliberately not
//! general body-body contact — an infinite plane welded to a body covers
//! "feet on a deck" at a fraction of a broad-phase's cost, and the exclusion
//! flag keeps the board's own wheels out of it.

use phyz_gpu::GpuBatchSimulator;
use phyz_gpu::contact_pipeline::{BodyContactGains, BodyPlane};
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{GeomInstance, Geometry, Joint, Model, ModelBuilder};

/// Contact natural frequency, rad/s — `omega = 1/tc` for MuJoCo's 0.02 s
/// solref default, so every body rests at `g/omega^2` regardless of mass.
const OMEGA: f64 = 50.0;

fn box_inertia(mass: f64, h: Vec3) -> SpatialInertia {
    let (lx, ly, lz) = (2.0 * h.x, 2.0 * h.y, 2.0 * h.z);
    SpatialInertia::new(
        mass,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(
            mass / 12.0 * (ly * ly + lz * lz),
            mass / 12.0 * (lx * lx + lz * lz),
            mass / 12.0 * (lx * lx + ly * ly),
        )),
    )
}

/// Two free boxes: a deck and a rider. `gravity` toggles the two scenarios.
fn deck_and_rider(gravity: f64) -> (Model, Vec3, Vec3) {
    let deck_half = Vec3::new(0.4, 0.2, 0.05);
    let rider_half = Vec3::new(0.1, 0.1, 0.1);
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, gravity))
        .dt(0.001)
        .add_body(
            "deck",
            -1,
            Joint::free(SpatialTransform::identity()),
            box_inertia(4.0, deck_half),
        )
        .add_body(
            "rider",
            -1,
            Joint::free(SpatialTransform::identity()),
            box_inertia(1.0, rider_half),
        )
        .build();
    model.bodies[0].collisions = vec![GeomInstance::centered(Geometry::Box {
        half_extents: deck_half,
    })];
    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: deck_half,
    });
    model.bodies[1].geometry = Some(Geometry::Box {
        half_extents: rider_half,
    });
    (model, deck_half, rider_half)
}

fn plane(deck_half: Vec3) -> BodyPlane {
    BodyPlane {
        body: 0,
        offset: deck_half.z,
        max_depth: 0.05,
        exclude: vec![],
    }
}

/// Gravity on, deck resting on the ground plane, rider dropped onto the
/// deck: the stack settles at the right heights instead of the rider
/// ghosting through.
#[test]
fn a_rider_settles_on_the_deck() {
    let (model, deck_half, rider_half) = deck_and_rider(-9.81);
    let Ok(mut sim) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.enable_ground_contact_with_plane(
        0.0,
        0.8,
        &BodyContactGains::uniform_frequency(&model, OMEGA, 1.0),
        Some(&plane(deck_half)),
    )
    .expect("contact");

    let mut s = model.default_state();
    // Free-joint q is [exp-coords(3), pos(3)] — angular first.
    s.q[5] = deck_half.z; // deck resting on the ground
    s.q[11] = deck_half.z * 2.0 + rider_half.z + 0.02; // rider 2 cm above the deck
    sim.load_states(std::slice::from_ref(&s));
    for _ in 0..1500 {
        sim.step();
    }
    let out = &sim.readback_states()[0];

    let deck_z = out.q[5];
    let rider_z = out.q[11];
    let deck_top = deck_z + deck_half.z;
    assert!(
        (deck_z - deck_half.z).abs() < 0.02,
        "deck sank or flew: z = {deck_z:.4}"
    );
    assert!(
        (rider_z - (deck_top + rider_half.z)).abs() < 0.02,
        "rider is not sitting on the deck: rider z {rider_z:.4}, deck top {deck_top:.4}"
    );
    // And it is at rest, not vibrating.
    let speed: f64 = (0..model.nv).map(|i| out.v[i] * out.v[i]).sum::<f64>().sqrt();
    assert!(speed < 0.05, "stack still moving at {speed:.3}");
}

/// Zero gravity, rider driven into the deck: the plane must push BOTH
/// bodies — linear momentum along the normal is conserved, which fails
/// instantly if the reaction on the plane body is missing or misframed.
#[test]
fn the_deck_feels_the_rider() {
    let (model, deck_half, rider_half) = deck_and_rider(0.0);
    let Ok(mut sim) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.enable_ground_contact_with_plane(
        // Ground far below so only the deck plane acts.
        -100.0,
        0.8,
        &BodyContactGains::uniform_frequency(&model, OMEGA, 1.0),
        Some(&plane(deck_half)),
    )
    .expect("contact");

    let (m_deck, m_rider) = (4.0, 1.0);
    let v0 = -0.5;
    let mut s = model.default_state();
    s.q[11] = deck_half.z + rider_half.z + 0.005; // just above the face
    s.v[11] = v0; // rider's linear z DOF: second free joint, [ang(3), lin(3)]
    sim.load_states(std::slice::from_ref(&s));
    for _ in 0..400 {
        sim.step();
    }
    let out = &sim.readback_states()[0];

    let (vd, vr) = (out.v[5], out.v[11]);
    assert!(vd < -1e-3, "deck picked up no recoil: v_z = {vd:.5}");
    assert!(vr > v0, "rider was not decelerated: v_z = {vr:.5}");
    let p0 = m_rider * v0;
    let p1 = m_deck * vd + m_rider * vr;
    assert!(
        (p1 - p0).abs() < 0.05 * p0.abs(),
        "momentum leaked: before {p0:.4}, after {p1:.4}"
    );
}

/// Static friction must hold a standing load, not creep under it.
///
/// The regression that motivated the regularized Coulomb law: friction
/// capped at `d * vt` is ~zero at standing slip speeds, so a box with a
/// steady sideways push slid indefinitely. Here a box rests on the ground
/// under gravity with a lateral force well inside the friction cone; it must
/// stay put.
#[test]
fn a_pushed_box_does_not_creep() {
    let half = Vec3::new(0.15, 0.15, 0.05);
    let mass = 4.0;
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_body(
            "box",
            -1,
            Joint::free(SpatialTransform::identity()),
            box_inertia(mass, half),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box { half_extents: half });

    let Ok(mut sim) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.enable_ground_contact_per_body(
        0.0,
        0.8,
        &BodyContactGains::uniform_frequency(&model, OMEGA, 1.0),
    )
    .expect("contact");
    let mut s = model.default_state();
    s.q[5] = half.z;
    sim.load_states(std::slice::from_ref(&s));

    // A steady lateral force at 25% of the friction cone (mu = 0.8).
    let push = 0.25 * 0.8 * mass * 9.81;
    let mut ctrl = vec![0.0; model.nv];
    ctrl[3] = push;
    sim.set_controls(&[ctrl]);
    for _ in 0..2000 {
        sim.step();
    }
    let out = &sim.readback_states()[0];
    let slid = out.q[3].abs();
    assert!(
        slid < 0.01,
        "box crept {slid:.4} m under a force well inside the friction cone"
    );
}
