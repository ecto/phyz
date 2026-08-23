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
        half_x: deck_half.x,
        half_y: deck_half.y,
        exclude: vec![],
        tilt: phyz_math::Mat3::identity(),
        center: phyz_math::Vec3::zeros(),
    }
}

/// The face supports what is over it and nothing else.
///
/// An unbounded face is harmless while the surface is level and wrong the
/// moment it pitches — a skateboard pop measured 14 deg of nose lift on the
/// GPU against the CPU solver's 22 deg, because a foot beyond the nose was
/// still pressing down through the phantom part of the plane.
#[test]
fn the_face_does_not_support_what_is_past_its_edge() {
    let (model, deck_half, rider_half) = deck_and_rider(-9.81);
    let Ok(mut sim) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.enable_ground_contact_with_plane(
        -100.0,
        0.8,
        &BodyContactGains::uniform_frequency(&model, OMEGA, 1.0),
        std::slice::from_ref(&plane(deck_half)),
    )
    .expect("contact");

    // Rider placed just past the deck's edge, at face height.
    let mut s = model.default_state();
    s.q[9] = deck_half.x + rider_half.x + 0.05;
    s.q[11] = deck_half.z + rider_half.z;
    sim.load_states(std::slice::from_ref(&s));
    for _ in 0..300 {
        sim.step();
    }
    let out = &sim.readback_states()[0];
    assert!(
        out.v[11] < -1e-3,
        "a body past the face's edge was still held up by it (v_z = {:.5})",
        out.v[11]
    );
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
        std::slice::from_ref(&plane(deck_half)),
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
    let speed: f64 = (0..model.nv)
        .map(|i| out.v[i] * out.v[i])
        .sum::<f64>()
        .sqrt();
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
        std::slice::from_ref(&plane(deck_half)),
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

/// The face pass reports what it solved, in a block of its own.
///
/// Until ecto/phyz#85 it reported nothing: the readback slot is per body and
/// already held the GROUND result, so writing a face contact there would have
/// erased whatever else that body was standing on. The consequence was that a
/// foot pressing on a deck produced no row at all from the host — the exact
/// contact set the pre-tip parity gap turns on was unobservable, and its
/// absence read like the rider standing on nothing.
///
/// So: rider resting on the deck, deck resting on the ground.
///   - the DECK's ground block is unchanged and still reports the ground;
///   - the RIDER's face block reports the deck, 4 corners, normal +z, and a
///     total normal force equal to the rider's weight.
#[test]
fn the_face_pass_reports_its_own_contacts() {
    let (model, deck_half, rider_half) = deck_and_rider(-9.81);
    let Ok(mut sim) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.enable_ground_contact_with_plane(
        0.0,
        0.8,
        &BodyContactGains::uniform_frequency(&model, OMEGA, 1.0),
        std::slice::from_ref(&plane(deck_half)),
    )
    .expect("contact");

    let mut s = model.default_state();
    s.q[5] = deck_half.z;
    s.q[11] = 2.0 * deck_half.z + rider_half.z;
    sim.load_states(std::slice::from_ref(&s));
    for _ in 0..2000 {
        sim.step();
    }

    let c = &sim.readback_contacts().expect("contact readback")[0];
    let (deck, rider) = (&c[0], &c[1]);

    // The deck still reports the GROUND in its own block, untouched by the
    // face pass — the whole reason the face got a second block.
    assert!(deck.touching, "the deck is on the ground: {deck:?}");
    assert!(
        deck.force.z > 30.0,
        "the deck's ground block should carry both masses (~49 N), got {:?}",
        deck.force
    );
    // ...and no face block of its own: nothing is standing on the deck's
    // *underside*, and the deck is excluded from its own face.
    assert!(
        !deck.plane.touching,
        "the deck is on no face: {:?}",
        deck.plane
    );

    // The rider is on the deck and on nothing else.
    assert!(!rider.touching, "the rider is not on the ground: {rider:?}");
    let f = &rider.plane;
    assert!(f.touching, "the rider is on the deck's face: {f:?}");
    assert_eq!(f.points, 4, "a box rests on a face through four corners");
    assert!(
        f.penetration > 0.0 && f.penetration < 5e-3,
        "settled penetration should be sub-millimetric, got {:.4} mm",
        f.penetration * 1000.0
    );
    assert!(
        (f.force.z - 9.81).abs() < 1.0,
        "the face should carry the rider's weight (9.81 N), got {:?}",
        f.force
    );
    // Every point names the same face, agrees on the normal, and the normal
    // forces sum to the aggregate — this is what makes the block usable as a
    // per-pair manifold table against the CPU narrow phase.
    let mut total = 0.0;
    for (k, d) in f.detail[..f.points].iter().enumerate() {
        assert_eq!(d.plane, Some(0), "point {k} names the wrong face: {d:?}");
        assert!(
            (d.normal - Vec3::new(0.0, 0.0, 1.0)).norm() < 1e-5,
            "point {k} normal should be world +z, got {:?}",
            d.normal
        );
        total += d.normal_force;
    }
    assert!(
        (total - f.force.z).abs() < 1e-2 * (1.0 + f.force.z.abs()),
        "per-point normal forces {total:.4} N do not sum to the aggregate {:.4} N",
        f.force.z
    );
    assert_eq!(
        f.points_on(0).count(),
        4,
        "points_on should group the manifold by face"
    );
}

/// A compound face set ranks by DEPTH, not by table order.
///
/// This is a concave deck: a low centre strip and two rails a centimetre
/// proud of it, as a convex decomposition produces. A foot lying across it
/// rests on the RAILS — that is what concave is for — and the centre strip
/// carries nothing.
///
/// The pass used to run face-major and hand each body its warm-start ranks in
/// FACE order. With two faces that is harmless. With a strip set it is not:
/// the centre strip is first in the table and shallowest, so it would claim
/// slots that the load-bearing rails needed, and the rider would sink through
/// the rails to the centre plane. Hence the centre face is deliberately
/// listed first here.
///
/// Measured cost of getting this wrong, on ipse's pre-tip: modelling the deck
/// as one flat rectangle put the rider's rear foot 5.4 mm ABOVE the face at
/// the spawn, so the device found no foot/deck contact at all where the CPU
/// found eleven points carrying ~95% of the rig (ecto/phyz#85).
#[test]
fn a_strip_set_ranks_by_depth_not_by_table_order() {
    let deck_half = Vec3::new(0.4, 0.2, 0.05);
    let rider_half = Vec3::new(0.1, 0.2, 0.1);
    let rail_rise = 0.01;
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
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
    model.bodies[1].geometry = Some(Geometry::Box {
        half_extents: rider_half,
    });

    let strip = |cy: f64, hy: f64, lift: f64| BodyPlane {
        body: 0,
        offset: deck_half.z,
        max_depth: 0.05,
        half_x: deck_half.x,
        half_y: hy,
        exclude: vec![],
        tilt: Mat3::identity(),
        center: Vec3::new(0.0, cy, lift),
    };
    // Centre first, on purpose: table order must not decide the manifold.
    let planes = [
        strip(0.0, 0.08, 0.0),
        strip(0.14, 0.06, rail_rise),
        strip(-0.14, 0.06, rail_rise),
    ];

    let Ok(mut sim) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.enable_ground_contact_with_plane(
        0.0,
        0.8,
        &BodyContactGains::uniform_frequency(&model, OMEGA, 1.0),
        &planes,
    )
    .expect("contact");

    let mut s = model.default_state();
    s.q[5] = deck_half.z;
    s.q[11] = 2.0 * deck_half.z + rail_rise + rider_half.z + 0.01;
    sim.load_states(std::slice::from_ref(&s));
    for _ in 0..3000 {
        sim.step();
    }

    let out = &sim.readback_states()[0];
    // Measured deck-to-rider, so the deck's own sag into the ground does not
    // enter: a penalty contact rests at g/omega^2 by construction and both
    // stacks carry different loads.
    let gap = out.q[11] - out.q[5];
    let on_rails = deck_half.z + rail_rise + rider_half.z;
    let on_centre = deck_half.z + rider_half.z;
    assert!(
        gap > 0.5 * (on_rails + on_centre),
        "the rider should rest on the RAILS (gap ~{on_rails:.4}), not sink to \
         the centre strip (~{on_centre:.4}); got {gap:.4}"
    );

    let f = &sim.readback_contacts().expect("readback")[0][1].plane;
    assert!(
        f.touching && f.points > 0,
        "the rider is on the deck: {f:?}"
    );
    for (k, d) in f.detail[..f.points].iter().enumerate() {
        assert_ne!(
            d.plane,
            Some(0),
            "point {k} landed on the centre strip, which carries nothing here: {d:?}"
        );
    }
    // ...and it is on BOTH rails, not balanced on one.
    assert!(
        f.points_on(1).count() > 0 && f.points_on(2).count() > 0,
        "the rider should be carried by both rails: {f:?}"
    );
}
