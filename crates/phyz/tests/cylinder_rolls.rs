//! A cylinder released rolling must roll.
//!
//! This is the end-to-end half of the analytic cylinder-ground contact: the
//! detector tests in `phyz-contact` pin the geometry, and these pin what the
//! integrated rollout does with it.
//!
//! The measurement that motivated the change, run through this file's own
//! fixture at `r = 0.1 m`, released rolling without slipping at 1.2 m/s:
//!
//! | contact model            | distance in 3 s | speed at 3 s | steady axle ripple |
//! |--------------------------|-----------------|--------------|--------------------|
//! | eight body-frame samples | **0.227 m**     | 0.000 m/s    | 4e-16 m            |
//! | analytic lowest line     | **3.905 m**     | 1.399 m/s    | 8.7e-10 m          |
//! | sphere, same radius      | 3.931 m         | 1.417 m/s    | 1.7e-9 m           |
//!
//! The old cylinder stopped after 23 cm because it was a polygon: it climbed
//! `r(1 − cos 45°)` — 29 mm on this wheel — four times per turn, spent its
//! kinetic energy doing it, and fell onto a corner. Its ripple column reads
//! `4e-16` for the same reason its speed column reads zero: by the time the
//! window opens it is lying still. The new one tracks a sphere of the same
//! radius to within 0.7%, and the sphere's own numbers are unchanged to the
//! bit, which is the check that this change touched only cylinders.
//!
//! # What these tests do not claim
//!
//! Both the cylinder and the sphere *gain* about 5% of speed per second here,
//! and that is a property of the contact model rather than of either shape:
//! detection reports the contact point on the midsurface, half a penetration
//! depth above the surface the impulse is really applied at, so the friction
//! impulse's moment arm is slightly too long for a rolling body. It is
//! pre-existing, it is identical for the sphere, and it is why the comparison
//! below is cylinder-against-sphere rather than cylinder-against-conservation.

use phyz::{
    ContactMaterial, Geometry, Mat3, Model, ModelBuilder, Simulator, SpatialInertia,
    SpatialTransform, State, Vec3,
};

const GROUND: f64 = 0.0;
const R: f64 = 0.1;
const HEIGHT: f64 = 0.04;
const MASS: f64 = 2.0;
/// Rolling without slipping at `omega * r`.
const OMEGA: f64 = 12.0;

fn body(geom: Geometry, inertia: Mat3) -> Model {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(1e-3)
        .add_free_body(
            "b",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(MASS, Vec3::zeros(), inertia),
        )
        .build();
    model.bodies[0].geometry = Some(geom);
    model
}

/// A wheel: axis laid along world `−y`, rolling in `+x`.
fn wheel() -> (Model, State) {
    let i_axial = 0.5 * MASS * R * R;
    let i_diam = MASS * (3.0 * R * R + HEIGHT * HEIGHT) / 12.0;
    let model = body(
        Geometry::Cylinder {
            radius: R,
            height: HEIGHT,
        },
        Mat3::from_diagonal(&Vec3::new(i_diam, i_diam, i_axial)),
    );
    let mut state = model.default_state();
    state.q[0] = std::f64::consts::FRAC_PI_2;
    state.q[5] = R + 2e-3;
    // The axle is the shape's own z. `v[2]` is the body's z; `v[1]` would be
    // the body's y, which after the roll points at the sky.
    state.v[2] = -OMEGA;
    state.v[3] = OMEGA * R;
    (model, state)
}

/// The same wheel as a sphere: same radius, same mass, same rolling release.
fn ball() -> (Model, State) {
    let model = body(
        Geometry::Sphere { radius: R },
        Mat3::identity() * (0.4 * MASS * R * R),
    );
    let mut state = model.default_state();
    state.q[5] = R + 2e-3;
    state.v[1] = OMEGA;
    state.v[3] = OMEGA * R;
    (model, state)
}

struct Roll {
    distance: f64,
    /// Mean forward speed over the last 0.5 s.
    final_speed: f64,
    /// Peak-to-peak variation of the body's height over the last second —
    /// steady state, with the release transient excluded. Over the whole roll
    /// this reads 1.5e-4 m for the cylinder and 5.9e-5 m for the sphere, and
    /// both of those are the 2 mm release height damping out, not the shape.
    height_ripple: f64,
}

fn roll(model: &Model, state: &State, steps: usize) -> Roll {
    let sim = Simulator::new();
    let mat = ContactMaterial::default();
    let mut s = state.clone();
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut x_at_minus_half = 0.0;
    for step in 0..steps {
        sim.step_with_contacts(model, &mut s, GROUND, &mat);
        if step >= steps.saturating_sub(1000) {
            lo = lo.min(s.q[5]);
            hi = hi.max(s.q[5]);
        }
        if step + 1 == steps.saturating_sub(500) {
            x_at_minus_half = s.q[3];
        }
    }
    Roll {
        distance: s.q[3],
        final_speed: (s.q[3] - x_at_minus_half) / 0.5,
        height_ripple: hi - lo,
    }
}

/// A rolling cylinder must behave like a rolling sphere of the same radius,
/// not like a polygon.
///
/// The single number this test exists for: 3.9 m instead of 0.23 m.
#[test]
fn a_rolling_cylinder_coasts_like_a_ball_of_the_same_radius() {
    let (wm, ws) = wheel();
    let (bm, bs) = ball();
    let w = roll(&wm, &ws, 3000);
    let b = roll(&bm, &bs, 3000);
    println!(
        "cylinder: {:.4} m, {:.4} m/s | sphere: {:.4} m, {:.4} m/s",
        w.distance, w.final_speed, b.distance, b.final_speed
    );

    // The old sampler stopped this wheel at 0.227 m. Anything in that class is
    // a regression to a polygon, and the bar is set far enough above it that
    // the test is about rolling rather than about the exact number.
    assert!(
        w.distance > 3.0,
        "the cylinder travelled {:.4} m in 3 s from a 1.2 m/s rolling release; \
         the eight-rim-sample contact managed 0.227 m and this is that class of \
         failure",
        w.distance
    );
    // And it must track the sphere, which is the shape-independent statement:
    // whatever the contact model does to a rolling body, it must do the same
    // thing to both.
    let rel = (w.distance - b.distance).abs() / b.distance;
    assert!(
        rel < 0.05,
        "cylinder {:.4} m vs sphere {:.4} m: {:.1}% apart, so rolling still \
         costs something the shape should not cost",
        w.distance,
        b.distance,
        100.0 * rel
    );
    // Neither is allowed to lose its roll: a wheel that dissipates is the bug.
    assert!(
        w.final_speed > 0.9 * OMEGA * R,
        "the cylinder ended at {:.4} m/s from {:.4} m/s",
        w.final_speed,
        OMEGA * R
    );
}

/// The axle height must not move while the wheel turns.
///
/// This is the ripple of `GAP_CYLINDER_GROUND` measured through the
/// integrator rather than through the detector: 29 mm of climb per quarter
/// turn on a 100 mm wheel becomes 0.9 nm of solver residual — half of what
/// the sphere in the same fixture shows.
#[test]
fn the_axle_does_not_climb_its_own_rim() {
    let (wm, ws) = wheel();
    let w = roll(&wm, &ws, 3000);
    let (bm, bs) = ball();
    let b = roll(&bm, &bs, 3000);
    println!(
        "ripple cyl {:.3e} m  sphere {:.3e} m",
        w.height_ripple, b.height_ripple
    );
    // The old contact's ripple is r(1 - cos 45 deg) = 29 mm here. Seven orders
    // of magnitude below that is the claim; the bar is set at the sphere's own
    // residual times ten, because what is left is the penetration breathing
    // under the solver and not the shape.
    assert!(
        w.height_ripple < 10.0 * b.height_ripple,
        "the axle height moved {:.3e} m while rolling, against a sphere's \
         {:.3e} m in the same fixture — the wheel is still climbing its own rim",
        w.height_ripple,
        b.height_ripple
    );
}

/// Two rollouts of the same rolling wheel must be bit-identical.
///
/// Rolling is the case where the contact *point* moves every step while the
/// contact is physically the same feature, so it is the one that catches a
/// detector whose output order depends on anything but the state.
#[test]
fn a_rolling_wheel_replays_bit_for_bit() {
    let (wm, ws) = wheel();
    let a = roll(&wm, &ws, 1500);
    let b = roll(&wm, &ws, 1500);
    assert_eq!(
        a.distance.to_bits(),
        b.distance.to_bits(),
        "two rolls of the same wheel ended at {} and {}",
        a.distance,
        b.distance
    );
    assert_eq!(a.height_ripple.to_bits(), b.height_ripple.to_bits());
}
