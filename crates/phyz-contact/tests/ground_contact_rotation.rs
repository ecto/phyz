//! A tilted body's support points must tilt the same way the body does.
//!
//! `find_ground_contacts` built its candidate support points as
//! `pos + rot * offset`. But `SpatialTransform::rot` is the **world→body**
//! rotation — `phyz_rigid::jacobian` says so in a comment and takes the
//! transpose for exactly this reason, when building the Jacobian for these very
//! contacts. Rotating a body-frame offset out to world therefore needs
//! `rot.transpose()`.
//!
//! Using `rot` directly rotates every support offset by `-θ` instead of `+θ`,
//! an error of `2θ`. At identity the two agree exactly, so a level stance looks
//! perfect and the bug only appears the instant something tilts — and then the
//! contact points and the contact Jacobian disagree about which way the body is
//! tilted, which inverts the sign of the whole ankle-torque → COP → COM loop
//! for a walking machine.
//!
//! The observable contradiction is simple enough to state without any of that
//! context: **a rigid box pitched toe-down cannot be loaded at its heel.**

use phyz_collision::Collision;
use phyz_contact::find_ground_contacts;
use phyz_math::{GRAVITY, Mat3, Quat, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder, State};

const HALF_X: f64 = 0.09;
const HALF_Y: f64 = 0.035;
const HALF_Z: f64 = 0.018;

/// A single free box, foot-shaped: long in x, thin in z.
fn foot_model() -> Model {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_free_body(
            "foot",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: Vec3::new(HALF_X, HALF_Y, HALF_Z),
    });
    model
}

/// Place the foot at `height`, pitched by `pitch` radians about +y.
///
/// Positive pitch about +y drops the +x end: toe-down.
fn tilted_state(model: &Model, height: f64, pitch: f64) -> State {
    let mut state = model.default_state();
    // `SpatialTransform::rot` is world→body, so a body physically pitched by
    // `+pitch` carries the rotation matrix for `-pitch`.
    state.body_xform[0] = SpatialTransform {
        rot: Quat::from_axis_angle(Vec3::y(), -pitch).to_matrix(),
        pos: Vec3::new(0.0, 0.0, height),
    };
    state
}

fn geoms(model: &Model) -> Vec<Option<Geometry>> {
    model.bodies.iter().map(|b| b.geometry.clone()).collect()
}

/// The deepest contact on a toe-down foot must be at the toe.
///
/// This is the assertion the old code could not satisfy: it reported the
/// deepest point at the heel, i.e. the box was simultaneously pitched one way
/// and resting on the other end.
#[test]
fn a_toe_down_foot_is_loaded_at_the_toe() {
    let model = foot_model();
    let g = geoms(&model);
    let pitch = 0.05;

    // Low enough that the toe corner penetrates and the heel corner does not.
    let state = tilted_state(&model, HALF_Z, pitch);
    let contacts = find_ground_contacts(&state, &g, 0.0, 0.0);

    assert!(!contacts.is_empty(), "a lowered tilted foot should touch");

    let deepest = contacts
        .iter()
        .max_by(|a, b| {
            a.penetration_depth
                .partial_cmp(&b.penetration_depth)
                .unwrap()
        })
        .expect("non-empty");

    assert!(
        deepest.contact_point.x > 0.0,
        "foot is pitched toe-down (+x end lowered) but the deepest contact is \
         at x = {:.4}, the heel. The support points are being rotated the wrong \
         way — `rot` is world→body, so mapping a body-frame offset to world \
         needs `rot.transpose()`.",
        deepest.contact_point.x
    );
}

/// Pitching the other way must move the load the other way.
#[test]
fn a_heel_down_foot_is_loaded_at_the_heel() {
    let model = foot_model();
    let g = geoms(&model);

    let state = tilted_state(&model, HALF_Z, -0.05);
    let contacts = find_ground_contacts(&state, &g, 0.0, 0.0);
    assert!(!contacts.is_empty());

    let deepest = contacts
        .iter()
        .max_by(|a, b| {
            a.penetration_depth
                .partial_cmp(&b.penetration_depth)
                .unwrap()
        })
        .expect("non-empty");

    assert!(
        deepest.contact_point.x < 0.0,
        "heel-down foot loaded at x = {:.4}, expected the heel (negative x)",
        deepest.contact_point.x
    );
}

/// The corner geometry must match a hand-computed rotation, not merely have the
/// right sign.
///
/// A sign check alone would also pass for a rotation of the wrong magnitude.
#[test]
fn the_support_points_match_the_analytic_corner_positions() {
    let model = foot_model();
    let g = geoms(&model);
    let pitch = 0.08;
    let height = 0.5;

    let state = tilted_state(&model, height, pitch);
    // Margin deep enough to admit every corner as a candidate. The manifold is
    // then capped at `MAX_MANIFOLD_POINTS` deepest-first, so what comes back is
    // the bottom face — which is where the toe-bottom corner lives.
    let contacts = find_ground_contacts(&state, &g, 0.0, 1.0);
    assert_eq!(
        contacts.len(),
        phyz_collision::MAX_MANIFOLD_POINTS,
        "expected a full manifold"
    );

    // The toe-bottom corner `(+HALF_X, ·, -HALF_Z)` under `R_y(pitch)`:
    //   x' =  x·cos + z·sin
    //   z' = -x·sin + z·cos
    let (c, s) = (pitch.cos(), pitch.sin());
    let want_x = HALF_X * c + (-HALF_Z) * s;
    let corner_z = height + (-HALF_X * s + (-HALF_Z) * c);
    // `contact_point` sits on the midsurface between the vertex and the plane,
    // so with the ground at z = 0 it reports half the corner's height.
    let want_z = corner_z * 0.5;

    let best = contacts
        .iter()
        .min_by(|a, b| {
            let da = (a.contact_point.x - want_x).abs() + (a.contact_point.z - want_z).abs();
            let db = (b.contact_point.x - want_x).abs() + (b.contact_point.z - want_z).abs();
            da.partial_cmp(&db).unwrap()
        })
        .expect("non-empty");

    assert!(
        (best.contact_point.x - want_x).abs() < 1e-12
            && (best.contact_point.z - want_z).abs() < 1e-12,
        "closest corner to the analytic toe-bottom ({want_x:.6}, {want_z:.6}) \
         was ({:.6}, {:.6})",
        best.contact_point.x,
        best.contact_point.z
    );
}

/// At identity the two conventions coincide, which is why this survived.
#[test]
fn a_level_foot_is_unaffected_because_identity_hides_the_bug() {
    let model = foot_model();
    let g = geoms(&model);

    let state = tilted_state(&model, HALF_Z * 0.5, 0.0);
    let contacts = find_ground_contacts(&state, &g, 0.0, 0.0);

    assert_eq!(contacts.len(), 4, "a level foot rests on four corners");
    for c in &contacts {
        assert!(
            (c.penetration_depth - HALF_Z * 0.5).abs() < 1e-12,
            "level corners should share one depth, got {}",
            c.penetration_depth
        );
    }
}
