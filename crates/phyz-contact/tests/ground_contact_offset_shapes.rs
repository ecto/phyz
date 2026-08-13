//! Ground detection over the full collision set — offsets included.
//!
//! `find_ground_contacts` reads one *centred* shape per body
//! (`Body::geometry`). Sources that place shapes at offsets inside a body
//! (URDF `<collision><origin>`, MJCF geoms) parse into `Body::collisions`,
//! which the ground pipeline never read: on a Booster K1 humanoid exactly 2 of
//! 24 bodies ended up with contact geometry — the two feet — so the robot
//! could not fall onto its side or catch itself with a hand.
//!
//! `find_ground_contacts_model` closes that: every shape in
//! `Body::collisions`, each at its own offset and orientation, competes for
//! the same per-body manifold. These tests pin down:
//! - exact parity with the legacy path for centred single-shape bodies,
//! - correct placement of offset shapes, including under body rotation
//!   (offsets are body-frame directions: world = `rot.transpose() * offset`),
//! - correct orientation composition for shapes rotated inside the body,
//! - the per-body manifold cap across shapes,
//! - the per-contact world-axis drop the convex adjoint anchors need.

use phyz_contact::{
    find_ground_contacts, find_ground_contacts_model, find_ground_contacts_model_with_drop,
};
use phyz_math::{GRAVITY, Mat3, Quat, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{GeomInstance, Geometry, Model, ModelBuilder, State};

fn free_body_model() -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_free_body(
            "b",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01),
        )
        .build()
}

/// Place the body at `pos`, physically rotated by `angle` about `axis`.
fn posed_state(model: &Model, pos: Vec3, axis: Vec3, angle: f64) -> State {
    let mut state = model.default_state();
    // `SpatialTransform::rot` is world→body, so a body physically rotated by
    // `+angle` carries the matrix for `-angle`.
    state.body_xform[0] = SpatialTransform {
        rot: Quat::from_axis_angle(axis, -angle).to_matrix(),
        pos,
    };
    state
}

/// A `GeomInstance` origin for a shape physically rotated by `angle` about
/// `axis` within the body, at `pos` (body coordinates). `origin.rot` is the
/// body→shape coordinate transform, so it carries the matrix for `-angle`.
fn shape_origin(pos: Vec3, axis: Vec3, angle: f64) -> SpatialTransform {
    SpatialTransform {
        rot: Quat::from_axis_angle(axis, -angle).to_matrix(),
        pos,
    }
}

// -------------------------------------------------------------------------
// Parity with the legacy path
// -------------------------------------------------------------------------

/// A centred single shape must produce bit-identical contacts through either
/// entry point, level or tilted, with or without margin — via the legacy
/// `geometry` field and via a centred `collisions` entry alike.
#[test]
fn centred_shape_parity_with_legacy_path() {
    let shapes = [
        Geometry::Box {
            half_extents: Vec3::new(0.09, 0.035, 0.018),
        },
        Geometry::Sphere { radius: 0.05 },
        Geometry::Capsule {
            radius: 0.03,
            length: 0.1,
        },
        Geometry::Cylinder {
            radius: 0.04,
            height: 0.08,
        },
    ];
    for geom in shapes {
        for (height, angle, margin) in [
            (0.01, 0.0, 0.0),
            (0.015, 0.05, 0.0),
            (0.05, -0.08, 1e-2),
            (0.02, 0.3, 1e-3),
        ] {
            let mut model = free_body_model();
            model.bodies[0].geometry = Some(geom.clone());
            let state = posed_state(&model, Vec3::new(0.0, 0.0, height), Vec3::y(), angle);
            let geoms: Vec<Option<Geometry>> =
                model.bodies.iter().map(|b| b.geometry.clone()).collect();

            let legacy = find_ground_contacts(&state, &geoms, 0.0, margin);
            let modeled = find_ground_contacts_model(&model, &state, 0.0, margin);
            assert_contacts_identical(&legacy, &modeled, "legacy geometry field");

            // The same shape as a centred GeomInstance must also agree.
            model.bodies[0].collisions = vec![GeomInstance::centered(geom.clone())];
            let via_collisions = find_ground_contacts_model(&model, &state, 0.0, margin);
            assert_contacts_identical(&legacy, &via_collisions, "centred collisions entry");
        }
    }
}

fn assert_contacts_identical(
    a: &[phyz_collision::Collision],
    b: &[phyz_collision::Collision],
    what: &str,
) {
    assert_eq!(a.len(), b.len(), "{what}: contact count");
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.body_i, y.body_i, "{what}: body");
        assert_eq!(
            (x.contact_point, x.penetration_depth),
            (y.contact_point, y.penetration_depth),
            "{what}: contact identity"
        );
    }
}

// -------------------------------------------------------------------------
// Offset shapes
// -------------------------------------------------------------------------

/// A box hung 10 cm forward of and 5 cm below the body origin must touch the
/// ground where the *box* is, not where the body origin is.
#[test]
fn offset_box_touches_where_the_box_is() {
    let half = Vec3::new(0.02, 0.02, 0.02);
    let offset = Vec3::new(0.10, 0.0, -0.05);
    let mut model = free_body_model();
    model.bodies[0].collisions = vec![GeomInstance::new(
        Geometry::Box { half_extents: half },
        shape_origin(offset, Vec3::z(), 0.0),
    )];

    // Body origin at z = 0.07: box bottom sits at 0.07 − 0.05 − 0.02 = 0.
    let state = posed_state(&model, Vec3::new(0.0, 0.0, 0.069), Vec3::z(), 0.0);
    let contacts = find_ground_contacts_model(&model, &state, 0.0, 0.0);

    assert_eq!(contacts.len(), 4, "one supporting face = four corners");
    for c in &contacts {
        assert!(
            (c.contact_point.x - offset.x).abs() <= half.x + 1e-12,
            "contact at x = {:.4} is not under the offset box",
            c.contact_point.x
        );
        assert!((c.penetration_depth - 0.001).abs() < 1e-9);
    }

    // The legacy path sees nothing: `geometry` is empty, which is precisely
    // the K1 trunk/limb situation this fixes.
    let geoms: Vec<Option<Geometry>> = model.bodies.iter().map(|b| b.geometry.clone()).collect();
    assert!(find_ground_contacts(&state, &geoms, 0.0, 0.0).is_empty());
}

/// The offset is a body-frame vector: yaw the body 180° and the box must
/// swing to the other side. (`rot` used raw instead of transposed is invisible
/// at identity — this is the rotated-state test that catches it.)
#[test]
fn offset_rotates_with_the_body() {
    let half = Vec3::new(0.02, 0.02, 0.02);
    let mut model = free_body_model();
    model.bodies[0].collisions = vec![GeomInstance::new(
        Geometry::Box { half_extents: half },
        shape_origin(Vec3::new(0.10, 0.0, -0.05), Vec3::z(), 0.0),
    )];

    let state = posed_state(
        &model,
        Vec3::new(0.0, 0.0, 0.069),
        Vec3::z(),
        std::f64::consts::PI,
    );
    let contacts = find_ground_contacts_model(&model, &state, 0.0, 0.0);
    assert_eq!(contacts.len(), 4);
    for c in &contacts {
        assert!(
            (c.contact_point.x + 0.10).abs() <= half.x + 1e-9,
            "body yawed π but the box stayed at x = {:.4}; the shape offset \
             is being mapped with `rot` instead of `rot.transpose()`",
            c.contact_point.x
        );
    }
}

/// A shape's own orientation composes with the body's. A cylinder pitched 90°
/// inside the body lies on its side: its lowest material points are on the
/// rim at `z = centre − radius`, not on an end cap.
#[test]
fn shape_orientation_composes_with_body_orientation() {
    let (radius, height) = (0.03, 0.2);
    let mut model = free_body_model();
    // Cylinder axis is local z; pitching the *shape* 90° about y lays the
    // axis along body x.
    model.bodies[0].collisions = vec![GeomInstance::new(
        Geometry::Cylinder { radius, height },
        shape_origin(Vec3::zeros(), Vec3::y(), std::f64::consts::FRAC_PI_2),
    )];

    // Body level, centre at exactly the radius: the lying cylinder just
    // touches. An upright cylinder (orientation ignored) would be 0.07 deep.
    let state = posed_state(&model, Vec3::new(0.0, 0.0, radius - 1e-4), Vec3::z(), 0.0);
    let contacts = find_ground_contacts_model(&model, &state, 0.0, 0.0);
    assert!(!contacts.is_empty(), "a lying cylinder at z < r touches");
    for c in &contacts {
        assert!(
            c.penetration_depth < 2e-4,
            "depth {:.4} m: the shape's own rotation was dropped and the \
             cylinder is being treated as upright",
            c.penetration_depth
        );
        // Rim points of the lying cylinder are at x = ±height/2, not 0.
        assert!(
            (c.contact_point.x.abs() - height * 0.5).abs() < 1e-9,
            "contact at x = {:.4} is not on an end-cap rim of the lying \
             cylinder",
            c.contact_point.x
        );
    }
}

// -------------------------------------------------------------------------
// Multi-shape bodies
// -------------------------------------------------------------------------

/// Two shapes on one body: only the one that reaches the ground contributes,
/// and the per-body manifold cap holds across shapes.
#[test]
fn shapes_compete_for_one_per_body_manifold() {
    let half = Vec3::new(0.02, 0.02, 0.02);
    let mut model = free_body_model();
    model.bodies[0].collisions = vec![
        GeomInstance::new(
            Geometry::Box { half_extents: half },
            shape_origin(Vec3::new(0.1, 0.0, -0.05), Vec3::z(), 0.0),
        ),
        GeomInstance::new(
            Geometry::Box { half_extents: half },
            shape_origin(Vec3::new(-0.1, 0.0, 0.05), Vec3::z(), 0.0),
        ),
    ];

    // Only the low box reaches.
    let state = posed_state(&model, Vec3::new(0.0, 0.0, 0.069), Vec3::z(), 0.0);
    let contacts = find_ground_contacts_model(&model, &state, 0.0, 0.0);
    assert_eq!(contacts.len(), 4);
    assert!(contacts.iter().all(|c| c.contact_point.x > 0.0));

    // Drop the body far enough that both boxes penetrate: 16 corner
    // candidates rank together and the per-body cap still applies.
    let state = posed_state(&model, Vec3::new(0.0, 0.0, -0.2), Vec3::z(), 0.0);
    let contacts = find_ground_contacts_model(&model, &state, 0.0, 0.0);
    assert_eq!(
        contacts.len(),
        phyz_collision::MAX_MANIFOLD_POINTS,
        "two fully-penetrating boxes must still respect the per-body cap"
    );
    // Deepest-first ranking: the cap must keep the *low* box's corners.
    assert!(
        contacts.iter().all(|c| c.contact_point.x > 0.0),
        "the manifold kept shallower corners over deeper ones"
    );
}

// -------------------------------------------------------------------------
// World-axis drop for the adjoint
// -------------------------------------------------------------------------

/// Sphere and capsule contacts carry their radius as the world-axis drop;
/// material-point contacts carry zero. The adjoint anchors on this.
#[test]
fn drop_reports_the_supporting_shape_radius() {
    let mut model = free_body_model();
    model.bodies[0].collisions = vec![
        GeomInstance::new(
            Geometry::Sphere { radius: 0.05 },
            shape_origin(Vec3::new(0.2, 0.0, 0.0), Vec3::z(), 0.0),
        ),
        GeomInstance::new(
            Geometry::Box {
                half_extents: Vec3::new(0.02, 0.02, 0.05),
            },
            shape_origin(Vec3::new(-0.2, 0.0, 0.0), Vec3::z(), 0.0),
        ),
    ];

    let state = posed_state(&model, Vec3::new(0.0, 0.0, 0.049), Vec3::z(), 0.0);
    let contacts = find_ground_contacts_model_with_drop(&model, &state, 0.0, 0.0);
    assert!(!contacts.is_empty());
    for (c, drop) in &contacts {
        if c.contact_point.x > 0.0 {
            assert_eq!(*drop, 0.05, "sphere contact must carry its radius");
        } else {
            assert_eq!(*drop, 0.0, "box corner is a material point");
        }
    }
    assert!(contacts.iter().any(|(c, _)| c.contact_point.x > 0.0));
    assert!(contacts.iter().any(|(c, _)| c.contact_point.x < 0.0));
}
