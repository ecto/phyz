//! Body-body contact over the full collision set, and the topology filter
//! that makes it usable.
//!
//! `find_contacts` used to read one centred shape per body, the same gap
//! ecto/phyz#51 closed for the ground. Closing it here is not enough on its
//! own: the moment a humanoid's links carry real geometry, *every joint*
//! becomes a colliding pair, because the two links a joint connects overlap at
//! the joint by construction. Those contacts exist in every configuration, so
//! they carry no information and would bury the one contact anybody wants —
//! the hand that reached the thigh.
//!
//! So detection and filtering ship together, and these tests pin both:
//! offset shapes collide, structurally-adjacent bodies never do, and a
//! genuine self-touch between two non-adjacent limbs does.

use phyz_contact::find_contacts;
use phyz_math::{Mat3, Quat, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{GeomInstance, Geometry, ModelBuilder, State};

fn inertia() -> SpatialInertia {
    SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01)
}

fn unit_box(h: f64) -> Geometry {
    Geometry::Box {
        half_extents: Vec3::new(h, h, h),
    }
}

/// Place a body's world transform directly, bypassing FK, so the geometry
/// under test is unambiguous.
fn place(state: &mut State, body: usize, pos: Vec3) {
    state.body_xform[body] = SpatialTransform::new(Mat3::identity(), pos);
}

// -------------------------------------------------------------------------
// Offset shapes take part at all
// -------------------------------------------------------------------------

/// Two bodies whose *only* geometry is offset inside the link frame must
/// collide when those offset shapes overlap — and must not when only the link
/// origins would.
#[test]
fn offset_shapes_collide_where_the_shapes_are() {
    // Two independent free bodies, so nothing is structurally adjacent.
    let mut model = ModelBuilder::new()
        .add_free_body("a", -1, SpatialTransform::identity(), inertia())
        .add_free_body("b", -1, SpatialTransform::identity(), inertia())
        .build();

    // Each body's box hangs 0.5 m out along +x of its own frame.
    for i in 0..2 {
        model.bodies[i].collisions = vec![GeomInstance::new(
            unit_box(0.1),
            SpatialTransform::new(Mat3::identity(), Vec3::new(0.5, 0.0, 0.0)),
        )];
    }

    let mut state = model.default_state();

    // Link origins 1.0 m apart — but the shapes are both at origin + 0.5 x̂,
    // so they are coincident and must collide.
    place(&mut state, 0, Vec3::zeros());
    place(&mut state, 1, Vec3::new(0.0, 0.0, 0.0));
    assert!(
        !find_contacts(&model, &state).is_empty(),
        "two coincident offset boxes did not collide; offsets are being ignored"
    );

    // Now put the *origins* on top of each other but the shapes far apart, by
    // yawing body b 180° so its box swings to −x.
    state.body_xform[1] = SpatialTransform::new(
        Quat::from_axis_angle(Vec3::z(), -std::f64::consts::PI).to_matrix(),
        Vec3::zeros(),
    );
    assert!(
        find_contacts(&model, &state).is_empty(),
        "boxes 1 m apart reported a contact — the offset is not rotating with \
         the body, so the shape is being placed at the link origin"
    );
}

// -------------------------------------------------------------------------
// The topology filter
// -------------------------------------------------------------------------

/// A parent and its child overlap at the joint by construction. That is not a
/// contact.
#[test]
fn a_joint_is_not_a_collision() {
    let mut model = ModelBuilder::new()
        .add_free_body("root", -1, SpatialTransform::identity(), inertia())
        .add_revolute_body("child", 0, SpatialTransform::identity(), inertia())
        .build();
    for i in 0..2 {
        model.bodies[i].geometry = Some(unit_box(0.2));
    }

    let mut state = model.default_state();
    // Fully coincident: the deepest overlap two boxes can have.
    place(&mut state, 0, Vec3::zeros());
    place(&mut state, 1, Vec3::zeros());

    assert!(
        find_contacts(&model, &state).is_empty(),
        "parent and child reported a contact; every joint in a humanoid would \
         produce one and they would swamp every real self-touch"
    );
}

/// Bodies made rigid by a chain of fixed joints are one physical part. `a` and
/// `c` are not parent and child, so only the weld rule excludes them.
#[test]
fn a_fixed_joint_chain_is_one_part() {
    let mut model = ModelBuilder::new()
        .add_free_body("a", -1, SpatialTransform::identity(), inertia())
        .add_fixed_body("b", 0, SpatialTransform::identity(), inertia())
        .add_fixed_body("c", 1, SpatialTransform::identity(), inertia())
        .build();
    for i in 0..3 {
        model.bodies[i].geometry = Some(unit_box(0.2));
    }

    let mut state = model.default_state();
    for i in 0..3 {
        place(&mut state, i, Vec3::zeros());
    }

    let welds = model.weld_groups();
    assert_eq!(
        welds[0], welds[2],
        "a and c are welded through b but landed in different groups"
    );
    assert!(
        find_contacts(&model, &state).is_empty(),
        "welded bodies reported contacts; a fixed-joint chain is one rigid \
         part and cannot collide with itself"
    );
}

/// The escape hatch, for geometry that overlaps across a *moving* joint — a
/// modelling artifact no rule derived from the tree can distinguish from a
/// real touch.
#[test]
fn an_explicit_exclusion_is_honoured() {
    let mut model = ModelBuilder::new()
        .add_free_body("a", -1, SpatialTransform::identity(), inertia())
        .add_free_body("b", -1, SpatialTransform::identity(), inertia())
        .build();
    for i in 0..2 {
        model.bodies[i].geometry = Some(unit_box(0.2));
    }
    let mut state = model.default_state();
    place(&mut state, 0, Vec3::zeros());
    place(&mut state, 1, Vec3::new(0.1, 0.0, 0.0));

    assert!(
        !find_contacts(&model, &state).is_empty(),
        "two unrelated overlapping bodies should collide"
    );

    model.contact_exclude.push((1, 0)); // order must not matter
    assert!(
        find_contacts(&model, &state).is_empty(),
        "the explicit exclusion was ignored"
    );
}

// -------------------------------------------------------------------------
// The thing this is all for
// -------------------------------------------------------------------------

/// A hand reaching its own thigh: two bodies on different branches of the
/// tree, neither adjacent nor welded, touching because the configuration put
/// them together. This is the contact the filter has to *keep*.
#[test]
fn a_limb_can_touch_another_limb() {
    // root ─ arm (branch 1)
    //      └ leg (branch 2)
    let mut model = ModelBuilder::new()
        .add_free_body("trunk", -1, SpatialTransform::identity(), inertia())
        .add_revolute_body("arm", 0, SpatialTransform::identity(), inertia())
        .add_revolute_body("leg", 0, SpatialTransform::identity(), inertia())
        .build();
    model.bodies[0].geometry = Some(unit_box(0.15));
    model.bodies[1].geometry = Some(unit_box(0.05));
    model.bodies[2].geometry = Some(unit_box(0.05));

    let mut state = model.default_state();
    // Trunk at the origin; arm and leg both out at +x, apart.
    place(&mut state, 0, Vec3::zeros());
    place(&mut state, 1, Vec3::new(1.0, 0.0, 0.3));
    place(&mut state, 2, Vec3::new(1.0, 0.0, -0.3));
    assert!(
        find_contacts(&model, &state).is_empty(),
        "arm and leg are 0.6 m apart"
    );

    // Bring them together. Both are children of the trunk but not of each
    // other, so nothing structural excludes the pair.
    place(&mut state, 1, Vec3::new(1.0, 0.0, 0.03));
    place(&mut state, 2, Vec3::new(1.0, 0.0, -0.03));
    let contacts = find_contacts(&model, &state);
    assert!(
        !contacts.is_empty(),
        "an arm and a leg touching each other produced no contact — this is \
         the whole point of body-body detection"
    );
    for c in &contacts {
        let pair = (c.body_i.min(c.body_j), c.body_i.max(c.body_j));
        assert_eq!(pair, (1, 2), "unexpected pair {pair:?}");
        assert!(
            c.penetration_depth > 0.0,
            "reported a contact at depth {}",
            c.penetration_depth
        );
        assert!(
            c.contact_normal.norm() > 0.5,
            "degenerate contact normal {:?}",
            c.contact_normal
        );
    }
}

/// Two shapes on the *same* body never collide, however they overlap.
#[test]
fn one_body_does_not_collide_with_itself() {
    let mut model = ModelBuilder::new()
        .add_free_body("a", -1, SpatialTransform::identity(), inertia())
        .build();
    model.bodies[0].collisions = vec![
        GeomInstance::centered(unit_box(0.2)),
        GeomInstance::new(
            unit_box(0.2),
            SpatialTransform::new(Mat3::identity(), Vec3::new(0.05, 0.0, 0.0)),
        ),
    ];
    let mut state = model.default_state();
    place(&mut state, 0, Vec3::zeros());
    assert!(find_contacts(&model, &state).is_empty());
}

/// A body with a non-finite transform contributes nothing rather than a
/// contact with a NaN normal.
#[test]
fn a_poisoned_transform_produces_no_contact() {
    let mut model = ModelBuilder::new()
        .add_free_body("a", -1, SpatialTransform::identity(), inertia())
        .add_free_body("b", -1, SpatialTransform::identity(), inertia())
        .build();
    for i in 0..2 {
        model.bodies[i].geometry = Some(unit_box(0.2));
    }
    let mut state = model.default_state();
    place(&mut state, 0, Vec3::zeros());
    place(&mut state, 1, Vec3::new(f64::NAN, 0.0, 0.0));
    assert!(find_contacts(&model, &state).is_empty());
}
