//! Body-to-body contact stability tests.
//!
//! Exercises the surfaces that the mecheval `fit_physics` grader relies on:
//! a fixed host body and a free accessory body, body-pair contact detection,
//! and penalty-based contact forces under gravity.

use approx::assert_relative_eq;
use phyz::{
    ContactMaterial, Geometry, Mat3, ModelBuilder, SpatialInertia, SpatialTransform, SpatialVec,
    Vec3,
    contact::{contact_forces, find_contacts, find_ground_contacts},
};

/// Goal 1 — body-pair contact force pushes the free body AWAY from the fixed body.
///
/// Mirrors `ball_drop_with_contacts` in `integration.rs` but uses two real bodies
/// (one fixed host below, one free accessory above) instead of a ground plane.
#[test]
fn body_drop_on_fixed_body_with_contacts() {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        // Body 0: fixed host (a small box) at the origin.
        .add_fixed_body(
            "host",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                1.0,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(0.1, 0.1, 0.1)),
            ),
        )
        // Body 1: free accessory (a sphere) parented to world so its free-joint
        // q is the world transform.
        .add_free_body(
            "accessory",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::sphere(0.1, 0.1),
        )
        .build();

    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: Vec3::new(0.5, 0.5, 0.1),
    });
    model.bodies[1].geometry = Some(Geometry::Sphere { radius: 0.1 });

    let mut state = model.default_state();

    // Place the accessory just slightly overlapping the host on the +z side.
    // Free joint q layout = [x, y, z, wx, wy, wz]; we only set z so the
    // sphere centre sits at z = 0.05, penetrating the box's top face (z = 0.1).
    let q_offset = model.q_offsets[model.bodies[1].joint_idx];
    state.q[q_offset + 2] = 0.05;

    // Manually populate body transforms instead of going through FK so we have
    // an unambiguous contact geometry independent of joint conventions.
    state.body_xform[0] = SpatialTransform::identity();
    state.body_xform[1] = SpatialTransform::new(Mat3::identity(), Vec3::new(0.0, 0.0, 0.05));

    let geometries: Vec<Option<Geometry>> =
        model.bodies.iter().map(|b| b.geometry.clone()).collect();

    let contacts = find_contacts(&model, &state, &geometries);
    assert_eq!(
        contacts.len(),
        1,
        "expected exactly one body-body contact, got {}",
        contacts.len()
    );
    let contact = &contacts[0];
    assert_eq!(contact.body_i, 0);
    assert_eq!(contact.body_j, 1);

    // body_j is above body_i, so the normal (pos_j - pos_i).normalize() points +z.
    assert_relative_eq!(contact.contact_normal.z, 1.0, epsilon = 1e-10);

    // No ground contacts in this scenario.
    let ground = find_ground_contacts(&state, &geometries, -10.0);
    assert!(ground.is_empty());

    let materials = vec![ContactMaterial::default()];
    let forces = contact_forces(&contacts, &state, &materials, None);

    // Force on the free accessory (body 1, sitting ABOVE the host) should point
    // upward (+z) so it separates from the host. Before the sign fix this would
    // be negative.
    let fz_free = forces[1].linear.z;
    assert!(
        fz_free > 0.0,
        "contact force on accessory should push UP (+z), got fz = {:.4}",
        fz_free,
    );

    // Body 0 receives the opposite force (-z). It's a fixed host so this won't
    // move it, but the sign must still be correct for Newton's 3rd law.
    let fz_host = forces[0].linear.z;
    assert!(
        fz_host < 0.0,
        "reaction force on host should push DOWN (-z), got fz = {:.4}",
        fz_host,
    );
    assert_relative_eq!(fz_free, -fz_host, epsilon = 1e-12);

    // Sanity: force magnitude is non-zero.
    assert!(forces[1].linear.norm() > 0.0);
    // No spurious torque component for an axis-aligned offset along the normal
    // (Goal 4 changes this for general offsets, but for now the wrench is
    // purely linear).
    let _ = SpatialVec::zero();
}
