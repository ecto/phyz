//! A sphere dropped onto a fixed box must rest on it, not fall through.
//!
//! Companion to `body_body_resting.rs`, which proves the box-on-box case. The
//! sphere case fails differently: while the sphere is in the margin band the
//! reported normal is correct, but once it penetrates, the sphere-vs-box
//! manifold's normal comes back pointing the other way, the non-penetration
//! row measures approach instead of separation, and the solver drives the
//! sphere down through the plate. Found by the newt marble spike, where a
//! 10 mm marble on a 10 mm plate fell 7 m in 1.2 s.

use phyz::Simulator;
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, ModelBuilder};

const R: f64 = 0.01;

fn drop_sphere_on_plate(plate_half_thickness: f64) -> (f64, f64) {
    let si = SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01);
    let mi = SpatialInertia::new(0.012, Vec3::zeros(), Mat3::identity() * (0.4 * 0.012 * R * R));
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_fixed_body("plate", -1, SpatialTransform::from_translation(Vec3::new(0.0, 0.0, 0.2)), si)
        .add_free_body("sphere", -1, SpatialTransform::identity(), mi)
        .build();
    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: Vec3::new(0.15, 0.10, plate_half_thickness),
    });
    model.bodies[1].geometry = Some(Geometry::Sphere { radius: R });

    let top = 0.2 + plate_half_thickness;
    let mut s = model.default_state();
    s.q[5] = top + R + 0.0005;
    let sim = Simulator::new();
    for _ in 0..600 {
        sim.step_with_contacts(&model, &mut s, -100.0, &Default::default());
    }
    (s.q[5], top + R)
}

#[test]
fn a_sphere_dropped_on_a_thin_fixed_box_rests_on_it() {
    let (z, rest) = drop_sphere_on_plate(0.005);
    assert!(
        (z - rest).abs() < 2e-3,
        "sphere settled at z = {z:.4}; resting on the plate is z = {rest:.4}. \
         Falling below the plate means the sphere-box contact normal is inverted."
    );
}

#[test]
fn a_sphere_dropped_on_a_thick_fixed_box_rests_on_it() {
    let (z, rest) = drop_sphere_on_plate(0.02);
    assert!((z - rest).abs() < 2e-3, "sphere settled at z = {z:.4}; expected {rest:.4}");
}

/// The manifold itself, at a penetrating pose: the normal must point the way
/// body_i (the plate) moves to separate, i.e. away from the sphere.
#[test]
fn sphere_box_manifold_normal_points_from_box_to_sphere_when_penetrating() {
    use phyz_collision::{Geometry as G, contact_manifold_within};
    let plate = G::Box { half_extents: Vec3::new(0.15, 0.10, 0.005) };
    let sphere = G::Sphere { radius: R };
    let m = contact_manifold_within(
        &plate,
        &sphere,
        &Vec3::new(0.0, 0.0, 0.2),
        &Mat3::identity(),
        &Vec3::new(0.0, 0.0, 0.205 + R - 0.003),
        &Mat3::identity(),
        1e-3,
    )
    .expect("penetrating by 3 mm");
    assert!(m.normal.z > 0.9, "manifold normal {:?} should point +z (from plate toward sphere)", m.normal);
    assert!((m.points[0].depth - 0.003).abs() < 1e-4, "depth {}", m.points[0].depth);
}
