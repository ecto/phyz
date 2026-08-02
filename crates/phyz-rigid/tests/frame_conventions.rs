//! Locks down the frame convention of [`forward_kinematics`].
//!
//! The returned transforms are documented as "world_to_body", which reads as
//! though the world pose needs `.inverse()`. It does not: the `pos` field is
//! already the body origin's position **in world coordinates**, while `rot` is
//! the world→body rotation (so `rot.transpose()` is body→world).
//!
//! Getting this backwards silently mirrors every model through the origin —
//! exactly the kind of bug that produces plausible-looking but wrong contact
//! and observation data. Hence this test.

use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Joint, ModelBuilder};
use phyz_rigid::forward_kinematics;

#[test]
fn xform_pos_is_the_world_position_not_its_inverse() {
    let model = ModelBuilder::new()
        .add_body(
            "root",
            -1,
            Joint::free(SpatialTransform::identity()),
            SpatialInertia::new(
                1.0,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(0.1, 0.1, 0.1)),
            ),
        )
        .add_revolute_body(
            "child",
            0,
            SpatialTransform::from_translation(Vec3::new(0.0, 0.0, -1.0)),
            SpatialInertia::point_mass(1.0, Vec3::zeros()),
        )
        .build();

    let mut s = model.default_state();
    // Free joint q = [wx, wy, wz, x, y, z]; z is slot 5.
    s.q[5] = 2.5; // lift the root to z = 2.5

    let (x, _) = forward_kinematics(&model, &s);

    assert!(
        (x[0].pos.z - 2.5).abs() < 1e-12,
        "root world z: {:?}",
        x[0].pos
    );
    assert!(
        (x[1].pos.z - 1.5).abs() < 1e-12,
        "child hangs 1 m below the root: {:?}",
        x[1].pos
    );

    // The inverse negates it — the trap this test exists to catch.
    assert!((x[0].inverse().pos.z + 2.5).abs() < 1e-12);
}

#[test]
fn xform_rot_transposed_maps_body_to_world() {
    let model = ModelBuilder::new()
        .add_revolute_body(
            "link",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
        )
        .build();

    let mut s = model.default_state();
    s.q[0] = std::f64::consts::FRAC_PI_2; // 90 deg about the default z axis

    let (x, _) = forward_kinematics(&model, &s);
    let body_to_world = x[0].rot.transpose();

    // Body +x maps to world +y under a +90 deg rotation about z.
    let mapped = body_to_world * Vec3::new(1.0, 0.0, 0.0);
    assert!(mapped.x.abs() < 1e-12, "{mapped:?}");
    assert!((mapped.y - 1.0).abs() < 1e-12, "{mapped:?}");
}
