//! 6D spatial algebra following Featherstone's "Rigid Body Dynamics Algorithms".
//!
//! Convention: spatial vectors are [angular; linear] (Featherstone order).
//! A spatial motion vector (twist): [w; v]
//! A spatial force vector (wrench): [tau; f]
//!
//! This module re-exports tang's block-structured spatial types with f64 specialization.

/// 6D spatial vector -- either a motion vector (twist) or force vector (wrench).
///
/// Stored as two Vec3s (angular, linear) instead of a flat Vec6.
pub type SpatialVec = tang::SpatialVec<f64>;

/// 6x6 spatial matrix stored as four 3x3 blocks.
///
/// ```text
/// | upper_left  upper_right |
/// | lower_left  lower_right |
/// ```
pub type SpatialMat = tang::SpatialMat<f64>;

/// Plucker transform: rigid body transformation acting on spatial vectors.
///
/// Represents a coordinate transform from frame A to frame B.
/// Stored as rotation R and translation p (position of B's origin in A's frame).
///
/// **Do not write `xf.rot * v` or `xf.rot.transpose() * v` by hand.** For a
/// body transform from forward kinematics, `rot` is the *world→body* rotation
/// and `pos` is the body origin in world coordinates — a convention every raw
/// call site must independently remember, and getting it backwards is
/// invisible at identity. Use the named, direction-carrying methods on
/// [`SpatialTransformExt`] instead: [`SpatialTransformExt::body_to_world_point`],
/// [`SpatialTransformExt::world_to_body_point`],
/// [`SpatialTransformExt::body_to_world_dir`], and
/// [`SpatialTransformExt::world_to_body_dir`].
pub type SpatialTransform = tang::SpatialTransform<f64>;

/// Spatial inertia of a rigid body about its center of mass.
///
/// Stored as mass, center of mass offset, and rotational inertia.
/// Can be converted to a 6x6 spatial inertia matrix.
pub type SpatialInertia = tang::SpatialInertia<f64>;

/// Direction-carrying frame conversions for [`SpatialTransform`].
///
/// **The convention, stated once and authoritatively:** in a body transform
/// produced by forward kinematics, `rot` is the **world→body** rotation
/// (Featherstone's `E`) and `pos` is the **body origin in world coordinates**.
/// So mapping a body-frame vector out to world requires `rot.transpose()`,
/// and mapping a world-frame vector into the body frame requires `rot`.
///
/// Every call site that writes `xf.rot * v` or `xf.rot.transpose() * v` by
/// hand must independently remember which direction `rot` points — and a
/// mistake is invisible at identity, only inverting once the body tilts.
/// These methods exist so no call site ever writes that arithmetic by hand
/// again: the method name carries the direction, and the point/dir split
/// carries whether the translation applies.
///
/// * `*_point` methods transform **positions** (rotation *and* translation).
/// * `*_dir` methods transform **directions** — axes, forces, velocities,
///   offsets — with rotation only.
pub trait SpatialTransformExt {
    /// Map a point in body coordinates to world coordinates:
    /// `pos + rot.transpose() * p`.
    fn body_to_world_point(&self, p: crate::Vec3) -> crate::Vec3;

    /// Map a point in world coordinates to body coordinates:
    /// `rot * (p - pos)`.
    fn world_to_body_point(&self, p: crate::Vec3) -> crate::Vec3;

    /// Rotate a direction (axis, force, velocity, offset) from body to world:
    /// `rot.transpose() * d`. No translation.
    fn body_to_world_dir(&self, d: crate::Vec3) -> crate::Vec3;

    /// Rotate a direction (axis, force, velocity, offset) from world to body:
    /// `rot * d`. No translation.
    fn world_to_body_dir(&self, d: crate::Vec3) -> crate::Vec3;
}

impl SpatialTransformExt for SpatialTransform {
    #[inline]
    fn body_to_world_point(&self, p: crate::Vec3) -> crate::Vec3 {
        self.pos + self.rot.transpose() * p
    }

    #[inline]
    fn world_to_body_point(&self, p: crate::Vec3) -> crate::Vec3 {
        self.rot * (p - self.pos)
    }

    #[inline]
    fn body_to_world_dir(&self, d: crate::Vec3) -> crate::Vec3 {
        self.rot.transpose() * d
    }

    #[inline]
    fn world_to_body_dir(&self, d: crate::Vec3) -> crate::Vec3 {
        self.rot * d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Mat3, Vec3};

    const EPS: f64 = 1e-10;

    fn assert_vec_eq(a: Vec3, b: Vec3, msg: &str) {
        assert!((a.x - b.x).abs() < EPS, "{msg}: x: {} vs {}", a.x, b.x);
        assert!((a.y - b.y).abs() < EPS, "{msg}: y: {} vs {}", a.y, b.y);
        assert!((a.z - b.z).abs() < EPS, "{msg}: z: {} vs {}", a.z, b.z);
    }

    fn assert_sv_eq(a: &SpatialVec, b: &SpatialVec, msg: &str) {
        assert_vec_eq(a.angular, b.angular, &format!("{msg} angular"));
        assert_vec_eq(a.linear, b.linear, &format!("{msg} linear"));
    }

    #[test]
    fn test_frame_conversion_round_trips() {
        let xf = SpatialTransform::new(
            Mat3::rotation_z(0.5).mul_mat(&Mat3::rotation_x(-0.3)),
            Vec3::new(1.0, -2.0, 3.0),
        );
        let p = Vec3::new(0.7, -1.1, 2.9);
        assert_vec_eq(
            xf.world_to_body_point(xf.body_to_world_point(p)),
            p,
            "point round trip b->w->b",
        );
        assert_vec_eq(
            xf.body_to_world_point(xf.world_to_body_point(p)),
            p,
            "point round trip w->b->w",
        );
        assert_vec_eq(
            xf.world_to_body_dir(xf.body_to_world_dir(p)),
            p,
            "dir round trip b->w->b",
        );
        assert_vec_eq(
            xf.body_to_world_dir(xf.world_to_body_dir(p)),
            p,
            "dir round trip w->b->w",
        );
    }

    #[test]
    fn test_frame_conversion_known_rotation() {
        use std::f64::consts::FRAC_PI_2;
        // Body frame rotated +90 deg about world Y: body->world is R_y(pi/2),
        // so the stored world->body rotation is R_y(-pi/2).
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let xf = SpatialTransform::new(Mat3::rotation_y(-FRAC_PI_2), pos);

        // R_y(pi/2) * (1,0,0) = (0,0,-1): the body x-axis points down world -z.
        assert_vec_eq(
            xf.body_to_world_dir(Vec3::x()),
            Vec3::new(0.0, 0.0, -1.0),
            "body x-axis in world",
        );
        // Points get the translation on top of the rotation.
        assert_vec_eq(
            xf.body_to_world_point(Vec3::x()),
            Vec3::new(1.0, 2.0, 2.0),
            "body point (1,0,0) in world",
        );
        // The inverse maps agree with hand-computed values.
        assert_vec_eq(
            xf.world_to_body_dir(Vec3::new(0.0, 0.0, -1.0)),
            Vec3::x(),
            "world -z in body",
        );
        assert_vec_eq(
            xf.world_to_body_point(Vec3::new(1.0, 2.0, 2.0)),
            Vec3::x(),
            "world point back to body",
        );
    }

    #[test]
    fn test_point_vs_dir_distinction() {
        let xf = SpatialTransform::new(Mat3::rotation_z(0.8), Vec3::new(5.0, -4.0, 3.0));
        let v = Vec3::new(1.0, 2.0, 3.0);
        // Directions never see the translation.
        assert_vec_eq(
            xf.body_to_world_point(v) - xf.body_to_world_dir(v),
            xf.pos,
            "point = dir + translation",
        );
        assert_vec_eq(
            xf.body_to_world_dir(Vec3::zero()),
            Vec3::zero(),
            "zero dir stays zero",
        );
        assert_vec_eq(
            xf.body_to_world_point(Vec3::zero()),
            xf.pos,
            "body origin maps to pos",
        );
        assert_vec_eq(
            xf.world_to_body_point(xf.pos),
            Vec3::zero(),
            "pos maps to body origin",
        );
    }

    #[test]
    fn test_spatial_vec_cross_motion() {
        let v1 = SpatialVec::new(Vec3::new(0.0, 0.0, 1.0), Vec3::zero());
        let v2 = SpatialVec::new(Vec3::new(1.0, 0.0, 0.0), Vec3::zero());
        let result = v1.cross_motion(&v2);
        // [0,0,1] x [1,0,0] = [0,1,0]
        assert!((result.angular.y - 1.0).abs() < EPS);
    }

    #[test]
    fn test_transform_identity() {
        let xf = SpatialTransform::identity();
        let v = SpatialVec::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
        let result = xf.apply_motion(&v);
        assert_sv_eq(&result, &v, "identity");
    }

    #[test]
    fn test_transform_inverse_roundtrip() {
        let xf = SpatialTransform::new(Mat3::rotation_z(0.5), Vec3::new(1.0, 2.0, 3.0));
        let v = SpatialVec::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));

        let forward = xf.apply_motion(&v);
        let back = xf.inv_apply_motion(&forward);
        assert_sv_eq(&back, &v, "roundtrip");
    }

    #[test]
    fn test_compose_transforms() {
        let xf1 = SpatialTransform::from_translation(Vec3::new(1.0, 0.0, 0.0));
        let xf2 = SpatialTransform::from_translation(Vec3::new(0.0, 2.0, 0.0));
        let composed = xf1.compose(&xf2);
        assert_vec_eq(composed.pos, Vec3::new(1.0, 2.0, 0.0), "compose pos");
    }

    #[test]
    fn test_spatial_inertia_point_mass() {
        let si = SpatialInertia::point_mass(2.0, Vec3::new(0.0, 1.0, 0.0));
        let mat = si.to_matrix();
        // Bottom-right should be 2*I3
        assert!((mat.lower_right[(0, 0)] - 2.0).abs() < EPS);
        assert!((mat.lower_right[(1, 1)] - 2.0).abs() < EPS);
        assert!((mat.lower_right[(2, 2)] - 2.0).abs() < EPS);
    }

    #[test]
    fn test_spatial_inertia_sphere() {
        let si = SpatialInertia::sphere(5.0, 0.1);
        let expected_i = 2.0 / 5.0 * 5.0 * 0.01;
        assert!((si.inertia[(0, 0)] - expected_i).abs() < EPS);
        assert!((si.inertia[(1, 1)] - expected_i).abs() < EPS);
        assert!((si.inertia[(2, 2)] - expected_i).abs() < EPS);
    }
}

#[cfg(test)]
mod prop_tests {
    use super::*;
    use crate::Quat;
    use crate::{Mat3, Vec3};
    use proptest::prelude::*;

    const EPS: f64 = 1e-9;

    fn arb_pos() -> impl Strategy<Value = Vec3> {
        (-10.0..10.0_f64, -10.0..10.0_f64, -10.0..10.0_f64).prop_map(|(x, y, z)| Vec3::new(x, y, z))
    }

    fn arb_angle() -> impl Strategy<Value = f64> {
        -std::f64::consts::PI..std::f64::consts::PI
    }

    fn arb_unit_vec() -> impl Strategy<Value = Vec3> {
        (-1.0..1.0_f64, -1.0..1.0_f64, -1.0..1.0_f64)
            .prop_filter("non-zero axis", |(x, y, z)| x * x + y * y + z * z > 0.01)
            .prop_map(|(x, y, z)| Vec3::new(x, y, z).normalize())
    }

    fn arb_transform() -> impl Strategy<Value = SpatialTransform> {
        (arb_unit_vec(), arb_angle(), arb_pos()).prop_map(|(axis, angle, pos)| {
            let rot = Mat3::rotation_axis(axis, angle);
            SpatialTransform::new(rot, pos)
        })
    }

    fn arb_spatial_vec() -> impl Strategy<Value = SpatialVec> {
        (arb_pos(), arb_pos()).prop_map(|(a, l)| SpatialVec::new(a, l))
    }

    proptest! {
        #[test]
        fn compose_with_inverse_is_identity(xf in arb_transform()) {
            let result = xf.compose(&xf.inverse());
            let id = SpatialTransform::identity();
            for i in 0..3 {
                for j in 0..3 {
                    prop_assert!((result.rot[(i, j)] - id.rot[(i, j)]).abs() < EPS,
                        "rot[{},{}]: {} vs {}", i, j, result.rot[(i, j)], id.rot[(i, j)]);
                }
            }
            for coord in [result.pos.x, result.pos.y, result.pos.z].iter().zip([id.pos.x, id.pos.y, id.pos.z].iter()) {
                let (a, b) = coord;
                prop_assert!((a - b).abs() < EPS, "pos: {} vs {}", a, b);
            }
        }

        #[test]
        fn compose_is_associative(
            a in arb_transform(),
            b in arb_transform(),
            c in arb_transform(),
        ) {
            let ab_c = a.compose(&b).compose(&c);
            let a_bc = a.compose(&b.compose(&c));
            for i in 0..3 {
                for j in 0..3 {
                    prop_assert!((ab_c.rot[(i, j)] - a_bc.rot[(i, j)]).abs() < EPS,
                        "rot[{},{}]: {} vs {}", i, j, ab_c.rot[(i, j)], a_bc.rot[(i, j)]);
                }
            }
            for coord in [(ab_c.pos.x, a_bc.pos.x), (ab_c.pos.y, a_bc.pos.y), (ab_c.pos.z, a_bc.pos.z)] {
                let (a, b) = coord;
                prop_assert!((a - b).abs() < EPS, "pos: {} vs {}", a, b);
            }
        }

        #[test]
        fn apply_force_matches_matrix(xf in arb_transform(), f in arb_spatial_vec()) {
            let applied = xf.apply_force(&f);
            let mat_result = xf.to_force_matrix().mul_vec(&f);
            let a = applied.as_array();
            let b = mat_result.as_array();
            for i in 0..6 {
                prop_assert!((a[i] - b[i]).abs() < EPS,
                    "component {}: {} vs {}", i, a[i], b[i]);
            }
        }

        #[test]
        fn apply_motion_matches_matrix(xf in arb_transform(), v in arb_spatial_vec()) {
            let applied = xf.apply_motion(&v);
            let mat_result = xf.to_motion_matrix().mul_vec(&v);
            let a = applied.as_array();
            let b = mat_result.as_array();
            for i in 0..6 {
                prop_assert!((a[i] - b[i]).abs() < EPS,
                    "component {}: {} vs {}", i, a[i], b[i]);
            }
        }

        #[test]
        fn sphere_inertia_matrix_is_symmetric(
            mass in 0.1..100.0_f64,
            radius in 0.01..10.0_f64,
        ) {
            let si = SpatialInertia::sphere(mass, radius);
            let mat = si.to_matrix();
            // Check that upper_right == lower_left^T
            for i in 0..3 {
                for j in 0..3 {
                    prop_assert!((mat.upper_right[(i, j)] - mat.lower_left[(j, i)]).abs() < EPS,
                        "ur/ll not symmetric at ({},{})", i, j);
                }
            }
            // Check ul and lr symmetric individually
            for i in 0..3 {
                for j in 0..3 {
                    prop_assert!((mat.upper_left[(i, j)] - mat.upper_left[(j, i)]).abs() < EPS,
                        "ul not symmetric at ({},{})", i, j);
                    prop_assert!((mat.lower_right[(i, j)] - mat.lower_right[(j, i)]).abs() < EPS,
                        "lr not symmetric at ({},{})", i, j);
                }
            }
        }

        #[test]
        fn quat_to_matrix_is_rotation(
            axis in arb_unit_vec(),
            angle in arb_angle(),
        ) {
            let q = Quat::from_axis_angle(axis, angle).normalize();
            let m = q.to_matrix();
            // determinant should be 1
            let det = m.determinant();
            prop_assert!((det - 1.0).abs() < EPS, "det = {}", det);
            // R * R^T should be identity
            let rrt = m * m.transpose();
            let id = Mat3::identity();
            for i in 0..3 {
                for j in 0..3 {
                    prop_assert!((rrt[(i, j)] - id[(i, j)]).abs() < EPS,
                        "R*R^T[{},{}]: {} vs {}", i, j, rrt[(i, j)], id[(i, j)]);
                }
            }
        }

        #[test]
        fn quat_matrix_roundtrip(
            axis in arb_unit_vec(),
            angle in arb_angle(),
        ) {
            let q = Quat::from_axis_angle(axis, angle).normalize();
            let m = q.to_matrix();
            let q2 = Quat::from_matrix(&m).normalize();
            // q and -q represent the same rotation
            let same = (q.w - q2.w).abs() < EPS
                && (q.v.x - q2.v.x).abs() < EPS
                && (q.v.y - q2.v.y).abs() < EPS
                && (q.v.z - q2.v.z).abs() < EPS;
            let neg = (q.w + q2.w).abs() < EPS
                && (q.v.x + q2.v.x).abs() < EPS
                && (q.v.y + q2.v.y).abs() < EPS
                && (q.v.z + q2.v.z).abs() < EPS;
            prop_assert!(same || neg,
                "roundtrip failed: q={:?}, q2={:?}", q, q2);
        }
    }
}
