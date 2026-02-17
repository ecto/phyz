//! 6D spatial algebra following Featherstone's "Rigid Body Dynamics Algorithms".
//!
//! Convention: spatial vectors are [angular; linear] (Featherstone order).
//! A spatial motion vector (twist): [ω; v]
//! A spatial force vector (wrench): [τ; f]

use crate::{Mat3, Mat6, Vec3, Vec6, skew};
use nalgebra as na;

/// 6D spatial vector — either a motion vector (twist) or force vector (wrench).
#[derive(Debug, Clone, Copy)]
pub struct SpatialVec {
    /// The underlying 6D vector [angular(3); linear(3)].
    pub data: Vec6,
}

impl SpatialVec {
    /// Create from angular and linear parts.
    #[inline]
    pub fn new(angular: Vec3, linear: Vec3) -> Self {
        Self {
            data: Vec6::new(
                angular.x, angular.y, angular.z, linear.x, linear.y, linear.z,
            ),
        }
    }

    /// Zero spatial vector.
    #[inline]
    pub fn zero() -> Self {
        Self {
            data: Vec6::zeros(),
        }
    }

    /// Angular (top 3) component.
    #[inline]
    pub fn angular(&self) -> Vec3 {
        Vec3::new(self.data[0], self.data[1], self.data[2])
    }

    /// Linear (bottom 3) component.
    #[inline]
    pub fn linear(&self) -> Vec3 {
        Vec3::new(self.data[3], self.data[4], self.data[5])
    }

    /// Spatial cross product for motion vectors: v ×ₘ w
    /// Used in velocity propagation.
    pub fn cross_motion(&self, other: &SpatialVec) -> SpatialVec {
        let w = self.angular();
        let v = self.linear();
        let w2 = other.angular();
        let v2 = other.linear();
        SpatialVec::new(w.cross(&w2), w.cross(&v2) + v.cross(&w2))
    }

    /// Spatial cross product for force vectors: v ×f f
    /// Used in bias force computation.
    pub fn cross_force(&self, other: &SpatialVec) -> SpatialVec {
        let w = self.angular();
        let v = self.linear();
        let t = other.angular();
        let f = other.linear();
        SpatialVec::new(w.cross(&t) + v.cross(&f), w.cross(&f))
    }

    /// Dot product of two spatial vectors.
    #[inline]
    pub fn dot(&self, other: &SpatialVec) -> f64 {
        self.data.dot(&other.data)
    }

    /// Create from linear and angular parts (convenient alternative order).
    #[inline]
    pub fn from_linear_angular(linear: Vec3, angular: Vec3) -> Self {
        Self::new(angular, linear)
    }
}

impl std::ops::Add for SpatialVec {
    type Output = SpatialVec;
    #[inline]
    fn add(self, rhs: SpatialVec) -> SpatialVec {
        SpatialVec {
            data: self.data + rhs.data,
        }
    }
}

impl std::ops::Sub for SpatialVec {
    type Output = SpatialVec;
    #[inline]
    fn sub(self, rhs: SpatialVec) -> SpatialVec {
        SpatialVec {
            data: self.data - rhs.data,
        }
    }
}

impl std::ops::Mul<f64> for SpatialVec {
    type Output = SpatialVec;
    #[inline]
    fn mul(self, rhs: f64) -> SpatialVec {
        SpatialVec {
            data: self.data * rhs,
        }
    }
}

impl std::ops::Neg for SpatialVec {
    type Output = SpatialVec;
    #[inline]
    fn neg(self) -> SpatialVec {
        SpatialVec { data: -self.data }
    }
}

/// 6x6 spatial matrix (inertia, transforms acting on spatial vectors).
#[derive(Debug, Clone, Copy)]
pub struct SpatialMat {
    pub data: Mat6,
}

impl SpatialMat {
    /// Create from a 6x6 nalgebra matrix.
    #[inline]
    pub fn from_mat6(data: Mat6) -> Self {
        Self { data }
    }

    /// Zero matrix.
    #[inline]
    pub fn zero() -> Self {
        Self {
            data: Mat6::zeros(),
        }
    }

    /// Identity matrix.
    #[inline]
    pub fn identity() -> Self {
        Self {
            data: Mat6::identity(),
        }
    }

    /// Multiply by a spatial vector.
    #[inline]
    pub fn mul_vec(&self, v: &SpatialVec) -> SpatialVec {
        SpatialVec {
            data: self.data * v.data,
        }
    }

    /// Matrix-matrix multiply.
    #[inline]
    pub fn mul_mat(&self, other: &SpatialMat) -> SpatialMat {
        SpatialMat {
            data: self.data * other.data,
        }
    }

    /// Transpose.
    #[inline]
    pub fn transpose(&self) -> SpatialMat {
        SpatialMat {
            data: self.data.transpose(),
        }
    }
}

impl std::ops::Add for SpatialMat {
    type Output = SpatialMat;
    #[inline]
    fn add(self, rhs: SpatialMat) -> SpatialMat {
        SpatialMat {
            data: self.data + rhs.data,
        }
    }
}

impl std::ops::Sub for SpatialMat {
    type Output = SpatialMat;
    #[inline]
    fn sub(self, rhs: SpatialMat) -> SpatialMat {
        SpatialMat {
            data: self.data - rhs.data,
        }
    }
}

/// Plücker transform: rigid body transformation acting on spatial vectors.
///
/// Represents a coordinate transform from frame A to frame B.
/// Stored as rotation R and translation p (position of B's origin in A's frame).
#[derive(Debug, Clone, Copy)]
pub struct SpatialTransform {
    /// Rotation from frame A to frame B.
    pub rot: Mat3,
    /// Position of frame B's origin expressed in frame A.
    pub pos: Vec3,
}

impl SpatialTransform {
    /// Create from rotation matrix and translation.
    pub fn new(rot: Mat3, pos: Vec3) -> Self {
        Self { rot, pos }
    }

    /// Identity transform.
    pub fn identity() -> Self {
        Self {
            rot: Mat3::identity(),
            pos: Vec3::zeros(),
        }
    }

    /// Pure rotation about the X axis.
    pub fn rot_x(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            rot: Mat3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c),
            pos: Vec3::zeros(),
        }
    }

    /// Pure rotation about the Y axis.
    pub fn rot_y(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            rot: Mat3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c),
            pos: Vec3::zeros(),
        }
    }

    /// Pure rotation about the Z axis.
    pub fn rot_z(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            rot: Mat3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0),
            pos: Vec3::zeros(),
        }
    }

    /// Pure translation.
    pub fn translation(pos: Vec3) -> Self {
        Self {
            rot: Mat3::identity(),
            pos,
        }
    }

    /// Rotation about an arbitrary axis.
    pub fn rot_axis(axis: &na::Unit<Vec3>, angle: f64) -> Self {
        let rot = na::Rotation3::from_axis_angle(axis, angle);
        Self {
            rot: *rot.matrix(),
            pos: Vec3::zeros(),
        }
    }

    /// Get the 6x6 Plücker transform matrix for motion vectors.
    ///
    /// X = | R    0 |
    ///     | -R[p]× R |
    ///
    /// This transforms spatial motion vectors from frame A to frame B.
    pub fn to_motion_matrix(&self) -> Mat6 {
        let r = self.rot;
        let px = skew(&self.pos);
        let neg_rpx = -r * px;

        let mut m = Mat6::zeros();
        // Top-left: R
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        // Bottom-left: -R[p]×
        m.fixed_view_mut::<3, 3>(3, 0).copy_from(&neg_rpx);
        // Bottom-right: R
        m.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        m
    }

    /// Get the 6x6 Plücker transform matrix for force vectors.
    ///
    /// X* = | R    -R[p]× |
    ///      | 0      R    |
    ///
    /// This is the transpose-inverse of the motion transform.
    pub fn to_force_matrix(&self) -> Mat6 {
        let r = self.rot;
        let px = skew(&self.pos);
        let neg_rpx = -r * px;

        let mut m = Mat6::zeros();
        // Top-left: R
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        // Top-right: -R[p]×
        m.fixed_view_mut::<3, 3>(0, 3).copy_from(&neg_rpx);
        // Bottom-right: R
        m.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        m
    }

    /// Transform a spatial motion vector from frame A to frame B.
    pub fn apply_motion(&self, v: &SpatialVec) -> SpatialVec {
        let w = v.angular();
        let vel = v.linear();
        let new_w = self.rot * w;
        let new_v = self.rot * (vel - self.pos.cross(&w));
        SpatialVec::new(new_w, new_v)
    }

    /// Transform a spatial force vector from frame A to frame B.
    pub fn apply_force(&self, f: &SpatialVec) -> SpatialVec {
        let tau = f.angular();
        let force = f.linear();
        let new_f = self.rot * force;
        let new_tau = self.rot * (tau - self.pos.cross(&force));
        SpatialVec::new(new_tau, new_f)
    }

    /// Inverse transform a spatial motion vector (from B to A).
    pub fn inv_apply_motion(&self, v: &SpatialVec) -> SpatialVec {
        let rt = self.rot.transpose();
        let w = v.angular();
        let vel = v.linear();
        let new_w = rt * w;
        let new_v = rt * vel + self.pos.cross(&(rt * w));
        SpatialVec::new(new_w, new_v)
    }

    /// Inverse transform a spatial force vector (from B to A).
    pub fn inv_apply_force(&self, f: &SpatialVec) -> SpatialVec {
        let rt = self.rot.transpose();
        let tau = f.angular();
        let force = f.linear();
        let new_f = rt * force;
        let new_tau = rt * tau + self.pos.cross(&(rt * force));
        SpatialVec::new(new_tau, new_f)
    }

    /// Compose two transforms: self ∘ other.
    pub fn compose(&self, other: &SpatialTransform) -> SpatialTransform {
        SpatialTransform {
            rot: self.rot * other.rot,
            pos: other.pos + other.rot.transpose() * self.pos,
        }
    }

    /// Inverse of this transform.
    pub fn inverse(&self) -> SpatialTransform {
        let rt = self.rot.transpose();
        SpatialTransform {
            rot: rt,
            pos: -(self.rot * self.pos),
        }
    }

    /// Get the translation vector.
    pub fn translation_vector(&self) -> Vec3 {
        self.pos
    }

    /// Get the rotation matrix.
    pub fn rotation_matrix(&self) -> Mat3 {
        self.rot
    }
}

/// Spatial inertia of a rigid body about its center of mass.
///
/// Stored as mass, center of mass offset, and rotational inertia.
/// Can be converted to a 6x6 spatial inertia matrix.
#[derive(Debug, Clone, Copy)]
pub struct SpatialInertia {
    /// Mass of the body.
    pub mass: f64,
    /// Center of mass position in body frame.
    pub com: Vec3,
    /// Rotational inertia about the center of mass (3x3 symmetric).
    pub inertia: Mat3,
}

impl SpatialInertia {
    /// Create a spatial inertia with the given mass, CoM offset, and inertia matrix.
    pub fn new(mass: f64, com: Vec3, inertia: Mat3) -> Self {
        Self { mass, com, inertia }
    }

    /// Create spatial inertia for a point mass at a given position.
    pub fn point_mass(mass: f64, pos: Vec3) -> Self {
        Self {
            mass,
            com: pos,
            inertia: Mat3::zeros(),
        }
    }

    /// Create spatial inertia for a uniform rod of given mass and length along Y axis.
    /// Rod is centered at origin.
    pub fn rod(mass: f64, length: f64) -> Self {
        let i = mass * length * length / 12.0;
        Self {
            mass,
            com: Vec3::zeros(),
            inertia: Mat3::new(i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i),
        }
    }

    /// Create spatial inertia for a uniform sphere.
    pub fn sphere(mass: f64, radius: f64) -> Self {
        let i = 2.0 / 5.0 * mass * radius * radius;
        Self {
            mass,
            com: Vec3::zeros(),
            inertia: Mat3::from_diagonal(&Vec3::new(i, i, i)),
        }
    }

    /// Convert to 6x6 spatial inertia matrix (about the body frame origin).
    ///
    /// I_spatial = | I + m[c]×[c]×ᵀ   m[c]× |
    ///             | m[c]×ᵀ             mE    |
    pub fn to_matrix(&self) -> SpatialMat {
        let cx = skew(&self.com);
        let m = self.mass;
        let i3 = Mat3::identity() * m;

        let mut mat = Mat6::zeros();
        // Top-left: I + m * cx * cx^T
        let top_left = self.inertia + cx * cx.transpose() * m;
        mat.fixed_view_mut::<3, 3>(0, 0).copy_from(&top_left);
        // Top-right: m * cx
        let mcx = cx * m;
        mat.fixed_view_mut::<3, 3>(0, 3).copy_from(&mcx);
        // Bottom-left: m * cx^T
        mat.fixed_view_mut::<3, 3>(3, 0).copy_from(&mcx.transpose());
        // Bottom-right: m * E
        mat.fixed_view_mut::<3, 3>(3, 3).copy_from(&i3);

        SpatialMat::from_mat6(mat)
    }

    /// Transform this inertia to a new frame.
    pub fn transform(&self, xform: &SpatialTransform) -> SpatialInertia {
        // For simplicity, transform via 6x6 matrix.
        // I_B = X* I_A X*^T  (in force transform convention)
        let x_force = SpatialMat::from_mat6(xform.to_force_matrix());
        let i_mat = self.to_matrix();
        let transformed = x_force
            .mul_mat(&i_mat)
            .mul_mat(&SpatialMat::from_mat6(xform.to_motion_matrix()).transpose());

        // Extract mass (unchanged), com, and inertia from the 6x6
        let mass = transformed.data[(3, 3)];
        let com = if mass.abs() > 1e-12 {
            Vec3::new(
                transformed.data[(2, 3)] / mass,
                -transformed.data[(0, 3)] / mass, // Note: skew sign
                transformed.data[(1, 3)] / mass,  // Note: skew sign
            )
        } else {
            Vec3::zeros()
        };
        // Rotational inertia: top-left 3x3 minus mass * [c]x [c]x^T
        let cx = skew(&com);
        let inertia_with_com = transformed.data.fixed_view::<3, 3>(0, 0).into_owned();
        let inertia = inertia_with_com - cx * cx.transpose() * mass;

        SpatialInertia { mass, com, inertia }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_spatial_vec_cross_motion() {
        let v1 = SpatialVec::new(Vec3::new(0.0, 0.0, 1.0), Vec3::zeros());
        let v2 = SpatialVec::new(Vec3::new(1.0, 0.0, 0.0), Vec3::zeros());
        let result = v1.cross_motion(&v2);
        // [0,0,1] × [1,0,0] = [0,1,0]
        assert_relative_eq!(result.angular().y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_transform_identity() {
        let xf = SpatialTransform::identity();
        let v = SpatialVec::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
        let result = xf.apply_motion(&v);
        assert_relative_eq!(result.data, v.data, epsilon = 1e-10);
    }

    #[test]
    fn test_transform_inverse_roundtrip() {
        let xf = SpatialTransform::new(
            *na::Rotation3::from_axis_angle(&na::Vector3::z_axis(), 0.5).matrix(),
            Vec3::new(1.0, 2.0, 3.0),
        );
        let v = SpatialVec::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));

        let forward = xf.apply_motion(&v);
        let back = xf.inv_apply_motion(&forward);
        assert_relative_eq!(back.data, v.data, epsilon = 1e-10);
    }

    #[test]
    fn test_compose_transforms() {
        let xf1 = SpatialTransform::translation(Vec3::new(1.0, 0.0, 0.0));
        let xf2 = SpatialTransform::translation(Vec3::new(0.0, 2.0, 0.0));
        let composed = xf1.compose(&xf2);
        assert_relative_eq!(composed.pos, Vec3::new(1.0, 2.0, 0.0), epsilon = 1e-10);
    }

    #[test]
    fn test_spatial_inertia_point_mass() {
        let si = SpatialInertia::point_mass(2.0, Vec3::new(0.0, 1.0, 0.0));
        let mat = si.to_matrix();
        // Mass is 2, com at (0,1,0)
        // Bottom-right should be 2*I3
        assert_relative_eq!(mat.data[(3, 3)], 2.0, epsilon = 1e-10);
        assert_relative_eq!(mat.data[(4, 4)], 2.0, epsilon = 1e-10);
        assert_relative_eq!(mat.data[(5, 5)], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spatial_inertia_sphere() {
        let si = SpatialInertia::sphere(5.0, 0.1);
        let expected_i = 2.0 / 5.0 * 5.0 * 0.01;
        assert_relative_eq!(si.inertia[(0, 0)], expected_i, epsilon = 1e-10);
        assert_relative_eq!(si.inertia[(1, 1)], expected_i, epsilon = 1e-10);
        assert_relative_eq!(si.inertia[(2, 2)], expected_i, epsilon = 1e-10);
    }
}

#[cfg(test)]
mod prop_tests {
    use super::*;
    use crate::quaternion::Quat;
    use proptest::prelude::*;

    const EPS: f64 = 1e-9;

    fn arb_pos() -> impl Strategy<Value = Vec3> {
        (-10.0..10.0_f64, -10.0..10.0_f64, -10.0..10.0_f64)
            .prop_map(|(x, y, z)| Vec3::new(x, y, z))
    }

    fn arb_angle() -> impl Strategy<Value = f64> {
        -std::f64::consts::PI..std::f64::consts::PI
    }

    fn arb_unit_axis() -> impl Strategy<Value = na::Unit<Vec3>> {
        // Generate a non-zero vector, then normalize
        (-1.0..1.0_f64, -1.0..1.0_f64, -1.0..1.0_f64)
            .prop_filter("non-zero axis", |(x, y, z)| x * x + y * y + z * z > 0.01)
            .prop_map(|(x, y, z)| na::Unit::new_normalize(Vec3::new(x, y, z)))
    }

    fn arb_transform() -> impl Strategy<Value = SpatialTransform> {
        (arb_unit_axis(), arb_angle(), arb_pos()).prop_map(|(axis, angle, pos)| {
            let rot = na::Rotation3::from_axis_angle(&axis, angle);
            SpatialTransform::new(*rot.matrix(), pos)
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
            for i in 0..3 {
                prop_assert!((result.pos[i] - id.pos[i]).abs() < EPS,
                    "pos[{}]: {} vs {}", i, result.pos[i], id.pos[i]);
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
            for i in 0..3 {
                prop_assert!((ab_c.pos[i] - a_bc.pos[i]).abs() < EPS,
                    "pos[{}]: {} vs {}", i, ab_c.pos[i], a_bc.pos[i]);
            }
        }

        #[test]
        fn apply_force_matches_matrix(xf in arb_transform(), f in arb_spatial_vec()) {
            let applied = xf.apply_force(&f);
            let mat_result = SpatialMat::from_mat6(xf.to_force_matrix()).mul_vec(&f);
            for i in 0..6 {
                prop_assert!((applied.data[i] - mat_result.data[i]).abs() < EPS,
                    "component {}: {} vs {}", i, applied.data[i], mat_result.data[i]);
            }
        }

        #[test]
        fn apply_motion_matches_matrix(xf in arb_transform(), v in arb_spatial_vec()) {
            let applied = xf.apply_motion(&v);
            let mat_result = SpatialMat::from_mat6(xf.to_motion_matrix()).mul_vec(&v);
            for i in 0..6 {
                prop_assert!((applied.data[i] - mat_result.data[i]).abs() < EPS,
                    "component {}: {} vs {}", i, applied.data[i], mat_result.data[i]);
            }
        }

        #[test]
        fn sphere_inertia_matrix_is_symmetric(
            mass in 0.1..100.0_f64,
            radius in 0.01..10.0_f64,
        ) {
            let si = SpatialInertia::sphere(mass, radius);
            let mat = si.to_matrix().data;
            for i in 0..6 {
                for j in 0..6 {
                    prop_assert!((mat[(i, j)] - mat[(j, i)]).abs() < EPS,
                        "not symmetric at ({},{}): {} vs {}", i, j, mat[(i, j)], mat[(j, i)]);
                }
            }
        }

        #[test]
        fn quat_to_matrix_is_rotation(
            axis in arb_unit_axis(),
            angle in arb_angle(),
        ) {
            let q = Quat::from_axis_angle(&axis.into_inner(), angle).normalize();
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
            axis in arb_unit_axis(),
            angle in arb_angle(),
        ) {
            let q = Quat::from_axis_angle(&axis.into_inner(), angle).normalize();
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
