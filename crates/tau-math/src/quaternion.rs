//! Quaternion utilities for 3D rotations.
//!
//! Convention: q = [w; x; y; z] where w is scalar, (x,y,z) is vector part.

use crate::{Mat3, Vec3};

/// A unit quaternion representing a 3D rotation.
#[derive(Debug, Clone, Copy)]
pub struct Quat {
    /// Scalar part (w).
    pub w: f64,
    /// Vector part (x, y, z).
    pub v: Vec3,
}

impl Quat {
    /// Create a new quaternion from scalar and vector parts.
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self {
            w,
            v: Vec3::new(x, y, z),
        }
    }

    /// Identity quaternion (no rotation).
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            v: Vec3::zeros(),
        }
    }

    /// Create quaternion from axis-angle representation.
    /// axis should be a unit vector, angle in radians.
    pub fn from_axis_angle(axis: &Vec3, angle: f64) -> Self {
        let half_angle = angle * 0.5;
        let (s, c) = half_angle.sin_cos();
        Self { w: c, v: axis * s }
    }

    /// Normalize this quaternion to unit length.
    pub fn normalize(&self) -> Self {
        let norm = (self.w * self.w + self.v.norm_squared()).sqrt();
        if norm < 1e-12 {
            return Self::identity();
        }
        Self {
            w: self.w / norm,
            v: self.v / norm,
        }
    }

    /// Quaternion multiplication: self * other.
    pub fn mul(&self, other: &Quat) -> Quat {
        Quat {
            w: self.w * other.w - self.v.dot(&other.v),
            v: self.v.cross(&other.v) + other.v * self.w + self.v * other.w,
        }
    }

    /// Conjugate of the quaternion (inverse for unit quaternions).
    pub fn conjugate(&self) -> Quat {
        Quat {
            w: self.w,
            v: -self.v,
        }
    }

    /// Convert quaternion to 3x3 rotation matrix.
    pub fn to_matrix(&self) -> Mat3 {
        let w = self.w;
        let x = self.v.x;
        let y = self.v.y;
        let z = self.v.z;

        let x2 = x * x;
        let y2 = y * y;
        let z2 = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;

        Mat3::new(
            1.0 - 2.0 * (y2 + z2),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (x2 + z2),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (x2 + y2),
        )
    }

    /// Convert rotation matrix to quaternion.
    /// Reference: Shepperd's method (stable for all rotation matrices).
    pub fn from_matrix(m: &Mat3) -> Quat {
        let trace = m[(0, 0)] + m[(1, 1)] + m[(2, 2)];

        if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0; // s = 4*w
            Quat {
                w: 0.25 * s,
                v: Vec3::new(
                    (m[(2, 1)] - m[(1, 2)]) / s,
                    (m[(0, 2)] - m[(2, 0)]) / s,
                    (m[(1, 0)] - m[(0, 1)]) / s,
                ),
            }
        } else if m[(0, 0)] > m[(1, 1)] && m[(0, 0)] > m[(2, 2)] {
            let s = (1.0 + m[(0, 0)] - m[(1, 1)] - m[(2, 2)]).sqrt() * 2.0; // s = 4*x
            Quat {
                w: (m[(2, 1)] - m[(1, 2)]) / s,
                v: Vec3::new(
                    0.25 * s,
                    (m[(0, 1)] + m[(1, 0)]) / s,
                    (m[(0, 2)] + m[(2, 0)]) / s,
                ),
            }
        } else if m[(1, 1)] > m[(2, 2)] {
            let s = (1.0 + m[(1, 1)] - m[(0, 0)] - m[(2, 2)]).sqrt() * 2.0; // s = 4*y
            Quat {
                w: (m[(0, 2)] - m[(2, 0)]) / s,
                v: Vec3::new(
                    (m[(0, 1)] + m[(1, 0)]) / s,
                    0.25 * s,
                    (m[(1, 2)] + m[(2, 1)]) / s,
                ),
            }
        } else {
            let s = (1.0 + m[(2, 2)] - m[(0, 0)] - m[(1, 1)]).sqrt() * 2.0; // s = 4*z
            Quat {
                w: (m[(1, 0)] - m[(0, 1)]) / s,
                v: Vec3::new(
                    (m[(0, 2)] + m[(2, 0)]) / s,
                    (m[(1, 2)] + m[(2, 1)]) / s,
                    0.25 * s,
                ),
            }
        }
    }

    /// Exponential map: exp(θ u) where θ u is axis-angle representation.
    /// Converts axis-angle to quaternion via q = [cos(θ/2), sin(θ/2) * u].
    /// For small angles, uses first-order approximation.
    pub fn exp(w: &Vec3) -> Quat {
        let theta = w.norm();
        if theta < 1e-10 {
            // First-order approximation for small angles
            Quat {
                w: 1.0,
                v: *w * 0.5,
            }
            .normalize()
        } else {
            let half_theta = theta * 0.5;
            Quat {
                w: half_theta.cos(),
                v: *w * (half_theta.sin() / theta),
            }
        }
    }

    /// Logarithmic map: log(q) returns the axis-angle vector θ u such that q = exp(θ u).
    pub fn log(&self) -> Vec3 {
        let v_norm = self.v.norm();
        if v_norm < 1e-10 {
            return Vec3::zeros();
        }
        // For quaternion q = [cos(θ/2), sin(θ/2) * u], we have:
        // θ = 2 * atan2(sin(θ/2), cos(θ/2)) = 2 * atan2(|v|, w)
        let angle = 2.0 * v_norm.atan2(self.w);
        self.v * (angle / v_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity() {
        let q = Quat::identity();
        assert_eq!(q.w, 1.0);
        assert_eq!(q.v, Vec3::zeros());
    }

    #[test]
    fn test_axis_angle() {
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let angle = std::f64::consts::FRAC_PI_2; // 90 degrees
        let q = Quat::from_axis_angle(&axis, angle);

        let expected_w = (angle / 2.0).cos();
        let expected_z = (angle / 2.0).sin();

        assert_relative_eq!(q.w, expected_w, epsilon = 1e-10);
        assert_relative_eq!(q.v.z, expected_z, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize() {
        let q = Quat::new(1.0, 2.0, 3.0, 4.0);
        let normalized = q.normalize();
        let norm = (normalized.w * normalized.w + normalized.v.norm_squared()).sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multiplication() {
        // 90 degree rotation about Z, then 90 degree rotation about Z
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let q1 = Quat::from_axis_angle(&axis, std::f64::consts::FRAC_PI_2);
        let q2 = Quat::from_axis_angle(&axis, std::f64::consts::FRAC_PI_2);
        let result = q1.mul(&q2);

        // Should equal 180 degree rotation about Z
        let expected = Quat::from_axis_angle(&axis, std::f64::consts::PI);

        assert_relative_eq!(result.w, expected.w, epsilon = 1e-10);
        assert_relative_eq!(result.v, expected.v, epsilon = 1e-10);
    }

    #[test]
    fn test_to_matrix() {
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let angle = std::f64::consts::FRAC_PI_2;
        let q = Quat::from_axis_angle(&axis, angle);
        let m = q.to_matrix();

        // 90 degree rotation about Z should map X to Y
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = m * x;
        assert_relative_eq!(y, Vec3::new(0.0, 1.0, 0.0), epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_roundtrip() {
        let axis = Vec3::new(1.0, 2.0, 3.0).normalize();
        let angle = 0.7;
        let q = Quat::from_axis_angle(&axis, angle);
        let m = q.to_matrix();
        let q2 = Quat::from_matrix(&m);

        // Quaternions q and -q represent the same rotation
        let same = (q.w - q2.w).abs() < 1e-10 && (q.v - q2.v).norm() < 1e-10;
        let negated = (q.w + q2.w).abs() < 1e-10 && (q.v + q2.v).norm() < 1e-10;

        assert!(same || negated);
    }

    #[test]
    fn test_exp_log() {
        let w = Vec3::new(0.1, 0.2, 0.3);
        let q = Quat::exp(&w);
        let w2 = q.log();
        assert_relative_eq!(w, w2, epsilon = 1e-10);
    }

    #[test]
    fn test_conjugate() {
        let q = Quat::new(0.5, 0.5, 0.5, 0.5).normalize();
        let conj = q.conjugate();
        let result = q.mul(&conj);
        assert_relative_eq!(result.w, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.v.norm(), 0.0, epsilon = 1e-10);
    }
}
