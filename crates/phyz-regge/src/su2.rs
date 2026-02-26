//! SU(2) group elements via unit quaternion representation.
//!
//! An SU(2) element is represented as a unit quaternion q = (q0, q1, q2, q3)
//! where q0 is the real (scalar) part and (q1, q2, q3) is the imaginary (vector) part.
//! The correspondence to 2×2 unitary matrices is:
//!
//!   U = q0·I + i·(q1·σ1 + q2·σ2 + q3·σ3)
//!
//! where σ_i are the Pauli matrices. Re(Tr(U)) = 2·q0.

/// SU(2) group element (unit quaternion).
#[derive(Debug, Clone, Copy)]
pub struct Su2 {
    /// Quaternion components [q0, q1, q2, q3]. q0 = real/scalar part.
    pub q: [f64; 4],
}

impl Su2 {
    /// Identity element.
    pub fn identity() -> Self {
        Self {
            q: [1.0, 0.0, 0.0, 0.0],
        }
    }

    /// Construct from components.
    pub fn new(q0: f64, q1: f64, q2: f64, q3: f64) -> Self {
        Self {
            q: [q0, q1, q2, q3],
        }
    }

    /// Quaternion norm squared.
    pub fn norm_sq(&self) -> f64 {
        self.q.iter().map(|x| x * x).sum()
    }

    /// Quaternion norm.
    pub fn norm(&self) -> f64 {
        self.norm_sq().sqrt()
    }

    /// Normalize to unit quaternion.
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-30 {
            return Self::identity();
        }
        Self {
            q: [self.q[0] / n, self.q[1] / n, self.q[2] / n, self.q[3] / n],
        }
    }

    /// Quaternion multiplication.
    pub fn mul(&self, other: &Su2) -> Su2 {
        let (a0, a1, a2, a3) = (self.q[0], self.q[1], self.q[2], self.q[3]);
        let (b0, b1, b2, b3) = (other.q[0], other.q[1], other.q[2], other.q[3]);

        Su2 {
            q: [
                a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
                a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
                a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
                a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
            ],
        }
    }

    /// Quaternion conjugate (= inverse for unit quaternions).
    pub fn inv(&self) -> Su2 {
        Su2 {
            q: [self.q[0], -self.q[1], -self.q[2], -self.q[3]],
        }
    }

    /// Re(Tr(U)) = 2·q0 (trace of the 2×2 matrix).
    pub fn re_trace(&self) -> f64 {
        2.0 * self.q[0]
    }

    /// Exponential map: su(2) algebra → SU(2) group.
    ///
    /// exp(θ_a · σ_a / 2i) = cos(|θ|/2) + sin(|θ|/2) · θ̂ · (σ/2i)
    ///
    /// Input: algebra element [θ1, θ2, θ3] (3 components).
    pub fn exp(algebra: &[f64; 3]) -> Su2 {
        let theta_sq = algebra[0] * algebra[0] + algebra[1] * algebra[1] + algebra[2] * algebra[2];
        let theta = theta_sq.sqrt();

        if theta < 1e-15 {
            // First-order approximation for small angles.
            Su2::new(1.0, algebra[0] / 2.0, algebra[1] / 2.0, algebra[2] / 2.0).normalize()
        } else {
            let half_theta = theta / 2.0;
            let c = half_theta.cos();
            let s = half_theta.sin() / theta;
            Su2::new(c, s * algebra[0], s * algebra[1], s * algebra[2])
        }
    }

    /// Logarithmic map: SU(2) group → su(2) algebra.
    ///
    /// Returns [θ1, θ2, θ3] such that self = exp(θ).
    pub fn log(&self) -> [f64; 3] {
        let sin_half =
            (self.q[1] * self.q[1] + self.q[2] * self.q[2] + self.q[3] * self.q[3]).sqrt();

        if sin_half < 1e-15 {
            // Near identity: θ ≈ 2·(q1, q2, q3).
            [2.0 * self.q[1], 2.0 * self.q[2], 2.0 * self.q[3]]
        } else {
            let half_theta = sin_half.atan2(self.q[0]);
            let factor = 2.0 * half_theta / sin_half;
            [factor * self.q[1], factor * self.q[2], factor * self.q[3]]
        }
    }

    /// Adjoint representation: Ad(U)·v = q·v·q* acting on 3-vectors.
    ///
    /// This is the SO(3) rotation matrix corresponding to the SU(2) element.
    /// Returns a 3×3 matrix (row-major): Ad(U)_{ab} rotates vector component b to a.
    pub fn adjoint(&self) -> [[f64; 3]; 3] {
        let (q0, q1, q2, q3) = (self.q[0], self.q[1], self.q[2], self.q[3]);
        [
            [
                1.0 - 2.0 * (q2 * q2 + q3 * q3),
                2.0 * (q1 * q2 - q0 * q3),
                2.0 * (q1 * q3 + q0 * q2),
            ],
            [
                2.0 * (q1 * q2 + q0 * q3),
                1.0 - 2.0 * (q1 * q1 + q3 * q3),
                2.0 * (q2 * q3 - q0 * q1),
            ],
            [
                2.0 * (q1 * q3 - q0 * q2),
                2.0 * (q2 * q3 + q0 * q1),
                1.0 - 2.0 * (q1 * q1 + q2 * q2),
            ],
        ]
    }

    /// Gradient of Re(Tr(U)) with respect to the algebra element θ_a,
    /// where U = V · exp(θ) · W (staple form).
    ///
    /// d/dθ_a [Re(Tr(exp(θ)))] evaluated at θ=0 gives [-q1, -q2, -q3]
    /// (the imaginary part, negated).
    pub fn re_trace_grad(&self) -> [f64; 3] {
        [-self.q[1], -self.q[2], -self.q[3]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let id = Su2::identity();
        assert!((id.re_trace() - 2.0).abs() < 1e-15);
        assert!((id.norm() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_mul_identity() {
        let a = Su2::new(0.5, 0.5, 0.5, 0.5);
        let id = Su2::identity();
        let b = a.mul(&id);
        for i in 0..4 {
            assert!((a.q[i] - b.q[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_inv() {
        let a = Su2::new(0.5, 0.5, 0.5, 0.5); // unit quaternion
        let b = a.mul(&a.inv());
        assert!((b.q[0] - 1.0).abs() < 1e-14);
        assert!(b.q[1].abs() < 1e-14);
        assert!(b.q[2].abs() < 1e-14);
        assert!(b.q[3].abs() < 1e-14);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let theta = [0.3, -0.5, 0.1];
        let u = Su2::exp(&theta);
        let recovered = u.log();
        for i in 0..3 {
            assert!(
                (theta[i] - recovered[i]).abs() < 1e-10,
                "axis {i}: {:.6} vs {:.6}",
                theta[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_exp_zero_is_identity() {
        let u = Su2::exp(&[0.0, 0.0, 0.0]);
        assert!((u.q[0] - 1.0).abs() < 1e-14);
        assert!(u.q[1].abs() < 1e-14);
    }

    #[test]
    fn test_re_trace() {
        let id = Su2::identity();
        assert!((id.re_trace() - 2.0).abs() < 1e-15);

        let u = Su2::exp(&[std::f64::consts::PI, 0.0, 0.0]);
        // exp(π·σ1/2i) has q0 = cos(π/2) = 0
        assert!(u.re_trace().abs() < 1e-10);
    }

    #[test]
    fn test_quaternion_associativity() {
        let a = Su2::exp(&[0.1, 0.2, 0.3]);
        let b = Su2::exp(&[0.4, -0.1, 0.2]);
        let c = Su2::exp(&[-0.2, 0.3, -0.1]);

        let ab_c = a.mul(&b).mul(&c);
        let a_bc = a.mul(&b.mul(&c));

        for i in 0..4 {
            assert!(
                (ab_c.q[i] - a_bc.q[i]).abs() < 1e-12,
                "associativity failed at component {i}"
            );
        }
    }
}
