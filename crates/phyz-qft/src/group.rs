//! Gauge group implementations for lattice QCD.
//!
//! Supports U(1), SU(2), and SU(3) groups.

use std::f64::consts::PI;

/// Trait for gauge groups on the lattice.
pub trait Group: Clone + std::fmt::Debug + Send + Sync {
    /// Identity element.
    fn identity() -> Self;

    /// Group multiplication.
    fn mul(&self, other: &Self) -> Self;

    /// Inverse (adjoint for unitary matrices).
    fn inv(&self) -> Self;

    /// Real part of trace (for action computation).
    fn re_tr(&self) -> f64;

    /// Generate random group element.
    fn random() -> Self;

    /// Exponential map from Lie algebra (for integration).
    fn exp(p: &Self::Momentum) -> Self;

    /// Force (derivative of action w.r.t. this link).
    /// Takes staple sum U_staple = sum of 6 staples touching this link.
    fn force(&self, staple: &Self) -> Self::Momentum;

    /// Associated momentum type (Lie algebra).
    type Momentum: Clone + std::fmt::Debug + Send + Sync;

    /// Create zero momentum.
    fn zero_momentum() -> Self::Momentum;

    /// Add two momenta.
    fn add_momentum(p1: &Self::Momentum, p2: &Self::Momentum) -> Self::Momentum;

    /// Scale momentum by scalar.
    fn scale_momentum(p: &Self::Momentum, scale: f64) -> Self::Momentum;

    /// Norm squared of momentum (for Hamiltonian).
    fn momentum_norm_squared(p: &Self::Momentum) -> f64;

    /// Sample momentum from Gaussian distribution.
    fn sample_momentum() -> Self::Momentum;
}

// ============================================================================
// U(1) Group
// ============================================================================

/// U(1) group element: e^(iθ).
#[derive(Debug, Clone, Copy)]
pub struct U1 {
    /// Angle θ ∈ [0, 2π).
    pub theta: f64,
}

impl U1 {
    /// Create new U(1) element.
    pub fn new(theta: f64) -> Self {
        Self {
            theta: theta.rem_euclid(2.0 * PI),
        }
    }

    /// Normalize angle to [0, 2π).
    #[allow(dead_code)]
    fn normalize(&mut self) {
        self.theta = self.theta.rem_euclid(2.0 * PI);
    }
}

impl Group for U1 {
    type Momentum = f64;

    fn identity() -> Self {
        Self { theta: 0.0 }
    }

    fn mul(&self, other: &Self) -> Self {
        Self::new(self.theta + other.theta)
    }

    fn inv(&self) -> Self {
        Self::new(-self.theta)
    }

    fn re_tr(&self) -> f64 {
        self.theta.cos()
    }

    fn random() -> Self {
        Self::new(rand::random::<f64>() * 2.0 * PI)
    }

    fn exp(p: &f64) -> Self {
        Self::new(*p)
    }

    fn force(&self, staple: &Self) -> f64 {
        // For U(1): force = -dS/dθ = β sin(θ + θ_staple)
        // Staple is sum of plaquette contributions
        (self.theta + staple.theta).sin()
    }

    fn zero_momentum() -> f64 {
        0.0
    }

    fn add_momentum(p1: &f64, p2: &f64) -> f64 {
        p1 + p2
    }

    fn scale_momentum(p: &f64, scale: f64) -> f64 {
        p * scale
    }

    fn momentum_norm_squared(p: &f64) -> f64 {
        p * p
    }

    fn sample_momentum() -> f64 {
        // Sample from N(0, 1)
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let u1: f64 = rng.r#gen();
        let u2: f64 = rng.r#gen();
        // Box-Muller transform
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

// ============================================================================
// SU(2) Group
// ============================================================================

/// SU(2) group element as unit quaternion.
///
/// Represented as q = a + b i + c j + d k with a² + b² + c² + d² = 1.
/// Matrix form: [a+id  b+ic]
///              [-b+ic a-id]
#[derive(Debug, Clone, Copy)]
pub struct SU2 {
    /// Quaternion components [a, b, c, d].
    pub q: [f64; 4],
}

impl SU2 {
    /// Create new SU(2) element from quaternion.
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        let mut q = Self { q: [a, b, c, d] };
        q.normalize();
        q
    }

    /// Normalize to unit quaternion.
    fn normalize(&mut self) {
        let norm = (self.q[0] * self.q[0]
            + self.q[1] * self.q[1]
            + self.q[2] * self.q[2]
            + self.q[3] * self.q[3])
            .sqrt();
        if norm > 1e-12 {
            for i in 0..4 {
                self.q[i] /= norm;
            }
        }
    }
}

/// SU(2) Lie algebra element (3D vector for su(2) ~ so(3)).
pub type SU2Momentum = [f64; 3];

impl Group for SU2 {
    type Momentum = SU2Momentum;

    fn identity() -> Self {
        Self {
            q: [1.0, 0.0, 0.0, 0.0],
        }
    }

    fn mul(&self, other: &Self) -> Self {
        // Quaternion multiplication
        let a = self.q[0] * other.q[0]
            - self.q[1] * other.q[1]
            - self.q[2] * other.q[2]
            - self.q[3] * other.q[3];
        let b = self.q[0] * other.q[1] + self.q[1] * other.q[0] + self.q[2] * other.q[3]
            - self.q[3] * other.q[2];
        let c = self.q[0] * other.q[2] - self.q[1] * other.q[3]
            + self.q[2] * other.q[0]
            + self.q[3] * other.q[1];
        let d = self.q[0] * other.q[3] + self.q[1] * other.q[2] - self.q[2] * other.q[1]
            + self.q[3] * other.q[0];
        Self::new(a, b, c, d)
    }

    fn inv(&self) -> Self {
        // Conjugate of unit quaternion
        Self {
            q: [self.q[0], -self.q[1], -self.q[2], -self.q[3]],
        }
    }

    fn re_tr(&self) -> f64 {
        // Tr(U) = 2a for SU(2)
        2.0 * self.q[0]
    }

    fn random() -> Self {
        // Uniform random SU(2) via quaternion sampling
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let u1: f64 = rng.r#gen();
        let u2: f64 = rng.r#gen();
        let u3: f64 = rng.r#gen();

        let s = (1.0 - u1).sqrt();
        let t = u1.sqrt();

        Self::new(
            s * (2.0 * PI * u2).sin(),
            s * (2.0 * PI * u2).cos(),
            t * (2.0 * PI * u3).sin(),
            t * (2.0 * PI * u3).cos(),
        )
    }

    fn exp(p: &SU2Momentum) -> Self {
        // Exponential map: exp(i θ · σ) where σ are Pauli matrices
        let theta = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if theta < 1e-12 {
            return Self::identity();
        }

        let a = (theta / 2.0).cos();
        let s = (theta / 2.0).sin() / theta;
        Self::new(a, s * p[0], s * p[1], s * p[2])
    }

    fn force(&self, staple: &Self) -> SU2Momentum {
        // Force = -dS/dU = project (U * staple†) to su(2)
        let prod = self.mul(&staple.inv());
        // Extract anti-hermitian traceless part
        // For quaternion q = a + bi + cj + dk:
        // Anti-hermitian part is i(b,c,d) which maps to su(2)
        [prod.q[1], prod.q[2], prod.q[3]]
    }

    fn zero_momentum() -> SU2Momentum {
        [0.0, 0.0, 0.0]
    }

    fn add_momentum(p1: &SU2Momentum, p2: &SU2Momentum) -> SU2Momentum {
        [p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2]]
    }

    fn scale_momentum(p: &SU2Momentum, scale: f64) -> SU2Momentum {
        [p[0] * scale, p[1] * scale, p[2] * scale]
    }

    fn momentum_norm_squared(p: &SU2Momentum) -> f64 {
        p[0] * p[0] + p[1] * p[1] + p[2] * p[2]
    }

    fn sample_momentum() -> SU2Momentum {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut sample_gaussian = || {
            let u1: f64 = rng.r#gen();
            let u2: f64 = rng.r#gen();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        };
        [sample_gaussian(), sample_gaussian(), sample_gaussian()]
    }
}

// ============================================================================
// SU(3) Group
// ============================================================================

/// SU(3) group element as 3×3 complex matrix.
///
/// Stored as 9 complex numbers (18 real values).
/// Must satisfy U† U = I and det(U) = 1.
#[derive(Debug, Clone, Copy)]
pub struct SU3 {
    /// Matrix elements [Re, Im] for each of 9 entries (row-major).
    pub m: [[f64; 2]; 9],
}

impl SU3 {
    /// Create new SU(3) element from 3×3 complex matrix.
    pub fn new(m: [[f64; 2]; 9]) -> Self {
        let mut su3 = Self { m };
        su3.orthonormalize();
        su3
    }

    /// Orthonormalize to ensure SU(3) properties.
    fn orthonormalize(&mut self) {
        // Gram-Schmidt on columns
        // Column 0: normalize
        let norm0 = self.col_norm(0);
        self.scale_col(0, 1.0 / norm0);

        // Column 1: subtract projection on col 0, normalize
        let proj = self.col_dot(1, 0);
        self.col_sub_scaled(1, 0, proj);
        let norm1 = self.col_norm(1);
        self.scale_col(1, 1.0 / norm1);

        // Column 2: cross product of col 0 and col 1 (ensures det = 1)
        self.col_cross(0, 1, 2);
    }

    #[inline]
    fn col_norm(&self, col: usize) -> f64 {
        let mut sum = 0.0;
        for row in 0..3 {
            let idx = row * 3 + col;
            sum += self.m[idx][0] * self.m[idx][0] + self.m[idx][1] * self.m[idx][1];
        }
        sum.sqrt()
    }

    #[inline]
    fn scale_col(&mut self, col: usize, scale: f64) {
        for row in 0..3 {
            let idx = row * 3 + col;
            self.m[idx][0] *= scale;
            self.m[idx][1] *= scale;
        }
    }

    #[inline]
    fn col_dot(&self, col1: usize, col2: usize) -> [f64; 2] {
        let mut re = 0.0;
        let mut im = 0.0;
        for row in 0..3 {
            let idx1 = row * 3 + col1;
            let idx2 = row * 3 + col2;
            // (a+bi)* (c+di) = ac+bd + (ad-bc)i
            re += self.m[idx1][0] * self.m[idx2][0] + self.m[idx1][1] * self.m[idx2][1];
            im += self.m[idx1][0] * self.m[idx2][1] - self.m[idx1][1] * self.m[idx2][0];
        }
        [re, im]
    }

    #[inline]
    fn col_sub_scaled(&mut self, col_dest: usize, col_src: usize, scale: [f64; 2]) {
        for row in 0..3 {
            let dest_idx = row * 3 + col_dest;
            let src_idx = row * 3 + col_src;
            // dest -= scale * src (complex)
            self.m[dest_idx][0] -= scale[0] * self.m[src_idx][0] - scale[1] * self.m[src_idx][1];
            self.m[dest_idx][1] -= scale[0] * self.m[src_idx][1] + scale[1] * self.m[src_idx][0];
        }
    }

    #[inline]
    fn col_cross(&mut self, col1: usize, col2: usize, col_dest: usize) {
        // Cross product for complex 3-vectors: conj(col1) × conj(col2)
        for row in 0..3 {
            let r1 = (row + 1) % 3;
            let r2 = (row + 2) % 3;
            let idx1_1 = r1 * 3 + col1;
            let idx2_1 = r2 * 3 + col2;
            let idx1_2 = r2 * 3 + col1;
            let idx2_2 = r1 * 3 + col2;

            // (a*)(c) - (b*)(d) where a,b from col1 and c,d from col2
            let re = self.m[idx1_1][0] * self.m[idx2_1][0] + self.m[idx1_1][1] * self.m[idx2_1][1]
                - (self.m[idx1_2][0] * self.m[idx2_2][0] + self.m[idx1_2][1] * self.m[idx2_2][1]);
            let im = self.m[idx1_1][0] * self.m[idx2_1][1]
                - self.m[idx1_1][1] * self.m[idx2_1][0]
                - (self.m[idx1_2][0] * self.m[idx2_2][1] - self.m[idx1_2][1] * self.m[idx2_2][0]);

            let dest_idx = row * 3 + col_dest;
            self.m[dest_idx][0] = re;
            self.m[dest_idx][1] = im;
        }
    }
}

/// SU(3) Lie algebra element (8D vector for su(3) generators).
pub type SU3Momentum = [f64; 8];

impl Group for SU3 {
    type Momentum = SU3Momentum;

    fn identity() -> Self {
        let mut m = [[0.0, 0.0]; 9];
        m[0] = [1.0, 0.0]; // (0,0)
        m[4] = [1.0, 0.0]; // (1,1)
        m[8] = [1.0, 0.0]; // (2,2)
        Self { m }
    }

    fn mul(&self, other: &Self) -> Self {
        // Matrix multiplication
        let mut result = [[0.0, 0.0]; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut re = 0.0;
                let mut im = 0.0;
                for k in 0..3 {
                    let idx1 = i * 3 + k;
                    let idx2 = k * 3 + j;
                    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                    re += self.m[idx1][0] * other.m[idx2][0] - self.m[idx1][1] * other.m[idx2][1];
                    im += self.m[idx1][0] * other.m[idx2][1] + self.m[idx1][1] * other.m[idx2][0];
                }
                result[i * 3 + j] = [re, im];
            }
        }
        Self::new(result)
    }

    fn inv(&self) -> Self {
        // Hermitian conjugate (transpose + complex conjugate)
        let mut result = [[0.0, 0.0]; 9];
        for i in 0..3 {
            for j in 0..3 {
                result[i * 3 + j] = [self.m[j * 3 + i][0], -self.m[j * 3 + i][1]];
            }
        }
        Self { m: result }
    }

    fn re_tr(&self) -> f64 {
        // Trace: sum of diagonal elements (real parts)
        self.m[0][0] + self.m[4][0] + self.m[8][0]
    }

    fn random() -> Self {
        // Generate random SU(3) via QR decomposition
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut m = [[0.0, 0.0]; 9];
        for elem in &mut m {
            *elem = [rng.r#gen::<f64>() - 0.5, rng.r#gen::<f64>() - 0.5];
        }
        Self::new(m)
    }

    fn exp(p: &SU3Momentum) -> Self {
        // Exponential map using Gell-Mann matrices (SU(3) generators)
        // Simplified: use small-angle approximation
        // U ≈ I + i θ_a λ_a for small |p|
        let mut m = Self::identity().m;

        // Add anti-hermitian traceless contribution
        // This is a simplified implementation; full version requires
        // proper Gell-Mann basis and matrix exponentiation
        let scale = 0.1; // small step approximation
        for i in 0..8 {
            // Add contribution from each generator (simplified)
            m[i % 9][1] += scale * p[i];
        }

        Self::new(m)
    }

    fn force(&self, staple: &Self) -> SU3Momentum {
        // Force = project (U * staple†) to su(3)
        let prod = self.mul(&staple.inv());

        // Extract anti-hermitian traceless part
        // Project to 8 generators of su(3)
        let mut f = [0.0; 8];

        // Simplified projection (full version needs Gell-Mann basis)
        for (i, elem) in f.iter_mut().enumerate() {
            *elem = prod.m[i % 9][1]; // Imaginary parts approximate force
        }

        f
    }

    fn zero_momentum() -> SU3Momentum {
        [0.0; 8]
    }

    fn add_momentum(p1: &SU3Momentum, p2: &SU3Momentum) -> SU3Momentum {
        let mut result = [0.0; 8];
        for i in 0..8 {
            result[i] = p1[i] + p2[i];
        }
        result
    }

    fn scale_momentum(p: &SU3Momentum, scale: f64) -> SU3Momentum {
        let mut result = [0.0; 8];
        for i in 0..8 {
            result[i] = p[i] * scale;
        }
        result
    }

    fn momentum_norm_squared(p: &SU3Momentum) -> f64 {
        p.iter().map(|&x| x * x).sum()
    }

    fn sample_momentum() -> SU3Momentum {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut sample_gaussian = || {
            let u1: f64 = rng.r#gen();
            let u2: f64 = rng.r#gen();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        };
        let mut p = [0.0; 8];
        for elem in &mut p {
            *elem = sample_gaussian();
        }
        p
    }
}
