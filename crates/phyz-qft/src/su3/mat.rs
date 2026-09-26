//! Complex numbers and 3×3 complex matrices for SU(3) lattice gauge theory.
//!
//! [`Su3`] is a general complex 3×3 matrix, not just a group element: a staple
//! sum or a clover leaf average leaves the group, and the heatbath and field
//! strength code needs both.

use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// Complex number, kept minimal and `Copy` so the 3×3 kernels stay flat.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
}

impl Add for C64 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
}

impl AddAssign for C64 {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        self.re += o.re;
        self.im += o.im;
    }
}

impl Sub for C64 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
}

impl SubAssign for C64 {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        self.re -= o.re;
        self.im -= o.im;
    }
}

impl Mul for C64 {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl Neg for C64 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

/// Complex 3×3 matrix, row-major. SU(3) when unitary with unit determinant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Su3 {
    pub m: [[C64; 3]; 3],
}

impl Default for Su3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Su3 {
    pub const ZERO: Self = Self {
        m: [[C64::ZERO; 3]; 3],
    };

    pub const IDENTITY: Self = Self {
        m: [
            [C64::ONE, C64::ZERO, C64::ZERO],
            [C64::ZERO, C64::ONE, C64::ZERO],
            [C64::ZERO, C64::ZERO, C64::ONE],
        ],
    };

    /// Conjugate transpose.
    #[inline]
    pub fn dagger(&self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] = self.m[j][i].conj();
            }
        }
        r
    }

    #[inline]
    pub fn trace(&self) -> C64 {
        self.m[0][0] + self.m[1][1] + self.m[2][2]
    }

    #[inline]
    pub fn re_tr(&self) -> f64 {
        self.m[0][0].re + self.m[1][1].re + self.m[2][2].re
    }

    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        let mut r = *self;
        for row in &mut r.m {
            for e in row {
                *e = e.scale(s);
            }
        }
        r
    }

    /// `self · other†` without materialising the dagger.
    #[inline]
    pub fn mul_dag(&self, o: &Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                let mut s = C64::ZERO;
                for k in 0..3 {
                    s += self.m[i][k] * o.m[j][k].conj();
                }
                r.m[i][j] = s;
            }
        }
        r
    }

    /// `self† · other` without materialising the dagger.
    #[inline]
    pub fn dag_mul(&self, o: &Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                let mut s = C64::ZERO;
                for k in 0..3 {
                    s += self.m[k][i].conj() * o.m[k][j];
                }
                r.m[i][j] = s;
            }
        }
        r
    }

    /// Frobenius distance squared, used by tests for unitarity checks.
    pub fn dist_sqr(&self, o: &Self) -> f64 {
        let mut s = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                s += (self.m[i][j] - o.m[i][j]).norm_sqr();
            }
        }
        s
    }

    pub fn det(&self) -> C64 {
        let m = &self.m;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    /// Project back onto SU(3): Gram-Schmidt the first two rows, then set the
    /// third to the conjugate cross product so det = 1 exactly.
    pub fn reunitarize(&mut self) {
        let [mut r0, mut r1, _] = self.m;

        let n0 = r0.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        for c in &mut r0 {
            *c = c.scale(1.0 / n0);
        }

        // r1 -= <r0, r1> r0
        let mut dot = C64::ZERO;
        for k in 0..3 {
            dot += r0[k].conj() * r1[k];
        }
        for k in 0..3 {
            r1[k] -= dot * r0[k];
        }
        let n1 = r1.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        for c in &mut r1 {
            *c = c.scale(1.0 / n1);
        }

        let r2 = [
            (r0[1] * r1[2] - r0[2] * r1[1]).conj(),
            (r0[2] * r1[0] - r0[0] * r1[2]).conj(),
            (r0[0] * r1[1] - r0[1] * r1[0]).conj(),
        ];

        self.m = [r0, r1, r2];
    }

    /// Anti-Hermitian traceless part: `(M − M†)/2 − Tr(M − M†)/6 · 1`.
    pub fn traceless_antihermitian(&self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] = (self.m[i][j] - self.m[j][i].conj()).scale(0.5);
            }
        }
        let tr = r.trace().scale(1.0 / 3.0);
        for i in 0..3 {
            r.m[i][i] -= tr;
        }
        r
    }

    /// Matrix exponential of an anti-Hermitian traceless `Q`, via a truncated
    /// Taylor series followed by reunitarization. Accurate to machine precision
    /// for the |Q| ≲ 1 steps that smearing and flow take.
    pub fn exp_antihermitian(q: &Self) -> Self {
        let mut result = Self::IDENTITY;
        let mut term = Self::IDENTITY;
        for n in 1..=12 {
            term = (term * *q).scale(1.0 / n as f64);
            result += term;
        }
        result.reunitarize();
        result
    }
}

impl Add for Su3 {
    type Output = Self;
    #[inline]
    fn add(mut self, o: Self) -> Self {
        self += o;
        self
    }
}

impl AddAssign for Su3 {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        for i in 0..3 {
            for j in 0..3 {
                self.m[i][j] += o.m[i][j];
            }
        }
    }
}

impl Sub for Su3 {
    type Output = Self;
    #[inline]
    fn sub(mut self, o: Self) -> Self {
        for i in 0..3 {
            for j in 0..3 {
                self.m[i][j] -= o.m[i][j];
            }
        }
        self
    }
}

impl Mul for Su3 {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                let mut s = C64::ZERO;
                for k in 0..3 {
                    s += self.m[i][k] * o.m[k][j];
                }
                r.m[i][j] = s;
            }
        }
        r
    }
}

/// Unit quaternion `a0 + i(a1 σ1 + a2 σ2 + a3 σ3)` as an SU(2) matrix
/// `[[a0 + i a3, a2 + i a1], [−a2 + i a1, a0 − i a3]]`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Quat(pub [f64; 4]);

impl Quat {
    /// Project a complex 2×2 block onto the real span of SU(2). This is the
    /// orthogonal projection under `Re Tr`, so `Re Tr(X w) = Re Tr(X · proj(w))`
    /// for any SU(2) `X`.
    #[inline]
    pub fn project(w: [[C64; 2]; 2]) -> Self {
        Self([
            0.5 * (w[0][0].re + w[1][1].re),
            0.5 * (w[0][1].im + w[1][0].im),
            0.5 * (w[0][1].re - w[1][0].re),
            0.5 * (w[0][0].im - w[1][1].im),
        ])
    }

    #[inline]
    pub fn norm(&self) -> f64 {
        self.0.iter().map(|a| a * a).sum::<f64>().sqrt()
    }

    #[inline]
    pub fn conj(&self) -> Self {
        let [a0, a1, a2, a3] = self.0;
        Self([a0, -a1, -a2, -a3])
    }

    #[inline]
    pub fn mul(&self, o: &Self) -> Self {
        // (a0 + i a·σ)(b0 + i b·σ) = a0 b0 − a·b + i(a0 b + b0 a − a × b)
        let [a0, a1, a2, a3] = self.0;
        let [b0, b1, b2, b3] = o.0;
        Self([
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + b0 * a1 - (a2 * b3 - a3 * b2),
            a0 * b2 + b0 * a2 - (a3 * b1 - a1 * b3),
            a0 * b3 + b0 * a3 - (a1 * b2 - a2 * b1),
        ])
    }

    #[inline]
    pub fn to_mat(self) -> [[C64; 2]; 2] {
        let [a0, a1, a2, a3] = self.0;
        [
            [C64::new(a0, a3), C64::new(a2, a1)],
            [C64::new(-a2, a1), C64::new(a0, -a3)],
        ]
    }
}

/// The three SU(2) subgroups that Cabibbo-Marinari cycles through.
pub(crate) const SUBGROUPS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

/// Left-multiply rows `i`, `j` of `u` by the 2×2 matrix `x`.
#[inline]
pub(crate) fn left_mul_sub(u: &mut Su3, (i, j): (usize, usize), x: &[[C64; 2]; 2]) {
    for col in 0..3 {
        let ui = u.m[i][col];
        let uj = u.m[j][col];
        u.m[i][col] = x[0][0] * ui + x[0][1] * uj;
        u.m[j][col] = x[1][0] * ui + x[1][1] * uj;
    }
}

#[inline]
pub(crate) fn sub_block(u: &Su3, (i, j): (usize, usize)) -> [[C64; 2]; 2] {
    [[u.m[i][i], u.m[i][j]], [u.m[j][i], u.m[j][j]]]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Su3 {
        let mut m = Su3::ZERO;
        let vals = [0.3, -1.2, 0.7, 0.1, 0.9, -0.4, 1.1, 0.2, -0.8];
        for i in 0..3 {
            for j in 0..3 {
                m.m[i][j] = C64::new(vals[i * 3 + j], vals[(i * 3 + j + 4) % 9]);
            }
        }
        m
    }

    #[test]
    fn reunitarize_gives_su3() {
        let mut u = sample();
        u.reunitarize();
        assert!(u.mul_dag(&u).dist_sqr(&Su3::IDENTITY) < 1e-24);
        let d = u.det();
        assert!((d.re - 1.0).abs() < 1e-12 && d.im.abs() < 1e-12);
    }

    #[test]
    fn dagger_products_match() {
        let a = sample();
        let mut b = sample().scale(0.5);
        b.m[0][2] = C64::new(2.0, -1.0);
        assert!(a.mul_dag(&b).dist_sqr(&(a * b.dagger())) < 1e-24);
        assert!(a.dag_mul(&b).dist_sqr(&(a.dagger() * b)) < 1e-24);
    }

    #[test]
    fn exp_is_unitary_and_matches_series() {
        let mut q = sample().traceless_antihermitian().scale(0.3);
        assert!(q.trace().norm_sqr() < 1e-24);
        let e = Su3::exp_antihermitian(&q);
        assert!(e.mul_dag(&e).dist_sqr(&Su3::IDENTITY) < 1e-24);
        // exp(Q) exp(−Q) = 1
        q = q.scale(-1.0);
        let e_inv = Su3::exp_antihermitian(&q);
        assert!((e * e_inv).dist_sqr(&Su3::IDENTITY) < 1e-20);
    }

    #[test]
    fn quat_matches_matrix_product() {
        let a = Quat([0.5, 0.5, 0.5, 0.5]);
        let b = Quat([0.8, 0.0, 0.6, 0.0]);
        let ab = a.mul(&b).to_mat();
        let (am, bm) = (a.to_mat(), b.to_mat());
        for i in 0..2 {
            for j in 0..2 {
                let want = am[i][0] * bm[0][j] + am[i][1] * bm[1][j];
                assert!((ab[i][j] - want).norm_sqr() < 1e-24);
            }
        }
        // projection of an SU(2) matrix is itself
        let p = Quat::project(am);
        for k in 0..4 {
            assert!((p.0[k] - a.0[k]).abs() < 1e-12);
        }
    }
}
