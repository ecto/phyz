//! A minimal in-place complex FFT, enough to drive the PME reciprocal sum.
//!
//! PME only needs transforms on a mesh whose dimensions we choose ourselves, so
//! restricting to powers of two (rounded up from the requested mesh) buys a
//! ~150-line radix-2 implementation instead of a dependency.

use std::f64::consts::TAU;

/// A complex number in the PME mesh transforms.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Cplx {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Cplx {
    /// A complex number from its parts.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// A purely real complex number.
    #[inline]
    pub const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    /// `e^{iθ}`.
    #[inline]
    pub fn expi(theta: f64) -> Self {
        Self {
            re: theta.cos(),
            im: theta.sin(),
        }
    }

    /// Complex product.
    #[inline]
    #[allow(clippy::should_implement_trait)] // deliberately inherent: no operator overloading for a private numeric helper
    pub fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }

    /// Squared modulus `|z|²`.
    #[inline]
    pub fn norm2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Complex conjugate.
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

/// Round `n` up to the next power of two (minimum 1).
pub fn next_pow2(n: usize) -> usize {
    n.max(1).next_power_of_two()
}

/// In-place radix-2 FFT of a power-of-two-length slice.
///
/// `sign = -1.0` is the forward (unnormalized) transform
/// `X_k = Σ_j x_j e^{-2πijk/N}`; `sign = +1.0` is the unnormalized inverse.
/// Neither direction applies a `1/N` factor — the PME convolution is arranged
/// so the normalization cancels.
pub fn fft_1d(buf: &mut [Cplx], sign: f64) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two(), "fft_1d requires a power-of-two length");
    if n < 2 {
        return;
    }
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    // Danielson–Lanczos butterflies.
    let mut len = 2;
    while len <= n {
        let ang = sign * TAU / len as f64;
        let wlen = Cplx::expi(ang);
        let mut start = 0;
        while start < n {
            let mut w = Cplx::real(1.0);
            for k in 0..len / 2 {
                let u = buf[start + k];
                let v = buf[start + k + len / 2].mul(w);
                buf[start + k] = Cplx::new(u.re + v.re, u.im + v.im);
                buf[start + k + len / 2] = Cplx::new(u.re - v.re, u.im - v.im);
                w = w.mul(wlen);
            }
            start += len;
        }
        len <<= 1;
    }
}

/// In-place 3D FFT over a flat array in row-major (`x` slowest, `z` fastest)
/// order with power-of-two dimensions.
pub fn fft_3d(buf: &mut [Cplx], dims: [usize; 3], sign: f64) {
    let [nx, ny, nz] = dims;
    debug_assert_eq!(buf.len(), nx * ny * nz);

    // Transform along z (contiguous).
    for plane in buf.chunks_mut(nz) {
        fft_1d(plane, sign);
    }
    // Transform along y and x with a gather/scatter through a scratch line.
    let mut line = vec![Cplx::default(); ny.max(nx)];
    for ix in 0..nx {
        for iz in 0..nz {
            for (iy, slot) in line[..ny].iter_mut().enumerate() {
                *slot = buf[(ix * ny + iy) * nz + iz];
            }
            fft_1d(&mut line[..ny], sign);
            for iy in 0..ny {
                buf[(ix * ny + iy) * nz + iz] = line[iy];
            }
        }
    }
    for iy in 0..ny {
        for iz in 0..nz {
            for (ix, slot) in line[..nx].iter_mut().enumerate() {
                *slot = buf[(ix * ny + iy) * nz + iz];
            }
            fft_1d(&mut line[..nx], sign);
            for ix in 0..nx {
                buf[(ix * ny + iy) * nz + iz] = line[ix];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dft_naive(x: &[Cplx], sign: f64) -> Vec<Cplx> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut acc = Cplx::default();
                for (j, xj) in x.iter().enumerate() {
                    let w = Cplx::expi(sign * TAU * (j * k) as f64 / n as f64);
                    let t = xj.mul(w);
                    acc.re += t.re;
                    acc.im += t.im;
                }
                acc
            })
            .collect()
    }

    #[test]
    fn fft_matches_naive_dft() {
        let n = 16;
        let x: Vec<Cplx> = (0..n)
            .map(|i| Cplx::new((i as f64 * 0.7).sin(), (i as f64 * 1.3).cos()))
            .collect();
        let want = dft_naive(&x, -1.0);
        let mut got = x.clone();
        fft_1d(&mut got, -1.0);
        for (g, w) in got.iter().zip(&want) {
            assert!((g.re - w.re).abs() < 1e-10 && (g.im - w.im).abs() < 1e-10);
        }
    }

    #[test]
    fn forward_then_inverse_recovers_input_scaled_by_n() {
        let n = 32;
        let x: Vec<Cplx> = (0..n).map(|i| Cplx::new(i as f64, -(i as f64))).collect();
        let mut y = x.clone();
        fft_1d(&mut y, -1.0);
        fft_1d(&mut y, 1.0);
        for (yi, xi) in y.iter().zip(&x) {
            assert!((yi.re / n as f64 - xi.re).abs() < 1e-9);
            assert!((yi.im / n as f64 - xi.im).abs() < 1e-9);
        }
    }

    #[test]
    fn fft_3d_roundtrips() {
        let dims = [4usize, 8, 8];
        let n = dims[0] * dims[1] * dims[2];
        let x: Vec<Cplx> = (0..n)
            .map(|i| Cplx::new((i as f64 * 0.31).sin(), 0.0))
            .collect();
        let mut y = x.clone();
        fft_3d(&mut y, dims, -1.0);
        fft_3d(&mut y, dims, 1.0);
        for (yi, xi) in y.iter().zip(&x) {
            assert!((yi.re / n as f64 - xi.re).abs() < 1e-9);
        }
    }

    #[test]
    fn fft_3d_matches_separable_dft_on_a_delta() {
        // A delta at the origin transforms to a constant 1 everywhere.
        let dims = [4usize, 4, 4];
        let mut buf = vec![Cplx::default(); 64];
        buf[0] = Cplx::real(1.0);
        fft_3d(&mut buf, dims, -1.0);
        for b in &buf {
            assert!((b.re - 1.0).abs() < 1e-12 && b.im.abs() < 1e-12);
        }
    }
}
