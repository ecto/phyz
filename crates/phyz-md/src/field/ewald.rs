//! Long-range electrostatics: Ewald summation and smooth particle mesh Ewald.
//!
//! Under periodic boundaries the Coulomb lattice sum is only conditionally
//! convergent — truncating it at a cutoff does not converge to the right answer
//! at all, it converges to a different one. Ewald's splitting
//!
//! ```text
//! 1/r = erfc(αr)/r  +  erf(αr)/r
//! ```
//!
//! sends the first (short-ranged) piece to a real-space sum inside `r_cut` and
//! the second (smooth) piece to an absolutely convergent reciprocal-space sum,
//! with a self-energy term removing each charge's interaction with its own
//! screening cloud.
//!
//! Two reciprocal-space back ends share that real-space half:
//!
//! - [`Ewald`] — the direct `Σ_k` structure-factor sum. O(N^{3/2}) when tuned,
//!   exact to the requested k-space cutoff, and the reference implementation.
//! - [`Pme`] — smooth particle mesh Ewald (Essmann et al., 1995): charges are
//!   B-spline-interpolated onto a mesh and the convolution is done by FFT,
//!   giving O(N log N). This is the production path.
//!
//! Both are validated against the Madelung constant of rocksalt NaCl, the
//! canonical exact result for a periodic ionic lattice.

use std::f64::consts::PI;

use super::cell::{Lattice, min_image, vec3};
use super::fft::{Cplx, fft_3d, next_pow2};
use super::special::{erf, erfc};
use super::units::KE_COULOMB;
use super::virial::Contribution;

/// Ewald summation with an explicit reciprocal-space sum.
#[derive(Debug, Clone, PartialEq)]
pub struct Ewald {
    /// Splitting parameter in 1/Å. Larger α shifts work from real to
    /// reciprocal space.
    pub alpha: f64,
    /// Real-space cutoff in Å.
    pub r_cut: f64,
    /// Maximum integer reciprocal index along each lattice direction.
    pub k_max: [i32; 3],
}

impl Ewald {
    /// Ewald with explicit parameters.
    pub fn new(alpha: f64, r_cut: f64, k_max: [i32; 3]) -> Self {
        Self {
            alpha,
            r_cut,
            k_max,
        }
    }

    /// Choose `alpha` and `k_max` for a target relative accuracy (e.g. `1e-8`)
    /// at the given real-space cutoff.
    ///
    /// `alpha` is set so the real-space term is negligible beyond `r_cut`, then
    /// `k_max` so the reciprocal term is negligible beyond the corresponding
    /// wavevector.
    pub fn tuned(cell: &Lattice, r_cut: f64, accuracy: f64) -> Self {
        let alpha = tune_alpha(r_cut, accuracy);
        let k_max = tune_kmax(cell, alpha, accuracy);
        Self::new(alpha, r_cut, k_max)
    }

    /// The complete electrostatic energy, forces, and virial.
    ///
    /// `pairs` is a neighbor list covering at least every pair within `r_cut`;
    /// pass `None` to use the all-pairs minimum-image loop. `exclusions` lists
    /// bonded pairs whose direct interaction is not wanted — their
    /// reciprocal-space contribution is subtracted analytically, which is the
    /// only correct way to exclude a pair from a mesh/k-space sum.
    pub fn compute(
        &self,
        charges: &[f64],
        positions: &[[f64; 3]],
        cell: Option<&Lattice>,
        pairs: Option<&[(usize, usize)]>,
        exclusions: &[(usize, usize)],
    ) -> Contribution {
        let mut acc = short_range(self.alpha, self.r_cut, charges, positions, cell, pairs);
        acc.merge(&exclusion_correction(
            self.alpha, charges, positions, cell, exclusions,
        ));
        acc.energy += self_energy(self.alpha, charges);
        if let Some(c) = cell {
            acc.merge(&self.reciprocal(charges, positions, c));
            acc.merge(&background(self.alpha, charges, c.volume().abs()));
        }
        acc
    }

    /// The reciprocal-space sum alone.
    pub fn reciprocal(
        &self,
        charges: &[f64],
        positions: &[[f64; 3]],
        cell: &Lattice,
    ) -> Contribution {
        let n = positions.len();
        let mut acc = Contribution::zeros(n);
        let Some(recip) = cell.reciprocal() else {
            return acc;
        };
        let volume = cell.volume().abs();
        if volume <= 0.0 {
            return acc;
        }
        let pref = 2.0 * PI * KE_COULOMB / volume;
        let inv_4a2 = 1.0 / (4.0 * self.alpha * self.alpha);

        // Scratch reused across k-vectors.
        let mut cos_kr = vec![0.0f64; n];
        let mut sin_kr = vec![0.0f64; n];

        for m0 in -self.k_max[0]..=self.k_max[0] {
            for m1 in -self.k_max[1]..=self.k_max[1] {
                for m2 in -self.k_max[2]..=self.k_max[2] {
                    if m0 == 0 && m1 == 0 && m2 == 0 {
                        continue;
                    }
                    let k = [
                        m0 as f64 * recip[0][0] + m1 as f64 * recip[1][0] + m2 as f64 * recip[2][0],
                        m0 as f64 * recip[0][1] + m1 as f64 * recip[1][1] + m2 as f64 * recip[2][1],
                        m0 as f64 * recip[0][2] + m1 as f64 * recip[1][2] + m2 as f64 * recip[2][2],
                    ];
                    let k2 = vec3::norm2(k);
                    if k2 <= 0.0 {
                        continue;
                    }
                    let damp = (-k2 * inv_4a2).exp() / k2;
                    if damp < 1e-30 {
                        continue;
                    }
                    // S(k) = Σ_j q_j e^{i k·r_j}
                    let mut s_re = 0.0;
                    let mut s_im = 0.0;
                    for j in 0..n {
                        let kr = vec3::dot(k, positions[j]);
                        let (s, c) = kr.sin_cos();
                        cos_kr[j] = c;
                        sin_kr[j] = s;
                        s_re += charges[j] * c;
                        s_im += charges[j] * s;
                    }
                    let s2 = s_re * s_re + s_im * s_im;
                    let term = pref * damp * s2;
                    acc.energy += term;

                    // F_i = 2·pref·damp·q_i·k·Im[conj(S) e^{i k·r_i}]
                    //     = 2·pref·damp·q_i·k·(S_re sin(k·r_i) − S_im cos(k·r_i))
                    for i in 0..n {
                        let im = s_re * sin_kr[i] - s_im * cos_kr[i];
                        let scale = 2.0 * pref * damp * charges[i] * im;
                        vec3::add_assign(&mut acc.forces[i], vec3::scale(k, scale));
                    }

                    // W_ab = Σ_k term·[δ_ab − 2(1/k² + 1/4α²) k_a k_b]
                    let coef = 2.0 * (1.0 / k2 + inv_4a2);
                    for a in 0..3 {
                        for b in 0..3 {
                            let delta = if a == b { 1.0 } else { 0.0 };
                            acc.virial[a][b] += term * (delta - coef * k[a] * k[b]);
                        }
                    }
                }
            }
        }
        acc
    }
}

/// Smooth particle mesh Ewald: the reciprocal sum by B-spline interpolation
/// onto a mesh plus an FFT convolution.
#[derive(Debug, Clone, PartialEq)]
pub struct Pme {
    /// Splitting parameter in 1/Å.
    pub alpha: f64,
    /// Real-space cutoff in Å.
    pub r_cut: f64,
    /// Mesh dimensions. Rounded up to powers of two by the constructors, as
    /// required by the bundled radix-2 FFT.
    pub mesh: [usize; 3],
    /// B-spline interpolation order (even, ≥ 4 in practice).
    pub order: usize,
}

impl Pme {
    /// PME with explicit parameters. `mesh` is rounded up to powers of two.
    pub fn new(alpha: f64, r_cut: f64, mesh: [usize; 3], order: usize) -> Self {
        let order = order.max(3);
        let mesh = [
            next_pow2(mesh[0].max(order + 1)),
            next_pow2(mesh[1].max(order + 1)),
            next_pow2(mesh[2].max(order + 1)),
        ];
        Self {
            alpha,
            r_cut,
            mesh,
            order,
        }
    }

    /// Choose parameters for a target accuracy at the given cutoff, with a mesh
    /// spacing of roughly `spacing` Å (1.0 Å is a common production value).
    pub fn tuned(cell: &Lattice, r_cut: f64, accuracy: f64, spacing: f64) -> Self {
        let alpha = tune_alpha(r_cut, accuracy);
        let widths = cell.perp_widths();
        let mesh = [
            (widths[0] / spacing).ceil().max(4.0) as usize,
            (widths[1] / spacing).ceil().max(4.0) as usize,
            (widths[2] / spacing).ceil().max(4.0) as usize,
        ];
        Self::new(alpha, r_cut, mesh, 6)
    }

    /// The complete electrostatic energy, forces, and virial. Same contract as
    /// [`Ewald::compute`].
    pub fn compute(
        &self,
        charges: &[f64],
        positions: &[[f64; 3]],
        cell: Option<&Lattice>,
        pairs: Option<&[(usize, usize)]>,
        exclusions: &[(usize, usize)],
    ) -> Contribution {
        let mut acc = short_range(self.alpha, self.r_cut, charges, positions, cell, pairs);
        acc.merge(&exclusion_correction(
            self.alpha, charges, positions, cell, exclusions,
        ));
        acc.energy += self_energy(self.alpha, charges);
        if let Some(c) = cell {
            acc.merge(&self.reciprocal(charges, positions, c));
            acc.merge(&background(self.alpha, charges, c.volume().abs()));
        }
        acc
    }

    /// The mesh-based reciprocal-space sum alone.
    // The mesh loops index several parallel arrays (spline weights per axis,
    // per-axis B-spline moduli, the flat grid) by the same counter; iterator
    // pairs would obscure rather than clarify the index arithmetic.
    #[allow(clippy::needless_range_loop)]
    pub fn reciprocal(
        &self,
        charges: &[f64],
        positions: &[[f64; 3]],
        cell: &Lattice,
    ) -> Contribution {
        let n = positions.len();
        let mut acc = Contribution::zeros(n);
        let (Some(hinv), volume) = (cell.inverse(), cell.volume().abs()) else {
            return acc;
        };
        if volume <= 0.0 {
            return acc;
        }
        let [kx, ky, kz] = self.mesh;
        let ngrid = kx * ky * kz;
        let order = self.order;

        // --- Charge spreading -------------------------------------------
        // Scaled fractional coordinates u = K · (H⁻¹ r), wrapped to [0, K).
        let mut base = vec![[0i64; 3]; n];
        // Per-atom spline weights and derivatives, laid out [atom][t][axis].
        let mut spline = vec![vec![[0.0f64; 3]; order]; n];
        let mut dspline = vec![vec![[0.0f64; 3]; order]; n];

        for i in 0..n {
            let s = [
                hinv[0][0] * positions[i][0]
                    + hinv[0][1] * positions[i][1]
                    + hinv[0][2] * positions[i][2],
                hinv[1][0] * positions[i][0]
                    + hinv[1][1] * positions[i][1]
                    + hinv[1][2] * positions[i][2],
                hinv[2][0] * positions[i][0]
                    + hinv[2][1] * positions[i][1]
                    + hinv[2][2] * positions[i][2],
            ];
            let dims = [kx, ky, kz];
            for ax in 0..3 {
                let u = s[ax].rem_euclid(1.0) * dims[ax] as f64;
                let b = u.floor();
                let w = u - b;
                base[i][ax] = b as i64;
                for t in 0..order {
                    // Grid point base − t receives M_order(w + t).
                    spline[i][t][ax] = bspline_direct(w + t as f64, order);
                    dspline[i][t][ax] = dbspline(w + t as f64, order);
                }
            }
        }

        let mut q = vec![Cplx::default(); ngrid];
        for i in 0..n {
            if charges[i] == 0.0 {
                continue;
            }
            for tx in 0..order {
                let gx = wrap_index(base[i][0] - tx as i64, kx);
                let wx = spline[i][tx][0];
                if wx == 0.0 {
                    continue;
                }
                for ty in 0..order {
                    let gy = wrap_index(base[i][1] - ty as i64, ky);
                    let wxy = wx * spline[i][ty][1];
                    if wxy == 0.0 {
                        continue;
                    }
                    for tz in 0..order {
                        let gz = wrap_index(base[i][2] - tz as i64, kz);
                        let w = wxy * spline[i][tz][2];
                        q[(gx * ky + gy) * kz + gz].re += charges[i] * w;
                    }
                }
            }
        }

        // --- Convolution -------------------------------------------------
        let mut fq = q.clone();
        fft_3d(&mut fq, self.mesh, -1.0);

        let bx = bspline_moduli(kx, order);
        let by = bspline_moduli(ky, order);
        let bz = bspline_moduli(kz, order);
        let inv_4a2 = 1.0 / (4.0 * self.alpha * self.alpha);

        let mut theta_fq = vec![Cplx::default(); ngrid];
        for ix in 0..kx {
            let m0 = signed_index(ix, kx);
            for iy in 0..ky {
                let m1 = signed_index(iy, ky);
                for iz in 0..kz {
                    let m2 = signed_index(iz, kz);
                    let idx = (ix * ky + iy) * kz + iz;
                    if m0 == 0 && m1 == 0 && m2 == 0 {
                        continue;
                    }
                    // k = 2π (m0 A* + m1 B* + m2 C*), rows of H⁻¹ times 2π.
                    let k = [
                        std::f64::consts::TAU
                            * (m0 as f64 * hinv[0][0]
                                + m1 as f64 * hinv[1][0]
                                + m2 as f64 * hinv[2][0]),
                        std::f64::consts::TAU
                            * (m0 as f64 * hinv[0][1]
                                + m1 as f64 * hinv[1][1]
                                + m2 as f64 * hinv[2][1]),
                        std::f64::consts::TAU
                            * (m0 as f64 * hinv[0][2]
                                + m1 as f64 * hinv[1][2]
                                + m2 as f64 * hinv[2][2]),
                    ];
                    let k2 = vec3::norm2(k);
                    if k2 <= 0.0 {
                        continue;
                    }
                    let b = bx[ix] * by[iy] * bz[iz];
                    let theta = 2.0 * PI * KE_COULOMB / volume * (-k2 * inv_4a2).exp() / k2 * b;
                    // E = Σ_{m≠0} Θ(m)|F(Q)(m)|². The mesh sum runs over both
                    // +m and −m, exactly like the direct sum's `Σ_{k≠0}`, so
                    // there is no extra ½ here.
                    let e_term = theta * fq[idx].norm2();
                    acc.energy += e_term;

                    // Same reciprocal virial as the direct Ewald sum.
                    let coef = 2.0 * (1.0 / k2 + inv_4a2);
                    for a in 0..3 {
                        for bb in 0..3 {
                            let delta = if a == bb { 1.0 } else { 0.0 };
                            acc.virial[a][bb] += e_term * (delta - coef * k[a] * k[bb]);
                        }
                    }

                    // dE/dQ[g] = Re(conv[g]) where conv is the unnormalized
                    // inverse transform of 2Θ·F(Q).
                    theta_fq[idx] = Cplx::new(2.0 * theta * fq[idx].re, 2.0 * theta * fq[idx].im);
                }
            }
        }

        // conv = unnormalized inverse FFT of Θ·F(Q); then dE/dQ[g] = Re conv[g].
        let mut conv = theta_fq;
        fft_3d(&mut conv, self.mesh, 1.0);

        // --- Force interpolation -----------------------------------------
        // du_ax/dr_c = K_ax · H⁻¹[ax][c]
        let dims = [kx as f64, ky as f64, kz as f64];
        for i in 0..n {
            if charges[i] == 0.0 {
                continue;
            }
            // dE/du along each mesh axis.
            let mut de_du = [0.0f64; 3];
            for tx in 0..order {
                let gx = wrap_index(base[i][0] - tx as i64, kx);
                for ty in 0..order {
                    let gy = wrap_index(base[i][1] - ty as i64, ky);
                    for tz in 0..order {
                        let gz = wrap_index(base[i][2] - tz as i64, kz);
                        let c = conv[(gx * ky + gy) * kz + gz].re;
                        let (sx, sy, sz) = (spline[i][tx][0], spline[i][ty][1], spline[i][tz][2]);
                        de_du[0] += c * charges[i] * dspline[i][tx][0] * sy * sz;
                        de_du[1] += c * charges[i] * sx * dspline[i][ty][1] * sz;
                        de_du[2] += c * charges[i] * sx * sy * dspline[i][tz][2];
                    }
                }
            }
            for c in 0..3 {
                let mut g = 0.0;
                for ax in 0..3 {
                    g += de_du[ax] * dims[ax] * hinv[ax][c];
                }
                acc.forces[i][c] -= g;
            }
        }

        acc
    }
}

/// Real-space Ewald sum: `Σ ke q_i q_j erfc(α r)/r` inside `r_cut`.
fn short_range(
    alpha: f64,
    r_cut: f64,
    charges: &[f64],
    positions: &[[f64; 3]],
    cell: Option<&Lattice>,
    pairs: Option<&[(usize, usize)]>,
) -> Contribution {
    let n = positions.len();
    let mut acc = Contribution::zeros(n);
    let rc2 = r_cut * r_cut;
    let two_a_sqrtpi = 2.0 * alpha / PI.sqrt();

    let handle = |i: usize, j: usize, acc: &mut Contribution| {
        if charges[i] == 0.0 || charges[j] == 0.0 {
            return;
        }
        let d = min_image(vec3::sub(positions[i], positions[j]), cell);
        let r2 = vec3::norm2(d);
        if r2 > rc2 || r2 < 1e-24 {
            return;
        }
        let r = r2.sqrt();
        let qq = KE_COULOMB * charges[i] * charges[j];
        let ar = alpha * r;
        acc.energy += qq * erfc(ar) / r;
        // −dE/dr = qq [erfc(αr)/r² + (2α/√π) e^{−α²r²}/r]; times d/r for the
        // vector force on i.
        let fmag = qq * (erfc(ar) / r + two_a_sqrtpi * (-ar * ar).exp()) / r2;
        let f = vec3::scale(d, fmag);
        acc.add_pair_force(i, j, d, f);
    };

    match pairs {
        Some(p) => {
            for &(i, j) in p {
                handle(i, j, &mut acc);
            }
        }
        None => {
            for i in 0..n {
                for j in (i + 1)..n {
                    handle(i, j, &mut acc);
                }
            }
        }
    }
    acc
}

/// `−ke α/√π Σ q_i²`: each charge's interaction with its own screening cloud.
fn self_energy(alpha: f64, charges: &[f64]) -> f64 {
    let sum_q2: f64 = charges.iter().map(|q| q * q).sum();
    -KE_COULOMB * alpha / PI.sqrt() * sum_q2
}

/// Neutralizing-background term, `−ke π (Σq)² / (2 α² V)`.
///
/// Zero for a charge-neutral cell, but a non-neutral cell in a periodic box is
/// only meaningful against a uniform compensating background, and this is its
/// energy. It depends on volume, so it carries a virial.
fn background(alpha: f64, charges: &[f64], volume: f64) -> Contribution {
    let mut acc = Contribution::default();
    if volume <= 0.0 {
        return acc;
    }
    let qtot: f64 = charges.iter().sum();
    if qtot.abs() < 1e-14 {
        return acc;
    }
    let e = -KE_COULOMB * PI * qtot * qtot / (2.0 * alpha * alpha * volume);
    acc.energy = e;
    // E ∝ 1/V, so Tr W = −3V dE/dV = 3 E, split isotropically.
    for a in 0..3 {
        acc.virial[a][a] += e;
    }
    acc
}

/// Remove the reciprocal-space interaction of excluded (bonded) pairs.
///
/// The mesh/k-space sum interacts *every* pair, including 1-2 and 1-3 partners
/// that a bonded force field handles explicitly. Simply omitting them from the
/// real-space loop is not enough — the smooth `erf(αr)/r` part still counts
/// them, so it has to be subtracted analytically.
fn exclusion_correction(
    alpha: f64,
    charges: &[f64],
    positions: &[[f64; 3]],
    cell: Option<&Lattice>,
    exclusions: &[(usize, usize)],
) -> Contribution {
    let mut acc = Contribution::zeros(positions.len());
    if exclusions.is_empty() {
        return acc;
    }
    let two_a_sqrtpi = 2.0 * alpha / PI.sqrt();
    for &(i, j) in exclusions {
        if i == j || charges[i] == 0.0 || charges[j] == 0.0 {
            continue;
        }
        let d = min_image(vec3::sub(positions[i], positions[j]), cell);
        let r2 = vec3::norm2(d);
        if r2 < 1e-24 {
            continue;
        }
        let r = r2.sqrt();
        let qq = KE_COULOMB * charges[i] * charges[j];
        let ar = alpha * r;
        acc.energy -= qq * erf(ar) / r;
        // E = −qq g(r) with g = erf(αr)/r and
        // g'(r) = (2α/√π) e^{−α²r²}/r − erf(αr)/r², so
        // F_i = −dE/dr · d̂ = qq g'(r) · d/r.
        let fmag = qq * (two_a_sqrtpi * (-ar * ar).exp() - erf(ar) / r) / r2;
        let f = vec3::scale(d, fmag);
        acc.add_pair_force(i, j, d, f);
    }
    acc
}

/// Solve `erfc(α r_cut)/r_cut = accuracy` for α by bisection.
fn tune_alpha(r_cut: f64, accuracy: f64) -> f64 {
    let target = accuracy.clamp(1e-16, 1e-1);
    let (mut lo, mut hi) = (0.0f64, 10.0f64 / r_cut.max(1e-6));
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if erfc(mid * r_cut) / r_cut > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Choose per-axis reciprocal cutoffs so `exp(−k²/4α²)/k² < accuracy`.
fn tune_kmax(cell: &Lattice, alpha: f64, accuracy: f64) -> [i32; 3] {
    let target = accuracy.clamp(1e-16, 1e-1);
    // Solve exp(−k²/4α²) = target for k, ignoring the 1/k² prefactor (which
    // only helps): k = 2α √(−ln target).
    let k_cut = 2.0 * alpha * (-target.ln()).sqrt();
    let Some(recip) = cell.reciprocal() else {
        return [1, 1, 1];
    };
    let mut out = [1i32; 3];
    for ax in 0..3 {
        let len = vec3::norm(recip[ax]);
        out[ax] = if len > 1e-12 {
            ((k_cut / len).ceil() as i32).clamp(1, 64)
        } else {
            1
        };
    }
    out
}

/// Cardinal B-spline `M_n(u)`, zero outside `[0, n]`.
///
/// Evaluated by the upward recursion
/// `M_n(u) = u/(n−1) M_{n−1}(u) + (n−u)/(n−1) M_{n−1}(u−1)` from
/// `M_2(u) = 1 − |u − 1|`, carrying the whole shifted row `M_k(u), M_k(u−1), …`
/// so each order costs O(n) rather than the O(2ⁿ) of a naive recursion.
fn bspline_direct(u: f64, n: usize) -> f64 {
    debug_assert!((2..=32).contains(&n));
    if u <= 0.0 || u >= n as f64 {
        return 0.0;
    }
    // row[j] holds M_k(u − j).
    let mut row = [0.0f64; 33];
    for (j, slot) in row.iter_mut().enumerate().take(n) {
        let uj = u - j as f64;
        *slot = if uj > 0.0 && uj < 2.0 {
            1.0 - (uj - 1.0).abs()
        } else {
            0.0
        };
    }
    for k in 3..=n {
        let inv = 1.0 / (k - 1) as f64;
        for j in 0..n {
            let uj = u - j as f64;
            let next = if j + 1 < n { row[j + 1] } else { 0.0 };
            row[j] = (uj * row[j] + (k as f64 - uj) * next) * inv;
        }
    }
    row[0]
}

/// `dM_n/du = M_{n−1}(u) − M_{n−1}(u − 1)`.
fn dbspline(u: f64, n: usize) -> f64 {
    bspline_direct(u, n - 1) - bspline_direct(u - 1.0, n - 1)
}

/// `|b(m)|²` for each mesh index along one axis (Essmann eq. 4.4).
fn bspline_moduli(k: usize, order: usize) -> Vec<f64> {
    // Denominator: Σ_{t=0}^{n−2} M_n(t+1) e^{2πi m t / K}
    let mut out = vec![0.0f64; k];
    for (m, slot) in out.iter_mut().enumerate() {
        let mut re = 0.0;
        let mut im = 0.0;
        for t in 0..=(order - 2) {
            let w = bspline_direct(t as f64 + 1.0, order);
            let ang = std::f64::consts::TAU * (m * t) as f64 / k as f64;
            re += w * ang.cos();
            im += w * ang.sin();
        }
        let den = re * re + im * im;
        // |numerator| = 1 (it is a pure phase), so |b|² = 1/|den|.
        *slot = if den > 1e-30 { 1.0 / den } else { 0.0 };
    }
    out
}

/// Map a mesh index to its signed reciprocal index in `(−K/2, K/2]`.
#[inline]
fn signed_index(i: usize, k: usize) -> i64 {
    if i * 2 <= k {
        i as i64
    } else {
        i as i64 - k as i64
    }
}

#[inline]
fn wrap_index(i: i64, k: usize) -> usize {
    i.rem_euclid(k as i64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rocksalt NaCl: a conventional cubic cell of 8 ions, alternating charge.
    /// `a` is the conventional lattice constant; nearest-neighbour distance is
    /// `a/2`.
    fn nacl(a: f64) -> (Vec<f64>, Vec<[f64; 3]>, Lattice) {
        let mut charges = Vec::new();
        let mut positions = Vec::new();
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    positions.push([i as f64 * a / 2.0, j as f64 * a / 2.0, k as f64 * a / 2.0]);
                    charges.push(if (i + j + k) % 2 == 0 { 1.0 } else { -1.0 });
                }
            }
        }
        (charges, positions, Lattice::cubic(a))
    }

    /// The Madelung constant implied by a total electrostatic energy.
    ///
    /// The Madelung energy per ion is `−M ke q²/r₀`, and the total lattice
    /// energy counts each pair once, so `U = −(N/2) M ke q²/r₀`.
    fn madelung_from_energy(energy: f64, n: usize, r_nn: f64) -> f64 {
        -energy * r_nn / (KE_COULOMB * n as f64 / 2.0)
    }

    /// The exact NaCl (rocksalt) Madelung constant.
    const MADELUNG_NACL: f64 = 1.747_564_594_633_1;

    #[test]
    fn ewald_reproduces_the_nacl_madelung_constant() {
        let a = 5.64; // Å, the real NaCl lattice constant
        let (charges, positions, cell) = nacl(a);
        let ewald = Ewald::tuned(&cell, a / 2.0 - 1e-9, 1e-12);
        let c = ewald.compute(&charges, &positions, Some(&cell), None, &[]);
        let m = madelung_from_energy(c.energy, charges.len(), a / 2.0);
        assert!(
            (m - MADELUNG_NACL).abs() < 1e-6,
            "Ewald Madelung constant = {m}, want {MADELUNG_NACL}"
        );
    }

    #[test]
    fn ewald_madelung_is_independent_of_lattice_constant() {
        // The Madelung constant is dimensionless: the same number must come
        // back at any lattice spacing.
        for &a in &[4.0, 5.64, 8.0] {
            let (charges, positions, cell) = nacl(a);
            let ewald = Ewald::tuned(&cell, a / 2.0 - 1e-9, 1e-12);
            let c = ewald.compute(&charges, &positions, Some(&cell), None, &[]);
            let m = madelung_from_energy(c.energy, charges.len(), a / 2.0);
            assert!((m - MADELUNG_NACL).abs() < 1e-6, "a = {a} gave M = {m}");
        }
    }

    #[test]
    fn ewald_madelung_is_independent_of_the_splitting_parameter() {
        // α is a purely computational split; the total must not depend on it.
        let a = 5.64;
        let (charges, positions, cell) = nacl(a);
        let mut seen = Vec::new();
        // α must be large enough that erfc(α·r_cut) is negligible at the
        // 2.82 Å cutoff a 5.64 Å cell allows, and k_max large enough to
        // converge the reciprocal sum at the largest α.
        for &alpha in &[1.4, 1.6, 1.8] {
            let ewald = Ewald::new(alpha, a / 2.0 - 1e-9, [24, 24, 24]);
            let c = ewald.compute(&charges, &positions, Some(&cell), None, &[]);
            seen.push(madelung_from_energy(c.energy, charges.len(), a / 2.0));
        }
        for m in &seen {
            assert!((m - MADELUNG_NACL).abs() < 1e-5, "got {seen:?}");
        }
    }

    #[test]
    fn pme_reproduces_the_nacl_madelung_constant() {
        let a = 5.64;
        let (charges, positions, cell) = nacl(a);
        let pme = Pme::new(
            tune_alpha(a / 2.0 - 1e-9, 1e-8),
            a / 2.0 - 1e-9,
            [32, 32, 32],
            8,
        );
        let c = pme.compute(&charges, &positions, Some(&cell), None, &[]);
        let m = madelung_from_energy(c.energy, charges.len(), a / 2.0);
        assert!(
            (m - MADELUNG_NACL).abs() < 1e-4,
            "PME Madelung constant = {m}, want {MADELUNG_NACL}"
        );
    }

    #[test]
    fn pme_matches_ewald_on_a_disordered_system() {
        let (charges, positions, cell) = disordered(24, 12.0);
        let r_cut = 5.0;
        let alpha = tune_alpha(r_cut, 1e-8);
        let ewald = Ewald::new(alpha, r_cut, tune_kmax(&cell, alpha, 1e-10));
        let pme = Pme::new(alpha, r_cut, [32, 32, 32], 8);

        let a = ewald.compute(&charges, &positions, Some(&cell), None, &[]);
        let b = pme.compute(&charges, &positions, Some(&cell), None, &[]);
        let rel = (a.energy - b.energy).abs() / a.energy.abs().max(1e-12);
        assert!(rel < 1e-5, "PME {} vs Ewald {}", b.energy, a.energy);

        let fmax = a
            .forces
            .iter()
            .flat_map(|f| f.iter())
            .fold(0.0f64, |m, v| m.max(v.abs()));
        for (fa, fb) in a.forces.iter().zip(&b.forces) {
            for k in 0..3 {
                assert!(
                    (fa[k] - fb[k]).abs() < 1e-4 * fmax.max(1.0),
                    "force mismatch {fa:?} vs {fb:?}"
                );
            }
        }
    }

    /// A charge-neutral disordered ionic configuration in a cubic box.
    fn disordered(n_pairs: usize, l: f64) -> (Vec<f64>, Vec<[f64; 3]>, Lattice) {
        let mut charges = Vec::new();
        let mut positions = Vec::new();
        // Deterministic low-discrepancy placement — no RNG dependency.
        let (g1, g2, g3) = (
            0.819_172_513_396_164,
            0.671_043_606_703_25,
            0.549_700_477_901_74,
        );
        for i in 0..2 * n_pairs {
            let t = i as f64 + 1.0;
            positions.push([
                (t * g1).fract() * l,
                (t * g2).fract() * l,
                (t * g3).fract() * l,
            ]);
            charges.push(if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        (charges, positions, Lattice::cubic(l))
    }

    fn check_forces<F>(positions: &[[f64; 3]], mut eval: F, tol: f64)
    where
        F: FnMut(&[[f64; 3]]) -> Contribution,
    {
        let h = 1e-6;
        let forces = eval(positions).forces;
        for i in 0..positions.len().min(6) {
            for a in 0..3 {
                let mut plus = positions.to_vec();
                let mut minus = positions.to_vec();
                plus[i][a] += h;
                minus[i][a] -= h;
                let fd = -(eval(&plus).energy - eval(&minus).energy) / (2.0 * h);
                assert!(
                    (fd - forces[i][a]).abs() < tol,
                    "atom {i} axis {a}: analytic {} vs fd {fd}",
                    forces[i][a]
                );
            }
        }
    }

    #[test]
    fn ewald_forces_match_the_energy_gradient() {
        let (charges, positions, cell) = disordered(8, 12.0);
        let ewald = Ewald::tuned(&cell, 5.0, 1e-10);
        check_forces(
            &positions,
            |p| ewald.compute(&charges, p, Some(&cell), None, &[]),
            1e-5,
        );
    }

    #[test]
    fn pme_forces_match_the_energy_gradient() {
        let (charges, positions, cell) = disordered(8, 12.0);
        let pme = Pme::new(tune_alpha(5.0, 1e-8), 5.0, [32, 32, 32], 8);
        check_forces(
            &positions,
            |p| pme.compute(&charges, p, Some(&cell), None, &[]),
            1e-4,
        );
    }

    /// The virial must reproduce `−dE/dV` under uniform scaling. This is the
    /// only check that exercises the reciprocal-space virial, which has no
    /// simpler closed form.
    fn check_virial<F>(positions: &[[f64; 3]], cell: &Lattice, mut eval: F, tol: f64)
    where
        F: FnMut(&[[f64; 3]], &Lattice) -> Contribution,
    {
        let base = eval(positions, cell);
        let v = cell.volume().abs();
        let h = 1e-5;
        let scale = |lam: f64| -> (Vec<[f64; 3]>, Lattice) {
            let p = positions.iter().map(|r| vec3::scale(*r, lam)).collect();
            let c = Lattice {
                a: vec3::scale(cell.a, lam),
                b: vec3::scale(cell.b, lam),
                c: vec3::scale(cell.c, lam),
                periodic: cell.periodic,
            };
            (p, c)
        };
        let (pp, cp) = scale(1.0 + h);
        let (pm, cm) = scale(1.0 - h);
        let ep = eval(&pp, &cp).energy;
        let em = eval(&pm, &cm).energy;
        let vp = cp.volume().abs();
        let vm = cm.volume().abs();
        let de_dv = (ep - em) / (vp - vm);
        let p_virial = base.scalar_virial() / (3.0 * v);
        assert!(
            (p_virial + de_dv).abs() < tol * p_virial.abs().max(1e-6),
            "virial pressure {p_virial} vs −dE/dV {}",
            -de_dv
        );
    }

    #[test]
    fn ewald_virial_matches_minus_de_dv() {
        let (charges, positions, cell) = disordered(16, 12.0);
        let alpha = tune_alpha(5.0, 1e-10);
        let kmax = tune_kmax(&cell, alpha, 1e-12);
        check_virial(
            &positions,
            &cell,
            |p, c| {
                // α and k_max fixed across the perturbation: the split is a
                // computational device, not a physical degree of freedom.
                Ewald::new(alpha, 5.0, kmax).compute(&charges, p, Some(c), None, &[])
            },
            1e-3,
        );
    }

    #[test]
    fn pme_virial_matches_minus_de_dv() {
        let (charges, positions, cell) = disordered(16, 12.0);
        let alpha = tune_alpha(5.0, 1e-8);
        check_virial(
            &positions,
            &cell,
            |p, c| Pme::new(alpha, 5.0, [32, 32, 32], 8).compute(&charges, p, Some(c), None, &[]),
            1e-3,
        );
    }

    #[test]
    fn exclusion_correction_cancels_the_pair_completely() {
        // Two charges in a big box, excluded from each other: what remains
        // must be only the (α-independent) self and image terms, so the energy
        // must not depend on how close they are.
        let cell = Lattice::cubic(40.0);
        let charges = [1.0, -1.0];
        let alpha = tune_alpha(12.0, 1e-10);
        let ewald = Ewald::new(alpha, 12.0, tune_kmax(&cell, alpha, 1e-12));
        let near = ewald.compute(
            &charges,
            &[[0.0; 3], [1.5, 0.0, 0.0]],
            Some(&cell),
            Some(&[]),
            &[(0, 1)],
        );
        let far = ewald.compute(
            &charges,
            &[[0.0; 3], [3.0, 0.0, 0.0]],
            Some(&cell),
            Some(&[]),
            &[(0, 1)],
        );
        // Residual is the periodic image interaction only — small in a 40 Å box
        // and slowly varying, so the two energies agree closely.
        assert!(
            (near.energy - far.energy).abs() < 5e-3,
            "{} vs {}",
            near.energy,
            far.energy
        );
    }

    #[test]
    fn bspline_partitions_unity() {
        // Σ_t M_n(w + t) = 1 for any fractional offset — the property that
        // makes charge spreading conserve total charge.
        for n in [4usize, 6, 8] {
            for i in 0..20 {
                let w = i as f64 / 20.0;
                let s: f64 = (0..n).map(|t| bspline_direct(w + t as f64, n)).sum();
                assert!((s - 1.0).abs() < 1e-12, "order {n} offset {w}: {s}");
            }
        }
    }

    #[test]
    fn bspline_derivative_matches_finite_difference() {
        let h = 1e-7;
        for n in [4usize, 6] {
            for i in 1..60 {
                let u = i as f64 * 0.1;
                let fd = (bspline_direct(u + h, n) - bspline_direct(u - h, n)) / (2.0 * h);
                assert!((fd - dbspline(u, n)).abs() < 1e-5, "n={n} u={u}");
            }
        }
    }

    #[test]
    fn cutoff_coulomb_gets_the_madelung_constant_badly_wrong() {
        // The motivation for all of the above: a bare cutoff sum is not a poor
        // approximation to the lattice energy, it is a different number.
        let a = 5.64;
        let (charges, positions, cell) = nacl(a);
        let cut = super::super::potentials::Coulomb {
            cutoff: a / 2.0 - 1e-9,
        };
        let c = cut.compute_all(&charges, &positions, Some(&cell));
        let m = madelung_from_energy(c.energy, charges.len(), a / 2.0);
        assert!(
            (m - MADELUNG_NACL).abs() > 0.1,
            "cutoff Coulomb accidentally agreed: {m}"
        );
    }
}
