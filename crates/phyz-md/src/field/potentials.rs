//! Whole-system interatomic potentials over structure-of-arrays slices.
//!
//! Each potential exposes two entry points:
//!
//! - `compute_pairs(.., pairs, ..)` — evaluate over a precomputed neighbor list
//!   (see [`super::neighbor::NeighborList`]), returning a
//!   [`Contribution`] with energy, forces, and virial. This is the path an MD
//!   run should take: O(N) per step.
//! - `compute(..)` — the self-contained all-pairs form, for one-off evaluations
//!   and as the reference the neighbor-list path is tested against.
//!
//! Energies are eV, forces eV/Å, virials eV. Periodic boundaries use the
//! minimum-image convention from [`super::cell::min_image`].

use std::collections::HashMap;

use super::cell::{Lattice, min_image, vec3};
use super::units::KE_COULOMB;
use super::virial::Contribution;

/// Chunk size for the parallel pair loop. Large enough that per-chunk
/// accumulator allocation is amortized, small enough to keep threads fed.
#[cfg(feature = "parallel")]
const PAIR_CHUNK: usize = 4096;

/// Accumulate a pairwise kernel over a neighbor list.
///
/// The kernel receives the pair and the accumulator it should write into. With
/// the `parallel` feature the pair list is chunked across rayon's pool, each
/// chunk folding into its own force array before a final reduction — a
/// half-neighbor-list force loop writes to both `i` and `j`, so per-thread
/// accumulators are what make it safely parallel without atomics.
///
/// The reduction order is not fixed, so summation rounding can differ between
/// runs at the 1-ulp level and a long chaotic trajectory will diverge from a
/// bit-identical repeat. This is the usual trade-off for parallel MD (LAMMPS
/// and GROMACS make it too); build without the `parallel` feature when
/// bit-for-bit reproducibility matters more than throughput.
fn accumulate_pairs<K>(n: usize, pairs: &[(usize, usize)], kernel: K) -> Contribution
where
    K: Fn(usize, usize, &mut Contribution) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        if pairs.len() >= PAIR_CHUNK {
            return pairs
                .par_chunks(PAIR_CHUNK)
                .fold(
                    || Contribution::zeros(n),
                    |mut acc, chunk| {
                        for &(i, j) in chunk {
                            kernel(i, j, &mut acc);
                        }
                        acc
                    },
                )
                .reduce(
                    || Contribution::zeros(n),
                    |mut a, b| {
                        a.merge(&b);
                        a
                    },
                );
        }
    }
    let mut acc = Contribution::zeros(n);
    for &(i, j) in pairs {
        kernel(i, j, &mut acc);
    }
    acc
}

/// Every `(i, j)` pair with `i < j` — the all-pairs fallback list.
fn all_pairs(n: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(n * n / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            v.push((i, j));
        }
    }
    v
}

/// Lennard-Jones potential with per-species parameters and Lorentz-Berthelot
/// mixing, a spherical cutoff, and optional energy shift for continuity.
#[derive(Debug, Clone)]
pub struct LennardJones {
    /// Per-species `(epsilon eV, sigma Å)` keyed by species id (e.g. atomic
    /// number).
    pub params: HashMap<u32, (f64, f64)>,
    /// Fallback parameters for species with no entry.
    pub default: (f64, f64),
    /// Cutoff radius in Å.
    pub cutoff: f64,
    /// Shift the potential so `E(cutoff) == 0`.
    pub shift: bool,
}

impl LennardJones {
    /// A single-species LJ fluid (e.g. argon: eps=0.0103 eV, sigma=3.4 Å).
    pub fn monatomic(epsilon: f64, sigma: f64, cutoff: f64) -> Self {
        Self {
            params: HashMap::new(),
            default: (epsilon, sigma),
            cutoff,
            shift: true,
        }
    }

    /// Argon: ε = 0.0103 eV, σ = 3.4 Å, cutoff 2.5σ.
    pub fn argon() -> Self {
        Self::monatomic(0.0103, 3.4, 2.5 * 3.4)
    }

    #[inline]
    fn pair_params(&self, zi: u32, zj: u32) -> (f64, f64) {
        let (ei, si) = self.params.get(&zi).copied().unwrap_or(self.default);
        let (ej, sj) = self.params.get(&zj).copied().unwrap_or(self.default);
        // Lorentz-Berthelot: eps = sqrt(ei ej), sigma = (si+sj)/2.
        ((ei * ej).sqrt(), 0.5 * (si + sj))
    }

    /// Energy, forces, and virial over a precomputed neighbor list.
    pub fn compute_pairs(
        &self,
        numbers: &[u32],
        positions: &[[f64; 3]],
        pairs: &[(usize, usize)],
        cell: Option<&Lattice>,
    ) -> Contribution {
        let rc2 = self.cutoff * self.cutoff;
        accumulate_pairs(positions.len(), pairs, |i, j, acc| {
            let d = min_image(vec3::sub(positions[i], positions[j]), cell);
            let r2 = vec3::norm2(d);
            if r2 > rc2 || r2 < 1e-12 {
                return;
            }
            let (eps, sigma) = self.pair_params(numbers[i], numbers[j]);
            let inv_r2 = 1.0 / r2;
            let s2 = sigma * sigma * inv_r2;
            let s6 = s2 * s2 * s2;
            let s12 = s6 * s6;
            let mut e = 4.0 * eps * (s12 - s6);
            if self.shift {
                let sc2 = sigma * sigma / rc2;
                let sc6 = sc2 * sc2 * sc2;
                e -= 4.0 * eps * (sc6 * sc6 - sc6);
            }
            acc.energy += e;
            // E = 4 eps (s12 - s6), s6 = sigma^6 (r2)^-3.
            // dE/dr2 = 4 eps (-6 s12 + 3 s6) / r2.
            // F_i = -dE/dr_i = -dE/dr2 * d(r2)/dr_i = -dE/dr2 * 2 d.
            let de_dr2 = 4.0 * eps * (-6.0 * s12 + 3.0 * s6) * inv_r2;
            let f = vec3::scale(d, -2.0 * de_dr2);
            acc.add_pair_force(i, j, d, f);
        })
    }

    /// All-pairs energy and forces for the given species ids and positions.
    ///
    /// Retained as the reference form and for callers with no neighbor list;
    /// prefer [`Self::compute_pairs`] inside a time-stepping loop.
    pub fn compute(
        &self,
        numbers: &[u32],
        positions: &[[f64; 3]],
        cell: Option<&Lattice>,
    ) -> (f64, Vec<[f64; 3]>) {
        let c = self.compute_all(numbers, positions, cell);
        (c.energy, c.forces)
    }

    /// All-pairs form returning the full contribution including the virial.
    pub fn compute_all(
        &self,
        numbers: &[u32],
        positions: &[[f64; 3]],
        cell: Option<&Lattice>,
    ) -> Contribution {
        let pairs = all_pairs(positions.len());
        self.compute_pairs(numbers, positions, &pairs, cell)
    }

    /// Long-range tail correction to the energy (eV) for a homogeneous fluid
    /// of `n` atoms of the default species in volume `v` (Å³).
    ///
    /// `E_tail = (8/3) π N ρ ε σ³ [(1/3)(σ/rc)⁹ - (σ/rc)³]`. Only meaningful
    /// for a single-species, unshifted, isotropic system.
    pub fn tail_energy(&self, n: usize, volume: f64) -> f64 {
        if volume <= 0.0 || n == 0 {
            return 0.0;
        }
        let (eps, sigma) = self.default;
        let rho = n as f64 / volume;
        let s3 = (sigma / self.cutoff).powi(3);
        (8.0 / 3.0)
            * std::f64::consts::PI
            * n as f64
            * rho
            * eps
            * sigma.powi(3)
            * (s3 * s3 * s3 / 3.0 - s3)
    }

    /// Long-range tail correction to the pressure (eV/Å³) for the same
    /// homogeneous-fluid assumption as [`Self::tail_energy`].
    pub fn tail_pressure(&self, n: usize, volume: f64) -> f64 {
        if volume <= 0.0 || n == 0 {
            return 0.0;
        }
        let (eps, sigma) = self.default;
        let rho = n as f64 / volume;
        let s3 = (sigma / self.cutoff).powi(3);
        (16.0 / 3.0)
            * std::f64::consts::PI
            * rho
            * rho
            * eps
            * sigma.powi(3)
            * (2.0 * s3 * s3 * s3 / 3.0 - s3)
    }
}

/// Harmonic bond stretching: `E = 0.5 k (r - r0)²` per bond.
#[derive(Debug, Clone)]
pub struct HarmonicBonds {
    /// Force constant in eV/Å².
    pub k: f64,
    /// Equilibrium length in Å (uniform if `per_bond` is empty).
    pub r0: f64,
    /// Optional per-bond equilibrium lengths, matching the bond-list order.
    pub per_bond: Vec<f64>,
    /// Optional per-bond force constants, matching the bond-list order.
    pub per_bond_k: Vec<f64>,
}

impl HarmonicBonds {
    /// Uniform harmonic bonds.
    pub fn uniform(k: f64, r0: f64) -> Self {
        Self {
            k,
            r0,
            per_bond: Vec::new(),
            per_bond_k: Vec::new(),
        }
    }

    /// Energy, forces, and virial for the given `(i, j)` bond pairs.
    pub fn compute_all(
        &self,
        bonds: &[(usize, usize)],
        positions: &[[f64; 3]],
        cell: Option<&Lattice>,
    ) -> Contribution {
        let mut acc = Contribution::zeros(positions.len());
        for (bi, &(i, j)) in bonds.iter().enumerate() {
            let r0 = self.per_bond.get(bi).copied().unwrap_or(self.r0);
            let k = self.per_bond_k.get(bi).copied().unwrap_or(self.k);
            let d = min_image(vec3::sub(positions[i], positions[j]), cell);
            let r = vec3::norm(d);
            if r < 1e-12 {
                continue;
            }
            let dr = r - r0;
            acc.energy += 0.5 * k * dr * dr;
            // F_i = -k (r - r0) * rhat
            let f = vec3::scale(d, -k * dr / r);
            acc.add_pair_force(i, j, d, f);
        }
        acc
    }

    /// Total energy and per-atom forces for the given `(i, j)` bond pairs.
    pub fn compute(
        &self,
        bonds: &[(usize, usize)],
        positions: &[[f64; 3]],
        cell: Option<&Lattice>,
    ) -> (f64, Vec<[f64; 3]>) {
        let c = self.compute_all(bonds, positions, cell);
        (c.energy, c.forces)
    }
}

/// Harmonic angle bending: `E = 0.5 k (θ - θ0)²` over `(i, j, k)` triples with
/// `j` the apex.
#[derive(Debug, Clone, Default)]
pub struct HarmonicAngles {
    /// `(i, apex, k)` atom-index triples.
    pub triples: Vec<(usize, usize, usize)>,
    /// Force constant in eV/rad².
    pub k: f64,
    /// Equilibrium angle in radians.
    pub theta0: f64,
    /// Optional per-triple `(k, theta0)` overrides, matching `triples` order.
    pub per_angle: Vec<(f64, f64)>,
}

impl HarmonicAngles {
    /// Uniform harmonic angles over the given triples.
    pub fn uniform(triples: Vec<(usize, usize, usize)>, k: f64, theta0: f64) -> Self {
        Self {
            triples,
            k,
            theta0,
            per_angle: Vec::new(),
        }
    }

    /// Energy, forces, and virial over the stored triples.
    pub fn compute_all(&self, positions: &[[f64; 3]], cell: Option<&Lattice>) -> Contribution {
        let mut acc = Contribution::zeros(positions.len());
        for (ai, &(ia, ja, ka)) in self.triples.iter().enumerate() {
            let (k, theta0) = self
                .per_angle
                .get(ai)
                .copied()
                .unwrap_or((self.k, self.theta0));
            let rij = min_image(vec3::sub(positions[ia], positions[ja]), cell);
            let rkj = min_image(vec3::sub(positions[ka], positions[ja]), cell);
            let lij = vec3::norm(rij);
            let lkj = vec3::norm(rkj);
            if lij < 1e-9 || lkj < 1e-9 {
                continue;
            }
            let cos_t = (vec3::dot(rij, rkj) / (lij * lkj)).clamp(-1.0, 1.0);
            let theta = cos_t.acos();
            let dtheta = theta - theta0;
            acc.energy += 0.5 * k * dtheta * dtheta;
            let sin_t = (1.0 - cos_t * cos_t).sqrt().max(1e-9);
            let de_dtheta = k * dtheta;
            // F_i = -dV/dr_i = (dV/dθ / sinθ) · d(cosθ)/dr_i, with
            // d(cosθ)/dr_i = rkj/(lij·lkj) − cosθ·rij/lij².
            let coef = de_dtheta / sin_t;
            let fi = vec3::scale(
                vec3::sub(
                    vec3::scale(rkj, 1.0 / (lij * lkj)),
                    vec3::scale(rij, cos_t / (lij * lij)),
                ),
                coef,
            );
            let fk = vec3::scale(
                vec3::sub(
                    vec3::scale(rij, 1.0 / (lij * lkj)),
                    vec3::scale(rkj, cos_t / (lkj * lkj)),
                ),
                coef,
            );
            vec3::add_assign(&mut acc.forces[ia], fi);
            vec3::add_assign(&mut acc.forces[ka], fk);
            vec3::add_assign(&mut acc.forces[ja], vec3::scale(vec3::add(fi, fk), -1.0));
            // Virial of a three-body term: Σ_legs d ⊗ f, with the apex as the
            // reference point so the result is translation invariant.
            acc.virial_from_pair(rij, fi);
            acc.virial_from_pair(rkj, fk);
        }
        acc
    }

    /// Total energy and per-atom forces over the stored triples.
    pub fn compute(&self, positions: &[[f64; 3]], cell: Option<&Lattice>) -> (f64, Vec<[f64; 3]>) {
        let c = self.compute_all(positions, cell);
        (c.energy, c.forces)
    }
}

/// Direct (cutoff) Coulomb interaction: `E = KE q_i q_j / r`.
///
/// This is the plain real-space sum. It is correct for isolated clusters, but
/// **not** for a periodic charged system — the neglected long-range tail is
/// conditionally convergent, and truncating it gets both energies and structure
/// qualitatively wrong. Use [`super::ewald::Ewald`] or [`super::ewald::Pme`]
/// under periodic boundaries.
#[derive(Debug, Clone)]
pub struct Coulomb {
    /// Cutoff radius in Å.
    pub cutoff: f64,
}

impl Coulomb {
    /// Energy, forces, and virial over a precomputed neighbor list.
    pub fn compute_pairs(
        &self,
        charges: &[f64],
        positions: &[[f64; 3]],
        pairs: &[(usize, usize)],
        cell: Option<&Lattice>,
    ) -> Contribution {
        let rc2 = self.cutoff * self.cutoff;
        accumulate_pairs(positions.len(), pairs, |i, j, acc| {
            if charges[i] == 0.0 || charges[j] == 0.0 {
                return;
            }
            let d = min_image(vec3::sub(positions[i], positions[j]), cell);
            let r2 = vec3::norm2(d);
            if r2 > rc2 || r2 < 1e-12 {
                return;
            }
            let r = r2.sqrt();
            let qq = KE_COULOMB * charges[i] * charges[j];
            acc.energy += qq / r;
            // F_i = qq / r^2 * rhat = qq / r^3 * d
            let f = vec3::scale(d, qq / (r2 * r));
            acc.add_pair_force(i, j, d, f);
        })
    }

    /// All-pairs form returning the full contribution.
    pub fn compute_all(
        &self,
        charges: &[f64],
        positions: &[[f64; 3]],
        cell: Option<&Lattice>,
    ) -> Contribution {
        let pairs = all_pairs(positions.len());
        self.compute_pairs(charges, positions, &pairs, cell)
    }

    /// Total energy and per-atom forces for the given per-atom charges (e).
    pub fn compute(
        &self,
        charges: &[f64],
        positions: &[[f64; 3]],
        cell: Option<&Lattice>,
    ) -> (f64, Vec<[f64; 3]>) {
        let c = self.compute_all(charges, positions, cell);
        (c.energy, c.forces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::neighbor::NeighborList;

    /// Central finite-difference check that forces are minus the energy
    /// gradient — the single most useful invariant a potential can have.
    fn check_forces<F>(positions: &[[f64; 3]], mut eval: F, tol: f64)
    where
        F: FnMut(&[[f64; 3]]) -> (f64, Vec<[f64; 3]>),
    {
        let h = 1e-6;
        let (_, forces) = eval(positions);
        for i in 0..positions.len() {
            for a in 0..3 {
                let mut plus = positions.to_vec();
                let mut minus = positions.to_vec();
                plus[i][a] += h;
                minus[i][a] -= h;
                let (ep, _) = eval(&plus);
                let (em, _) = eval(&minus);
                let fd = -(ep - em) / (2.0 * h);
                assert!(
                    (fd - forces[i][a]).abs() < tol,
                    "atom {i} axis {a}: analytic {} vs finite-difference {fd}",
                    forces[i][a]
                );
            }
        }
    }

    #[test]
    fn lj_dimer_minimum_is_at_two_to_the_sixth_sigma() {
        let lj = LennardJones {
            params: HashMap::new(),
            default: (0.0103, 3.4),
            cutoff: 12.0,
            shift: false,
        };
        let r_min = 2.0f64.powf(1.0 / 6.0) * 3.4;
        let (e, f) = lj.compute(&[18, 18], &[[0.0; 3], [r_min, 0.0, 0.0]], None);
        assert!((e - (-0.0103)).abs() < 1e-9);
        assert!(f[0][0].abs() < 1e-9);
    }

    #[test]
    fn lj_forces_match_the_energy_gradient() {
        let lj = LennardJones::argon();
        let numbers = [18u32; 4];
        let pos = vec![
            [0.0, 0.0, 0.0],
            [3.7, 0.3, -0.2],
            [1.1, 4.0, 0.6],
            [-2.0, 1.5, 3.3],
        ];
        check_forces(&pos, |p| lj.compute(&numbers, p, None), 1e-7);
    }

    #[test]
    fn lj_neighbor_list_path_matches_all_pairs() {
        // Cutoff chosen so a 20 Å box still gives ≥ 3 cells per axis and the
        // binned path (not the fallback) is what gets compared.
        let lj = LennardJones::monatomic(0.0103, 3.4, 6.0);
        let mut pos = Vec::new();
        let mut numbers = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    let jit = ((i * 3 + j * 5 + k * 7) % 9) as f64 * 0.02;
                    pos.push([
                        i as f64 * 4.0 + jit,
                        j as f64 * 4.0 - jit,
                        k as f64 * 4.0 + 0.5 * jit,
                    ]);
                    numbers.push(18u32);
                }
            }
        }
        let cell = Lattice::cubic(20.0);
        let mut nl = NeighborList::new(lj.cutoff, 0.6);
        nl.build(&pos, Some(&cell));
        assert!(!nl.used_fallback());

        let via_list = lj.compute_pairs(&numbers, &pos, nl.pairs(), Some(&cell));
        let via_all = lj.compute_all(&numbers, &pos, Some(&cell));
        assert!((via_list.energy - via_all.energy).abs() < 1e-10);
        for (a, b) in via_list.forces.iter().zip(&via_all.forces) {
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 1e-10);
            }
        }
        assert!((via_list.scalar_virial() - via_all.scalar_virial()).abs() < 1e-10);
    }

    #[test]
    fn coulomb_dimer_energy_matches_closed_form() {
        let c = Coulomb { cutoff: 20.0 };
        let (e, f) = c.compute(&[1.0, -1.0], &[[0.0; 3], [2.0, 0.0, 0.0]], None);
        assert!((e - (-KE_COULOMB / 2.0)).abs() < 1e-12);
        // Opposite charges attract: force on atom 0 points toward atom 1 (+x).
        assert!(f[0][0] > 0.0);
    }

    #[test]
    fn harmonic_bond_energy_and_force() {
        let hb = HarmonicBonds::uniform(100.0, 1.0);
        let (e, f) = hb.compute(&[(0, 1)], &[[0.0; 3], [1.5, 0.0, 0.0]], None);
        assert!((e - 12.5).abs() < 1e-12);
        assert!((f[0][0] - 50.0).abs() < 1e-12);
    }

    #[test]
    fn angle_forces_match_the_energy_gradient() {
        let angles = HarmonicAngles::uniform(vec![(0, 1, 2)], 5.0, 1.9);
        let pos = vec![[1.0, 0.2, 0.0], [0.0, 0.0, 0.0], [-0.3, 0.95, 0.1]];
        check_forces(&pos, |p| angles.compute(p, None), 1e-6);
    }

    #[test]
    fn angle_forces_sum_to_zero() {
        let angles = HarmonicAngles::uniform(vec![(0, 1, 2)], 5.0, 1.9);
        let pos = vec![[1.0, 0.2, 0.0], [0.0, 0.0, 0.0], [-0.3, 0.95, 0.1]];
        let c = angles.compute_all(&pos, None);
        for a in 0..3 {
            let sum: f64 = c.forces.iter().map(|f| f[a]).sum();
            assert!(sum.abs() < 1e-12);
        }
    }
}
