//! Whole-system interatomic potentials over structure-of-arrays slices.
//!
//! Each potential's `compute` returns total energy (eV) and per-atom forces
//! (eV/Å). Periodic boundaries use the minimum-image convention from
//! [`super::cell::min_image`].

use std::collections::HashMap;

use super::cell::{Lattice, min_image, vec3};
use super::units::KE_COULOMB;

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

    #[inline]
    fn pair_params(&self, zi: u32, zj: u32) -> (f64, f64) {
        let (ei, si) = self.params.get(&zi).copied().unwrap_or(self.default);
        let (ej, sj) = self.params.get(&zj).copied().unwrap_or(self.default);
        // Lorentz-Berthelot: eps = sqrt(ei ej), sigma = (si+sj)/2.
        ((ei * ej).sqrt(), 0.5 * (si + sj))
    }

    /// Total energy and per-atom forces for the given species ids and
    /// positions.
    pub fn compute(
        &self,
        numbers: &[u32],
        positions: &[[f64; 3]],
        cell: Option<&Lattice>,
    ) -> (f64, Vec<[f64; 3]>) {
        let n = positions.len();
        let mut forces = vec![[0.0; 3]; n];
        let mut energy = 0.0;
        let rc2 = self.cutoff * self.cutoff;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = min_image(vec3::sub(positions[i], positions[j]), cell);
                let r2 = vec3::norm2(d);
                if r2 > rc2 || r2 < 1e-12 {
                    continue;
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
                energy += e;
                // E = 4 eps (s12 - s6), s6 = sigma^6 (r2)^-3.
                // dE/dr2 = 4 eps (-6 s12 + 3 s6) / r2.
                // F_i = -dE/dr_i = -dE/dr2 * d(r2)/dr_i = -dE/dr2 * 2 d.
                let de_dr2 = 4.0 * eps * (-6.0 * s12 + 3.0 * s6) * inv_r2;
                let fmag = -2.0 * de_dr2; // multiply by d gives force on i
                let f = vec3::scale(d, fmag);
                vec3::add_assign(&mut forces[i], f);
                vec3::add_assign(&mut forces[j], vec3::scale(f, -1.0));
            }
        }
        (energy, forces)
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
}

impl HarmonicBonds {
    /// Uniform harmonic bonds.
    pub fn uniform(k: f64, r0: f64) -> Self {
        Self {
            k,
            r0,
            per_bond: Vec::new(),
        }
    }

    /// Total energy and per-atom forces for the given `(i, j)` bond pairs.
    pub fn compute(
        &self,
        bonds: &[(usize, usize)],
        positions: &[[f64; 3]],
        cell: Option<&Lattice>,
    ) -> (f64, Vec<[f64; 3]>) {
        let mut forces = vec![[0.0; 3]; positions.len()];
        let mut energy = 0.0;
        for (bi, &(i, j)) in bonds.iter().enumerate() {
            let r0 = self.per_bond.get(bi).copied().unwrap_or(self.r0);
            let d = min_image(vec3::sub(positions[i], positions[j]), cell);
            let r = vec3::norm(d);
            if r < 1e-12 {
                continue;
            }
            let dr = r - r0;
            energy += 0.5 * self.k * dr * dr;
            // F_i = -k (r - r0) * rhat
            let fmag = -self.k * dr / r;
            let f = vec3::scale(d, fmag);
            vec3::add_assign(&mut forces[i], f);
            vec3::add_assign(&mut forces[j], vec3::scale(f, -1.0));
        }
        (energy, forces)
    }
}

/// Harmonic angle bending: `E = 0.5 k (θ - θ0)²` over `(i, j, k)` triples with
/// `j` the apex.
#[derive(Debug, Clone)]
pub struct HarmonicAngles {
    /// `(i, apex, k)` atom-index triples.
    pub triples: Vec<(usize, usize, usize)>,
    /// Force constant in eV/rad².
    pub k: f64,
    /// Equilibrium angle in radians.
    pub theta0: f64,
}

impl HarmonicAngles {
    /// Total energy and per-atom forces over the stored triples.
    pub fn compute(&self, positions: &[[f64; 3]], cell: Option<&Lattice>) -> (f64, Vec<[f64; 3]>) {
        let mut forces = vec![[0.0; 3]; positions.len()];
        let mut energy = 0.0;
        for &(ia, ja, ka) in &self.triples {
            let rij = min_image(vec3::sub(positions[ia], positions[ja]), cell);
            let rkj = min_image(vec3::sub(positions[ka], positions[ja]), cell);
            let lij = vec3::norm(rij);
            let lkj = vec3::norm(rkj);
            if lij < 1e-9 || lkj < 1e-9 {
                continue;
            }
            let cos_t = (vec3::dot(rij, rkj) / (lij * lkj)).clamp(-1.0, 1.0);
            let theta = cos_t.acos();
            let dtheta = theta - self.theta0;
            energy += 0.5 * self.k * dtheta * dtheta;
            let sin_t = (1.0 - cos_t * cos_t).sqrt().max(1e-9);
            let de_dtheta = self.k * dtheta;
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
            vec3::add_assign(&mut forces[ia], fi);
            vec3::add_assign(&mut forces[ka], fk);
            vec3::add_assign(&mut forces[ja], vec3::scale(vec3::add(fi, fk), -1.0));
        }
        (energy, forces)
    }
}

/// Direct (cutoff) Coulomb interaction: `E = KE q_i q_j / r`.
///
/// This is the simple real-space sum, not Ewald — adequate for clusters and
/// short-range screening, not for long-range lattice sums.
#[derive(Debug, Clone)]
pub struct Coulomb {
    /// Cutoff radius in Å.
    pub cutoff: f64,
}

impl Coulomb {
    /// Total energy and per-atom forces for the given per-atom charges (e).
    pub fn compute(
        &self,
        charges: &[f64],
        positions: &[[f64; 3]],
        cell: Option<&Lattice>,
    ) -> (f64, Vec<[f64; 3]>) {
        let n = positions.len();
        let mut forces = vec![[0.0; 3]; n];
        let mut energy = 0.0;
        let rc2 = self.cutoff * self.cutoff;
        for i in 0..n {
            if charges[i] == 0.0 {
                continue;
            }
            for j in (i + 1)..n {
                if charges[j] == 0.0 {
                    continue;
                }
                let d = min_image(vec3::sub(positions[i], positions[j]), cell);
                let r2 = vec3::norm2(d);
                if r2 > rc2 || r2 < 1e-12 {
                    continue;
                }
                let r = r2.sqrt();
                let qq = KE_COULOMB * charges[i] * charges[j];
                energy += qq / r;
                // F_i = qq / r^2 * rhat = qq / r^3 * d
                let f = vec3::scale(d, qq / (r2 * r));
                vec3::add_assign(&mut forces[i], f);
                vec3::add_assign(&mut forces[j], vec3::scale(f, -1.0));
            }
        }
        (energy, forces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
