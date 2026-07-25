//! Energy / force / virial bundles and the pressure they feed.
//!
//! Every potential in this module reports a [`Contribution`]: the energy it
//! adds (eV), the force it puts on each atom (eV/Å), and its virial tensor
//! (eV). The virial is what makes pressure — and therefore any constant-
//! pressure ensemble — computable at all.
//!
//! The convention is `W_ab = Σ_pairs d_a f_b` where `d = r_i - r_j` and `f` is
//! the force on `i`, which gives the standard result
//!
//! ```text
//! P = (2 K + Tr W) / (3 V)
//! ```
//!
//! with `K` the kinetic energy. [`Contribution::virial_from_pair`] applies it
//! for a pairwise term; many-body terms (angles, torsions, the Ewald
//! reciprocal sum) accumulate their own analytic forms.

use super::cell::vec3;
use super::units::EV_PER_A3_TO_GPA;

/// The energy, forces, and virial produced by one interaction term.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Contribution {
    /// Potential energy in eV.
    pub energy: f64,
    /// Force on each atom in eV/Å.
    pub forces: Vec<[f64; 3]>,
    /// Virial tensor in eV.
    pub virial: [[f64; 3]; 3],
}

impl Contribution {
    /// A zeroed contribution sized for `n` atoms.
    pub fn zeros(n: usize) -> Self {
        Self {
            energy: 0.0,
            forces: vec![[0.0; 3]; n],
            virial: [[0.0; 3]; 3],
        }
    }

    /// Accumulate `other` into `self`.
    ///
    /// Force arrays must be the same length; a shorter one is treated as
    /// zero-padded so terms that touch only a subset can be merged.
    pub fn merge(&mut self, other: &Self) {
        self.energy += other.energy;
        if self.forces.len() < other.forces.len() {
            self.forces.resize(other.forces.len(), [0.0; 3]);
        }
        for (dst, src) in self.forces.iter_mut().zip(&other.forces) {
            vec3::add_assign(dst, *src);
        }
        for a in 0..3 {
            for b in 0..3 {
                self.virial[a][b] += other.virial[a][b];
            }
        }
    }

    /// Apply a pairwise force `f` on atom `i` (and `-f` on `j`) separated by
    /// the minimum-image displacement `d = r_i - r_j`, accumulating both the
    /// forces and the pair's virial contribution.
    #[inline]
    pub fn add_pair_force(&mut self, i: usize, j: usize, d: [f64; 3], f: [f64; 3]) {
        vec3::add_assign(&mut self.forces[i], f);
        vec3::add_assign(&mut self.forces[j], vec3::scale(f, -1.0));
        self.virial_from_pair(d, f);
    }

    /// Accumulate only the virial of a pair with displacement `d` and force `f`
    /// on the first atom.
    #[inline]
    pub fn virial_from_pair(&mut self, d: [f64; 3], f: [f64; 3]) {
        for (a, row) in self.virial.iter_mut().enumerate() {
            for (b, w) in row.iter_mut().enumerate() {
                *w += d[a] * f[b];
            }
        }
    }

    /// `Tr W`, the scalar virial `Σ d · f`.
    #[inline]
    pub fn scalar_virial(&self) -> f64 {
        self.virial[0][0] + self.virial[1][1] + self.virial[2][2]
    }
}

/// Instantaneous pressure in eV/Å³ from the kinetic energy (eV), scalar virial
/// (eV), and cell volume (Å³).
///
/// `P = (2K + Tr W) / 3V`.
#[inline]
pub fn pressure(kinetic_energy: f64, scalar_virial: f64, volume: f64) -> f64 {
    if volume <= 0.0 {
        return 0.0;
    }
    (2.0 * kinetic_energy + scalar_virial) / (3.0 * volume)
}

/// Full pressure tensor in eV/Å³ from the kinetic tensor and the virial.
///
/// `P_ab = (Σ_i m_i v_ia v_ib / FORCE_TO_ACCEL + W_ab) / V`.
pub fn pressure_tensor(
    velocities: &[[f64; 3]],
    masses: &[f64],
    virial: &[[f64; 3]; 3],
    volume: f64,
) -> [[f64; 3]; 3] {
    let mut p = [[0.0; 3]; 3];
    if volume <= 0.0 {
        return p;
    }
    for (v, &m) in velocities.iter().zip(masses) {
        for a in 0..3 {
            for b in 0..3 {
                p[a][b] += m * v[a] * v[b] / super::units::FORCE_TO_ACCEL;
            }
        }
    }
    for a in 0..3 {
        for b in 0..3 {
            p[a][b] = (p[a][b] + virial[a][b]) / volume;
        }
    }
    p
}

/// Convert a pressure in eV/Å³ to GPa.
#[inline]
pub fn to_gpa(p_ev_per_a3: f64) -> f64 {
    p_ev_per_a3 * EV_PER_A3_TO_GPA
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_virial_is_symmetric_for_central_forces() {
        let mut c = Contribution::zeros(2);
        let d = [1.5, -0.5, 0.25];
        // A central force is parallel to d, so d ⊗ f is symmetric.
        let f = vec3::scale(d, 2.3);
        c.add_pair_force(0, 1, d, f);
        for a in 0..3 {
            for b in 0..3 {
                assert!((c.virial[a][b] - c.virial[b][a]).abs() < 1e-12);
            }
        }
        assert!((c.scalar_virial() - 2.3 * vec3::norm2(d)).abs() < 1e-12);
        assert!((c.forces[0][0] + c.forces[1][0]).abs() < 1e-12);
    }

    #[test]
    fn ideal_gas_pressure_recovers_nkt_over_v() {
        // With no interactions, P V = 2K/3 = N k T.
        let volume = 1000.0;
        let kinetic = 12.0;
        assert!((pressure(kinetic, 0.0, volume) - 2.0 * kinetic / (3.0 * volume)).abs() < 1e-15);
    }

    #[test]
    fn merge_sums_all_three_channels() {
        let mut a = Contribution::zeros(2);
        a.energy = 1.0;
        a.forces[0] = [1.0, 0.0, 0.0];
        a.virial[0][0] = 2.0;
        let mut b = Contribution::zeros(2);
        b.energy = 3.0;
        b.forces[0] = [0.0, 2.0, 0.0];
        b.virial[0][0] = 5.0;
        a.merge(&b);
        assert_eq!(a.energy, 4.0);
        assert_eq!(a.forces[0], [1.0, 2.0, 0.0]);
        assert_eq!(a.virial[0][0], 7.0);
    }
}
