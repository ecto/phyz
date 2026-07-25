//! Torsional terms: proper dihedrals and improper (out-of-plane) dihedrals.
//!
//! Bonds and angles alone cannot express a biomolecular force field — AMBER,
//! CHARMM, and OPLS all place their conformational preferences (backbone
//! φ/ψ, sugar puckers, sp² planarity) in torsion terms. This module supplies
//! both flavours:
//!
//! - [`PeriodicDihedrals`] — `E = Σ k (1 + cos(n φ − δ))`, the AMBER/CHARMM/
//!   OPLS proper-torsion form. Multiple terms on the same quartet are just
//!   multiple entries.
//! - [`HarmonicImpropers`] — `E = ½ k (φ − φ₀)²`, the CHARMM improper form used
//!   to keep planar groups planar and preserve chirality.
//!
//! Both share the torsion-angle geometry and its gradient.

use super::cell::{Lattice, min_image, vec3};
use super::virial::Contribution;

/// A proper torsion term over the quartet `(i, j, k, l)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DihedralTerm {
    /// Atom indices, bonded i–j–k–l.
    pub atoms: (usize, usize, usize, usize),
    /// Barrier height in eV.
    pub k: f64,
    /// Multiplicity (number of minima per turn).
    pub n: u32,
    /// Phase offset in radians.
    pub delta: f64,
}

/// An improper torsion term over the quartet `(i, j, k, l)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImproperTerm {
    /// Atom indices; the torsion is measured about the j–k axis.
    pub atoms: (usize, usize, usize, usize),
    /// Force constant in eV/rad².
    pub k: f64,
    /// Equilibrium angle in radians (0 or π for planar groups).
    pub phi0: f64,
}

/// Proper dihedrals with the periodic (cosine-series) functional form.
#[derive(Debug, Clone, Default)]
pub struct PeriodicDihedrals {
    /// The torsion terms. Several terms may share a quartet to build up a
    /// Fourier series.
    pub terms: Vec<DihedralTerm>,
}

impl PeriodicDihedrals {
    /// Dihedrals from a list of terms.
    pub fn new(terms: Vec<DihedralTerm>) -> Self {
        Self { terms }
    }

    /// Energy, forces, and virial for all terms.
    pub fn compute_all(&self, positions: &[[f64; 3]], cell: Option<&Lattice>) -> Contribution {
        let mut acc = Contribution::zeros(positions.len());
        for t in &self.terms {
            let Some(g) = TorsionGeometry::new(t.atoms, positions, cell) else {
                continue;
            };
            let arg = t.n as f64 * g.phi - t.delta;
            // E = k (1 + cos(n φ − δ)); dE/dφ = −k n sin(n φ − δ).
            acc.energy += t.k * (1.0 + arg.cos());
            let de_dphi = -t.k * t.n as f64 * arg.sin();
            g.apply(de_dphi, &mut acc);
        }
        acc
    }

    /// Total energy and per-atom forces.
    pub fn compute(&self, positions: &[[f64; 3]], cell: Option<&Lattice>) -> (f64, Vec<[f64; 3]>) {
        let c = self.compute_all(positions, cell);
        (c.energy, c.forces)
    }
}

/// Improper dihedrals with a harmonic restraint on the torsion angle.
#[derive(Debug, Clone, Default)]
pub struct HarmonicImpropers {
    /// The improper terms.
    pub terms: Vec<ImproperTerm>,
}

impl HarmonicImpropers {
    /// Impropers from a list of terms.
    pub fn new(terms: Vec<ImproperTerm>) -> Self {
        Self { terms }
    }

    /// Energy, forces, and virial for all terms.
    pub fn compute_all(&self, positions: &[[f64; 3]], cell: Option<&Lattice>) -> Contribution {
        let mut acc = Contribution::zeros(positions.len());
        for t in &self.terms {
            let Some(g) = TorsionGeometry::new(t.atoms, positions, cell) else {
                continue;
            };
            // Wrap into (−π, π] so a restraint at ±π does not see a 2π jump.
            let mut d = g.phi - t.phi0;
            while d > std::f64::consts::PI {
                d -= std::f64::consts::TAU;
            }
            while d <= -std::f64::consts::PI {
                d += std::f64::consts::TAU;
            }
            acc.energy += 0.5 * t.k * d * d;
            g.apply(t.k * d, &mut acc);
        }
        acc
    }

    /// Total energy and per-atom forces.
    pub fn compute(&self, positions: &[[f64; 3]], cell: Option<&Lattice>) -> (f64, Vec<[f64; 3]>) {
        let c = self.compute_all(positions, cell);
        (c.energy, c.forces)
    }
}

/// The torsion angle about the j–k bond and its gradient with respect to the
/// four atom positions.
struct TorsionGeometry {
    atoms: (usize, usize, usize, usize),
    /// Torsion angle in radians, in (−π, π].
    phi: f64,
    /// `∂φ/∂r` for each of the four atoms, in `atoms` order.
    dphi: [[f64; 3]; 4],
    /// Positions relative to atom `j`, in `atoms` order. Used for the virial
    /// so it stays independent of how the atoms happen to be wrapped.
    local: [[f64; 3]; 4],
}

impl TorsionGeometry {
    fn new(
        atoms: (usize, usize, usize, usize),
        positions: &[[f64; 3]],
        cell: Option<&Lattice>,
    ) -> Option<Self> {
        let (i, j, k, l) = atoms;
        // b1 = r_j − r_i, b2 = r_k − r_j, b3 = r_l − r_k.
        let b1 = min_image(vec3::sub(positions[j], positions[i]), cell);
        let b2 = min_image(vec3::sub(positions[k], positions[j]), cell);
        let b3 = min_image(vec3::sub(positions[l], positions[k]), cell);

        let n1 = cross(b1, b2);
        let n2 = cross(b2, b3);
        let n1sq = vec3::norm2(n1);
        let n2sq = vec3::norm2(n2);
        let b2len = vec3::norm(b2);
        // Collinear triples leave φ undefined; the term contributes nothing.
        if n1sq < 1e-18 || n2sq < 1e-18 || b2len < 1e-9 {
            return None;
        }

        // φ = atan2( (n1 × n2)·b̂2 , n1·n2 )
        let phi = (vec3::dot(cross(n1, n2), b2) / b2len).atan2(vec3::dot(n1, n2));

        // Gradients (Bekker/van Gunsteren form). u and w are the perpendicular
        // "lever arms" of the end atoms about the central bond.
        let u = vec3::scale(n1, b2len / n1sq);
        let w = vec3::scale(n2, b2len / n2sq);
        let p = vec3::dot(b1, b2) / (b2len * b2len);
        let q = vec3::dot(b3, b2) / (b2len * b2len);

        // φ depends on the three bond vectors, so with B_n = ∂φ/∂b_n the chain
        // rule gives ∂φ/∂r_i = −B₁, ∂φ/∂r_j = B₁ − B₂, ∂φ/∂r_k = B₂ − B₃,
        // ∂φ/∂r_l = B₃ — which sums to zero by construction. Here B₁ = u,
        // B₃ = w, and B₂ = −(p u + q w).
        let dphi_i = vec3::scale(u, -1.0);
        let dphi_l = w;
        let dphi_j = vec3::add(vec3::scale(u, 1.0 + p), vec3::scale(w, q));
        let dphi_k = vec3::sub(vec3::scale(u, -p), vec3::scale(w, 1.0 + q));

        Some(Self {
            atoms,
            phi,
            dphi: [dphi_i, dphi_j, dphi_k, dphi_l],
            // r_j at the origin: r_i = −b1, r_k = b2, r_l = b2 + b3.
            local: [vec3::scale(b1, -1.0), [0.0; 3], b2, vec3::add(b2, b3)],
        })
    }

    /// Distribute `F_a = −(dE/dφ) ∂φ/∂r_a` into the accumulator, along with
    /// the term's virial `Σ_a r_a ⊗ f_a` (well defined because the forces sum
    /// to zero).
    fn apply(&self, de_dphi: f64, acc: &mut Contribution) {
        let idx = [self.atoms.0, self.atoms.1, self.atoms.2, self.atoms.3];
        for ((&at, &dphi), &local) in idx.iter().zip(&self.dphi).zip(&self.local) {
            let f = vec3::scale(dphi, -de_dphi);
            vec3::add_assign(&mut acc.forces[at], f);
            acc.virial_from_pair(local, f);
        }
    }
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Butane-like quartet whose torsion angle is exactly `phi`.
    ///
    /// With i–j along x and the j–k bond along −y, atom `l` is placed on the
    /// unit circle in the x–z plane, so the measured torsion is `atan2(sin φ,
    /// cos φ) = φ`.
    fn quartet(phi: f64) -> Vec<[f64; 3]> {
        vec![
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [phi.cos(), 0.0, phi.sin()],
        ]
    }

    fn measured_phi(pos: &[[f64; 3]]) -> f64 {
        TorsionGeometry::new((0, 1, 2, 3), pos, None).unwrap().phi
    }

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
                let fd = -(eval(&plus).0 - eval(&minus).0) / (2.0 * h);
                assert!(
                    (fd - forces[i][a]).abs() < tol,
                    "atom {i} axis {a}: analytic {} vs fd {fd}",
                    forces[i][a]
                );
            }
        }
    }

    #[test]
    fn torsion_angle_matches_the_constructed_geometry() {
        // The quartet places atom 3 at angle `phi` around the j–k axis, so the
        // measured torsion must come back as ±phi consistently.
        let a = measured_phi(&quartet(0.0));
        assert!(a.abs() < 1e-12, "cis geometry should read 0, got {a}");
        let b = measured_phi(&quartet(PI));
        assert!(
            (b.abs() - PI).abs() < 1e-12,
            "trans should read ±π, got {b}"
        );
        // Monotone and antisymmetric about the cis configuration.
        let p = measured_phi(&quartet(0.7));
        let m = measured_phi(&quartet(-0.7));
        assert!((p + m).abs() < 1e-12, "{p} vs {m}");
        assert!((p.abs() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn periodic_dihedral_forces_match_the_energy_gradient() {
        let d = PeriodicDihedrals::new(vec![
            DihedralTerm {
                atoms: (0, 1, 2, 3),
                k: 0.13,
                n: 3,
                delta: 0.0,
            },
            DihedralTerm {
                atoms: (0, 1, 2, 3),
                k: 0.05,
                n: 1,
                delta: PI,
            },
        ]);
        // A generic, low-symmetry geometry: symmetric ones can hide sign errors.
        let pos = vec![
            [1.2, 0.9, 0.1],
            [0.1, 1.05, -0.2],
            [-0.05, -0.1, 0.15],
            [0.8, -0.7, 1.1],
        ];
        check_forces(&pos, |p| d.compute(p, None), 1e-7);
    }

    #[test]
    fn dihedral_forces_and_torques_vanish_for_the_whole_quartet() {
        let d = PeriodicDihedrals::new(vec![DihedralTerm {
            atoms: (0, 1, 2, 3),
            k: 0.13,
            n: 3,
            delta: 0.4,
        }]);
        let pos = vec![
            [1.2, 0.9, 0.1],
            [0.1, 1.05, -0.2],
            [-0.05, -0.1, 0.15],
            [0.8, -0.7, 1.1],
        ];
        let c = d.compute_all(&pos, None);
        for a in 0..3 {
            let sum: f64 = c.forces.iter().map(|f| f[a]).sum();
            assert!(sum.abs() < 1e-12, "net force along {a} = {sum}");
        }
        // Net torque about the origin must also vanish for an internal term.
        let mut torque = [0.0f64; 3];
        for (r, f) in pos.iter().zip(&c.forces) {
            let t = cross(*r, *f);
            vec3::add_assign(&mut torque, t);
        }
        for (a, t) in torque.iter().enumerate() {
            assert!(t.abs() < 1e-12, "net torque along {a}");
        }
    }

    #[test]
    fn periodic_dihedral_energy_has_the_right_minima() {
        // n = 3, δ = 0 → E = k(1 + cos 3φ), minima at φ = ±π/3, π.
        let d = PeriodicDihedrals::new(vec![DihedralTerm {
            atoms: (0, 1, 2, 3),
            k: 0.1,
            n: 3,
            delta: 0.0,
        }]);
        let e_min = d.compute(&quartet(PI / 3.0), None).0;
        let e_max = d.compute(&quartet(0.0), None).0;
        assert!(e_min.abs() < 1e-9, "expected a minimum, got {e_min}");
        assert!((e_max - 0.2).abs() < 1e-9, "expected 2k, got {e_max}");
    }

    #[test]
    fn improper_forces_match_the_energy_gradient() {
        let imp = HarmonicImpropers::new(vec![ImproperTerm {
            atoms: (0, 1, 2, 3),
            k: 4.0,
            phi0: 0.0,
        }]);
        let pos = vec![
            [1.2, 0.9, 0.1],
            [0.1, 1.05, -0.2],
            [-0.05, -0.1, 0.15],
            [0.8, -0.7, 1.1],
        ];
        check_forces(&pos, |p| imp.compute(p, None), 1e-6);
    }

    #[test]
    fn improper_restrains_a_planar_group() {
        let imp = HarmonicImpropers::new(vec![ImproperTerm {
            atoms: (0, 1, 2, 3),
            k: 4.0,
            phi0: 0.0,
        }]);
        // Planar (φ = 0) sits at zero energy; puckering costs energy.
        assert!(imp.compute(&quartet(0.0), None).0.abs() < 1e-12);
        let puckered = imp.compute(&quartet(0.3), None).0;
        assert!((puckered - 0.5 * 4.0 * 0.09).abs() < 1e-9, "{puckered}");
    }

    #[test]
    fn dihedral_is_invariant_under_minimum_image_wrapping() {
        let d = PeriodicDihedrals::new(vec![DihedralTerm {
            atoms: (0, 1, 2, 3),
            k: 0.13,
            n: 3,
            delta: 0.4,
        }]);
        let cell = Lattice::cubic(10.0);
        let pos = vec![
            [1.2, 0.9, 0.1],
            [0.1, 1.05, -0.2],
            [-0.05, -0.1, 0.15],
            [0.8, -0.7, 1.1],
        ];
        let unwrapped = d.compute_all(&pos, Some(&cell));
        let wrapped: Vec<[f64; 3]> = pos.iter().map(|p| cell.wrap(*p)).collect();
        let after = d.compute_all(&wrapped, Some(&cell));
        assert!((unwrapped.energy - after.energy).abs() < 1e-12);
        for (a, b) in unwrapped.forces.iter().zip(&after.forces) {
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 1e-12);
            }
        }
    }
}
