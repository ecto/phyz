//! Force field models for molecular dynamics.

use tau_math::Vec3;

/// Force field trait for computing interatomic forces.
pub trait ForceField: Send + Sync {
    /// Compute force on particle i due to particle j at distance vector r_ij = r_j - r_i.
    /// Returns (force on i, potential energy).
    fn compute_force(&self, r_ij: Vec3, atom_type_i: u32, atom_type_j: u32) -> (Vec3, f64);

    /// Cutoff radius for this force field.
    fn cutoff(&self) -> f64;
}

/// Lennard-Jones 12-6 potential: V(r) = 4ε [(σ/r)^12 - (σ/r)^6].
#[derive(Clone, Debug)]
pub struct LennardJones {
    /// Well depth (eV or similar energy units).
    pub epsilon: f64,
    /// Zero-crossing distance (Å or nm).
    pub sigma: f64,
    /// Cutoff radius (typically 2.5σ).
    pub r_cut: f64,
}

impl LennardJones {
    /// Create Lennard-Jones potential with given parameters.
    pub fn new(epsilon: f64, sigma: f64, r_cut: f64) -> Self {
        Self {
            epsilon,
            sigma,
            r_cut,
        }
    }

    /// Create argon Lennard-Jones potential (ε=0.0103 eV, σ=3.4 Å).
    pub fn argon() -> Self {
        Self::new(0.0103, 3.4, 8.5) // r_cut = 2.5σ
    }
}

impl ForceField for LennardJones {
    fn compute_force(&self, r_ij: Vec3, _atom_type_i: u32, _atom_type_j: u32) -> (Vec3, f64) {
        let r = r_ij.norm();
        if r > self.r_cut || r < 1e-10 {
            return (Vec3::zeros(), 0.0);
        }

        let s_r = self.sigma / r;
        let s_r6 = s_r.powi(6);
        let s_r12 = s_r6 * s_r6;

        // V(r) = 4ε [(σ/r)^12 - (σ/r)^6]
        let potential = 4.0 * self.epsilon * (s_r12 - s_r6);

        // dV/dr = 24ε/r [(σ/r)^6 - 2(σ/r)^12]
        // F_i = dV/dr * r_ij/r (positive derivative gives force toward j)
        let dv_dr = 24.0 * self.epsilon / r * (s_r6 - 2.0 * s_r12);

        // Force on i
        let force = dv_dr * r_ij / r;

        (force, potential)
    }

    fn cutoff(&self) -> f64 {
        self.r_cut
    }
}

/// Coulomb electrostatic potential: V(r) = k q_i q_j / r.
#[derive(Clone, Debug)]
pub struct Coulomb {
    /// Coulomb constant k (eV·Å/e² or similar).
    pub k: f64,
    /// Charges for each atom type.
    pub charges: Vec<f64>,
    /// Cutoff radius.
    pub r_cut: f64,
}

impl Coulomb {
    /// Create Coulomb potential with given parameters.
    /// k in eV·Å/e² ≈ 14.4 for atomic units.
    pub fn new(k: f64, charges: Vec<f64>, r_cut: f64) -> Self {
        Self { k, charges, r_cut }
    }
}

impl ForceField for Coulomb {
    fn compute_force(&self, r_ij: Vec3, atom_type_i: u32, atom_type_j: u32) -> (Vec3, f64) {
        let r = r_ij.norm();
        if r > self.r_cut || r < 1e-10 {
            return (Vec3::zeros(), 0.0);
        }

        let q_i = self
            .charges
            .get(atom_type_i as usize)
            .copied()
            .unwrap_or(0.0);
        let q_j = self
            .charges
            .get(atom_type_j as usize)
            .copied()
            .unwrap_or(0.0);

        // V(r) = k q_i q_j / r
        let potential = self.k * q_i * q_j / r;

        // dV/dr = -k q_i q_j / r²
        // F_i = dV/dr * r_ij/r = -k q_i q_j / r² * r_ij/r
        let dv_dr = -self.k * q_i * q_j / (r * r);

        // Force on i
        let force = dv_dr * r_ij / r;

        (force, potential)
    }

    fn cutoff(&self) -> f64 {
        self.r_cut
    }
}

/// Harmonic bond potential: V(r) = 0.5 k (r - r0)².
#[derive(Clone, Debug)]
pub struct HarmonicBond {
    /// Spring constant (eV/Å² or similar).
    pub k: f64,
    /// Equilibrium bond length (Å).
    pub r0: f64,
}

impl HarmonicBond {
    /// Create harmonic bond potential.
    pub fn new(k: f64, r0: f64) -> Self {
        Self { k, r0 }
    }

    /// Compute force and potential for a bond.
    pub fn compute(&self, r_ij: Vec3) -> (Vec3, f64) {
        let r = r_ij.norm();
        if r < 1e-10 {
            return (Vec3::zeros(), 0.0);
        }

        let dr = r - self.r0;
        // V(r) = 0.5 k (r - r0)²
        let potential = 0.5 * self.k * dr * dr;

        // dV/dr = k (r - r0)
        // F_i = dV/dr * r_ij/r = k (r - r0) * r_ij/r
        let dv_dr = self.k * dr;

        // Force on i
        let force = dv_dr * r_ij / r;

        (force, potential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lj_argon() {
        let lj = LennardJones::argon();
        let r = Vec3::new(3.4, 0.0, 0.0); // At σ

        let (force, pot) = lj.compute_force(r, 0, 0);

        // At r=σ, V(σ) = 0
        assert_relative_eq!(pot, 0.0, epsilon = 1e-10);

        // Force should be repulsive at r=σ
        // dV/dr = 24ε/σ [1 - 2] = -24ε/σ < 0
        // F = dV/dr * r_ij/r < 0 * r_ij/r, points opposite to r_ij (repulsive)
        assert!(force.x < 0.0); // Negative x means repulsive
    }

    #[test]
    fn test_lj_minimum() {
        let lj = LennardJones::argon();
        // Minimum at r = 2^(1/6) σ ≈ 1.122σ
        let r_min = 1.122462 * lj.sigma;
        let r = Vec3::new(r_min, 0.0, 0.0);

        let (force, pot) = lj.compute_force(r, 0, 0);

        // At minimum, V ≈ -ε
        assert_relative_eq!(pot, -lj.epsilon, epsilon = 1e-3);

        // Force should be approximately zero at minimum
        assert!(force.norm() < 1e-2);
    }

    #[test]
    fn test_coulomb() {
        let coulomb = Coulomb::new(14.4, vec![1.0, -1.0], 10.0);
        let r = Vec3::new(2.0, 0.0, 0.0);

        let (force, pot) = coulomb.compute_force(r, 0, 1);

        // V(2Å) = 14.4 * 1 * (-1) / 2 = -7.2 eV
        assert_relative_eq!(pot, -7.2, epsilon = 1e-10);

        // Force should be attractive (positive x-component pulling i toward j)
        // dV/dr = -k q_i q_j / r² = -14.4 * 1 * (-1) / 4 = 3.6
        // F = dV/dr * r_ij/r = 3.6 * (2,0,0) / 2 = (3.6, 0, 0)
        assert_relative_eq!(force.x, 3.6, epsilon = 1e-10);
    }

    #[test]
    fn test_harmonic_bond() {
        let bond = HarmonicBond::new(100.0, 1.5);
        let r = Vec3::new(2.0, 0.0, 0.0);

        let (force, pot) = bond.compute(r);

        // dr = 2.0 - 1.5 = 0.5
        // V = 0.5 * 100 * 0.25 = 12.5
        assert_relative_eq!(pot, 12.5, epsilon = 1e-10);

        // dV/dr = k (r - r0) = 100 * 0.5 = 50
        // F = dV/dr * r_ij/r = 50 * (2,0,0) / 2 = 50 * (1,0,0) = (50, 0, 0)
        // Positive means pushing i toward j (attractive back to equilibrium)
        assert_relative_eq!(force.x, 50.0, epsilon = 1e-10);
    }
}
