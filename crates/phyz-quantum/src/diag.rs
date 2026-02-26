//! Dense eigendecomposition for small Hamiltonians.
//!
//! Uses `SymmetricEigen` for real symmetric matrices.
//! Suitable for Hilbert spaces up to ~50K dimension.

use phyz_math::{DMat, DVec};

/// Eigenvalues and eigenstates from diagonalization.
#[derive(Debug, Clone)]
pub struct Spectrum {
    /// Eigenvalues in ascending order.
    pub energies: Vec<f64>,
    /// Corresponding eigenstates (columns of the unitary matrix).
    pub states: Vec<DVec>,
}

impl Spectrum {
    /// Ground state energy.
    pub fn ground_energy(&self) -> f64 {
        self.energies[0]
    }

    /// Ground state vector.
    pub fn ground_state(&self) -> &DVec {
        &self.states[0]
    }

    /// Spectral gap (E_1 - E_0).
    pub fn gap(&self) -> f64 {
        if self.energies.len() < 2 {
            return 0.0;
        }
        self.energies[1] - self.energies[0]
    }
}

/// Threshold: use Lanczos when dim >= this AND n_lowest is small.
const LANCZOS_DENSE_THRESHOLD: usize = 64;

/// Diagonalize a real symmetric matrix.
///
/// If `n_lowest` is `Some(n)`, only the `n` lowest eigenvalues/states
/// are returned. When `n` is small relative to dim, uses Lanczos with
/// dense matvec (O(n_iter * dim²)) instead of full eigen (O(dim³)).
pub fn diagonalize(h: &DMat, n_lowest: Option<usize>) -> Spectrum {
    let dim = h.nrows();
    let k = n_lowest.unwrap_or(dim);

    // Use Lanczos when we only need a few eigenvalues of a large matrix.
    if k < dim / 2 && dim >= LANCZOS_DENSE_THRESHOLD {
        let matvec = |v: &DVec| h * v;
        let max_iter = (20 * k).max(100).min(dim);
        return crate::lanczos::lanczos(matvec, dim, k, max_iter, 1e-12);
    }

    diagonalize_full(h, n_lowest)
}

/// Full dense diagonalization (always computes all eigenvalues).
pub fn diagonalize_full(h: &DMat, n_lowest: Option<usize>) -> Spectrum {
    let eig = h.symmetric_eigen();

    // Sort by eigenvalue.
    let mut indexed: Vec<(usize, f64)> = eig
        .eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let n = match n_lowest {
        Some(n) => n.min(indexed.len()),
        None => indexed.len(),
    };

    let energies: Vec<f64> = indexed[..n].iter().map(|&(_, e)| e).collect();
    let states: Vec<DVec> = indexed[..n]
        .iter()
        .map(|&(i, _)| eig.eigenvectors.column(i))
        .collect();

    Spectrum { energies, states }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_spectrum() {
        let h = DMat::identity(3);
        let spec = diagonalize(&h, None);

        assert_eq!(spec.energies.len(), 3);
        for &e in &spec.energies {
            assert!((e - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_diagonal_matrix() {
        let h = DMat::from_diagonal(&DVec::from_vec(vec![3.0, 1.0, 2.0]));
        let spec = diagonalize(&h, None);

        assert!((spec.energies[0] - 1.0).abs() < 1e-12);
        assert!((spec.energies[1] - 2.0).abs() < 1e-12);
        assert!((spec.energies[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_n_lowest() {
        let h = DMat::from_diagonal(&DVec::from_vec(vec![3.0, 1.0, 2.0]));
        let spec = diagonalize(&h, Some(2));

        assert_eq!(spec.energies.len(), 2);
        assert!((spec.energies[0] - 1.0).abs() < 1e-12);
        assert!((spec.energies[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_gap() {
        let h = DMat::from_diagonal(&DVec::from_vec(vec![3.0, 1.0, 2.0]));
        let spec = diagonalize(&h, None);
        assert!((spec.gap() - 1.0).abs() < 1e-12);
    }
}
