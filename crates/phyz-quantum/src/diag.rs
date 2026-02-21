//! Dense eigendecomposition for small Hamiltonians.
//!
//! Uses nalgebra's `SymmetricEigen` for real symmetric matrices.
//! Suitable for Hilbert spaces up to ~50K dimension.

use nalgebra::{DMatrix, DVector};

/// Eigenvalues and eigenstates from diagonalization.
#[derive(Debug, Clone)]
pub struct Spectrum {
    /// Eigenvalues in ascending order.
    pub energies: Vec<f64>,
    /// Corresponding eigenstates (columns of the unitary matrix).
    pub states: Vec<DVector<f64>>,
}

impl Spectrum {
    /// Ground state energy.
    pub fn ground_energy(&self) -> f64 {
        self.energies[0]
    }

    /// Ground state vector.
    pub fn ground_state(&self) -> &DVector<f64> {
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

/// Diagonalize a real symmetric matrix.
///
/// If `n_lowest` is `Some(n)`, only the `n` lowest eigenvalues/states
/// are returned (still computed via full diagonalization).
pub fn diagonalize(h: &DMatrix<f64>, n_lowest: Option<usize>) -> Spectrum {
    let eig = h.clone().symmetric_eigen();

    // Sort by eigenvalue.
    let mut indexed: Vec<(usize, f64)> = eig.eigenvalues.iter().enumerate().map(|(i, &e)| (i, e)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let n = match n_lowest {
        Some(n) => n.min(indexed.len()),
        None => indexed.len(),
    };

    let energies: Vec<f64> = indexed[..n].iter().map(|&(_, e)| e).collect();
    let states: Vec<DVector<f64>> = indexed[..n]
        .iter()
        .map(|&(i, _)| eig.eigenvectors.column(i).into_owned())
        .collect();

    Spectrum { energies, states }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_spectrum() {
        let h = DMatrix::identity(3, 3);
        let spec = diagonalize(&h, None);

        assert_eq!(spec.energies.len(), 3);
        for &e in &spec.energies {
            assert!((e - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_diagonal_matrix() {
        let h = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 1.0, 2.0]));
        let spec = diagonalize(&h, None);

        assert!((spec.energies[0] - 1.0).abs() < 1e-12);
        assert!((spec.energies[1] - 2.0).abs() < 1e-12);
        assert!((spec.energies[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_n_lowest() {
        let h = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 1.0, 2.0]));
        let spec = diagonalize(&h, Some(2));

        assert_eq!(spec.energies.len(), 2);
        assert!((spec.energies[0] - 1.0).abs() < 1e-12);
        assert!((spec.energies[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_gap() {
        let h = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 1.0, 2.0]));
        let spec = diagonalize(&h, None);
        assert!((spec.gap() - 1.0).abs() < 1e-12);
    }
}
