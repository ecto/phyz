//! CSR (Compressed Sparse Row) matrix for GPU-friendly Hamiltonian storage.
//!
//! Converts the on-the-fly `hamiltonian_matvec` into an explicit sparse matrix
//! that can be uploaded to GPU once and reused for all Lanczos iterations.

use crate::hilbert::U1HilbertSpace;
use crate::su2_quantum::Su2HilbertSpace;
use phyz_regge::SimplicialComplex;

/// Sparse matrix in Compressed Sparse Row format.
///
/// All values stored as f64 (source of truth). Conversion to f32 happens
/// at GPU upload time if the device lacks f64 support.
pub struct CsrMatrix {
    pub nrows: usize,
    pub row_ptr: Vec<u32>,
    pub col_indices: Vec<u32>,
    pub values: Vec<f64>,
}

impl CsrMatrix {
    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Compute y = A * x on CPU (for testing).
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.nrows];
        for i in 0..self.nrows {
            let start = self.row_ptr[i] as usize;
            let end = self.row_ptr[i + 1] as usize;
            let mut sum = 0.0;
            for idx in start..end {
                sum += self.values[idx] * x[self.col_indices[idx] as usize];
            }
            y[i] = sum;
        }
        y
    }
}

/// Build the KS Hamiltonian as a CSR sparse matrix.
///
/// Same physics as `hamiltonian_matvec` (lanczos.rs) but stored explicitly.
/// Construction is one-time cost — sequential for simplicity.
pub fn build_csr(
    hilbert: &U1HilbertSpace,
    complex: &SimplicialComplex,
    g_squared: f64,
    metric_weights: Option<&[f64]>,
) -> CsrMatrix {
    let dim = hilbert.dim();
    let e_coeff = g_squared / 2.0;
    let b_coeff = -1.0 / g_squared * 0.5;

    // Accumulate entries per row: Vec<(col, val)>
    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); dim];

    // Electric term: diagonal
    for i in 0..dim {
        let config = hilbert.index_to_config(i);
        let energy: f64 = config.iter().map(|&n| (n as f64) * (n as f64)).sum();
        let val = e_coeff * energy;
        if val.abs() > 0.0 {
            rows[i].push((i as u32, val));
        }
    }

    // Magnetic term: off-diagonal plaquette shifts
    let signs: [i32; 3] = [1, -1, 1];
    let lam = hilbert.lambda as i32;

    for (ti, _tri) in complex.triangles.iter().enumerate() {
        let edge_indices = complex.tri_edge_indices(ti);
        let weight = metric_weights.map_or(1.0, |w| w[ti]);
        let c = b_coeff * weight;

        for i in 0..dim {
            let config = hilbert.index_to_config(i);

            // Forward shift: U_tri
            let mut new_config = config.to_vec();
            let mut valid = true;
            for k in 0..3 {
                new_config[edge_indices[k]] += signs[k];
                if new_config[edge_indices[k]] < -lam || new_config[edge_indices[k]] > lam {
                    valid = false;
                    break;
                }
            }
            if valid {
                if let Some(j) = hilbert.config_to_index(&new_config) {
                    // H[i,j] = c and H[j,i] = c (symmetric)
                    rows[i].push((j as u32, c));
                    rows[j].push((i as u32, c));
                }
            }
        }
    }

    // Sort each row by column index, merge duplicates, build CSR
    let mut row_ptr = Vec::with_capacity(dim + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptr.push(0u32);

    for row in &mut rows {
        row.sort_by_key(|&(col, _)| col);

        // Merge duplicates
        let mut merged: Vec<(u32, f64)> = Vec::new();
        for &(col, val) in row.iter() {
            if let Some(last) = merged.last_mut() {
                if last.0 == col {
                    last.1 += val;
                    continue;
                }
            }
            merged.push((col, val));
        }

        for (col, val) in merged {
            col_indices.push(col);
            values.push(val);
        }
        row_ptr.push(col_indices.len() as u32);
    }

    CsrMatrix {
        nrows: dim,
        row_ptr,
        col_indices,
        values,
    }
}

/// Build the SU(2) j=1/2 Hamiltonian as a CSR sparse matrix.
///
/// H = (3g²/8) Σ_e n_e − (1/2g²) Σ_tri B_tri
///
/// Electric term is diagonal (popcount-based), magnetic term flips all 3
/// edges of each triangle via XOR. Gauss law is preserved by construction.
pub fn build_csr_su2(
    hilbert: &Su2HilbertSpace,
    complex: &SimplicialComplex,
    g_squared: f64,
) -> CsrMatrix {
    let dim = hilbert.dim();
    let e_coeff = 3.0 * g_squared / 8.0;
    let b_coeff = -0.5 / g_squared;

    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); dim];

    // Electric term: diagonal, (3g²/8) × popcount(state)
    for (i, &state) in hilbert.basis.iter().enumerate() {
        let val = e_coeff * state.count_ones() as f64;
        if val.abs() > 0.0 {
            rows[i].push((i as u32, val));
        }
    }

    // Magnetic term: plaquette flips
    for ti in 0..complex.n_triangles() {
        let [e0, e1, e2] = complex.tri_edge_indices(ti);
        let flip_mask: u64 = (1 << e0) | (1 << e1) | (1 << e2);

        for (i, &state) in hilbert.basis.iter().enumerate() {
            let flipped = state ^ flip_mask;
            if let Some(j) = hilbert.config_to_index(flipped) {
                rows[i].push((j as u32, b_coeff));
            }
        }
    }

    // Sort each row by column index, merge duplicates, build CSR
    let mut row_ptr = Vec::with_capacity(dim + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptr.push(0u32);

    for row in &mut rows {
        row.sort_by_key(|&(col, _)| col);

        let mut merged: Vec<(u32, f64)> = Vec::new();
        for &(col, val) in row.iter() {
            if let Some(last) = merged.last_mut() {
                if last.0 == col {
                    last.1 += val;
                    continue;
                }
            }
            merged.push((col, val));
        }

        for (col, val) in merged {
            col_indices.push(col);
            values.push(val);
        }
        row_ptr.push(col_indices.len() as u32);
    }

    CsrMatrix {
        nrows: dim,
        row_ptr,
        col_indices,
        values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lanczos::hamiltonian_matvec;
    use nalgebra::DVector;

    fn single_pentachoron() -> SimplicialComplex {
        SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]])
    }

    #[test]
    fn test_csr_matvec_matches_hamiltonian_matvec() {
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let csr = build_csr(&hs, &complex, 1.0, None);

        for seed in 0..5u64 {
            let mut v = DVector::zeros(hs.dim());
            for i in 0..hs.dim() {
                v[i] = ((i as u64 + seed * 137) as f64 * 0.618).fract() - 0.5;
            }

            let hv_matvec = hamiltonian_matvec(&v, &hs, &complex, 1.0, None);
            let hv_csr = csr.matvec(v.as_slice());

            let diff: f64 = hv_matvec
                .iter()
                .zip(hv_csr.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                diff < 1e-10,
                "CSR matvec mismatch at seed {seed}: max_diff={diff}"
            );
        }
    }

    #[test]
    fn test_csr_symmetric() {
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let csr = build_csr(&hs, &complex, 1.0, None);

        // Verify A*e_i == column i matches row i entries
        // Simpler: check (Ax)·y == x·(Ay) for random vectors
        let dim = hs.dim();
        let mut x = vec![0.0; dim];
        let mut y = vec![0.0; dim];
        for i in 0..dim {
            x[i] = ((i as f64 + 1.0) * 0.618).fract() - 0.5;
            y[i] = ((i as f64 + 2.0) * 0.314).fract() - 0.5;
        }

        let ax = csr.matvec(&x);
        let ay = csr.matvec(&y);

        let ax_dot_y: f64 = ax.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let x_dot_ay: f64 = x.iter().zip(ay.iter()).map(|(a, b)| a * b).sum();

        let diff = (ax_dot_y - x_dot_ay).abs();
        assert!(diff < 1e-10, "CSR not symmetric: diff={diff}");
    }

    #[test]
    fn test_csr_with_metric_weights() {
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let weights = vec![2.0; complex.n_triangles()];
        let csr = build_csr(&hs, &complex, 1.0, Some(&weights));

        let mut v = DVector::zeros(hs.dim());
        for i in 0..hs.dim() {
            v[i] = ((i as f64 + 1.0) * 0.618).fract() - 0.5;
        }

        let hv_matvec = hamiltonian_matvec(&v, &hs, &complex, 1.0, Some(&weights));
        let hv_csr = csr.matvec(v.as_slice());

        let diff: f64 = hv_matvec
            .iter()
            .zip(hv_csr.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            diff < 1e-10,
            "CSR matvec with weights mismatch: max_diff={diff}"
        );
    }

    #[test]
    fn test_csr_dimensions() {
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let csr = build_csr(&hs, &complex, 1.0, None);

        assert_eq!(csr.nrows, hs.dim());
        assert_eq!(csr.row_ptr.len(), hs.dim() + 1);
        assert!(csr.nnz() > 0);
        assert_eq!(csr.col_indices.len(), csr.values.len());
    }

    #[test]
    fn test_csr_su2_matvec_matches_dense() {
        use crate::su2_quantum::{build_su2_hamiltonian, Su2HilbertSpace};

        let complex = single_pentachoron();
        let hs = Su2HilbertSpace::new(&complex);
        let csr = build_csr_su2(&hs, &complex, 1.0);
        let h_dense = build_su2_hamiltonian(&hs, &complex, 1.0);

        for seed in 0..5u64 {
            let mut v = DVector::zeros(hs.dim());
            for i in 0..hs.dim() {
                v[i] = ((i as u64 + seed * 137) as f64 * 0.618).fract() - 0.5;
            }

            let hv_dense: Vec<f64> = (&h_dense * &v).iter().copied().collect();
            let hv_csr = csr.matvec(v.as_slice());

            let diff: f64 = hv_dense
                .iter()
                .zip(hv_csr.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                diff < 1e-10,
                "SU(2) CSR matvec mismatch at seed {seed}: max_diff={diff}"
            );
        }
    }
}
