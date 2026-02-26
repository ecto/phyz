//! Lanczos eigensolver for large sparse Hamiltonians.
//!
//! Finds the k lowest eigenvalues and eigenvectors of a real symmetric matrix
//! using the Lanczos algorithm with full reorthogonalization.
//!
//! This enables exact diagonalization of systems too large for dense methods:
//! - 2-pentachoron at Λ=2 (dim ~269K)
//! - 3-pentachoron at Λ=1 (dim ~47K)
//!
//! The key insight: we never store the full Hamiltonian matrix. Instead, we
//! apply H|v⟩ on-the-fly by iterating over basis states and plaquettes.

use crate::diag::Spectrum;
use crate::hilbert::U1HilbertSpace;
use phyz_math::{DMat, DVec};
use phyz_regge::SimplicialComplex;

/// Hamiltonian-vector product without storing the full matrix.
///
/// Computes H|v⟩ = H_E|v⟩ + H_B|v⟩ directly from the Hilbert space and
/// complex structure.
pub fn hamiltonian_matvec(
    v: &DVec,
    hilbert: &U1HilbertSpace,
    complex: &SimplicialComplex,
    g_squared: f64,
    metric_weights: Option<&[f64]>,
) -> DVec {
    let dim = hilbert.dim();
    let mut result = DVec::zeros(dim);

    let e_coeff = g_squared / 2.0;
    let b_coeff = -1.0 / g_squared * 0.5;

    // Electric term: diagonal, H_E|v⟩_i = (g²/2) Σ_e n_e² · v_i
    for i in 0..dim {
        let config = hilbert.index_to_config(i);
        let energy: f64 = config.iter().map(|&n| (n as f64) * (n as f64)).sum();
        result[i] += e_coeff * energy * v[i];
    }

    // Magnetic term: off-diagonal plaquette shifts
    let signs: [i32; 3] = [1, -1, 1]; // holonomy convention
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
                    // result[i] += c * v[j] and result[j] += c * v[i]
                    result[i] += c * v[j];
                    result[j] += c * v[i];
                }
            }
        }
    }

    result
}

/// Lanczos algorithm with full reorthogonalization.
///
/// Builds a tridiagonal matrix T of size m×m from the Krylov subspace,
/// then diagonalizes T to get approximate eigenvalues/eigenvectors.
///
/// # Arguments
/// * `matvec` — function that computes H|v⟩
/// * `dim` — dimension of the Hilbert space
/// * `n_eigenvalues` — number of lowest eigenvalues to find
/// * `max_iter` — maximum Lanczos iterations (typically 100-300)
/// * `tol` — convergence tolerance on eigenvalue change
pub fn lanczos<F>(
    matvec: F,
    dim: usize,
    n_eigenvalues: usize,
    max_iter: usize,
    tol: f64,
) -> Spectrum
where
    F: Fn(&DVec) -> DVec,
{
    let m = max_iter.min(dim);
    let k = n_eigenvalues.min(m);

    // Lanczos vectors (stored for reorthogonalization and eigenvector recovery)
    let mut q_vecs: Vec<DVec> = Vec::with_capacity(m + 1);

    // Tridiagonal elements
    let mut alpha: Vec<f64> = Vec::with_capacity(m); // diagonal
    let mut beta: Vec<f64> = Vec::with_capacity(m); // off-diagonal

    // Initial random vector (deterministic seed for reproducibility)
    let mut q = DVec::zeros(dim);
    // Use a simple deterministic initialization
    for i in 0..dim {
        q[i] = ((i as f64 + 1.0) * 0.618033988749895).fract() - 0.5;
    }
    let norm = q.norm();
    q *= 1.0 / norm;
    q_vecs.push(q.clone());

    let mut prev_eigenvalues = vec![f64::MAX; k];

    for j in 0..m {
        // w = H * q_j
        let mut w = matvec(&q_vecs[j]);

        // α_j = q_j · w
        let a = q_vecs[j].dot(&w);
        alpha.push(a);

        // w = w - α_j * q_j - β_{j-1} * q_{j-1}
        w -= &(&q_vecs[j] * a);
        if j > 0 {
            w -= &(&q_vecs[j - 1] * beta[j - 1]);
        }

        // Full reorthogonalization (crucial for numerical stability)
        for qi in &q_vecs {
            let overlap = qi.dot(&w);
            w -= &(qi * overlap);
        }

        let b = w.norm();

        // Check for convergence periodically
        if (j + 1) % 10 == 0 || j == m - 1 || b < 1e-14 {
            let spec = diagonalize_tridiagonal(&alpha, &beta, k);
            let max_change = spec
                .iter()
                .zip(prev_eigenvalues.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);

            if max_change < tol {
                eprintln!(
                    "  Lanczos converged at iteration {} (change={:.2e})",
                    j + 1,
                    max_change
                );
                // Recover eigenvectors
                return recover_eigenvectors(&alpha, &beta, &q_vecs, k);
            }
            prev_eigenvalues = spec;
        }

        if b < 1e-14 {
            eprintln!("  Lanczos: invariant subspace found at iteration {}", j + 1);
            return recover_eigenvectors(&alpha, &beta, &q_vecs, k);
        }

        beta.push(b);
        let q_next = &w * (1.0 / b);
        q_vecs.push(q_next);
    }

    eprintln!("  Lanczos: max iterations ({m}) reached");
    recover_eigenvectors(&alpha, &beta, &q_vecs, k)
}

/// Diagonalize the tridiagonal matrix to get eigenvalues only.
fn diagonalize_tridiagonal(alpha: &[f64], beta: &[f64], k: usize) -> Vec<f64> {
    let m = alpha.len();
    let mut t = DMat::zeros(m, m);
    for i in 0..m {
        t[(i, i)] = alpha[i];
        if i > 0 {
            t[(i, i - 1)] = beta[i - 1];
            t[(i - 1, i)] = beta[i - 1];
        }
    }
    let eig = t.symmetric_eigen();
    let mut vals: Vec<f64> = eig.eigenvalues.iter().copied().collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals.truncate(k);
    vals
}

/// Recover eigenvectors from Lanczos vectors and tridiagonal eigenvectors.
fn recover_eigenvectors(alpha: &[f64], beta: &[f64], q_vecs: &[DVec], k: usize) -> Spectrum {
    let m = alpha.len();
    let mut t = DMat::zeros(m, m);
    for i in 0..m {
        t[(i, i)] = alpha[i];
        if i > 0 {
            t[(i, i - 1)] = beta[i - 1];
            t[(i - 1, i)] = beta[i - 1];
        }
    }
    let eig = t.symmetric_eigen();

    // Sort by eigenvalue
    let mut indexed: Vec<(usize, f64)> = eig
        .eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let n = k.min(indexed.len());
    let dim = q_vecs[0].len();
    let n_q = q_vecs.len().min(m); // number of Lanczos vectors we actually have

    let mut energies = Vec::with_capacity(n);
    let mut states = Vec::with_capacity(n);

    for &(idx, eval) in indexed.iter().take(n) {
        energies.push(eval);

        // Eigenvector in original space: Σ_j s_j * q_j
        let mut v = DVec::zeros(dim);
        for j in 0..n_q {
            let coeff = eig.eigenvectors[(j, idx)];
            v += &(&q_vecs[j] * coeff);
        }
        // Normalize (should already be ~1 but enforce)
        let norm = v.norm();
        if norm > 1e-15 {
            v *= 1.0 / norm;
        }
        states.push(v);
    }

    Spectrum { energies, states }
}

/// Convenience: run Lanczos on a simplicial KS Hamiltonian.
pub fn lanczos_diagonalize(
    hilbert: &U1HilbertSpace,
    complex: &SimplicialComplex,
    g_squared: f64,
    metric_weights: Option<&[f64]>,
    n_eigenvalues: usize,
    max_iter: Option<usize>,
) -> Spectrum {
    let dim = hilbert.dim();

    // Choose max iterations: caller override or default heuristic, capped at dim.
    let max_iter = max_iter
        .unwrap_or_else(|| (20 * n_eigenvalues).max(100))
        .min(dim);
    let tol = 1e-10;

    eprintln!("  Lanczos: dim={dim}, k={n_eigenvalues}, max_iter={max_iter}");

    let mw = metric_weights.map(|w| w.to_vec());
    let matvec = |v: &DVec| hamiltonian_matvec(v, hilbert, complex, g_squared, mw.as_deref());

    lanczos(matvec, dim, n_eigenvalues, max_iter, tol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag;
    use crate::hamiltonian::{KSParams, build_hamiltonian};

    fn single_pentachoron() -> SimplicialComplex {
        SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]])
    }

    #[test]
    fn test_lanczos_vs_dense_ground_state() {
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);

        // Dense reference
        let params = KSParams {
            g_squared: 1.0,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);
        let dense = diag::diagonalize_full(&h, Some(5));

        // Lanczos
        let lanc = lanczos_diagonalize(&hs, &complex, 1.0, None, 5, None);

        // Ground state energy should match to high precision
        let e0_diff = (dense.ground_energy() - lanc.ground_energy()).abs();
        assert!(
            e0_diff < 1e-8,
            "E₀ mismatch: dense={}, lanczos={}, diff={e0_diff}",
            dense.ground_energy(),
            lanc.ground_energy()
        );

        // Gap should match
        let gap_diff = (dense.gap() - lanc.gap()).abs();
        assert!(
            gap_diff < 1e-8,
            "Gap mismatch: dense={}, lanczos={}, diff={gap_diff}",
            dense.gap(),
            lanc.gap()
        );
    }

    #[test]
    fn test_lanczos_vs_dense_distinct_levels() {
        // Lanczos may not fully resolve degenerate multiplets (the 6-fold E₁
        // level of the pentachoron is particularly tricky). So we compare
        // distinct energy levels rather than individual eigenvalues.
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);

        let params = KSParams {
            g_squared: 1.0,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);
        let dense = diag::diagonalize_full(&h, Some(10));

        let lanc = lanczos_diagonalize(&hs, &complex, 1.0, None, 10, None);

        // Extract distinct energy levels from dense spectrum
        let mut dense_levels = vec![dense.energies[0]];
        for &e in &dense.energies[1..] {
            if (e - dense_levels.last().unwrap()).abs() > 1e-4 {
                dense_levels.push(e);
            }
        }

        // Check that Lanczos finds the same distinct levels
        let mut lanc_levels = vec![lanc.energies[0]];
        for &e in &lanc.energies[1..] {
            if (e - lanc_levels.last().unwrap()).abs() > 1e-4 {
                lanc_levels.push(e);
            }
        }

        // At least 3 distinct levels should match
        let n_check = 3.min(dense_levels.len()).min(lanc_levels.len());
        for i in 0..n_check {
            let diff = (dense_levels[i] - lanc_levels[i]).abs();
            assert!(
                diff < 1e-4,
                "Level {i} mismatch: dense={}, lanczos={}, diff={diff}",
                dense_levels[i],
                lanc_levels[i]
            );
        }
    }

    #[test]
    fn test_lanczos_strong_coupling() {
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);

        let lanc = lanczos_diagonalize(&hs, &complex, 1e6, None, 3, None);
        assert!(
            lanc.ground_energy().abs() < 1e-2,
            "Strong coupling E₀ = {}",
            lanc.ground_energy()
        );
    }

    #[test]
    #[ignore] // slow in debug mode (dim=3135, unoptimized matvec)
    fn test_lanczos_two_pentachorons() {
        let complex = SimplicialComplex::from_pentachorons(6, &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]]);
        let hs = U1HilbertSpace::new(&complex, 1);

        // Dense reference
        let params = KSParams {
            g_squared: 1.0,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);
        let dense = diag::diagonalize_full(&h, Some(5));

        // Lanczos
        let lanc = lanczos_diagonalize(&hs, &complex, 1.0, None, 5, None);

        let e0_diff = (dense.ground_energy() - lanc.ground_energy()).abs();
        assert!(
            e0_diff < 1e-6,
            "2-pent E₀ mismatch: dense={}, lanczos={}, diff={e0_diff}",
            dense.ground_energy(),
            lanc.ground_energy()
        );
    }

    #[test]
    fn test_matvec_matches_dense() {
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);

        let params = KSParams {
            g_squared: 1.0,
            metric_weights: None,
        };
        let h_dense = build_hamiltonian(&hs, &complex, &params);

        // Test matvec on a few random-ish vectors
        for seed in 0..5 {
            let mut v = DVec::zeros(hs.dim());
            for i in 0..hs.dim() {
                v[i] = ((i + seed * 137) as f64 * 0.618).fract() - 0.5;
            }

            let hv_dense = &h_dense * &v;
            let hv_lanczos = hamiltonian_matvec(&v, &hs, &complex, 1.0, None);

            let diff = (&hv_dense - &hv_lanczos).norm();
            assert!(diff < 1e-10, "matvec mismatch at seed {seed}: diff={diff}");
        }
    }
}
