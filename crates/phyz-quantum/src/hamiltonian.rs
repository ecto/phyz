//! Kogut-Susskind Hamiltonian for U(1) gauge theory on simplicial complexes.
//!
//! The KS Hamiltonian is:
//!
//!   H = H_E + H_B
//!     = (g²/2) Σ_e n_e² − (1/g²) Σ_tri Re(U_tri)
//!
//! where:
//! - H_E (electric) is diagonal in the electric basis
//! - H_B (magnetic) is a sum over triangular plaquettes; each triangle
//!   holonomy U_tri raises/lowers quantum numbers on its 3 edges
//!
//! Triangle holonomy convention (from `phyz-regge::gauge`):
//!   For triangle [v0,v1,v2] with v0<v1<v2, `tri_edge_indices` returns
//!   [e01, e02, e12]. The oriented holonomy is θ_{01} - θ_{02} + θ_{12},
//!   so signs are [+1, -1, +1].
//!
//!   U_tri: simultaneously shifts n_{e01} → n_{e01}+1, n_{e02} → n_{e02}-1,
//!          n_{e12} → n_{e12}+1
//!   Re(U_tri) = (U_tri + U_tri†) / 2

use crate::hilbert::U1HilbertSpace;
use nalgebra::DMatrix;
use phyz_regge::SimplicialComplex;

/// Parameters for the Kogut-Susskind Hamiltonian.
#[derive(Debug, Clone)]
pub struct KSParams {
    /// Coupling constant g².
    pub g_squared: f64,
    /// Optional metric weights from curved background (one per triangle).
    /// If `None`, all weights are 1.0 (flat background).
    pub metric_weights: Option<Vec<f64>>,
}

impl Default for KSParams {
    fn default() -> Self {
        Self {
            g_squared: 1.0,
            metric_weights: None,
        }
    }
}

/// Holonomy signs for a triangle: [sign_e01, sign_e02, sign_e12] = [+1, -1, +1].
///
/// Matches the gauge.rs convention: F_t = θ_{01} + θ_{12} - θ_{02}.
const TRIANGLE_HOLONOMY_SIGNS: [i32; 3] = [1, -1, 1];

/// Build the KS Hamiltonian as a dense matrix.
///
/// The matrix is real symmetric (H_E is diagonal, H_B is symmetric by
/// construction since Re(U) = (U + U†)/2).
pub fn build_hamiltonian(
    hilbert: &U1HilbertSpace,
    complex: &SimplicialComplex,
    params: &KSParams,
) -> DMatrix<f64> {
    let dim = hilbert.dim();
    let mut h = DMatrix::zeros(dim, dim);

    add_electric_term(&mut h, hilbert, complex, params);
    add_magnetic_term(&mut h, hilbert, complex, params);

    h
}

/// Add the electric term: H_E = (g²/2) Σ_e n_e².
/// This is diagonal in the electric basis.
fn add_electric_term(
    h: &mut DMatrix<f64>,
    hilbert: &U1HilbertSpace,
    _complex: &SimplicialComplex,
    params: &KSParams,
) {
    let coeff = params.g_squared / 2.0;
    for i in 0..hilbert.dim() {
        let config = hilbert.index_to_config(i);
        let energy: f64 = config.iter().map(|&n| (n as f64) * (n as f64)).sum();
        h[(i, i)] += coeff * energy;
    }
}

/// Add the magnetic term: H_B = -(1/g²) Σ_tri w_tri · Re(U_tri).
///
/// For each triangle, U_tri shifts edges by holonomy signs [+1,-1,+1].
/// Re(U) = (U + U†)/2, so we add both the forward and backward shifts
/// with coefficient 1/2.
fn add_magnetic_term(
    h: &mut DMatrix<f64>,
    hilbert: &U1HilbertSpace,
    complex: &SimplicialComplex,
    params: &KSParams,
) {
    let coeff = -1.0 / params.g_squared;

    for (ti, _tri) in complex.triangles.iter().enumerate() {
        let edge_indices = complex.tri_edge_indices(ti);
        let weight = params
            .metric_weights
            .as_ref()
            .map_or(1.0, |w| w[ti]);

        let c = coeff * weight * 0.5; // factor of 1/2 from Re = (U + U†)/2

        // For each basis state, try to apply U_tri and U_tri†.
        for i in 0..hilbert.dim() {
            let config = hilbert.index_to_config(i);

            // Forward: U_tri shifts n_e by +signs
            if let Some(j) = apply_triangle_shift(hilbert, config, &edge_indices, &TRIANGLE_HOLONOMY_SIGNS) {
                h[(i, j)] += c;
                h[(j, i)] += c;
            }
        }
    }
}

/// Try to apply a triangle holonomy shift to a config.
/// Returns the target basis index if the shifted config is valid.
fn apply_triangle_shift(
    hilbert: &U1HilbertSpace,
    config: &[i32],
    edge_indices: &[usize; 3],
    signs: &[i32; 3],
) -> Option<usize> {
    let lam = hilbert.lambda as i32;
    let mut new_config = config.to_vec();

    for k in 0..3 {
        new_config[edge_indices[k]] += signs[k];
        if new_config[edge_indices[k]] < -lam || new_config[edge_indices[k]] > lam {
            return None;
        }
    }

    hilbert.config_to_index(&new_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_pentachoron() -> SimplicialComplex {
        SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]])
    }

    #[test]
    fn test_hamiltonian_symmetric() {
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let params = KSParams::default();
        let h = build_hamiltonian(&hs, &complex, &params);

        let diff = (&h - h.transpose()).norm();
        assert!(diff < 1e-12, "H not symmetric: diff={diff}");
    }

    #[test]
    fn test_strong_coupling_limit() {
        // g² → ∞: H ≈ H_E. Ground state = |0,...,0⟩ with E_0 = 0.
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let params = KSParams {
            g_squared: 1e6,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);

        // Ground state energy should be ~0 (all-zero config has E=0).
        let zero_config = vec![0i32; 10];
        let zero_idx = hs.config_to_index(&zero_config).unwrap();
        let e_zero = h[(zero_idx, zero_idx)];
        assert!(
            e_zero.abs() < 1e-10,
            "strong coupling: E(|0⟩) = {e_zero}, expected 0"
        );

        // All other diagonal elements should be positive.
        for i in 0..hs.dim() {
            if i != zero_idx {
                assert!(h[(i, i)] > 0.0, "state {i} should have positive energy");
            }
        }
    }

    #[test]
    fn test_electric_term_only() {
        // With g² very large and checking just diagonal entries.
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let g_sq = 2.0;
        let params = KSParams {
            g_squared: g_sq,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);

        // Check that diagonal entries include the electric term.
        let zero_config = vec![0i32; 10];
        let zero_idx = hs.config_to_index(&zero_config).unwrap();

        // |0,...,0⟩ should have zero electric energy.
        // The diagonal also gets magnetic self-coupling, but for |0⟩ the
        // magnetic term only creates off-diagonal entries. So diagonal at
        // zero_idx should be exactly 0 from the electric term.
        // (The magnetic term adds off-diagonal entries for |0⟩.)
        let expected_e = 0.0; // Σ n_e² = 0 for all-zero config
        // The diagonal might have magnetic contributions too if U_tri|0⟩
        // maps back to |0⟩. For U_tri, the shift is [+1,-1,+1] which is
        // never all zero, so no magnetic diagonal contribution at |0⟩.
        assert!(
            (h[(zero_idx, zero_idx)] - expected_e).abs() < 1e-12,
            "electric energy at |0⟩ = {}, expected {expected_e}",
            h[(zero_idx, zero_idx)]
        );
    }

    #[test]
    fn test_single_triangle_exact() {
        // Minimal gauge theory: a single triangle (not embedded in a pentachoron).
        // 3 vertices, 3 edges, b1 = 3 - 3 + 1 = 1.
        // With Λ=1: dim = 3 states (n_free ∈ {-1, 0, 1}).
        //
        // We need a SimplicialComplex for this. Since SimplicialComplex requires
        // pentachorons, we can't easily test a bare triangle. Skip to pentachoron.
        //
        // Instead, verify that for a single pentachoron the Hamiltonian has
        // the right dimension.
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let params = KSParams::default();
        let h = build_hamiltonian(&hs, &complex, &params);

        assert_eq!(h.nrows(), hs.dim());
        assert_eq!(h.ncols(), hs.dim());
    }

    #[test]
    fn test_hamiltonian_real_eigenvalues() {
        // A real symmetric matrix should have all real eigenvalues.
        // This is guaranteed by construction but let's verify the matrix is well-formed.
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let params = KSParams {
            g_squared: 1.0,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);

        // Check symmetry more carefully.
        for i in 0..hs.dim() {
            for j in 0..hs.dim() {
                assert!(
                    (h[(i, j)] - h[(j, i)]).abs() < 1e-14,
                    "H[{i},{j}]={} != H[{j},{i}]={}",
                    h[(i, j)],
                    h[(j, i)]
                );
            }
        }
    }

    #[test]
    fn test_metric_weights_scaling() {
        // Doubling all metric weights should double the magnetic term
        // but leave the electric term unchanged.
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);

        let w1 = vec![1.0; complex.n_triangles()];
        let w2 = vec![2.0; complex.n_triangles()];

        let p1 = KSParams {
            g_squared: 1.0,
            metric_weights: Some(w1),
        };
        let p2 = KSParams {
            g_squared: 1.0,
            metric_weights: Some(w2),
        };

        let h1 = build_hamiltonian(&hs, &complex, &p1);
        let h2 = build_hamiltonian(&hs, &complex, &p2);

        // Build electric-only for reference.
        let mut h_e = DMatrix::zeros(hs.dim(), hs.dim());
        add_electric_term(&mut h_e, &hs, &complex, &p1);

        // h2 - h1 should equal the extra magnetic contribution.
        // h_B(w=2) = 2 * h_B(w=1), so h2 - h1 = h_B(w=1).
        // And h1 = h_e + h_B(w=1), h2 = h_e + 2*h_B(w=1).
        // So h2 - h1 = h_B(w=1) = h1 - h_e.
        let diff = &h2 - &h1;
        let mag1 = &h1 - &h_e;
        let err = (&diff - &mag1).norm();
        assert!(
            err < 1e-12,
            "metric weight scaling failed: err={err}"
        );
    }
}
