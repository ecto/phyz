//! SU(2) lattice gauge theory at j_max = 1/2 (Z₂ reduction).
//!
//! At the lowest truncation j_max = 1/2, SU(2) gauge theory reduces to
//! a Z₂ gauge theory. Each edge carries j ∈ {0, 1/2}, represented as a
//! bit. Gauss law requires even parity (even number of j = 1/2 edges) at
//! each vertex.
//!
//! Hamiltonian:
//!
//!   H = (3g²/8) Σ_e n_e − (1/2g²) Σ_tri B_tri
//!
//! where:
//! - n_e ∈ {0, 1} is the j = 1/2 occupation of edge e
//! - B_tri flips all 3 edges of triangle tri (Z₂ plaquette operator)
//! - 3g²/8 = (g²/2) × j(j+1)|_{j=1/2}
//!
//! The gauge-invariant Hilbert space has dimension 2^b₁ where
//! b₁ = E − V + 1 is the first Betti number.

use crate::observables;
use crate::ryu_takayanagi;
use nalgebra::{DMatrix, DVector};
use phyz_regge::complex::SimplicialComplex;
use std::collections::HashMap;

/// SU(2) gauge-invariant Hilbert space at j_max = 1/2.
///
/// States are binary strings on edges, subject to even-parity Gauss law.
/// Dimension = 2^b₁ where b₁ = E − V + 1.
pub struct Su2HilbertSpace {
    /// Number of edges.
    pub n_edges: usize,
    /// Gauge-invariant basis states (as bitmasks).
    pub basis: Vec<u64>,
    /// Basis index lookup.
    index_map: HashMap<u64, usize>,
    /// Basis as Vec<Vec<i32>> for entropy computation.
    basis_vecs: Vec<Vec<i32>>,
}

impl Su2HilbertSpace {
    /// Build the gauge-invariant SU(2) Hilbert space at j_max = 1/2.
    pub fn new(complex: &SimplicialComplex) -> Self {
        let n_edges = complex.n_edges();
        assert!(n_edges <= 63, "Su2HilbertSpace supports up to 63 edges");

        // Build vertex-edge adjacency.
        let mut vertex_edges: Vec<Vec<usize>> = vec![Vec::new(); complex.n_vertices];
        for (ei, edge) in complex.edges.iter().enumerate() {
            vertex_edges[edge[0]].push(ei);
            vertex_edges[edge[1]].push(ei);
        }

        // Enumerate gauge-invariant states.
        let mut basis = Vec::new();
        for mask in 0u64..(1u64 << n_edges) {
            if satisfies_gauss_law(mask, &vertex_edges, complex.n_vertices) {
                basis.push(mask);
            }
        }

        let index_map: HashMap<u64, usize> = basis
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, i))
            .collect();

        let basis_vecs: Vec<Vec<i32>> = basis
            .iter()
            .map(|&s| bitmask_to_vec(s, n_edges))
            .collect();

        Self {
            n_edges,
            basis,
            index_map,
            basis_vecs,
        }
    }

    /// Hilbert space dimension.
    pub fn dim(&self) -> usize {
        self.basis.len()
    }

    /// Look up basis index for a state.
    pub fn config_to_index(&self, state: u64) -> Option<usize> {
        self.index_map.get(&state).copied()
    }

    /// Get basis as Vec<Vec<i32>> for entanglement entropy computation.
    pub fn basis_vecs(&self) -> &[Vec<i32>] {
        &self.basis_vecs
    }
}

/// Check Gauss law: even parity at each vertex.
fn satisfies_gauss_law(state: u64, vertex_edges: &[Vec<usize>], n_vertices: usize) -> bool {
    for v in 0..n_vertices {
        let parity: u32 = vertex_edges[v]
            .iter()
            .map(|&ei| ((state >> ei) & 1) as u32)
            .sum();
        if !parity.is_multiple_of(2) {
            return false;
        }
    }
    true
}

/// Convert bitmask to Vec<i32>.
fn bitmask_to_vec(state: u64, n_edges: usize) -> Vec<i32> {
    (0..n_edges)
        .map(|e| ((state >> e) & 1) as i32)
        .collect()
}

/// Build the SU(2) j_max = 1/2 Hamiltonian.
///
/// H = (3g²/8) Σ_e n_e − (1/2g²) Σ_tri B_tri
pub fn build_su2_hamiltonian(
    hilbert: &Su2HilbertSpace,
    complex: &SimplicialComplex,
    g_squared: f64,
) -> DMatrix<f64> {
    let dim = hilbert.dim();
    let mut h = DMatrix::zeros(dim, dim);

    // Electric term: (g²/2) × j(j+1)|_{j=1/2} × n_e = (3g²/8) × Σ n_e
    let e_coeff = 3.0 * g_squared / 8.0;
    for (i, &state) in hilbert.basis.iter().enumerate() {
        let n_excited = state.count_ones();
        h[(i, i)] += e_coeff * n_excited as f64;
    }

    // Magnetic term: −(1/2g²) × B_tri (plaquette flip)
    let b_coeff = -0.5 / g_squared;
    for ti in 0..complex.n_triangles() {
        let [e0, e1, e2] = complex.tri_edge_indices(ti);
        let flip_mask: u64 = (1 << e0) | (1 << e1) | (1 << e2);

        for (i, &state) in hilbert.basis.iter().enumerate() {
            let flipped = state ^ flip_mask;
            if let Some(j) = hilbert.config_to_index(flipped) {
                h[(i, j)] += b_coeff;
            }
        }
    }

    h
}

/// Entanglement entropy for a vertex partition using SU(2) states.
///
/// Uses the algebraic prescription: edges_a = edges with both endpoints in A.
pub fn su2_entanglement_for_partition(
    hilbert: &Su2HilbertSpace,
    state: &DVector<f64>,
    complex: &SimplicialComplex,
    partition_a: &[usize],
) -> f64 {
    let (edges_a, _, _) = ryu_takayanagi::classify_edges(complex, partition_a);
    observables::entanglement_entropy_raw(hilbert.basis_vecs(), hilbert.n_edges, state, &edges_a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag;

    fn single_pentachoron() -> SimplicialComplex {
        SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]])
    }

    #[test]
    fn test_su2_dimension() {
        let complex = single_pentachoron();
        let hs = Su2HilbertSpace::new(&complex);
        // b₁ = E - V + 1 = 10 - 5 + 1 = 6, dim = 2^6 = 64
        assert_eq!(hs.dim(), 64);
    }

    #[test]
    fn test_su2_2pent_dimension() {
        let complex =
            SimplicialComplex::from_pentachorons(6, &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]]);
        let hs = Su2HilbertSpace::new(&complex);
        let b1 = complex.n_edges() - complex.n_vertices + 1;
        assert_eq!(hs.dim(), 1 << b1);
    }

    #[test]
    fn test_su2_zero_state_present() {
        let complex = single_pentachoron();
        let hs = Su2HilbertSpace::new(&complex);
        assert!(hs.config_to_index(0).is_some());
    }

    #[test]
    fn test_su2_hamiltonian_symmetric() {
        let complex = single_pentachoron();
        let hs = Su2HilbertSpace::new(&complex);
        let h = build_su2_hamiltonian(&hs, &complex, 1.0);
        let diff = (&h - h.transpose()).norm();
        assert!(diff < 1e-12, "H not symmetric: diff={diff}");
    }

    #[test]
    fn test_su2_strong_coupling() {
        // Strong coupling: ground state ≈ |0,...,0⟩ with E₀ ≈ 0.
        let complex = single_pentachoron();
        let hs = Su2HilbertSpace::new(&complex);
        let h = build_su2_hamiltonian(&hs, &complex, 1e6);
        let spec = diag::diagonalize(&h, Some(1));
        assert!(
            spec.ground_energy().abs() < 1e-2,
            "E₀ = {}",
            spec.ground_energy()
        );
    }

    #[test]
    fn test_su2_entanglement_nonnegative() {
        let complex = single_pentachoron();
        let hs = Su2HilbertSpace::new(&complex);
        let h = build_su2_hamiltonian(&hs, &complex, 1.0);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        for part in ryu_takayanagi::vertex_bipartitions(5) {
            let s = su2_entanglement_for_partition(&hs, gs, &complex, &part);
            assert!(s >= -1e-10, "partition {:?}: S = {s} < 0", part);
        }
    }

    #[test]
    fn test_su2_entanglement_strong_coupling_zero() {
        // Strong coupling ground state ≈ product → S ≈ 0.
        let complex = single_pentachoron();
        let hs = Su2HilbertSpace::new(&complex);
        let h = build_su2_hamiltonian(&hs, &complex, 1e6);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        let part = vec![0, 1];
        let s = su2_entanglement_for_partition(&hs, gs, &complex, &part);
        assert!(s < 0.1, "strong coupling S = {s}, expected ~0");
    }

    #[test]
    fn test_su2_plaquette_preserves_gauss() {
        // Verify: flipping all 3 edges of any triangle preserves even parity.
        let complex = single_pentachoron();
        let hs = Su2HilbertSpace::new(&complex);

        for ti in 0..complex.n_triangles() {
            let [e0, e1, e2] = complex.tri_edge_indices(ti);
            let flip_mask: u64 = (1 << e0) | (1 << e1) | (1 << e2);

            for &state in &hs.basis {
                let flipped = state ^ flip_mask;
                assert!(
                    hs.config_to_index(flipped).is_some(),
                    "triangle {ti}: flipped state not in basis"
                );
            }
        }
    }
}
