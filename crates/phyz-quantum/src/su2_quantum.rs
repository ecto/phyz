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
    ///
    /// Uses cycle basis generation: gauge-invariant states are exactly the cycle
    /// space of the graph over Z₂, so we enumerate all 2^b₁ XOR combinations of
    /// fundamental cycles. This is O(2^b₁) instead of O(2^E).
    pub fn new(complex: &SimplicialComplex) -> Self {
        let n_edges = complex.n_edges();
        assert!(n_edges <= 63, "Su2HilbertSpace supports up to 63 edges");

        let basis = cycle_basis_enumerate(complex);

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

/// Convert bitmask to Vec<i32>.
fn bitmask_to_vec(state: u64, n_edges: usize) -> Vec<i32> {
    (0..n_edges)
        .map(|e| ((state >> e) & 1) as i32)
        .collect()
}

/// Enumerate gauge-invariant states via cycle basis.
///
/// The gauge-invariant Hilbert space (even parity at each vertex) is exactly
/// the cycle space of the graph over Z₂. We find fundamental cycles from a
/// spanning tree, then enumerate all 2^b₁ XOR combinations.
fn cycle_basis_enumerate(complex: &SimplicialComplex) -> Vec<u64> {
    let n_v = complex.n_vertices;
    let n_e = complex.n_edges();

    // Build adjacency: for each vertex, list of (neighbor, edge_index).
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_v];
    for (ei, edge) in complex.edges.iter().enumerate() {
        adj[edge[0]].push((edge[1], ei));
        adj[edge[1]].push((edge[0], ei));
    }

    // BFS spanning tree from vertex 0.
    let mut parent: Vec<Option<(usize, usize)>> = vec![None; n_v]; // (parent_vertex, edge_index)
    let mut visited = vec![false; n_v];
    let mut queue = std::collections::VecDeque::new();
    visited[0] = true;
    queue.push_back(0);
    let mut in_tree = vec![false; n_e];

    while let Some(v) = queue.pop_front() {
        for &(u, ei) in &adj[v] {
            if !visited[u] {
                visited[u] = true;
                parent[u] = Some((v, ei));
                in_tree[ei] = true;
                queue.push_back(u);
            }
        }
    }

    // Cotree edges generate fundamental cycles.
    let mut cycle_masks: Vec<u64> = Vec::new();
    for ei in 0..n_e {
        if in_tree[ei] {
            continue;
        }
        // Cotree edge ei connects endpoints of complex.edges[ei].
        let (a, b) = (complex.edges[ei][0], complex.edges[ei][1]);

        // Find path from a to root.
        let path_a = path_to_root(a, &parent);
        let path_b = path_to_root(b, &parent);

        // Cycle = cotree edge + symmetric difference of paths to LCA.
        let mut mask: u64 = 1 << ei; // the cotree edge itself
        // Find LCA: walk both paths, mark edges in symmetric difference.
        let set_a: std::collections::HashSet<usize> = path_a.iter().copied().collect();
        let set_b: std::collections::HashSet<usize> = path_b.iter().copied().collect();
        for &v in &path_a {
            if !set_b.contains(&v) {
                // edge from v to parent(v) is in the cycle
                if let Some((_, pe)) = parent[v] {
                    mask |= 1 << pe;
                }
            }
        }
        for &v in &path_b {
            if !set_a.contains(&v) {
                if let Some((_, pe)) = parent[v] {
                    mask |= 1 << pe;
                }
            }
        }
        cycle_masks.push(mask);
    }

    let b1 = cycle_masks.len();
    // Enumerate all 2^b₁ XOR combinations.
    let mut basis = Vec::with_capacity(1 << b1);
    for bits in 0u64..(1u64 << b1) {
        let mut state = 0u64;
        for (j, cm) in cycle_masks.iter().enumerate() {
            if (bits >> j) & 1 == 1 {
                state ^= cm;
            }
        }
        basis.push(state);
    }

    basis.sort_unstable();
    basis
}

/// Walk from vertex v to root via parent pointers, returning vertices visited.
fn path_to_root(mut v: usize, parent: &[Option<(usize, usize)>]) -> Vec<usize> {
    let mut path = vec![v];
    while let Some((p, _)) = parent[v] {
        path.push(p);
        v = p;
    }
    path
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

/// SU(2) j=1/2 Wilson loop expectation value.
///
/// For the Z₂ reduction, the Wilson loop is diagonal:
/// W|s⟩ = (-1)^(popcount(s & mask)) |s⟩
/// where mask = Σ (1 << e) for edges in the loop.
///
/// Returns ⟨ψ|W|ψ⟩ = Σ_i |ψ_i|² × (-1)^(popcount(basis[i] & mask)).
pub fn su2_wilson_loop(
    hilbert: &Su2HilbertSpace,
    state: &DVector<f64>,
    loop_edges: &[usize],
) -> f64 {
    let mask: u64 = loop_edges.iter().fold(0u64, |m, &e| m | (1 << e));
    let mut expectation = 0.0;
    for (i, &basis_state) in hilbert.basis.iter().enumerate() {
        let sign = if (basis_state & mask).count_ones() % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        expectation += state[i] * state[i] * sign;
    }
    expectation
}

/// Fundamental loops for SU(2) j=1/2 (edge indices only).
///
/// Same topology as [`crate::observables::fundamental_loops`] but returns
/// just edge indices without direction signs (not needed for Z₂).
pub fn su2_fundamental_loops(complex: &SimplicialComplex) -> Vec<Vec<usize>> {
    crate::observables::fundamental_loops(complex)
        .into_iter()
        .map(|lp| lp.into_iter().map(|(ei, _dir)| ei).collect())
        .collect()
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

    #[test]
    fn test_su2_wilson_loop_zero_state() {
        let complex = single_pentachoron();
        let hs = Su2HilbertSpace::new(&complex);

        // Pure |0...0⟩ state: popcount(0 & anything) = 0, so W = 1.
        let zero_idx = hs.config_to_index(0).unwrap();
        let mut state = DVector::zeros(hs.dim());
        state[zero_idx] = 1.0;

        let loops = su2_fundamental_loops(&complex);
        for (li, lp) in loops.iter().enumerate() {
            let w = su2_wilson_loop(&hs, &state, lp);
            assert!(
                (w - 1.0).abs() < 1e-12,
                "loop {li}: W = {w}, expected 1.0 for zero state"
            );
        }
    }

    #[test]
    fn test_su2_wilson_loop_bounded() {
        let complex = single_pentachoron();
        let hs = Su2HilbertSpace::new(&complex);
        let h = build_su2_hamiltonian(&hs, &complex, 1.0);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        let loops = su2_fundamental_loops(&complex);
        for (li, lp) in loops.iter().enumerate() {
            let w = su2_wilson_loop(&hs, gs, lp);
            assert!(
                w >= -1.0 - 1e-12 && w <= 1.0 + 1e-12,
                "loop {li}: W = {w}, expected ∈ [-1, 1]"
            );
        }
    }
}
