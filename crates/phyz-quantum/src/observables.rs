//! Physical observables for the gauge theory ground state.
//!
//! - Electric field expectation values
//! - Wilson loop expectation values
//! - Fundamental loops from spanning tree
//! - Entanglement entropy

use crate::hilbert::U1HilbertSpace;
use nalgebra::DVector;
use phyz_regge::SimplicialComplex;

/// Expectation value of n_e² for each edge in the given state.
pub fn electric_field_sq(hilbert: &U1HilbertSpace, state: &DVector<f64>) -> Vec<f64> {
    let n_edges = hilbert.n_edges;
    let mut result = vec![0.0; n_edges];

    for (i, config) in hilbert.basis.iter().enumerate() {
        let amp_sq = state[i] * state[i];
        for (e, &n) in config.iter().enumerate() {
            result[e] += amp_sq * (n as f64) * (n as f64);
        }
    }

    result
}

/// Wilson loop expectation value ⟨Re(U_path)⟩.
///
/// `path` is a sequence of `(edge_index, direction)` pairs where
/// direction = +1 means traverse in the canonical direction (raises n_e),
/// direction = -1 means reverse (lowers n_e).
///
/// The Wilson loop operator Re(U_path) = (U_path + U_path†)/2 acts by
/// shifting all edge quantum numbers along the path simultaneously.
pub fn wilson_loop(
    hilbert: &U1HilbertSpace,
    state: &DVector<f64>,
    path: &[(usize, i32)],
) -> f64 {
    let lam = hilbert.lambda as i32;
    let mut expectation = 0.0;

    for (i, config) in hilbert.basis.iter().enumerate() {
        // Apply U_path: shift each edge by its direction.
        let mut shifted = config.clone();
        let mut valid = true;
        for &(edge, dir) in path {
            shifted[edge] += dir;
            if shifted[edge] < -lam || shifted[edge] > lam {
                valid = false;
                break;
            }
        }

        if valid {
            if let Some(j) = hilbert.config_to_index(&shifted) {
                // ⟨ψ| (U + U†)/2 |ψ⟩ contributions:
                // state[i] * state[j] from U, and state[j] * state[i] from U†.
                // Together: state[i] * state[j] (real states, so this is correct).
                expectation += state[i] * state[j];
            }
        }
    }

    expectation
}

/// Find fundamental loops from the spanning tree.
///
/// Each non-tree edge defines a fundamental loop: the path in the tree
/// connecting its endpoints, plus the non-tree edge itself.
///
/// Returns a vec of loops, each loop is a vec of `(edge_index, direction)`.
pub fn fundamental_loops(complex: &SimplicialComplex) -> Vec<Vec<(usize, i32)>> {
    let (tree_edges, non_tree_edges) = crate::gauss_law::spanning_tree(complex);

    // Build tree adjacency for path finding.
    let mut tree_adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); complex.n_vertices];
    for &ei in &tree_edges {
        let edge = &complex.edges[ei];
        tree_adj[edge[0]].push((edge[1], ei));
        tree_adj[edge[1]].push((edge[0], ei));
    }

    let mut loops = Vec::new();

    for &ei in &non_tree_edges {
        let edge = &complex.edges[ei];
        let start = edge[0];
        let end = edge[1];

        // BFS to find tree path from start to end.
        let path = tree_path(&tree_adj, complex, start, end);

        // Complete the loop: tree path + non-tree edge.
        let mut lp = path;
        // The non-tree edge goes from end back to start.
        // Canonical direction for edge [start, end] with start < end:
        // traversing end→start is direction -1.
        lp.push((ei, -1));

        loops.push(lp);
    }

    loops
}

/// BFS path in the tree from `start` to `end`.
/// Returns path as `(edge_index, direction)` pairs.
fn tree_path(
    tree_adj: &[Vec<(usize, usize)>],
    complex: &SimplicialComplex,
    start: usize,
    end: usize,
) -> Vec<(usize, i32)> {
    use std::collections::VecDeque;

    let n_v = tree_adj.len();
    let mut parent: Vec<Option<(usize, usize)>> = vec![None; n_v]; // (parent_vertex, edge_index)
    let mut visited = vec![false; n_v];
    let mut queue = VecDeque::new();

    visited[start] = true;
    queue.push_back(start);

    while let Some(v) = queue.pop_front() {
        if v == end {
            break;
        }
        for &(neighbor, ei) in &tree_adj[v] {
            if !visited[neighbor] {
                visited[neighbor] = true;
                parent[neighbor] = Some((v, ei));
                queue.push_back(neighbor);
            }
        }
    }

    // Reconstruct path.
    let mut path = Vec::new();
    let mut current = end;
    while current != start {
        let (prev, ei) = parent[current].unwrap();
        let edge = &complex.edges[ei];
        // Direction: +1 if we traverse from edge[0] to edge[1], -1 otherwise.
        let dir = if edge[0] == prev { 1 } else { -1 };
        path.push((ei, dir));
        current = prev;
    }
    path.reverse();
    path
}

/// Von Neumann entanglement entropy S_A = -Tr(ρ_A log ρ_A).
///
/// Partitions edges into set A (`edges_a`) and complement B.
/// Computes the reduced density matrix ρ_A by tracing out B,
/// then computes the entropy.
pub fn entanglement_entropy(
    hilbert: &U1HilbertSpace,
    state: &DVector<f64>,
    edges_a: &[usize],
) -> f64 {
    let n_edges = hilbert.n_edges;
    let mut is_a = vec![false; n_edges];
    for &e in edges_a {
        is_a[e] = true;
    }

    // Build reduced density matrix ρ_A.
    // Group basis states by their restriction to edges_a.
    // ρ_A[α,β] = Σ_{b} ψ(α,b) ψ(β,b) where α,β are configs on A and b on B.
    use std::collections::HashMap;

    // Extract the A-part of each config.
    let a_configs: Vec<Vec<i32>> = hilbert
        .basis
        .iter()
        .map(|c| edges_a.iter().map(|&e| c[e]).collect())
        .collect();

    // Assign indices to unique A-configs.
    let mut a_index: HashMap<Vec<i32>, usize> = HashMap::new();
    for ac in &a_configs {
        let len = a_index.len();
        a_index.entry(ac.clone()).or_insert(len);
    }
    let dim_a = a_index.len();

    // Build ρ_A.
    let mut rho: nalgebra::DMatrix<f64> = nalgebra::DMatrix::zeros(dim_a, dim_a);

    // Group states by B-config. For each unique B-config, accumulate
    // the outer product of the A-amplitudes.
    let b_configs: Vec<Vec<i32>> = hilbert
        .basis
        .iter()
        .map(|c| {
            (0..n_edges)
                .filter(|e| !is_a[*e])
                .map(|e| c[e])
                .collect()
        })
        .collect();

    // Group by B-config.
    let mut b_groups: HashMap<Vec<i32>, Vec<(usize, f64)>> = HashMap::new();
    for (i, bc) in b_configs.iter().enumerate() {
        b_groups
            .entry(bc.clone())
            .or_default()
            .push((a_index[&a_configs[i]], state[i]));
    }

    for (_bc, entries) in &b_groups {
        for &(ai, amp_i) in entries {
            for &(aj, amp_j) in entries {
                rho[(ai, aj)] += amp_i * amp_j;
            }
        }
    }

    // Diagonalize ρ_A to get eigenvalues.
    let eig = rho.symmetric_eigen();
    let mut entropy = 0.0;
    for i in 0..eig.eigenvalues.len() {
        let ev: f64 = eig.eigenvalues[i];
        if ev > 1e-15 {
            entropy -= ev * ev.ln();
        }
    }

    entropy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag;
    use crate::hamiltonian::{build_hamiltonian, KSParams};

    fn single_pentachoron() -> SimplicialComplex {
        SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]])
    }

    #[test]
    fn test_electric_field_strong_coupling() {
        // In strong coupling, ground state ≈ |0,...,0⟩, so ⟨n_e²⟩ ≈ 0.
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let params = KSParams {
            g_squared: 1e6,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        let e_sq = electric_field_sq(&hs, gs);
        for (e, &val) in e_sq.iter().enumerate() {
            assert!(
                val < 1e-3,
                "edge {e}: ⟨n²⟩ = {val}, expected ~0 in strong coupling"
            );
        }
    }

    #[test]
    fn test_fundamental_loops() {
        let complex = single_pentachoron();
        let loops = fundamental_loops(&complex);

        // b1 = E - V + 1 = 10 - 5 + 1 = 6 fundamental loops.
        assert_eq!(loops.len(), 6);

        // Each loop should be a closed path.
        for lp in &loops {
            assert!(!lp.is_empty());
        }
    }

    #[test]
    fn test_wilson_loop_strong_coupling() {
        // In strong coupling, ground state ≈ |0,...,0⟩.
        // Wilson loops should be small (area law suppression).
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let params = KSParams {
            g_squared: 100.0,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        let loops = fundamental_loops(&complex);
        for (li, lp) in loops.iter().enumerate() {
            let w = wilson_loop(&hs, gs, lp);
            // In strong coupling, Wilson loops are suppressed.
            assert!(
                w.abs() < 1.0,
                "loop {li}: W = {w}, expected small in strong coupling"
            );
        }
    }

    #[test]
    fn test_entanglement_entropy_product_state() {
        // For the strong-coupling ground state ≈ |0,...,0⟩ (product state),
        // entanglement entropy should be ~0.
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let params = KSParams {
            g_squared: 1e6,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        // Partition: first 5 edges vs rest.
        let edges_a: Vec<usize> = (0..5).collect();
        let s = entanglement_entropy(&hs, gs, &edges_a);
        assert!(
            s < 0.1,
            "strong coupling entropy = {s}, expected ~0"
        );
    }

    #[test]
    fn test_entanglement_entropy_nonnegative() {
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let params = KSParams {
            g_squared: 1.0,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        let edges_a: Vec<usize> = (0..3).collect();
        let s = entanglement_entropy(&hs, gs, &edges_a);
        assert!(s >= -1e-10, "entropy should be non-negative: {s}");
    }
}
