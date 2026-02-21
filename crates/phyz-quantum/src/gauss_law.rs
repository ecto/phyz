//! Gauss law constraint and gauge-invariant basis enumeration.
//!
//! For U(1) lattice gauge theory in the electric basis, states are labeled
//! by integer quantum numbers n_e ∈ [-Λ, Λ] on each edge. Gauss law at
//! vertex v requires:
//!
//!   Σ_{e: v∈e} σ(v,e) · n_e = 0
//!
//! where σ(v,e) = +1 if v is the higher-index endpoint, -1 if lower.
//! This matches the gauge generator convention in `phyz-regge::gauge`.
//!
//! We reduce the Hilbert space by choosing a spanning tree of the 1-skeleton.
//! Tree edges are determined by Gauss law from the free (non-tree) edges.
//! Only configurations where all tree edges also fall within [-Λ, Λ] survive.

use phyz_regge::SimplicialComplex;
use std::collections::VecDeque;

/// Vertex-edge adjacency with orientation signs.
///
/// Returns `adj[v]` = vec of `(edge_index, sign)` where:
/// - sign = +1 if `v == edge[1]` (v is higher-index endpoint)
/// - sign = -1 if `v == edge[0]` (v is lower-index endpoint)
pub fn vertex_edge_adjacency(complex: &SimplicialComplex) -> Vec<Vec<(usize, i32)>> {
    let mut adj = vec![Vec::new(); complex.n_vertices];
    for (ei, edge) in complex.edges.iter().enumerate() {
        adj[edge[0]].push((ei, -1));
        adj[edge[1]].push((ei, 1));
    }
    adj
}

/// BFS spanning tree of the 1-skeleton.
///
/// Returns `(tree_edge_indices, non_tree_edge_indices)`.
/// The tree edges connect all vertices (assuming the complex is connected).
pub fn spanning_tree(complex: &SimplicialComplex) -> (Vec<usize>, Vec<usize>) {
    let adj = vertex_edge_adjacency(complex);
    let n_v = complex.n_vertices;
    let mut visited = vec![false; n_v];
    let mut tree_edges = Vec::new();
    let mut is_tree = vec![false; complex.n_edges()];

    let mut queue = VecDeque::new();
    visited[0] = true;
    queue.push_back(0);

    while let Some(v) = queue.pop_front() {
        for &(ei, _sign) in &adj[v] {
            let edge = &complex.edges[ei];
            let other = if edge[0] == v { edge[1] } else { edge[0] };
            if !visited[other] {
                visited[other] = true;
                tree_edges.push(ei);
                is_tree[ei] = true;
                queue.push_back(other);
            }
        }
    }

    let non_tree: Vec<usize> = (0..complex.n_edges()).filter(|&ei| !is_tree[ei]).collect();
    (tree_edges, non_tree)
}

/// Enumerate all gauge-invariant configurations.
///
/// Assigns free (non-tree) edges all values in `[-lambda, lambda]`,
/// then solves for tree edges via Gauss law (leaf peeling).
/// Configs where any tree edge exceeds `[-lambda, lambda]` are discarded.
///
/// Returns a vec of configs, each of length `n_edges`.
pub fn enumerate_gauge_invariant(complex: &SimplicialComplex, lambda: u32) -> Vec<Vec<i32>> {
    let adj = vertex_edge_adjacency(complex);
    let (tree_edges, non_tree_edges) = spanning_tree(complex);
    let n_edges = complex.n_edges();
    let n_free = non_tree_edges.len();
    let lam = lambda as i32;
    let base = 2 * lam + 1;

    let mut is_tree = vec![false; n_edges];
    for &ei in &tree_edges {
        is_tree[ei] = true;
    }

    // Precompute tree-solving order: peel leaves from the tree.
    // At each step, find a vertex with exactly one unsolved tree edge,
    // solve that edge, mark it done.
    let solve_order = tree_solve_order(complex, &tree_edges, &adj);

    // Total number of free-edge configs.
    let n_configs: u64 = (base as u64).pow(n_free as u32);

    let mut basis = Vec::new();

    for ci in 0..n_configs {
        let mut config = vec![0i32; n_edges];

        // Assign free edges via mixed-radix decoding.
        let mut rem = ci;
        for &fi in &non_tree_edges {
            let digit = (rem % base as u64) as i32;
            config[fi] = digit - lam;
            rem /= base as u64;
        }

        // Solve tree edges via Gauss law.
        if solve_tree_edges(&adj, &solve_order, &mut config, lam) {
            basis.push(config);
        }
    }

    basis
}

/// Determine the order in which to solve tree edges (leaf peeling).
///
/// Returns vec of `(vertex, edge_index)` pairs: at each step, `vertex`
/// has exactly one remaining unsolved tree edge `edge_index`, and
/// Gauss law at that vertex determines the edge's value.
fn tree_solve_order(
    complex: &SimplicialComplex,
    tree_edges: &[usize],
    adj: &[Vec<(usize, i32)>],
) -> Vec<(usize, usize)> {
    let n_v = complex.n_vertices;
    let n_edges = complex.n_edges();

    let mut is_tree = vec![false; n_edges];
    for &ei in tree_edges {
        is_tree[ei] = true;
    }

    // Count unsolved tree edges per vertex.
    let mut tree_degree = vec![0u32; n_v];
    for v in 0..n_v {
        for &(ei, _) in &adj[v] {
            if is_tree[ei] {
                tree_degree[v] += 1;
            }
        }
    }

    let mut solved = vec![false; n_edges];
    let mut order = Vec::with_capacity(tree_edges.len());
    let mut queue = VecDeque::new();

    // Seed with leaves (vertices with exactly 1 tree edge).
    for v in 0..n_v {
        if tree_degree[v] == 1 {
            queue.push_back(v);
        }
    }

    while let Some(v) = queue.pop_front() {
        // Find the one unsolved tree edge at v.
        let mut found = None;
        for &(ei, _) in &adj[v] {
            if is_tree[ei] && !solved[ei] {
                found = Some(ei);
                break;
            }
        }
        let Some(ei) = found else { continue };
        if solved[ei] {
            continue;
        }

        solved[ei] = true;
        order.push((v, ei));

        // Update neighbor's tree degree.
        let edge = &complex.edges[ei];
        let other = if edge[0] == v { edge[1] } else { edge[0] };
        tree_degree[other] -= 1;
        if tree_degree[other] == 1 {
            queue.push_back(other);
        }
    }

    order
}

/// Solve tree edges given free-edge assignments. Returns false if any
/// tree edge exceeds the truncation bound.
fn solve_tree_edges(
    adj: &[Vec<(usize, i32)>],
    solve_order: &[(usize, usize)],
    config: &mut [i32],
    lam: i32,
) -> bool {
    for &(v, target_ei) in solve_order {
        // Gauss law at v: Σ σ(v,e) · n_e = 0
        // n_{target} = -(Σ_{e ≠ target} σ(v,e) · n_e) / σ(v,target)
        let mut sum = 0i32;
        let mut target_sign = 0i32;
        for &(ei, sign) in &adj[v] {
            if ei == target_ei {
                target_sign = sign;
            } else {
                sum += sign * config[ei];
            }
        }
        debug_assert_ne!(target_sign, 0);
        // n_target = -sum / target_sign. Since target_sign is ±1:
        let n_target = -sum * target_sign;
        if n_target < -lam || n_target > lam {
            return false;
        }
        config[target_ei] = n_target;
    }
    true
}

/// Verify that a configuration satisfies Gauss law at every vertex.
pub fn check_gauss_law(complex: &SimplicialComplex, config: &[i32]) -> bool {
    let adj = vertex_edge_adjacency(complex);
    for v in 0..complex.n_vertices {
        let mut sum = 0i32;
        for &(ei, sign) in &adj[v] {
            sum += sign * config[ei];
        }
        if sum != 0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_pentachoron() -> SimplicialComplex {
        SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]])
    }

    #[test]
    fn test_vertex_edge_adjacency() {
        let complex = single_pentachoron();
        let adj = vertex_edge_adjacency(&complex);

        // 5 vertices, each connected to 4 others = 4 edges per vertex.
        assert_eq!(adj.len(), 5);
        for v_adj in &adj {
            assert_eq!(v_adj.len(), 4);
        }

        // Check sign convention: for edge [i,j] with i<j,
        // vertex i gets -1, vertex j gets +1.
        for (ei, edge) in complex.edges.iter().enumerate() {
            let i = edge[0];
            let j = edge[1];
            assert!(adj[i].contains(&(ei, -1)));
            assert!(adj[j].contains(&(ei, 1)));
        }
    }

    #[test]
    fn test_spanning_tree() {
        let complex = single_pentachoron();
        let (tree, non_tree) = spanning_tree(&complex);

        // Spanning tree of 5 vertices has 4 edges.
        assert_eq!(tree.len(), 4);
        assert_eq!(non_tree.len(), 6); // 10 - 4 = 6 = b1

        // No duplicates.
        let mut all: Vec<usize> = tree.iter().chain(non_tree.iter()).copied().collect();
        all.sort();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_gauss_law_satisfaction() {
        let complex = single_pentachoron();
        let basis = enumerate_gauge_invariant(&complex, 1);

        assert!(!basis.is_empty());
        for config in &basis {
            assert!(
                check_gauss_law(&complex, config),
                "Gauss law violated: {config:?}"
            );
        }
    }

    #[test]
    fn test_basis_dimension_pentachoron_l1() {
        // Single pentachoron: V=5, E=10, b1=6.
        // Upper bound: 3^6 = 729 (before tree-edge filtering).
        // Actual dimension should be less or equal.
        let complex = single_pentachoron();
        let basis = enumerate_gauge_invariant(&complex, 1);

        assert!(basis.len() <= 729);
        assert!(basis.len() > 0);

        // All-zero state must be in the basis.
        let zero = vec![0i32; 10];
        assert!(basis.contains(&zero));
    }

    #[test]
    fn test_basis_no_duplicates() {
        let complex = single_pentachoron();
        let basis = enumerate_gauge_invariant(&complex, 1);

        let mut sorted = basis.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), basis.len(), "duplicate basis states found");
    }

    #[test]
    fn test_two_pentachorons() {
        let complex = SimplicialComplex::from_pentachorons(6, &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]]);
        let basis = enumerate_gauge_invariant(&complex, 1);

        // V=6, E=?, b1=E-V+1. All states should satisfy Gauss law.
        for config in &basis {
            assert!(check_gauss_law(&complex, config));
        }
    }
}
