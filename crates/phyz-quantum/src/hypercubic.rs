//! Kogut-Susskind Hamiltonian on hypercubic lattices for comparison.
//!
//! Standard formulation with **square** plaquettes and 4-edge holonomies.
//! This serves as the baseline to compare against the simplicial (triangular
//! plaquette) formulation.
//!
//! We build a small periodic hypercubic lattice directly (no SimplicialComplex
//! needed) with square plaquettes defined by pairs of coordinate axes.

use crate::diag;
use phyz_math::DMat;
use std::collections::HashMap;

/// A small periodic hypercubic lattice for KS Hamiltonian construction.
#[derive(Debug, Clone)]
pub struct HypercubicLattice {
    /// Grid size along each dimension.
    pub n: usize,
    /// Number of spatial dimensions.
    pub dim: usize,
    /// Number of vertices (n^dim).
    pub n_vertices: usize,
    /// Edges: (vertex_a, vertex_b, axis).
    pub edges: Vec<(usize, usize, usize)>,
    /// Square plaquettes: [edge0, edge1, edge2, edge3] with signs [+1,+1,-1,-1].
    /// The holonomy goes: +axis1, +axis2, -axis1, -axis2.
    pub plaquettes: Vec<([usize; 4], [i32; 4])>,
}

impl HypercubicLattice {
    /// Build a periodic d-dimensional hypercubic lattice with n sites per axis.
    pub fn new(n: usize, dim: usize) -> Self {
        let n_vertices = n.pow(dim as u32);

        // Build edges: one edge per (vertex, axis) pair.
        // Edge goes from vertex v to v + e_axis (periodic).
        let mut edges = Vec::new();
        let mut edge_map: HashMap<(usize, usize), usize> = HashMap::new();

        for v in 0..n_vertices {
            for axis in 0..dim {
                let neighbor = shift_vertex(v, axis, 1, n, dim);
                let (a, b) = if v < neighbor { (v, neighbor) } else { (neighbor, v) };
                let len = edges.len();
                edge_map.entry((a, b)).or_insert_with(|| {
                    edges.push((a, b, axis));
                    len
                });
            }
        }

        // Build plaquettes: one per (vertex, axis1, axis2) with axis1 < axis2.
        // Holonomy: v→v+e1 (+axis1), v+e1→v+e1+e2 (+axis2),
        //           v+e1+e2→v+e2 (-axis1), v+e2→v (-axis2).
        let mut plaquettes = Vec::new();

        for v in 0..n_vertices {
            for a1 in 0..dim {
                for a2 in (a1 + 1)..dim {
                    let v_a1 = shift_vertex(v, a1, 1, n, dim);
                    let v_a2 = shift_vertex(v, a2, 1, n, dim);
                    let v_a1a2 = shift_vertex(v_a1, a2, 1, n, dim);

                    // Four edges of the plaquette.
                    let e0 = lookup_edge(&edge_map, v, v_a1);
                    let e1 = lookup_edge(&edge_map, v_a1, v_a1a2);
                    let e2 = lookup_edge(&edge_map, v_a2, v_a1a2);
                    let e3 = lookup_edge(&edge_map, v, v_a2);

                    // Signs: direction of traversal relative to canonical edge orientation.
                    let s0 = if v < v_a1 { 1 } else { -1 };
                    let s1 = if v_a1 < v_a1a2 { 1 } else { -1 };
                    let s2 = if v_a2 < v_a1a2 { -1 } else { 1 }; // reverse
                    let s3 = if v < v_a2 { -1 } else { 1 }; // reverse

                    plaquettes.push(([e0, e1, e2, e3], [s0, s1, s2, s3]));
                }
            }
        }

        Self {
            n,
            dim,
            n_vertices,
            edges,
            plaquettes,
        }
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn n_plaquettes(&self) -> usize {
        self.plaquettes.len()
    }
}

fn shift_vertex(v: usize, axis: usize, delta: usize, n: usize, dim: usize) -> usize {
    let mut coords = vec![0usize; dim];
    let mut rem = v;
    for d in 0..dim {
        coords[d] = rem % n;
        rem /= n;
    }
    coords[axis] = (coords[axis] + delta) % n;
    let mut result = 0;
    let mut factor = 1;
    for d in 0..dim {
        result += coords[d] * factor;
        factor *= n;
    }
    result
}

fn lookup_edge(edge_map: &HashMap<(usize, usize), usize>, a: usize, b: usize) -> usize {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    edge_map[&(lo, hi)]
}

/// Gauge-invariant Hilbert space for the hypercubic lattice.
///
/// Same spanning-tree approach as the simplicial version.
pub struct HypercubicHilbert {
    pub n_edges: usize,
    pub lambda: u32,
    pub basis: Vec<Vec<i32>>,
    index_map: HashMap<Vec<i32>, usize>,
}

impl HypercubicHilbert {
    pub fn new(lattice: &HypercubicLattice, lambda: u32) -> Self {
        let n_edges = lattice.n_edges();
        let n_v = lattice.n_vertices;
        let lam = lambda as i32;
        let base = 2 * lam + 1;

        // Build adjacency.
        let mut adj: Vec<Vec<(usize, i32)>> = vec![Vec::new(); n_v];
        for (ei, &(a, b, _)) in lattice.edges.iter().enumerate() {
            adj[a].push((ei, -1)); // a is lower
            adj[b].push((ei, 1));  // b is higher
        }

        // BFS spanning tree.
        let mut visited = vec![false; n_v];
        let mut is_tree = vec![false; n_edges];
        let mut tree_edges = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        visited[0] = true;
        queue.push_back(0);
        while let Some(v) = queue.pop_front() {
            for &(ei, _) in &adj[v] {
                let (a, b, _) = lattice.edges[ei];
                let other = if a == v { b } else { a };
                if !visited[other] {
                    visited[other] = true;
                    is_tree[ei] = true;
                    tree_edges.push(ei);
                    queue.push_back(other);
                }
            }
        }

        let non_tree: Vec<usize> = (0..n_edges).filter(|ei| !is_tree[*ei]).collect();
        let n_free = non_tree.len();

        // Leaf-peeling solve order.
        let mut tree_degree = vec![0u32; n_v];
        for v in 0..n_v {
            for &(ei, _) in &adj[v] {
                if is_tree[ei] {
                    tree_degree[v] += 1;
                }
            }
        }
        let mut solved = vec![false; n_edges];
        let mut solve_order = Vec::new();
        let mut peel_queue = std::collections::VecDeque::new();
        for v in 0..n_v {
            if tree_degree[v] == 1 {
                peel_queue.push_back(v);
            }
        }
        while let Some(v) = peel_queue.pop_front() {
            let mut found = None;
            for &(ei, _) in &adj[v] {
                if is_tree[ei] && !solved[ei] {
                    found = Some(ei);
                    break;
                }
            }
            let Some(ei) = found else { continue };
            if solved[ei] { continue; }
            solved[ei] = true;
            solve_order.push((v, ei));
            let (a, b, _) = lattice.edges[ei];
            let other = if a == v { b } else { a };
            tree_degree[other] -= 1;
            if tree_degree[other] == 1 {
                peel_queue.push_back(other);
            }
        }

        // Enumerate gauge-invariant configs.
        let n_configs = (base as u64).pow(n_free as u32);
        let mut basis = Vec::new();

        for ci in 0..n_configs {
            let mut config = vec![0i32; n_edges];
            let mut rem = ci;
            for &fi in &non_tree {
                let digit = (rem % base as u64) as i32;
                config[fi] = digit - lam;
                rem /= base as u64;
            }

            // Solve tree edges.
            let mut valid = true;
            for &(v, target_ei) in &solve_order {
                let mut sum = 0i32;
                let mut target_sign = 0i32;
                for &(ei, sign) in &adj[v] {
                    if ei == target_ei {
                        target_sign = sign;
                    } else {
                        sum += sign * config[ei];
                    }
                }
                let n_target = -sum * target_sign;
                if n_target < -lam || n_target > lam {
                    valid = false;
                    break;
                }
                config[target_ei] = n_target;
            }
            if valid {
                basis.push(config);
            }
        }

        let index_map: HashMap<Vec<i32>, usize> =
            basis.iter().enumerate().map(|(i, c)| (c.clone(), i)).collect();

        Self {
            n_edges,
            lambda,
            basis,
            index_map,
        }
    }

    pub fn dim(&self) -> usize {
        self.basis.len()
    }

    pub fn config_to_index(&self, config: &[i32]) -> Option<usize> {
        self.index_map.get(config).copied()
    }
}

/// Build the KS Hamiltonian for a hypercubic lattice.
///
/// Same structure as simplicial, but plaquettes are **squares** with 4 edges.
pub fn build_hypercubic_hamiltonian(
    hilbert: &HypercubicHilbert,
    lattice: &HypercubicLattice,
    g_squared: f64,
) -> DMat {
    let dim = hilbert.dim();
    let mut h = DMat::zeros(dim, dim);
    let lam = hilbert.lambda as i32;

    // Electric term: (g²/2) Σ_e n_e²
    let e_coeff = g_squared / 2.0;
    for i in 0..dim {
        let config = &hilbert.basis[i];
        let energy: f64 = config.iter().map(|&n| (n as f64) * (n as f64)).sum();
        h[(i, i)] += e_coeff * energy;
    }

    // Magnetic term: -(1/g²) Σ_plaq Re(U_plaq)
    let b_coeff = -1.0 / g_squared * 0.5;
    for (edge_indices, signs) in &lattice.plaquettes {
        for i in 0..dim {
            let config = &hilbert.basis[i];

            // Apply U_plaq: shift edges by signs.
            let mut shifted = config.clone();
            let mut valid = true;
            for k in 0..4 {
                shifted[edge_indices[k]] += signs[k];
                if shifted[edge_indices[k]] < -lam || shifted[edge_indices[k]] > lam {
                    valid = false;
                    break;
                }
            }
            if valid {
                if let Some(j) = hilbert.config_to_index(&shifted) {
                    h[(i, j)] += b_coeff;
                    h[(j, i)] += b_coeff;
                }
            }
        }
    }

    h
}

/// Run comparison: simplicial vs hypercubic spectra at a given coupling.
///
/// Returns (simplicial_spectrum, hypercubic_spectrum).
pub fn compare_spectra(
    complex: &phyz_regge::SimplicialComplex,
    lambda: u32,
    g_squared: f64,
    n_lowest: usize,
) -> (diag::Spectrum, diag::Spectrum) {
    // Simplicial
    let hs = crate::hilbert::U1HilbertSpace::new(complex, lambda);
    let params = crate::hamiltonian::KSParams {
        g_squared,
        metric_weights: None,
    };
    let h_simp = crate::hamiltonian::build_hamiltonian(&hs, complex, &params);
    let spec_simp = diag::diagonalize(&h_simp, Some(n_lowest));

    // Hypercubic: use a 2D periodic lattice for comparable size.
    // A 2×2 lattice in 2D has 4 vertices, 8 edges, 4 plaquettes.
    let lattice = HypercubicLattice::new(2, 2);
    let hh = HypercubicHilbert::new(&lattice, lambda);
    let h_hyp = build_hypercubic_hamiltonian(&hh, &lattice, g_squared);
    let spec_hyp = diag::diagonalize(&h_hyp, Some(n_lowest));

    (spec_simp, spec_hyp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2d_lattice_counts() {
        let lat = HypercubicLattice::new(2, 2);
        assert_eq!(lat.n_vertices, 4);
        // 2×2 periodic torus: each forward edge wraps, e.g. (0,0)→(1,0)
        // and (1,0)→(0,0) map to the same undirected edge.
        // 4 vertices × 2 axes = 8 directed → 4 undirected.
        assert_eq!(lat.n_edges(), 4);
        // One plaquette per vertex (single xy-plane): 4 plaquettes.
        assert_eq!(lat.n_plaquettes(), 4);
    }

    #[test]
    fn test_3d_lattice_counts() {
        let lat = HypercubicLattice::new(2, 3);
        assert_eq!(lat.n_vertices, 8);
        // 8 vertices × 3 axes = 24 directed → 12 undirected edges.
        assert_eq!(lat.n_edges(), 12);
    }

    #[test]
    fn test_hypercubic_hamiltonian_symmetric() {
        let lat = HypercubicLattice::new(2, 2);
        let hs = HypercubicHilbert::new(&lat, 1);
        let h = build_hypercubic_hamiltonian(&hs, &lat, 1.0);

        let diff = (&h - &h.transpose()).norm();
        assert!(diff < 1e-12, "H not symmetric: diff={diff}");
    }

    #[test]
    fn test_hypercubic_strong_coupling() {
        let lat = HypercubicLattice::new(2, 2);
        let hs = HypercubicHilbert::new(&lat, 1);
        let h = build_hypercubic_hamiltonian(&hs, &lat, 1e6);
        let spec = diag::diagonalize(&h, Some(1));

        assert!(spec.ground_energy().abs() < 1e-2, "E₀ = {}", spec.ground_energy());
    }

    #[test]
    fn test_hypercubic_dim() {
        let lat = HypercubicLattice::new(2, 2);
        let hs = HypercubicHilbert::new(&lat, 1);
        // 8 edges, 4 vertices, b1 = 8 - 4 + 1 = 5. dim ≤ 3^5 = 243.
        assert!(hs.dim() > 0);
        assert!(hs.dim() <= 243);
    }

    #[test]
    fn test_comparison_runs() {
        let complex =
            phyz_regge::SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);
        let (s_simp, s_hyp) = compare_spectra(&complex, 1, 1.0, 5);
        assert!(s_simp.energies.len() >= 3);
        assert!(s_hyp.energies.len() >= 3);
        // Both should have finite ground state energies.
        assert!(s_simp.ground_energy().is_finite());
        assert!(s_hyp.ground_energy().is_finite());
    }
}
