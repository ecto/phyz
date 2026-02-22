//! Triangulated torus for matched-topology comparison.
//!
//! Takes the same n×n periodic 2D lattice as `HypercubicLattice::new(n, 2)`
//! and triangulates each square face by adding a diagonal edge. This gives
//! a simplicial lattice on the **same torus** with **triangular plaquettes**
//! instead of square ones.
//!
//! Comparison with the hypercubic version isolates the effect of plaquette
//! shape (triangular vs square) from topology effects.

use nalgebra::DMatrix;
use std::collections::HashMap;

/// Triangulated 2D periodic torus.
///
/// Same vertices as `HypercubicLattice::new(n, 2)`, but each square face
/// is split into 2 triangles by adding a diagonal edge.
///
/// For an n×n torus:
/// - Vertices: n²
/// - Original edges: 2n² (horizontal + vertical)
/// - Diagonal edges: n² (one per square)
/// - Total edges: 3n²
/// - Triangular plaquettes: 2n² (two per square)
#[derive(Debug, Clone)]
pub struct TriangulatedTorus {
    /// Grid size.
    pub n: usize,
    /// Number of vertices (n²).
    pub n_vertices: usize,
    /// Edges: (vertex_a, vertex_b) with a < b.
    pub edges: Vec<(usize, usize)>,
    /// Triangular plaquettes: [edge0, edge1, edge2] with signs [s0, s1, s2].
    /// The holonomy goes around the triangle with appropriate orientations.
    pub plaquettes: Vec<([usize; 3], [i32; 3])>,
}

impl TriangulatedTorus {
    /// Build a triangulated n×n periodic 2D torus.
    ///
    /// Each square (i,j)→(i+1,j)→(i+1,j+1)→(i,j+1) is split by the
    /// diagonal from (i,j) to (i+1,j+1) into:
    ///   - Lower triangle: (i,j)→(i+1,j)→(i+1,j+1)
    ///   - Upper triangle: (i,j)→(i+1,j+1)→(i,j+1)
    pub fn new(n: usize) -> Self {
        let n_vertices = n * n;
        let mut edges = Vec::new();
        let mut edge_map: HashMap<(usize, usize), usize> = HashMap::new();

        let vertex = |i: usize, j: usize| -> usize { (i % n) * n + (j % n) };

        let mut get_edge = |a: usize, b: usize| -> usize {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            let len = edges.len();
            *edge_map.entry((lo, hi)).or_insert_with(|| {
                edges.push((lo, hi));
                len
            })
        };

        let mut plaquettes = Vec::new();

        for i in 0..n {
            for j in 0..n {
                // Four vertices of the square
                let v00 = vertex(i, j);
                let v10 = vertex(i + 1, j);
                let v11 = vertex(i + 1, j + 1);
                let v01 = vertex(i, j + 1);

                // Edges of the square + diagonal
                let e_bot = get_edge(v00, v10);   // bottom: (i,j)→(i+1,j)
                let e_right = get_edge(v10, v11); // right: (i+1,j)→(i+1,j+1)
                let e_diag = get_edge(v00, v11);  // diagonal: (i,j)→(i+1,j+1)
                let e_top = get_edge(v01, v11);   // top: (i,j+1)→(i+1,j+1)
                let e_left = get_edge(v00, v01);  // left: (i,j)→(i,j+1)

                // Lower triangle: v00 → v10 → v11
                // Holonomy: +e_bot, +e_right, -e_diag
                // (traverse bottom, then right, then back along diagonal)
                let s_bot = if v00 < v10 { 1 } else { -1 };
                let s_right = if v10 < v11 { 1 } else { -1 };
                let s_diag_lower = if v00 < v11 { -1 } else { 1 }; // reverse direction
                plaquettes.push((
                    [e_bot, e_right, e_diag],
                    [s_bot, s_right, s_diag_lower],
                ));

                // Upper triangle: v00 → v11 → v01
                // Holonomy: +e_diag, +e_top(reversed), -e_left
                // Traverse: v00→v11 along diagonal, v11→v01 along top, v01→v00 along left
                let s_diag_upper = if v00 < v11 { 1 } else { -1 }; // forward direction
                // Canonical edge is (min, max). If v01 < v11, edge goes v01→v11,
                // so traversing v11→v01 is direction -1.
                let s_top_corrected = if v01 < v11 { -1 } else { 1 };
                let s_left = if v00 < v01 { -1 } else { 1 }; // v01→v00 = reverse
                plaquettes.push((
                    [e_diag, e_top, e_left],
                    [s_diag_upper, s_top_corrected, s_left],
                ));
            }
        }

        Self {
            n,
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

/// Gauge-invariant Hilbert space for the triangulated torus.
///
/// Same spanning-tree + leaf-peeling approach as the other lattice types.
pub struct TriangulatedTorusHilbert {
    pub n_edges: usize,
    pub lambda: u32,
    pub basis: Vec<Vec<i32>>,
    index_map: HashMap<Vec<i32>, usize>,
}

impl TriangulatedTorusHilbert {
    pub fn new(torus: &TriangulatedTorus, lambda: u32) -> Self {
        let n_edges = torus.n_edges();
        let n_v = torus.n_vertices;
        let lam = lambda as i32;
        let base = 2 * lam + 1;

        // Build adjacency.
        let mut adj: Vec<Vec<(usize, i32)>> = vec![Vec::new(); n_v];
        for (ei, &(a, b)) in torus.edges.iter().enumerate() {
            adj[a].push((ei, -1)); // a < b by construction
            adj[b].push((ei, 1));
        }

        // BFS spanning tree.
        let mut visited = vec![false; n_v];
        let mut is_tree = vec![false; n_edges];
        let mut queue = std::collections::VecDeque::new();
        visited[0] = true;
        queue.push_back(0);
        while let Some(v) = queue.pop_front() {
            for &(ei, _) in &adj[v] {
                let (a, b) = torus.edges[ei];
                let other = if a == v { b } else { a };
                if !visited[other] {
                    visited[other] = true;
                    is_tree[ei] = true;
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
            if solved[ei] {
                continue;
            }
            solved[ei] = true;
            solve_order.push((v, ei));
            let (a, b) = torus.edges[ei];
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

/// Build the KS Hamiltonian for a triangulated torus.
///
/// Same structure as the simplicial Hamiltonian but using the torus topology
/// with triangular plaquettes.
pub fn build_triangulated_torus_hamiltonian(
    hilbert: &TriangulatedTorusHilbert,
    torus: &TriangulatedTorus,
    g_squared: f64,
) -> DMatrix<f64> {
    let dim = hilbert.dim();
    let mut h = DMatrix::zeros(dim, dim);
    let lam = hilbert.lambda as i32;

    // Electric term: (g²/2) Σ_e n_e²
    let e_coeff = g_squared / 2.0;
    for i in 0..dim {
        let config = &hilbert.basis[i];
        let energy: f64 = config.iter().map(|&n| (n as f64) * (n as f64)).sum();
        h[(i, i)] += e_coeff * energy;
    }

    // Magnetic term: -(1/g²) Σ_tri Re(U_tri)
    let b_coeff = -1.0 / g_squared * 0.5;
    for (edge_indices, signs) in &torus.plaquettes {
        for i in 0..dim {
            let config = &hilbert.basis[i];

            let mut shifted = config.clone();
            let mut valid = true;
            for k in 0..3 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag;

    #[test]
    fn test_torus_counts_2x2() {
        let t = TriangulatedTorus::new(2);
        assert_eq!(t.n_vertices, 4);
        // 2×2 torus: 4 horizontal + 4 vertical + 4 diagonal = 12? No...
        // 2×2 periodic: 2 horizontal edges per row × 2 rows = 4 horizontal.
        // Wait, n=2 means 2 vertices per axis. Horizontal: vertex(i,j)→vertex(i+1,j).
        // With wrapping: (0,0)→(1,0), (1,0)→(0,0) — same edge! So 2 horizontal.
        // Similarly 2 vertical. Plus 2 diagonals? Let's just check.
        assert_eq!(t.n_edges(), 6); // 2 horiz + 2 vert + 2 diag
        assert_eq!(t.n_plaquettes(), 8); // 2 per square × 4 squares
    }

    #[test]
    fn test_torus_counts_3x3() {
        let t = TriangulatedTorus::new(3);
        assert_eq!(t.n_vertices, 9);
        // 3×3 periodic: 9 horizontal + 9 vertical + 9 diagonal = 27 edges.
        assert_eq!(t.n_edges(), 27);
        // 2 triangles per square × 9 squares = 18 plaquettes.
        assert_eq!(t.n_plaquettes(), 18);
    }

    #[test]
    fn test_triangulated_hamiltonian_symmetric() {
        let t = TriangulatedTorus::new(2);
        let hs = TriangulatedTorusHilbert::new(&t, 1);
        let h = build_triangulated_torus_hamiltonian(&hs, &t, 1.0);

        let diff = (&h - h.transpose()).norm();
        assert!(diff < 1e-12, "H not symmetric: diff={diff}");
    }

    #[test]
    fn test_triangulated_strong_coupling() {
        let t = TriangulatedTorus::new(2);
        let hs = TriangulatedTorusHilbert::new(&t, 1);
        let h = build_triangulated_torus_hamiltonian(&hs, &t, 1e6);
        let spec = diag::diagonalize(&h, Some(1));

        assert!(spec.ground_energy().abs() < 1e-2, "E₀ = {}", spec.ground_energy());
    }

    #[test]
    fn test_b1_correct() {
        // b₁ = E - V + 1 for connected lattice on torus (genus 1).
        // Actually for a torus, b₁ = E - V + 1 = E - V + χ where χ(T²) = 0.
        // So b₁ = E - V for a torus? No...
        // For a graph on a surface: b₁ = E - V + connected components.
        // Connected, so b₁ = E - V + 1.
        let t2 = TriangulatedTorus::new(2);
        let b1 = t2.n_edges() - t2.n_vertices + 1;
        // 2×2: b₁ = 6 - 4 + 1 = 3
        assert_eq!(b1, 3);

        let t3 = TriangulatedTorus::new(3);
        let b1_3 = t3.n_edges() - t3.n_vertices + 1;
        // 3×3: b₁ = 27 - 9 + 1 = 19
        assert_eq!(b1_3, 19);
    }
}
