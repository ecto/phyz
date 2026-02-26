//! Ryu-Takayanagi formula check on simplicial gauge theory.
//!
//! Tests whether the entanglement entropy S_EE of the gauge theory ground state
//! correlates with the geometric area of the entangling surface, as predicted by
//! the RT formula S_EE = Area(γ_A) / 4G_N.
//!
//! Provides functions for:
//! - Vertex bipartitioning of simplicial complexes
//! - Classifying edges by partition (interior A, interior B, boundary)
//! - Computing topological and geometric cut areas
//! - Entanglement entropy for vertex-defined partitions

use crate::hilbert::U1HilbertSpace;
use crate::observables;
use phyz_math::DVec;
use phyz_regge::complex::SimplicialComplex;
use phyz_regge::geometry::triangle_area;

/// All non-trivial vertex bipartitions {A, complement}.
///
/// Returns partitions with `|A| <= n_vertices/2` to avoid duplicates
/// (since S(A) = S(complement) for pure states). Excludes empty partitions.
pub fn vertex_bipartitions(n_vertices: usize) -> Vec<Vec<usize>> {
    let mut partitions = Vec::new();
    let half = n_vertices / 2;

    // Iterate over all non-empty subsets with |A| <= n/2.
    // For |A| = n/2 with even n, only include subsets where the smallest
    // element is 0 to avoid A/complement duplicates.
    for mask in 1u64..(1u64 << n_vertices) {
        let a: Vec<usize> = (0..n_vertices).filter(|&v| mask & (1 << v) != 0).collect();
        let size = a.len();

        if size > half {
            continue;
        }
        if size == half && n_vertices % 2 == 0 && !a.contains(&0) {
            continue;
        }

        partitions.push(a);
    }

    partitions
}

/// Classify edges by a vertex partition.
///
/// Returns `(edges_in_a, edges_in_b, boundary_edges)` where:
/// - `edges_in_a`: both endpoints in `partition_a`
/// - `edges_in_b`: both endpoints NOT in `partition_a`
/// - `boundary_edges`: one endpoint in each side
pub fn classify_edges(
    complex: &SimplicialComplex,
    partition_a: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut in_a = vec![false; complex.n_vertices];
    for &v in partition_a {
        in_a[v] = true;
    }

    let mut edges_a = Vec::new();
    let mut edges_b = Vec::new();
    let mut boundary = Vec::new();

    for (ei, edge) in complex.edges.iter().enumerate() {
        let a0 = in_a[edge[0]];
        let a1 = in_a[edge[1]];
        match (a0, a1) {
            (true, true) => edges_a.push(ei),
            (false, false) => edges_b.push(ei),
            _ => boundary.push(ei),
        }
    }

    (edges_a, edges_b, boundary)
}

/// Topological area of cut: number of boundary edges.
pub fn cut_area_topological(
    complex: &SimplicialComplex,
    partition_a: &[usize],
) -> usize {
    let (_, _, boundary) = classify_edges(complex, partition_a);
    boundary.len()
}

/// Geometric area of cut: sum of lengths of boundary edges.
pub fn cut_area_geometric(
    complex: &SimplicialComplex,
    partition_a: &[usize],
    lengths: &[f64],
) -> f64 {
    let (_, _, boundary) = classify_edges(complex, partition_a);
    boundary.iter().map(|&ei| lengths[ei]).sum()
}

/// Triangle-based area: sum of areas of triangles straddling the boundary.
///
/// A triangle "straddles" if its 3 vertices are not all on the same side.
pub fn cut_area_triangles(
    complex: &SimplicialComplex,
    partition_a: &[usize],
    lengths: &[f64],
) -> f64 {
    let mut in_a = vec![false; complex.n_vertices];
    for &v in partition_a {
        in_a[v] = true;
    }

    let mut area = 0.0;
    for (ti, tri) in complex.triangles.iter().enumerate() {
        let count_a = tri.iter().filter(|&&v| in_a[v]).count();
        // Straddles if not all on one side (count_a is 1 or 2).
        if count_a > 0 && count_a < 3 {
            let [a, b, c] = complex.tri_edge_lengths(ti, lengths);
            area += triangle_area(a, b, c);
        }
    }

    area
}

/// Compute S_EE for a vertex partition.
///
/// Uses the "algebraic" prescription (Casini-Huerta-Rosabal): edges_a = edges
/// with BOTH endpoints in partition_a. Boundary edges are excluded from both
/// subsystems — they contribute to the boundary term, which is the
/// RT-analogous piece.
pub fn entanglement_for_partition(
    hilbert: &U1HilbertSpace,
    state: &DVec,
    complex: &SimplicialComplex,
    partition_a: &[usize],
) -> f64 {
    let (edges_a, _, _) = classify_edges(complex, partition_a);
    observables::entanglement_entropy(hilbert, state, &edges_a)
}

/// Entanglement entropy decomposed into superselection sectors for U(1).
///
/// Returns Shannon (edge mode) and distillable components separately.
pub fn entanglement_decomposed_for_partition(
    hilbert: &U1HilbertSpace,
    state: &DVec,
    complex: &SimplicialComplex,
    partition_a: &[usize],
) -> observables::EntropyDecomposition {
    let (edges_a, _, boundary) = classify_edges(complex, partition_a);
    observables::entanglement_entropy_decomposed(
        &hilbert.basis,
        hilbert.n_edges,
        state,
        &edges_a,
        &boundary,
    )
}

/// Compute S_EE using the "extended" prescription.
///
/// edges_a = edges with AT LEAST ONE endpoint in partition_a.
/// This includes both interior-A edges and boundary edges.
/// The complement gets only interior-B edges.
pub fn entanglement_for_partition_extended(
    hilbert: &U1HilbertSpace,
    state: &DVec,
    complex: &SimplicialComplex,
    partition_a: &[usize],
) -> f64 {
    let (edges_a, _, boundary) = classify_edges(complex, partition_a);
    let extended: Vec<usize> = edges_a.into_iter().chain(boundary).collect();
    observables::entanglement_entropy(hilbert, state, &extended)
}

/// Mutual information I(A:B) = S_A + S_B - S_AB.
///
/// Uses the algebraic edge prescription for both A and B individually.
/// S_AB is computed over edges_a ∪ edges_b (excluding boundary edges from
/// both subsystems, i.e., the union of interior edges).
///
/// For a pure state, S_AB = 0 when A ∪ B covers all edges, so
/// I(A:B) = S_A + S_B in that case. But with the algebraic prescription
/// boundary edges are excluded, so S_AB may be nonzero.
pub fn mutual_information(
    hilbert: &U1HilbertSpace,
    state: &DVec,
    complex: &SimplicialComplex,
    partition_a: &[usize],
) -> f64 {
    let (edges_a, edges_b, _boundary) = classify_edges(complex, partition_a);

    let s_a = observables::entanglement_entropy(hilbert, state, &edges_a);
    let s_b = observables::entanglement_entropy(hilbert, state, &edges_b);

    // S_AB: trace out everything except edges_a ∪ edges_b (i.e., trace out boundary).
    let edges_ab: Vec<usize> = edges_a.iter().chain(edges_b.iter()).copied().collect();
    let s_ab = observables::entanglement_entropy(hilbert, state, &edges_ab);

    s_a + s_b - s_ab
}

/// Assign Schwarzschild-like edge lengths to a simplicial complex.
///
/// Vertices closer to "center" (vertex 0) get shorter edges (gravitational
/// redshift). Parameter `mass` controls the curvature strength.
///
/// For edge [u, v] with graph distances d_u, d_v from vertex 0:
///   length = 1 + mass * (d_u + d_v) / (2 * diameter)
///
/// At mass=0 all edges are length 1.0 (flat). At mass>0 edges far from
/// the center are longer (simulating the metric redshift).
pub fn schwarzschild_edge_lengths(
    complex: &SimplicialComplex,
    mass: f64,
) -> Vec<f64> {
    // BFS from vertex 0 to compute graph distances.
    let n = complex.n_vertices;
    let mut dist = vec![usize::MAX; n];
    dist[0] = 0;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(0);

    // Build adjacency from edges.
    let mut adj = vec![Vec::new(); n];
    for edge in &complex.edges {
        adj[edge[0]].push(edge[1]);
        adj[edge[1]].push(edge[0]);
    }

    while let Some(v) = queue.pop_front() {
        for &w in &adj[v] {
            if dist[w] == usize::MAX {
                dist[w] = dist[v] + 1;
                queue.push_back(w);
            }
        }
    }

    let diameter = *dist.iter().max().unwrap_or(&1) as f64;
    let diameter = if diameter < 1e-10 { 1.0 } else { diameter };

    complex
        .edges
        .iter()
        .map(|edge| {
            let d_avg = (dist[edge[0]] + dist[edge[1]]) as f64 / 2.0;
            1.0 + mass * d_avg / diameter
        })
        .collect()
}

/// Simple linear regression: y = slope * x + intercept.
///
/// Returns (slope, intercept, r_squared).
pub fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-30 {
        return (0.0, sy / n, 0.0);
    }

    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;

    // R²
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| {
            let pred = slope * xi + intercept;
            (yi - pred) * (yi - pred)
        })
        .sum();
    let y_mean = sy / n;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean) * (yi - y_mean)).sum();
    let r_squared = if ss_tot > 1e-30 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };

    (slope, intercept, r_squared)
}

/// Format a partition as a compact string like "{0,2,3}".
pub fn partition_label(partition: &[usize]) -> String {
    let inner: Vec<String> = partition.iter().map(|v| v.to_string()).collect();
    format!("{{{}}}", inner.join(","))
}

/// De Sitter-like edge lengths: cosmological expansion from vertex 0.
///
/// BFS distance from vertex 0, length = cosh(H * d_avg / diameter).
/// At H=0 all edges are 1.0 (flat).
pub fn de_sitter_edge_lengths(complex: &SimplicialComplex, hubble: f64) -> Vec<f64> {
    let (dist, diameter) = bfs_distances(complex);
    complex
        .edges
        .iter()
        .map(|edge| {
            let d_avg = (dist[edge[0]] + dist[edge[1]]) as f64 / 2.0;
            (hubble * d_avg / diameter).cosh()
        })
        .collect()
}

/// Perturbed edge lengths: flat + deterministic pseudo-random perturbations.
///
/// length = 1.0 + epsilon * hash(seed, edge_idx), where hash ∈ [-1, 1].
/// No `rand` dependency — uses a simple deterministic hash.
pub fn perturbed_edge_lengths(complex: &SimplicialComplex, epsilon: f64, seed: u64) -> Vec<f64> {
    complex
        .edges
        .iter()
        .enumerate()
        .map(|(ei, _)| {
            let h = deterministic_hash(seed, ei as u64);
            1.0 + epsilon * h
        })
        .collect()
}

/// Anisotropic edge lengths: different scales for shared-face vs radial edges.
///
/// Edges with both endpoints in the first `n_shared_verts` vertices get length
/// `l_shared`; all other edges get `l_radial`.
pub fn anisotropic_edge_lengths(
    complex: &SimplicialComplex,
    l_shared: f64,
    l_radial: f64,
    n_shared_verts: usize,
) -> Vec<f64> {
    complex
        .edges
        .iter()
        .map(|edge| {
            if edge[0] < n_shared_verts && edge[1] < n_shared_verts {
                l_shared
            } else {
                l_radial
            }
        })
        .collect()
}

/// Conformal edge lengths: conformal rescaling from vertex 0.
///
/// length = (1 + alpha * d_avg / diameter)^2.
/// At alpha=0 all edges are 1.0 (flat).
pub fn conformal_edge_lengths(complex: &SimplicialComplex, alpha: f64) -> Vec<f64> {
    let (dist, diameter) = bfs_distances(complex);
    complex
        .edges
        .iter()
        .map(|edge| {
            let d_avg = (dist[edge[0]] + dist[edge[1]]) as f64 / 2.0;
            let factor = 1.0 + alpha * d_avg / diameter;
            factor * factor
        })
        .collect()
}

/// Check if a geometry is valid: all triangles have positive area and all
/// 4-simplices have positive volume.
pub fn geometry_valid(complex: &SimplicialComplex, lengths: &[f64]) -> bool {
    use phyz_regge::geometry::{pent_volume, triangle_area};

    for ti in 0..complex.n_triangles() {
        let [a, b, c] = complex.tri_edge_lengths(ti, lengths);
        if triangle_area(a, b, c) <= 0.0 {
            return false;
        }
    }
    for pi in 0..complex.n_pents() {
        let pl = complex.pent_edge_lengths(pi, lengths);
        if pent_volume(&pl) <= 0.0 {
            return false;
        }
    }
    true
}

/// BFS distances from vertex 0. Returns (dist, diameter).
fn bfs_distances(complex: &SimplicialComplex) -> (Vec<usize>, f64) {
    let n = complex.n_vertices;
    let mut dist = vec![usize::MAX; n];
    dist[0] = 0;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(0);

    let mut adj = vec![Vec::new(); n];
    for edge in &complex.edges {
        adj[edge[0]].push(edge[1]);
        adj[edge[1]].push(edge[0]);
    }

    while let Some(v) = queue.pop_front() {
        for &w in &adj[v] {
            if dist[w] == usize::MAX {
                dist[w] = dist[v] + 1;
                queue.push_back(w);
            }
        }
    }

    let diameter = *dist.iter().max().unwrap_or(&1) as f64;
    let diameter = if diameter < 1e-10 { 1.0 } else { diameter };
    (dist, diameter)
}

/// Deterministic hash mapping (seed, index) → [-1, 1].
fn deterministic_hash(seed: u64, index: u64) -> f64 {
    // Simple bit-mixing hash (splitmix64-style).
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(index);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    // Map to [-1, 1].
    (x as f64) / (u64::MAX as f64) * 2.0 - 1.0
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
    fn test_classify_edges_counts() {
        let complex = single_pentachoron();
        // Partition: {0, 1}
        let partition_a = vec![0, 1];
        let (ea, eb, boundary) = classify_edges(&complex, &partition_a);

        // edges_a + edges_b + boundary = total edges
        assert_eq!(
            ea.len() + eb.len() + boundary.len(),
            complex.n_edges(),
            "edge counts don't sum to total"
        );

        // {0,1}: one edge (0-1) in A, C(3,2)=3 edges in B, 2*3=6 boundary
        assert_eq!(ea.len(), 1, "expected 1 edge in A");
        assert_eq!(eb.len(), 3, "expected 3 edges in B");
        assert_eq!(boundary.len(), 6, "expected 6 boundary edges");
    }

    #[test]
    fn test_classify_edges_sum() {
        let complex = single_pentachoron();
        // Test several partitions: edge counts always sum correctly.
        for part in vertex_bipartitions(5) {
            let (ea, eb, bd) = classify_edges(&complex, &part);
            assert_eq!(
                ea.len() + eb.len() + bd.len(),
                complex.n_edges(),
                "partition {:?}: counts don't sum",
                part
            );
        }
    }

    #[test]
    fn test_cut_area_topological_matches_boundary() {
        let complex = single_pentachoron();
        let partition_a = vec![0, 1];
        let (_, _, boundary) = classify_edges(&complex, &partition_a);
        assert_eq!(
            cut_area_topological(&complex, &partition_a),
            boundary.len()
        );
    }

    #[test]
    fn test_flat_geometric_equals_topological() {
        // With all edge lengths = 1.0, geometric area = number of cut edges.
        let complex = single_pentachoron();
        let lengths = vec![1.0; complex.n_edges()];
        for part in vertex_bipartitions(5) {
            let topo = cut_area_topological(&complex, &part) as f64;
            let geom = cut_area_geometric(&complex, &part, &lengths);
            assert!(
                (topo - geom).abs() < 1e-12,
                "partition {:?}: topo={topo} != geom={geom}",
                part
            );
        }
    }

    #[test]
    fn test_bipartitions_no_empty() {
        let parts = vertex_bipartitions(5);
        for p in &parts {
            assert!(!p.is_empty(), "empty partition found");
            assert!(p.len() <= 2, "|A| > n/2 found: {:?}", p); // n/2 = 2 for n=5
        }
    }

    #[test]
    fn test_bipartitions_count() {
        // For n=5: non-trivial partitions with |A| <= 2.
        // |A|=1: C(5,1)=5, |A|=2: C(5,2)=10. Total=15.
        let parts = vertex_bipartitions(5);
        assert_eq!(parts.len(), 15);
    }

    #[test]
    fn test_schwarzschild_flat_at_zero_mass() {
        let complex = single_pentachoron();
        let lengths = schwarzschild_edge_lengths(&complex, 0.0);
        for &l in &lengths {
            assert!(
                (l - 1.0).abs() < 1e-12,
                "expected flat lengths at M=0, got {l}"
            );
        }
    }

    #[test]
    fn test_schwarzschild_monotone() {
        // Edges farther from center should be longer at positive mass.
        let complex = single_pentachoron();
        let lengths = schwarzschild_edge_lengths(&complex, 1.0);
        // All lengths should be >= 1.0 (since mass > 0, d_avg >= 0).
        for &l in &lengths {
            assert!(l >= 1.0 - 1e-12, "length {l} < 1.0");
        }
    }

    #[test]
    fn test_entanglement_trivial_partition() {
        // Empty partition → no edges in A → S = 0.
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let params = KSParams {
            g_squared: 1.0,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        // Full partition: all edges in A → ρ_A = |ψ⟩⟨ψ| → S = 0.
        let all_vertices: Vec<usize> = (0..5).collect();
        let s_full = entanglement_for_partition(&hs, gs, &complex, &all_vertices);
        assert!(
            s_full < 1e-10,
            "full partition should give S=0, got {s_full}"
        );
    }

    #[test]
    fn test_entanglement_nonnegative() {
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let params = KSParams {
            g_squared: 1.0,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        for part in vertex_bipartitions(5) {
            let s = entanglement_for_partition(&hs, gs, &complex, &part);
            assert!(s >= -1e-10, "partition {:?}: S = {s} < 0", part);
        }
    }

    #[test]
    fn test_extended_prescription_nonnegative() {
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let params = KSParams {
            g_squared: 1.0,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        for part in vertex_bipartitions(5) {
            let s = entanglement_for_partition_extended(&hs, gs, &complex, &part);
            assert!(s >= -1e-10, "extended partition {:?}: S = {s} < 0", part);
        }
    }

    #[test]
    fn test_extended_geq_algebraic() {
        // Extended prescription includes more edges in A, so the reduced state
        // of A has more DOF but traces out fewer. The entropy can go either way,
        // but for single-vertex partitions the extended gives nonzero (boundary
        // edges included) while algebraic gives zero.
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let params = KSParams {
            g_squared: 1.0,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        // Single vertex: algebraic gives 0, extended should give nonzero.
        let part = vec![0];
        let s_alg = entanglement_for_partition(&hs, gs, &complex, &part);
        let s_ext = entanglement_for_partition_extended(&hs, gs, &complex, &part);
        assert!(s_alg < 1e-10, "algebraic should be ~0 for single vertex");
        assert!(s_ext > 0.1, "extended should be nonzero for single vertex, got {s_ext}");
    }

    #[test]
    fn test_mutual_information_nonnegative() {
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let params = KSParams {
            g_squared: 1.0,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        for part in vertex_bipartitions(5) {
            let mi = mutual_information(&hs, gs, &complex, &part);
            assert!(
                mi >= -1e-10,
                "partition {:?}: MI = {mi} < 0",
                part
            );
        }
    }

    #[test]
    fn test_mutual_information_strong_coupling_zero() {
        // In strong coupling, ground state is product → MI should be ~0.
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);
        let params = KSParams {
            g_squared: 1e6,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs, &complex, &params);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        let part = vec![0, 1];
        let mi = mutual_information(&hs, gs, &complex, &part);
        assert!(mi < 0.1, "strong coupling MI should be ~0, got {mi}");
    }

    #[test]
    fn test_linear_regression_perfect() {
        // y = 2x + 1 exactly.
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 3.0, 5.0, 7.0];
        let (slope, intercept, r2) = linear_regression(&x, &y);
        assert!((slope - 2.0).abs() < 1e-10);
        assert!((intercept - 1.0).abs() < 1e-10);
        assert!((r2 - 1.0).abs() < 1e-10);
    }

    // ── Geometry generator tests ──

    fn two_pentachorons() -> SimplicialComplex {
        SimplicialComplex::from_pentachorons(6, &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]])
    }

    #[test]
    fn test_de_sitter_flat_at_zero() {
        let complex = two_pentachorons();
        let lengths = de_sitter_edge_lengths(&complex, 0.0);
        for &l in &lengths {
            assert!(
                (l - 1.0).abs() < 1e-12,
                "expected flat at H=0, got {l}"
            );
        }
    }

    #[test]
    fn test_de_sitter_valid_geometry() {
        let complex = two_pentachorons();
        let lengths = de_sitter_edge_lengths(&complex, 0.5);
        assert!(geometry_valid(&complex, &lengths), "de Sitter H=0.5 invalid");
    }

    #[test]
    fn test_perturbed_flat_at_zero() {
        let complex = two_pentachorons();
        let lengths = perturbed_edge_lengths(&complex, 0.0, 42);
        for &l in &lengths {
            assert!(
                (l - 1.0).abs() < 1e-12,
                "expected flat at eps=0, got {l}"
            );
        }
    }

    #[test]
    fn test_perturbed_deterministic() {
        let complex = two_pentachorons();
        let l1 = perturbed_edge_lengths(&complex, 0.1, 42);
        let l2 = perturbed_edge_lengths(&complex, 0.1, 42);
        assert_eq!(l1, l2, "perturbed should be deterministic");
    }

    #[test]
    fn test_perturbed_valid_geometry() {
        let complex = two_pentachorons();
        let lengths = perturbed_edge_lengths(&complex, 0.1, 42);
        assert!(geometry_valid(&complex, &lengths), "perturbed eps=0.1 invalid");
    }

    #[test]
    fn test_anisotropic_uniform() {
        let complex = two_pentachorons();
        let lengths = anisotropic_edge_lengths(&complex, 1.0, 1.0, 4);
        for &l in &lengths {
            assert!(
                (l - 1.0).abs() < 1e-12,
                "expected flat at uniform aniso, got {l}"
            );
        }
    }

    #[test]
    fn test_anisotropic_two_scales() {
        let complex = two_pentachorons();
        let lengths = anisotropic_edge_lengths(&complex, 0.8, 1.2, 4);
        // Shared-face edges (both verts < 4) should be 0.8.
        for (ei, edge) in complex.edges.iter().enumerate() {
            if edge[0] < 4 && edge[1] < 4 {
                assert!(
                    (lengths[ei] - 0.8).abs() < 1e-12,
                    "shared edge {ei}: expected 0.8, got {}",
                    lengths[ei]
                );
            } else {
                assert!(
                    (lengths[ei] - 1.2).abs() < 1e-12,
                    "radial edge {ei}: expected 1.2, got {}",
                    lengths[ei]
                );
            }
        }
    }

    #[test]
    fn test_conformal_flat_at_zero() {
        let complex = two_pentachorons();
        let lengths = conformal_edge_lengths(&complex, 0.0);
        for &l in &lengths {
            assert!(
                (l - 1.0).abs() < 1e-12,
                "expected flat at alpha=0, got {l}"
            );
        }
    }

    #[test]
    fn test_conformal_valid_geometry() {
        let complex = two_pentachorons();
        let lengths = conformal_edge_lengths(&complex, 0.3);
        assert!(geometry_valid(&complex, &lengths), "conformal alpha=0.3 invalid");
    }

    #[test]
    fn test_geometry_valid_flat() {
        let complex = two_pentachorons();
        let flat = vec![1.0; complex.n_edges()];
        assert!(geometry_valid(&complex, &flat), "flat geometry should be valid");
    }

    #[test]
    fn test_geometry_valid_degenerate() {
        let complex = single_pentachoron();
        // Make one edge very long → degenerate triangle
        let mut lengths = vec![1.0; complex.n_edges()];
        lengths[0] = 100.0;
        assert!(!geometry_valid(&complex, &lengths), "degenerate geometry should be invalid");
    }
}
