//! Traced quantum solver — builds the full computation as an expression graph.
//!
//! Instead of computing numbers, this module traces the entire quantum solver
//! pipeline through `ExprId` to produce a `WireGraph` that tang-mesh can
//! dispatch to any worker.
//!
//! # Pipeline
//!
//! 1. Inputs: coupling g², metric weights per triangle (as `Var` nodes)
//! 2. Build Hamiltonian as `DMat<ExprId>`
//! 3. Eigendecompose via branchless Jacobi
//! 4. Compute ground state energy + entanglement entropy per partition

use tang::Scalar;
use tang_expr::node::ExprId;
use tang_expr::ExprGraph;
use tang_la::{branchless_jacobi_eigen, DMat};

use crate::hilbert::U1HilbertSpace;
use phyz_regge::SimplicialComplex;

/// Holonomy signs for a triangle: [+1, -1, +1].
const TRIANGLE_HOLONOMY_SIGNS: [i32; 3] = [1, -1, 1];

/// Result of tracing the quantum solver.
pub struct TracedSolver {
    /// The expression graph containing the full computation.
    pub graph: ExprGraph,
    /// Ground state energy output.
    pub ground_energy: ExprId,
    /// Entanglement entropy for each vertex bipartition.
    pub entropies: Vec<ExprId>,
    /// Boundary area (sum of boundary triangle weights) per partition.
    pub boundary_areas: Vec<ExprId>,
    /// Number of input variables.
    pub n_inputs: u16,
}

/// Trace the quantum entropy computation as an expression graph.
///
/// # Inputs (as Var nodes)
///
/// - `Var(0)` = coupling g²
/// - `Var(1)..Var(n_triangles)` = metric weights (one per triangle)
///
/// The `partitions` parameter specifies which vertex sets to compute entropy
/// for. Each partition is a set of vertex indices defining region A.
pub fn trace_quantum_entropy(
    complex: &SimplicialComplex,
    hilbert: &U1HilbertSpace,
    partitions: &[Vec<usize>],
    n_sweeps: usize,
) -> TracedSolver {
    let dim = hilbert.dim();
    let n_tri = complex.n_triangles();
    let n_inputs = (1 + n_tri) as u16;

    // Pre-compute all integer topology (indices, configs, shifts) before tracing
    let ham_structure = precompute_hamiltonian_structure(hilbert, complex);
    let partition_structures: Vec<_> = partitions
        .iter()
        .map(|p| precompute_partition_structure(complex, hilbert, p))
        .collect();

    let (graph, outputs) = tang_expr::trace(|| {
        // Input variables
        let g_squared = ExprId::var(0);
        let metric_weights: Vec<ExprId> = (0..n_tri)
            .map(|i| ExprId::var((i + 1) as u16))
            .collect();

        // Build Hamiltonian
        let h = build_traced_hamiltonian(
            dim,
            g_squared,
            &metric_weights,
            &ham_structure,
        );

        // Eigendecompose
        let (eigenvalues, eigenvectors) = branchless_jacobi_eigen(&h, n_sweeps);

        let ground_energy = eigenvalues[0];
        let ground_state: Vec<ExprId> = (0..dim).map(|i| eigenvectors.get(i, 0)).collect();

        // Entropy + boundary areas per partition
        let mut entropies = Vec::new();
        let mut boundary_areas = Vec::new();

        for ps in &partition_structures {
            let entropy = traced_entanglement_entropy(
                &ground_state,
                &ps.b_groups,
                ps.dim_a,
                n_sweeps,
            );
            entropies.push(entropy);

            // Boundary area = sum of metric weights for triangles touching boundary edges
            let mut area = ExprId::ZERO;
            for &ti in &ps.boundary_triangles {
                area = area + metric_weights[ti];
            }
            boundary_areas.push(area);
        }

        let mut all = vec![ground_energy];
        all.extend_from_slice(&entropies);
        all.extend_from_slice(&boundary_areas);
        all
    });

    let n_parts = partitions.len();
    let ground_energy = outputs[0];
    let entropies = outputs[1..1 + n_parts].to_vec();
    let boundary_areas = outputs[1 + n_parts..1 + 2 * n_parts].to_vec();

    TracedSolver {
        graph,
        ground_energy,
        entropies,
        boundary_areas,
        n_inputs,
    }
}

// --- Pre-computed integer structures (no ExprId, just topology) ---

/// Pre-computed Hamiltonian structure: which basis states connect to which.
struct HamiltonianStructure {
    /// For each basis state: Σ n_e² (the electric energy as f64)
    electric_energies: Vec<f64>,
    /// (triangle_index, basis_state_i, basis_state_j) for magnetic term entries
    magnetic_entries: Vec<(usize, usize, usize)>,
}

/// Pre-computed partition structure for entropy computation.
struct PartitionStructure {
    /// Dimension of reduced density matrix
    dim_a: usize,
    /// Groups by B-config: Vec of (a_index, basis_state_index) pairs
    b_groups: Vec<Vec<(usize, usize)>>,
    /// Triangle indices that touch boundary edges
    boundary_triangles: Vec<usize>,
}

fn precompute_hamiltonian_structure(
    hilbert: &U1HilbertSpace,
    complex: &SimplicialComplex,
) -> HamiltonianStructure {
    let dim = hilbert.dim();

    let electric_energies: Vec<f64> = (0..dim)
        .map(|i| {
            let config = hilbert.index_to_config(i);
            config.iter().map(|&n| (n as f64) * (n as f64)).sum()
        })
        .collect();

    let mut magnetic_entries = Vec::new();
    for (ti, _tri) in complex.triangles.iter().enumerate() {
        let edge_indices = complex.tri_edge_indices(ti);
        for i in 0..dim {
            let config = hilbert.index_to_config(i);
            if let Some(j) = apply_triangle_shift_static(
                hilbert, config, &edge_indices, &TRIANGLE_HOLONOMY_SIGNS,
            ) {
                magnetic_entries.push((ti, i, j));
            }
        }
    }

    HamiltonianStructure {
        electric_energies,
        magnetic_entries,
    }
}

fn precompute_partition_structure(
    complex: &SimplicialComplex,
    hilbert: &U1HilbertSpace,
    partition_a: &[usize],
) -> PartitionStructure {
    let n_edges = hilbert.n_edges;
    let n_tri = complex.n_triangles();

    let (edges_a, _edges_b, boundary_edges) = classify_edges_static(complex, partition_a);

    let mut is_a = vec![false; n_edges];
    for &e in &edges_a {
        is_a[e] = true;
    }

    use std::collections::HashMap;

    let a_configs: Vec<Vec<i32>> = hilbert
        .basis
        .iter()
        .map(|c| edges_a.iter().map(|&e| c[e]).collect())
        .collect();

    let mut a_index_map: HashMap<Vec<i32>, usize> = HashMap::new();
    for ac in &a_configs {
        let len = a_index_map.len();
        a_index_map.entry(ac.clone()).or_insert(len);
    }
    let dim_a = a_index_map.len();

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

    let mut b_group_map: HashMap<Vec<i32>, Vec<(usize, usize)>> = HashMap::new();
    for (i, bc) in b_configs.iter().enumerate() {
        b_group_map
            .entry(bc.clone())
            .or_default()
            .push((a_index_map[&a_configs[i]], i));
    }
    let b_groups: Vec<Vec<(usize, usize)>> = b_group_map.into_values().collect();

    // Find triangles that touch boundary edges
    let mut boundary_tri_set = std::collections::HashSet::new();
    for &e in &boundary_edges {
        for ti in 0..n_tri {
            let tri_edges = complex.tri_edge_indices(ti);
            if tri_edges.contains(&e) {
                boundary_tri_set.insert(ti);
            }
        }
    }
    let boundary_triangles: Vec<usize> = boundary_tri_set.into_iter().collect();

    PartitionStructure {
        dim_a,
        b_groups,
        boundary_triangles,
    }
}

/// Build the KS Hamiltonian as a dense matrix of ExprId nodes.
fn build_traced_hamiltonian(
    dim: usize,
    g_squared: ExprId,
    metric_weights: &[ExprId],
    structure: &HamiltonianStructure,
) -> DMat<ExprId> {
    let mut h = DMat::<ExprId>::zeros(dim, dim);

    // Electric term: H_E = (g²/2) Σ_e n_e²
    let half = ExprId::from_f64(0.5);
    let coeff_e = g_squared * half;

    for i in 0..dim {
        let energy = ExprId::from_f64(structure.electric_energies[i]);
        h.set(i, i, coeff_e * energy);
    }

    // Magnetic term: H_B = -(1/g²) Σ_tri w_tri · (1/2) Re(U_tri)
    let neg_half = ExprId::from_f64(-0.5);
    let recip_g2 = g_squared.recip();
    let base_coeff = neg_half * recip_g2;

    for &(ti, i, j) in &structure.magnetic_entries {
        let c = base_coeff * metric_weights[ti];
        let old_ij = h.get(i, j);
        h.set(i, j, old_ij + c);
        let old_ji = h.get(j, i);
        h.set(j, i, old_ji + c);
    }

    h
}

/// Same as hamiltonian::apply_triangle_shift.
fn apply_triangle_shift_static(
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

/// Classify edges into region A, region B, and boundary.
fn classify_edges_static(
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

    for (i, edge) in complex.edges.iter().enumerate() {
        let a0 = in_a[edge[0]];
        let a1 = in_a[edge[1]];
        match (a0, a1) {
            (true, true) => edges_a.push(i),
            (false, false) => edges_b.push(i),
            _ => boundary.push(i),
        }
    }

    (edges_a, edges_b, boundary)
}

/// Compute entanglement entropy as a traced expression.
fn traced_entanglement_entropy(
    state: &[ExprId],
    b_groups: &[Vec<(usize, usize)>],
    dim_a: usize,
    n_sweeps: usize,
) -> ExprId {
    // Build reduced density matrix ρ_A
    let mut rho = DMat::<ExprId>::zeros(dim_a, dim_a);

    for entries in b_groups {
        for &(ai, state_i) in entries {
            for &(aj, state_j) in entries {
                let old = rho.get(ai, aj);
                rho.set(ai, aj, old + state[state_i] * state[state_j]);
            }
        }
    }

    // Eigendecompose ρ_A
    let (rho_evals, _) = branchless_jacobi_eigen(&rho, n_sweeps);

    // S = -Σ select(λ - ε, λ * ln(λ), 0)
    let eps = ExprId::from_f64(1e-15);
    let mut entropy = ExprId::ZERO;
    for i in 0..dim_a {
        let lam = rho_evals[i];
        let ln_lam = Scalar::ln(lam);
        let contribution = lam * ln_lam;
        let safe = ExprId::select(lam - eps, contribution, ExprId::ZERO);
        entropy = entropy - safe;
    }

    entropy
}

impl TracedSolver {
    /// All output ExprIds (for WireGraph serialization).
    pub fn all_outputs(&self) -> Vec<ExprId> {
        let mut out = vec![self.ground_energy];
        out.extend_from_slice(&self.entropies);
        out.extend_from_slice(&self.boundary_areas);
        out
    }

    /// Total number of nodes in the expression graph.
    pub fn node_count(&self) -> usize {
        self.graph.len()
    }

    /// Evaluate with concrete inputs.
    pub fn eval(&self, inputs: &[f64]) -> Vec<f64> {
        let outputs = self.all_outputs();
        self.graph.eval_many(&outputs, inputs)
    }

    /// Convert to a WireGraph for distributed execution.
    #[cfg(feature = "mesh")]
    pub fn to_wire_graph(&self) -> tang_mesh::protocol::WireGraph {
        let outputs = self.all_outputs();
        tang_mesh::protocol::WireGraph::from_expr_graph(&self.graph, &outputs, self.n_inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test the core branchless Jacobi eigen tracing on a small synthetic matrix.
    /// Uses a 4×4 Hamiltonian to keep the graph small and tests fast.
    #[test]
    fn traced_small_eigen() {
        // Build a small 4×4 symmetric matrix and trace the eigendecomposition
        let (graph, (energy, _evecs_flat)) = tang_expr::trace(|| {
            let x = ExprId::var(0); // parametric input

            let mut h = DMat::<ExprId>::zeros(4, 4);
            let two = ExprId::from_f64(2.0);
            let one = ExprId::ONE;
            // Simple tridiagonal: diag = [x, 2, 3, 4], off-diag = [1, 1, 1]
            h.set(0, 0, x);
            h.set(1, 1, two);
            h.set(2, 2, ExprId::from_f64(3.0));
            h.set(3, 3, ExprId::from_f64(4.0));
            h.set(0, 1, one);
            h.set(1, 0, one);
            h.set(1, 2, one);
            h.set(2, 1, one);
            h.set(2, 3, one);
            h.set(3, 2, one);

            let (evals, evecs) = branchless_jacobi_eigen(&h, 20);
            let ground = evals[0];

            // Extract first eigenvector
            let v: Vec<ExprId> = (0..4).map(|i| evecs.get(i, 0)).collect();
            (ground, v)
        });

        // Evaluate at x=1
        let e0: f64 = graph.eval(energy, &[1.0]);

        // Reference: eigenvalues of [[1,1,0,0],[1,2,1,0],[0,1,3,1],[0,0,1,4]]
        let ref_mat = tang_la::DMat::<f64>::from_fn(4, 4, |i, j| {
            if i == j {
                [1.0, 2.0, 3.0, 4.0][i]
            } else if (i as i32 - j as i32).abs() == 1 {
                1.0
            } else {
                0.0
            }
        });
        let eig_ref = tang_la::SymmetricEigen::new(&ref_mat);
        let ref_e0 = eig_ref.eigenvalues[0];

        assert!(
            (e0 - ref_e0).abs() < 1e-8,
            "traced eigen mismatch: traced={e0}, ref={ref_e0}"
        );
    }

    /// Test the select-based entropy computation on a small density matrix.
    #[test]
    fn traced_entropy_small() {
        let (graph, entropy) = tang_expr::trace(|| {
            // Build a 2×2 density matrix: [[0.7, 0.1], [0.1, 0.3]]
            let mut rho = DMat::<ExprId>::zeros(2, 2);
            rho.set(0, 0, ExprId::from_f64(0.7));
            rho.set(0, 1, ExprId::from_f64(0.1));
            rho.set(1, 0, ExprId::from_f64(0.1));
            rho.set(1, 1, ExprId::from_f64(0.3));

            let (evals, _) = branchless_jacobi_eigen(&rho, 20);

            let eps = ExprId::from_f64(1e-15);
            let mut s = ExprId::ZERO;
            for i in 0..2 {
                let lam = evals[i];
                let ln_lam = Scalar::ln(lam);
                let contribution = lam * ln_lam;
                let safe = ExprId::select(lam - eps, contribution, ExprId::ZERO);
                s = s - safe;
            }
            s
        });

        let traced_s: f64 = graph.eval(entropy, &[]);

        // Reference: eigenvalues of [[0.7,0.1],[0.1,0.3]]
        let ref_mat = tang_la::DMat::<f64>::from_fn(2, 2, |i, j| {
            [[0.7, 0.1], [0.1, 0.3]][i][j]
        });
        let eig = tang_la::SymmetricEigen::new(&ref_mat);
        let ref_s: f64 = eig
            .eigenvalues
            .iter()
            .filter(|&&l| l > 1e-15)
            .map(|&l| -l * l.ln())
            .sum();

        assert!(
            (traced_s - ref_s).abs() < 1e-8,
            "entropy mismatch: traced={traced_s}, ref={ref_s}"
        );
    }

    /// Test WireGraph roundtrip on traced eigendecomposition.
    #[test]
    #[cfg(feature = "mesh")]
    fn traced_wire_roundtrip() {
        let (graph, energy) = tang_expr::trace(|| {
            let x = ExprId::var(0);
            let mut h = DMat::<ExprId>::zeros(3, 3);
            h.set(0, 0, x);
            h.set(1, 1, ExprId::from_f64(2.0));
            h.set(2, 2, ExprId::from_f64(3.0));
            h.set(0, 1, ExprId::ONE);
            h.set(1, 0, ExprId::ONE);
            h.set(1, 2, ExprId::ONE);
            h.set(2, 1, ExprId::ONE);

            let (evals, _) = branchless_jacobi_eigen(&h, 15);
            evals[0]
        });

        let wire = tang_mesh::protocol::WireGraph::from_expr_graph(&graph, &[energy], 1);
        let (graph2, outputs2) = wire.to_expr_graph();

        let orig: f64 = graph.eval(energy, &[1.5]);
        let rt: f64 = graph2.eval(outputs2[0], &[1.5]);

        assert!(
            (orig - rt).abs() < 1e-10,
            "roundtrip mismatch: orig={orig}, rt={rt}"
        );
    }

    /// Full pipeline test with actual pentachoron (slow, ignored by default).
    #[test]
    #[ignore]
    fn traced_pentachoron_ground_energy() {
        let complex = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);
        let hs = U1HilbertSpace::new(&complex, 1);
        let n_tri = complex.n_triangles();

        let g_sq = 2.0;
        let weights = vec![1.0; n_tri];

        let params = crate::hamiltonian::KSParams {
            g_squared: g_sq,
            metric_weights: Some(weights.clone()),
        };
        let h = crate::hamiltonian::build_hamiltonian(&hs, &complex, &params);
        let spec = crate::diag::diagonalize(&h, Some(1));
        let ref_energy = spec.ground_energy();

        let solver = trace_quantum_entropy(&complex, &hs, &[], 30);

        let mut inputs = vec![g_sq];
        inputs.extend_from_slice(&weights);
        let results = solver.eval(&inputs);

        assert!(
            (results[0] - ref_energy).abs() < 1e-6,
            "ground energy mismatch: traced={}, ref={ref_energy}",
            results[0]
        );
    }
}
