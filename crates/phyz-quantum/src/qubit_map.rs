//! Qubit encoding and resource estimates for quantum simulation.
//!
//! Maps the U(1) electric basis truncated to [-Λ, Λ] onto qubits,
//! and estimates circuit resources for Trotterized time evolution.

use phyz_regge::SimplicialComplex;

/// Resource estimate for quantum simulation of a simplicial gauge theory.
#[derive(Debug, Clone)]
pub struct ResourceEstimate {
    /// Number of edges in the complex.
    pub n_edges: usize,
    /// Number of vertices.
    pub n_vertices: usize,
    /// Number of triangular plaquettes.
    pub n_triangles: usize,
    /// First Betti number b₁ = E - V + 1 (independent loops).
    pub b1: usize,
    /// Truncation parameter Λ.
    pub lambda: u32,
    /// Qubits per edge: ⌈log₂(2Λ+1)⌉.
    pub qubits_per_edge: u32,
    /// Total physical qubits (edges × qubits_per_edge).
    pub total_qubits: u32,
    /// Gauge-invariant qubits (b₁ × qubits_per_edge).
    pub gauge_invariant_qubits: u32,
    /// Hilbert space dimension (gauge-invariant sector).
    pub hilbert_dim: usize,
    /// Number of Trotter gates per step (electric + magnetic).
    pub trotter_gates_per_step: u64,
}

/// Compute resource estimates for quantum simulation.
pub fn resource_estimate(
    complex: &SimplicialComplex,
    lambda: u32,
    hilbert_dim: usize,
) -> ResourceEstimate {
    let n_edges = complex.n_edges();
    let n_vertices = complex.n_vertices;
    let n_triangles = complex.n_triangles();
    let b1 = n_edges - n_vertices + 1;

    let base = 2 * lambda + 1;
    let qubits_per_edge = (base as f64).log2().ceil() as u32;

    let total_qubits = n_edges as u32 * qubits_per_edge;
    let gauge_invariant_qubits = b1 as u32 * qubits_per_edge;

    // Trotter gate count per step:
    // Electric term: n_edges single-edge gates (diagonal, easy)
    // Magnetic term: n_triangles plaquette gates, each involves 3 edges
    //   Each plaquette gate decomposes into O(qubits_per_edge²) CNOT gates
    let electric_gates = n_edges as u64;
    let magnetic_gates = n_triangles as u64 * (qubits_per_edge as u64).pow(2);
    let trotter_gates_per_step = electric_gates + magnetic_gates;

    ResourceEstimate {
        n_edges,
        n_vertices,
        n_triangles,
        b1,
        lambda,
        qubits_per_edge,
        total_qubits,
        gauge_invariant_qubits,
        hilbert_dim,
        trotter_gates_per_step,
    }
}

impl std::fmt::Display for ResourceEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Resource Estimate")?;
        writeln!(f, "  Vertices:       {}", self.n_vertices)?;
        writeln!(f, "  Edges:          {}", self.n_edges)?;
        writeln!(f, "  Triangles:      {}", self.n_triangles)?;
        writeln!(f, "  b₁:            {}", self.b1)?;
        writeln!(f, "  Λ:             {}", self.lambda)?;
        writeln!(f, "  Qubits/edge:    {}", self.qubits_per_edge)?;
        writeln!(f, "  Total qubits:   {}", self.total_qubits)?;
        writeln!(f, "  GI qubits:      {}", self.gauge_invariant_qubits)?;
        writeln!(f, "  Hilbert dim:    {}", self.hilbert_dim)?;
        writeln!(f, "  Trotter gates:  {}", self.trotter_gates_per_step)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_pentachoron() {
        let complex = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);
        let est = resource_estimate(&complex, 1, 729);

        assert_eq!(est.n_edges, 10);
        assert_eq!(est.n_vertices, 5);
        assert_eq!(est.n_triangles, 10);
        assert_eq!(est.b1, 6);
        assert_eq!(est.qubits_per_edge, 2); // ceil(log2(3)) = 2
        assert_eq!(est.total_qubits, 20);
        assert_eq!(est.gauge_invariant_qubits, 12);
    }

    #[test]
    fn test_lambda_scaling() {
        let complex = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);

        let e1 = resource_estimate(&complex, 1, 100);
        let e3 = resource_estimate(&complex, 3, 1000);

        // Higher Λ needs more qubits per edge.
        assert!(e3.qubits_per_edge >= e1.qubits_per_edge);
        assert!(e3.total_qubits >= e1.total_qubits);
    }

    #[test]
    fn test_display() {
        let complex = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);
        let est = resource_estimate(&complex, 1, 729);
        let s = format!("{est}");
        assert!(s.contains("Resource Estimate"));
        assert!(s.contains("10")); // edges
    }
}
