//! Trace the quantum solver as a WireGraph and compare against the existing solver.
//!
//! Demonstrates:
//! 1. Building the full quantum computation as an expression graph
//! 2. Serializing as WireGraph for distributed execution
//! 3. Evaluating with concrete inputs
//! 4. Comparing against the existing dense solver

use phyz_quantum::hilbert::U1HilbertSpace;
use phyz_quantum::traced::trace_quantum_entropy;
use phyz_regge::SimplicialComplex;

fn main() {
    println!("=== Traced Quantum Solver → WireGraph ===\n");

    let complex = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);
    let hs = U1HilbertSpace::new(&complex, 1);
    let n_tri = complex.n_triangles();

    println!("Complex: single pentachoron");
    println!("  vertices: {}", complex.n_vertices);
    println!("  edges: {}", complex.n_edges());
    println!("  triangles: {n_tri}");
    println!("  Hilbert space dim: {}", hs.dim());
    println!();

    // Reference computation with existing solver
    let g_sq = 2.0;
    let weights = vec![1.0; n_tri];
    let params = phyz_quantum::KSParams {
        g_squared: g_sq,
        metric_weights: Some(weights.clone()),
    };
    let h = phyz_quantum::hamiltonian::build_hamiltonian(&hs, &complex, &params);
    let spec = phyz_quantum::diag::diagonalize(&h, Some(1));
    println!("Reference ground energy: {:.10}", spec.ground_energy());
    println!();

    // Trace the solver (no partitions for speed)
    println!("Tracing quantum solver with 20 Jacobi sweeps...");
    let solver = trace_quantum_entropy(&complex, &hs, &[], 20);
    println!("  Graph nodes: {}", solver.node_count());
    println!("  Input vars: {}", solver.n_inputs);
    println!("  Outputs: {}", solver.all_outputs().len());
    println!();

    // Evaluate
    let mut inputs = vec![g_sq];
    inputs.extend_from_slice(&weights);
    let results = solver.eval(&inputs);
    let traced_energy = results[0];
    println!("Traced ground energy:    {:.10}", traced_energy);
    println!(
        "Error: {:.2e}",
        (traced_energy - spec.ground_energy()).abs()
    );
    println!();

    // Serialize as WireGraph
    let wire = solver.to_wire_graph();
    let bytes = wire.to_bytes().unwrap();
    println!("WireGraph serialized: {} bytes ({:.1} KB)", bytes.len(), bytes.len() as f64 / 1024.0);

    // Roundtrip
    let wire2 = tang_mesh::protocol::WireGraph::from_bytes(&bytes).unwrap();
    let (graph2, outputs2) = wire2.to_expr_graph();
    let rt_results: Vec<f64> = graph2.eval_many(&outputs2, &inputs);
    println!(
        "Roundtrip energy:        {:.10} (diff: {:.2e})",
        rt_results[0],
        (rt_results[0] - traced_energy).abs()
    );

    // WGSL generation check
    let wgsl = wire.to_wgsl();
    println!("\nWGSL shader: {} lines, {} bytes",
        wgsl.source.lines().count(),
        wgsl.source.len()
    );
    println!("  workgroup_size: {}", wgsl.workgroup_size);
    println!("  n_inputs: {}", wgsl.n_inputs);
    println!("  n_outputs: {}", wgsl.n_outputs);
}
