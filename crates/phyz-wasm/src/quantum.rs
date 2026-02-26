use wasm_bindgen::prelude::*;

use phyz_quantum::jacobson::{boundary_5simplex, su2_ground_state_with_energy, subdivided_s4};
use phyz_quantum::ryu_takayanagi::{
    cut_area_geometric, perturbed_edge_lengths, vertex_bipartitions,
};
use phyz_quantum::su2_quantum::{Su2HilbertSpace, su2_entanglement_for_partition};
use phyz_regge::complex::SimplicialComplex;

#[wasm_bindgen]
pub struct QuantumSolver {
    complex: SimplicialComplex,
    hilbert: Su2HilbertSpace,
}

#[wasm_bindgen]
impl QuantumSolver {
    #[wasm_bindgen(constructor)]
    pub fn new(triangulation: &str) -> Result<QuantumSolver, JsError> {
        let complex = match triangulation {
            "s4_level0" => boundary_5simplex(),
            "s4_level1" => subdivided_s4(1),
            "s4_level2" => subdivided_s4(2),
            "s4_level3" => subdivided_s4(3),
            other => return Err(JsError::new(&format!("unknown triangulation: {other}"))),
        };
        let hilbert = Su2HilbertSpace::new(&complex);
        Ok(QuantumSolver { complex, hilbert })
    }

    /// Solve for all partitions at once. Returns JSON with ground_state_energy,
    /// entropy_per_partition, and walltime_ms.
    pub fn solve_all_partitions(
        &self,
        coupling_g2: f64,
        geometry_seed: u64,
        perturbation_type: &str,
        perturbation_index: i32,
        perturbation_direction: f64,
        fd_epsilon: f64,
    ) -> Result<String, JsError> {
        let start = js_sys::Date::now();

        // Generate base edge lengths from seed
        let mut lengths = perturbed_edge_lengths(&self.complex, 0.1, geometry_seed);

        // Apply edge perturbation if requested
        if perturbation_type == "edge" {
            let idx = perturbation_index as usize;
            if idx >= lengths.len() {
                return Err(JsError::new(&format!(
                    "edge index {} out of range ({})",
                    idx,
                    lengths.len()
                )));
            }
            lengths[idx] += perturbation_direction * fd_epsilon;
        }

        // Single eigensolve
        let (gs, energy) =
            su2_ground_state_with_energy(&self.hilbert, &self.complex, &lengths, coupling_g2);

        // Compute entropy for every non-trivial bipartition
        let partitions = vertex_bipartitions(self.complex.n_vertices);
        let entropies: Vec<f64> = partitions
            .iter()
            .map(|part| su2_entanglement_for_partition(&self.hilbert, &gs, &self.complex, part))
            .collect();

        let areas: Vec<f64> = partitions
            .iter()
            .map(|part| cut_area_geometric(&self.complex, part, &lengths))
            .collect();

        let walltime_ms = js_sys::Date::now() - start;

        let result = serde_json::json!({
            "ground_state_energy": energy,
            "entropy_per_partition": entropies,
            "boundary_area_per_partition": areas,
            "walltime_ms": walltime_ms,
        });

        Ok(result.to_string())
    }

    pub fn info(&self) -> String {
        serde_json::json!({
            "n_vertices": self.complex.n_vertices,
            "n_edges": self.complex.n_edges(),
            "n_triangles": self.complex.n_triangles(),
            "n_pentachora": self.complex.n_pents(),
            "dim": self.hilbert.dim(),
        })
        .to_string()
    }
}
