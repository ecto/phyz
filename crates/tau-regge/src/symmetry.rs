//! Known symmetry generators for the Einstein-Maxwell system.
//!
//! A `Generator` is a vector G ∈ R^{2·n_edges} acting on packed fields
//! `[δl..., δθ...]`. For infinitesimal transform φ → φ + ε·G, the action
//! changes by δS = ε·(∇S · G). Exact symmetry iff ∇S · G = 0 for all
//! field configurations.

use crate::action::Fields;
use crate::complex::SimplicialComplex;

/// A symmetry generator acting on packed fields [δl..., δθ...].
#[derive(Debug, Clone)]
pub struct Generator {
    /// Infinitesimal change to edge lengths.
    pub delta_lengths: Vec<f64>,
    /// Infinitesimal change to edge phases.
    pub delta_phases: Vec<f64>,
    /// Human-readable name.
    pub name: String,
}

impl Generator {
    /// Pack into a single flat vector: [delta_lengths..., delta_phases...].
    pub fn pack(&self) -> Vec<f64> {
        let mut v = self.delta_lengths.clone();
        v.extend_from_slice(&self.delta_phases);
        v
    }

    /// Euclidean norm.
    pub fn norm(&self) -> f64 {
        let sum_sq: f64 = self
            .delta_lengths
            .iter()
            .chain(self.delta_phases.iter())
            .map(|x| x * x)
            .sum();
        sum_sq.sqrt()
    }

    /// Return a unit-norm copy.
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n < 1e-30 {
            return self.clone();
        }
        Self {
            delta_lengths: self.delta_lengths.iter().map(|x| x / n).collect(),
            delta_phases: self.delta_phases.iter().map(|x| x / n).collect(),
            name: self.name.clone(),
        }
    }

    /// Dot product with another generator.
    pub fn dot(&self, other: &Generator) -> f64 {
        self.delta_lengths
            .iter()
            .zip(other.delta_lengths.iter())
            .chain(self.delta_phases.iter().zip(other.delta_phases.iter()))
            .map(|(a, b)| a * b)
            .sum()
    }
}

/// U(1) gauge generator at vertex v.
///
/// Under a gauge transformation with parameter ε at vertex v:
///   θ_{ij} → θ_{ij} + ε·(δ_{j,v} − δ_{i,v})
///
/// So δθ_e = +1 if the higher-index endpoint is v, −1 if the lower is v.
/// Edge lengths are unaffected: δl = 0.
pub fn gauge_generator(complex: &SimplicialComplex, vertex: usize) -> Generator {
    let n = complex.n_edges();
    let mut delta_phases = vec![0.0; n];

    for (ei, edge) in complex.edges.iter().enumerate() {
        if edge[0] == vertex {
            delta_phases[ei] = -1.0;
        } else if edge[1] == vertex {
            delta_phases[ei] = 1.0;
        }
    }

    Generator {
        delta_lengths: vec![0.0; n],
        delta_phases,
        name: format!("gauge_v{vertex}"),
    }
}

/// One gauge generator per vertex.
pub fn all_gauge_generators(complex: &SimplicialComplex) -> Vec<Generator> {
    (0..complex.n_vertices)
        .map(|v| gauge_generator(complex, v))
        .collect()
}

/// Conformal (global rescaling) generator: δl_e = l_e, δθ = 0.
///
/// This is NOT a symmetry of 4D Einstein-Hilbert or Maxwell — useful as
/// a control direction that should give non-zero Noether current.
pub fn conformal_generator(complex: &SimplicialComplex, fields: &Fields) -> Generator {
    Generator {
        delta_lengths: fields.lengths.clone(),
        delta_phases: vec![0.0; complex.n_edges()],
        name: "conformal".to_string(),
    }
}

/// T⁴ lattice translation generator along axis μ ∈ {0,1,2,3}.
///
/// For a periodic n×n×n×n mesh, shifts all fields by one lattice site
/// along axis μ. The generator is δφ = φ(shifted) − φ(original).
pub fn translation_generator(
    complex: &SimplicialComplex,
    fields: &Fields,
    axis: usize,
    n: usize,
) -> Generator {
    // Build vertex permutation: shift along axis by +1 (mod n).
    let mut vertex_map = vec![0usize; complex.n_vertices];
    for v in 0..complex.n_vertices {
        let mut coords = vertex_coords_4d(v, n);
        coords[axis] = (coords[axis] + 1) % n;
        vertex_map[v] = vertex_index_4d(&coords, n);
    }

    let n_edges = complex.n_edges();
    let mut delta_lengths = vec![0.0; n_edges];
    let mut delta_phases = vec![0.0; n_edges];

    for (ei, edge) in complex.edges.iter().enumerate() {
        let mut shifted_edge = [vertex_map[edge[0]], vertex_map[edge[1]]];
        shifted_edge.sort_unstable();
        if let Some(&shifted_ei) = complex.edge_index.get(&shifted_edge) {
            delta_lengths[ei] = fields.lengths[shifted_ei] - fields.lengths[ei];
            delta_phases[ei] = fields.phases[shifted_ei] - fields.phases[ei];
        }
    }

    Generator {
        delta_lengths,
        delta_phases,
        name: format!("translation_axis{axis}"),
    }
}

fn vertex_coords_4d(idx: usize, n: usize) -> [usize; 4] {
    [
        idx % n,
        (idx / n) % n,
        (idx / (n * n)) % n,
        idx / (n * n * n),
    ]
}

fn vertex_index_4d(coords: &[usize; 4], n: usize) -> usize {
    coords[0] + n * coords[1] + n * n * coords[2] + n * n * n * coords[3]
}

/// Gram-Schmidt orthonormalization of generators.
///
/// Linearly dependent generators are dropped (norm < 1e-12 after projection).
pub fn orthonormalize(generators: &[Generator]) -> Vec<Generator> {
    if generators.is_empty() {
        return Vec::new();
    }

    let n_edges = generators[0].delta_lengths.len();
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut names: Vec<String> = Vec::new();

    for g in generators {
        let mut v = g.pack();

        // Subtract projections onto existing basis vectors.
        for b in &basis {
            let dot: f64 = v.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
            for (vi, bi) in v.iter_mut().zip(b.iter()) {
                *vi -= dot * bi;
            }
        }

        // Normalize.
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for vi in v.iter_mut() {
                *vi /= norm;
            }
            basis.push(v);
            names.push(g.name.clone());
        }
    }

    basis
        .into_iter()
        .zip(names)
        .map(|(v, name)| Generator {
            delta_lengths: v[..n_edges].to_vec(),
            delta_phases: v[n_edges..].to_vec(),
            name,
        })
        .collect()
}

/// Remove the component of `vector` lying in the span of orthonormal generators.
pub fn project_out_span(vector: &mut [f64], orthonormal_generators: &[Generator]) {
    for g in orthonormal_generators {
        let packed = g.pack();
        let dot: f64 = vector.iter().zip(packed.iter()).map(|(a, b)| a * b).sum();
        for (vi, gi) in vector.iter_mut().zip(packed.iter()) {
            *vi -= dot * gi;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{einstein_maxwell_grad, noether_current, ActionParams};
    use crate::mesh;

    #[test]
    fn test_gauge_noether_current_zero() {
        // Gauge generators should give zero Noether current on any config.
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(123);
        let phases: Vec<f64> = (0..complex.n_edges())
            .map(|_| rng.r#gen::<f64>() * 0.3)
            .collect();

        let fields = Fields::new(lengths, phases);
        let params = ActionParams::default();

        for v in 0..complex.n_vertices {
            let g = gauge_generator(&complex, v);
            let j = noether_current(
                &complex,
                &fields,
                &g.delta_lengths,
                &g.delta_phases,
                &params,
            );
            assert!(
                j.abs() < 1e-10,
                "gauge generator at v{v}: Noether current = {j} (expected ~0)"
            );
        }
    }

    #[test]
    fn test_conformal_noether_current_nonzero() {
        // Conformal scaling is NOT a symmetry on a curved mesh.
        let (complex, lengths) = mesh::reissner_nordstrom(3, 1.0, 0.1, 0.0, 0.5);

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(456);
        let phases: Vec<f64> = (0..complex.n_edges())
            .map(|_| rng.r#gen::<f64>() * 0.1)
            .collect();

        let fields = Fields::new(lengths, phases);
        let params = ActionParams::default();
        let g = conformal_generator(&complex, &fields);

        let j = noether_current(
            &complex,
            &fields,
            &g.delta_lengths,
            &g.delta_phases,
            &params,
        );
        assert!(
            j.abs() > 1e-6,
            "conformal generator: Noether current = {j} (expected non-zero on curved mesh)"
        );
    }

    #[test]
    fn test_orthonormalize() {
        let (complex, _) = mesh::flat_hypercubic(2, 1.0);
        let generators = all_gauge_generators(&complex);
        let ortho = orthonormalize(&generators);

        // Should have n_vertices - 1 independent generators
        // (sum of all gauge generators is zero).
        assert_eq!(
            ortho.len(),
            complex.n_vertices - 1,
            "expected {} orthonormal generators, got {}",
            complex.n_vertices - 1,
            ortho.len()
        );

        // Check orthonormality.
        for (i, gi) in ortho.iter().enumerate() {
            let ni = gi.norm();
            assert!(
                (ni - 1.0).abs() < 1e-10,
                "generator {i} norm = {ni} (expected 1)"
            );
            for (j, gj) in ortho.iter().enumerate() {
                if i != j {
                    let d = gi.dot(gj);
                    assert!(
                        d.abs() < 1e-10,
                        "generators {i},{j} dot = {d} (expected 0)"
                    );
                }
            }
        }

        // Check that the orthonormalized set spans the same subspace:
        // each original generator should be expressible in the ortho basis.
        for g in &generators {
            let packed = g.pack();
            let mut residual = packed.clone();
            for ob in &ortho {
                let ob_packed = ob.pack();
                let dot: f64 = residual.iter().zip(ob_packed.iter()).map(|(a, b)| a * b).sum();
                for (ri, bi) in residual.iter_mut().zip(ob_packed.iter()) {
                    *ri -= dot * bi;
                }
            }
            let res_norm: f64 = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                res_norm < 1e-10,
                "original generator not in span: residual norm = {res_norm}"
            );
        }
    }

    #[test]
    fn test_gauge_generator_sum_zero() {
        // Sum of all gauge generators should be zero (constant gauge = no effect).
        let (complex, _) = mesh::flat_hypercubic(2, 1.0);
        let generators = all_gauge_generators(&complex);

        let n = complex.n_edges();
        let mut sum_phases = vec![0.0; n];
        for g in &generators {
            for (i, &d) in g.delta_phases.iter().enumerate() {
                sum_phases[i] += d;
            }
        }

        for (i, &s) in sum_phases.iter().enumerate() {
            assert!(
                s.abs() < 1e-15,
                "sum of gauge generators at edge {i} = {s} (expected 0)"
            );
        }
    }

    #[test]
    fn test_gauge_invariance_via_gradient_dot() {
        // ∇S · G_gauge = 0 directly (equivalent to Noether current = 0).
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(789);
        let phases: Vec<f64> = (0..complex.n_edges())
            .map(|_| rng.r#gen::<f64>() * 0.2)
            .collect();

        let fields = Fields::new(lengths, phases);
        let params = ActionParams::default();
        let grad = einstein_maxwell_grad(&complex, &fields, &params);

        for v in 0..complex.n_vertices {
            let g = gauge_generator(&complex, v);
            let packed = g.pack();
            let dot: f64 = grad.iter().zip(packed.iter()).map(|(a, b)| a * b).sum();
            assert!(
                dot.abs() < 1e-10,
                "∇S · G_gauge(v{v}) = {dot} (expected 0)"
            );
        }
    }
}
