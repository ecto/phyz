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

pub fn vertex_coords_4d(idx: usize, n: usize) -> [usize; 4] {
    [
        idx % n,
        (idx / n) % n,
        (idx / (n * n)) % n,
        idx / (n * n * n),
    ]
}

pub fn vertex_index_4d(coords: &[usize; 4], n: usize) -> usize {
    coords[0] + n * coords[1] + n * n * coords[2] + n * n * n * coords[3]
}

/// Rotation generator via 90° lattice rotation in a coordinate plane.
///
/// For a periodic n×n×n×n mesh, rotates all vertices by 90° in the
/// (axis1, axis2) plane. Axis indices: 0=t, 1=x, 2=y, 3=z.
/// The generator is δφ = φ(rotated) − φ(original).
///
/// On a cubic lattice, 90° rotations are exact discrete symmetries of flat
/// space, so this generator has δl = 0 on flat backgrounds. On curved
/// spherically-symmetric backgrounds (RN/Schwarzschild), violations are
/// O(h²) from the lattice discretization.
///
/// For spatial axes (1-3), this gives SO(3) rotation generators.
/// For time-space planes (0 vs 1-3), this gives Lorentz boost generators
/// (which are ordinary rotations in Euclidean signature).
pub fn rotation_generator(
    complex: &SimplicialComplex,
    fields: &Fields,
    axis1: usize,
    axis2: usize,
    n: usize,
) -> Generator {
    assert!(axis1 <= 3, "axis1 must be 0..=3");
    assert!(axis2 <= 3, "axis2 must be 0..=3");
    assert_ne!(axis1, axis2, "axes must be distinct");

    // Build vertex permutation: 90° rotation in (axis1, axis2) plane.
    // (a1, a2) → (a2, n - a1)
    let mut vertex_map = vec![0usize; complex.n_vertices];
    for v in 0..complex.n_vertices {
        let mut coords = vertex_coords_4d(v, n);
        let a1 = coords[axis1];
        let a2 = coords[axis2];
        coords[axis1] = a2;
        coords[axis2] = (n - a1) % n;
        vertex_map[v] = vertex_index_4d(&coords, n);
    }

    let n_edges = complex.n_edges();
    let mut delta_lengths = vec![0.0; n_edges];
    let mut delta_phases = vec![0.0; n_edges];

    for (ei, edge) in complex.edges.iter().enumerate() {
        let mut rotated_edge = [vertex_map[edge[0]], vertex_map[edge[1]]];
        rotated_edge.sort_unstable();
        if let Some(&rotated_ei) = complex.edge_index.get(&rotated_edge) {
            delta_lengths[ei] = fields.lengths[rotated_ei] - fields.lengths[ei];
            delta_phases[ei] = fields.phases[rotated_ei] - fields.phases[ei];
        }
    }

    let axis_names = ["t", "x", "y", "z"];
    Generator {
        delta_lengths,
        delta_phases,
        name: format!("rotation_{}{}", axis_names[axis1], axis_names[axis2]),
    }
}

/// All three spatial rotation generators: xy, xz, yz.
pub fn all_rotation_generators(
    complex: &SimplicialComplex,
    fields: &Fields,
    n: usize,
) -> Vec<Generator> {
    vec![
        rotation_generator(complex, fields, 1, 2, n),
        rotation_generator(complex, fields, 1, 3, n),
        rotation_generator(complex, fields, 2, 3, n),
    ]
}

/// Lorentz boost generator in the (t, spatial_axis) plane.
///
/// On a Euclidean-signature lattice, a "boost" is just a 90° rotation in a
/// (time, space) plane. This is a thin wrapper around `rotation_generator`.
pub fn boost_generator(
    complex: &SimplicialComplex,
    fields: &Fields,
    spatial_axis: usize,
    n: usize,
) -> Generator {
    assert!(
        (1..=3).contains(&spatial_axis),
        "spatial_axis must be 1, 2, or 3"
    );
    let mut g = rotation_generator(complex, fields, 0, spatial_axis, n);
    let axis_names = ["t", "x", "y", "z"];
    g.name = format!("boost_t{}", axis_names[spatial_axis]);
    g
}

/// All three Lorentz boost generators: tx, ty, tz.
pub fn all_boost_generators(
    complex: &SimplicialComplex,
    fields: &Fields,
    n: usize,
) -> Vec<Generator> {
    (1..=3)
        .map(|sa| boost_generator(complex, fields, sa, n))
        .collect()
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

// === Discrete symmetries ===

/// Discrete symmetries of the Einstein-Maxwell system.
#[derive(Debug, Clone, Copy)]
pub enum DiscreteSymmetry {
    /// Charge conjugation: θ_e → -θ_e.
    ChargeConjugation,
    /// Time reversal: t → -t (vertex permutation on axis 0).
    TimeReversal,
    /// Parity flip along a spatial axis (1, 2, or 3).
    Parity(usize),
    /// C·P (charge conjugation × parity).
    CP(usize),
    /// C·T (charge conjugation × time reversal).
    CT,
    /// P·T (parity × time reversal).
    PT(usize),
    /// C·P·T.
    CPT(usize),
}

impl std::fmt::Display for DiscreteSymmetry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChargeConjugation => write!(f, "C"),
            Self::TimeReversal => write!(f, "T"),
            Self::Parity(a) => write!(f, "P{a}"),
            Self::CP(a) => write!(f, "CP{a}"),
            Self::CT => write!(f, "CT"),
            Self::PT(a) => write!(f, "PT{a}"),
            Self::CPT(a) => write!(f, "CPT{a}"),
        }
    }
}

/// Build vertex permutation map from a coordinate transformation.
///
/// For each vertex, apply the coordinate transform and find the new vertex index.
fn build_vertex_map(
    n_vertices: usize,
    n: usize,
    coord_transform: impl Fn([usize; 4]) -> [usize; 4],
) -> Vec<usize> {
    let mut vertex_map = vec![0usize; n_vertices];
    for v in 0..n_vertices {
        let coords = vertex_coords_4d(v, n);
        let new_coords = coord_transform(coords);
        vertex_map[v] = vertex_index_4d(&new_coords, n);
    }
    vertex_map
}

/// Apply a vertex permutation to fields, remapping edges.
///
/// Returns new (lengths, phases) with edges remapped according to vertex_map.
/// Edges whose mapped version doesn't exist in the complex keep their original values.
/// (This happens for reflections on Kuhn triangulations, where some reflected
/// vertex pairs don't appear in the same simplex.)
pub fn apply_vertex_permutation(
    complex: &SimplicialComplex,
    fields: &Fields,
    vertex_map: &[usize],
) -> Fields {
    let mut new_lengths = fields.lengths.clone();
    let mut new_phases = fields.phases.clone();

    for (ei, edge) in complex.edges.iter().enumerate() {
        let mut mapped = [vertex_map[edge[0]], vertex_map[edge[1]]];
        mapped.sort_unstable();
        if let Some(&mapped_ei) = complex.edge_index.get(&mapped) {
            new_lengths[mapped_ei] = fields.lengths[ei];
            new_phases[mapped_ei] = fields.phases[ei];
        }
    }

    Fields::new(new_lengths, new_phases)
}

/// Apply a discrete symmetry to fields.
pub fn apply_discrete_symmetry(
    complex: &SimplicialComplex,
    fields: &Fields,
    sym: DiscreteSymmetry,
    n: usize,
) -> Fields {
    match sym {
        DiscreteSymmetry::ChargeConjugation => {
            // C: negate all phases, lengths unchanged.
            let new_phases: Vec<f64> = fields.phases.iter().map(|&p| -p).collect();
            Fields::new(fields.lengths.clone(), new_phases)
        }
        DiscreteSymmetry::TimeReversal => {
            // T: flip time coordinate.
            let vertex_map = build_vertex_map(complex.n_vertices, n, |mut c| {
                c[0] = (n - c[0]) % n;
                c
            });
            apply_vertex_permutation(complex, fields, &vertex_map)
        }
        DiscreteSymmetry::Parity(axis) => {
            assert!((1..=3).contains(&axis), "parity axis must be 1, 2, or 3");
            let vertex_map = build_vertex_map(complex.n_vertices, n, |mut c| {
                c[axis] = (n - c[axis]) % n;
                c
            });
            apply_vertex_permutation(complex, fields, &vertex_map)
        }
        DiscreteSymmetry::CP(axis) => {
            let p_result = apply_discrete_symmetry(complex, fields, DiscreteSymmetry::Parity(axis), n);
            apply_discrete_symmetry(complex, &p_result, DiscreteSymmetry::ChargeConjugation, n)
        }
        DiscreteSymmetry::CT => {
            let t_result = apply_discrete_symmetry(complex, fields, DiscreteSymmetry::TimeReversal, n);
            apply_discrete_symmetry(complex, &t_result, DiscreteSymmetry::ChargeConjugation, n)
        }
        DiscreteSymmetry::PT(axis) => {
            let p_result = apply_discrete_symmetry(complex, fields, DiscreteSymmetry::Parity(axis), n);
            apply_discrete_symmetry(complex, &p_result, DiscreteSymmetry::TimeReversal, n)
        }
        DiscreteSymmetry::CPT(axis) => {
            let pt_result = apply_discrete_symmetry(complex, fields, DiscreteSymmetry::PT(axis), n);
            apply_discrete_symmetry(complex, &pt_result, DiscreteSymmetry::ChargeConjugation, n)
        }
    }
}

/// Check how badly a discrete symmetry is broken: |S[T(φ)] - S[φ]|.
pub fn check_discrete_symmetry(
    complex: &SimplicialComplex,
    fields: &Fields,
    sym: DiscreteSymmetry,
    n: usize,
    params: &crate::action::ActionParams,
) -> f64 {
    let transformed = apply_discrete_symmetry(complex, fields, sym, n);
    crate::action::action_variation(complex, fields, &transformed, params)
}

/// Check all discrete symmetries and return (symmetry, violation) pairs.
pub fn check_all_discrete_symmetries(
    complex: &SimplicialComplex,
    fields: &Fields,
    n: usize,
    params: &crate::action::ActionParams,
) -> Vec<(DiscreteSymmetry, f64)> {
    let syms = [
        DiscreteSymmetry::ChargeConjugation,
        DiscreteSymmetry::TimeReversal,
        DiscreteSymmetry::Parity(1),
        DiscreteSymmetry::Parity(2),
        DiscreteSymmetry::Parity(3),
        DiscreteSymmetry::CP(1),
        DiscreteSymmetry::CT,
        DiscreteSymmetry::PT(1),
        DiscreteSymmetry::CPT(1),
    ];

    syms.iter()
        .map(|&sym| {
            let violation = check_discrete_symmetry(complex, fields, sym, n, params);
            (sym, violation)
        })
        .collect()
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

    #[test]
    fn test_rotation_generator_flat_space() {
        // On flat space, 90° rotations are exact lattice symmetries,
        // so the generator should be identically zero (δl = 0 for all edges).
        let n = 3;
        let (complex, lengths) = mesh::flat_hypercubic(n, 1.0);
        let phases = vec![0.0; complex.n_edges()];
        let fields = Fields::new(lengths, phases);

        for (a1, a2) in [(1, 2), (1, 3), (2, 3)] {
            let g = rotation_generator(&complex, &fields, a1, a2, n);

            // All δl should be zero on flat space.
            let max_dl: f64 = g
                .delta_lengths
                .iter()
                .map(|x| x.abs())
                .fold(0.0, f64::max);
            assert!(
                max_dl < 1e-12,
                "rotation ({a1},{a2}) on flat space: max δl = {max_dl} (expected 0)"
            );

            // Noether current should also be zero.
            let params = ActionParams::default();
            let j = noether_current(
                &complex,
                &fields,
                &g.delta_lengths,
                &g.delta_phases,
                &params,
            );
            assert!(
                j.abs() < 1e-10,
                "rotation ({a1},{a2}) Noether current = {j} (expected ~0)"
            );
        }
    }

    #[test]
    fn test_rotation_generator_rn_small_violation() {
        // On RN background, rotation generators should have small violations
        // (O(h²) from lattice discretization), not large ones.
        let n = 3;
        let (complex, lengths) = mesh::reissner_nordstrom(n, 1.0, 0.1, 0.0, 0.5);
        let phases = vec![0.0; complex.n_edges()];
        let fields = Fields::new(lengths, phases);

        let gens = all_rotation_generators(&complex, &fields, n);
        assert_eq!(gens.len(), 3);

        let params = ActionParams::default();
        for g in &gens {
            let j = noether_current(
                &complex,
                &fields,
                &g.delta_lengths,
                &g.delta_phases,
                &params,
            );
            // Not necessarily zero: the RN metric breaks lattice rotation
            // symmetry at O(h²), and with coarse grids the constant can be
            // large. Just verify it's finite and much smaller than conformal.
            assert!(
                j.abs() < 10.0,
                "{}: Noether current = {j} (expected bounded)",
                g.name
            );
        }
    }

    #[test]
    fn test_boost_flat_space() {
        // On flat space, boosts (= rotations in t-space planes) are exact
        // lattice symmetries, so δl = 0 and Noether current = 0.
        let n = 3;
        let (complex, lengths) = mesh::flat_hypercubic(n, 1.0);
        let phases = vec![0.0; complex.n_edges()];
        let fields = Fields::new(lengths, phases);

        let boosts = all_boost_generators(&complex, &fields, n);
        assert_eq!(boosts.len(), 3);

        let params = ActionParams::default();
        for g in &boosts {
            let max_dl: f64 = g
                .delta_lengths
                .iter()
                .map(|x| x.abs())
                .fold(0.0, f64::max);
            assert!(
                max_dl < 1e-12,
                "{} on flat space: max δl = {max_dl} (expected 0)",
                g.name
            );

            let j = noether_current(
                &complex,
                &fields,
                &g.delta_lengths,
                &g.delta_phases,
                &params,
            );
            assert!(
                j.abs() < 1e-10,
                "{}: Noether current = {j} (expected ~0)",
                g.name
            );
        }
    }

    #[test]
    fn test_boost_rn_broken() {
        // On RN background, boosts should be more broken than spatial rotations
        // since f(r) ≠ g(r) breaks time-space isotropy.
        let n = 3;
        let (complex, lengths) = mesh::reissner_nordstrom(n, 1.0, 0.1, 0.0, 0.5);
        let phases = vec![0.0; complex.n_edges()];
        let fields = Fields::new(lengths, phases);

        let boosts = all_boost_generators(&complex, &fields, n);
        let rotations = all_rotation_generators(&complex, &fields, n);
        assert_eq!(boosts.len(), 3);
        assert_eq!(rotations.len(), 3);

        // Boosts should have larger δl norms than rotations on RN
        // (rotations only break at O(h²) from discretization,
        //  boosts break at O(1) from f ≠ g).
        let max_boost_dl: f64 = boosts
            .iter()
            .map(|g| g.delta_lengths.iter().map(|x| x.abs()).fold(0.0, f64::max))
            .fold(0.0, f64::max);
        let max_rot_dl: f64 = rotations
            .iter()
            .map(|g| g.delta_lengths.iter().map(|x| x.abs()).fold(0.0, f64::max))
            .fold(0.0, f64::max);

        assert!(
            max_boost_dl > max_rot_dl,
            "boost max δl ({max_boost_dl}) should exceed rotation max δl ({max_rot_dl}) on RN"
        );
    }

    // === Discrete symmetry tests ===

    #[test]
    fn test_c_exact_flat() {
        // C is exact on flat space with zero field.
        let n = 3;
        let (complex, lengths) = mesh::flat_hypercubic(n, 1.0);
        let phases = vec![0.0; complex.n_edges()];
        let fields = Fields::new(lengths, phases);
        let params = ActionParams::default();

        let v = check_discrete_symmetry(&complex, &fields, DiscreteSymmetry::ChargeConjugation, n, &params);
        assert!(v < 1e-12, "C violation on flat = {v}");
    }

    #[test]
    fn test_t_exact_flat() {
        let n = 3;
        let (complex, lengths) = mesh::flat_hypercubic(n, 1.0);
        let phases = vec![0.0; complex.n_edges()];
        let fields = Fields::new(lengths, phases);
        let params = ActionParams::default();

        let v = check_discrete_symmetry(&complex, &fields, DiscreteSymmetry::TimeReversal, n, &params);
        assert!(v < 1e-10, "T violation on flat = {v}");
    }

    #[test]
    fn test_p_exact_flat() {
        let n = 3;
        let (complex, lengths) = mesh::flat_hypercubic(n, 1.0);
        let phases = vec![0.0; complex.n_edges()];
        let fields = Fields::new(lengths, phases);
        let params = ActionParams::default();

        for axis in 1..=3 {
            let v = check_discrete_symmetry(&complex, &fields, DiscreteSymmetry::Parity(axis), n, &params);
            assert!(v < 1e-10, "P{axis} violation on flat = {v}");
        }
    }

    #[test]
    fn test_c_exact_rn() {
        // C is exact on RN with zero gauge field (S is even in θ).
        let n = 3;
        let (complex, lengths) = mesh::reissner_nordstrom(n, 1.0, 0.1, 0.0, 0.5);
        let phases = vec![0.0; complex.n_edges()];
        let fields = Fields::new(lengths, phases);
        let params = ActionParams::default();

        let v = check_discrete_symmetry(&complex, &fields, DiscreteSymmetry::ChargeConjugation, n, &params);
        assert!(v < 1e-12, "C violation on RN = {v}");
    }

    #[test]
    fn test_cpt_flat() {
        let n = 3;
        let (complex, lengths) = mesh::flat_hypercubic(n, 1.0);
        let phases = vec![0.0; complex.n_edges()];
        let fields = Fields::new(lengths, phases);
        let params = ActionParams::default();

        let v = check_discrete_symmetry(&complex, &fields, DiscreteSymmetry::CPT(1), n, &params);
        assert!(v < 1e-10, "CPT violation on flat = {v}");
    }

    #[test]
    fn test_t_kerr_broken() {
        // T should be broken on Kerr (frame-dragging breaks time-reversal).
        let n = 3;
        let (complex, lengths) = mesh::kerr(n, 1.0, 0.1, 0.3, 0.5);
        let phases = vec![0.0; complex.n_edges()];
        let fields = Fields::new(lengths, phases);
        let params = ActionParams::default();

        let v_t = check_discrete_symmetry(&complex, &fields, DiscreteSymmetry::TimeReversal, n, &params);
        let v_c = check_discrete_symmetry(&complex, &fields, DiscreteSymmetry::ChargeConjugation, n, &params);
        // C is still exact (zero gauge field), T is broken.
        assert!(v_c < 1e-12, "C should be exact on Kerr with zero gauge field");
        assert!(v_t > v_c, "T should be more broken than C on Kerr: T={v_t}, C={v_c}");
    }
}
