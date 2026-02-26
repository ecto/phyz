//! SU(2) Yang-Mills gauge field on a simplicial complex.
//!
//! Each edge carries an SU(2) group element (unit quaternion).
//! The Wilson action on each triangle is:
//!
//!   S_YM = Σ_t (1 - Re(Tr(U_t))/2) · W_t
//!
//! where U_t is the oriented holonomy around the triangle and W_t is
//! the metric weight (same as for U(1)).

use crate::complex::SimplicialComplex;
use crate::gauge::{GaugeField, metric_weights};
use crate::su2::Su2;

/// SU(2) gauge field: one SU(2) group element per edge.
#[derive(Debug, Clone)]
pub struct Su2Field(pub Vec<Su2>);

impl GaugeField for Su2Field {
    const DOF_PER_EDGE: usize = 3;

    fn flat(n_edges: usize) -> Self {
        Su2Field(vec![Su2::identity(); n_edges])
    }

    fn action(complex: &SimplicialComplex, lengths: &[f64], field: &Self) -> f64 {
        wilson_action(complex, lengths, &field.0)
    }

    fn grad_field(complex: &SimplicialComplex, lengths: &[f64], field: &Self) -> Vec<f64> {
        wilson_action_grad_field(complex, lengths, &field.0)
    }

    fn grad_lengths(complex: &SimplicialComplex, lengths: &[f64], field: &Self) -> Vec<f64> {
        wilson_action_grad_lengths(complex, lengths, &field.0)
    }

    fn pack(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.0.len() * 3);
        for u in &self.0 {
            let theta = u.log();
            v.extend_from_slice(&theta);
        }
        v
    }

    fn unpack(flat: &[f64], n_edges: usize) -> Self {
        let mut elements = Vec::with_capacity(n_edges);
        for i in 0..n_edges {
            let theta = [flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]];
            elements.push(Su2::exp(&theta));
        }
        Su2Field(elements)
    }

    fn n_gauge_generators_per_vertex() -> usize {
        3
    }

    fn gauge_generator(complex: &SimplicialComplex, vertex: usize, direction: usize) -> Vec<f64> {
        assert!(direction < 3, "SU(2) has 3 gauge directions");
        let n = complex.n_edges();
        let mut g = vec![0.0; 3 * n];
        for (ei, edge) in complex.edges.iter().enumerate() {
            if edge[0] == vertex {
                g[3 * ei + direction] = -1.0;
            } else if edge[1] == vertex {
                g[3 * ei + direction] = 1.0;
            }
        }
        g
    }
}

/// Compute the oriented holonomy around a triangle.
///
/// For triangle [v0, v1, v2] with v0 < v1 < v2:
///   U_t = U_{01} · U_{12} · U_{02}^{-1}
fn triangle_holonomy(complex: &SimplicialComplex, elements: &[Su2], tri: &[usize; 3]) -> Su2 {
    let e01 = complex.edge_index[&[tri[0], tri[1]]];
    let e02 = complex.edge_index[&[tri[0], tri[2]]];
    let e12 = complex.edge_index[&[tri[1], tri[2]]];

    // U_t = U_{01} · U_{12} · U_{02}^{-1}
    elements[e01].mul(&elements[e12]).mul(&elements[e02].inv())
}

/// Wilson action for SU(2) gauge field.
///
///   S_YM = Σ_t (1 - Re(Tr(U_t))/2) · W_t
pub fn wilson_action(complex: &SimplicialComplex, lengths: &[f64], elements: &[Su2]) -> f64 {
    let w = metric_weights(complex, lengths);
    let mut action = 0.0;

    for (ti, tri) in complex.triangles.iter().enumerate() {
        let u_t = triangle_holonomy(complex, elements, tri);
        let plaquette = 1.0 - u_t.re_trace() / 2.0;
        action += plaquette * w[ti];
    }

    action
}

/// Gradient of Wilson action w.r.t. SU(2) field DOF (left-invariant basis).
///
/// Uses left-invariant variations: δU_e = ε^a · (e_a/2) · U_e.
/// For each triangle t with holonomy U_t = U01 · U12 · U02^{-1}:
///
/// ∂S_t/∂ε^a_{01} = W_t · q_a(U_t) / 2
/// ∂S_t/∂ε^a_{12} = W_t · q_a(U12 · U02^{-1} · U01) / 2
/// ∂S_t/∂ε^a_{02} = -W_t · q_a(U_t) / 2
///
/// where q_a(V) is the a-th imaginary quaternion component of V.
pub fn wilson_action_grad_field(
    complex: &SimplicialComplex,
    lengths: &[f64],
    elements: &[Su2],
) -> Vec<f64> {
    let w = metric_weights(complex, lengths);
    let n_edges = complex.n_edges();
    let mut grad = vec![0.0; 3 * n_edges];

    for (ti, tri) in complex.triangles.iter().enumerate() {
        let e01 = complex.edge_index[&[tri[0], tri[1]]];
        let e02 = complex.edge_index[&[tri[0], tri[2]]];
        let e12 = complex.edge_index[&[tri[1], tri[2]]];

        let u01 = &elements[e01];
        let u12 = &elements[e12];
        let u02 = &elements[e02];

        let wt = w[ti];

        // Holonomy: U_t = U01 · U12 · U02^{-1}
        let u_t = u01.mul(&u12.mul(&u02.inv()));

        // For e01 (left variation): ∂S_t/∂ε^a = W_t/2 · q_a(U_t)
        for a in 0..3 {
            grad[3 * e01 + a] += wt * 0.5 * u_t.q[a + 1];
        }

        // For e12 (left variation): cyclic shift → V12 = U12 · U02^{-1} · U01
        let v12 = u12.mul(&u02.inv().mul(u01));
        for a in 0..3 {
            grad[3 * e12 + a] += wt * 0.5 * v12.q[a + 1];
        }

        // For e02 (enters as inverse): ∂S_t/∂ε^a = -W_t/2 · q_a(U_t)
        for a in 0..3 {
            grad[3 * e02 + a] -= wt * 0.5 * u_t.q[a + 1];
        }
    }

    grad
}

/// Gradient of Wilson action w.r.t. edge lengths.
///
/// Same weight-gradient structure as U(1), with F_t²/2 replaced by (1 - Re(Tr)/2).
pub fn wilson_action_grad_lengths(
    complex: &SimplicialComplex,
    lengths: &[f64],
    elements: &[Su2],
) -> Vec<f64> {
    use crate::geometry::{pent_volume_grad, triangle_area, triangle_area_grad_lsq};

    let n_edges = complex.n_edges();
    let n_tris = complex.n_triangles();
    let n_pents = complex.n_pents();
    let mut grad = vec![0.0; n_edges];

    let w = metric_weights(complex, lengths);

    // Precompute plaquette values.
    let mut plaquettes = vec![0.0; n_tris];
    for (ti, tri) in complex.triangles.iter().enumerate() {
        let u_t = triangle_holonomy(complex, elements, tri);
        plaquettes[ti] = 1.0 - u_t.re_trace() / 2.0;
    }

    // Precompute triangle areas.
    let mut areas = vec![0.0; n_tris];
    for ti in 0..n_tris {
        let [a, b, c] = complex.tri_edge_lengths(ti, lengths);
        areas[ti] = triangle_area(a, b, c);
    }

    // Precompute volume gradients.
    let mut pent_vol_grads: Vec<[f64; 10]> = Vec::with_capacity(n_pents);
    let mut pent_edge_idx: Vec<[usize; 10]> = Vec::with_capacity(n_pents);
    for pi in 0..n_pents {
        let pl = complex.pent_edge_lengths(pi, lengths);
        pent_vol_grads.push(pent_volume_grad(&pl));
        pent_edge_idx.push(complex.pent_edge_indices(pi));
    }

    // Term 1: volume gradient contribution.
    for ti in 0..n_tris {
        let area = areas[ti];
        if area < 1e-30 {
            continue;
        }
        let coeff = plaquettes[ti] / area;

        for &pi in &complex.tri_to_pents[ti] {
            let vol_grad = &pent_vol_grads[pi];
            let edge_idx = &pent_edge_idx[pi];
            for k in 0..10 {
                grad[edge_idx[k]] += coeff * (1.0 / 10.0) * vol_grad[k];
            }
        }
    }

    // Term 2: area gradient contribution.
    for ti in 0..n_tris {
        let area = areas[ti];
        if area < 1e-30 {
            continue;
        }
        let coeff = -plaquettes[ti] * w[ti] / area;

        let [a, b, c] = complex.tri_edge_lengths(ti, lengths);
        let da_dlsq = triangle_area_grad_lsq(a, b, c);
        let edge_indices = complex.tri_edge_indices(ti);
        let tri_lengths = [a, b, c];

        for k in 0..3 {
            let da_dl = da_dlsq[k] * 2.0 * tri_lengths[k];
            grad[edge_indices[k]] += coeff * da_dl;
        }
    }

    grad
}

/// Compute the field-dependent SU(2) gauge generator at a given vertex and direction.
///
/// In the left-invariant parametrization (U_e → exp(ε·T_a)·U_e), the gauge
/// transformation at vertex v has a non-trivial adjoint action on edges where
/// v is the target vertex:
///
/// - Edge (v, j): δε^a = +λ^a  (left multiplication)
/// - Edge (i, v): δε^b = -Ad(U_e)^{ba}·λ^a  (right multiplication → adjoint)
pub fn su2_gauge_generator(
    complex: &SimplicialComplex,
    elements: &[Su2],
    vertex: usize,
    direction: usize,
) -> Vec<f64> {
    assert!(direction < 3, "SU(2) has 3 gauge directions");
    let n = complex.n_edges();
    let mut g = vec![0.0; 3 * n];

    for (ei, edge) in complex.edges.iter().enumerate() {
        if edge[0] == vertex {
            // Left multiplication: δε^a = +λ^a
            g[3 * ei + direction] += 1.0;
        } else if edge[1] == vertex {
            // Right multiplication: δε^b = -Ad(U_e)^{ba}·λ^a
            let adj = elements[ei].adjoint();
            for b in 0..3 {
                g[3 * ei + b] -= adj[b][direction];
            }
        }
    }

    g
}

/// Build all field-dependent SU(2) gauge generators (3 per vertex).
///
/// Returns generators in the full DOF space (n_edges lengths + 3·n_edges field).
pub fn all_su2_gauge_generators(
    complex: &SimplicialComplex,
    elements: &[Su2],
) -> Vec<crate::symmetry::Generator> {
    let n_edges = complex.n_edges();
    let mut generators = Vec::with_capacity(3 * complex.n_vertices);

    for v in 0..complex.n_vertices {
        for dir in 0..3 {
            let gauge_dof = su2_gauge_generator(complex, elements, v, dir);
            generators.push(crate::symmetry::Generator {
                name: format!("su2_gauge_v{v}_d{dir}"),
                delta_lengths: vec![0.0; n_edges],
                delta_phases: gauge_dof,
            });
        }
    }

    generators
}

/// Translation generator for SU(2) Einstein-Yang-Mills on a periodic lattice.
///
/// Shifts all fields by one lattice site along the given axis.
/// The SU(2) field transport uses left-invariant differences:
///   δε_e = log(U_{shifted(e)} · U_e⁻¹)
pub fn su2_translation_generator(
    complex: &SimplicialComplex,
    lengths: &[f64],
    elements: &[Su2],
    axis: usize,
    n: usize,
) -> crate::symmetry::Generator {
    use crate::symmetry::{vertex_coords_4d, vertex_index_4d};

    let mut vertex_map = vec![0usize; complex.n_vertices];
    for v in 0..complex.n_vertices {
        let mut coords = vertex_coords_4d(v, n);
        coords[axis] = (coords[axis] + 1) % n;
        vertex_map[v] = vertex_index_4d(&coords, n);
    }

    let n_edges = complex.n_edges();
    let mut delta_lengths = vec![0.0; n_edges];
    let mut delta_phases = vec![0.0; 3 * n_edges];

    for (ei, edge) in complex.edges.iter().enumerate() {
        let mut shifted_edge = [vertex_map[edge[0]], vertex_map[edge[1]]];
        shifted_edge.sort_unstable();
        if let Some(&shifted_ei) = complex.edge_index.get(&shifted_edge) {
            delta_lengths[ei] = lengths[shifted_ei] - lengths[ei];
            // Left-invariant difference: log(U_shifted · U_orig⁻¹)
            let diff = elements[shifted_ei].mul(&elements[ei].inv()).log();
            for a in 0..3 {
                delta_phases[3 * ei + a] = diff[a];
            }
        }
    }

    let axis_names = ["t", "x", "y", "z"];
    crate::symmetry::Generator {
        delta_lengths,
        delta_phases,
        name: format!("su2_translation_axis{}", axis_names[axis]),
    }
}

/// Rotation generator for SU(2) Einstein-Yang-Mills on a periodic lattice.
///
/// 90° lattice rotation in the (axis1, axis2) plane with SU(2) field transport.
pub fn su2_rotation_generator(
    complex: &SimplicialComplex,
    lengths: &[f64],
    elements: &[Su2],
    axis1: usize,
    axis2: usize,
    n: usize,
) -> crate::symmetry::Generator {
    use crate::symmetry::{vertex_coords_4d, vertex_index_4d};

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
    let mut delta_phases = vec![0.0; 3 * n_edges];

    for (ei, edge) in complex.edges.iter().enumerate() {
        let mut rotated_edge = [vertex_map[edge[0]], vertex_map[edge[1]]];
        rotated_edge.sort_unstable();
        if let Some(&rotated_ei) = complex.edge_index.get(&rotated_edge) {
            delta_lengths[ei] = lengths[rotated_ei] - lengths[ei];
            let diff = elements[rotated_ei].mul(&elements[ei].inv()).log();
            for a in 0..3 {
                delta_phases[3 * ei + a] = diff[a];
            }
        }
    }

    let axis_names = ["t", "x", "y", "z"];
    crate::symmetry::Generator {
        delta_lengths,
        delta_phases,
        name: format!("su2_rotation_{}{}", axis_names[axis1], axis_names[axis2]),
    }
}

/// Conformal generator for SU(2) Einstein-Yang-Mills (pure length scaling).
pub fn su2_conformal_generator(
    complex: &SimplicialComplex,
    lengths: &[f64],
) -> crate::symmetry::Generator {
    crate::symmetry::Generator {
        delta_lengths: lengths.to_vec(),
        delta_phases: vec![0.0; 3 * complex.n_edges()],
        name: "su2_conformal".to_string(),
    }
}

/// Combined Einstein-Yang-Mills action: S = S_R + α · S_YM.
pub fn einstein_yang_mills_action(
    complex: &SimplicialComplex,
    lengths: &[f64],
    elements: &[Su2],
    alpha: f64,
) -> f64 {
    let s_r = crate::regge::regge_action(complex, lengths);
    let s_ym = wilson_action(complex, lengths, elements);
    s_r + alpha * s_ym
}

/// Gradient of Einstein-Yang-Mills action w.r.t. all DOF.
///
/// Returns packed vector: [∂S/∂l₁, ..., ∂S/∂l_n, ∂S/∂θ¹₁, ..., ∂S/∂θ³_n].
/// Total length = n_edges + 3 * n_edges = 4 * n_edges.
pub fn einstein_yang_mills_grad(
    complex: &SimplicialComplex,
    lengths: &[f64],
    elements: &[Su2],
    alpha: f64,
) -> Vec<f64> {
    let n_edges = complex.n_edges();

    let grad_r = crate::regge::regge_action_grad(complex, lengths);
    let grad_ym_lengths = wilson_action_grad_lengths(complex, lengths, elements);
    let grad_ym_field = wilson_action_grad_field(complex, lengths, elements);

    let mut grad = Vec::with_capacity(n_edges + 3 * n_edges);

    for i in 0..n_edges {
        grad.push(grad_r[i] + alpha * grad_ym_lengths[i]);
    }

    for i in 0..(3 * n_edges) {
        grad.push(alpha * grad_ym_field[i]);
    }

    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh;
    use crate::su2::Su2;

    #[test]
    fn test_zero_field_zero_action() {
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);
        let elements = vec![Su2::identity(); complex.n_edges()];
        let action = wilson_action(&complex, &lengths, &elements);
        assert!(action.abs() < 1e-15, "Wilson action = {action}");
    }

    #[test]
    fn test_su2_gauge_invariance() {
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);

        // Random SU(2) field.
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        let elements: Vec<Su2> = (0..complex.n_edges())
            .map(|_| {
                let theta = [
                    rng.r#gen::<f64>() * 0.2 - 0.1,
                    rng.r#gen::<f64>() * 0.2 - 0.1,
                    rng.r#gen::<f64>() * 0.2 - 0.1,
                ];
                Su2::exp(&theta)
            })
            .collect();

        let s_original = wilson_action(&complex, &lengths, &elements);

        // Gauge transform: U_e → G_i · U_e · G_j^{-1} for edge (i, j).
        let gauge: Vec<Su2> = (0..complex.n_vertices)
            .map(|_| {
                let theta = [
                    rng.r#gen::<f64>() * 2.0 - 1.0,
                    rng.r#gen::<f64>() * 2.0 - 1.0,
                    rng.r#gen::<f64>() * 2.0 - 1.0,
                ];
                Su2::exp(&theta)
            })
            .collect();

        let mut transformed = elements.clone();
        for (ei, edge) in complex.edges.iter().enumerate() {
            // U'_{ij} = G_i · U_{ij} · G_j^{-1}
            transformed[ei] = gauge[edge[0]].mul(&elements[ei]).mul(&gauge[edge[1]].inv());
        }

        let s_transformed = wilson_action(&complex, &lengths, &transformed);
        assert!(
            (s_original - s_transformed).abs() < 1e-10,
            "Wilson action not gauge-invariant: {s_original} vs {s_transformed}"
        );
    }

    #[test]
    fn test_field_grad_vs_fd() {
        // Use left-invariant variations: U_e → exp(ε·e_a) · U_e.
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(123);
        let elements: Vec<Su2> = (0..complex.n_edges())
            .map(|_| {
                Su2::exp(&[
                    rng.r#gen::<f64>() * 0.1,
                    rng.r#gen::<f64>() * 0.1,
                    rng.r#gen::<f64>() * 0.1,
                ])
            })
            .collect();

        let grad = wilson_action_grad_field(&complex, &lengths, &elements);

        let eps = 1e-6;
        for ei in 0..complex.n_edges().min(10) {
            for a in 0..3 {
                let mut plus = elements.clone();
                let mut minus = elements.clone();

                // Left-invariant variation: U → exp(±ε·e_a) · U
                let mut delta = [0.0; 3];
                delta[a] = eps;
                let exp_plus = Su2::exp(&delta);
                delta[a] = -eps;
                let exp_minus = Su2::exp(&delta);

                plus[ei] = exp_plus.mul(&elements[ei]);
                minus[ei] = exp_minus.mul(&elements[ei]);

                let sp = wilson_action(&complex, &lengths, &plus);
                let sm = wilson_action(&complex, &lengths, &minus);
                let fd = (sp - sm) / (2.0 * eps);

                let abs_err = (grad[3 * ei + a] - fd).abs();
                let scale = fd.abs().max(1e-10);
                assert!(
                    abs_err / scale < 0.05 || abs_err < 1e-6,
                    "edge {ei} dir {a}: analytical={:.6e}, fd={:.6e}",
                    grad[3 * ei + a],
                    fd
                );
            }
        }
    }

    #[test]
    fn test_abelian_embedding() {
        // U(1) ⊂ SU(2): embedding θ → exp(θ·σ3) should give the same action
        // as U(1) with the same θ.
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(456);
        let phases: Vec<f64> = (0..complex.n_edges())
            .map(|_| rng.r#gen::<f64>() * 0.1)
            .collect();

        // U(1) action.
        let s_u1 = crate::gauge::maxwell_action(&complex, &lengths, &phases);

        // SU(2) with abelian embedding along σ3.
        let elements: Vec<Su2> = phases.iter().map(|&p| Su2::exp(&[0.0, 0.0, p])).collect();
        let s_su2 = wilson_action(&complex, &lengths, &elements);

        // Normalization: Wilson uses (1 - cos(F/2)), Maxwell uses F²/2.
        // For small F: 1 - cos(F/2) ≈ F²/8, so ratio = 1/4.
        let ratio = if s_u1.abs() > 1e-15 {
            s_su2 / s_u1
        } else {
            0.25
        };
        assert!(
            (ratio - 0.25).abs() < 0.05,
            "abelian embedding: SU(2)={s_su2:.6e}, U(1)={s_u1:.6e}, ratio={ratio:.4} (expected ~0.25)"
        );
    }

    #[test]
    fn test_trait_compat() {
        // U1Field through GaugeField trait should match direct calls.
        use crate::gauge::U1Field;
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(789);
        let phases: Vec<f64> = (0..complex.n_edges())
            .map(|_| rng.r#gen::<f64>() * 0.1)
            .collect();

        let s_direct = crate::gauge::maxwell_action(&complex, &lengths, &phases);
        let field = U1Field(phases.clone());
        let s_trait = U1Field::action(&complex, &lengths, &field);
        assert!((s_direct - s_trait).abs() < 1e-15);

        let g_direct = crate::gauge::maxwell_action_grad_phases(&complex, &lengths, &phases);
        let g_trait = U1Field::grad_field(&complex, &lengths, &field);
        for (a, b) in g_direct.iter().zip(g_trait.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }
}
