//! U(1) gauge field on a simplicial complex.
//!
//! The gauge field assigns a phase θ_e ∈ [0, 2π) to each edge.
//! The field strength on each triangle is the oriented holonomy:
//!
//!   F_t = θ_{01} + θ_{12} + θ_{20}   (mod 2π, centered at [-π, π))
//!
//! where the edges are oriented around the triangle boundary.
//!
//! The discrete Maxwell action on a curved background is:
//!
//!   S_M = (1/2) Σ_triangles  F_t² · W_t
//!
//! where W_t = V*_t / A_t is the ratio of the dual volume to the area,
//! encoding the metric coupling. In the simplicial setting, V*_t is the
//! circumcentric dual area (the "diamond" volume in the dual lattice).
//!
//! For a 4D complex, the dual 2-cell of a triangle is constructed from
//! circumcenters of the 4-simplices, tetrahedra, and the triangle itself.
//! We approximate V*_t using the sum of dual contributions from each
//! 4-simplex sharing the triangle.

use crate::complex::SimplicialComplex;
use crate::geometry::{pent_volume, pent_volume_grad, triangle_area, triangle_area_grad_lsq};
use std::f64::consts::PI;

// === GaugeField trait ===

/// Trait for gauge field types (U(1), SU(2), etc.) on a simplicial complex.
pub trait GaugeField: Clone {
    /// Number of DOF per edge (1 for U(1), 3 for SU(2)).
    const DOF_PER_EDGE: usize;

    /// Flat connection (identity/zero).
    fn flat(n_edges: usize) -> Self;

    /// Gauge/Yang-Mills action on curved background.
    fn action(complex: &SimplicialComplex, lengths: &[f64], field: &Self) -> f64;

    /// Gradient w.r.t. gauge DOF (packed flat: DOF_PER_EDGE * n_edges).
    fn grad_field(complex: &SimplicialComplex, lengths: &[f64], field: &Self) -> Vec<f64>;

    /// Gradient w.r.t. edge lengths.
    fn grad_lengths(complex: &SimplicialComplex, lengths: &[f64], field: &Self) -> Vec<f64>;

    /// Pack into flat vector.
    fn pack(&self) -> Vec<f64>;

    /// Unpack from flat vector.
    fn unpack(flat: &[f64], n_edges: usize) -> Self;

    /// Number of gauge generators per vertex.
    fn n_gauge_generators_per_vertex() -> usize;

    /// Gauge generator at vertex v, direction a (0..n_gauge_generators).
    /// Returns a flat vector of length DOF_PER_EDGE * n_edges.
    fn gauge_generator(complex: &SimplicialComplex, vertex: usize, direction: usize) -> Vec<f64>;
}

// === U(1) implementation ===

/// U(1) gauge field: one phase per edge.
#[derive(Debug, Clone)]
pub struct U1Field(pub Vec<f64>);

impl GaugeField for U1Field {
    const DOF_PER_EDGE: usize = 1;

    fn flat(n_edges: usize) -> Self {
        U1Field(vec![0.0; n_edges])
    }

    fn action(complex: &SimplicialComplex, lengths: &[f64], field: &Self) -> f64 {
        maxwell_action(complex, lengths, &field.0)
    }

    fn grad_field(complex: &SimplicialComplex, lengths: &[f64], field: &Self) -> Vec<f64> {
        maxwell_action_grad_phases(complex, lengths, &field.0)
    }

    fn grad_lengths(complex: &SimplicialComplex, lengths: &[f64], field: &Self) -> Vec<f64> {
        maxwell_action_grad_lengths(complex, lengths, &field.0)
    }

    fn pack(&self) -> Vec<f64> {
        self.0.clone()
    }

    fn unpack(flat: &[f64], _n_edges: usize) -> Self {
        U1Field(flat.to_vec())
    }

    fn n_gauge_generators_per_vertex() -> usize {
        1
    }

    fn gauge_generator(complex: &SimplicialComplex, vertex: usize, _direction: usize) -> Vec<f64> {
        let n = complex.n_edges();
        let mut g = vec![0.0; n];
        for (ei, edge) in complex.edges.iter().enumerate() {
            if edge[0] == vertex {
                g[ei] = -1.0;
            } else if edge[1] == vertex {
                g[ei] = 1.0;
            }
        }
        g
    }
}

/// Compute field strengths F_t on all triangles.
///
/// F_t = oriented sum of edge phases around triangle boundary,
/// wrapped to [-π, π).
///
/// We use a convention where edges are oriented from lower to higher
/// vertex index. The triangle [v0, v1, v2] with v0 < v1 < v2 has
/// boundary: (v0→v1) + (v1→v2) - (v0→v2), i.e., edges [v0,v1], [v1,v2]
/// with positive orientation and [v0,v2] with negative orientation.
pub fn field_strengths(complex: &SimplicialComplex, phases: &[f64]) -> Vec<f64> {
    let n_tri = complex.n_triangles();
    let mut f = vec![0.0; n_tri];

    for (ti, tri) in complex.triangles.iter().enumerate() {
        // Oriented sum: θ_{01} + θ_{12} - θ_{02}
        // (This is the discrete exterior derivative dA evaluated on the triangle.)
        let e01 = complex.edge_index[&[tri[0], tri[1]]];
        let e02 = complex.edge_index[&[tri[0], tri[2]]];
        let e12 = complex.edge_index[&[tri[1], tri[2]]];

        let raw = phases[e01] + phases[e12] - phases[e02];

        // Wrap to [-π, π) for compact U(1).
        f[ti] = wrap_angle(raw);
    }

    f
}

/// Wrap angle to [-π, π).
fn wrap_angle(x: f64) -> f64 {
    let mut a = x % (2.0 * PI);
    if a >= PI {
        a -= 2.0 * PI;
    } else if a < -PI {
        a += 2.0 * PI;
    }
    a
}

/// Compute the metric weight W_t for each triangle.
///
/// Public for use by other gauge field implementations (e.g., SU(2)).
///
/// In the continuum, the Maxwell action density is (1/2) F_μν F^μν √g.
/// In the discrete setting, this becomes:
///
///   S_M = (1/2) Σ_t F_t² · W_t
///
/// where W_t encodes the "volume per area" from the metric.
///
/// We use the approximation:
///   W_t = Σ_{σ∋t} V_σ / (n_triangles_in_σ)  /  A_t
///
/// where V_σ is the 4-volume of the pentachoron σ and A_t is the
/// triangle area. This distributes each 4-simplex's volume equally
/// among its 10 triangles, then divides by the area to get the correct
/// dimensions.
///
/// A more accurate formula uses circumcentric dual volumes, but this
/// barycentric approximation is sufficient for the symmetry search
/// and is exact for regular triangulations.
pub fn metric_weights(complex: &SimplicialComplex, lengths: &[f64]) -> Vec<f64> {
    let n_tri = complex.n_triangles();
    let mut weights = vec![0.0; n_tri];

    // Compute 4-volumes.
    let mut pent_volumes = Vec::with_capacity(complex.n_pents());
    for pi in 0..complex.n_pents() {
        let pl = complex.pent_edge_lengths(pi, lengths);
        pent_volumes.push(pent_volume(&pl));
    }

    // Distribute volume to triangles.
    // Each 4-simplex has 10 triangles; distribute V/10 to each.
    for (ti, pent_list) in complex.tri_to_pents.iter().enumerate() {
        let mut vol_share = 0.0;
        for &pi in pent_list {
            vol_share += pent_volumes[pi] / 10.0;
        }

        let [a, b, c] = complex.tri_edge_lengths(ti, lengths);
        let area = triangle_area(a, b, c);
        if area > 1e-30 {
            weights[ti] = vol_share / area;
        }
    }

    weights
}

/// Discrete Maxwell action on the curved simplicial background.
///
///   S_M = (1/2) Σ_t F_t² · W_t
pub fn maxwell_action(complex: &SimplicialComplex, lengths: &[f64], phases: &[f64]) -> f64 {
    let f = field_strengths(complex, phases);
    let w = metric_weights(complex, lengths);

    let mut action = 0.0;
    for ti in 0..complex.n_triangles() {
        action += 0.5 * f[ti] * f[ti] * w[ti];
    }
    action
}

/// Gradient of Maxwell action w.r.t. edge phases: ∂S_M/∂θ_e.
///
/// ∂S_M/∂θ_e = Σ_{t∋e} F_t · W_t · (∂F_t/∂θ_e)
///
/// where ∂F_t/∂θ_e = ±1 depending on orientation.
pub fn maxwell_action_grad_phases(
    complex: &SimplicialComplex,
    lengths: &[f64],
    phases: &[f64],
) -> Vec<f64> {
    let f = field_strengths(complex, phases);
    let w = metric_weights(complex, lengths);
    let n_edges = complex.n_edges();
    let mut grad = vec![0.0; n_edges];

    for (ti, tri) in complex.triangles.iter().enumerate() {
        let fw = f[ti] * w[ti];

        let e01 = complex.edge_index[&[tri[0], tri[1]]];
        let e02 = complex.edge_index[&[tri[0], tri[2]]];
        let e12 = complex.edge_index[&[tri[1], tri[2]]];

        // F_t = θ_{01} + θ_{12} - θ_{02}
        grad[e01] += fw; // ∂F/∂θ_{01} = +1
        grad[e12] += fw; // ∂F/∂θ_{12} = +1
        grad[e02] -= fw; // ∂F/∂θ_{02} = -1
    }

    grad
}

/// Gradient of Maxwell action w.r.t. edge lengths: ∂S_M/∂l_e.
///
/// This is the EM stress-energy contribution to the Regge equations.
/// Uses analytical derivatives via Cayley-Menger cofactors.
///
/// ∂S_M/∂l_e = (1/2) Σ_t F_t² · ∂W_t/∂l_e
///
/// where W_t = (Σ_{σ∋t} V_σ/10) / A_t, so:
///
/// ∂W_t/∂l_e = (1/A_t) Σ_{σ∋t} (1/10) ∂V_σ/∂l_e − (W_t/A_t) ∂A_t/∂l_e
pub fn maxwell_action_grad_lengths(
    complex: &SimplicialComplex,
    lengths: &[f64],
    phases: &[f64],
) -> Vec<f64> {
    let n_edges = complex.n_edges();
    let n_tris = complex.n_triangles();
    let n_pents = complex.n_pents();
    let mut grad = vec![0.0; n_edges];

    // Precompute field strengths and metric weights.
    let f = field_strengths(complex, phases);
    let w = metric_weights(complex, lengths);

    // Precompute triangle areas and F_t² for each triangle.
    let mut areas = vec![0.0; n_tris];
    for ti in 0..n_tris {
        let [a, b, c] = complex.tri_edge_lengths(ti, lengths);
        areas[ti] = triangle_area(a, b, c);
    }

    // Precompute volume gradients and global edge indices for each pentachoron.
    let mut pent_vol_grads: Vec<[f64; 10]> = Vec::with_capacity(n_pents);
    let mut pent_edge_idx: Vec<[usize; 10]> = Vec::with_capacity(n_pents);
    for pi in 0..n_pents {
        let pl = complex.pent_edge_lengths(pi, lengths);
        pent_vol_grads.push(pent_volume_grad(&pl));
        pent_edge_idx.push(complex.pent_edge_indices(pi));
    }

    // Term 1 (volume): For each triangle, for each pentachoron containing it,
    // accumulate (1/2) · (F_t²/A_t) · (1/10) · ∂V_σ/∂l_e
    for ti in 0..n_tris {
        let area = areas[ti];
        if area < 1e-30 {
            continue;
        }
        let coeff = 0.5 * f[ti] * f[ti] / area;

        for &pi in &complex.tri_to_pents[ti] {
            let vol_grad = &pent_vol_grads[pi];
            let edge_idx = &pent_edge_idx[pi];
            for k in 0..10 {
                grad[edge_idx[k]] += coeff * (1.0 / 10.0) * vol_grad[k];
            }
        }
    }

    // Term 2 (area): For each triangle, accumulate
    // -(1/2) · F_t² · W_t / A_t · ∂A_t/∂l_e
    // where ∂A_t/∂l_e = ∂A/∂(l²) · 2l
    for ti in 0..n_tris {
        let area = areas[ti];
        if area < 1e-30 {
            continue;
        }
        let coeff = -0.5 * f[ti] * f[ti] * w[ti] / area;

        let [a, b, c] = complex.tri_edge_lengths(ti, lengths);
        let da_dlsq = triangle_area_grad_lsq(a, b, c);
        let edge_indices = complex.tri_edge_indices(ti);
        let tri_lengths = [a, b, c];

        for k in 0..3 {
            // ∂A/∂l_e = ∂A/∂(l²) · 2l
            let da_dl = da_dlsq[k] * 2.0 * tri_lengths[k];
            grad[edge_indices[k]] += coeff * da_dl;
        }
    }

    grad
}

/// Gradient of Maxwell action w.r.t. edge lengths via finite differences.
///
/// Kept for testing against the analytical version.
#[cfg(test)]
fn maxwell_action_grad_lengths_fd(
    complex: &SimplicialComplex,
    lengths: &[f64],
    phases: &[f64],
    eps: f64,
) -> Vec<f64> {
    let n_edges = complex.n_edges();
    let mut grad = vec![0.0; n_edges];
    let mut l_work = lengths.to_vec();

    for i in 0..n_edges {
        l_work[i] = lengths[i] + eps;
        let s_plus = maxwell_action(complex, &l_work, phases);

        l_work[i] = lengths[i] - eps;
        let s_minus = maxwell_action(complex, &l_work, phases);

        grad[i] = (s_plus - s_minus) / (2.0 * eps);
        l_work[i] = lengths[i];
    }

    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh;

    #[test]
    fn test_zero_field_zero_action() {
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);
        let phases = vec![0.0; complex.n_edges()];
        let action = maxwell_action(&complex, &lengths, &phases);
        assert!(action.abs() < 1e-15, "action = {action}");
    }

    #[test]
    fn test_field_strength_wrap() {
        assert!((wrap_angle(0.0)).abs() < 1e-15);
        assert!((wrap_angle(PI + 0.1) - (-PI + 0.1)).abs() < 1e-12);
        assert!((wrap_angle(-PI - 0.1) - (PI - 0.1)).abs() < 1e-12);
    }

    #[test]
    fn test_nonzero_field_positive_action() {
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);

        // Random phases → non-zero field strengths → positive action.
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let phases: Vec<f64> = (0..complex.n_edges())
            .map(|_| rng.r#gen::<f64>() * 0.1)
            .collect();

        let action = maxwell_action(&complex, &lengths, &phases);
        assert!(action > 0.0, "action should be positive for non-zero field");
    }

    #[test]
    fn test_gauge_invariance() {
        // Under a gauge transformation θ_e → θ_e + φ_j - φ_i for edge (i,j),
        // the field strength F_t should be invariant.
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let phases: Vec<f64> = (0..complex.n_edges())
            .map(|_| rng.r#gen::<f64>() * 0.5)
            .collect();

        // Random gauge transformation: one scalar per vertex.
        let gauge: Vec<f64> = (0..complex.n_vertices)
            .map(|_| rng.r#gen::<f64>() * 2.0 * PI)
            .collect();

        // Apply gauge transform.
        let mut phases_transformed = phases.clone();
        for (ei, edge) in complex.edges.iter().enumerate() {
            // θ_{ij} → θ_{ij} + φ_j - φ_i (edge oriented i→j with i<j)
            phases_transformed[ei] += gauge[edge[1]] - gauge[edge[0]];
        }

        let f_original = field_strengths(&complex, &phases);
        let f_transformed = field_strengths(&complex, &phases_transformed);

        let s_original = maxwell_action(&complex, &lengths, &phases);
        let s_transformed = maxwell_action(&complex, &lengths, &phases_transformed);

        // Field strengths should match (mod wrapping).
        for ti in 0..complex.n_triangles() {
            let diff = wrap_angle(f_original[ti] - f_transformed[ti]);
            assert!(
                diff.abs() < 1e-10,
                "field strength mismatch at tri {ti}: {} vs {}",
                f_original[ti],
                f_transformed[ti]
            );
        }

        assert!(
            (s_original - s_transformed).abs() < 1e-10,
            "Maxwell action not gauge-invariant: {s_original} vs {s_transformed}"
        );
    }

    #[test]
    fn test_phase_gradient_vs_fd() {
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let phases: Vec<f64> = (0..complex.n_edges())
            .map(|_| rng.r#gen::<f64>() * 0.1)
            .collect();

        let grad = maxwell_action_grad_phases(&complex, &lengths, &phases);

        // Finite-difference check.
        let eps = 1e-7;
        let mut phases_work = phases.clone();
        for i in 0..complex.n_edges().min(20) {
            phases_work[i] = phases[i] + eps;
            let sp = maxwell_action(&complex, &lengths, &phases_work);
            phases_work[i] = phases[i] - eps;
            let sm = maxwell_action(&complex, &lengths, &phases_work);
            phases_work[i] = phases[i];

            let fd = (sp - sm) / (2.0 * eps);
            assert!(
                (grad[i] - fd).abs() < 1e-5,
                "edge {i}: analytical={}, fd={}",
                grad[i],
                fd
            );
        }
    }

    #[test]
    fn test_analytical_length_grad_vs_fd() {
        // Perturbed mesh with random phases — analytical should match FD.
        let (complex, mut lengths) = mesh::flat_hypercubic(2, 1.0);

        // Perturb lengths slightly to break symmetry.
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for l in lengths.iter_mut() {
            *l *= 1.0 + 0.05 * (rng.r#gen::<f64>() - 0.5);
        }

        let phases: Vec<f64> = (0..complex.n_edges())
            .map(|_| rng.r#gen::<f64>() * 0.3)
            .collect();

        let grad_analytical = maxwell_action_grad_lengths(&complex, &lengths, &phases);
        let grad_fd = maxwell_action_grad_lengths_fd(&complex, &lengths, &phases, 1e-7);

        for i in 0..complex.n_edges() {
            let abs_err = (grad_analytical[i] - grad_fd[i]).abs();
            let scale = grad_fd[i].abs().max(1e-12);
            let rel_err = abs_err / scale;
            assert!(
                rel_err < 1e-4 || abs_err < 1e-10,
                "edge {i}: analytical={}, fd={}, rel_err={}",
                grad_analytical[i],
                grad_fd[i],
                rel_err
            );
        }
    }
}
