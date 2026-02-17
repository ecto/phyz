//! Regge action and its gradient.
//!
//! The Regge action is the discrete Einstein-Hilbert action:
//!
//!   S_R = Σ_triangles  A_t · δ_t
//!
//! where A_t is the area of triangle t and δ_t is the deficit angle
//! (2π minus the sum of dihedral angles from all 4-simplices meeting at t).
//!
//! For interior triangles in a closed manifold, δ_t = 2π - Σ_σ∋t θ_t(σ).
//! For boundary triangles, the formula differs (we handle closed manifolds first).
//!
//! The Regge equations (discrete Einstein equations) are:
//!
//!   ∂S_R/∂l_e = Σ_t∋e  (∂A_t/∂l_e) · δ_t
//!
//! The Schläfli identity guarantees that Σ A_t ∂δ_t/∂l_e = 0 for each edge,
//! so only the area gradient contributes.

use crate::complex::SimplicialComplex;
use crate::geometry::{all_dihedral_angles, triangle_area, triangle_area_grad_lsq};
use std::f64::consts::PI;

/// Compute deficit angles at all triangles.
///
/// δ_t = 2π - Σ_{σ∋t} θ_t(σ)
///
/// Returns a vector of deficit angles indexed by triangle index.
pub fn deficit_angles(complex: &SimplicialComplex, lengths: &[f64]) -> Vec<f64> {
    let n_tri = complex.n_triangles();
    let mut deficit = vec![2.0 * PI; n_tri];

    // For each 4-simplex, compute all 10 dihedral angles and subtract
    // from the deficit of the corresponding triangles.
    for (pi, pent) in complex.pents.iter().enumerate() {
        let pent_lengths = complex.pent_edge_lengths(pi, lengths);
        let angles = all_dihedral_angles(&pent_lengths);

        // Map each of the 10 dihedral angles to the corresponding triangle.
        // angles[idx] corresponds to the triangle opposite to edge (l,m) in local ordering.
        let mut idx = 0;
        for l in 0..5usize {
            for m in (l + 1)..5 {
                // The triangle is the one formed by the 3 vertices NOT equal to l or m.
                let mut tri_verts = [0usize; 3];
                let mut ti_local = 0;
                for v in 0..5 {
                    if v != l && v != m {
                        tri_verts[ti_local] = pent[v];
                        ti_local += 1;
                    }
                }
                tri_verts.sort_unstable();
                let tri_idx = complex.tri_index[&tri_verts];
                deficit[tri_idx] -= angles[idx];
                idx += 1;
            }
        }
    }

    deficit
}

/// Compute the Regge action: S_R = Σ_t A_t · δ_t.
pub fn regge_action(complex: &SimplicialComplex, lengths: &[f64]) -> f64 {
    let deficits = deficit_angles(complex, lengths);
    let mut action = 0.0;

    for (ti, &delta) in deficits.iter().enumerate() {
        let [a, b, c] = complex.tri_edge_lengths(ti, lengths);
        let area = triangle_area(a, b, c);
        action += area * delta;
    }

    action
}

/// Gradient of the Regge action w.r.t. edge lengths: ∂S_R/∂l_e.
///
/// By the Schläfli identity, only the area gradient contributes:
///
///   ∂S_R/∂l_e = Σ_{t∋e} (∂A_t/∂l_e) · δ_t
///
/// This is exact (not an approximation) and is one of the beautiful
/// properties of Regge calculus.
pub fn regge_action_grad(complex: &SimplicialComplex, lengths: &[f64]) -> Vec<f64> {
    let deficits = deficit_angles(complex, lengths);
    let n_edges = complex.n_edges();
    let mut grad = vec![0.0; n_edges];

    for (ti, &delta) in deficits.iter().enumerate() {
        let edge_indices = complex.tri_edge_indices(ti);
        let [a, b, c] = complex.tri_edge_lengths(ti, lengths);

        // ∂A/∂(l²) for each of the 3 edges.
        let da_dlsq = triangle_area_grad_lsq(a, b, c);

        // ∂S/∂l_e = ∂A/∂(l²) · 2l · δ_t
        for (k, &ei) in edge_indices.iter().enumerate() {
            let l_e = lengths[ei];
            grad[ei] += da_dlsq[k] * 2.0 * l_e * delta;
        }
    }

    grad
}

/// Compute Regge action via finite differences for gradient verification.
pub fn regge_action_grad_fd(
    complex: &SimplicialComplex,
    lengths: &[f64],
    eps: f64,
) -> Vec<f64> {
    let n_edges = complex.n_edges();
    let mut grad = vec![0.0; n_edges];
    let mut l_work = lengths.to_vec();

    for i in 0..n_edges {
        l_work[i] = lengths[i] + eps;
        let s_plus = regge_action(complex, &l_work);

        l_work[i] = lengths[i] - eps;
        let s_minus = regge_action(complex, &l_work);

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
    fn test_flat_space_zero_action() {
        // A flat triangulation should have zero deficit angles everywhere
        // (interior triangles), hence zero Regge action.
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);

        let deficits = deficit_angles(&complex, &lengths);
        let action = regge_action(&complex, &lengths);

        // Interior triangles should have deficit ≈ 0.
        // Boundary triangles will have non-zero deficit, but in a periodic
        // (toroidal) mesh all triangles are interior.
        let max_deficit = deficits.iter().map(|d| d.abs()).fold(0.0, f64::max);
        assert!(
            max_deficit < 1e-8,
            "max deficit angle = {max_deficit} (expected ~0 for flat space)"
        );
        assert!(
            action.abs() < 1e-8,
            "Regge action = {action} (expected 0 for flat space)"
        );
    }

    #[test]
    fn test_gradient_vs_finite_diff() {
        // Use a small mesh with slightly perturbed edge lengths.
        let (complex, mut lengths) = mesh::flat_hypercubic(2, 1.0);

        // Perturb lengths slightly to get non-zero curvature.
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for l in lengths.iter_mut() {
            *l *= 1.0 + 0.001 * (rng.r#gen::<f64>() - 0.5);
        }

        let grad_analytical = regge_action_grad(&complex, &lengths);
        let grad_fd = regge_action_grad_fd(&complex, &lengths, 1e-7);

        // Compare relative error where gradients are non-negligible.
        let max_grad = grad_fd
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-10);

        let max_rel_err = grad_analytical
            .iter()
            .zip(grad_fd.iter())
            .map(|(a, b)| (a - b).abs() / max_grad)
            .fold(0.0, f64::max);

        assert!(
            max_rel_err < 0.05,
            "max relative gradient error = {max_rel_err} (analytical vs finite diff)"
        );
    }
}
