//! Lorentzian Regge action and its gradient.
//!
//! The Lorentzian Regge action:
//!
//!   S_R = Σ_spacelike_hinges  A_t · (2π − Σ_σ θ_t(σ))
//!       + Σ_timelike_hinges   A_t · (−Σ_σ η_t(σ))
//!
//! Hinge type is determined from the CM cofactor structure per-pentachoron:
//! - Both cofactors same sign, |ratio| ≤ 1: real angle (spacelike hinge)
//! - Both cofactors same sign, |ratio| > 1: boost (timelike hinge)
//! - Mixed-sign cofactors: boost (timelike hinge)
//!
//! When both cofactors are negative, a sign correction is applied:
//! sqrt(C_ll) * sqrt(C_mm) = i√|C_ll| · i√|C_mm| = −√(C_ll · C_mm).
//!
//! The gradient uses the Schläfli identity (holds in Lorentzian signature):
//!   ∂S_R/∂s_e = Σ_{t∋e} (∂A_t/∂s_e) · δ_t

use crate::complex::SimplicialComplex;
use crate::geometry::cofactor_6x6;
use crate::lorentzian::{
    cm_matrix_signed, triangle_area_sq_grad, triangle_area_sq_lorentzian, HingeType,
};
use std::f64::consts::PI;

/// Compute dihedral value and hinge type from CM cofactors.
///
/// The hinge type is determined by the cofactor structure:
/// - product > 0 (both same sign), |ratio| ≤ 1 → real angle
/// - product > 0 (both same sign), |ratio| > 1 → boost
/// - product < 0 (mixed signs) → boost
///
/// When both cofactors are negative, the sign of the ratio is flipped
/// to account for the analytic continuation: i² = −1.
fn compute_dihedral(c_lm: f64, c_ll: f64, c_mm: f64) -> (f64, HingeType) {
    let product = c_ll * c_mm;

    if product.abs() < 1e-60 {
        return (0.0, HingeType::Spacelike);
    }

    if product > 0.0 {
        // Both cofactors have the same sign.
        // When both are negative, the denominator picks up a sign flip:
        //   sqrt(C_ll) * sqrt(C_mm) = i*sqrt(|C_ll|) * i*sqrt(|C_mm|) = -sqrt(product)
        let sign_corr = if c_ll < 0.0 { -1.0 } else { 1.0 };
        let denom = product.sqrt();
        let ratio = sign_corr * (-c_lm) / denom;

        if ratio.abs() <= 1.0 + 1e-12 {
            // Real rotation angle
            let cos_theta = ratio.clamp(-1.0, 1.0);
            (cos_theta.acos(), HingeType::Spacelike)
        } else {
            // Boost parameter
            let boost = ratio.abs().acosh();
            (ratio.signum() * boost, HingeType::Timelike)
        }
    } else {
        // Mixed-sign cofactors → always a boost.
        // Denominator is i * sqrt(|product|), giving η = arcsinh(-C_lm / sqrt(|product|)).
        let abs_denom = (-product).sqrt();
        let ratio = -c_lm / abs_denom;
        (ratio.asinh(), HingeType::Timelike)
    }
}

/// Compute Lorentzian deficit angles at all triangles.
///
/// Returns (deficit, hinge_type) for each triangle.
pub fn lorentzian_deficit_angles(
    complex: &SimplicialComplex,
    sq_lengths: &[f64],
) -> Vec<(f64, HingeType)> {
    let n_tri = complex.n_triangles();
    let mut hinge_types: Vec<Option<HingeType>> = vec![None; n_tri];
    let mut angle_sums = vec![0.0; n_tri];

    for (pi, pent) in complex.pents.iter().enumerate() {
        let pent_sq = pent_sq_lengths(complex, pi, sq_lengths);
        let cm = cm_matrix_signed(&pent_sq);

        // Precompute cofactors (symmetric matrix, indices 1..5)
        #[allow(clippy::needless_range_loop)]
        let cofactors = {
            let mut cof = [[0.0; 6]; 6];
            for i in 1..6 {
                for j in i..6 {
                    let c = cofactor_6x6(&cm, i, j);
                    cof[i][j] = c;
                    cof[j][i] = c;
                }
            }
            cof
        };

        for l in 0..5usize {
            for m in (l + 1)..5 {
                let rl = l + 1;
                let rm = m + 1;

                let (val, ht) = compute_dihedral(
                    cofactors[rl][rm],
                    cofactors[rl][rl],
                    cofactors[rm][rm],
                );

                // Find the triangle index
                let mut tri_verts = [0usize; 3];
                let mut ti = 0;
                for (v, &pv) in pent.iter().enumerate() {
                    if v != l && v != m {
                        tri_verts[ti] = pv;
                        ti += 1;
                    }
                }
                tri_verts.sort_unstable();
                let tri_idx = complex.tri_index[&tri_verts];

                // Set hinge type on first encounter
                if hinge_types[tri_idx].is_none() {
                    hinge_types[tri_idx] = Some(ht);
                }

                angle_sums[tri_idx] += val;
            }
        }
    }

    (0..n_tri)
        .map(|ti| {
            let ht = hinge_types[ti].unwrap_or(HingeType::Spacelike);
            let deficit = match ht {
                HingeType::Spacelike => 2.0 * PI - angle_sums[ti],
                HingeType::Timelike => -angle_sums[ti],
            };
            (deficit, ht)
        })
        .collect()
}

/// Absolute area of a triangle: √|A²| (always non-negative).
///
/// The Lorentzian Regge action uses positive areas for both spacelike and
/// timelike triangles. This is required for the Schläfli identity to cancel
/// the dihedral angle gradients.
fn abs_area(complex: &SimplicialComplex, tri_idx: usize, sq_lengths: &[f64]) -> f64 {
    let t = &complex.triangles[tri_idx];
    let e01 = complex.edge_index[&sorted2(t[0], t[1])];
    let e02 = complex.edge_index[&sorted2(t[0], t[2])];
    let e12 = complex.edge_index[&sorted2(t[1], t[2])];

    let a_sq = triangle_area_sq_lorentzian(sq_lengths[e01], sq_lengths[e02], sq_lengths[e12]);
    a_sq.abs().sqrt()
}

/// Lorentzian Regge action: S_R = Σ_t |A_t| · δ_t.
///
/// Uses positive areas for both spacelike and timelike triangles,
/// as required by the Lorentzian Schläfli identity (Barrett 1993).
pub fn lorentzian_regge_action(complex: &SimplicialComplex, sq_lengths: &[f64]) -> f64 {
    let deficits = lorentzian_deficit_angles(complex, sq_lengths);
    let mut action = 0.0;

    for (ti, &(deficit, _htype)) in deficits.iter().enumerate() {
        let area = abs_area(complex, ti, sq_lengths);
        action += area * deficit;
    }

    action
}

/// Gradient of Lorentzian Regge action w.r.t. signed squared edge lengths.
///
/// By the Lorentzian Schläfli identity (Barrett 1993):
///   ∂S_R/∂s_e = Σ_{t∋e} (∂|A_t|/∂s_e) · δ_t
///
/// where ∂|A_t|/∂s_e = sign(A²_t) · ∂(A²_t)/∂s_e / (2|A_t|).
pub fn lorentzian_regge_action_grad(
    complex: &SimplicialComplex,
    sq_lengths: &[f64],
) -> Vec<f64> {
    let deficits = lorentzian_deficit_angles(complex, sq_lengths);
    let n_edges = complex.n_edges();
    let mut grad = vec![0.0; n_edges];

    for (ti, &(deficit, _htype)) in deficits.iter().enumerate() {
        let t = &complex.triangles[ti];
        let e_indices = complex.tri_edge_indices(ti);

        let e01 = complex.edge_index[&sorted2(t[0], t[1])];
        let e02 = complex.edge_index[&sorted2(t[0], t[2])];
        let e12 = complex.edge_index[&sorted2(t[1], t[2])];

        let s01 = sq_lengths[e01];
        let s02 = sq_lengths[e02];
        let s12 = sq_lengths[e12];

        let a_sq = triangle_area_sq_lorentzian(s01, s02, s12);
        let da_sq = triangle_area_sq_grad(s01, s02, s12);

        let abs_a = a_sq.abs().sqrt();
        if abs_a < 1e-30 {
            continue;
        }
        // ∂|A|/∂s = sign(A²) · ∂(A²)/∂s / (2|A|)
        let sign_a = if a_sq >= 0.0 { 1.0 } else { -1.0 };
        let factor = sign_a / (2.0 * abs_a);

        for (k, &ei) in e_indices.iter().enumerate() {
            grad[ei] += factor * da_sq[k] * deficit;
        }
    }

    grad
}

/// Finite-difference gradient for verification.
pub fn lorentzian_regge_action_grad_fd(
    complex: &SimplicialComplex,
    sq_lengths: &[f64],
    eps: f64,
) -> Vec<f64> {
    let n_edges = complex.n_edges();
    let mut grad = vec![0.0; n_edges];
    let mut work = sq_lengths.to_vec();

    for i in 0..n_edges {
        work[i] = sq_lengths[i] + eps;
        let s_plus = lorentzian_regge_action(complex, &work);
        work[i] = sq_lengths[i] - eps;
        let s_minus = lorentzian_regge_action(complex, &work);
        grad[i] = (s_plus - s_minus) / (2.0 * eps);
        work[i] = sq_lengths[i];
    }

    grad
}

/// Get the 10 signed squared edge lengths of a 4-simplex.
fn pent_sq_lengths(
    complex: &SimplicialComplex,
    pent_idx: usize,
    sq_lengths: &[f64],
) -> [f64; 10] {
    let p = &complex.pents[pent_idx];
    let mut result = [0.0; 10];
    let mut idx = 0;
    for i in 0..5 {
        for j in (i + 1)..5 {
            let e = sorted2(p[i], p[j]);
            result[idx] = sq_lengths[complex.edge_index[&e]];
            idx += 1;
        }
    }
    result
}

fn sorted2(a: usize, b: usize) -> [usize; 2] {
    if a < b { [a, b] } else { [b, a] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh;

    /// Flat Euclidean space with positive squared lengths should give zero action.
    #[test]
    fn test_flat_euclidean_zero_action() {
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);
        let sq_lengths: Vec<f64> = lengths.iter().map(|l| l * l).collect();

        let action = lorentzian_regge_action(&complex, &sq_lengths);
        assert!(
            action.abs() < 1e-8,
            "flat Euclidean action = {action} (expected ~0)"
        );
    }

    /// Gradient vs finite difference on perturbed Euclidean lattice.
    #[test]
    fn test_gradient_vs_fd_euclidean() {
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        let sq_lengths: Vec<f64> = lengths
            .iter()
            .map(|l| {
                let l_pert = l * (1.0 + 0.001 * (rng.r#gen::<f64>() - 0.5));
                l_pert * l_pert
            })
            .collect();

        let grad_analytical = lorentzian_regge_action_grad(&complex, &sq_lengths);
        let grad_fd = lorentzian_regge_action_grad_fd(&complex, &sq_lengths, 1e-7);

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
            "max relative gradient error = {max_rel_err}"
        );
    }

    /// Lorentzian Regge action on all-positive squared lengths should match
    /// the standard (Euclidean) Regge action.
    #[test]
    fn test_matches_euclidean_regge() {
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(99);
        let lengths_pert: Vec<f64> = lengths
            .iter()
            .map(|l| l * (1.0 + 0.002 * (rng.r#gen::<f64>() - 0.5)))
            .collect();
        let sq_lengths: Vec<f64> = lengths_pert.iter().map(|l| l * l).collect();

        let s_euclidean = crate::regge::regge_action(&complex, &lengths_pert);
        let s_lorentzian = lorentzian_regge_action(&complex, &sq_lengths);

        assert!(
            (s_euclidean - s_lorentzian).abs() / (s_euclidean.abs() + 1e-15) < 0.01,
            "Euclidean={s_euclidean}, Lorentzian={s_lorentzian}"
        );
    }

    /// Diagnose CM cofactor structure for a Lorentzian Kuhn pentachoron.
    #[test]
    fn test_lorentzian_cofactor_structure() {
        let dt = 0.3;
        let dx = 1.0;
        let sq_lor = [
            -(dt * dt),
            -(dt * dt) + dx * dx,
            -(dt * dt) + 2.0 * dx * dx,
            -(dt * dt) + 3.0 * dx * dx,
            dx * dx,
            2.0 * dx * dx,
            3.0 * dx * dx,
            dx * dx,
            2.0 * dx * dx,
            dx * dx,
        ];

        let cm = cm_matrix_signed(&sq_lor);
        let mut cof = [[0.0; 6]; 6];
        for i in 1..6 {
            for j in i..6 {
                cof[i][j] = cofactor_6x6(&cm, i, j);
                cof[j][i] = cof[i][j];
            }
        }

        // Verify each (l,m) pair gives a well-defined dihedral
        for l in 0..5usize {
            for m in (l + 1)..5 {
                let rl = l + 1;
                let rm = m + 1;
                let (val, ht) = compute_dihedral(cof[rl][rm], cof[rl][rl], cof[rm][rm]);
                assert!(val.is_finite(), "({l},{m}): dihedral not finite: {val}");
                eprintln!("({l},{m}): val={val:.6} type={ht:?}");
            }
        }
    }
}
