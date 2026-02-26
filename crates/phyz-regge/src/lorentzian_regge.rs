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
    HingeType, all_lorentzian_dihedrals_jacobian, cm_matrix_signed, tri_area_sq_grad_in_pent,
    triangle_area_sq_grad, triangle_area_sq_lorentzian,
};
use phyz_math::DMat;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
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

                let (val, ht) =
                    compute_dihedral(cofactors[rl][rm], cofactors[rl][rl], cofactors[rm][rm]);

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
pub fn lorentzian_regge_action_grad(complex: &SimplicialComplex, sq_lengths: &[f64]) -> Vec<f64> {
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

/// Analytical Hessian of the Lorentzian Regge action w.r.t. squared edge lengths.
///
/// H_{ef} = ∂²S/∂s_e∂s_f = Σ_t [ (∂²|A_t|/∂s_e∂s_f)·δ_t + (∂|A_t|/∂s_e)·(∂δ_t/∂s_f) ]
///
/// - Term 1: area Hessian × deficit angle (fully analytical)
/// - Term 2: area gradient × deficit gradient (deficit grad via per-pentachoron FD on dihedrals)
///
/// Cost: O(n_pent) — same as a single gradient evaluation.
pub fn lorentzian_regge_action_hessian(
    complex: &SimplicialComplex,
    sq_lengths: &[f64],
    fd_eps: f64,
) -> Vec<Vec<f64>> {
    let n_edges = complex.n_edges();
    let mut hessian = vec![vec![0.0; n_edges]; n_edges];

    let deficits = lorentzian_deficit_angles(complex, sq_lengths);

    // Local edge index for a pair of local vertices in a pentachoron.
    let local_edge_idx = |a: usize, b: usize| -> usize {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        match (lo, hi) {
            (0, 1) => 0,
            (0, 2) => 1,
            (0, 3) => 2,
            (0, 4) => 3,
            (1, 2) => 4,
            (1, 3) => 5,
            (1, 4) => 6,
            (2, 3) => 7,
            (2, 4) => 8,
            (3, 4) => 9,
            _ => unreachable!(),
        }
    };

    // Term 1: Σ_t (∂²|A_t|/∂s_e∂s_f) · δ_t
    //
    // A² = (2*s_ab*s_ac + 2*s_ac*s_bc + 2*s_bc*s_ab - s_ab² - s_ac² - s_bc²) / 16
    // The A² Hessian (w.r.t. the 3 triangle edge lengths) is constant:
    //   ∂²(A²)/∂s_i² = -2/16 = -1/8
    //   ∂²(A²)/∂s_i∂s_j = 2/16 = 1/8  (i ≠ j)
    //
    // |A| = sqrt(sign(A²) * A²) where sign handles the Lorentzian case.
    // ∂²|A|/∂s_e∂s_f = sign(A²) * [H(A²)_{ef}/(2|A|) - ∂(A²)/∂s_e · ∂(A²)/∂s_f / (4|A|³)]
    for (ti, &(deficit, _)) in deficits.iter().enumerate() {
        let e_idx = complex.tri_edge_indices(ti);
        let t = &complex.triangles[ti];

        let e01 = complex.edge_index[&sorted2(t[0], t[1])];
        let e02 = complex.edge_index[&sorted2(t[0], t[2])];
        let e12 = complex.edge_index[&sorted2(t[1], t[2])];

        let s01 = sq_lengths[e01];
        let s02 = sq_lengths[e02];
        let s12 = sq_lengths[e12];

        let a_sq = triangle_area_sq_lorentzian(s01, s02, s12);
        let abs_a = a_sq.abs().sqrt();
        if abs_a < 1e-30 {
            continue;
        }
        let sign_a = if a_sq >= 0.0 { 1.0 } else { -1.0 };
        let da_sq = triangle_area_sq_grad(s01, s02, s12);

        // A² Hessian: H_a2[i][j] = -1/8 if i==j, +1/8 if i!=j (3×3)
        // |A| Hessian: sign(A²) * [H(A²)/(2|A|) - ∇(A²)·∇(A²)ᵀ/(4|A|³)] · deficit
        let inv_2a = 1.0 / (2.0 * abs_a);
        let inv_4a3 = 1.0 / (4.0 * abs_a * abs_a * abs_a);

        for ii in 0..3 {
            for jj in 0..3 {
                let h_a2 = if ii == jj { -1.0 / 8.0 } else { 1.0 / 8.0 };
                let h_abs_a = sign_a * h_a2 * inv_2a - da_sq[ii] * da_sq[jj] * inv_4a3;
                hessian[e_idx[ii]][e_idx[jj]] += h_abs_a * deficit;
            }
        }
    }

    // Term 2: Σ_σ Σ_{t∈σ} (∂|A_t|/∂s_e) · (-∂θ_t(σ)/∂s_f)
    //
    // deficit_t = 2π - Σ_σ θ_t(σ)  (spacelike) or -Σ_σ η_t(σ)  (timelike)
    // So ∂δ_t/∂s_f = -Σ_σ ∂θ_t(σ)/∂s_f
    // The per-pentachoron contribution is: (∂|A_t|/∂s_e) · (-∂θ_t(σ)/∂s_f)
    for pi in 0..complex.pents.len() {
        let pent_sq = pent_sq_lengths(complex, pi, sq_lengths);
        let global_edge = complex.pent_edge_indices(pi);
        let dihedral_jac = all_lorentzian_dihedrals_jacobian(&pent_sq, fd_eps);

        // For each triangle in the pentachoron (= each (l,m) pair)
        let mut dih_idx = 0;
        for l in 0..5usize {
            for m in (l + 1)..5 {
                // Triangle opposite to (l,m): local vertices {0..4}\{l,m}
                let mut tri_local = [0usize; 3];
                let mut ti_local = 0;
                for v in 0..5 {
                    if v != l && v != m {
                        tri_local[ti_local] = v;
                        ti_local += 1;
                    }
                }

                // Compute ∂(A²)/∂s in local pent coords (10 components, 3 nonzero)
                let area_grad_local = tri_area_sq_grad_in_pent(&pent_sq, tri_local);

                // Compute |A| from local pent edges
                let e_ab = local_edge_idx(tri_local[0], tri_local[1]);
                let e_ac = local_edge_idx(tri_local[0], tri_local[2]);
                let e_bc = local_edge_idx(tri_local[1], tri_local[2]);
                let a_sq = triangle_area_sq_lorentzian(pent_sq[e_ab], pent_sq[e_ac], pent_sq[e_bc]);
                let abs_a = a_sq.abs().sqrt();
                if abs_a < 1e-30 {
                    dih_idx += 1;
                    continue;
                }
                let sign_a = if a_sq >= 0.0 { 1.0 } else { -1.0 };
                let area_factor = sign_a / (2.0 * abs_a);

                // Accumulate: (∂|A_t|/∂s_e) · (-∂θ_t(σ)/∂s_f)
                for e_local in 0..10 {
                    let da_e = area_factor * area_grad_local[e_local];
                    if da_e.abs() < 1e-30 {
                        continue;
                    }
                    let ge = global_edge[e_local];
                    for f_local in 0..10 {
                        let gf = global_edge[f_local];
                        hessian[ge][gf] += da_e * (-dihedral_jac[dih_idx][f_local]);
                    }
                }

                dih_idx += 1;
            }
        }
    }

    hessian
}

/// Compute Lorentzian deficit angles only at triangles relevant to a set of free edges.
///
/// "Relevant" = any triangle reachable from a free edge via `edge_to_tris`.
/// Deficit angles are computed correctly using ALL pents per triangle (via `tri_to_pents`).
///
/// Returns a HashMap: tri_idx → (deficit, hinge_type).
pub fn local_deficit_angles(
    complex: &SimplicialComplex,
    sq_lengths: &[f64],
    free_edges: &[usize],
) -> HashMap<usize, (f64, HingeType)> {
    // Collect relevant triangles
    let mut relevant_tris: HashSet<usize> = HashSet::new();
    for &ei in free_edges {
        for &ti in &complex.edge_to_tris[ei] {
            relevant_tris.insert(ti);
        }
    }

    let relevant_tris_vec: Vec<usize> = relevant_tris.into_iter().collect();

    #[cfg(feature = "parallel")]
    let iter = relevant_tris_vec.par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = relevant_tris_vec.iter();

    iter.map(|&ti| {
        let mut hinge_type: Option<HingeType> = None;
        let mut angle_sum = 0.0;

        // Use ALL pents touching this triangle (critical for correct deficit)
        for &pi in &complex.tri_to_pents[ti] {
            let pent = &complex.pents[pi];
            let tri = &complex.triangles[ti];
            let (l, m) = SimplicialComplex::opposite_local_indices(pent, tri);

            let pent_sq = pent_sq_lengths(complex, pi, sq_lengths);
            let cm = cm_matrix_signed(&pent_sq);

            let rl = l + 1;
            let rm = m + 1;
            let c_lm = cofactor_6x6(&cm, rl, rm);
            let c_ll = cofactor_6x6(&cm, rl, rl);
            let c_mm = cofactor_6x6(&cm, rm, rm);

            let (val, ht) = compute_dihedral(c_lm, c_ll, c_mm);

            if hinge_type.is_none() {
                hinge_type = Some(ht);
            }
            angle_sum += val;
        }

        let ht = hinge_type.unwrap_or(HingeType::Spacelike);
        let deficit = match ht {
            HingeType::Spacelike => 2.0 * PI - angle_sum,
            HingeType::Timelike => -angle_sum,
        };
        (ti, (deficit, ht))
    })
    .collect()
}

/// Local Hessian of the Lorentzian Regge action restricted to free edges.
///
/// Only processes pentachorons and triangles touching the free edges.
/// Returns an n_free × n_free DMatrix directly — no full n_edges × n_edges allocation.
pub fn local_lorentzian_regge_hessian(
    complex: &SimplicialComplex,
    sq_lengths: &[f64],
    free_edges: &[usize],
    fd_eps: f64,
) -> DMat {
    let n_free = free_edges.len();
    let free_set: HashSet<usize> = free_edges.iter().copied().collect();
    let free_idx: HashMap<usize, usize> = free_edges
        .iter()
        .enumerate()
        .map(|(i, &e)| (e, i))
        .collect();

    let mut hessian = DMat::zeros(n_free, n_free);

    // Compute local deficit angles
    let deficits = local_deficit_angles(complex, sq_lengths, free_edges);

    // Local edge index for a pair of local vertices in a pentachoron.
    let local_edge_idx = |a: usize, b: usize| -> usize {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        match (lo, hi) {
            (0, 1) => 0,
            (0, 2) => 1,
            (0, 3) => 2,
            (0, 4) => 3,
            (1, 2) => 4,
            (1, 3) => 5,
            (1, 4) => 6,
            (2, 3) => 7,
            (2, 4) => 8,
            (3, 4) => 9,
            _ => unreachable!(),
        }
    };

    // Term 1: Σ_t (∂²|A_t|/∂s_e∂s_f) · δ_t  — only relevant triangles
    for (&ti, &(deficit, _)) in &deficits {
        let e_idx = complex.tri_edge_indices(ti);
        let t = &complex.triangles[ti];

        // Check if any edge of this triangle is free
        let has_free = e_idx.iter().any(|ei| free_set.contains(ei));
        if !has_free {
            continue;
        }

        let e01 = complex.edge_index[&sorted2(t[0], t[1])];
        let e02 = complex.edge_index[&sorted2(t[0], t[2])];
        let e12 = complex.edge_index[&sorted2(t[1], t[2])];

        let s01 = sq_lengths[e01];
        let s02 = sq_lengths[e02];
        let s12 = sq_lengths[e12];

        let a_sq = triangle_area_sq_lorentzian(s01, s02, s12);
        let abs_a = a_sq.abs().sqrt();
        if abs_a < 1e-30 {
            continue;
        }
        let sign_a = if a_sq >= 0.0 { 1.0 } else { -1.0 };
        let da_sq = triangle_area_sq_grad(s01, s02, s12);

        let inv_2a = 1.0 / (2.0 * abs_a);
        let inv_4a3 = 1.0 / (4.0 * abs_a * abs_a * abs_a);

        for ii in 0..3 {
            let Some(&fi) = free_idx.get(&e_idx[ii]) else {
                continue;
            };
            for jj in 0..3 {
                let Some(&fj) = free_idx.get(&e_idx[jj]) else {
                    continue;
                };
                let h_a2 = if ii == jj { -1.0 / 8.0 } else { 1.0 / 8.0 };
                let h_abs_a = sign_a * h_a2 * inv_2a - da_sq[ii] * da_sq[jj] * inv_4a3;
                hessian[(fi, fj)] += h_abs_a * deficit;
            }
        }
    }

    // Term 2: per-pentachoron dihedral Jacobian — only relevant pents
    let mut relevant_pents: HashSet<usize> = HashSet::new();
    for &ei in free_edges {
        for &pi in &complex.edge_to_pents[ei] {
            relevant_pents.insert(pi);
        }
    }
    let relevant_pents_vec: Vec<usize> = relevant_pents.into_iter().collect();

    let accumulate = |mut local_h: DMat, &pi: &usize| -> DMat {
        let pent_sq = pent_sq_lengths(complex, pi, sq_lengths);
        let global_edge = complex.pent_edge_indices(pi);
        let dihedral_jac = all_lorentzian_dihedrals_jacobian(&pent_sq, fd_eps);

        let mut dih_idx = 0;
        for l in 0..5usize {
            for m in (l + 1)..5 {
                // Triangle opposite to (l,m)
                let mut tri_local = [0usize; 3];
                let mut ti_local = 0;
                for v in 0..5 {
                    if v != l && v != m {
                        tri_local[ti_local] = v;
                        ti_local += 1;
                    }
                }

                let area_grad_local = tri_area_sq_grad_in_pent(&pent_sq, tri_local);

                let e_ab = local_edge_idx(tri_local[0], tri_local[1]);
                let e_ac = local_edge_idx(tri_local[0], tri_local[2]);
                let e_bc = local_edge_idx(tri_local[1], tri_local[2]);
                let a_sq = triangle_area_sq_lorentzian(pent_sq[e_ab], pent_sq[e_ac], pent_sq[e_bc]);
                let abs_a = a_sq.abs().sqrt();
                if abs_a < 1e-30 {
                    dih_idx += 1;
                    continue;
                }
                let sign_a = if a_sq >= 0.0 { 1.0 } else { -1.0 };
                let area_factor = sign_a / (2.0 * abs_a);

                for e_local in 0..10 {
                    let ge = global_edge[e_local];
                    let Some(&fi) = free_idx.get(&ge) else {
                        continue;
                    };
                    let da_e = area_factor * area_grad_local[e_local];
                    if da_e.abs() < 1e-30 {
                        continue;
                    }
                    for f_local in 0..10 {
                        let gf = global_edge[f_local];
                        let Some(&fj) = free_idx.get(&gf) else {
                            continue;
                        };
                        local_h[(fi, fj)] += da_e * (-dihedral_jac[dih_idx][f_local]);
                    }
                }

                dih_idx += 1;
            }
        }
        local_h
    };

    #[cfg(feature = "parallel")]
    let term2 = relevant_pents_vec
        .par_iter()
        .fold(
            || DMat::zeros(n_free, n_free),
            |local_h, pi| accumulate(local_h, pi),
        )
        .reduce(|| DMat::zeros(n_free, n_free), |a, b| &a + &b);

    #[cfg(not(feature = "parallel"))]
    let term2 = relevant_pents_vec
        .iter()
        .fold(DMat::zeros(n_free, n_free), |local_h, pi| {
            accumulate(local_h, pi)
        });

    &hessian + &term2
}

/// Local gradient of the Lorentzian Regge action restricted to free edges.
///
/// Only processes triangles touching free edges, returning a Vec<f64> of length n_free.
pub fn local_lorentzian_regge_action_grad(
    complex: &SimplicialComplex,
    sq_lengths: &[f64],
    free_edges: &[usize],
) -> Vec<f64> {
    let n_free = free_edges.len();
    let free_idx: HashMap<usize, usize> = free_edges
        .iter()
        .enumerate()
        .map(|(i, &e)| (e, i))
        .collect();

    let deficits = local_deficit_angles(complex, sq_lengths, free_edges);
    let mut grad = vec![0.0; n_free];

    for (&ti, &(deficit, _)) in &deficits {
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
        let sign_a = if a_sq >= 0.0 { 1.0 } else { -1.0 };
        let factor = sign_a / (2.0 * abs_a);

        for (k, &ei) in e_indices.iter().enumerate() {
            if let Some(&fi) = free_idx.get(&ei) {
                grad[fi] += factor * da_sq[k] * deficit;
            }
        }
    }

    grad
}

/// Get the 10 signed squared edge lengths of a 4-simplex.
pub(crate) fn pent_sq_lengths(
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

    /// Analytical Hessian vs full finite-difference Jacobian of the gradient.
    #[test]
    fn test_hessian_vs_fd_jacobian() {
        use crate::foliation::{flat_minkowski_sq_lengths, foliated_hypercubic};
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let fc = foliated_hypercubic(2, 2);
        let flat_sq = flat_minkowski_sq_lengths(&fc, 1.0, 0.3);
        let mut sq_lengths = flat_sq.clone();

        // Perturb slightly so we're not at a degenerate point
        let mut rng = StdRng::seed_from_u64(77);
        for s in &mut sq_lengths {
            *s *= 1.0 + 0.002 * (rng.r#gen::<f64>() - 0.5);
        }

        let complex = &fc.complex;
        let n = complex.n_edges();
        let hessian = lorentzian_regge_action_hessian(complex, &sq_lengths, 1e-7);

        // FD Jacobian of the gradient
        let eps = 1e-6;
        let mut fd_hessian = vec![vec![0.0; n]; n];
        let mut work = sq_lengths.clone();
        for j in 0..n {
            let old = work[j];
            work[j] = old + eps;
            let grad_p = lorentzian_regge_action_grad(complex, &work);
            work[j] = old - eps;
            let grad_m = lorentzian_regge_action_grad(complex, &work);
            work[j] = old;
            for i in 0..n {
                fd_hessian[i][j] = (grad_p[i] - grad_m[i]) / (2.0 * eps);
            }
        }

        // Compare
        let mut max_err = 0.0f64;
        let mut max_scale = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let err = (hessian[i][j] - fd_hessian[i][j]).abs();
                max_err = max_err.max(err);
                max_scale = max_scale.max(fd_hessian[i][j].abs());
            }
        }

        let rel_err = max_err / max_scale.max(1e-15);
        eprintln!(
            "Hessian vs FD: max_err={max_err:.2e}, max_scale={max_scale:.2e}, rel={rel_err:.2e}"
        );
        assert!(
            rel_err < 1e-3,
            "Hessian relative error too large: {rel_err:.2e}"
        );
    }

    /// Hessian is approximately symmetric.
    #[test]
    fn test_hessian_symmetry() {
        use crate::foliation::{flat_minkowski_sq_lengths, foliated_hypercubic};
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let fc = foliated_hypercubic(2, 2);
        let mut sq_lengths = flat_minkowski_sq_lengths(&fc, 1.0, 0.3);
        let mut rng = StdRng::seed_from_u64(55);
        for s in &mut sq_lengths {
            *s *= 1.0 + 0.002 * (rng.r#gen::<f64>() - 0.5);
        }

        let hessian = lorentzian_regge_action_hessian(&fc.complex, &sq_lengths, 1e-7);
        let n = fc.complex.n_edges();
        let mut max_asym = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                max_asym = max_asym.max((hessian[i][j] - hessian[j][i]).abs());
            }
        }
        eprintln!("max asymmetry = {max_asym:.2e}");
        assert!(
            max_asym < 1e-6,
            "Hessian not symmetric: max_asym={max_asym:.2e}"
        );
    }

    /// Local Hessian matches the submatrix of the full Hessian at free edges.
    #[test]
    fn test_local_hessian_matches_full() {
        use crate::foliation::{flat_minkowski_sq_lengths, foliated_hypercubic};
        use crate::tent_move::tent_edges_for_vertex;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let fc = foliated_hypercubic(4, 2);
        let mut sq_lengths = flat_minkowski_sq_lengths(&fc, 1.0, 0.3);
        let mut rng = StdRng::seed_from_u64(88);
        for s in &mut sq_lengths {
            *s *= 1.0 + 0.002 * (rng.r#gen::<f64>() - 0.5);
        }

        let v = fc.global_vertex(1, 0);
        let free = tent_edges_for_vertex(&fc, v, 2);
        let n_free = free.len();

        let full_hessian = lorentzian_regge_action_hessian(&fc.complex, &sq_lengths, 1e-7);
        let local_hessian = local_lorentzian_regge_hessian(&fc.complex, &sq_lengths, &free, 1e-7);

        let mut max_err = 0.0f64;
        let mut max_scale = 0.0f64;
        for (i, &ei) in free.iter().enumerate() {
            for (j, &ej) in free.iter().enumerate() {
                let full_val = full_hessian[ei][ej];
                let local_val = local_hessian[(i, j)];
                max_err = max_err.max((full_val - local_val).abs());
                max_scale = max_scale.max(full_val.abs());
            }
        }

        let rel_err = max_err / max_scale.max(1e-15);
        eprintln!(
            "local vs full Hessian: n_free={n_free}, max_err={max_err:.2e}, rel={rel_err:.2e}"
        );
        assert!(
            rel_err < 1e-10,
            "local Hessian relative error too large: {rel_err:.2e}"
        );
    }

    /// Local gradient matches the full gradient at free edges.
    #[test]
    fn test_local_gradient_matches_full() {
        use crate::foliation::{flat_minkowski_sq_lengths, foliated_hypercubic};
        use crate::tent_move::tent_edges_for_vertex;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let fc = foliated_hypercubic(4, 2);
        let mut sq_lengths = flat_minkowski_sq_lengths(&fc, 1.0, 0.3);
        let mut rng = StdRng::seed_from_u64(88);
        for s in &mut sq_lengths {
            *s *= 1.0 + 0.002 * (rng.r#gen::<f64>() - 0.5);
        }

        let v = fc.global_vertex(1, 0);
        let free = tent_edges_for_vertex(&fc, v, 2);

        let full_grad = lorentzian_regge_action_grad(&fc.complex, &sq_lengths);
        let local_grad = local_lorentzian_regge_action_grad(&fc.complex, &sq_lengths, &free);

        let mut max_err = 0.0f64;
        let mut max_scale = 0.0f64;
        for (i, &ei) in free.iter().enumerate() {
            max_err = max_err.max((full_grad[ei] - local_grad[i]).abs());
            max_scale = max_scale.max(full_grad[ei].abs());
        }

        let rel_err = max_err / max_scale.max(1e-15);
        eprintln!("local vs full gradient: max_err={max_err:.2e}, rel={rel_err:.2e}");
        assert!(
            rel_err < 1e-10,
            "local gradient relative error too large: {rel_err:.2e}"
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
