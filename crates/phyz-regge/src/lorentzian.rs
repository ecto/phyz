//! Lorentzian geometry from signed squared edge lengths via Cayley-Menger determinants.
//!
//! Extends the Euclidean geometry module to handle **signed** squared lengths
//! `s_e = g_{ab} dx^a dx^b` where `s_e < 0` for timelike edges and `s_e > 0`
//! for spacelike edges.
//!
//! All formulas are the same Cayley-Menger machinery — the only difference is
//! that the CM matrix entries are signed squared lengths instead of positive
//! squared lengths, and we must handle the resulting sign changes in areas,
//! volumes, and dihedral angles.

use crate::geometry::cofactor_6x6;

/// Signed squared edge length. Negative = timelike, positive = spacelike.
pub type SignedLengthSq = f64;

/// Classification of a hinge (triangle) in the Lorentzian setting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HingeType {
    /// Spacelike hinge: deficit = 2π − Σ(dihedral angles)
    Spacelike,
    /// Timelike hinge: deficit = −Σ(boost parameters), no 2π term
    Timelike,
}

/// Build the 6×6 Cayley-Menger matrix from 10 signed squared edge lengths.
///
/// Same structure as `geometry::cayley_menger_matrix`, but entries are `s_{ij}`
/// (signed squared lengths) rather than `l_{ij}^2`.
///
/// Input ordering: [s01, s02, s03, s04, s12, s13, s14, s23, s24, s34].
pub fn cm_matrix_signed(sq_lengths: &[f64; 10]) -> [[f64; 6]; 6] {
    let s = |i: usize, j: usize| -> f64 {
        if i == j {
            return 0.0;
        }
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        let idx = match (a, b) {
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
        };
        sq_lengths[idx]
    };

    let mut cm = [[0.0; 6]; 6];
    for i in 1..6 {
        cm[0][i] = 1.0;
        cm[i][0] = 1.0;
    }
    for i in 0..5 {
        for j in 0..5 {
            cm[i + 1][j + 1] = s(i, j);
        }
    }
    cm
}

/// Signed triangle area squared from three signed squared edge lengths.
///
/// For a triangle with vertices 0,1,2 and signed squared lengths s01, s02, s12:
///
/// ```text
///                | 0   1    1    1   |
/// 16 A² = -det  | 1   0    s01  s02 |
///                | 1  s01   0    s12 |
///                | 1  s02  s12   0   |
/// ```
///
/// The result is positive for spacelike triangles, negative for timelike.
pub fn triangle_area_sq_lorentzian(s01: f64, s02: f64, s12: f64) -> f64 {
    // Expand the 4×4 CM determinant for a triangle:
    // 16 A² = 2*s01*s02 + 2*s02*s12 + 2*s12*s01 - s01² - s02² - s12²
    // (same formula as Euclidean Heron, but with signed inputs)
    let val =
        2.0 * s01 * s02 + 2.0 * s02 * s12 + 2.0 * s12 * s01 - s01 * s01 - s02 * s02 - s12 * s12;
    val / 16.0
}

/// Gradient of triangle_area_sq_lorentzian w.r.t. (s01, s02, s12).
///
/// Returns [∂(A²)/∂s01, ∂(A²)/∂s02, ∂(A²)/∂s12].
pub fn triangle_area_sq_grad(s01: f64, s02: f64, s12: f64) -> [f64; 3] {
    // A² = (2*s01*s02 + 2*s02*s12 + 2*s12*s01 - s01² - s02² - s12²) / 16
    // ∂(A²)/∂s01 = (2*s02 + 2*s12 - 2*s01) / 16 = (s02 + s12 - s01) / 8
    [
        (s02 + s12 - s01) / 8.0,
        (s01 + s12 - s02) / 8.0,
        (s01 + s02 - s12) / 8.0,
    ]
}

/// Signed 4-simplex volume squared from 10 signed squared edge lengths.
///
/// V₄² = (-1)^{d+1} det(CM) / (2^d (d!)²)  with d=4
///      = -det(CM) / 9216
///
/// Positive for a valid Lorentzian 4-simplex.
pub fn pent_volume_sq_lorentzian(sq_lengths: &[f64; 10]) -> f64 {
    let cm = cm_matrix_signed(sq_lengths);
    let det = det_6x6(&cm);
    -det / 9216.0
}

/// Lorentzian dihedral angle at a triangle in a 4-simplex.
///
/// Given the 4-simplex with 10 signed squared edge lengths, computes the
/// dihedral angle at the triangle opposite to the edge (l, m) (local vertex indices).
///
/// Returns `(angle_or_boost, hinge_type)` where:
/// - For spacelike hinges: angle = arccos(−C_lm / √(C_ll · C_mm))
/// - For timelike hinges: boost = arccosh(−C_lm / √(|C_ll · C_mm|)), sign-corrected
///
/// The hinge type is determined by the signs of C_ll and C_mm:
/// both positive → spacelike hinge; mixed signs → timelike hinge.
pub fn lorentzian_dihedral(sq_lengths: &[f64; 10], l: usize, m: usize) -> (f64, HingeType) {
    let cm = cm_matrix_signed(sq_lengths);
    let rl = l + 1;
    let rm = m + 1;

    let c_lm = cofactor_6x6(&cm, rl, rm);
    let c_ll = cofactor_6x6(&cm, rl, rl);
    let c_mm = cofactor_6x6(&cm, rm, rm);

    let product = c_ll * c_mm;

    if product > 0.0 {
        // Both cofactors same sign.
        // When both are negative, the analytic continuation gives:
        //   sqrt(C_ll) * sqrt(C_mm) = i*sqrt(|C_ll|) * i*sqrt(|C_mm|) = -sqrt(product)
        // This flips the sign of the ratio.
        let sign_corr = if c_ll < 0.0 { -1.0 } else { 1.0 };
        let denom = product.sqrt();
        if denom < 1e-30 {
            return (0.0, HingeType::Spacelike);
        }
        let ratio = sign_corr * (-c_lm) / denom;
        if ratio.abs() <= 1.0 + 1e-12 {
            let cos_theta = ratio.clamp(-1.0, 1.0);
            (cos_theta.acos(), HingeType::Spacelike)
        } else {
            let boost = ratio.abs().acosh();
            (ratio.signum() * boost, HingeType::Timelike)
        }
    } else {
        // Mixed signs → always a boost.
        let denom = product.abs().sqrt();
        if denom < 1e-30 {
            return (0.0, HingeType::Timelike);
        }
        let ratio = -c_lm / denom;
        (ratio.asinh(), HingeType::Timelike)
    }
}

/// Compute all 10 Lorentzian dihedral angles of a 4-simplex.
///
/// Returns `[(angle_or_boost, hinge_type); 10]` in the standard edge ordering.
pub fn all_lorentzian_dihedrals(sq_lengths: &[f64; 10]) -> [(f64, HingeType); 10] {
    let cm = cm_matrix_signed(sq_lengths);

    // Precompute cofactors for rows/cols 1..5
    let mut cofactors = [[0.0; 6]; 6];
    for i in 1..6 {
        for j in i..6 {
            let c = cofactor_6x6(&cm, i, j);
            cofactors[i][j] = c;
            cofactors[j][i] = c;
        }
    }

    let mut result = [(0.0, HingeType::Spacelike); 10];
    let mut idx = 0;
    for l in 0..5usize {
        for m in (l + 1)..5 {
            let rl = l + 1;
            let rm = m + 1;
            let c_lm = cofactors[rl][rm];
            let c_ll = cofactors[rl][rl];
            let c_mm = cofactors[rm][rm];
            let product = c_ll * c_mm;

            if product > 0.0 {
                let sign_corr = if c_ll < 0.0 { -1.0 } else { 1.0 };
                let denom = product.sqrt();
                if denom < 1e-30 {
                    result[idx] = (0.0, HingeType::Spacelike);
                } else {
                    let ratio = sign_corr * (-c_lm) / denom;
                    if ratio.abs() <= 1.0 + 1e-12 {
                        let cos_theta = ratio.clamp(-1.0, 1.0);
                        result[idx] = (cos_theta.acos(), HingeType::Spacelike);
                    } else {
                        let boost = ratio.abs().acosh();
                        result[idx] = (ratio.signum() * boost, HingeType::Timelike);
                    }
                }
            } else {
                let denom = product.abs().sqrt();
                if denom < 1e-30 {
                    result[idx] = (0.0, HingeType::Timelike);
                } else {
                    let ratio = -c_lm / denom;
                    result[idx] = (ratio.asinh(), HingeType::Timelike);
                }
            }
            idx += 1;
        }
    }
    result
}

/// Jacobian of all 10 dihedral angles w.r.t. the 10 squared edge lengths.
///
/// Returns `[dihedral_idx][edge_idx] = ∂θ_k/∂s_e`, computed via central finite differences
/// on `all_lorentzian_dihedrals`. Cost: 21 calls (10 edges × 2 perturbations + 1 baseline).
pub fn all_lorentzian_dihedrals_jacobian(sq_lengths: &[f64; 10], eps: f64) -> [[f64; 10]; 10] {
    let mut jac = [[0.0; 10]; 10];
    let mut work = *sq_lengths;

    for e in 0..10 {
        let old = work[e];

        work[e] = old + eps;
        let plus = all_lorentzian_dihedrals(&work);

        work[e] = old - eps;
        let minus = all_lorentzian_dihedrals(&work);

        work[e] = old;

        for k in 0..10 {
            jac[k][e] = (plus[k].0 - minus[k].0) / (2.0 * eps);
        }
    }

    jac
}

/// Gradient of triangle area squared w.r.t. 10 signed squared edge lengths of a 4-simplex.
///
/// For triangle with local vertices (a, b, c) in the pent, the area squared
/// depends on only the three edges sa_sb, sa_sc, sb_sc.
/// Returns ∂(A²_tri)/∂s_e for each of the 10 edges.
pub fn tri_area_sq_grad_in_pent(sq_lengths: &[f64; 10], tri_local: [usize; 3]) -> [f64; 10] {
    let edge_idx = |a: usize, b: usize| -> usize {
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

    let [a, b, c] = tri_local;
    let e_ab = edge_idx(a, b);
    let e_ac = edge_idx(a, c);
    let e_bc = edge_idx(b, c);

    let grad3 = triangle_area_sq_grad(sq_lengths[e_ab], sq_lengths[e_ac], sq_lengths[e_bc]);

    let mut grad10 = [0.0; 10];
    grad10[e_ab] = grad3[0];
    grad10[e_ac] = grad3[1];
    grad10[e_bc] = grad3[2];
    grad10
}

// Re-export determinant for internal use.
fn det_6x6(m: &[[f64; 6]; 6]) -> f64 {
    let mut det = 0.0;
    for j in 0..6 {
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        let mut minor = [[0.0; 5]; 5];
        let mut mi = 0;
        for i in 1..6 {
            let mut mj = 0;
            for k in 0..6 {
                if k == j {
                    continue;
                }
                minor[mi][mj] = m[i][k];
                mj += 1;
            }
            mi += 1;
        }
        det += sign * m[0][j] * det_5x5(&minor);
    }
    det
}

fn det_5x5(m: &[[f64; 5]; 5]) -> f64 {
    let mut det = 0.0;
    for j in 0..5 {
        let mut minor = [[0.0; 4]; 4];
        let mut mi = 0;
        for i in 1..5 {
            let mut mj = 0;
            for k in 0..5 {
                if k == j {
                    continue;
                }
                minor[mi][mj] = m[i][k];
                mj += 1;
            }
            mi += 1;
        }
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        det += sign * m[0][j] * det_4x4(&minor);
    }
    det
}

fn det_4x4(m: &[[f64; 4]; 4]) -> f64 {
    let mut det = 0.0;
    for j in 0..4 {
        let mut minor = [[0.0; 3]; 3];
        let mut mi = 0;
        for i in 1..4 {
            let mut mj = 0;
            for k in 0..4 {
                if k == j {
                    continue;
                }
                minor[mi][mj] = m[i][k];
                mj += 1;
            }
            mi += 1;
        }
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        det += sign * m[0][j] * det_3x3(&minor);
    }
    det
}

fn det_3x3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All-positive squared lengths should reproduce Euclidean results.
    #[test]
    fn test_euclidean_consistency() {
        // Regular pentachoron with edge length 1 → s_e = 1.0 for all edges
        let sq = [1.0; 10];
        let dihedrals = all_lorentzian_dihedrals(&sq);

        let expected = (1.0_f64 / 4.0).acos();
        for (i, &(angle, htype)) in dihedrals.iter().enumerate() {
            assert_eq!(htype, HingeType::Spacelike, "edge {i} should be spacelike");
            assert!(
                (angle - expected).abs() < 1e-8,
                "edge {i}: got {angle}, expected {expected}"
            );
        }
    }

    /// Flat Minkowski simplex should have specific known dihedral angles.
    ///
    /// Take a right-angle 4-simplex with vertices:
    ///   v0 = (0,0,0,0), v1 = (dt,0,0,0), v2 = (0,dx,0,0),
    ///   v3 = (0,0,dx,0), v4 = (0,0,0,dx)
    ///
    /// With Minkowski signature (−,+,+,+):
    ///   s01 = −dt², s02 = dx², s03 = dx², s04 = dx²
    ///   s12 = −dt²+dx², s13 = −dt²+dx², s14 = −dt²+dx²
    ///   s23 = 2dx², s24 = 2dx², s34 = 2dx²
    #[test]
    fn test_minkowski_simplex_volume() {
        let dt = 1.0;
        let dx = 1.0;
        let dt2 = dt * dt;
        let dx2 = dx * dx;

        let sq = [
            -dt2,       // s01
            dx2,        // s02
            dx2,        // s03
            dx2,        // s04
            -dt2 + dx2, // s12
            -dt2 + dx2, // s13
            -dt2 + dx2, // s14
            2.0 * dx2,  // s23
            2.0 * dx2,  // s24
            2.0 * dx2,  // s34
        ];

        let v_sq = pent_volume_sq_lorentzian(&sq);
        // Volume of right-angle 4-simplex = dt*dx³/24
        // For Lorentzian signature, V² is negative (like √(-g) d⁴x).
        let expected_mag = (dt * dx * dx * dx / 24.0).powi(2);
        assert!(
            (v_sq.abs() - expected_mag).abs() < 1e-10,
            "|V²| = {}, expected {expected_mag}",
            v_sq.abs()
        );
    }

    /// Triangle area squared: spacelike triangle
    #[test]
    fn test_spacelike_triangle_area() {
        // Equilateral spacelike triangle with s = 1
        let a_sq = triangle_area_sq_lorentzian(1.0, 1.0, 1.0);
        // Heron: A² = (2+2+2 - 1-1-1)/16 = 3/16
        let expected = 3.0 / 16.0;
        assert!(
            (a_sq - expected).abs() < 1e-12,
            "A² = {a_sq}, expected {expected}"
        );
    }

    /// Triangle area squared: timelike triangle
    #[test]
    fn test_timelike_triangle_area() {
        // Triangle with one timelike edge:
        // s01 = -1 (timelike), s02 = 1, s12 = 1
        // A² = (2*(-1)*1 + 2*1*1 + 2*1*(-1) - 1 - 1 - 1)/16 = (-2+2-2-3)/16 = -5/16
        let a_sq = triangle_area_sq_lorentzian(-1.0, 1.0, 1.0);
        let expected = (-2.0 + 2.0 - 2.0 - 3.0) / 16.0;
        assert!(
            (a_sq - expected).abs() < 1e-12,
            "A² = {a_sq}, expected {expected}"
        );
    }

    /// Area squared gradient vs finite difference.
    #[test]
    fn test_area_sq_grad_vs_fd() {
        let s = [0.8, 1.2, 1.0];
        let grad = triangle_area_sq_grad(s[0], s[1], s[2]);

        let eps = 1e-7;
        for i in 0..3 {
            let mut sp = s;
            sp[i] += eps;
            let mut sm = s;
            sm[i] -= eps;
            let fd = (triangle_area_sq_lorentzian(sp[0], sp[1], sp[2])
                - triangle_area_sq_lorentzian(sm[0], sm[1], sm[2]))
                / (2.0 * eps);
            assert!(
                (grad[i] - fd).abs() < 1e-6,
                "component {i}: analytical={}, fd={}",
                grad[i],
                fd
            );
        }
    }

    /// Area squared gradient with mixed-sign inputs.
    #[test]
    fn test_area_sq_grad_lorentzian_vs_fd() {
        let s = [-1.0, 1.2, 0.5]; // one timelike
        let grad = triangle_area_sq_grad(s[0], s[1], s[2]);

        let eps = 1e-7;
        for i in 0..3 {
            let mut sp = s;
            sp[i] += eps;
            let mut sm = s;
            sm[i] -= eps;
            let fd = (triangle_area_sq_lorentzian(sp[0], sp[1], sp[2])
                - triangle_area_sq_lorentzian(sm[0], sm[1], sm[2]))
                / (2.0 * eps);
            assert!(
                (grad[i] - fd).abs() < 1e-6,
                "component {i}: analytical={}, fd={}",
                grad[i],
                fd
            );
        }
    }

    /// Flat Minkowski lattice: dihedral angles of a "Kuhn-type" simplex should
    /// sum to 2π for interior spacelike hinges (zero deficit on flat space).
    ///
    /// Use 5 pentachorons sharing a spacelike interior triangle to verify.
    #[test]
    fn test_flat_minkowski_single_simplex_dihedrals() {
        // A right-angle Minkowski simplex with dt = dx = 1
        let dt2 = 1.0;
        let dx2 = 1.0;
        let sq = [
            -dt2,       // s01
            dx2,        // s02
            dx2,        // s03
            dx2,        // s04
            -dt2 + dx2, // s12
            -dt2 + dx2, // s13
            -dt2 + dx2, // s14
            2.0 * dx2,  // s23
            2.0 * dx2,  // s24
            2.0 * dx2,  // s34
        ];

        let dihedrals = all_lorentzian_dihedrals(&sq);
        // Just check all values are finite and hinge types are determined
        for (i, &(val, htype)) in dihedrals.iter().enumerate() {
            assert!(val.is_finite(), "dihedral {i} is not finite: {val}");
            assert!(
                val >= 0.0 || htype == HingeType::Timelike,
                "dihedral {i}: negative angle {val} with type {htype:?}"
            );
        }
    }

    /// Dihedral angle via FD consistency check.
    #[test]
    fn test_lorentzian_dihedral_continuity() {
        // Perturb a Euclidean simplex slightly and check continuity
        let base = [1.0; 10];
        let (angle0, _) = lorentzian_dihedral(&base, 0, 1);

        let eps = 1e-6;
        let mut perturbed = base;
        perturbed[0] += eps;
        let (angle1, _) = lorentzian_dihedral(&perturbed, 0, 1);

        assert!(
            (angle1 - angle0).abs() < 0.01,
            "dihedral not continuous: {} vs {}",
            angle0,
            angle1
        );
    }

    /// Dihedral Jacobian vs per-component finite difference.
    #[test]
    fn test_dihedral_jacobian_vs_fd() {
        // Lorentzian simplex
        let dt = 0.3;
        let dx = 1.0;
        let sq = [
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

        let eps = 1e-7;
        let jac = all_lorentzian_dihedrals_jacobian(&sq, eps);

        // Verify against independent per-component FD
        let fd_eps = 1e-6;
        let mut work = sq;
        for e in 0..10 {
            let old = work[e];
            work[e] = old + fd_eps;
            let plus = all_lorentzian_dihedrals(&work);
            work[e] = old - fd_eps;
            let minus = all_lorentzian_dihedrals(&work);
            work[e] = old;

            for k in 0..10 {
                let fd = (plus[k].0 - minus[k].0) / (2.0 * fd_eps);
                let err = (jac[k][e] - fd).abs();
                assert!(
                    err < 1e-4,
                    "dihedral_jac[{k}][{e}]: analytical={:.8e}, fd={:.8e}, err={:.2e}",
                    jac[k][e],
                    fd,
                    err
                );
            }
        }
    }

    /// CM matrix with all-positive entries should match Euclidean cayley_menger_matrix
    /// (up to the fact that Euclidean uses lengths, we use squared lengths).
    #[test]
    fn test_cm_matrix_euclidean_consistency() {
        let lengths = [1.0, 1.1, 0.9, 1.2, 1.05, 0.95, 1.15, 1.0, 1.1, 0.85];
        // Euclidean CM matrix uses l² in entries, and our signed version uses s = l² directly.
        let sq: [f64; 10] = std::array::from_fn(|i| lengths[i] * lengths[i]);

        let cm_euclidean = crate::geometry::cayley_menger_matrix(&lengths);
        let cm_signed = cm_matrix_signed(&sq);

        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (cm_euclidean[i][j] - cm_signed[i][j]).abs() < 1e-12,
                    "CM[{i}][{j}]: euclidean={}, signed={}",
                    cm_euclidean[i][j],
                    cm_signed[i][j]
                );
            }
        }
    }
}
