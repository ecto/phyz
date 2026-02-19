//! Geometric computations from edge lengths via Cayley-Menger determinants.
//!
//! All geometry (areas, volumes, dihedral angles) is computed purely from
//! edge lengths — no embedding coordinates needed. This is essential for
//! Regge calculus where edge lengths are the fundamental variables.

/// Triangle area from edge lengths via Heron's formula.
///
/// Given edges a, b, c:
/// A = (1/4) sqrt(2a²b² + 2b²c² + 2c²a² - a⁴ - b⁴ - c⁴)
pub fn triangle_area(a: f64, b: f64, c: f64) -> f64 {
    let a2 = a * a;
    let b2 = b * b;
    let c2 = c * c;
    let val = 2.0 * a2 * b2 + 2.0 * b2 * c2 + 2.0 * c2 * a2 - a2 * a2 - b2 * b2 - c2 * c2;
    if val <= 0.0 {
        return 0.0;
    }
    0.25 * val.sqrt()
}

/// Derivative of triangle area w.r.t. edge length squared.
///
/// Returns (∂A/∂(a²), ∂A/∂(b²), ∂A/∂(c²)).
/// Useful because the Regge action gradient ∂S/∂l_e = (∂S/∂l²) · 2l.
pub fn triangle_area_grad_lsq(a: f64, b: f64, c: f64) -> [f64; 3] {
    let a2 = a * a;
    let b2 = b * b;
    let c2 = c * c;
    let val = 2.0 * a2 * b2 + 2.0 * b2 * c2 + 2.0 * c2 * a2 - a2 * a2 - b2 * b2 - c2 * c2;
    if val <= 0.0 {
        return [0.0; 3];
    }
    // A = 0.25 * sqrt(val), so ∂A/∂(a²) = 0.25 * 1/(2*sqrt(val)) * ∂val/∂(a²)
    let inv_4_sqrt = 0.125 / val.sqrt();

    // ∂val/∂(a²) = 2b² + 2c² - 2a²
    // ∂val/∂(b²) = 2a² + 2c² - 2b²
    // ∂val/∂(c²) = 2a² + 2b² - 2c²
    [
        inv_4_sqrt * (2.0 * b2 + 2.0 * c2 - 2.0 * a2),
        inv_4_sqrt * (2.0 * a2 + 2.0 * c2 - 2.0 * b2),
        inv_4_sqrt * (2.0 * a2 + 2.0 * b2 - 2.0 * c2),
    ]
}

/// 6×6 Cayley-Menger matrix for a 4-simplex.
///
/// For vertices 0..4 with squared distances d²_{ij}:
///
/// ```text
/// CM = | 0  1      1      1      1      1     |
///      | 1  0      d01²   d02²   d03²   d04²  |
///      | 1  d01²   0      d12²   d13²   d14²  |
///      | 1  d02²   d12²   0      d23²   d24²  |
///      | 1  d03²   d13²   d23²   0      d34²  |
///      | 1  d04²   d14²   d24²   d34²  0      |
/// ```
///
/// Input: 10 edge lengths in order [l01,l02,l03,l04,l12,l13,l14,l23,l24,l34].
pub fn cayley_menger_matrix(lengths: &[f64; 10]) -> [[f64; 6]; 6] {
    // Map (i,j) with i<j in 0..5 to index in lengths array.
    // (0,1)→0, (0,2)→1, (0,3)→2, (0,4)→3, (1,2)→4, (1,3)→5, (1,4)→6, (2,3)→7, (2,4)→8, (3,4)→9
    let d2 = |i: usize, j: usize| -> f64 {
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
        lengths[idx] * lengths[idx]
    };

    let mut cm = [[0.0; 6]; 6];

    // Row/col 0 is the border.
    for i in 1..6 {
        cm[0][i] = 1.0;
        cm[i][0] = 1.0;
    }

    // Fill squared distances. CM[i+1][j+1] = d²_{ij}.
    for i in 0..5 {
        for j in 0..5 {
            cm[i + 1][j + 1] = d2(i, j);
        }
    }

    cm
}

/// Compute the (i,j) cofactor of a 6×6 matrix.
///
/// Cofactor C_{ij} = (-1)^{i+j} det(minor_{ij}).
pub fn cofactor_6x6(m: &[[f64; 6]; 6], row: usize, col: usize) -> f64 {
    // Build 5×5 minor by deleting row and col.
    let mut minor = [[0.0; 5]; 5];
    let mut mi = 0;
    for i in 0..6 {
        if i == row {
            continue;
        }
        let mut mj = 0;
        for j in 0..6 {
            if j == col {
                continue;
            }
            minor[mi][mj] = m[i][j];
            mj += 1;
        }
        mi += 1;
    }

    let sign = if (row + col).is_multiple_of(2) { 1.0 } else { -1.0 };
    sign * det_5x5(&minor)
}

/// Determinant of a 5×5 matrix via cofactor expansion along first row.
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

/// Determinant of a 4×4 matrix.
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

/// Determinant of a 3×3 matrix.
fn det_3x3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Determinant of the full 6×6 Cayley-Menger matrix.
pub fn cayley_menger_det(lengths: &[f64; 10]) -> f64 {
    let cm = cayley_menger_matrix(lengths);
    det_6x6(&cm)
}

/// Determinant of a 6×6 matrix via cofactor expansion.
fn det_6x6(m: &[[f64; 6]; 6]) -> f64 {
    let mut det = 0.0;
    for j in 0..6 {
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        // cofactor_6x6 includes the sign, so just use the minor determinant directly.
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

/// Volume of a 4-simplex from 10 edge lengths.
///
/// V₄ = sqrt(|det(CM)| / 2⁴ / (4!)²)
///     = sqrt(|det(CM)| / 9216)
///
/// Note: det(CM) should be negative for a valid 4-simplex in 4D.
/// V₄² = (-1)^5 det(CM) / (2^4 (4!)^2) = -det(CM) / 9216.
pub fn pent_volume(lengths: &[f64; 10]) -> f64 {
    let det = cayley_menger_det(lengths);
    // For a 4-simplex in 4D: V² = (-1)^{d+1} det(CM) / (2^d (d!)^2)
    // d=4: V² = -det(CM) / (16 * 576) = -det(CM) / 9216
    let v_sq = -det / 9216.0;
    if v_sq <= 0.0 {
        return 0.0;
    }
    v_sq.sqrt()
}

/// Gradient of 4-simplex volume w.r.t. its 10 edge lengths.
///
/// Uses Cayley-Menger cofactors:
///   ∂V/∂l_e = −2·l_e · C_{a+1,b+1} / (9216·V)
///
/// where (a,b) is the vertex pair for edge e, and C_{i,j} is the cofactor
/// of the 6×6 CM matrix. The factor of 2 accounts for the symmetric
/// off-diagonal entries (l² appears at both (r,c) and (c,r)).
///
/// Returns zeros for degenerate simplices (V ≤ 0).
pub fn pent_volume_grad(lengths: &[f64; 10]) -> [f64; 10] {
    let cm = cayley_menger_matrix(lengths);
    let det = det_6x6(&cm);
    let v_sq = -det / 9216.0;
    if v_sq <= 0.0 {
        return [0.0; 10];
    }
    let v = v_sq.sqrt();

    // Edge ordering: (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)
    let pairs: [(usize, usize); 10] = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ];

    let mut grad = [0.0; 10];
    for (idx, &(a, b)) in pairs.iter().enumerate() {
        // CM rows/cols are offset by 1 (row 0 is the border).
        let cof = cofactor_6x6(&cm, a + 1, b + 1);
        // ∂(det)/∂(l²) = 2·cofactor (symmetric matrix, l² at (r,c) and (c,r))
        // ∂V/∂l = ∂V/∂(l²) · 2l = [-1/(2·9216·V)] · 2·cof · 2l
        // Simplify: ∂V/∂l = -2·l·cof / (9216·V)
        grad[idx] = -2.0 * lengths[idx] * cof / (9216.0 * v);
    }

    grad
}

/// Dihedral angle at a triangle in a 4-simplex.
///
/// In a 4-simplex [v0,v1,v2,v3,v4], the dihedral angle at triangle
/// [vi,vj,vk] (between the two tetrahedra sharing that triangle) is
/// computed from the Cayley-Menger cofactors.
///
/// The triangle [vi,vj,vk] is "opposite" to the edge [vl,vm].
/// In the CM matrix (with border row/col 0), vertices map to rows/cols 1..5.
///
/// cos(θ) = -C_{l+1,m+1} / sqrt(C_{l+1,l+1} · C_{m+1,m+1})
///
/// where C_{ij} is the (i,j) cofactor of the 6×6 CM matrix, and
/// l,m are the local indices (0..4) of the two opposite vertices.
///
/// # Arguments
/// * `lengths` - 10 edge lengths of the 4-simplex
/// * `local_l` - local vertex index (0..4) of first opposite vertex
/// * `local_m` - local vertex index (0..4) of second opposite vertex
pub fn dihedral_angle(lengths: &[f64; 10], local_l: usize, local_m: usize) -> f64 {
    let cm = cayley_menger_matrix(lengths);

    // CM indices are offset by 1 (row/col 0 is border).
    let rl = local_l + 1;
    let rm = local_m + 1;

    let c_lm = cofactor_6x6(&cm, rl, rm);
    let c_ll = cofactor_6x6(&cm, rl, rl);
    let c_mm = cofactor_6x6(&cm, rm, rm);

    let denom = (c_ll * c_mm).abs().sqrt();
    if denom < 1e-30 {
        return 0.0;
    }

    let cos_theta = -c_lm / denom;
    // Clamp to [-1, 1] for numerical safety.
    cos_theta.clamp(-1.0, 1.0).acos()
}

/// Compute all 10 dihedral angles of a 4-simplex at once.
///
/// Returns angles indexed the same way as edges:
/// [θ_{01}, θ_{02}, θ_{03}, θ_{04}, θ_{12}, θ_{13}, θ_{14}, θ_{23}, θ_{24}, θ_{34}]
///
/// where θ_{lm} is the dihedral angle at the triangle *opposite* to edge (l,m).
///
/// This is more efficient than calling `dihedral_angle` 10 times because
/// it computes the CM matrix and all needed cofactors once.
pub fn all_dihedral_angles(lengths: &[f64; 10]) -> [f64; 10] {
    let cm = cayley_menger_matrix(lengths);

    // Precompute all cofactors we need: C_{i,j} for i,j ∈ 1..5.
    // We need C_{ij} for i ≤ j (symmetric under transposition for real symmetric matrix).
    let mut cofactors = [[0.0; 6]; 6];
    for i in 1..6 {
        for j in i..6 {
            let c = cofactor_6x6(&cm, i, j);
            cofactors[i][j] = c;
            cofactors[j][i] = c;
        }
    }

    let mut angles = [0.0; 10];
    let mut idx = 0;
    for l in 0..5usize {
        for m in (l + 1)..5 {
            let rl = l + 1;
            let rm = m + 1;
            let c_lm = cofactors[rl][rm];
            let c_ll = cofactors[rl][rl];
            let c_mm = cofactors[rm][rm];

            let denom = (c_ll * c_mm).abs().sqrt();
            let cos_theta = if denom > 1e-30 {
                (-c_lm / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            angles[idx] = cos_theta.acos();
            idx += 1;
        }
    }

    angles
}

/// Gradient of a single dihedral angle w.r.t. edge lengths.
///
/// Computed via finite differences on the Cayley-Menger cofactor formula.
/// Returns ∂θ/∂l_e for each of the 10 edges.
pub fn dihedral_angle_grad(
    lengths: &[f64; 10],
    local_l: usize,
    local_m: usize,
    eps: f64,
) -> [f64; 10] {
    let theta_0 = dihedral_angle(lengths, local_l, local_m);
    let mut grad = [0.0; 10];
    for i in 0..10 {
        let mut l_plus = *lengths;
        l_plus[i] += eps;
        let theta_plus = dihedral_angle(&l_plus, local_l, local_m);

        let mut l_minus = *lengths;
        l_minus[i] -= eps;
        let theta_minus = dihedral_angle(&l_minus, local_l, local_m);

        grad[i] = (theta_plus - theta_minus) / (2.0 * eps);
    }
    let _ = theta_0;
    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_equilateral_triangle_area() {
        let a = 1.0;
        let area = triangle_area(a, a, a);
        let expected = (3.0_f64).sqrt() / 4.0;
        assert!((area - expected).abs() < 1e-12);
    }

    #[test]
    fn test_right_triangle_area() {
        let area = triangle_area(3.0, 4.0, 5.0);
        assert!((area - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_regular_pentachoron_volume() {
        // Regular 4-simplex with edge length 1.
        // Volume = sqrt(5) / 96 * a^4 ≈ 0.02329.
        let lengths = [1.0; 10];
        let vol = pent_volume(&lengths);
        let expected = 5.0_f64.sqrt() / 96.0;
        assert!(
            (vol - expected).abs() < 1e-6,
            "got {vol}, expected {expected}"
        );
    }

    #[test]
    fn test_regular_pentachoron_dihedral() {
        // Regular 4-simplex: all dihedral angles are arccos(1/4) ≈ 75.52°.
        let lengths = [1.0; 10];
        let angles = all_dihedral_angles(&lengths);

        let expected = (1.0_f64 / 4.0).acos();
        for (i, &angle) in angles.iter().enumerate() {
            assert!(
                (angle - expected).abs() < 1e-8,
                "angle[{i}] = {angle}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_right_angle_simplex() {
        // 4-simplex with vertices at origin and along axes:
        // v0 = (0,0,0,0), v1 = (1,0,0,0), v2 = (0,1,0,0), v3 = (0,0,1,0), v4 = (0,0,0,1)
        // Edge lengths:
        // l01=l02=l03=l04=1, l12=l13=l14=l23=l24=l34=sqrt(2)
        let s2 = 2.0_f64.sqrt();
        let lengths = [1.0, 1.0, 1.0, 1.0, s2, s2, s2, s2, s2, s2];

        let vol = pent_volume(&lengths);
        // Volume of standard 4-simplex = 1/24
        let expected = 1.0 / 24.0;
        assert!(
            (vol - expected).abs() < 1e-10,
            "got {vol}, expected {expected}"
        );
    }

    #[test]
    fn test_pent_volume_grad_vs_fd() {
        // Irregular simplex — analytical gradient should match finite differences.
        let lengths = [1.0, 1.1, 0.9, 1.2, 1.05, 0.95, 1.15, 1.0, 1.1, 0.85];
        let grad = pent_volume_grad(&lengths);

        let eps = 1e-7;
        for i in 0..10 {
            let mut lp = lengths;
            lp[i] += eps;
            let mut lm = lengths;
            lm[i] -= eps;
            let fd = (pent_volume(&lp) - pent_volume(&lm)) / (2.0 * eps);
            assert!(
                (grad[i] - fd).abs() < 1e-6,
                "edge {i}: analytical={}, fd={}",
                grad[i],
                fd
            );
        }
    }

    #[test]
    fn test_pent_volume_grad_regular_symmetry() {
        // Regular simplex: all gradients should be equal.
        let lengths = [1.0; 10];
        let grad = pent_volume_grad(&lengths);
        for i in 1..10 {
            assert!(
                (grad[i] - grad[0]).abs() < 1e-12,
                "grad[{i}]={} != grad[0]={}",
                grad[i],
                grad[0]
            );
        }
        // Should be positive (increasing edge → increasing volume for regular simplex).
        assert!(grad[0] > 0.0, "grad[0]={} should be positive", grad[0]);
    }

    #[test]
    fn test_dihedral_angle_finite_diff_consistency() {
        // Check that dihedral_angle and all_dihedral_angles agree.
        let lengths = [1.0, 1.1, 0.9, 1.2, 1.05, 0.95, 1.15, 1.0, 1.1, 0.85];
        let all = all_dihedral_angles(&lengths);

        // Check a specific one: angle at triangle opposite to edge (0,1)
        let single = dihedral_angle(&lengths, 0, 1);
        assert!(
            (all[0] - single).abs() < 1e-12,
            "all[0]={}, single={}",
            all[0],
            single
        );
    }
}
