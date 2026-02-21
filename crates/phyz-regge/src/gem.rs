//! Gravitoelectromagnetic (GEM) field extraction from Regge geometry.
//!
//! Extracts gravitoelectric and gravitomagnetic fields from the evolved
//! simplicial geometry by reconstructing the Riemann tensor from deficit angles.
//!
//! # Algorithm
//!
//! Each triangle's deficit angle encodes an integrated Riemann component:
//!   δ_t / A_t ≈ R_{abcd} · n^{ab}_t · n^{cd}_t
//!
//! where n^{ab}_t is the triangle's bivector. For each vertex, collect all
//! surrounding triangles, build a linear system for the 20 independent
//! Riemann components, and solve via least-squares (SVD).
//!
//! Then extract:
//! - E_g^{ij} = R^{0i0j} (gravitoelectric tidal tensor)
//! - B_g^{ij} = (1/2) ε^{ikl} R_{0jkl} (gravitomagnetic frame-drag tensor)

use crate::complex::SimplicialComplex;
use crate::foliation::FoliatedComplex;
use crate::lorentzian_regge::lorentzian_deficit_angles;

/// Gravitoelectric and gravitomagnetic field tensors at each vertex.
#[derive(Debug, Clone)]
pub struct GemFields {
    /// Gravitoelectric tidal tensor E_g^{ij} = R^{0i0j} at each vertex.
    /// Indexed as `e_grav[vertex][i][j]` where i,j ∈ {0,1,2} (spatial indices).
    pub e_grav: Vec<[[f64; 3]; 3]>,
    /// Gravitomagnetic frame-drag tensor B_g^{ij} at each vertex.
    /// Indexed as `b_grav[vertex][i][j]`.
    pub b_grav: Vec<[[f64; 3]; 3]>,
}

/// The 20 independent components of the Riemann tensor in 4D.
///
/// Riemann symmetries: R_{abcd} = -R_{bacd} = -R_{abdc} = R_{cdab},
/// plus the Bianchi identity R_{a[bcd]} = 0.
///
/// We store them in a canonical order for the linear system.
type Riemann20 = [f64; 20];

/// Map from a pair of antisymmetric indices (a,b) with a < b to a bivector index 0..5.
///
/// Ordering: (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=4, (2,3)=5
fn bivector_index(a: usize, b: usize) -> usize {
    debug_assert!(a < b && b < 4);
    match (a, b) {
        (0, 1) => 0,
        (0, 2) => 1,
        (0, 3) => 2,
        (1, 2) => 3,
        (1, 3) => 4,
        (2, 3) => 5,
        _ => unreachable!(),
    }
}

/// Map from a pair of bivector indices (p,q) with p <= q to one of the
/// 21 symmetric entries of the 6×6 Riemann matrix.
/// Then the Bianchi identity reduces this to 20 independent components.
///
/// Returns an index into the Riemann20 array.
/// Convention: we store the 6×6 upper triangle row by row (21 entries),
/// then the Bianchi identity eliminates one, leaving 20.
///
/// For simplicity, we use all 21 and project later.
fn symmetric_index(p: usize, q: usize) -> usize {
    let (lo, hi) = if p <= q { (p, q) } else { (q, p) };
    lo * 6 - lo * (lo + 1) / 2 + hi
}

/// Compute the bivector of a triangle given vertex coordinates.
///
/// The bivector B^{ab} = (v1 - v0)^a ∧ (v2 - v0)^b is a 4×4 antisymmetric tensor.
/// We return the 6 independent components in the canonical order.
fn triangle_bivector_from_coords(
    v0: [f64; 4],
    v1: [f64; 4],
    v2: [f64; 4],
) -> [f64; 6] {
    let e1: [f64; 4] = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2], v1[3] - v0[3]];
    let e2: [f64; 4] = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2], v2[3] - v0[3]];

    // B^{ab} = e1^a e2^b - e1^b e2^a
    let mut bv = [0.0; 6];
    let mut idx = 0;
    for a in 0..4 {
        for b in (a + 1)..4 {
            bv[idx] = e1[a] * e2[b] - e1[b] * e2[a];
            idx += 1;
        }
    }
    bv
}

/// Extract the approximate Riemann tensor at a vertex by least-squares
/// fitting deficit angles to bivector projections.
///
/// For each triangle t near the vertex:
///   δ_t / A_t ≈ R_{abcd} n^{ab} n^{cd} = Σ_{pq} R_pq · B_p · B_q
///
/// This gives a linear system: A · R = b, solved via SVD.
pub fn extract_riemann_at_vertex(
    complex: &SimplicialComplex,
    fc: &FoliatedComplex,
    sq_lengths: &[f64],
    deficits: &[(f64, crate::lorentzian::HingeType)],
    vertex: usize,
) -> Riemann20 {
    let tris = &complex.vertex_to_tris[vertex];
    let n_tris = tris.len();
    let n_t = fc.n_slices;
    let n_s = fc.n_spatial;

    if n_tris == 0 {
        return [0.0; 20];
    }

    // Build the linear system: each triangle gives one equation
    // δ_t / A_t = Σ_{p<=q} (1 + δ_{pq}) · R_pq · B_p(t) · B_q(t)
    let n_comps = 21; // 6×6 upper triangle
    let mut a_mat = vec![0.0; n_tris * n_comps];
    let mut b_vec = vec![0.0; n_tris];

    for (row, &ti) in tris.iter().enumerate() {
        let (deficit, _ht) = deficits[ti];

        // Compute area
        let t = &complex.triangles[ti];
        let a_sq = tri_area_sq(complex, ti, sq_lengths);
        let abs_a = a_sq.abs().sqrt();
        if abs_a < 1e-30 {
            continue;
        }

        b_vec[row] = deficit / abs_a;

        // Get vertex coordinates
        let coords: Vec<[f64; 4]> = t
            .iter()
            .map(|&v| vertex_coords(v, n_t, n_s, fc))
            .collect();
        let bv = triangle_bivector_from_coords(coords[0], coords[1], coords[2]);

        // Fill the row of A: for each pair (p,q) with p <= q
        for p in 0..6 {
            for q in p..6 {
                let idx = symmetric_index(p, q);
                let weight = if p == q { 1.0 } else { 2.0 };
                a_mat[row * n_comps + idx] = weight * bv[p] * bv[q];
            }
        }
    }

    // Solve via SVD (least-squares)
    let a_nalg = nalgebra::DMatrix::from_row_slice(n_tris, n_comps, &a_mat);
    let b_nalg = nalgebra::DVector::from_vec(b_vec);

    let svd = a_nalg.svd(true, true);
    let solution = svd.solve(&b_nalg, 1e-10).unwrap_or_else(|_| {
        nalgebra::DVector::zeros(n_comps)
    });

    // Pack into Riemann20 (first 20 of the 21 symmetric components)
    let mut riemann = [0.0; 20];
    for i in 0..20.min(n_comps) {
        riemann[i] = solution[i];
    }
    riemann
}

/// Extract GEM tidal tensors from the 20 Riemann components.
///
/// E_g^{ij} = R^{0i0j} (gravitoelectric)
/// B_g^{ij} = (1/2) ε^{ikl} R_{0jkl} (gravitomagnetic)
pub fn riemann_to_gem(riemann: &Riemann20) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
    // The Riemann components are stored in the 6×6 symmetric form.
    // Index mapping: bivector (a,b) → index p
    // (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=4, (2,3)=5
    //
    // R_{p,q} = R_{(a,b),(c,d)} = R_{abcd}
    //
    // E_g^{ij} = R^{0i0j} = R_{(0,i),(0,j)} for i,j ∈ {1,2,3}
    // In bivector indexing: (0,i) corresponds to i-1 (i.e., i=1→0, i=2→1, i=3→2)

    let r = |p: usize, q: usize| -> f64 {
        let idx = symmetric_index(p, q);
        if idx < 20 {
            riemann[idx]
        } else {
            // The 21st component is constrained by Bianchi identity
            0.0
        }
    };

    let mut e_grav = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            // R_{0(i+1)0(j+1)} = R_{p,q} where p = bivector_index(0, i+1), q = bivector_index(0, j+1)
            e_grav[i][j] = r(i, j); // bivector indices 0,1,2 correspond to (0,1),(0,2),(0,3)
        }
    }

    // B_g^{ij} = (1/2) ε^{ikl} R_{0j,kl}
    // In bivector indexing: R_{0j,kl} = R_{p,q} where p = j-1 (for (0,j)), q = bivector_index(k,l)
    let eps = |i: usize, j: usize, k: usize| -> f64 {
        if (i, j, k) == (0, 1, 2) || (i, j, k) == (1, 2, 0) || (i, j, k) == (2, 0, 1) {
            1.0
        } else if (i, j, k) == (0, 2, 1) || (i, j, k) == (2, 1, 0) || (i, j, k) == (1, 0, 2) {
            -1.0
        } else {
            0.0
        }
    };

    let mut b_grav = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                for l in 0..3 {
                    if k >= l {
                        continue;
                    }
                    let e = eps(i, k, l);
                    if e.abs() < 0.5 {
                        continue;
                    }
                    // R_{0(j+1),(k+1)(l+1)}
                    let p = j; // bivector (0, j+1) → index j
                    let q = bivector_index(k + 1, l + 1);
                    sum += e * r(p, q);
                }
            }
            b_grav[i][j] = 0.5 * sum;
        }
    }

    (e_grav, b_grav)
}

/// Extract GEM fields at all vertices of a foliated complex.
pub fn extract_gem_fields(
    complex: &SimplicialComplex,
    fc: &FoliatedComplex,
    sq_lengths: &[f64],
) -> GemFields {
    let deficits = lorentzian_deficit_angles(complex, sq_lengths);
    let n_v = complex.n_vertices;

    let mut e_grav = vec![[[0.0; 3]; 3]; n_v];
    let mut b_grav = vec![[[0.0; 3]; 3]; n_v];

    for v in 0..n_v {
        let riemann = extract_riemann_at_vertex(complex, fc, sq_lengths, &deficits, v);
        let (eg, bg) = riemann_to_gem(&riemann);
        e_grav[v] = eg;
        b_grav[v] = bg;
    }

    GemFields { e_grav, b_grav }
}

/// Compute the gravitomagnetic flux through a surface defined by triangles.
///
/// Φ_B = Σ_t B_g · n_t · A_t
pub fn gravitomagnetic_flux(
    gem: &GemFields,
    surface_tris: &[(usize, [f64; 3])],
) -> f64 {
    let mut flux = 0.0;
    for &(vi, normal) in surface_tris {
        // B_g · n gives the gravitomagnetic flux density
        let bg = &gem.b_grav[vi];
        for i in 0..3 {
            for j in 0..3 {
                flux += bg[i][j] * normal[i] * normal[j];
            }
        }
    }
    flux
}

/// Compute the induced GEM EMF around a loop.
///
/// EMF = -d(Φ_B)/dt ≈ -(Φ_B(t+dt) - Φ_B(t)) / dt
pub fn induced_gem_emf(
    gem_before: &GemFields,
    gem_after: &GemFields,
    loop_vertices: &[usize],
    dt: f64,
) -> f64 {
    // Compute average B_g over loop vertices at both times
    let avg_bg = |gem: &GemFields| -> [[f64; 3]; 3] {
        let mut avg = [[0.0; 3]; 3];
        for &v in loop_vertices {
            for i in 0..3 {
                for j in 0..3 {
                    avg[i][j] += gem.b_grav[v][i][j];
                }
            }
        }
        let n = loop_vertices.len() as f64;
        for row in &mut avg {
            for val in row {
                *val /= n;
            }
        }
        avg
    };

    let bg_before = avg_bg(gem_before);
    let bg_after = avg_bg(gem_after);

    // Trace of ΔB gives scalar measure
    let mut d_trace = 0.0;
    for i in 0..3 {
        d_trace += bg_after[i][i] - bg_before[i][i];
    }

    -d_trace / dt
}

/// Linearized gravitomagnetic field from a mass current loop.
///
/// In the weak-field (linearized GEM) regime, the gravitomagnetic field is
/// given by the Biot-Savart-like law:
///
///   B_g(r) = -4G/c² ∫ (J × r̂) / r² dl
///
/// In geometric units (G = c = 1):
///
///   B_g(r) = -4 ∫ (J_dl × r̂) / |r|² dl
///
/// where J_dl = mass_rate * dl_hat is the mass current element.
///
/// # Arguments
/// * `loop_coords` - 3D spatial coordinates of the current loop vertices
/// * `mass_rate` - Mass flow rate (dm/dt in geometric units)
/// * `field_point` - 3D spatial coordinates where B_g is evaluated
///
/// # Returns
/// The gravitomagnetic field vector [B_x, B_y, B_z] at `field_point`.
pub fn linearized_b_grav(
    loop_coords: &[[f64; 3]],
    mass_rate: f64,
    field_point: [f64; 3],
) -> [f64; 3] {
    let mut b = [0.0; 3];
    let n = loop_coords.len();
    if n < 2 {
        return b;
    }

    for i in 0..n {
        let p0 = loop_coords[i];
        let p1 = loop_coords[(i + 1) % n];

        // Current element: dl = p1 - p0, with mass_rate along it
        let dl = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];

        // Midpoint of the element
        let mid = [
            0.5 * (p0[0] + p1[0]),
            0.5 * (p0[1] + p1[1]),
            0.5 * (p0[2] + p1[2]),
        ];

        // r = field_point - midpoint
        let r = [
            field_point[0] - mid[0],
            field_point[1] - mid[1],
            field_point[2] - mid[2],
        ];
        let r_mag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        if r_mag < 1e-30 {
            continue;
        }

        // J_dl × r̂ / r² = mass_rate * (dl × r) / r³
        let cross = [
            dl[1] * r[2] - dl[2] * r[1],
            dl[2] * r[0] - dl[0] * r[2],
            dl[0] * r[1] - dl[1] * r[0],
        ];

        let factor = -4.0 * mass_rate / (r_mag * r_mag * r_mag);
        b[0] += factor * cross[0];
        b[1] += factor * cross[1];
        b[2] += factor * cross[2];
    }

    b
}

/// Compute the linearized GEM prediction for B_g at a set of field points,
/// given a primary current loop on a foliated lattice.
///
/// Returns B_g vectors at each field point.
pub fn linearized_gem_prediction(
    loop_coords: &[[f64; 3]],
    mass_rate: f64,
    field_points: &[[f64; 3]],
) -> Vec<[f64; 3]> {
    field_points
        .iter()
        .map(|&fp| linearized_b_grav(loop_coords, mass_rate, fp))
        .collect()
}

/// Compare Regge-extracted B_g to the linearized GEM prediction.
///
/// Returns (ratio, regge_magnitude, linearized_magnitude) for each field point.
/// Ratio = |B_g_regge| / |B_g_linearized|.
/// At weak fields this should be ~1.0; deviations indicate nonlinear GR corrections.
pub fn gem_comparison(
    regge_b_grav: &[[[f64; 3]; 3]],
    field_vertices: &[usize],
    loop_coords: &[[f64; 3]],
    mass_rate: f64,
    field_coords: &[[f64; 3]],
) -> Vec<GemComparisonPoint> {
    field_vertices
        .iter()
        .zip(field_coords.iter())
        .map(|(&vi, &fp)| {
            // Regge B_g magnitude: Frobenius norm of the 3×3 tensor
            let bg = &regge_b_grav[vi];
            let regge_mag: f64 = bg
                .iter()
                .flat_map(|row| row.iter())
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();

            // Linearized prediction magnitude
            let b_lin = linearized_b_grav(loop_coords, mass_rate, fp);
            let lin_mag = (b_lin[0] * b_lin[0] + b_lin[1] * b_lin[1] + b_lin[2] * b_lin[2]).sqrt();

            let ratio = if lin_mag > 1e-30 {
                regge_mag / lin_mag
            } else {
                f64::NAN
            };

            GemComparisonPoint {
                ratio,
                regge_magnitude: regge_mag,
                linearized_magnitude: lin_mag,
            }
        })
        .collect()
}

/// One point in the Regge vs linearized GEM comparison.
#[derive(Debug, Clone)]
pub struct GemComparisonPoint {
    /// Ratio: |B_g_regge| / |B_g_linearized|. ~1.0 in weak field.
    pub ratio: f64,
    /// Regge-extracted B_g magnitude (Frobenius norm of tensor).
    pub regge_magnitude: f64,
    /// Linearized (Biot-Savart) B_g magnitude.
    pub linearized_magnitude: f64,
}

/// Get 3D spatial coordinates for a vertex on the lattice.
///
/// The coordinate convention follows `vertex_coords` but returns only
/// the spatial part (x, y, z) scaled by the spacing.
pub fn vertex_spatial_coords(
    v: usize,
    fc: &FoliatedComplex,
    spacing: f64,
) -> [f64; 3] {
    let local = fc.vertex_local(v);
    let n = fc.n_spatial;
    let x = (local % n) as f64 * spacing;
    let y = ((local / n) % n) as f64 * spacing;
    let z = (local / (n * n)) as f64 * spacing;
    [x, y, z]
}

// --- Helper functions ---

fn tri_area_sq(complex: &SimplicialComplex, ti: usize, sq_lengths: &[f64]) -> f64 {
    let t = &complex.triangles[ti];
    let sorted2 = |a: usize, b: usize| -> [usize; 2] {
        if a < b { [a, b] } else { [b, a] }
    };
    let e01 = complex.edge_index[&sorted2(t[0], t[1])];
    let e02 = complex.edge_index[&sorted2(t[0], t[2])];
    let e12 = complex.edge_index[&sorted2(t[1], t[2])];
    crate::lorentzian::triangle_area_sq_lorentzian(sq_lengths[e01], sq_lengths[e02], sq_lengths[e12])
}

fn vertex_coords(v: usize, n_t: usize, n_s: usize, _fc: &FoliatedComplex) -> [f64; 4] {
    let t = (v % n_t) as f64;
    let rest = v / n_t;
    let x = (rest % n_s) as f64;
    let y = ((rest / n_s) % n_s) as f64;
    let z = (rest / (n_s * n_s)) as f64;
    [t, x, y, z]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foliation::{flat_minkowski_sq_lengths, foliated_hypercubic};

    /// Flat Minkowski space should give zero GEM fields.
    #[test]
    fn test_flat_space_zero_gem() {
        let fc = foliated_hypercubic(2, 2);
        let sq_lengths = flat_minkowski_sq_lengths(&fc, 1.0, 0.3);

        let gem = extract_gem_fields(&fc.complex, &fc, &sq_lengths);

        let max_e: f64 = gem
            .e_grav
            .iter()
            .flat_map(|m| m.iter().flat_map(|r| r.iter()))
            .map(|x| x.abs())
            .fold(0.0, f64::max);
        let max_b: f64 = gem
            .b_grav
            .iter()
            .flat_map(|m| m.iter().flat_map(|r| r.iter()))
            .map(|x| x.abs())
            .fold(0.0, f64::max);

        assert!(
            max_e < 1e-6,
            "flat space E_g should be ~0, got max {max_e:.2e}"
        );
        assert!(
            max_b < 1e-6,
            "flat space B_g should be ~0, got max {max_b:.2e}"
        );
    }

    /// Riemann-to-GEM conversion is consistent.
    #[test]
    fn test_riemann_to_gem_zero() {
        let riemann = [0.0; 20];
        let (e_grav, b_grav) = riemann_to_gem(&riemann);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(e_grav[i][j], 0.0);
                assert_eq!(b_grav[i][j], 0.0);
            }
        }
    }

    /// Bivector computation for a flat triangle.
    #[test]
    fn test_triangle_bivector() {
        let v0 = [0.0, 0.0, 0.0, 0.0];
        let v1 = [0.0, 1.0, 0.0, 0.0];
        let v2 = [0.0, 0.0, 1.0, 0.0];

        let bv = triangle_bivector_from_coords(v0, v1, v2);
        // B^{12} = e1^1 * e2^2 - e1^2 * e2^1 = 1*1 - 0*0 = 1
        // bivector_index(1,2) = 3
        assert!((bv[3] - 1.0).abs() < 1e-12, "B^{{12}} = {}", bv[3]);
        // All other components should be zero
        assert!(bv[0].abs() < 1e-12);
        assert!(bv[1].abs() < 1e-12);
        assert!(bv[2].abs() < 1e-12);
        assert!(bv[4].abs() < 1e-12);
        assert!(bv[5].abs() < 1e-12);
    }
}
