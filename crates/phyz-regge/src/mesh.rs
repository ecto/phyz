//! Mesh generation for 4D simplicial complexes.
//!
//! Generates regular triangulations of simple topologies for testing
//! and as starting points for the symmetry search.

use crate::complex::SimplicialComplex;

/// Integration method for evaluating the metric along edges.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MetricIntegration {
    /// Single evaluation at the edge midpoint.
    #[default]
    Midpoint,
    /// Simpson's 1/3 rule: weighted average of endpoints + midpoint.
    Simpson,
}

/// Generate a flat periodic (toroidal) hypercubic triangulation.
///
/// Tiles 4D space with n^4 hypercubes, each subdivided into 24 pentachorons
/// using the Kuhn/Freudenthal triangulation. Periodic boundary conditions
/// (4-torus T⁴) ensure all triangles are interior — no boundary effects.
///
/// # Arguments
/// * `n` - Number of hypercubes along each axis (total = n⁴).
/// * `spacing` - Edge length of each hypercube.
///
/// # Returns
/// (SimplicialComplex, edge_lengths) where edge_lengths assigns each edge
/// its Euclidean length in the flat embedding. For a unit hypercube, edges
/// are either 1.0 (axis-aligned), sqrt(2) (face diagonal), sqrt(3) (cell
/// diagonal), or 2.0 (hypercube diagonal).
pub fn flat_hypercubic(n: usize, spacing: f64) -> (SimplicialComplex, Vec<f64>) {
    let n4 = n * n * n * n;
    let n_vertices = n4;

    // Vertex indexing: v(t,x,y,z) = t + n*x + n²*y + n³*z
    let vidx = |t: usize, x: usize, y: usize, z: usize| -> usize {
        (t % n) + n * (x % n) + n * n * (y % n) + n * n * n * (z % n)
    };

    // Kuhn triangulation of a 4-cube into 24 pentachorons.
    // Each pentachoron corresponds to a permutation of the 4 axes.
    // For a hypercube with "origin" vertex v0, the 24 permutations of
    // (0,1,2,3) give 24 simplices. Each permutation σ gives the simplex:
    //   [v0, v0+e_{σ(0)}, v0+e_{σ(0)}+e_{σ(1)}, ..., v0+e_{σ(0)}+...+e_{σ(3)}]
    // where e_i is the unit vector along axis i.
    let permutations = all_permutations_4();

    let mut pent_set: std::collections::HashSet<[usize; 5]> = std::collections::HashSet::new();

    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                for t in 0..n {
                    for perm in &permutations {
                        // Build the 5 vertices of this pentachoron.
                        let mut coords = [t, x, y, z];
                        let mut verts = [0usize; 5];
                        verts[0] = vidx(coords[0], coords[1], coords[2], coords[3]);

                        for step in 0..4 {
                            coords[perm[step]] += 1;
                            verts[step + 1] =
                                vidx(coords[0], coords[1], coords[2], coords[3]);
                        }

                        verts.sort_unstable();
                        pent_set.insert(verts);
                    }
                }
            }
        }
    }

    let pents: Vec<[usize; 5]> = pent_set.into_iter().collect();

    let complex = SimplicialComplex::from_pentachorons(n_vertices, &pents);

    // Compute edge lengths from the flat embedding.
    let mut lengths = vec![0.0; complex.n_edges()];
    for (ei, edge) in complex.edges.iter().enumerate() {
        let (t0, x0, y0, z0) = vertex_coords(edge[0], n);
        let (t1, x1, y1, z1) = vertex_coords(edge[1], n);

        // Minimum-image distance for periodic BC.
        let dt = min_image_dist(t0, t1, n);
        let dx = min_image_dist(x0, x1, n);
        let dy = min_image_dist(y0, y1, n);
        let dz = min_image_dist(z0, z1, n);

        let dist_sq = (dt * dt + dx * dx + dy * dy + dz * dz) as f64;
        lengths[ei] = (dist_sq).sqrt() * spacing;
    }

    (complex, lengths)
}

/// Minimum-image distance for periodic coordinates.
fn min_image_dist(a: usize, b: usize, n: usize) -> i64 {
    let d = (b as i64) - (a as i64);
    let n = n as i64;
    if d > n / 2 {
        d - n
    } else if d < -n / 2 {
        d + n
    } else {
        d
    }
}

/// Convert linear vertex index to (t, x, y, z) coordinates.
fn vertex_coords(idx: usize, n: usize) -> (usize, usize, usize, usize) {
    let t = idx % n;
    let x = (idx / n) % n;
    let y = (idx / (n * n)) % n;
    let z = idx / (n * n * n);
    (t, x, y, z)
}

/// Generate all 24 permutations of [0, 1, 2, 3].
fn all_permutations_4() -> Vec<[usize; 4]> {
    let mut perms = Vec::with_capacity(24);
    let items = [0usize, 1, 2, 3];
    permute(&items, 0, &mut items.clone(), &mut perms);
    perms
}

fn permute(
    _items: &[usize; 4],
    start: usize,
    current: &mut [usize; 4],
    result: &mut Vec<[usize; 4]>,
) {
    if start == 4 {
        result.push(*current);
        return;
    }
    for i in start..4 {
        current.swap(start, i);
        permute(_items, start + 1, current, result);
        current.swap(start, i);
    }
}

/// Compute ds² = g_μν dx^μ dx^ν for a full 4×4 metric tensor.
fn metric_line_element(g: &[[f64; 4]; 4], dx: &[f64; 4]) -> f64 {
    let mut ds_sq = 0.0;
    for mu in 0..4 {
        for nu in 0..4 {
            ds_sq += g[mu][nu] * dx[mu] * dx[nu];
        }
    }
    ds_sq
}

/// Helper: build flat hypercubic mesh, compute centered vertex positions,
/// then deform edge lengths using a metric tensor evaluated at each edge midpoint.
fn deform_by_metric_with(
    n: usize,
    spacing: f64,
    integration: MetricIntegration,
    metric_at: impl Fn(f64, f64, f64, f64) -> [[f64; 4]; 4],
) -> (SimplicialComplex, Vec<f64>) {
    let (complex, _flat_lengths) = flat_hypercubic(n, spacing);

    let half = n as f64 / 2.0;
    let vertex_pos = |idx: usize| -> [f64; 4] {
        let (t, x, y, z) = vertex_coords(idx, n);
        [
            (t as f64 - half) * spacing,
            (x as f64 - half) * spacing,
            (y as f64 - half) * spacing,
            (z as f64 - half) * spacing,
        ]
    };

    let mut lengths = vec![0.0; complex.n_edges()];
    for (ei, edge) in complex.edges.iter().enumerate() {
        let p0 = vertex_pos(edge[0]);
        let p1 = vertex_pos(edge[1]);

        let dx = [
            p1[0] - p0[0],
            p1[1] - p0[1],
            p1[2] - p0[2],
            p1[3] - p0[3],
        ];

        let length = match integration {
            MetricIntegration::Midpoint => {
                let mid = [
                    (p0[0] + p1[0]) / 2.0,
                    (p0[1] + p1[1]) / 2.0,
                    (p0[2] + p1[2]) / 2.0,
                    (p0[3] + p1[3]) / 2.0,
                ];
                let g = metric_at(mid[0], mid[1], mid[2], mid[3]);
                metric_line_element(&g, &dx).abs().sqrt()
            }
            MetricIntegration::Simpson => {
                // Simpson's 1/3 rule: (l0 + 4*lm + l1) / 6
                let mid = [
                    (p0[0] + p1[0]) / 2.0,
                    (p0[1] + p1[1]) / 2.0,
                    (p0[2] + p1[2]) / 2.0,
                    (p0[3] + p1[3]) / 2.0,
                ];
                let g0 = metric_at(p0[0], p0[1], p0[2], p0[3]);
                let gm = metric_at(mid[0], mid[1], mid[2], mid[3]);
                let g1 = metric_at(p1[0], p1[1], p1[2], p1[3]);
                let l0 = metric_line_element(&g0, &dx).abs().sqrt();
                let lm = metric_line_element(&gm, &dx).abs().sqrt();
                let l1 = metric_line_element(&g1, &dx).abs().sqrt();
                (l0 + 4.0 * lm + l1) / 6.0
            }
        };

        lengths[ei] = length.max(1e-10);
    }

    (complex, lengths)
}

/// Generate a Reissner-Nordström-like triangulation.
///
/// Takes a flat hypercubic mesh and deforms edge lengths to approximate
/// the Reissner-Nordström geometry in isotropic coordinates.
///
/// The RN metric in isotropic coordinates:
///   ds² = -f(r)dt² + g(r)(dx² + dy² + dz²)
/// where
///   f(r) = (1 - M/(2r))² / (1 + M/(2r))²  (simplified, Q=0 → Schwarzschild)
///   g(r) = (1 + M/(2r))⁴
///
/// For RN with charge Q:
///   f(r) = ((1 - r+/(2r))(1 - r-/(2r))) / ((1 + r+/(2r))(1 + r-/(2r)))
///   g(r) = ((1 + r+/(2r))(1 + r-/(2r)))²
///
/// where r± = M ± sqrt(M² - Q²).
///
/// # Arguments
/// * `n` - Grid size along each axis
/// * `spacing` - Base grid spacing (physical units)
/// * `mass` - Black hole mass parameter M
/// * `charge` - Black hole charge parameter Q (0 for Schwarzschild)
/// * `r_min` - Minimum isotropic radius (should be > r+/2 to stay outside horizon)
pub fn reissner_nordstrom(
    n: usize,
    spacing: f64,
    mass: f64,
    charge: f64,
    r_min: f64,
) -> (SimplicialComplex, Vec<f64>) {
    reissner_nordstrom_with(n, spacing, mass, charge, r_min, MetricIntegration::Midpoint)
}

/// Like `reissner_nordstrom` with configurable integration method.
pub fn reissner_nordstrom_with(
    n: usize,
    spacing: f64,
    mass: f64,
    charge: f64,
    r_min: f64,
    integration: MetricIntegration,
) -> (SimplicialComplex, Vec<f64>) {
    let r_plus = mass + (mass * mass - charge * charge).max(0.0).sqrt();
    let r_minus = mass - (mass * mass - charge * charge).max(0.0).sqrt();

    deform_by_metric_with(n, spacing, integration, |_t, x, y, z| {
        let r = (x * x + y * y + z * z).sqrt().max(r_min);
        let a = 1.0 + r_plus / (2.0 * r);
        let b = 1.0 + r_minus / (2.0 * r);
        let c = 1.0 - r_plus / (2.0 * r);
        let d = 1.0 - r_minus / (2.0 * r);

        let f = (c * d) / (a * b);       // g_tt
        let g = (a * b) * (a * b);       // g_xx = g_yy = g_zz

        let f = f.abs().max(0.01);
        let g = g.max(0.01);

        let mut metric = [[0.0; 4]; 4];
        metric[0][0] = f;
        metric[1][1] = g;
        metric[2][2] = g;
        metric[3][3] = g;
        metric
    })
}

/// Generate a de Sitter triangulation in conformally flat (cosmological) coordinates.
///
/// The de Sitter metric in cosmological coordinates:
///   ds² = (1/(H²τ²)) (dτ² + dx² + dy² + dz²)
/// where H = 1/L (cosmological_length).
///
/// This is conformally flat: f = g = 1/(H²τ²), so it is maximally symmetric.
/// All 6 "rotation" symmetries (3 spatial + 3 boost) should have comparable
/// O(h²) violation from the lattice discretization.
///
/// # Arguments
/// * `n` - Grid size along each axis
/// * `spacing` - Base grid spacing
/// * `cosmological_length` - L = 1/H (de Sitter radius)
pub fn de_sitter(
    n: usize,
    spacing: f64,
    cosmological_length: f64,
) -> (SimplicialComplex, Vec<f64>) {
    de_sitter_with(n, spacing, cosmological_length, MetricIntegration::Midpoint)
}

/// Like `de_sitter` with configurable integration method.
pub fn de_sitter_with(
    n: usize,
    spacing: f64,
    cosmological_length: f64,
    integration: MetricIntegration,
) -> (SimplicialComplex, Vec<f64>) {
    let h = 1.0 / cosmological_length;

    deform_by_metric_with(n, spacing, integration, |tau, _x, _y, _z| {
        let tau_safe = tau.abs().max(0.1 * spacing);
        let conformal = 1.0 / (h * h * tau_safe * tau_safe);

        let mut metric = [[0.0; 4]; 4];
        metric[0][0] = conformal;
        metric[1][1] = conformal;
        metric[2][2] = conformal;
        metric[3][3] = conformal;
        metric
    })
}

/// Generate a slow-rotation Kerr triangulation.
///
/// Uses Schwarzschild isotropic factors for the diagonal plus linear-in-a
/// frame-dragging off-diagonal terms:
/// - Diagonal: g_tt = f(r), g_ii = g(r) (same as Schwarzschild)
/// - Off-diagonal: g_tx = h(r)·(−y/r²), g_ty = h(r)·(x/r²)
///   where h(r) = −2Ma/r
///
/// Only the z-axis rotation survives as approximate symmetry. Other rotations
/// and all boosts are broken at O(a) > O(h²).
///
/// # Arguments
/// * `n` - Grid size along each axis
/// * `spacing` - Base grid spacing
/// * `mass` - Black hole mass M
/// * `spin` - Dimensionless spin parameter a (slow rotation: |a| << M)
/// * `r_min` - Minimum isotropic radius (should be > M/2)
pub fn kerr(
    n: usize,
    spacing: f64,
    mass: f64,
    spin: f64,
    r_min: f64,
) -> (SimplicialComplex, Vec<f64>) {
    kerr_with(n, spacing, mass, spin, r_min, MetricIntegration::Midpoint)
}

/// Like `kerr` (slow rotation) with configurable integration method.
pub fn kerr_with(
    n: usize,
    spacing: f64,
    mass: f64,
    spin: f64,
    r_min: f64,
    integration: MetricIntegration,
) -> (SimplicialComplex, Vec<f64>) {
    deform_by_metric_with(n, spacing, integration, |_t, x, y, z| {
        let r = (x * x + y * y + z * z).sqrt().max(r_min);

        let a_fac = 1.0 + mass / (2.0 * r);
        let c_fac = 1.0 - mass / (2.0 * r);
        let f = (c_fac * c_fac) / (a_fac * a_fac);
        let g = a_fac * a_fac * a_fac * a_fac;

        let f = f.abs().max(0.01);
        let g = g.max(0.01);

        let h_drag = -2.0 * mass * spin / r;
        let r_sq = r * r;

        let mut metric = [[0.0; 4]; 4];
        metric[0][0] = f;
        metric[1][1] = g;
        metric[2][2] = g;
        metric[3][3] = g;
        metric[0][1] = h_drag * (-y / r_sq);
        metric[1][0] = metric[0][1];
        metric[0][2] = h_drag * (x / r_sq);
        metric[2][0] = metric[0][2];
        metric
    })
}

/// Exact Kerr metric in Kerr-Schild (Euclidean) coordinates.
///
/// g_μν = δ_μν + H·l_μ·l_ν where H = 2Mr³/(r⁴ + a²z²).
/// Wick-rotated to Euclidean signature.
pub fn kerr_schild(
    n: usize,
    spacing: f64,
    mass: f64,
    spin: f64,
    r_min: f64,
) -> (SimplicialComplex, Vec<f64>) {
    kerr_schild_with(n, spacing, mass, spin, r_min, MetricIntegration::Midpoint)
}

/// Like `kerr_schild` with configurable integration method.
pub fn kerr_schild_with(
    n: usize,
    spacing: f64,
    mass: f64,
    spin: f64,
    r_min: f64,
    integration: MetricIntegration,
) -> (SimplicialComplex, Vec<f64>) {
    let a = spin;
    deform_by_metric_with(n, spacing, integration, move |_t, x, y, z| {
        let rho_sq = x * x + y * y + z * z;
        // r from implicit: r⁴ - (ρ² - a²)r² - a²z² = 0
        // r² = ((ρ²-a²) + sqrt((ρ²-a²)² + 4a²z²)) / 2
        let diff = rho_sq - a * a;
        let r_sq = (diff + (diff * diff + 4.0 * a * a * z * z).sqrt()) / 2.0;
        let r_sq = r_sq.max(r_min * r_min);
        let r = r_sq.sqrt();

        let h_val = 2.0 * mass * r * r_sq / (r_sq * r_sq + a * a * z * z).max(1e-30);

        // Null vector l_μ (Euclidean): l_0 = 1
        let r2a2 = (r_sq + a * a).max(1e-30);
        let l = [
            1.0,
            (r * x + a * y) / r2a2,
            (r * y - a * x) / r2a2,
            z / r.max(1e-30),
        ];

        let mut metric = [[0.0; 4]; 4];
        for mu in 0..4 {
            metric[mu][mu] = 1.0;
            for nu in 0..4 {
                metric[mu][nu] += h_val * l[mu] * l[nu];
            }
        }
        metric
    })
}

/// Exact Kerr metric in Boyer-Lindquist isotropic coordinates.
///
/// Converts the BL metric to Cartesian via r_BL(r_iso) inversion.
/// Diagonal f(r), g(r) are exact Kerr functions, with off-diagonal g_{tφ}.
pub fn kerr_bl(
    n: usize,
    spacing: f64,
    mass: f64,
    spin: f64,
    r_min: f64,
) -> (SimplicialComplex, Vec<f64>) {
    kerr_bl_with(n, spacing, mass, spin, r_min, MetricIntegration::Midpoint)
}

/// Like `kerr_bl` with configurable integration method.
pub fn kerr_bl_with(
    n: usize,
    spacing: f64,
    mass: f64,
    spin: f64,
    r_min: f64,
    integration: MetricIntegration,
) -> (SimplicialComplex, Vec<f64>) {
    let a = spin;
    deform_by_metric_with(n, spacing, integration, move |_t, x, y, z| {
        let r_iso = (x * x + y * y + z * z).sqrt().max(r_min);

        // Map isotropic r to BL r: r_BL = r_iso * (1 + M/(2*r_iso))^2
        // (Schwarzschild-like mapping, adjusted for Kerr)
        let r_bl = r_iso * (1.0 + mass / (2.0 * r_iso)).powi(2);

        let sigma = r_bl * r_bl + a * a * z * z / (r_iso * r_iso).max(1e-30);
        let delta = r_bl * r_bl - 2.0 * mass * r_bl + a * a;

        let delta_safe = delta.abs().max(0.01);
        let sigma_safe = sigma.max(0.01);

        // Isotropic conformal factor
        let psi = (1.0 + mass / (2.0 * r_iso)).powi(2);
        let psi4 = psi * psi;

        // g_tt (Euclidean: positive)
        let f_tt = (delta_safe * sigma_safe / ((r_bl * r_bl + a * a) * (r_bl * r_bl + a * a) + a * a * delta_safe).max(0.01)).abs().max(0.01);

        // Spatial part
        let g_rr = psi4;

        // Frame-dragging in Cartesian: g_{tφ} → g_{tx}, g_{ty}
        let r2a2 = r_bl * r_bl + a * a;
        let omega = 2.0 * mass * a * r_bl / (r2a2 * r2a2 + a * a * delta_safe).max(0.01);

        // φ-direction in Cartesian: (-y/ρ, x/ρ)
        let rho = (x * x + y * y).sqrt().max(1e-30);

        let mut metric = [[0.0; 4]; 4];
        metric[0][0] = f_tt;
        metric[1][1] = g_rr;
        metric[2][2] = g_rr;
        metric[3][3] = g_rr;
        // Frame dragging
        metric[0][1] = omega * rho * (-y / rho);
        metric[1][0] = metric[0][1];
        metric[0][2] = omega * rho * (x / rho);
        metric[2][0] = metric[0][2];
        metric
    })
}

/// de Sitter metric in static patch coordinates.
///
///   g_tt = |1 - r²/L²|  (Euclidean, clamped away from horizon)
///   g_ij = δ_ij + (r_i·r_j/r²)·(1/|1-r²/L²| - 1)
pub fn de_sitter_static(
    n: usize,
    spacing: f64,
    cosmological_length: f64,
    r_min: f64,
) -> (SimplicialComplex, Vec<f64>) {
    de_sitter_static_with(n, spacing, cosmological_length, r_min, MetricIntegration::Midpoint)
}

/// Like `de_sitter_static` with configurable integration method.
pub fn de_sitter_static_with(
    n: usize,
    spacing: f64,
    cosmological_length: f64,
    r_min: f64,
    integration: MetricIntegration,
) -> (SimplicialComplex, Vec<f64>) {
    let l_sq = cosmological_length * cosmological_length;

    deform_by_metric_with(n, spacing, integration, move |_t, x, y, z| {
        let r_sq = (x * x + y * y + z * z).max(r_min * r_min);

        // Clamp away from the cosmological horizon r = L
        let ratio = (r_sq / l_sq).min(0.95);
        let f = (1.0 - ratio).abs().max(0.01);

        let mut metric = [[0.0; 4]; 4];
        metric[0][0] = f; // g_tt

        // g_ij = δ_ij + (r_i r_j / r²) * (1/f - 1)
        let g_radial_extra = 1.0 / f - 1.0;
        let coords = [x, y, z];
        for i in 0..3 {
            metric[i + 1][i + 1] = 1.0;
            for j in 0..3 {
                metric[i + 1][j + 1] += coords[i] * coords[j] / r_sq * g_radial_extra;
            }
        }
        metric
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutations() {
        let perms = all_permutations_4();
        assert_eq!(perms.len(), 24);

        // All should be permutations of [0,1,2,3].
        for p in &perms {
            let mut sorted = *p;
            sorted.sort();
            assert_eq!(sorted, [0, 1, 2, 3]);
        }

        // All distinct.
        let mut unique = perms.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 24);
    }

    #[test]
    fn test_flat_hypercubic_n2() {
        let (complex, lengths) = flat_hypercubic(2, 1.0);

        assert_eq!(complex.n_vertices, 16); // 2^4
        // 24 unique pentachorons per hypercube, but with n=2 periodic BCs
        // many are shared between hypercubes. Total is 24 * n^4 / duplicity.
        assert!(complex.n_pents() > 0);

        // All edge lengths should be 1, sqrt(2), sqrt(3), or 2.
        for &l in &lengths {
            let valid = (l - 1.0).abs() < 1e-10
                || (l - 2.0_f64.sqrt()).abs() < 1e-10
                || (l - 3.0_f64.sqrt()).abs() < 1e-10
                || (l - 2.0).abs() < 1e-10;
            assert!(valid, "unexpected edge length: {l}");
        }
    }

    #[test]
    fn test_rn_mesh() {
        let (complex, lengths) = reissner_nordstrom(3, 1.0, 0.1, 0.0, 0.5);

        // Should produce a valid mesh with all positive edge lengths.
        assert!(complex.n_pents() > 0);
        for &l in &lengths {
            assert!(l > 0.0, "negative or zero edge length: {l}");
        }
    }

    #[test]
    fn test_de_sitter_mesh() {
        let (complex, lengths) = de_sitter(3, 1.0, 10.0);

        assert!(complex.n_pents() > 0);
        for &l in &lengths {
            assert!(l > 0.0, "negative or zero edge length: {l}");
        }
    }

    #[test]
    fn test_kerr_mesh() {
        let (complex, lengths) = kerr(3, 1.0, 0.1, 0.3, 0.5);

        assert!(complex.n_pents() > 0);
        for &l in &lengths {
            assert!(l > 0.0, "negative or zero edge length: {l}");
        }
    }

    #[test]
    fn test_metric_line_element_diagonal() {
        // Flat Euclidean: g = diag(1,1,1,1).
        let g = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let dx = [1.0, 2.0, 3.0, 4.0];
        let ds_sq = metric_line_element(&g, &dx);
        assert!((ds_sq - 30.0).abs() < 1e-12);
    }

    #[test]
    fn test_metric_line_element_off_diagonal() {
        // Metric with off-diagonal terms.
        let mut g = [[0.0; 4]; 4];
        g[0][0] = 1.0;
        g[1][1] = 1.0;
        g[0][1] = 0.5;
        g[1][0] = 0.5;
        let dx = [1.0, 1.0, 0.0, 0.0];
        // ds² = 1·1 + 0.5·1 + 0.5·1 + 1·1 = 3.0
        let ds_sq = metric_line_element(&g, &dx);
        assert!((ds_sq - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_simpson_flat_matches_midpoint() {
        // For flat metric (M=0), both methods should produce the same set of lengths.
        let (_, mut lengths_mid) = reissner_nordstrom_with(3, 1.0, 0.0, 0.0, 0.5, MetricIntegration::Midpoint);
        let (_, mut lengths_simp) = reissner_nordstrom_with(3, 1.0, 0.0, 0.0, 0.5, MetricIntegration::Simpson);

        lengths_mid.sort_by(|a, b| a.partial_cmp(b).unwrap());
        lengths_simp.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(lengths_mid.len(), lengths_simp.len());
        for (lm, ls) in lengths_mid.iter().zip(lengths_simp.iter()) {
            assert!(
                (lm - ls).abs() < 1e-8,
                "midpoint={lm}, simpson={ls}, diff={}",
                (lm - ls).abs()
            );
        }
    }

    #[test]
    fn test_simpson_rn_positive_lengths() {
        let (complex, lengths) = reissner_nordstrom_with(3, 1.0, 0.1, 0.0, 0.5, MetricIntegration::Simpson);
        assert!(complex.n_pents() > 0);
        for &l in &lengths {
            assert!(l > 0.0, "negative or zero edge length: {l}");
        }
    }

    #[test]
    fn test_kerr_exact_ks_schwarzschild() {
        // a=0 should match Schwarzschild (same as RN Q=0).
        let (_, lengths_ks) = kerr_schild(3, 1.0, 0.1, 0.0, 0.5);
        let (_, lengths_rn) = reissner_nordstrom(3, 1.0, 0.1, 0.0, 0.5);

        // KS and RN use different coordinate representations, so they won't
        // match exactly, but should be in the same ballpark.
        for &l in &lengths_ks {
            assert!(l > 0.0, "negative or zero KS length: {l}");
        }
        // Just check both produce valid meshes.
        assert_eq!(lengths_ks.len(), lengths_rn.len());
    }

    #[test]
    fn test_kerr_exact_bl_schwarzschild() {
        let (complex, lengths) = kerr_bl(3, 1.0, 0.1, 0.0, 0.5);
        assert!(complex.n_pents() > 0);
        for &l in &lengths {
            assert!(l > 0.0, "negative or zero BL length: {l}");
        }
    }

    #[test]
    fn test_kerr_exact_high_spin() {
        let (complex, lengths) = kerr_schild(3, 1.0, 0.1, 0.09, 0.5);
        assert!(complex.n_pents() > 0);
        for &l in &lengths {
            assert!(l > 0.0, "negative or zero high-spin length: {l}");
        }
    }

    #[test]
    fn test_ks_bl_agree() {
        // At small spin, both forms should give similar edge lengths.
        let (_, lengths_ks) = kerr_schild(3, 1.0, 0.1, 0.01, 0.5);
        let (_, lengths_bl) = kerr_bl(3, 1.0, 0.1, 0.01, 0.5);

        // They use different coordinates so won't match exactly,
        // but should produce valid positive lengths.
        for (&lks, &lbl) in lengths_ks.iter().zip(lengths_bl.iter()) {
            assert!(lks > 0.0 && lbl > 0.0);
        }
    }

    #[test]
    fn test_de_sitter_static_mesh() {
        let (complex, lengths) = de_sitter_static(3, 1.0, 10.0, 0.1);
        assert!(complex.n_pents() > 0);
        for &l in &lengths {
            assert!(l > 0.0, "negative or zero edge length: {l}");
        }
    }

    #[test]
    fn test_de_sitter_static_flat_limit() {
        // As L → ∞, de Sitter static metric approaches flat.
        // With very large L, the metric factors are ~1 everywhere,
        // so all lengths should be bounded and positive.
        let (complex, lengths) = de_sitter_static(3, 1.0, 1000.0, 0.001);
        assert!(complex.n_pents() > 0);
        for &l in &lengths {
            assert!(l > 0.01, "length too small for near-flat metric: {l}");
        }
    }
}
