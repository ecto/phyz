//! Mesh generation for 4D simplicial complexes.
//!
//! Generates regular triangulations of simple topologies for testing
//! and as starting points for the symmetry search.

use crate::complex::SimplicialComplex;

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
    let (complex, _flat_lengths) = flat_hypercubic(n, spacing);

    let r_plus = mass + (mass * mass - charge * charge).max(0.0).sqrt();
    let r_minus = mass - (mass * mass - charge * charge).max(0.0).sqrt();

    // Compute vertex positions in isotropic coordinates.
    // Center the grid: coordinates go from -L/2 to +L/2.
    let half = n as f64 / 2.0;

    let vertex_pos = |idx: usize| -> (f64, f64, f64, f64) {
        let (t, x, y, z) = vertex_coords(idx, n);
        let tc = (t as f64 - half) * spacing;
        let xc = (x as f64 - half) * spacing;
        let yc = (y as f64 - half) * spacing;
        let zc = (z as f64 - half) * spacing;
        (tc, xc, yc, zc)
    };

    // Metric factors at a point.
    let metric_factors = |_t: f64, x: f64, y: f64, z: f64| -> (f64, f64) {
        let r = (x * x + y * y + z * z).sqrt().max(r_min);
        let a = 1.0 + r_plus / (2.0 * r);
        let b = 1.0 + r_minus / (2.0 * r);
        let c = 1.0 - r_plus / (2.0 * r);
        let d = 1.0 - r_minus / (2.0 * r);

        let f = (c * d) / (a * b); // -g_tt = f
        let g = (a * b) * (a * b); // g_xx = g_yy = g_zz = g

        (f.abs().max(0.01), g.max(0.01)) // Clamp to avoid singularity
    };

    // Deform edge lengths using metric at the midpoint of each edge.
    let mut lengths = vec![0.0; complex.n_edges()];
    for (ei, edge) in complex.edges.iter().enumerate() {
        let (t0, x0, y0, z0) = vertex_pos(edge[0]);
        let (t1, x1, y1, z1) = vertex_pos(edge[1]);

        // Midpoint metric.
        let tm = (t0 + t1) / 2.0;
        let xm = (x0 + x1) / 2.0;
        let ym = (y0 + y1) / 2.0;
        let zm = (z0 + z1) / 2.0;
        let (f, g) = metric_factors(tm, xm, ym, zm);

        // Edge displacement.
        let dt = t1 - t0;
        let dx = x1 - x0;
        let dy = y1 - y0;
        let dz = z1 - z0;

        // ds² = f dt² + g (dx² + dy² + dz²)
        // (Using Euclidean signature for Regge calculus; the Lorentzian
        // signature requires more careful treatment.)
        let ds_sq = f * dt * dt + g * (dx * dx + dy * dy + dz * dz);
        lengths[ei] = ds_sq.abs().sqrt().max(1e-10);
    }

    (complex, lengths)
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
}
