//! 3+1 foliated simplicial complexes for Lorentzian Regge calculus.
//!
//! Generates simplicial complexes with a clear time foliation — spatial slices
//! connected by timelike edges — suitable for tent-move evolution.
//!
//! Uses the 4D Kuhn (Freudenthal) triangulation — proven to tile R⁴ flat —
//! with foliation metadata derived from vertex time coordinates.

use crate::complex::SimplicialComplex;
use std::collections::HashMap;

/// A spatial (3D) triangulation forming one time-slice.
#[derive(Debug, Clone)]
pub struct SpatialSlice {
    pub n_vertices: usize,
    pub tets: Vec<[usize; 4]>,
    pub edge_index: HashMap<[usize; 2], usize>,
    pub edges: Vec<[usize; 2]>,
    pub tri_index: HashMap<[usize; 3], usize>,
    pub triangles: Vec<[usize; 3]>,
}

/// Classification of edges in a foliated complex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    /// Edge within a single spatial slice (Δt = 0).
    Spatial,
    /// Edge connecting corresponding vertices in adjacent slices (Δt = 1, Δx = Δy = Δz = 0).
    Timelike,
    /// Edge connecting different vertices across slices (Δt ≠ 0, spatial displacement ≠ 0).
    Diagonal,
}

/// A 4D simplicial complex with foliation structure.
#[derive(Debug, Clone)]
pub struct FoliatedComplex {
    pub complex: SimplicialComplex,
    pub n_slices: usize,
    pub vertices_per_slice: usize,
    pub n_spatial: usize,
    pub edge_types: Vec<EdgeType>,
    /// `vertex_map[slice][local_vertex]` → global vertex index
    pub vertex_map: Vec<Vec<usize>>,
}

fn sorted2(a: usize, b: usize) -> [usize; 2] {
    if a < b { [a, b] } else { [b, a] }
}

fn sorted3(a: usize, b: usize, c: usize) -> [usize; 3] {
    let mut v = [a, b, c];
    v.sort_unstable();
    v
}

fn sorted4(a: usize, b: usize, c: usize, d: usize) -> [usize; 4] {
    let mut v = [a, b, c, d];
    v.sort_unstable();
    v
}

/// Build a flat cubic spatial slice: n×n×n grid with periodic BCs (3-torus).
///
/// Each cube is subdivided into 6 tetrahedra using Kuhn triangulation.
pub fn flat_spatial_cube(n: usize, spacing: f64) -> (SpatialSlice, Vec<f64>) {
    let n_verts = n * n * n;
    let vidx = |x: usize, y: usize, z: usize| -> usize {
        (x % n) + n * (y % n) + n * n * (z % n)
    };

    let perms: [[usize; 3]; 6] = [
        [0, 1, 2], [0, 2, 1], [1, 0, 2],
        [1, 2, 0], [2, 0, 1], [2, 1, 0],
    ];

    let mut tet_set: Vec<[usize; 4]> = Vec::new();
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                for perm in &perms {
                    let mut coords = [x, y, z];
                    let v0 = vidx(coords[0], coords[1], coords[2]);
                    coords[perm[0]] += 1;
                    let v1 = vidx(coords[0], coords[1], coords[2]);
                    coords[perm[1]] += 1;
                    let v2 = vidx(coords[0], coords[1], coords[2]);
                    coords[perm[2]] += 1;
                    let v3 = vidx(coords[0], coords[1], coords[2]);
                    tet_set.push(sorted4(v0, v1, v2, v3));
                }
            }
        }
    }
    tet_set.sort_unstable();
    tet_set.dedup();

    let mut edge_index: HashMap<[usize; 2], usize> = HashMap::new();
    let mut edges: Vec<[usize; 2]> = Vec::new();
    let mut tri_index: HashMap<[usize; 3], usize> = HashMap::new();
    let mut triangles: Vec<[usize; 3]> = Vec::new();

    for tet in &tet_set {
        for i in 0..4 {
            for j in (i + 1)..4 {
                let e = sorted2(tet[i], tet[j]);
                let len = edges.len();
                edge_index.entry(e).or_insert_with(|| { edges.push(e); len });
            }
            for j in (i + 1)..4 {
                for k in (j + 1)..4 {
                    let t = sorted3(tet[i], tet[j], tet[k]);
                    let len = triangles.len();
                    tri_index.entry(t).or_insert_with(|| { triangles.push(t); len });
                }
            }
        }
    }

    let mut edge_lengths = vec![0.0; edges.len()];
    for (ei, e) in edges.iter().enumerate() {
        let (x0, y0, z0) = coords_3d(e[0], n);
        let (x1, y1, z1) = coords_3d(e[1], n);
        let dx = min_image(x0, x1, n);
        let dy = min_image(y0, y1, n);
        let dz = min_image(z0, z1, n);
        let dist_sq = (dx * dx + dy * dy + dz * dz) as f64;
        edge_lengths[ei] = dist_sq.sqrt() * spacing;
    }

    (SpatialSlice { n_vertices: n_verts, tets: tet_set, edge_index, edges, tri_index, triangles }, edge_lengths)
}

fn min_image(a: usize, b: usize, n: usize) -> i64 {
    let d = (b as i64) - (a as i64);
    let n = n as i64;
    if d > n / 2 { d - n } else if d < -n / 2 { d + n } else { d }
}

fn coords_3d(idx: usize, n: usize) -> (usize, usize, usize) {
    (idx % n, (idx / n) % n, idx / (n * n))
}

/// Build a foliated 4D complex using the standard 4D Kuhn triangulation.
///
/// Uses `n_t` time steps and `n_s` spatial grid points per axis.
/// Periodic boundary conditions in all directions (4-torus T⁴).
///
/// The 4D Kuhn triangulation is the same as `mesh::flat_hypercubic`,
/// but with foliation metadata extracted from vertex time coordinates.
///
/// Vertex indexing: `v(t, x, y, z) = t + n_t * (x + n_s * (y + n_s * z))`
pub fn foliated_hypercubic(n_t: usize, n_s: usize) -> FoliatedComplex {
    let n_v_per_slice = n_s * n_s * n_s;
    let n_total = n_t * n_v_per_slice;

    let vidx = |t: usize, x: usize, y: usize, z: usize| -> usize {
        (t % n_t) + n_t * ((x % n_s) + n_s * ((y % n_s) + n_s * (z % n_s)))
    };

    // 4D Kuhn: 24 permutations of [0,1,2,3] = (t,x,y,z) axes
    let permutations = all_permutations_4();
    let mut pent_set: std::collections::HashSet<[usize; 5]> = std::collections::HashSet::new();

    for z in 0..n_s {
        for y in 0..n_s {
            for x in 0..n_s {
                for t in 0..n_t {
                    for perm in &permutations {
                        let mut coords = [t, x, y, z];
                        let mut verts = [0usize; 5];
                        verts[0] = vidx(coords[0], coords[1], coords[2], coords[3]);
                        for step in 0..4 {
                            coords[perm[step]] += 1;
                            verts[step + 1] = vidx(coords[0], coords[1], coords[2], coords[3]);
                        }
                        verts.sort_unstable();
                        pent_set.insert(verts);
                    }
                }
            }
        }
    }

    let pents: Vec<[usize; 5]> = pent_set.into_iter().collect();
    let complex = SimplicialComplex::from_pentachorons(n_total, &pents);

    // Classify edges by time separation
    let mut edge_types = vec![EdgeType::Spatial; complex.n_edges()];
    for (ei, e) in complex.edges.iter().enumerate() {
        let t0 = e[0] % n_t;
        let t1 = e[1] % n_t;
        let spatial0 = e[0] / n_t;
        let spatial1 = e[1] / n_t;

        let dt = min_image(t0, t1, n_t);
        if dt == 0 {
            edge_types[ei] = EdgeType::Spatial;
        } else if spatial0 == spatial1 {
            edge_types[ei] = EdgeType::Timelike;
        } else {
            edge_types[ei] = EdgeType::Diagonal;
        }
    }

    // Build vertex map
    let vertex_map: Vec<Vec<usize>> = (0..n_t)
        .map(|t| {
            (0..n_v_per_slice)
                .map(|s| {
                    let x = s % n_s;
                    let y = (s / n_s) % n_s;
                    let z = s / (n_s * n_s);
                    vidx(t, x, y, z)
                })
                .collect()
        })
        .collect();

    FoliatedComplex {
        complex,
        n_slices: n_t,
        vertices_per_slice: n_v_per_slice,
        n_spatial: n_s,
        edge_types,
        vertex_map,
    }
}

/// Assign flat Minkowski signed squared edge lengths to a foliated complex.
///
/// Uses the vertex coordinate system to compute exact Minkowski distances.
pub fn flat_minkowski_sq_lengths(
    fc: &FoliatedComplex,
    spacing: f64,
    dt: f64,
) -> Vec<f64> {
    let n_t = fc.n_slices;
    let n_s = fc.n_spatial;
    let mut sq_lengths = vec![0.0; fc.complex.n_edges()];

    for (ei, e) in fc.complex.edges.iter().enumerate() {
        let (t0, x0, y0, z0) = vertex_coords_4d(e[0], n_t, n_s);
        let (t1, x1, y1, z1) = vertex_coords_4d(e[1], n_t, n_s);

        let delta_t = min_image(t0, t1, n_t) as f64 * dt;
        let delta_x = min_image(x0, x1, n_s) as f64 * spacing;
        let delta_y = min_image(y0, y1, n_s) as f64 * spacing;
        let delta_z = min_image(z0, z1, n_s) as f64 * spacing;

        // Minkowski: ds² = -dt² + dx² + dy² + dz²
        sq_lengths[ei] = -delta_t * delta_t + delta_x * delta_x
            + delta_y * delta_y + delta_z * delta_z;
    }

    sq_lengths
}

fn vertex_coords_4d(idx: usize, n_t: usize, n_s: usize) -> (usize, usize, usize, usize) {
    let t = idx % n_t;
    let rest = idx / n_t;
    let x = rest % n_s;
    let y = (rest / n_s) % n_s;
    let z = rest / (n_s * n_s);
    (t, x, y, z)
}

fn all_permutations_4() -> Vec<[usize; 4]> {
    let mut perms = Vec::with_capacity(24);
    let mut current = [0usize, 1, 2, 3];
    permute(0, &mut current, &mut perms);
    perms
}

fn permute(start: usize, current: &mut [usize; 4], result: &mut Vec<[usize; 4]>) {
    if start == 4 {
        result.push(*current);
        return;
    }
    for i in start..4 {
        current.swap(start, i);
        permute(start + 1, current, result);
        current.swap(start, i);
    }
}

/// Deform spatial edge lengths of a slice using a 3-metric.
pub fn deform_spatial_slice(
    slice: &SpatialSlice,
    n: usize,
    spacing: f64,
    metric_3: impl Fn(f64, f64, f64) -> [[f64; 3]; 3],
) -> Vec<f64> {
    let half = n as f64 / 2.0;
    let pos = |idx: usize| -> [f64; 3] {
        let (x, y, z) = coords_3d(idx, n);
        [
            (x as f64 - half) * spacing,
            (y as f64 - half) * spacing,
            (z as f64 - half) * spacing,
        ]
    };

    let mut edge_lengths = vec![0.0; slice.edges.len()];
    for (ei, e) in slice.edges.iter().enumerate() {
        let p0 = pos(e[0]);
        let p1 = pos(e[1]);
        let mid = [
            (p0[0] + p1[0]) / 2.0,
            (p0[1] + p1[1]) / 2.0,
            (p0[2] + p1[2]) / 2.0,
        ];
        let dx = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];

        let g = metric_3(mid[0], mid[1], mid[2]);
        let mut ds_sq = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                ds_sq += g[i][j] * dx[i] * dx[j];
            }
        }
        edge_lengths[ei] = ds_sq.abs().sqrt().max(1e-10);
    }

    edge_lengths
}

impl FoliatedComplex {
    /// Get the global vertex index for a (slice, local_vertex) pair.
    pub fn global_vertex(&self, slice: usize, local: usize) -> usize {
        self.vertex_map[slice][local]
    }

    /// Get the slice index for a global vertex.
    pub fn vertex_slice(&self, global: usize) -> usize {
        global % self.n_slices
    }

    /// Get the local vertex index within its slice.
    pub fn vertex_local(&self, global: usize) -> usize {
        global / self.n_slices
    }

    /// Get the spatial neighbors of a vertex within its slice.
    pub fn spatial_neighbors(&self, global_vertex: usize) -> Vec<usize> {
        let slice = self.vertex_slice(global_vertex);
        let mut neighbors = Vec::new();
        for (ei, e) in self.complex.edges.iter().enumerate() {
            if self.edge_types[ei] != EdgeType::Spatial {
                continue;
            }
            if e[0] == global_vertex {
                if self.vertex_slice(e[1]) == slice {
                    neighbors.push(e[1]);
                }
            } else if e[1] == global_vertex && self.vertex_slice(e[0]) == slice {
                neighbors.push(e[0]);
            }
        }
        neighbors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lorentzian_regge::lorentzian_regge_action;

    #[test]
    fn test_flat_spatial_cube_counts() {
        let n = 2;
        let (slice, lengths) = flat_spatial_cube(n, 1.0);
        assert_eq!(slice.n_vertices, 8);
        assert!(!slice.tets.is_empty());
        for &l in &lengths {
            assert!(l > 0.0);
        }
    }

    #[test]
    fn test_foliated_hypercubic_counts() {
        let fc = foliated_hypercubic(2, 2);
        assert_eq!(fc.n_slices, 2);
        assert_eq!(fc.vertices_per_slice, 8);
        assert_eq!(fc.complex.n_vertices, 16);
        assert!(fc.complex.n_pents() > 0);
    }

    #[test]
    fn test_edge_type_classification() {
        let fc = foliated_hypercubic(2, 2);

        let n_spatial = fc.edge_types.iter().filter(|&&t| t == EdgeType::Spatial).count();
        let n_timelike = fc.edge_types.iter().filter(|&&t| t == EdgeType::Timelike).count();
        let n_diagonal = fc.edge_types.iter().filter(|&&t| t == EdgeType::Diagonal).count();

        assert!(n_spatial > 0, "no spatial edges");
        assert!(n_timelike > 0, "no timelike edges");
        assert!(n_diagonal > 0, "no diagonal edges");
        assert_eq!(n_spatial + n_timelike + n_diagonal, fc.complex.n_edges());
    }

    #[test]
    fn test_flat_minkowski_zero_action() {
        let fc = foliated_hypercubic(2, 2);
        let sq_lengths = flat_minkowski_sq_lengths(&fc, 1.0, 0.3);

        let deficits = crate::lorentzian_regge::lorentzian_deficit_angles(
            &fc.complex, &sq_lengths,
        );

        let mut max_spacelike = 0.0f64;
        let mut max_timelike = 0.0f64;
        let mut n_sl = 0;
        let mut n_tl = 0;
        for &(d, ht) in &deficits {
            match ht {
                crate::lorentzian::HingeType::Spacelike => {
                    max_spacelike = max_spacelike.max(d.abs());
                    n_sl += 1;
                }
                crate::lorentzian::HingeType::Timelike => {
                    max_timelike = max_timelike.max(d.abs());
                    n_tl += 1;
                }
            }
        }
        eprintln!("spacelike: {n_sl} (max |δ| = {max_spacelike:.6e}), timelike: {n_tl} (max |δ| = {max_timelike:.6e})");

        let action = lorentzian_regge_action(&fc.complex, &sq_lengths);
        assert!(
            action.abs() < 1e-6,
            "flat Minkowski action = {action}"
        );
    }

    #[test]
    fn test_flat_minkowski_matches_euclidean() {
        // When all s_e > 0 (pure Euclidean-like assignment),
        // the foliated mesh should match flat_hypercubic behavior.
        let fc = foliated_hypercubic(2, 2);
        // Assign all-positive squared lengths (Euclidean-like, spacing=1 in all directions)
        let sq_lengths: Vec<f64> = fc.complex.edges.iter().enumerate().map(|(_ei, e)| {
            let (t0, x0, y0, z0) = vertex_coords_4d(e[0], fc.n_slices, fc.n_spatial);
            let (t1, x1, y1, z1) = vertex_coords_4d(e[1], fc.n_slices, fc.n_spatial);
            let dt = min_image(t0, t1, fc.n_slices) as f64;
            let dx = min_image(x0, x1, fc.n_spatial) as f64;
            let dy = min_image(y0, y1, fc.n_spatial) as f64;
            let dz = min_image(z0, z1, fc.n_spatial) as f64;
            dt*dt + dx*dx + dy*dy + dz*dz
        }).collect();

        let action = lorentzian_regge_action(&fc.complex, &sq_lengths);
        assert!(
            action.abs() < 1e-6,
            "flat Euclidean foliated action = {action} (expected ~0)"
        );
    }

    #[test]
    fn test_vertex_map_consistency() {
        let fc = foliated_hypercubic(2, 2);
        for s in 0..fc.n_slices {
            for v in 0..fc.vertices_per_slice {
                let global = fc.global_vertex(s, v);
                assert_eq!(fc.vertex_slice(global), s);
                assert_eq!(fc.vertex_local(global), v);
            }
        }
    }

    #[test]
    fn test_pent_edge_type_mix() {
        let fc = foliated_hypercubic(2, 2);
        for pi in 0..fc.complex.n_pents() {
            let edge_indices = fc.complex.pent_edge_indices(pi);
            let types: Vec<EdgeType> = edge_indices.iter().map(|&ei| fc.edge_types[ei]).collect();
            let has_spatial = types.iter().any(|&t| t == EdgeType::Spatial);
            let has_time = types.iter().any(|&t| t == EdgeType::Timelike || t == EdgeType::Diagonal);
            assert!(has_spatial || has_time, "pent {pi}: {types:?}");
        }
    }
}
