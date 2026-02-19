//! Simplicial complex in 4D.
//!
//! Stores the combinatorial structure: vertices, edges, triangles,
//! tetrahedra, and 4-simplices (pentachorons), plus adjacency maps
//! needed for Regge calculus (which triangles border which 4-simplices).

use std::collections::HashMap;

/// Sorted edge (2 vertices).
pub type Edge = [usize; 2];
/// Sorted triangle (3 vertices).
pub type Triangle = [usize; 3];
/// Sorted tetrahedron (4 vertices).
pub type Tet = [usize; 4];
/// Sorted 4-simplex / pentachoron (5 vertices).
pub type Pent = [usize; 5];

/// Sort a small array of vertex indices in place.
fn sorted<const N: usize>(mut a: [usize; N]) -> [usize; N] {
    a.sort_unstable();
    a
}

/// 4D simplicial complex with full incidence structure.
#[derive(Debug, Clone)]
pub struct SimplicialComplex {
    /// Number of vertices.
    pub n_vertices: usize,

    /// Edges: sorted pairs of vertex indices.
    pub edges: Vec<Edge>,
    /// Map from edge → index in `edges`.
    pub edge_index: HashMap<Edge, usize>,

    /// Triangles: sorted triples.
    pub triangles: Vec<Triangle>,
    /// Map from triangle → index in `triangles`.
    pub tri_index: HashMap<Triangle, usize>,

    /// Tetrahedra: sorted 4-tuples.
    pub tets: Vec<Tet>,
    /// Map from tet → index in `tets`.
    pub tet_index: HashMap<Tet, usize>,

    /// 4-simplices: sorted 5-tuples.
    pub pents: Vec<Pent>,

    /// For each triangle, the list of 4-simplices containing it.
    /// tri_to_pents[tri_idx] = vec of pent indices.
    pub tri_to_pents: Vec<Vec<usize>>,

    /// For each edge, the list of triangles containing it.
    /// edge_to_tris[edge_idx] = vec of triangle indices.
    pub edge_to_tris: Vec<Vec<usize>>,

    /// For each triangle in each 4-simplex: the two "opposite" vertices.
    /// Given triangle [a,b,c] in pent [a,b,c,d,e], the opposite edge is [d,e].
    /// tri_pent_opposite[tri_idx] stores (pent_idx, [d, e]) pairs.
    pub tri_pent_opposite: Vec<Vec<(usize, [usize; 2])>>,

    /// For each vertex, the list of triangles containing it.
    /// vertex_to_tris[vertex] = vec of triangle indices.
    pub vertex_to_tris: Vec<Vec<usize>>,
}

impl SimplicialComplex {
    /// Build a simplicial complex from a list of 4-simplices.
    ///
    /// Each 4-simplex is given as 5 vertex indices. The combinatorial
    /// structure (edges, triangles, tets, adjacency) is derived automatically.
    pub fn from_pentachorons(n_vertices: usize, pents: &[[usize; 5]]) -> Self {
        let mut edge_index: HashMap<Edge, usize> = HashMap::new();
        let mut edges: Vec<Edge> = Vec::new();

        let mut tri_index: HashMap<Triangle, usize> = HashMap::new();
        let mut triangles: Vec<Triangle> = Vec::new();

        let mut tet_index: HashMap<Tet, usize> = HashMap::new();
        let mut tets: Vec<Tet> = Vec::new();

        let mut sorted_pents: Vec<Pent> = Vec::with_capacity(pents.len());

        // Extract all sub-simplices from each pentachoron.
        for p in pents {
            let sp = sorted(*p);
            sorted_pents.push(sp);

            // 10 edges (5 choose 2)
            for i in 0..5 {
                for j in (i + 1)..5 {
                    let e = [sp[i], sp[j]];
                    let len = edges.len();
                    edge_index.entry(e).or_insert_with(|| {
                        edges.push(e);
                        len
                    });
                }
            }

            // 10 triangles (5 choose 3)
            for i in 0..5 {
                for j in (i + 1)..5 {
                    for k in (j + 1)..5 {
                        let t = [sp[i], sp[j], sp[k]];
                        let len = triangles.len();
                        tri_index.entry(t).or_insert_with(|| {
                            triangles.push(t);
                            len
                        });
                    }
                }
            }

            // 5 tetrahedra (5 choose 4)
            for skip in 0..5 {
                let mut tet = [0usize; 4];
                let mut idx = 0;
                for v in 0..5 {
                    if v != skip {
                        tet[idx] = sp[v];
                        idx += 1;
                    }
                }
                let len = tets.len();
                tet_index.entry(tet).or_insert_with(|| {
                    tets.push(tet);
                    len
                });
            }
        }

        // Build triangle → pentachoron adjacency.
        let n_tris = triangles.len();
        let mut tri_to_pents = vec![Vec::new(); n_tris];
        let mut tri_pent_opposite = vec![Vec::new(); n_tris];

        for (pi, sp) in sorted_pents.iter().enumerate() {
            for i in 0..5 {
                for j in (i + 1)..5 {
                    for k in (j + 1)..5 {
                        let t = [sp[i], sp[j], sp[k]];
                        let ti = tri_index[&t];
                        tri_to_pents[ti].push(pi);

                        // The two vertices NOT in the triangle.
                        let mut opp = [0usize; 2];
                        let mut oi = 0;
                        for v in 0..5 {
                            if sp[v] != t[0] && sp[v] != t[1] && sp[v] != t[2] {
                                opp[oi] = sp[v];
                                oi += 1;
                            }
                        }
                        opp.sort_unstable();
                        tri_pent_opposite[ti].push((pi, opp));
                    }
                }
            }
        }

        // Build edge → triangle adjacency.
        let n_edges = edges.len();
        let mut edge_to_tris = vec![Vec::new(); n_edges];
        for (ti, t) in triangles.iter().enumerate() {
            // 3 edges per triangle
            for i in 0..3 {
                for j in (i + 1)..3 {
                    let e = sorted([t[i], t[j]]);
                    if let Some(&ei) = edge_index.get(&e) {
                        edge_to_tris[ei].push(ti);
                    }
                }
            }
        }

        // Build vertex → triangle adjacency.
        let mut vertex_to_tris = vec![Vec::new(); n_vertices];
        for (ti, t) in triangles.iter().enumerate() {
            for &v in t {
                vertex_to_tris[v].push(ti);
            }
        }

        Self {
            n_vertices,
            edges,
            edge_index,
            triangles,
            tri_index,
            tets,
            tet_index,
            pents: sorted_pents,
            tri_to_pents,
            edge_to_tris,
            tri_pent_opposite,
            vertex_to_tris,
        }
    }

    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Number of triangles (hinges for Regge curvature in 4D).
    pub fn n_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// Number of 4-simplices.
    pub fn n_pents(&self) -> usize {
        self.pents.len()
    }

    /// Get the 10 edge lengths of a 4-simplex from the global edge-length array.
    ///
    /// Returns lengths in the order needed for the Cayley-Menger determinant:
    /// l_{01}, l_{02}, l_{03}, l_{04}, l_{12}, l_{13}, l_{14}, l_{23}, l_{24}, l_{34}
    pub fn pent_edge_lengths(&self, pent_idx: usize, lengths: &[f64]) -> [f64; 10] {
        let p = &self.pents[pent_idx];
        let mut result = [0.0; 10];
        let mut idx = 0;
        for i in 0..5 {
            for j in (i + 1)..5 {
                let e = sorted([p[i], p[j]]);
                result[idx] = lengths[self.edge_index[&e]];
                idx += 1;
            }
        }
        result
    }

    /// Get the 10 global edge indices of a 4-simplex.
    ///
    /// Returns indices in the same order as `pent_edge_lengths`:
    /// (01, 02, 03, 04, 12, 13, 14, 23, 24, 34) in local vertex ordering.
    pub fn pent_edge_indices(&self, pent_idx: usize) -> [usize; 10] {
        let p = &self.pents[pent_idx];
        let mut result = [0usize; 10];
        let mut idx = 0;
        for i in 0..5 {
            for j in (i + 1)..5 {
                let e = sorted([p[i], p[j]]);
                result[idx] = self.edge_index[&e];
                idx += 1;
            }
        }
        result
    }

    /// Get the 3 edge lengths of a triangle from the global edge-length array.
    pub fn tri_edge_lengths(&self, tri_idx: usize, lengths: &[f64]) -> [f64; 3] {
        let t = &self.triangles[tri_idx];
        [
            lengths[self.edge_index[&sorted([t[0], t[1]])]],
            lengths[self.edge_index[&sorted([t[0], t[2]])]],
            lengths[self.edge_index[&sorted([t[1], t[2]])]],
        ]
    }

    /// Get the indices of the 3 edges forming a triangle.
    pub fn tri_edge_indices(&self, tri_idx: usize) -> [usize; 3] {
        let t = &self.triangles[tri_idx];
        [
            self.edge_index[&sorted([t[0], t[1]])],
            self.edge_index[&sorted([t[0], t[2]])],
            self.edge_index[&sorted([t[1], t[2]])],
        ]
    }

    /// For a triangle in a 4-simplex, find the index pair (within 0..5)
    /// of the two "opposite" vertices (the ones not in the triangle).
    ///
    /// Returns (local_idx_d, local_idx_e) such that pent[d] and pent[e]
    /// are the vertices not in the triangle.
    pub fn opposite_local_indices(pent: &Pent, tri: &Triangle) -> (usize, usize) {
        let mut opp = [0usize; 2];
        let mut oi = 0;
        for (v, &pv) in pent.iter().enumerate() {
            if pv != tri[0] && pv != tri[1] && pv != tri[2] {
                opp[oi] = v;
                oi += 1;
            }
        }
        (opp[0], opp[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_pentachoron() {
        let complex = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);

        assert_eq!(complex.n_edges(), 10); // 5 choose 2
        assert_eq!(complex.n_triangles(), 10); // 5 choose 3
        assert_eq!(complex.tets.len(), 5); // 5 choose 4
        assert_eq!(complex.n_pents(), 1);

        // Each triangle should be in exactly 1 pentachoron.
        for adj in &complex.tri_to_pents {
            assert_eq!(adj.len(), 1);
        }
    }

    #[test]
    fn test_two_shared_pentachorons() {
        // Two pentachorons sharing a tetrahedron [0,1,2,3].
        let complex = SimplicialComplex::from_pentachorons(6, &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]]);

        assert_eq!(complex.n_pents(), 2);

        // Triangles in [0,1,2,3] should appear in both pentachorons.
        let shared_tri = sorted([0, 1, 2]);
        let ti = complex.tri_index[&shared_tri];
        assert_eq!(complex.tri_to_pents[ti].len(), 2);
    }

    #[test]
    fn test_edge_adjacency() {
        let complex = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);

        // Edge [0,1] appears in triangles [0,1,2], [0,1,3], [0,1,4] = 3 triangles.
        let ei = complex.edge_index[&[0, 1]];
        assert_eq!(complex.edge_to_tris[ei].len(), 3);
    }
}
