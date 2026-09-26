//! Deterministic greedy graph colouring of the vertex-interaction graph.
//!
//! # Why colouring is not optional
//!
//! VBD is Gauss–Seidel: vertex `i` reads the current positions of every vertex
//! it shares an energy term with. Two vertices that share a term therefore
//! *cannot* be updated concurrently, and — more subtly — the answer depends on
//! which of them went first even when you run serially. Colouring removes both
//! problems at once: within one colour no two vertices share a term, so the
//! sweep over a colour is order-independent, and a parallel backend may run it
//! however it likes and still land on the same positions as the serial code.
//!
//! # The edge that is easy to get wrong
//!
//! The graph is *not* the mesh's edge graph. A tetrahedron's energy is a
//! function of all four of its vertices, so all **six** pairs must be edges,
//! including the pairs that are only tet-diagonal in some other reading of the
//! mesh. Colouring the edge graph of a tet mesh happens to give the same answer
//! (a tet's edges are already the complete graph on its four vertices) — but
//! for elements with a non-complete edge set (a hex, a quad bending term) it
//! would silently place two coupled vertices in the same colour, and the
//! serial code would keep working while the parallel one produced garbage.
//! [`color_vertices`] therefore takes *element vertex lists*, not edges, and
//! completes each one.
//!
//! # Determinism
//!
//! Adjacency is a sorted, deduplicated `Vec<Vec<usize>>` — never a hash
//! container — vertices are visited in ascending index order, and each takes
//! the smallest colour not used by an already-coloured neighbour. Same mesh in,
//! same colouring out, on every platform.

/// Colour the vertices of `n_vertices` such that no two vertices sharing an
/// element get the same colour.
///
/// `elements` is a list of vertex-index groups — a tet contributes its four
/// indices, a spring its two. Returns colour classes, each sorted ascending,
/// ordered by colour number.
///
/// Isolated vertices (in no element) all land in colour 0, which is correct and
/// costs nothing: they have no incident energy, so their update is a no-op.
pub fn color_vertices(n_vertices: usize, elements: &[&[usize]]) -> Vec<Vec<usize>> {
    let adjacency = build_adjacency(n_vertices, elements);

    let mut color_of = vec![usize::MAX; n_vertices];
    // Reused scratch: `forbidden[c] == v + 1` means colour `c` is taken by a
    // neighbour of vertex `v`. Stamping with the vertex id avoids clearing the
    // whole vector per vertex, which would make colouring quadratic in the
    // colour count on large meshes.
    let mut forbidden = vec![0usize; 1];

    for v in 0..n_vertices {
        for &u in &adjacency[v] {
            let c = color_of[u];
            if c != usize::MAX {
                if c >= forbidden.len() {
                    forbidden.resize(c + 1, 0);
                }
                forbidden[c] = v + 1;
            }
        }
        let mut c = 0;
        while c < forbidden.len() && forbidden[c] == v + 1 {
            c += 1;
        }
        if c >= forbidden.len() {
            forbidden.resize(c + 1, 0);
        }
        color_of[v] = c;
    }

    let n_colors = color_of.iter().map(|c| c + 1).max().unwrap_or(0);
    let mut classes = vec![Vec::new(); n_colors];
    for (v, &c) in color_of.iter().enumerate() {
        classes[c].push(v);
    }
    classes
}

/// Sorted, deduplicated adjacency lists over the completed element graph.
fn build_adjacency(n_vertices: usize, elements: &[&[usize]]) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); n_vertices];
    for element in elements {
        for (a, &i) in element.iter().enumerate() {
            for &j in element.iter().skip(a + 1) {
                if i != j && i < n_vertices && j < n_vertices {
                    adjacency[i].push(j);
                    adjacency[j].push(i);
                }
            }
        }
    }
    for list in &mut adjacency {
        list.sort_unstable();
        list.dedup();
    }
    adjacency
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the whole method rests on: no element has two vertices of
    /// the same colour.
    fn assert_valid(n: usize, elements: &[&[usize]], classes: &[Vec<usize>]) {
        let mut color_of = vec![usize::MAX; n];
        for (c, class) in classes.iter().enumerate() {
            for &v in class {
                assert_eq!(color_of[v], usize::MAX, "vertex {v} coloured twice");
                color_of[v] = c;
            }
        }
        assert!(
            color_of.iter().all(|&c| c != usize::MAX),
            "vertex uncoloured"
        );
        for element in elements {
            for (a, &i) in element.iter().enumerate() {
                for &j in element.iter().skip(a + 1) {
                    assert_ne!(color_of[i], color_of[j], "conflict in element {element:?}");
                }
            }
        }
    }

    #[test]
    fn single_tet_needs_four_colors() {
        let tet: &[usize] = &[0, 1, 2, 3];
        let classes = color_vertices(4, &[tet]);
        assert_eq!(classes.len(), 4);
        assert_valid(4, &[tet], &classes);
    }

    #[test]
    fn two_tets_sharing_a_face() {
        let a: &[usize] = &[0, 1, 2, 3];
        let b: &[usize] = &[1, 2, 3, 4];
        let elements = [a, b];
        let classes = color_vertices(5, &elements);
        assert_valid(5, &elements, &classes);
        // Vertices 0 and 4 do not share an element, so a greedy pass must reuse
        // colour 0 for both — four colours total, not five.
        assert_eq!(classes.len(), 4);
    }

    #[test]
    fn springs_are_respected() {
        let s0: &[usize] = &[0, 1];
        let s1: &[usize] = &[1, 2];
        let elements = [s0, s1];
        let classes = color_vertices(3, &elements);
        assert_valid(3, &elements, &classes);
        assert_eq!(classes.len(), 2);
    }

    #[test]
    fn isolated_vertices_share_color_zero() {
        let classes = color_vertices(3, &[]);
        assert_eq!(classes, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn coloring_is_deterministic() {
        let a: &[usize] = &[0, 1, 2, 3];
        let b: &[usize] = &[2, 3, 4, 5];
        let c: &[usize] = &[4, 5, 6, 7];
        let elements = [a, b, c];
        let first = color_vertices(8, &elements);
        for _ in 0..8 {
            assert_eq!(color_vertices(8, &elements), first);
        }
    }

    #[test]
    fn classes_are_sorted_ascending() {
        let a: &[usize] = &[0, 3, 5, 7];
        let b: &[usize] = &[1, 2, 4, 6];
        let elements = [a, b];
        for class in color_vertices(8, &elements) {
            assert!(class.windows(2).all(|w| w[0] < w[1]));
        }
    }
}
