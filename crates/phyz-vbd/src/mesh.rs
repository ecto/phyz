//! Small deterministic tet-mesh generators, so the tests and examples have
//! something to bend without shipping mesh files.

use phyz_math::Vec3;

/// A box of `nx × ny × nz` cells spanning `[0, size]`, each cell split into six
/// tetrahedra.
///
/// Returns `(positions, tets)`. Vertex `(i, j, k)` is at index
/// `(i·(ny+1) + j)·(nz+1) + k`, so the `i = 0` face — the usual clamped end of a
/// cantilever — is the first `(ny+1)(nz+1)` indices.
///
/// The split is Kuhn's: the six tets of a cell are the six monotone paths from
/// corner `(0,0,0)` to corner `(1,1,1)`, one per axis permutation. Kuhn's
/// decomposition is used rather than the more compact five-tet one because it
/// is *conforming without alternation*: five-tet cells only match across a face
/// if neighbouring cells use mirrored splits, and getting that parity wrong
/// produces a mesh that looks fine and has invisible cracks along half its
/// internal faces. Six tets per cell costs 20% more elements and removes the
/// entire class of bug.
pub fn tet_box(nx: usize, ny: usize, nz: usize, size: Vec3) -> (Vec<Vec3>, Vec<[usize; 4]>) {
    assert!(nx > 0 && ny > 0 && nz > 0, "a box needs at least one cell");
    let index = |i: usize, j: usize, k: usize| (i * (ny + 1) + j) * (nz + 1) + k;

    let mut positions = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    for i in 0..=nx {
        for j in 0..=ny {
            for k in 0..=nz {
                positions.push(Vec3::new(
                    size.x * i as f64 / nx as f64,
                    size.y * j as f64 / ny as f64,
                    size.z * k as f64 / nz as f64,
                ));
            }
        }
    }

    const PERMUTATIONS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    let mut tets = Vec::with_capacity(6 * nx * ny * nz);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                for perm in PERMUTATIONS {
                    let mut offset = [0usize; 3];
                    let mut path = [index(i, j, k); 4];
                    for (step, &axis) in perm.iter().enumerate() {
                        offset[axis] = 1;
                        path[step + 1] = index(i + offset[0], j + offset[1], k + offset[2]);
                    }
                    tets.push(path);
                }
            }
        }
    }

    (positions, tets)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tet_volume(p: [Vec3; 4]) -> f64 {
        (p[1] - p[0]).cross(p[2] - p[0]).dot(p[3] - p[0]) / 6.0
    }

    /// The six tets must tile the cell exactly — no gaps, no overlaps. Total
    /// volume is the cheap global check for that.
    #[test]
    fn tets_tile_the_box_volume() {
        let size = Vec3::new(0.4, 0.1, 0.2);
        let (p, tets) = tet_box(4, 2, 3, size);
        let total: f64 = tets
            .iter()
            .map(|t| tet_volume([p[t[0]], p[t[1]], p[t[2]], p[t[3]]]).abs())
            .sum();
        let expected = size.x * size.y * size.z;
        assert!(
            (total - expected).abs() < 1e-12 * expected,
            "volume {total} vs {expected}"
        );
    }

    #[test]
    fn counts_are_what_the_grid_implies() {
        let (p, tets) = tet_box(3, 2, 1, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(p.len(), 4 * 3 * 2);
        assert_eq!(tets.len(), 6 * 3 * 2);
    }

    #[test]
    fn no_degenerate_tets() {
        let (p, tets) = tet_box(2, 2, 2, Vec3::new(1.0, 1.0, 1.0));
        for t in &tets {
            let v = tet_volume([p[t[0]], p[t[1]], p[t[2]], p[t[3]]]).abs();
            assert!(v > 1e-6, "degenerate tet {t:?}");
        }
    }

    #[test]
    fn generation_is_deterministic() {
        let a = tet_box(3, 2, 2, Vec3::new(1.0, 0.2, 0.2));
        let b = tet_box(3, 2, 2, Vec3::new(1.0, 0.2, 0.2));
        assert_eq!(a, b);
    }
}
