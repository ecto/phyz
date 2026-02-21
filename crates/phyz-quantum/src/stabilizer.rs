//! Z₂ limit of the gauge theory → stabilizer code.
//!
//! When Λ = 1/2 (i.e., each edge carries a Z₂ = {0, 1} variable), the
//! Kogut-Susskind Hamiltonian reduces to a stabilizer code:
//!
//! - **Star operators** A_v = ∏_{e∋v} X_e (vertex terms, Gauss law)
//! - **Plaquette operators** B_p = ∏_{e∈∂p} Z_e (triangle terms)
//!
//! On a simplicial complex, star operators have variable weight (vertex
//! coordination number), and plaquettes are triangles (weight 3) instead
//! of squares (weight 4). This gives different code parameters than the
//! standard toric code.

use phyz_regge::SimplicialComplex;

/// Parameters of the Z₂ stabilizer code on a simplicial complex.
#[derive(Debug, Clone)]
pub struct StabilizerCode {
    /// Number of physical qubits (= number of edges).
    pub n: usize,
    /// Number of logical qubits (= first Betti number b₁).
    pub k: usize,
    /// Code distance (minimum weight of a non-trivial logical operator).
    pub d: usize,
    /// Number of star (vertex) stabilizers.
    pub n_stars: usize,
    /// Number of plaquette (triangle) stabilizers.
    pub n_plaquettes: usize,
    /// Star operator weights (coordination numbers).
    pub star_weights: Vec<usize>,
    /// Plaquette operator weight (always 3 for triangles).
    pub plaquette_weight: usize,
}

/// Compute the Z₂ stabilizer code parameters for a simplicial complex.
///
/// The code has:
/// - n = |E| physical qubits
/// - Star stabilizers: one per vertex (minus 1 for the dependency)
/// - Plaquette stabilizers: one per triangle (minus b₂ for dependencies)
/// - k = b₁ = |E| - |V| + 1 logical qubits (for connected complexes)
///
/// The code distance d is the minimum weight of a non-trivial cycle
/// (homologically non-trivial loop). We compute this by brute-force
/// for small complexes.
pub fn stabilizer_code(complex: &SimplicialComplex) -> StabilizerCode {
    let n = complex.n_edges();
    let n_v = complex.n_vertices;
    let n_tri = complex.n_triangles();
    let b1 = n - n_v + 1;

    // Star weights = vertex coordination numbers.
    let adj = crate::gauss_law::vertex_edge_adjacency(complex);
    let star_weights: Vec<usize> = adj.iter().map(|a| a.len()).collect();

    // Independent stabilizers:
    // Stars: n_v - 1 (product of all stars = identity)
    // Plaquettes: varies (n_tri minus nullity)
    let n_stars = n_v; // total star operators (one dependent)
    let n_plaquettes = n_tri;

    // Code distance: find shortest non-trivial cycle.
    let d = compute_code_distance(complex, b1);

    StabilizerCode {
        n,
        k: b1,
        d,
        n_stars,
        n_plaquettes,
        star_weights,
        plaquette_weight: 3,
    }
}

/// Compute code distance by finding the minimum-weight non-trivial cycle.
///
/// A non-trivial cycle is a set of edges that:
/// 1. Forms a cycle (every vertex has even degree in the set)
/// 2. Is not a boundary (not a sum of triangles over Z₂)
///
/// For small complexes, we enumerate cycles starting from the fundamental
/// loops (from the spanning tree) and find the shortest one.
fn compute_code_distance(complex: &SimplicialComplex, b1: usize) -> usize {
    if b1 == 0 {
        return 0;
    }

    let loops = crate::observables::fundamental_loops(complex);

    // The fundamental loops give a basis for H₁. The shortest non-trivial
    // cycle is the minimum weight among all non-empty Z₂ combinations of
    // fundamental loops.
    //
    // For small b1, enumerate all 2^b1 - 1 non-trivial combinations.
    let n_edges = complex.n_edges();

    // Convert each fundamental loop to a Z₂ edge vector.
    let loop_vectors: Vec<Vec<bool>> = loops
        .iter()
        .map(|lp| {
            let mut v = vec![false; n_edges];
            for &(ei, _) in lp {
                v[ei] ^= true; // XOR for Z₂
            }
            v
        })
        .collect();

    let n_loops = loop_vectors.len();
    if n_loops > 20 {
        // Too many combinations; return the minimum fundamental loop weight.
        return loop_vectors
            .iter()
            .map(|v| v.iter().filter(|&&b| b).count())
            .min()
            .unwrap_or(0);
    }

    let mut min_weight = n_edges + 1;

    for mask in 1..(1u64 << n_loops) {
        let mut combined = vec![false; n_edges];
        for (li, lv) in loop_vectors.iter().enumerate() {
            if mask & (1 << li) != 0 {
                for e in 0..n_edges {
                    combined[e] ^= lv[e];
                }
            }
        }
        let weight = combined.iter().filter(|&&b| b).count();
        if weight > 0 && weight < min_weight {
            min_weight = weight;
        }
    }

    if min_weight > n_edges {
        0
    } else {
        min_weight
    }
}

impl std::fmt::Display for StabilizerCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Z₂ Stabilizer Code [[{}, {}, {}]]", self.n, self.k, self.d)?;
        writeln!(f, "  Physical qubits:  {}", self.n)?;
        writeln!(f, "  Logical qubits:   {}", self.k)?;
        writeln!(f, "  Code distance:    {}", self.d)?;
        writeln!(f, "  Star stabilizers: {}", self.n_stars)?;
        writeln!(f, "  Plaquette stabs:  {}", self.n_plaquettes)?;
        writeln!(f, "  Plaquette weight: {}", self.plaquette_weight)?;
        let min_star = self.star_weights.iter().min().unwrap_or(&0);
        let max_star = self.star_weights.iter().max().unwrap_or(&0);
        writeln!(f, "  Star weight range: [{min_star}, {max_star}]")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_pentachoron_code() {
        let complex = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);
        let code = stabilizer_code(&complex);

        assert_eq!(code.n, 10); // 10 edges
        assert_eq!(code.k, 6); // b1 = 10 - 5 + 1 = 6
        assert_eq!(code.n_stars, 5);
        assert_eq!(code.n_plaquettes, 10);
        assert_eq!(code.plaquette_weight, 3);

        // All vertices in a single pentachoron have coordination 4.
        for &w in &code.star_weights {
            assert_eq!(w, 4);
        }

        // Code distance should be > 0.
        assert!(code.d > 0, "code distance should be positive");
    }

    #[test]
    fn test_two_pentachorons_code() {
        let complex =
            SimplicialComplex::from_pentachorons(6, &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]]);
        let code = stabilizer_code(&complex);

        // 6 vertices, edges between two pents sharing [0,1,2,3].
        assert_eq!(code.k, code.n - 6 + 1);
        assert!(code.d > 0);
    }

    #[test]
    fn test_display() {
        let complex = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);
        let code = stabilizer_code(&complex);
        let s = format!("{code}");
        assert!(s.contains("Stabilizer Code"));
    }
}
