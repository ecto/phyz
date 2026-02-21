//! Gauge-invariant Hilbert space for U(1) lattice gauge theory.
//!
//! Wraps the basis enumeration from [`gauss_law`] into a struct with
//! fast config→index lookup via a sorted basis + binary search.

use crate::gauss_law;
use phyz_regge::SimplicialComplex;
use std::collections::HashMap;

/// Gauge-invariant U(1) Hilbert space on a simplicial complex.
///
/// States are labeled by integer quantum numbers `n_e ∈ [-Λ, Λ]` on each edge,
/// subject to Gauss law at every vertex. The basis is enumerated via
/// spanning-tree gauge fixing.
pub struct U1HilbertSpace {
    /// Number of edges in the complex.
    pub n_edges: usize,
    /// Truncation parameter Λ.
    pub lambda: u32,
    /// Gauge-invariant basis states. `basis[i]` = edge quantum numbers for state i.
    pub basis: Vec<Vec<i32>>,
    /// Config → basis index lookup.
    index_map: HashMap<Vec<i32>, usize>,
    /// Non-tree edge indices (free DOF).
    pub free_edges: Vec<usize>,
    /// Tree edge indices (determined by Gauss law).
    pub tree_edges: Vec<usize>,
}

impl U1HilbertSpace {
    /// Build the gauge-invariant Hilbert space.
    pub fn new(complex: &SimplicialComplex, lambda: u32) -> Self {
        let (tree_edges, free_edges) = gauss_law::spanning_tree(complex);
        let basis = gauss_law::enumerate_gauge_invariant(complex, lambda);

        let index_map: HashMap<Vec<i32>, usize> =
            basis.iter().enumerate().map(|(i, c)| (c.clone(), i)).collect();

        Self {
            n_edges: complex.n_edges(),
            lambda,
            basis,
            index_map,
            free_edges,
            tree_edges,
        }
    }

    /// Hilbert space dimension (number of gauge-invariant states).
    pub fn dim(&self) -> usize {
        self.basis.len()
    }

    /// Look up the basis index of a configuration. Returns `None` if the
    /// config is not in the gauge-invariant basis.
    pub fn config_to_index(&self, config: &[i32]) -> Option<usize> {
        self.index_map.get(config).copied()
    }

    /// Get the configuration for basis state `idx`.
    pub fn index_to_config(&self, idx: usize) -> &[i32] {
        &self.basis[idx]
    }

    /// Apply a raising/lowering operator to a configuration.
    ///
    /// Returns the new config and its basis index, or `None` if the
    /// resulting config is outside the truncation or not gauge-invariant.
    pub fn apply_shift(
        &self,
        config: &[i32],
        edge: usize,
        delta: i32,
    ) -> Option<(Vec<i32>, usize)> {
        let mut new_config = config.to_vec();
        new_config[edge] += delta;
        let lam = self.lambda as i32;
        if new_config[edge] < -lam || new_config[edge] > lam {
            return None;
        }
        let idx = self.config_to_index(&new_config)?;
        Some((new_config, idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_pentachoron() -> SimplicialComplex {
        SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]])
    }

    #[test]
    fn test_hilbert_space_basic() {
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);

        assert_eq!(hs.n_edges, 10);
        assert_eq!(hs.lambda, 1);
        assert!(hs.dim() > 0);
        assert!(hs.dim() <= 729); // 3^6

        // b1 = E - V + 1 = 10 - 5 + 1 = 6
        assert_eq!(hs.free_edges.len(), 6);
        assert_eq!(hs.tree_edges.len(), 4);
    }

    #[test]
    fn test_config_index_roundtrip() {
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);

        for i in 0..hs.dim() {
            let config = hs.index_to_config(i);
            let idx = hs.config_to_index(config).unwrap();
            assert_eq!(idx, i);
        }
    }

    #[test]
    fn test_zero_state_present() {
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);

        let zero = vec![0i32; 10];
        assert!(hs.config_to_index(&zero).is_some());
    }

    #[test]
    fn test_dim_grows_with_lambda() {
        let complex = single_pentachoron();
        let d1 = U1HilbertSpace::new(&complex, 1).dim();
        let d2 = U1HilbertSpace::new(&complex, 2).dim();
        let d3 = U1HilbertSpace::new(&complex, 3).dim();

        assert!(d2 > d1, "dim should grow: d1={d1}, d2={d2}");
        assert!(d3 > d2, "dim should grow: d2={d2}, d3={d3}");
    }

    #[test]
    fn test_apply_shift() {
        let complex = single_pentachoron();
        let hs = U1HilbertSpace::new(&complex, 1);

        let zero = vec![0i32; 10];
        let idx0 = hs.config_to_index(&zero).unwrap();

        // Shifting a free edge by +1 should give a valid state (if Gauss law
        // can be satisfied). Shifting a tree edge may not preserve Gauss law.
        // Just test that the method works without panicking.
        for e in 0..10 {
            let _ = hs.apply_shift(&hs.basis[idx0], e, 1);
            let _ = hs.apply_shift(&hs.basis[idx0], e, -1);
        }
    }
}
