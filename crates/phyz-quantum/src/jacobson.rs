//! Jacobson's entanglement equilibrium on a Regge lattice.
//!
//! Tests whether ∂S_EE/∂l_e ∝ ∂S_R/∂l_e — entanglement gradients encode
//! the discrete Einstein equations. This is the numerical realization of
//! Jacobson's entanglement equilibrium (1505.04753) on simplicial geometry.
//!
//! Three experiments:
//! 1. **On-shell equilibrium**: On flat geometry (∂S_R = 0), show ∂S_EE ≈ 0.
//! 2. **Off-shell correlation**: Perturbed geometry → linear correlation between
//!    ∂S_EE/∂l_e and ∂S_R/∂l_e across edges.
//! 3. **RT differential**: ∂S_EE/∂l_e ≈ (1/4G_N) · ∂Area_cut/∂l_e.

use crate::diag;
use crate::ryu_takayanagi::{self, geometry_valid};
use crate::su2_quantum::{
    build_su2_hamiltonian, su2_entanglement_for_partition, su2_lanczos_diagonalize,
    Su2HilbertSpace,
};
use phyz_regge::complex::SimplicialComplex;
use phyz_regge::gauge::metric_weights;
use phyz_regge::geometry::triangle_area_grad_lsq;

/// Configuration for entanglement equilibrium experiments.
pub struct EquilibriumConfig {
    /// Gauge coupling (default 1.0).
    pub g_squared: f64,
    /// Finite-difference step for ∂S_EE/∂l (default 1e-4).
    pub fd_eps: f64,
}

impl Default for EquilibriumConfig {
    fn default() -> Self {
        Self {
            g_squared: 1.0,
            fd_eps: 1e-4,
        }
    }
}

/// Result of the on-shell equilibrium test.
pub struct EquilibriumResult {
    /// max |∂S_EE/∂l_e| (should be ~0 on-shell).
    pub max_ds_ee: f64,
    /// mean |∂S_EE/∂l_e|.
    pub mean_ds_ee: f64,
    /// (∂S_EE/∂l_e, ∂S_R/∂l_e) per edge.
    pub per_edge: Vec<(f64, f64)>,
}

/// Result of the off-shell correlation test.
pub struct CorrelationResult {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    /// (∂S_EE/∂l_e, ∂S_R/∂l_e) pooled across partitions.
    pub per_edge: Vec<(f64, f64)>,
}

/// Result of the RT differential test.
pub struct RtDiffResult {
    /// Slope ≈ 1/(4G_N).
    pub slope: f64,
    pub r_squared: f64,
    /// Extracted G_N = 1/(4·slope).
    pub g_n: f64,
}

/// Build the boundary of the 5-simplex: ∂Δ⁵ = S⁴.
///
/// 6 pentachorons (one per omitted vertex), forming a closed 4-manifold
/// with V=6, E=15, T=20, P=6. Every triangle is shared by exactly 3 pents.
pub fn boundary_5simplex() -> SimplicialComplex {
    SimplicialComplex::from_pentachorons(6, &[
        [1, 2, 3, 4, 5],
        [0, 2, 3, 4, 5],
        [0, 1, 3, 4, 5],
        [0, 1, 2, 4, 5],
        [0, 1, 2, 3, 5],
        [0, 1, 2, 3, 4],
    ])
}

/// Build a subdivided S⁴ triangulation at the given refinement level.
///
/// Stellar subdivision: replace one pentachoron with 5 by adding a vertex at
/// its center. Each level adds +1V, +5E, +4P.
///
/// | Level | V | E | b₁ = E-V+1 | dim = 2^b₁ |
/// |-------|---|---|-------------|------------|
/// | 0     | 6 | 15| 10          | 1,024      |
/// | 1     | 7 | 20| 14          | 16,384     |
/// | 2     | 8 | 25| 18          | 262,144    |
pub fn subdivided_s4(level: usize) -> SimplicialComplex {
    // Start with ∂Δ⁵ pentachorons.
    let mut pents: Vec<[usize; 5]> = vec![
        [1, 2, 3, 4, 5],
        [0, 2, 3, 4, 5],
        [0, 1, 3, 4, 5],
        [0, 1, 2, 4, 5],
        [0, 1, 2, 3, 5],
        [0, 1, 2, 3, 4],
    ];
    let mut n_vertices = 6;

    // Pentachorons to subdivide at each level (by sorted vertex set).
    let targets: &[[usize; 5]] = &[
        [0, 1, 2, 3, 4], // level 1: subdivide with vertex 6
        [0, 2, 3, 4, 5], // level 2: subdivide with vertex 7
        [0, 1, 3, 4, 5], // level 3: subdivide with vertex 8
    ];

    for lv in 0..level.min(targets.len()) {
        let target = targets[lv];
        let new_vertex = n_vertices;
        n_vertices += 1;

        // Find and remove the target pentachoron.
        let pos = pents.iter().position(|p| {
            let mut s = *p;
            s.sort_unstable();
            s == target
        });
        if let Some(idx) = pos {
            pents.remove(idx);
        }

        // Add 5 new pentachorons: one per facet of the original.
        // Each replaces one vertex of the original with new_vertex.
        for skip in 0..5 {
            let mut new_pent = [0usize; 5];
            new_pent[0] = new_vertex;
            let mut idx = 1;
            for (k, &v) in target.iter().enumerate() {
                if k != skip {
                    new_pent[idx] = v;
                    idx += 1;
                }
            }
            pents.push(new_pent);
        }
    }

    SimplicialComplex::from_pentachorons(n_vertices, &pents)
}

/// Project out the conformal mode (uniform component) from a gradient vector.
///
/// Returns (projected_gradient, conformal_component) where the conformal
/// component is the coefficient along the normalized (1,1,...,1) direction.
pub fn project_out_conformal(grad: &[f64]) -> (Vec<f64>, f64) {
    let n = grad.len() as f64;
    let mean = grad.iter().sum::<f64>() / n;
    let projected: Vec<f64> = grad.iter().map(|&g| g - mean).collect();
    (projected, mean * n.sqrt())
}

/// Dimension threshold above which we use Lanczos instead of dense diag.
const LANCZOS_THRESHOLD: usize = 8192;

/// Compute the SU(2) ground state for a given geometry.
///
/// Builds metric weights, constructs the Hamiltonian, and diagonalizes
/// (dense or Lanczos depending on dimension). Returns the ground state vector.
///
/// Use this to avoid redundant solves when computing entropy for multiple partitions.
pub fn su2_ground_state(
    hilbert: &Su2HilbertSpace,
    complex: &SimplicialComplex,
    lengths: &[f64],
    g_squared: f64,
) -> phyz_math::DVec {
    su2_ground_state_with_energy(hilbert, complex, lengths, g_squared).0
}

/// Compute the SU(2) ground state and its energy for a given geometry.
///
/// Like [`su2_ground_state`] but also returns the ground-state energy E₀.
pub fn su2_ground_state_with_energy(
    hilbert: &Su2HilbertSpace,
    complex: &SimplicialComplex,
    lengths: &[f64],
    g_squared: f64,
) -> (phyz_math::DVec, f64) {
    let mw = metric_weights(complex, lengths);
    let spec = if hilbert.dim() > LANCZOS_THRESHOLD {
        su2_lanczos_diagonalize(hilbert, complex, g_squared, Some(&mw), 1, None)
    } else {
        let h = build_su2_hamiltonian(hilbert, complex, g_squared, Some(&mw));
        diag::diagonalize(&h, Some(1))
    };
    (spec.states[0].clone(), spec.energies[0])
}

/// Compute SU(2) ground-state entropy reusing a pre-built Hilbert space.
///
/// Avoids rebuilding the Hilbert space for each FD perturbation (the basis
/// doesn't change when only metric_weights change).
fn su2_ground_state_entropy_with_hs(
    hilbert: &Su2HilbertSpace,
    complex: &SimplicialComplex,
    lengths: &[f64],
    partition_a: &[usize],
    g_squared: f64,
) -> f64 {
    let gs = su2_ground_state(hilbert, complex, lengths, g_squared);
    su2_entanglement_for_partition(hilbert, &gs, complex, partition_a)
}

/// ∂S_EE/∂l_e via central finite differences for SU(2) j=1/2.
///
/// For each edge: perturb l_e ± eps, rebuild metric_weights, rebuild H,
/// rediagonalize, compute S_EE. Returns n_edges-length gradient.
///
/// Builds the Hilbert space once and reuses it across perturbations (the
/// gauge-invariant basis depends only on topology, not metric).
/// Automatically uses Lanczos when dim > 8192.
pub fn entanglement_gradient_su2(
    complex: &SimplicialComplex,
    lengths: &[f64],
    partition_a: &[usize],
    config: &EquilibriumConfig,
) -> Vec<f64> {
    let n_edges = complex.n_edges();
    let eps = config.fd_eps;
    let hs = Su2HilbertSpace::new(complex);
    let mut grad = vec![0.0; n_edges];

    for ei in 0..n_edges {
        let mut l_plus = lengths.to_vec();
        let mut l_minus = lengths.to_vec();
        l_plus[ei] += eps;
        l_minus[ei] -= eps;

        // Guard: skip if perturbed geometry is invalid.
        if !geometry_valid(complex, &l_plus) || !geometry_valid(complex, &l_minus) {
            continue;
        }

        let s_plus =
            su2_ground_state_entropy_with_hs(&hs, complex, &l_plus, partition_a, config.g_squared);
        let s_minus =
            su2_ground_state_entropy_with_hs(&hs, complex, &l_minus, partition_a, config.g_squared);
        grad[ei] = (s_plus - s_minus) / (2.0 * eps);
    }

    grad
}

/// ∂(cut_area_triangles)/∂l_e — analytical via triangle_area_grad_lsq.
///
/// Only boundary-straddling triangles contribute.
pub fn cut_area_gradient(
    complex: &SimplicialComplex,
    partition_a: &[usize],
    lengths: &[f64],
) -> Vec<f64> {
    let n_edges = complex.n_edges();
    let mut grad = vec![0.0; n_edges];

    let mut in_a = vec![false; complex.n_vertices];
    for &v in partition_a {
        in_a[v] = true;
    }

    for (ti, tri) in complex.triangles.iter().enumerate() {
        let count_a = tri.iter().filter(|&&v| in_a[v]).count();
        // Only straddling triangles contribute to the cut area.
        if count_a == 0 || count_a == 3 {
            continue;
        }

        let edge_indices = complex.tri_edge_indices(ti);
        let [a, b, c] = complex.tri_edge_lengths(ti, lengths);
        let da_dlsq = triangle_area_grad_lsq(a, b, c);

        // ∂Area/∂l_e = ∂A/∂(l²) · 2l
        for (k, &ei) in edge_indices.iter().enumerate() {
            grad[ei] += da_dlsq[k] * 2.0 * lengths[ei];
        }
    }

    grad
}

/// Compute entanglement gradient averaged over all non-trivial partitions.
///
/// Returns (mean_gradient, regge_gradient) for correlation analysis.
pub fn averaged_entanglement_gradient(
    complex: &SimplicialComplex,
    lengths: &[f64],
    config: &EquilibriumConfig,
) -> Vec<f64> {
    let partitions = ryu_takayanagi::vertex_bipartitions(complex.n_vertices);
    let n_edges = complex.n_edges();
    let mut avg_grad = vec![0.0; n_edges];
    let n_parts = partitions.len() as f64;

    for part in &partitions {
        let grad = entanglement_gradient_su2(complex, lengths, part, config);
        for (ei, &g) in grad.iter().enumerate() {
            avg_grad[ei] += g / n_parts;
        }
    }

    avg_grad
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_regge::regge::regge_action_grad;

    fn single_pentachoron() -> SimplicialComplex {
        SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]])
    }

    fn two_pentachorons() -> SimplicialComplex {
        SimplicialComplex::from_pentachorons(6, &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]])
    }

    #[test]
    fn test_entanglement_gradient_finite() {
        // FD gradient produces finite values on 1-pentachoron.
        let complex = single_pentachoron();
        let lengths = vec![1.0; complex.n_edges()];
        let config = EquilibriumConfig::default();
        let partition = vec![0, 1];

        let grad = entanglement_gradient_su2(&complex, &lengths, &partition, &config);
        assert_eq!(grad.len(), complex.n_edges());
        for &g in &grad {
            assert!(g.is_finite(), "non-finite gradient: {g}");
        }
    }

    #[test]
    fn test_entanglement_gradient_zero_full_partition() {
        // Full partition → S_EE = 0 always → gradient = 0.
        let complex = single_pentachoron();
        let lengths = vec![1.0; complex.n_edges()];
        let config = EquilibriumConfig::default();
        let all_vertices: Vec<usize> = (0..complex.n_vertices).collect();

        let grad = entanglement_gradient_su2(&complex, &lengths, &all_vertices, &config);
        for (ei, &g) in grad.iter().enumerate() {
            assert!(
                g.abs() < 1e-8,
                "full partition gradient[{ei}] = {g}, expected ~0"
            );
        }
    }

    #[test]
    fn test_cut_area_gradient_vs_fd() {
        // Analytical area gradient matches finite differences.
        let complex = single_pentachoron();
        let lengths = vec![1.0; complex.n_edges()];
        let partition = vec![0, 1];
        let eps = 1e-6;

        let analytical = cut_area_gradient(&complex, &partition, &lengths);

        for ei in 0..complex.n_edges() {
            let mut l_plus = lengths.clone();
            let mut l_minus = lengths.clone();
            l_plus[ei] += eps;
            l_minus[ei] -= eps;

            let a_plus = ryu_takayanagi::cut_area_triangles(&complex, &partition, &l_plus);
            let a_minus = ryu_takayanagi::cut_area_triangles(&complex, &partition, &l_minus);
            let fd = (a_plus - a_minus) / (2.0 * eps);

            assert!(
                (analytical[ei] - fd).abs() < 1e-5,
                "edge {ei}: analytical={}, fd={fd}",
                analytical[ei]
            );
        }
    }

    #[test]
    fn test_regge_grad_zero_on_flat_closed() {
        // Flat geometry on a closed manifold → ∂S_R/∂l_e = 0.
        // Uses periodic hypercubic mesh (closed, no boundary).
        let (complex, lengths) = phyz_regge::mesh::flat_hypercubic(2, 1.0);

        let grad = regge_action_grad(&complex, &lengths);
        for (ei, &g) in grad.iter().enumerate() {
            assert!(
                g.abs() < 1e-8,
                "flat Regge gradient[{ei}] = {g}, expected 0"
            );
        }
    }

    #[test]
    fn test_fd_step_convergence() {
        // Richardson extrapolation: eps vs eps/2 should agree within tolerance.
        let complex = single_pentachoron();
        let lengths = vec![1.0; complex.n_edges()];
        let partition = vec![0, 1];

        let config1 = EquilibriumConfig {
            fd_eps: 2e-4,
            ..Default::default()
        };
        let config2 = EquilibriumConfig {
            fd_eps: 1e-4,
            ..Default::default()
        };

        let grad1 = entanglement_gradient_su2(&complex, &lengths, &partition, &config1);
        let grad2 = entanglement_gradient_su2(&complex, &lengths, &partition, &config2);

        for ei in 0..complex.n_edges() {
            if grad2[ei].abs() > 1e-10 {
                let rel_diff = (grad1[ei] - grad2[ei]).abs() / grad2[ei].abs();
                assert!(
                    rel_diff < 0.1,
                    "edge {ei}: eps convergence failed, grad1={}, grad2={}, rel_diff={rel_diff}",
                    grad1[ei],
                    grad2[ei]
                );
            }
        }
    }

    #[test]
    fn test_cut_area_gradient_zero_interior() {
        // For a partition with no straddling triangles... well, on a pentachoron
        // every partition has straddling triangles. But verify gradient is zero
        // for the full partition (no boundary).
        let complex = single_pentachoron();
        let lengths = vec![1.0; complex.n_edges()];
        let all_verts: Vec<usize> = (0..complex.n_vertices).collect();

        let grad = cut_area_gradient(&complex, &all_verts, &lengths);
        for (ei, &g) in grad.iter().enumerate() {
            assert!(
                g.abs() < 1e-12,
                "full partition area gradient[{ei}] = {g}, expected 0"
            );
        }
    }

    #[test]
    fn test_cut_area_gradient_2pent() {
        // Verify cut area gradient works on 2-pentachoron.
        let complex = two_pentachorons();
        let lengths = vec![1.0; complex.n_edges()];
        let partition = vec![0, 1];

        let grad = cut_area_gradient(&complex, &partition, &lengths);
        assert_eq!(grad.len(), complex.n_edges());

        // At least some edges should have nonzero gradient.
        let nonzero = grad.iter().filter(|&&g| g.abs() > 1e-12).count();
        assert!(
            nonzero > 0,
            "expected some nonzero area gradient entries"
        );
    }

    #[test]
    fn test_boundary_5simplex_structure() {
        let complex = boundary_5simplex();
        assert_eq!(complex.n_vertices, 6);
        assert_eq!(complex.n_edges(), 15);
        assert_eq!(complex.n_triangles(), 20);
        assert_eq!(complex.n_pents(), 6);

        // Every triangle should be shared by exactly 3 pentachorons.
        for (ti, adj) in complex.tri_pent_opposite.iter().enumerate() {
            assert_eq!(
                adj.len(),
                3,
                "triangle {ti} has {} adjacent pents, expected 3",
                adj.len()
            );
        }
    }

    #[test]
    fn test_s4_regge_gradient_conformal() {
        // On equilateral S⁴, the Regge gradient should be uniform (pure conformal)
        // and positive (positive curvature).
        let complex = boundary_5simplex();
        let lengths = vec![1.0; complex.n_edges()];
        let grad = regge_action_grad(&complex, &lengths);

        // All components should be equal by S₆ symmetry.
        let g0 = grad[0];
        assert!(g0 > 0.0, "expected positive Regge gradient, got {g0}");
        for (ei, &g) in grad.iter().enumerate() {
            assert!(
                (g - g0).abs() < 1e-10,
                "Regge gradient not uniform: grad[0]={g0}, grad[{ei}]={g}"
            );
        }
    }

    #[test]
    fn test_s4_entanglement_gradient_conformal_projected() {
        // On equilateral S⁴, ∂S_EE should be purely conformal by symmetry.
        // After projecting out conformal mode, residual should be ~0.
        let complex = boundary_5simplex();
        let lengths = vec![1.0; complex.n_edges()];
        let config = EquilibriumConfig::default();
        let partition = vec![0, 1, 2];

        let grad = entanglement_gradient_su2(&complex, &lengths, &partition, &config);
        let (projected, conformal) = project_out_conformal(&grad);

        // Conformal component should be nonzero.
        assert!(
            conformal.abs() > 1e-8,
            "expected nonzero conformal component, got {conformal}"
        );

        // Projected (non-conformal) residual should be small relative to conformal.
        let max_resid = projected.iter().map(|g| g.abs()).fold(0.0, f64::max);
        let ratio = max_resid / conformal.abs();
        // Partition [0,1,2] breaks S₆ → S₃×S₃, so non-conformal residual isn't
        // exactly zero — just small relative to the conformal component.
        assert!(
            ratio < 0.25,
            "projected residual too large: max|resid|={max_resid:.4e}, |conformal|={:.4e}, ratio={ratio:.4e}",
            conformal.abs()
        );
    }

    #[test]
    fn test_subdivided_s4_level0() {
        let complex = subdivided_s4(0);
        assert_eq!(complex.n_vertices, 6);
        assert_eq!(complex.n_edges(), 15);
        assert_eq!(complex.n_pents(), 6);
        let b1 = complex.n_edges() - complex.n_vertices + 1;
        assert_eq!(b1, 10);
    }

    #[test]
    fn test_subdivided_s4_level1() {
        let complex = subdivided_s4(1);
        assert_eq!(complex.n_vertices, 7);
        assert_eq!(complex.n_edges(), 20);
        assert_eq!(complex.n_pents(), 10); // 6 - 1 + 5 = 10
        let b1 = complex.n_edges() - complex.n_vertices + 1;
        assert_eq!(b1, 14);
    }

    #[test]
    fn test_subdivided_s4_level2() {
        let complex = subdivided_s4(2);
        assert_eq!(complex.n_vertices, 8);
        assert_eq!(complex.n_edges(), 25);
        assert_eq!(complex.n_pents(), 14); // 10 - 1 + 5 = 14
        let b1 = complex.n_edges() - complex.n_vertices + 1;
        assert_eq!(b1, 18);
    }

    #[test]
    fn test_subdivided_s4_level3() {
        let complex = subdivided_s4(3);
        assert_eq!(complex.n_vertices, 9);
        assert_eq!(complex.n_edges(), 30);
        assert_eq!(complex.n_pents(), 18);
        let b1 = complex.n_edges() - complex.n_vertices + 1;
        assert_eq!(b1, 22);
    }

    #[test]
    fn test_subdivided_s4_closed_manifold() {
        // Every triangle should be shared by ≥2 pentachorons (closed manifold).
        for level in 0..=3 {
            let complex = subdivided_s4(level);
            for (ti, adj) in complex.tri_pent_opposite.iter().enumerate() {
                assert!(
                    adj.len() >= 2,
                    "level {level}: triangle {ti} has {} adjacent pents, expected ≥2",
                    adj.len()
                );
            }
        }
    }
}
