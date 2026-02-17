//! Combined Einstein-Maxwell action and gradient.
//!
//! The full action is:
//!
//!   S[l, θ] = S_R[l] + α · S_M[l, θ]
//!
//! where:
//!   S_R = Σ_t A_t δ_t         (Regge / Einstein-Hilbert)
//!   S_M = (1/2) Σ_t F_t² W_t  (discrete Maxwell on curved background)
//!   α   = coupling constant (8πG/c⁴ in physical units)
//!
//! The coupling enters through:
//! - l_e (edge lengths) appearing in both S_R and S_M
//! - W_t (metric weights in S_M) depending on l_e
//!
//! Varying w.r.t. l_e gives the discrete Einstein equations with EM source:
//!   ∂S/∂l_e = ∂S_R/∂l_e + α · ∂S_M/∂l_e = 0
//!
//! Varying w.r.t. θ_e gives discrete Maxwell equations on curved background:
//!   ∂S/∂θ_e = α · ∂S_M/∂θ_e = 0

use crate::complex::SimplicialComplex;
use crate::gauge;
use crate::regge;

/// Combined Einstein-Maxwell field configuration.
#[derive(Debug, Clone)]
pub struct Fields {
    /// Edge lengths (metric degrees of freedom).
    pub lengths: Vec<f64>,
    /// Edge phases (U(1) gauge field degrees of freedom).
    pub phases: Vec<f64>,
}

impl Fields {
    /// Create from separate length and phase arrays.
    pub fn new(lengths: Vec<f64>, phases: Vec<f64>) -> Self {
        Self { lengths, phases }
    }

    /// Pack into a single flat vector: [lengths..., phases...].
    pub fn pack(&self) -> Vec<f64> {
        let mut v = self.lengths.clone();
        v.extend_from_slice(&self.phases);
        v
    }

    /// Unpack from a flat vector.
    pub fn unpack(flat: &[f64], n_edges: usize) -> Self {
        Self {
            lengths: flat[..n_edges].to_vec(),
            phases: flat[n_edges..].to_vec(),
        }
    }

    /// Total number of degrees of freedom.
    pub fn n_dof(&self) -> usize {
        self.lengths.len() + self.phases.len()
    }
}

/// Parameters for the Einstein-Maxwell action.
#[derive(Debug, Clone)]
pub struct ActionParams {
    /// EM coupling constant α = 8πG/c⁴ (or 1.0 in geometric units).
    pub alpha: f64,
    /// Finite-difference step for length gradients of S_M.
    pub fd_eps: f64,
}

impl Default for ActionParams {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fd_eps: 1e-7,
        }
    }
}

/// Compute the full Einstein-Maxwell action.
///
///   S = S_R + α · S_M
pub fn einstein_maxwell_action(
    complex: &SimplicialComplex,
    fields: &Fields,
    params: &ActionParams,
) -> f64 {
    let s_r = regge::regge_action(complex, &fields.lengths);
    let s_m = gauge::maxwell_action(complex, &fields.lengths, &fields.phases);
    s_r + params.alpha * s_m
}

/// Gradient of the full action w.r.t. all degrees of freedom.
///
/// Returns a packed vector: [∂S/∂l₁, ..., ∂S/∂l_n, ∂S/∂θ₁, ..., ∂S/∂θ_n].
pub fn einstein_maxwell_grad(
    complex: &SimplicialComplex,
    fields: &Fields,
    params: &ActionParams,
) -> Vec<f64> {
    let n_edges = complex.n_edges();

    // ∂S/∂l_e = ∂S_R/∂l_e + α · ∂S_M/∂l_e
    let grad_r = regge::regge_action_grad(complex, &fields.lengths);
    let grad_m_lengths =
        gauge::maxwell_action_grad_lengths(complex, &fields.lengths, &fields.phases);

    // ∂S/∂θ_e = α · ∂S_M/∂θ_e
    let grad_m_phases =
        gauge::maxwell_action_grad_phases(complex, &fields.lengths, &fields.phases);

    let mut grad = Vec::with_capacity(2 * n_edges);

    // Length gradients.
    for i in 0..n_edges {
        grad.push(grad_r[i] + params.alpha * grad_m_lengths[i]);
    }

    // Phase gradients.
    for i in 0..n_edges {
        grad.push(params.alpha * grad_m_phases[i]);
    }

    grad
}

/// Compute action variation under a field transformation.
///
/// |S[T(φ)] - S[φ]|
///
/// This is the key quantity for the symmetry search: we look for
/// transformations T that make this small.
pub fn action_variation(
    complex: &SimplicialComplex,
    original: &Fields,
    transformed: &Fields,
    params: &ActionParams,
) -> f64 {
    let s_original = einstein_maxwell_action(complex, original, params);
    let s_transformed = einstein_maxwell_action(complex, transformed, params);
    (s_transformed - s_original).abs()
}

/// Apply an infinitesimal Lie-algebra transformation to the fields.
///
/// The transformation is parameterized as:
///   δl_e = Σ_a ε_a · G^l_{a,e}(l, θ)
///   δθ_e = Σ_a ε_a · G^θ_{a,e}(l, θ)
///
/// where G^l and G^θ are the generator actions on lengths and phases,
/// and ε_a are the transformation parameters.
///
/// For the symmetry search, G is what we're trying to find.
pub fn apply_infinitesimal_transform(
    fields: &Fields,
    generator_lengths: &[Vec<f64>],
    generator_phases: &[Vec<f64>],
    epsilons: &[f64],
) -> Fields {
    let n_edges = fields.lengths.len();
    let n_generators = epsilons.len();

    let mut new_lengths = fields.lengths.clone();
    let mut new_phases = fields.phases.clone();

    for a in 0..n_generators {
        for e in 0..n_edges {
            new_lengths[e] += epsilons[a] * generator_lengths[a][e];
            new_phases[e] += epsilons[a] * generator_phases[a][e];
        }
    }

    Fields::new(new_lengths, new_phases)
}

/// Compute the Noether current associated with a symmetry generator.
///
/// For an infinitesimal symmetry with generator (G^l, G^θ), the
/// Noether current (on-shell) is:
///
///   J = Σ_e (∂S/∂l_e · G^l_e + ∂S/∂θ_e · G^θ_e)
///
/// This should be zero (or constant) if the transformation is an exact symmetry.
pub fn noether_current(
    complex: &SimplicialComplex,
    fields: &Fields,
    generator_lengths: &[f64],
    generator_phases: &[f64],
    params: &ActionParams,
) -> f64 {
    let grad = einstein_maxwell_grad(complex, fields, params);
    let n_edges = complex.n_edges();

    let mut current = 0.0;
    for e in 0..n_edges {
        current += grad[e] * generator_lengths[e]; // ∂S/∂l_e · G^l_e
        current += grad[n_edges + e] * generator_phases[e]; // ∂S/∂θ_e · G^θ_e
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh;

    #[test]
    fn test_flat_vacuum_action() {
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);
        let phases = vec![0.0; complex.n_edges()];
        let fields = Fields::new(lengths, phases);
        let params = ActionParams::default();

        let action = einstein_maxwell_action(&complex, &fields, &params);
        assert!(
            action.abs() < 1e-8,
            "flat vacuum action = {action} (expected ~0)"
        );
    }

    #[test]
    fn test_gauge_invariance_full_action() {
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let phases: Vec<f64> = (0..complex.n_edges())
            .map(|_| rng.r#gen::<f64>() * 0.3)
            .collect();

        let fields = Fields::new(lengths.clone(), phases.clone());
        let params = ActionParams::default();
        let s_original = einstein_maxwell_action(&complex, &fields, &params);

        // Apply gauge transformation.
        let gauge: Vec<f64> = (0..complex.n_vertices)
            .map(|_| rng.r#gen::<f64>() * 2.0 * std::f64::consts::PI)
            .collect();

        let mut phases_t = phases.clone();
        for (ei, edge) in complex.edges.iter().enumerate() {
            phases_t[ei] += gauge[edge[1]] - gauge[edge[0]];
        }

        let fields_t = Fields::new(lengths, phases_t);
        let s_transformed = einstein_maxwell_action(&complex, &fields_t, &params);

        assert!(
            (s_original - s_transformed).abs() < 1e-10,
            "full action not gauge-invariant: {s_original} vs {s_transformed}"
        );
    }

    #[test]
    fn test_gradient_dimensions() {
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);
        let phases = vec![0.0; complex.n_edges()];
        let fields = Fields::new(lengths, phases);
        let params = ActionParams::default();

        let grad = einstein_maxwell_grad(&complex, &fields, &params);
        assert_eq!(grad.len(), 2 * complex.n_edges());
    }

    #[test]
    fn test_pack_unpack() {
        let fields = Fields::new(vec![1.0, 2.0, 3.0], vec![0.1, 0.2, 0.3]);
        let packed = fields.pack();
        assert_eq!(packed, vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3]);

        let unpacked = Fields::unpack(&packed, 3);
        assert_eq!(unpacked.lengths, fields.lengths);
        assert_eq!(unpacked.phases, fields.phases);
    }
}
