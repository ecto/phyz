//! Gradient descent solver for Einstein-Yang-Mills classical solutions.
//!
//! Finds on-shell configurations (local minima of the action) using
//! manifold-aware gradient descent with Armijo backtracking line search.
//!
//! **Update rules** respect the geometry of each DOF:
//! - Lengths (R⁺): `l' = l · exp(-step · g)` — guarantees positivity
//! - SU(2) (unit quaternions): `U' = exp(-step · g) · U` — left-invariant retraction
//!
//! Convergence is checked after projecting out gauge flat directions
//! (48 for SU(2) on n=2) to avoid stalling.

use crate::complex::SimplicialComplex;
use crate::su2::Su2;
use crate::symmetry::{Generator, orthonormalize, project_out_span};
use crate::yang_mills::{
    all_su2_gauge_generators, einstein_yang_mills_action, einstein_yang_mills_grad,
};

/// Solver configuration.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum number of gradient descent iterations.
    pub max_iter: usize,
    /// Convergence tolerance on (gauge-projected) gradient norm.
    pub grad_tol: f64,
    /// Initial step size for backtracking line search.
    pub initial_step: f64,
    /// Step shrink factor for Armijo backtracking.
    pub shrink: f64,
    /// Armijo sufficient decrease parameter.
    pub armijo_c: f64,
    /// Minimum step size before declaring failure.
    pub min_step: f64,
    /// Print progress every N iterations (0 = silent).
    pub print_every: usize,
    /// Renormalize SU(2) quaternions every N steps.
    pub renorm_every: usize,
    /// Maximum allowed edge length ratio max(l)/min(l) to prevent degenerate simplices.
    /// Set to 0.0 to disable.
    pub max_length_ratio: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iter: 5000,
            grad_tol: 1e-8,
            initial_step: 1e-3,
            shrink: 0.5,
            armijo_c: 1e-4,
            min_step: 1e-14,
            print_every: 0,
            renorm_every: 50,
            max_length_ratio: 3.0,
        }
    }
}

/// Result of the gradient descent solver.
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// Optimized edge lengths.
    pub lengths: Vec<f64>,
    /// Optimized SU(2) group elements.
    pub elements: Vec<Su2>,
    /// Final action value.
    pub action: f64,
    /// Final (gauge-projected) gradient norm.
    pub grad_norm: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the solver converged (grad_norm < grad_tol).
    pub converged: bool,
    /// Action history: (iteration, action) at logged steps.
    pub action_history: Vec<(usize, f64)>,
}

/// Apply a gradient step with manifold-aware updates.
///
/// - Lengths: `l' = l · exp(-step · g_l)` (stays positive)
/// - SU(2): `U' = exp(-step · g_su2) · U` (left-invariant retraction)
fn apply_update(
    lengths: &[f64],
    elements: &[Su2],
    grad: &[f64],
    step: f64,
    n_edges: usize,
) -> (Vec<f64>, Vec<Su2>) {
    let new_lengths: Vec<f64> = lengths
        .iter()
        .enumerate()
        .map(|(i, &l)| l * (-step * grad[i]).exp())
        .collect();

    let new_elements: Vec<Su2> = elements
        .iter()
        .enumerate()
        .map(|(i, u)| {
            let delta = [
                -step * grad[n_edges + 3 * i],
                -step * grad[n_edges + 3 * i + 1],
                -step * grad[n_edges + 3 * i + 2],
            ];
            Su2::exp(&delta).mul(u)
        })
        .collect();

    (new_lengths, new_elements)
}

/// Rebuild the orthonormal gauge + conformal basis from current configuration.
///
/// Projects out both gauge flat directions (field-dependent for SU(2)) and
/// the conformal mode (global length scaling) which is unbounded below in
/// the Regge action.
fn rebuild_flat_basis(
    complex: &SimplicialComplex,
    lengths: &[f64],
    elements: &[Su2],
) -> Vec<Generator> {
    let mut gens = all_su2_gauge_generators(complex, elements);
    // Add conformal mode (pure length scaling, no field component).
    gens.push(crate::yang_mills::su2_conformal_generator(complex, lengths));
    orthonormalize(&gens)
}

/// Minimize the Einstein-Yang-Mills action via gradient descent.
///
/// Uses Armijo backtracking line search with manifold-aware updates.
/// Convergence is measured after projecting out gauge directions.
pub fn minimize_einstein_yang_mills(
    complex: &SimplicialComplex,
    lengths: &[f64],
    elements: &[Su2],
    alpha: f64,
    config: &SolverConfig,
) -> SolverResult {
    let n_edges = complex.n_edges();
    let mut cur_lengths = lengths.to_vec();
    let mut cur_elements = elements.to_vec();
    let mut cur_action = einstein_yang_mills_action(complex, &cur_lengths, &cur_elements, alpha);
    let mut step = config.initial_step;
    let mut action_history = vec![(0, cur_action)];

    let mut iterations = 0;
    let mut grad_norm = f64::MAX;

    for iter in 1..=config.max_iter {
        iterations = iter;

        // Compute gradient.
        let grad = einstein_yang_mills_grad(complex, &cur_lengths, &cur_elements, alpha);

        // Compute gauge-projected gradient norm for convergence check.
        // Rebuild gauge basis from current elements (field-dependent).
        let ortho_gauge = rebuild_flat_basis(complex, &cur_lengths, &cur_elements);
        let mut projected = grad.clone();
        project_out_span(&mut projected, &ortho_gauge);
        grad_norm = projected.iter().map(|x| x * x).sum::<f64>().sqrt();

        if config.print_every > 0 && iter % config.print_every == 0 {
            eprintln!(
                "  iter {iter:5}: S = {cur_action:.6e}, |grad| = {grad_norm:.2e}, step = {step:.2e}"
            );
        }

        // Check convergence.
        if grad_norm < config.grad_tol {
            action_history.push((iter, cur_action));
            break;
        }

        // Use projected gradient as search direction (gauge-free).
        // Directional derivative along negative projected gradient.
        let dir_deriv: f64 = -projected.iter().map(|x| x * x).sum::<f64>();

        // Armijo backtracking.
        let mut accepted = false;
        let mut trial_step = step;
        for _ in 0..60 {
            let (trial_lengths, trial_elements) =
                apply_update(&cur_lengths, &cur_elements, &projected, trial_step, n_edges);

            // Reject degenerate configurations.
            if config.max_length_ratio > 0.0 {
                let l_min = trial_lengths.iter().copied().fold(f64::INFINITY, f64::min);
                let l_max = trial_lengths.iter().copied().fold(0.0_f64, f64::max);
                if l_min > 0.0 && l_max / l_min > config.max_length_ratio {
                    trial_step *= config.shrink;
                    if trial_step < config.min_step {
                        break;
                    }
                    continue;
                }
            }

            let trial_action =
                einstein_yang_mills_action(complex, &trial_lengths, &trial_elements, alpha);

            if trial_action <= cur_action + config.armijo_c * trial_step * dir_deriv {
                cur_lengths = trial_lengths;
                cur_elements = trial_elements;
                cur_action = trial_action;
                // Try to grow step for next iteration.
                step = trial_step * 1.5;
                accepted = true;
                break;
            }

            trial_step *= config.shrink;
            if trial_step < config.min_step {
                break;
            }
        }

        if !accepted {
            // Step too small — likely at a minimum or saddle.
            action_history.push((iter, cur_action));
            break;
        }

        // Renormalize SU(2) quaternions periodically.
        if iter % config.renorm_every == 0 {
            for u in &mut cur_elements {
                *u = u.normalize();
            }
            action_history.push((iter, cur_action));
        }
    }

    let converged = grad_norm < config.grad_tol;

    SolverResult {
        lengths: cur_lengths,
        elements: cur_elements,
        action: cur_action,
        grad_norm,
        iterations,
        converged,
        action_history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh;

    #[test]
    fn test_flat_vacuum_is_minimum() {
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);
        let elements = vec![Su2::identity(); complex.n_edges()];
        let alpha = 1.0;

        let grad = einstein_yang_mills_grad(&complex, &lengths, &elements, alpha);
        let norm: f64 = grad.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            norm < 1e-10,
            "flat vacuum gradient norm = {norm} (expected ~0)"
        );
    }

    #[test]
    fn test_solver_decreases_action() {
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);
        let n_edges = complex.n_edges();

        // Perturb away from flat.
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        let perturbed_lengths: Vec<f64> = lengths
            .iter()
            .map(|&l| l * (1.0 + 0.01 * (2.0 * rng.r#gen::<f64>() - 1.0)))
            .collect();
        let elements: Vec<Su2> = (0..n_edges)
            .map(|_| {
                Su2::exp(&[
                    rng.r#gen::<f64>() * 0.02 - 0.01,
                    rng.r#gen::<f64>() * 0.02 - 0.01,
                    rng.r#gen::<f64>() * 0.02 - 0.01,
                ])
            })
            .collect();

        let alpha = 1.0;
        let initial_action =
            einstein_yang_mills_action(&complex, &perturbed_lengths, &elements, alpha);

        let config = SolverConfig {
            max_iter: 100,
            grad_tol: 1e-12,
            ..SolverConfig::default()
        };
        let result =
            minimize_einstein_yang_mills(&complex, &perturbed_lengths, &elements, alpha, &config);

        assert!(
            result.action <= initial_action + 1e-15,
            "action should decrease: {initial_action:.6e} -> {:.6e}",
            result.action
        );
    }

    #[test]
    fn test_solver_terminates_and_decreases() {
        // The Regge action is unbounded below (conformal mode problem),
        // so on flat space the solver won't converge to grad=0. But it
        // should decrease the action and terminate when the step gets
        // too small, without panicking or producing NaN.
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);
        let n_edges = complex.n_edges();

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(123);
        let perturbed_lengths: Vec<f64> = lengths
            .iter()
            .map(|&l| l * (1.0 + 0.005 * (2.0 * rng.r#gen::<f64>() - 1.0)))
            .collect();
        let elements: Vec<Su2> = (0..n_edges)
            .map(|_| {
                Su2::exp(&[
                    rng.r#gen::<f64>() * 0.01 - 0.005,
                    rng.r#gen::<f64>() * 0.01 - 0.005,
                    rng.r#gen::<f64>() * 0.01 - 0.005,
                ])
            })
            .collect();

        let initial_action = crate::yang_mills::einstein_yang_mills_action(
            &complex,
            &perturbed_lengths,
            &elements,
            1.0,
        );

        let config = SolverConfig {
            max_iter: 500,
            ..SolverConfig::default()
        };
        let result =
            minimize_einstein_yang_mills(&complex, &perturbed_lengths, &elements, 1.0, &config);

        assert!(
            result.action <= initial_action + 1e-15,
            "action should not increase"
        );
        assert!(!result.action.is_nan(), "action should not be NaN");
        assert!(!result.grad_norm.is_nan(), "grad_norm should not be NaN");
    }

    #[test]
    fn test_length_positivity() {
        let (complex, lengths) = mesh::flat_hypercubic(2, 1.0);
        let n_edges = complex.n_edges();

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(456);
        let perturbed_lengths: Vec<f64> = lengths
            .iter()
            .map(|&l| l * (1.0 + 0.05 * (2.0 * rng.r#gen::<f64>() - 1.0)))
            .collect();
        let elements: Vec<Su2> = (0..n_edges)
            .map(|_| {
                Su2::exp(&[
                    rng.r#gen::<f64>() * 0.1 - 0.05,
                    rng.r#gen::<f64>() * 0.1 - 0.05,
                    rng.r#gen::<f64>() * 0.1 - 0.05,
                ])
            })
            .collect();

        let config = SolverConfig {
            max_iter: 200,
            ..SolverConfig::default()
        };
        let result =
            minimize_einstein_yang_mills(&complex, &perturbed_lengths, &elements, 1.0, &config);

        for (i, &l) in result.lengths.iter().enumerate() {
            assert!(l > 0.0, "edge {i} has non-positive length {l}");
        }
    }
}
