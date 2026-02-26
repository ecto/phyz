//! Tent-move time evolution for Lorentzian Regge calculus.
//!
//! Implements Sorkin's tent-move scheme: advance one vertex at a time by
//! solving a local Regge equation via Newton's method.
//!
//! The core function `solve_regge_at_edges` solves the system of Regge
//! equations (∂S/∂s_e = 0) at a specified set of "free" edges using
//! Newton's method with a finite-difference Jacobian and SVD solve
//! (to handle gauge degeneracies).
//!
//! Two evolution modes:
//! - `evolve_slice`: solve for all edges touching a target time slice at once
//! - `tent_move_vertex`: solve for edges in the tent around a single vertex

use crate::complex::SimplicialComplex;
use crate::foliation::FoliatedComplex;
use crate::lorentzian_regge::{
    local_lorentzian_regge_action_grad, local_lorentzian_regge_hessian,
    lorentzian_regge_action_grad,
};
use phyz_math::{DMat, DVec};

/// Configuration for tent-move time evolution.
#[derive(Debug, Clone)]
pub struct TentConfig {
    /// Newton convergence tolerance on residual norm.
    pub newton_tol: f64,
    /// Maximum Newton iterations.
    pub max_newton_iter: usize,
    /// Step size for finite-difference Jacobian.
    pub fd_eps: f64,
    /// SVD tolerance for singular values (gauge modes).
    pub svd_tol: f64,
    /// Use analytical Hessian for the Newton Jacobian (faster for large meshes).
    pub use_analytical_jacobian: bool,
}

impl Default for TentConfig {
    fn default() -> Self {
        Self {
            newton_tol: 1e-10,
            max_newton_iter: 50,
            fd_eps: 1e-7,
            svd_tol: 1e-10,
            use_analytical_jacobian: true,
        }
    }
}

/// Errors during tent-move evolution.
#[derive(Debug)]
pub enum TentMoveError {
    DidNotConverge { residual: f64, iterations: usize },
    SingularJacobian,
}

impl std::fmt::Display for TentMoveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DidNotConverge {
                residual,
                iterations,
            } => write!(
                f,
                "did not converge after {iterations} iterations (residual = {residual:.2e})"
            ),
            Self::SingularJacobian => write!(f, "singular Jacobian"),
        }
    }
}

impl std::error::Error for TentMoveError {}

/// Result of evolving one time step.
#[derive(Debug, Clone)]
pub struct EvolutionResult {
    pub residual: f64,
    pub newton_iters: usize,
}

/// Find edges touching a specific time slice in a foliated complex.
pub fn edges_touching_slice(fc: &FoliatedComplex, slice: usize) -> Vec<usize> {
    fc.complex
        .edges
        .iter()
        .enumerate()
        .filter(|(_, e)| fc.vertex_slice(e[0]) == slice || fc.vertex_slice(e[1]) == slice)
        .map(|(ei, _)| ei)
        .collect()
}

/// Solve the Regge equations at specified free edges via Newton's method.
///
/// Modifies `sq_lengths` in place to satisfy ∂S/∂s_e = 0 at each free edge.
/// Uses SVD for the Newton step to handle gauge degeneracies gracefully.
pub fn solve_regge_at_edges(
    complex: &SimplicialComplex,
    sq_lengths: &mut [f64],
    free_edges: &[usize],
    config: &TentConfig,
) -> Result<EvolutionResult, TentMoveError> {
    let n_free = free_edges.len();
    if n_free == 0 {
        return Ok(EvolutionResult {
            residual: 0.0,
            newton_iters: 0,
        });
    }

    for iter in 0..config.max_newton_iter {
        // Residual: Regge gradient restricted to free edges (local)
        let residual = local_lorentzian_regge_action_grad(complex, sq_lengths, free_edges);
        let res_norm = residual.iter().map(|r| r * r).sum::<f64>().sqrt();

        if res_norm < config.newton_tol {
            return Ok(EvolutionResult {
                residual: res_norm,
                newton_iters: iter,
            });
        }

        // Jacobian: local analytical Hessian or central finite differences
        let jac = if config.use_analytical_jacobian {
            local_lorentzian_regge_hessian(complex, sq_lengths, free_edges, config.fd_eps)
        } else {
            let mut j = DMat::zeros(n_free, n_free);
            for k in 0..n_free {
                let ek = free_edges[k];
                let old = sq_lengths[ek];

                sq_lengths[ek] = old + config.fd_eps;
                let grad_plus = lorentzian_regge_action_grad(complex, sq_lengths);

                sq_lengths[ek] = old - config.fd_eps;
                let grad_minus = lorentzian_regge_action_grad(complex, sq_lengths);

                sq_lengths[ek] = old;

                for (i, &ei) in free_edges.iter().enumerate() {
                    j[(i, k)] = (grad_plus[ei] - grad_minus[ei]) / (2.0 * config.fd_eps);
                }
            }
            j
        };

        // SVD solve: J * delta = -residual (handles gauge null directions)
        let rhs = DVec::from_fn(n_free, |i| -residual[i]);
        let svd = jac.svd(true, true);
        let delta = svd
            .solve(&rhs, config.svd_tol)
            .map_err(|_| TentMoveError::SingularJacobian)?;

        // Backtracking line search on ||residual||²
        let old_res_sq = res_norm * res_norm;
        let mut step = 1.0;
        let mut accepted = false;

        for _ in 0..20 {
            // Trial update
            for (j, &ej) in free_edges.iter().enumerate() {
                sq_lengths[ej] += step * delta[j];
            }

            let trial_grad = local_lorentzian_regge_action_grad(complex, sq_lengths, free_edges);
            let trial_res_sq: f64 = trial_grad.iter().map(|r| r.powi(2)).sum();

            if trial_res_sq < old_res_sq {
                accepted = true;
                break;
            }

            // Revert and shrink step
            for (j, &ej) in free_edges.iter().enumerate() {
                sq_lengths[ej] -= step * delta[j];
            }
            step *= 0.5;
        }

        if !accepted {
            // Accept the full step even if it didn't improve (Newton near convergence)
            for (j, &ej) in free_edges.iter().enumerate() {
                sq_lengths[ej] += step * delta[j];
            }
        }
    }

    // Final residual check
    let residual = local_lorentzian_regge_action_grad(complex, sq_lengths, free_edges);
    let res_norm: f64 = residual.iter().map(|r| r.powi(2)).sum::<f64>().sqrt();
    Err(TentMoveError::DidNotConverge {
        residual: res_norm,
        iterations: config.max_newton_iter,
    })
}

/// Evolve one time step by solving for all edges touching the target slice.
pub fn evolve_slice(
    fc: &FoliatedComplex,
    sq_lengths: &mut [f64],
    target_slice: usize,
    config: &TentConfig,
) -> Result<EvolutionResult, TentMoveError> {
    let free = edges_touching_slice(fc, target_slice);
    solve_regge_at_edges(&fc.complex, sq_lengths, &free, config)
}

/// Evolve multiple time steps sequentially.
pub fn evolve_n_steps(
    fc: &FoliatedComplex,
    sq_lengths: &mut [f64],
    target_slices: &[usize],
    config: &TentConfig,
) -> Result<Vec<EvolutionResult>, TentMoveError> {
    let mut results = Vec::new();
    for &slice in target_slices {
        let result = evolve_slice(fc, sq_lengths, slice, config)?;
        results.push(result);
    }
    Ok(results)
}

/// Find edges in the "tent" around a vertex: the tent pole and tent walls.
///
/// For a vertex v at slice `source_slice`, returns the edges connecting v
/// to the target slice and the edges connecting v's spatial neighbors to
/// the target slice.
pub fn tent_edges_for_vertex(
    fc: &FoliatedComplex,
    vertex: usize,
    target_slice: usize,
) -> Vec<usize> {
    let source_slice = fc.vertex_slice(vertex);
    let target_v = fc.global_vertex(target_slice, fc.vertex_local(vertex));

    let mut tent_edges = Vec::new();

    // Find all edges that connect source-slice vertices (v or its neighbors)
    // to target-slice vertices
    for (ei, e) in fc.complex.edges.iter().enumerate() {
        let s0 = fc.vertex_slice(e[0]);
        let s1 = fc.vertex_slice(e[1]);

        // Edge connects source to target slice
        let connects = (s0 == source_slice && s1 == target_slice)
            || (s1 == source_slice && s0 == target_slice);
        if !connects {
            continue;
        }

        // Check if one endpoint is v or its spatial neighbor,
        // and the other endpoint is target_v or a neighbor's counterpart
        let src_v = if s0 == source_slice { e[0] } else { e[1] };
        let tgt_v = if s0 == target_slice { e[0] } else { e[1] };

        // Include the tent pole (v → target_v) and tent walls
        if src_v == vertex || tgt_v == target_v {
            tent_edges.push(ei);
        }
    }

    // Also include spatial edges at the target slice touching target_v
    for (ei, e) in fc.complex.edges.iter().enumerate() {
        let s0 = fc.vertex_slice(e[0]);
        let s1 = fc.vertex_slice(e[1]);
        if s0 == target_slice && s1 == target_slice && (e[0] == target_v || e[1] == target_v) {
            tent_edges.push(ei);
        }
    }

    tent_edges.sort_unstable();
    tent_edges.dedup();
    tent_edges
}

/// Advance a single vertex via tent-move: solve the local Regge equations.
pub fn tent_move_vertex(
    fc: &FoliatedComplex,
    sq_lengths: &mut [f64],
    vertex: usize,
    target_slice: usize,
    config: &TentConfig,
) -> Result<EvolutionResult, TentMoveError> {
    let free = tent_edges_for_vertex(fc, vertex, target_slice);
    solve_regge_at_edges(&fc.complex, sq_lengths, &free, config)
}

/// Result of amplitude continuation: the full trajectory of converged solutions.
#[derive(Debug, Clone)]
pub struct ContinuationResult {
    /// (amplitude, squared_edge_lengths, residual) at each converged step.
    pub trajectory: Vec<(f64, Vec<f64>, f64)>,
}

/// Errors during continuation.
#[derive(Debug)]
pub enum ContinuationError {
    /// Failed to converge even at the starting amplitude.
    StartFailed(TentMoveError),
    /// Step size became smaller than the minimum without converging.
    StepTooSmall {
        last_converged_amplitude: f64,
        target_amplitude: f64,
    },
}

impl std::fmt::Display for ContinuationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StartFailed(e) => write!(f, "failed at starting amplitude: {e}"),
            Self::StepTooSmall {
                last_converged_amplitude,
                target_amplitude,
            } => write!(
                f,
                "step too small: stuck at A={last_converged_amplitude:.2e}, target={target_amplitude:.2e}"
            ),
        }
    }
}

impl std::error::Error for ContinuationError {}

/// Solve with amplitude continuation: ramp the source amplitude from
/// `a_start` to `a_target` in increments, using each converged solution
/// as initialization for the next step.
///
/// This overcomes Newton divergence at large amplitudes by staying within
/// the linearization basin at each step.
///
/// # Arguments
/// * `fc` - Foliated complex
/// * `flat_sq` - Flat Minkowski squared edge lengths (baseline)
/// * `a_start` - Starting amplitude (should be small enough for easy convergence)
/// * `a_target` - Target amplitude
/// * `n_steps` - Number of continuation steps
/// * `config` - Tent move configuration
/// * `make_source` - Closure that builds a StressEnergy source for a given amplitude
/// * `evolve_fn` - Closure that evolves all tent moves for one amplitude, given
///                  `(sq_lengths, source, config)` → max residual
///
/// # Adaptive stepping
/// If a step fails to converge, the increment is halved and retried.
/// This continues until convergence or the step size drops below
/// `(a_target - a_start) / (n_steps * 1024)`.
pub fn solve_with_continuation<S, F, E>(
    flat_sq: &[f64],
    a_start: f64,
    a_target: f64,
    n_steps: usize,
    make_source: F,
    evolve_fn: E,
) -> Result<ContinuationResult, ContinuationError>
where
    S: crate::matter::StressEnergy,
    F: Fn(f64) -> S,
    E: Fn(&mut [f64], &S, f64) -> Result<f64, TentMoveError>,
{
    let n_steps = n_steps.max(1);
    let base_da = (a_target - a_start) / n_steps as f64;
    let min_da = base_da.abs() / 1024.0;

    let mut trajectory = Vec::with_capacity(n_steps + 1);
    let mut current_sq = flat_sq.to_vec();
    let mut current_a = a_start;

    // Solve at starting amplitude
    let source = make_source(current_a);
    let mut trial_sq = current_sq.clone();
    match evolve_fn(&mut trial_sq, &source, current_a) {
        Ok(residual) => {
            current_sq = trial_sq;
            trajectory.push((current_a, current_sq.clone(), residual));
        }
        Err(e) => return Err(ContinuationError::StartFailed(e)),
    }

    // Ramp to target
    while (current_a - a_target).abs() > min_da * 0.5 {
        let remaining = a_target - current_a;
        let mut da = base_da.copysign(remaining);

        // Don't overshoot
        if da.abs() > remaining.abs() {
            da = remaining;
        }

        let mut converged = false;
        let mut attempts = 0;

        while !converged && da.abs() > min_da * 0.5 {
            attempts += 1;
            let trial_a = current_a + da;
            let source = make_source(trial_a);
            let mut trial_sq = current_sq.clone();

            match evolve_fn(&mut trial_sq, &source, trial_a) {
                Ok(residual) => {
                    current_a = trial_a;
                    current_sq = trial_sq;
                    trajectory.push((current_a, current_sq.clone(), residual));
                    converged = true;
                }
                Err(_) => {
                    // Halve the step and retry
                    da *= 0.5;
                }
            }

            if attempts > 20 {
                break;
            }
        }

        if !converged {
            return Err(ContinuationError::StepTooSmall {
                last_converged_amplitude: current_a,
                target_amplitude: a_target,
            });
        }
    }

    Ok(ContinuationResult { trajectory })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foliation::{flat_minkowski_sq_lengths, foliated_hypercubic};

    /// Flat Minkowski is a fixed point: zero residual at all edges.
    #[test]
    fn test_flat_minkowski_zero_residual() {
        let fc = foliated_hypercubic(4, 2);
        let sq_lengths = flat_minkowski_sq_lengths(&fc, 1.0, 0.3);

        let grad = lorentzian_regge_action_grad(&fc.complex, &sq_lengths);
        let max_grad = grad.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
        assert!(max_grad < 1e-10, "max |grad| = {max_grad:.2e}");
    }

    /// Perturb a small set of edges and recover flat via Newton.
    ///
    /// Uses tent_move_vertex for a single vertex (small local system)
    /// rather than the full slice (too expensive for FD Jacobian).
    #[test]
    fn test_recover_flat_tent_move() {
        let fc = foliated_hypercubic(4, 2);
        let flat_sq = flat_minkowski_sq_lengths(&fc, 1.0, 0.3);
        let mut sq_lengths = flat_sq.clone();

        // Pick a vertex in slice 1 and find its tent edges to slice 2
        let v = fc.global_vertex(1, 0);
        let free = tent_edges_for_vertex(&fc, v, 2);
        eprintln!("tent_move for vertex {v}: {} free edges", free.len());

        // Perturb these edges
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(123);
        for &ei in &free {
            sq_lengths[ei] *= 1.0 + 0.005 * (rng.r#gen::<f64>() - 0.5);
        }

        let config = TentConfig {
            max_newton_iter: 30,
            newton_tol: 1e-8,
            ..TentConfig::default()
        };
        let result = solve_regge_at_edges(&fc.complex, &mut sq_lengths, &free, &config)
            .expect("Newton should converge");

        eprintln!(
            "converged in {} iters, residual = {:.2e}",
            result.newton_iters, result.residual
        );
        assert!(result.residual < 1e-8, "residual = {:.2e}", result.residual);
    }

    /// Gradient vs FD on a perturbed Lorentzian lattice.
    #[test]
    fn test_lorentzian_grad_vs_fd() {
        use crate::lorentzian_regge::lorentzian_regge_action_grad_fd;
        let fc = foliated_hypercubic(2, 2);
        let mut sq_lengths = flat_minkowski_sq_lengths(&fc, 1.0, 0.3);

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        for s in &mut sq_lengths {
            *s *= 1.0 + 0.001 * (rng.r#gen::<f64>() - 0.5);
        }

        let grad_a = lorentzian_regge_action_grad(&fc.complex, &sq_lengths);
        let grad_fd = lorentzian_regge_action_grad_fd(&fc.complex, &sq_lengths, 1e-7);

        let max_grad = grad_fd
            .iter()
            .map(|x| x.abs())
            .fold(0.0f64, f64::max)
            .max(1e-10);
        let max_err = grad_a
            .iter()
            .zip(grad_fd.iter())
            .map(|(a, b)| (a - b).abs() / max_grad)
            .fold(0.0f64, f64::max);

        assert!(max_err < 0.05, "max gradient error = {max_err:.2e}");
    }

    /// Evolve_slice wrapper works.
    #[test]
    fn test_evolve_slice_flat() {
        let fc = foliated_hypercubic(4, 2);
        let mut sq_lengths = flat_minkowski_sq_lengths(&fc, 1.0, 0.3);

        // Already flat → should converge in 0 iterations
        let config = TentConfig::default();
        let result = evolve_slice(&fc, &mut sq_lengths, 2, &config).expect("should converge");
        assert_eq!(result.newton_iters, 0);
    }

    /// Both analytical and FD Jacobian paths converge on the same problem.
    #[test]
    fn test_analytical_vs_fd_convergence() {
        let fc = foliated_hypercubic(4, 2);
        let flat_sq = flat_minkowski_sq_lengths(&fc, 1.0, 0.3);

        let v = fc.global_vertex(1, 0);
        let free = tent_edges_for_vertex(&fc, v, 2);

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        // Analytical path
        let mut sq_analytical = flat_sq.clone();
        let mut rng = StdRng::seed_from_u64(200);
        for &ei in &free {
            sq_analytical[ei] *= 1.0 + 0.005 * (rng.r#gen::<f64>() - 0.5);
        }

        let config_a = TentConfig {
            max_newton_iter: 30,
            newton_tol: 1e-8,
            use_analytical_jacobian: true,
            ..TentConfig::default()
        };
        let result_a = solve_regge_at_edges(&fc.complex, &mut sq_analytical, &free, &config_a)
            .expect("analytical should converge");

        // FD path (same initial perturbation)
        let mut sq_fd = flat_sq.clone();
        let mut rng2 = StdRng::seed_from_u64(200);
        for &ei in &free {
            sq_fd[ei] *= 1.0 + 0.005 * (rng2.r#gen::<f64>() - 0.5);
        }

        let config_fd = TentConfig {
            max_newton_iter: 30,
            newton_tol: 1e-8,
            use_analytical_jacobian: false,
            ..TentConfig::default()
        };
        let result_fd = solve_regge_at_edges(&fc.complex, &mut sq_fd, &free, &config_fd)
            .expect("FD should converge");

        eprintln!(
            "analytical: {} iters, res={:.2e}; FD: {} iters, res={:.2e}",
            result_a.newton_iters, result_a.residual, result_fd.newton_iters, result_fd.residual
        );

        // Both should converge; iteration counts may differ by ±1-2
        assert!(result_a.residual < 1e-8);
        assert!(result_fd.residual < 1e-8);

        // Solutions should match
        let max_diff: f64 = sq_analytical
            .iter()
            .zip(sq_fd.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-4, "solutions differ: max_diff={max_diff:.2e}");
    }
}
