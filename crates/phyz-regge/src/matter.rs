//! Stress-energy source terms for the Regge equations.
//!
//! Provides matter coupling to the tent-move evolution via the discrete
//! Einstein equations: ∂S_R/∂s_e = −8π T_e.
//!
//! The source term T_e represents the integrated stress-energy along edge e,
//! obtained by contracting the continuous T^{μν} with the edge's dual volume.

use crate::complex::SimplicialComplex;
use crate::lorentzian_regge::{
    local_lorentzian_regge_action_grad, local_lorentzian_regge_hessian,
    lorentzian_regge_action_grad,
};
use crate::tent_move::{EvolutionResult, TentConfig, TentMoveError};

/// Trait for stress-energy sources on a simplicial lattice.
///
/// Implementations provide the source term T_e for each edge, which enters
/// the Regge equations as:  ∂S_R/∂s_e + 8π T_e = 0.
pub trait StressEnergy {
    /// Compute the source term for each edge.
    ///
    /// Returns a vector of length `n_edges`, where T_e is the integrated
    /// stress-energy contribution at edge e.
    fn edge_sources(&self, complex: &SimplicialComplex, sq_lengths: &[f64]) -> Vec<f64>;
}

/// A point particle following a specified worldline.
#[derive(Debug, Clone)]
pub struct PointParticle {
    /// Rest mass of the particle.
    pub mass: f64,
    /// Worldline as a sequence of (slice_index, local_vertex) pairs.
    pub worldline: Vec<(usize, usize)>,
}

/// A circulating mass current loop (for the gravitomagnetic transformer primary).
///
/// Models a steady mass-energy flow around a loop of vertices within a spatial slice.
/// The mass rate is dm/dt (mass per unit coordinate time crossing any point).
#[derive(Debug, Clone)]
pub struct MassCurrentLoop {
    /// Mass flow rate: dm/dt (in geometric units).
    pub mass_rate: f64,
    /// Ordered list of global vertex indices forming the loop.
    pub loop_vertices: Vec<usize>,
    /// Lorentz factor γ of the flowing mass.
    pub gamma: f64,
}

/// Distributed stress-energy on the lattice.
///
/// Directly specifies the momentum flux and energy density at each edge/vertex.
#[derive(Debug, Clone)]
pub struct DistributedSource {
    /// Momentum flux per edge: T^{0i} integrated over the dual area.
    /// Indexed by edge index.
    pub momentum_flux: Vec<f64>,
    /// Energy density per vertex: T^{00} integrated over the dual volume.
    /// Indexed by vertex index.
    pub energy_density: Vec<f64>,
}

impl StressEnergy for PointParticle {
    fn edge_sources(&self, complex: &SimplicialComplex, sq_lengths: &[f64]) -> Vec<f64> {
        let mut sources = vec![0.0; complex.n_edges()];

        // A point particle contributes T_e = m / (2 * |s_e|^{1/2}) along its worldline.
        // For each consecutive pair of worldline points connected by an edge,
        // the particle's mass-energy is distributed to that edge.
        for window in self.worldline.windows(2) {
            let (_, v0) = window[0];
            let (_, v1) = window[1];
            let edge = if v0 < v1 { [v0, v1] } else { [v1, v0] };
            if let Some(&ei) = complex.edge_index.get(&edge) {
                let s_e = sq_lengths[ei];
                let abs_len = s_e.abs().sqrt().max(1e-30);
                // Source coupling: T_e ~ m / L_e (mass per proper length)
                sources[ei] += self.mass / abs_len;
            }
        }

        sources
    }
}

impl StressEnergy for MassCurrentLoop {
    fn edge_sources(&self, complex: &SimplicialComplex, sq_lengths: &[f64]) -> Vec<f64> {
        let mut sources = vec![0.0; complex.n_edges()];
        let n = self.loop_vertices.len();
        if n < 2 {
            return sources;
        }

        // Each edge in the loop carries momentum flux proportional to the mass rate.
        // T^{0i} ~ γ² * mass_rate * v_i, distributed along the edge.
        for i in 0..n {
            let v0 = self.loop_vertices[i];
            let v1 = self.loop_vertices[(i + 1) % n];
            let edge = if v0 < v1 { [v0, v1] } else { [v1, v0] };
            if let Some(&ei) = complex.edge_index.get(&edge) {
                let s_e = sq_lengths[ei];
                let abs_len = s_e.abs().sqrt().max(1e-30);
                // T_e ~ γ² * mass_rate / L_e (momentum per proper length)
                sources[ei] += self.gamma * self.gamma * self.mass_rate / abs_len;
            }
        }

        sources
    }
}

impl StressEnergy for DistributedSource {
    fn edge_sources(&self, complex: &SimplicialComplex, _sq_lengths: &[f64]) -> Vec<f64> {
        let mut sources = vec![0.0; complex.n_edges()];

        // Direct momentum flux per edge
        for (ei, &flux) in self.momentum_flux.iter().enumerate() {
            if ei < sources.len() {
                sources[ei] += flux;
            }
        }

        // Distribute energy density from vertices to their edges
        for (vi, &rho) in self.energy_density.iter().enumerate() {
            if rho.abs() < 1e-30 {
                continue;
            }
            // Find edges touching this vertex, distribute equally
            let mut n_edges_at_v = 0;
            for e in &complex.edges {
                if e[0] == vi || e[1] == vi {
                    n_edges_at_v += 1;
                }
            }
            if n_edges_at_v == 0 {
                continue;
            }
            let per_edge = rho / n_edges_at_v as f64;
            for (ei, e) in complex.edges.iter().enumerate() {
                if e[0] == vi || e[1] == vi {
                    sources[ei] += per_edge;
                }
            }
        }

        sources
    }
}

/// Solve the Regge equations with matter source terms.
///
/// Instead of ∂S/∂s_e = 0, solves: ∂S/∂s_e + 8π T_e = 0 at each free edge.
pub fn solve_regge_with_source(
    complex: &SimplicialComplex,
    sq_lengths: &mut [f64],
    free_edges: &[usize],
    source: &dyn StressEnergy,
    config: &TentConfig,
) -> Result<EvolutionResult, TentMoveError> {
    let n_free = free_edges.len();
    if n_free == 0 {
        return Ok(EvolutionResult {
            residual: 0.0,
            newton_iters: 0,
        });
    }

    let eight_pi = 8.0 * std::f64::consts::PI;

    for iter in 0..config.max_newton_iter {
        // Residual: ∂S/∂s_e + 8π T_e (local Regge grad + full matter sources)
        let regge_grad = local_lorentzian_regge_action_grad(complex, sq_lengths, free_edges);
        let matter_sources = source.edge_sources(complex, sq_lengths);
        let residual: Vec<f64> = regge_grad
            .iter()
            .zip(free_edges.iter())
            .map(|(&rg, &ei)| rg + eight_pi * matter_sources[ei])
            .collect();
        let res_norm = residual.iter().map(|r| r * r).sum::<f64>().sqrt();

        if res_norm < config.newton_tol {
            return Ok(EvolutionResult {
                residual: res_norm,
                newton_iters: iter,
            });
        }

        // Jacobian: local analytical Hessian for geometry + FD for matter coupling
        let jac = if config.use_analytical_jacobian {
            let mut j =
                local_lorentzian_regge_hessian(complex, sq_lengths, free_edges, config.fd_eps);
            // Add matter Jacobian via FD (typically cheap)
            for k in 0..n_free {
                let ek = free_edges[k];
                let old = sq_lengths[ek];

                sq_lengths[ek] = old + config.fd_eps;
                let src_plus = source.edge_sources(complex, sq_lengths);

                sq_lengths[ek] = old - config.fd_eps;
                let src_minus = source.edge_sources(complex, sq_lengths);

                sq_lengths[ek] = old;

                for (i, &ei) in free_edges.iter().enumerate() {
                    j[(i, k)] += eight_pi * (src_plus[ei] - src_minus[ei]) / (2.0 * config.fd_eps);
                }
            }
            j
        } else {
            let mut j = phyz_math::DMat::zeros(n_free, n_free);
            for k in 0..n_free {
                let ek = free_edges[k];
                let old = sq_lengths[ek];

                sq_lengths[ek] = old + config.fd_eps;
                let grad_plus = lorentzian_regge_action_grad(complex, sq_lengths);
                let src_plus = source.edge_sources(complex, sq_lengths);

                sq_lengths[ek] = old - config.fd_eps;
                let grad_minus = lorentzian_regge_action_grad(complex, sq_lengths);
                let src_minus = source.edge_sources(complex, sq_lengths);

                sq_lengths[ek] = old;

                for (i, &ei) in free_edges.iter().enumerate() {
                    let d_regge = (grad_plus[ei] - grad_minus[ei]) / (2.0 * config.fd_eps);
                    let d_matter =
                        eight_pi * (src_plus[ei] - src_minus[ei]) / (2.0 * config.fd_eps);
                    j[(i, k)] = d_regge + d_matter;
                }
            }
            j
        };

        // SVD solve
        let rhs = phyz_math::DVec::from_fn(n_free, |i| -residual[i]);
        let svd = jac.svd(true, true);
        let delta = svd
            .solve(&rhs, config.svd_tol)
            .map_err(|_| TentMoveError::SingularJacobian)?;

        // Backtracking line search
        let old_res_sq = res_norm * res_norm;
        let mut step = 1.0;
        let mut accepted = false;

        for _ in 0..20 {
            for (j, &ej) in free_edges.iter().enumerate() {
                sq_lengths[ej] += step * delta[j];
            }

            let trial_regge = local_lorentzian_regge_action_grad(complex, sq_lengths, free_edges);
            let trial_src = source.edge_sources(complex, sq_lengths);
            let trial_res_sq: f64 = trial_regge
                .iter()
                .zip(free_edges.iter())
                .map(|(&rg, &ei)| (rg + eight_pi * trial_src[ei]).powi(2))
                .sum();

            if trial_res_sq < old_res_sq {
                accepted = true;
                break;
            }

            for (j, &ej) in free_edges.iter().enumerate() {
                sq_lengths[ej] -= step * delta[j];
            }
            step *= 0.5;
        }

        if !accepted {
            for (j, &ej) in free_edges.iter().enumerate() {
                sq_lengths[ej] += step * delta[j];
            }
        }
    }

    let regge_grad = local_lorentzian_regge_action_grad(complex, sq_lengths, free_edges);
    let matter_sources = source.edge_sources(complex, sq_lengths);
    let res_norm: f64 = regge_grad
        .iter()
        .zip(free_edges.iter())
        .map(|(&rg, &ei)| (rg + eight_pi * matter_sources[ei]).powi(2))
        .sum::<f64>()
        .sqrt();
    Err(TentMoveError::DidNotConverge {
        residual: res_norm,
        iterations: config.max_newton_iter,
    })
}

/// Zero source (vacuum): for testing.
pub struct VacuumSource;

impl StressEnergy for VacuumSource {
    fn edge_sources(&self, complex: &SimplicialComplex, _sq_lengths: &[f64]) -> Vec<f64> {
        vec![0.0; complex.n_edges()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foliation::{flat_minkowski_sq_lengths, foliated_hypercubic};
    use crate::tent_move::tent_edges_for_vertex;

    /// Vacuum source gives zero at all edges.
    #[test]
    fn test_vacuum_source_zero() {
        let fc = foliated_hypercubic(2, 2);
        let sq_lengths = flat_minkowski_sq_lengths(&fc, 1.0, 0.3);
        let source = VacuumSource;
        let sources = source.edge_sources(&fc.complex, &sq_lengths);
        assert!(sources.iter().all(|&s| s == 0.0));
    }

    /// Vacuum source reproduces the vacuum Regge solution.
    #[test]
    fn test_vacuum_source_solves_vacuum() {
        let fc = foliated_hypercubic(4, 2);
        let flat_sq = flat_minkowski_sq_lengths(&fc, 1.0, 0.3);
        let mut sq_lengths = flat_sq.clone();

        let v = fc.global_vertex(1, 0);
        let free = tent_edges_for_vertex(&fc, v, 2);

        // Perturb
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        for &ei in &free {
            sq_lengths[ei] *= 1.0 + 0.005 * (rng.r#gen::<f64>() - 0.5);
        }

        let config = TentConfig {
            max_newton_iter: 30,
            ..TentConfig::default()
        };
        let source = VacuumSource;
        let result = solve_regge_with_source(&fc.complex, &mut sq_lengths, &free, &source, &config)
            .expect("should converge");

        assert!(result.residual < 1e-8, "residual = {:.2e}", result.residual);
    }

    /// Mass current loop produces non-zero source terms on loop edges.
    #[test]
    fn test_mass_current_nonzero() {
        let fc = foliated_hypercubic(2, 2);
        let sq_lengths = flat_minkowski_sq_lengths(&fc, 1.0, 0.3);

        // Pick some vertices in slice 0 as the loop
        let loop_verts: Vec<usize> = (0..4).map(|i| fc.global_vertex(0, i)).collect();
        let source = MassCurrentLoop {
            mass_rate: 1.0,
            loop_vertices: loop_verts,
            gamma: 1.0,
        };

        let sources = source.edge_sources(&fc.complex, &sq_lengths);
        let nonzero_count = sources.iter().filter(|&&s| s.abs() > 1e-15).count();
        assert!(
            nonzero_count > 0,
            "mass current loop should produce nonzero sources"
        );
    }
}
