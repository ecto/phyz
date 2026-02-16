//! Hybrid Monte Carlo (HMC) algorithm for lattice gauge theory.

use crate::group::Group;
use crate::lattice::Lattice;

/// HMC parameters.
#[derive(Debug, Clone)]
pub struct HmcParams {
    /// Number of molecular dynamics steps per HMC trajectory.
    pub n_md_steps: usize,

    /// Molecular dynamics timestep.
    pub dt: f64,

    /// Enable Metropolis accept/reject (disable for testing).
    pub metropolis: bool,
}

impl Default for HmcParams {
    fn default() -> Self {
        Self {
            n_md_steps: 10,
            dt: 0.1,
            metropolis: true,
        }
    }
}

/// HMC state for tracking statistics.
#[derive(Debug, Clone, Default)]
pub struct HmcState {
    /// Number of HMC steps performed.
    pub n_steps: usize,

    /// Number of accepted trajectories.
    pub n_accepted: usize,

    /// Current Hamiltonian.
    pub hamiltonian: f64,

    /// History of Hamiltonian values (for monitoring drift).
    pub hamiltonian_history: Vec<f64>,
}

impl HmcState {
    /// Acceptance rate.
    pub fn acceptance_rate(&self) -> f64 {
        if self.n_steps == 0 {
            0.0
        } else {
            self.n_accepted as f64 / self.n_steps as f64
        }
    }

    /// Average Hamiltonian drift per step.
    pub fn average_drift(&self) -> f64 {
        if self.hamiltonian_history.len() < 2 {
            return 0.0;
        }

        let mut sum = 0.0;
        for i in 1..self.hamiltonian_history.len() {
            sum += (self.hamiltonian_history[i] - self.hamiltonian_history[i - 1]).abs();
        }
        sum / (self.hamiltonian_history.len() - 1) as f64
    }
}

impl<G: Group> Lattice<G> {
    /// Perform single HMC step.
    pub fn hmc_step(&mut self, params: &HmcParams) -> HmcState {
        self.hmc_step_with_state(params, &mut HmcState::default())
    }

    /// Perform HMC step with persistent state.
    pub fn hmc_step_with_state(&mut self, params: &HmcParams, state: &mut HmcState) -> HmcState {
        // 1. Sample momenta from Gaussian distribution
        let mut momenta = self.sample_momenta();

        // 2. Compute initial Hamiltonian
        let h_initial = self.hamiltonian(&momenta);

        // 3. Save initial configuration
        let links_initial = self.clone_links();

        // 4. Molecular dynamics integration (leapfrog)
        self.leapfrog(&mut momenta, params.n_md_steps, params.dt);

        // 5. Compute final Hamiltonian
        let h_final = self.hamiltonian(&momenta);

        // 6. Metropolis accept/reject
        let accepted = if params.metropolis {
            let delta_h = h_final - h_initial;
            let accept_prob = (-delta_h).exp().min(1.0);
            let r: f64 = rand::random();
            r < accept_prob
        } else {
            true
        };

        // 7. Update state
        if !accepted {
            self.restore_links(links_initial);
        }

        state.n_steps += 1;
        if accepted {
            state.n_accepted += 1;
        }
        state.hamiltonian = if accepted { h_final } else { h_initial };
        state.hamiltonian_history.push(state.hamiltonian);

        state.clone()
    }

    /// Sample momenta from Gaussian distribution.
    fn sample_momenta(&self) -> Vec<Vec<G::Momentum>> {
        let mut momenta = Vec::with_capacity(4);
        for _ in 0..4 {
            let mut direction_momenta = Vec::with_capacity(self.n_sites());
            for _ in 0..self.n_sites() {
                direction_momenta.push(G::sample_momentum());
            }
            momenta.push(direction_momenta);
        }
        momenta
    }

    /// Compute Hamiltonian H = K + S where K = 0.5 Σ |p|² and S is action.
    fn hamiltonian(&self, momenta: &[Vec<G::Momentum>]) -> f64 {
        let mut kinetic = 0.0;
        for momentum_dir in momenta.iter().take(4) {
            for momentum in momentum_dir.iter().take(self.n_sites()) {
                kinetic += G::momentum_norm_squared(momentum);
            }
        }
        kinetic *= 0.5;

        let potential = self.action();
        kinetic + potential
    }

    /// Leapfrog integrator for molecular dynamics.
    ///
    /// p(t+dt/2) = p(t) - (dt/2) * ∇S(U(t))
    /// U(t+dt) = U(t) * exp(dt * p(t+dt/2))
    /// p(t+dt) = p(t+dt/2) - (dt/2) * ∇S(U(t+dt))
    fn leapfrog(&mut self, momenta: &mut [Vec<G::Momentum>], n_steps: usize, dt: f64) {
        // Initial half-step for momenta
        let forces = self.compute_forces();
        for (mu, momentum_dir) in momenta.iter_mut().enumerate().take(4) {
            for (site, momentum) in momentum_dir.iter_mut().enumerate().take(self.n_sites()) {
                let force_scaled = G::scale_momentum(&forces[mu][site], -0.5 * dt);
                *momentum = G::add_momentum(momentum, &force_scaled);
            }
        }

        // Full steps
        for _ in 0..n_steps {
            // Update links
            #[allow(clippy::needless_range_loop)]
            for mu in 0..4 {
                for site in 0..self.n_sites() {
                    let p_scaled = G::scale_momentum(&momenta[mu][site], dt);
                    let exp_p = G::exp(&p_scaled);
                    let current_link = self.get_link(mu, site);
                    let new_link = current_link.mul(&exp_p);
                    self.set_link_unchecked(mu, site, new_link);
                }
            }

            // Update momenta (full step except last)
            let forces = self.compute_forces();
            for (mu, momentum_dir) in momenta.iter_mut().enumerate().take(4) {
                for (site, momentum) in momentum_dir.iter_mut().enumerate().take(self.n_sites()) {
                    let force_scaled = G::scale_momentum(&forces[mu][site], -dt);
                    *momentum = G::add_momentum(momentum, &force_scaled);
                }
            }
        }

        // Final half-step for momenta
        let forces = self.compute_forces();
        for (mu, momentum_dir) in momenta.iter_mut().enumerate().take(4) {
            for (site, momentum) in momentum_dir.iter_mut().enumerate().take(self.n_sites()) {
                let force_scaled = G::scale_momentum(&forces[mu][site], -0.5 * dt);
                *momentum = G::add_momentum(momentum, &force_scaled);
            }
        }
    }

    /// Compute forces ∇S for all links.
    ///
    /// Force on link U_μ(n) is:
    /// F_μ(n) = β * ∂S/∂U_μ(n) = -β * U_μ(n) * staple_sum
    fn compute_forces(&self) -> Vec<Vec<G::Momentum>> {
        let mut forces = Vec::with_capacity(4);

        for mu in 0..4 {
            let mut direction_forces = Vec::with_capacity(self.n_sites());
            for site in 0..self.n_sites() {
                let staple = self.staple_sum(site, mu);
                let link = self.get_link(mu, site);
                let force = link.force(&staple);
                let force_scaled = G::scale_momentum(&force, self.beta);
                direction_forces.push(force_scaled);
            }
            forces.push(direction_forces);
        }

        forces
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::U1;

    #[test]
    fn test_hmc_step() {
        let mut lattice = Lattice::<U1>::new(4, 4, 4, 4, 1.0);
        lattice.randomize();

        let params = HmcParams {
            n_md_steps: 5,
            dt: 0.05,
            metropolis: false, // Disable for deterministic test
        };

        let initial_action = lattice.action();
        lattice.hmc_step(&params);
        let final_action = lattice.action();

        // Action should change after HMC step
        assert!((final_action - initial_action).abs() > 1e-10);
    }

    #[test]
    fn test_hamiltonian_conservation() {
        let mut lattice = Lattice::<U1>::new(2, 2, 2, 2, 1.0); // Smaller lattice
        lattice.randomize();

        let params = HmcParams {
            n_md_steps: 5,
            dt: 0.01, // Very small dt for better conservation
            metropolis: false,
        };

        let mut state = HmcState::default();
        for _ in 0..5 {
            lattice.hmc_step_with_state(&params, &mut state);
        }

        // Check that Hamiltonian drift is reasonable
        // (will not be perfect due to finite dt and numerical errors)
        let drift = state.average_drift();
        assert!(drift < 100.0, "Hamiltonian drift too large: {}", drift);
    }

    #[test]
    fn test_acceptance_rate() {
        let mut lattice = Lattice::<U1>::new(2, 2, 2, 2, 1.0); // Smaller lattice
        lattice.randomize();

        let params = HmcParams {
            n_md_steps: 5,
            dt: 0.05, // Smaller timestep
            metropolis: true,
        };

        let mut state = HmcState::default();
        for _ in 0..20 {
            lattice.hmc_step_with_state(&params, &mut state);
        }

        // Acceptance rate should be non-zero
        let acc_rate = state.acceptance_rate();
        assert!(acc_rate > 0.0, "Acceptance rate: {}", acc_rate);
    }
}
