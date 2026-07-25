//! Molecular dynamics system with Velocity Verlet (NVE) and BAOAB Langevin
//! (NVT) integration.

use crate::{ForceField, HarmonicBond, NeighborList, Particle};
use phyz_math::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use std::sync::Arc;

/// Bond definition between two particles.
#[derive(Clone, Debug)]
pub struct Bond {
    pub i: usize,
    pub j: usize,
    pub potential: HarmonicBond,
}

/// Molecular dynamics system.
pub struct MdSystem {
    /// Particles in the system.
    pub particles: Vec<Particle>,
    /// Non-bonded force field.
    pub force_field: Arc<dyn ForceField>,
    /// Bonded interactions.
    pub bonds: Vec<Bond>,
    /// Neighbor list for non-bonded interactions.
    pub neighbor_list: NeighborList,
    /// Periodic box size (None = no PBC).
    pub box_size: Option<Vec3>,
    /// Current simulation time.
    pub time: f64,
    /// Current step count.
    pub step: usize,
    /// Timestep (fs or ps).
    pub dt: f64,
    /// Rebuild neighbor list every N steps.
    pub rebuild_frequency: usize,
    /// Thermostat parameters (Langevin).
    pub thermostat: Option<Thermostat>,
    /// Integration scheme.
    pub integrator: Integrator,
    /// Seed of the random number generator (for reproducibility).
    seed: u64,
    /// Random number generator used for velocity initialization and the thermostat.
    rng: ChaCha8Rng,
    /// Whether the 3 center-of-mass degrees of freedom have been removed.
    com_removed: bool,
}

/// Langevin thermostat parameters.
#[derive(Clone, Debug)]
pub struct Thermostat {
    /// Target temperature (K).
    pub temperature: f64,
    /// Damping coefficient (1/ps or 1/fs).
    pub gamma: f64,
    /// Boltzmann constant (eV/K or appropriate units).
    pub k_b: f64,
}

/// Time integration scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integrator {
    /// Plain NVE velocity Verlet (no thermostat coupling).
    VelocityVerlet,
    /// BAOAB Langevin splitting: the deterministic B/A pieces are velocity
    /// Verlet, and the O piece is an exact Ornstein-Uhlenbeck update that
    /// satisfies the fluctuation-dissipation relation.
    Baoab,
}

impl MdSystem {
    /// Create a new MD system with a randomly drawn seed.
    ///
    /// Use [`MdSystem::with_seed`] or [`MdSystem::set_seed`] when a
    /// reproducible trajectory is wanted.
    pub fn new(force_field: Arc<dyn ForceField>, dt: f64) -> Self {
        let seed = rand::random::<u64>();
        Self::with_seed(force_field, dt, seed)
    }

    /// Create a new MD system with an explicit RNG seed.
    ///
    /// Two systems built with the same seed and driven identically produce
    /// bit-identical trajectories.
    pub fn with_seed(force_field: Arc<dyn ForceField>, dt: f64, seed: u64) -> Self {
        let r_cut = force_field.cutoff();
        let neighbor_list = NeighborList::new(r_cut, 0.5);

        Self {
            integrator: Integrator::VelocityVerlet,
            seed,
            rng: ChaCha8Rng::seed_from_u64(seed),
            com_removed: false,
            particles: Vec::new(),
            force_field,
            bonds: Vec::new(),
            neighbor_list,
            box_size: None,
            time: 0.0,
            step: 0,
            dt,
            rebuild_frequency: 10,
            thermostat: None,
        }
    }

    /// Add a particle to the system.
    pub fn add_particle(&mut self, particle: Particle) {
        self.particles.push(particle);
    }

    /// Add a bond between particles i and j.
    pub fn add_bond(&mut self, i: usize, j: usize, k: f64, r0: f64) {
        self.bonds.push(Bond {
            i,
            j,
            potential: HarmonicBond::new(k, r0),
        });
    }

    /// Set periodic boundary conditions.
    pub fn set_box_size(&mut self, box_size: Vec3) {
        self.box_size = Some(box_size);
    }

    /// Re-seed the random number generator.
    ///
    /// Resets the RNG stream, so calling this before a run makes the run
    /// reproducible.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = ChaCha8Rng::seed_from_u64(seed);
    }

    /// Seed the random number generator was last set with.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Select the time integration scheme.
    pub fn set_integrator(&mut self, integrator: Integrator) {
        self.integrator = integrator;
    }

    /// Set Langevin thermostat and switch to the BAOAB integrator.
    pub fn set_thermostat(&mut self, temperature: f64, gamma: f64, k_b: f64) {
        self.thermostat = Some(Thermostat {
            temperature,
            gamma,
            k_b,
        });
        self.integrator = Integrator::Baoab;
    }

    /// Remove the thermostat and return to plain NVE velocity Verlet.
    pub fn clear_thermostat(&mut self) {
        self.thermostat = None;
        self.integrator = Integrator::VelocityVerlet;
    }

    /// Draw a standard normal variate from the system RNG.
    fn normal(&mut self) -> f64 {
        StandardNormal.sample(&mut self.rng)
    }

    /// Initialize velocities from Maxwell-Boltzmann distribution.
    pub fn initialize_velocities(&mut self, temperature: f64, k_b: f64) {
        for i in 0..self.particles.len() {
            let sigma = (k_b * temperature / self.particles[i].mass).sqrt();
            let v = Vec3::new(self.normal(), self.normal(), self.normal()) * sigma;
            self.particles[i].v = v;
        }

        // Remove center-of-mass motion
        self.remove_com_motion();
    }

    /// Remove center-of-mass motion.
    ///
    /// This eliminates 3 degrees of freedom, which [`MdSystem::temperature`]
    /// accounts for afterwards.
    pub fn remove_com_motion(&mut self) {
        self.com_removed = true;

        if self.particles.is_empty() {
            return;
        }

        let mut total_momentum = Vec3::zeros();
        let mut total_mass = 0.0;

        for particle in &self.particles {
            total_momentum += particle.mass * particle.v;
            total_mass += particle.mass;
        }

        let com_velocity = total_momentum / total_mass;

        for particle in &mut self.particles {
            particle.v -= com_velocity;
        }
    }

    /// Compute forces on all particles.
    fn compute_forces(&mut self) {
        // Reset forces
        for particle in &mut self.particles {
            particle.reset_force();
        }

        // Non-bonded forces via neighbor list
        for &(i, j) in &self.neighbor_list.pairs {
            let mut r_ij = self.particles[j].x - self.particles[i].x;

            // Apply minimum image if periodic
            if let Some(box_size) = self.box_size {
                r_ij = crate::neighbor::minimum_image(r_ij, box_size);
            }

            let (force_on_i, _) = self.force_field.compute_force(
                r_ij,
                self.particles[i].atom_type,
                self.particles[j].atom_type,
            );

            // Newton's third law: F_j = -F_i
            self.particles[i].add_force(force_on_i);
            self.particles[j].add_force(-force_on_i);
        }

        // Bonded forces
        for bond in &self.bonds {
            let mut r_ij = self.particles[bond.j].x - self.particles[bond.i].x;

            if let Some(box_size) = self.box_size {
                r_ij = crate::neighbor::minimum_image(r_ij, box_size);
            }

            let (force_on_i, _) = bond.potential.compute(r_ij);

            // Newton's third law
            self.particles[bond.i].add_force(force_on_i);
            self.particles[bond.j].add_force(-force_on_i);
        }
    }

    /// Perform one integration step with the configured [`Integrator`].
    pub fn step(&mut self) {
        // Rebuild neighbor list if needed
        if self.step.is_multiple_of(self.rebuild_frequency)
            || self
                .neighbor_list
                .needs_rebuild(&self.particles, self.box_size)
        {
            self.neighbor_list.build(&self.particles, self.box_size);
        }

        // Both schemes assume the force accumulator holds F(x(t)) on entry.
        if self.step == 0 {
            self.compute_forces();
        }

        match self.integrator {
            Integrator::VelocityVerlet => self.step_velocity_verlet(),
            Integrator::Baoab => self.step_baoab(),
        }

        self.time += self.dt;
        self.step += 1;
    }

    /// Wrap positions back into the periodic box.
    fn apply_pbc(&mut self) {
        let Some(box_size) = self.box_size else {
            return;
        };

        fn wrap_coord(val: &mut f64, size: f64) {
            if *val < 0.0 {
                *val += size;
            } else if *val >= size {
                *val -= size;
            }
        }

        for particle in &mut self.particles {
            wrap_coord(&mut particle.x.x, box_size.x);
            wrap_coord(&mut particle.x.y, box_size.y);
            wrap_coord(&mut particle.x.z, box_size.z);
        }
    }

    /// Half-step velocity kick with the current forces ("B" in BAOAB).
    fn kick(&mut self, dt: f64) {
        for particle in &mut self.particles {
            particle.v += (dt / particle.mass) * particle.f;
        }
    }

    /// Position drift ("A" in BAOAB).
    fn drift(&mut self, dt: f64) {
        for particle in &mut self.particles {
            particle.x += particle.v * dt;
        }
        self.apply_pbc();
    }

    /// Ornstein-Uhlenbeck velocity update ("O" in BAOAB).
    ///
    /// `v <- c v + sqrt(k_B T (1 - c^2) / m) * xi`, with `c = exp(-gamma dt)`.
    /// The friction and the noise amplitude are tied together exactly as the
    /// fluctuation-dissipation relation requires, so the stationary
    /// distribution of `v` is Maxwell-Boltzmann at the target temperature for
    /// any `dt`.
    fn ornstein_uhlenbeck(&mut self, dt: f64) {
        let Some(thermo) = self.thermostat.clone() else {
            return;
        };

        let c = (-thermo.gamma * dt).exp();
        let noise_scale = (thermo.k_b * thermo.temperature * (1.0 - c * c)).sqrt();

        for i in 0..self.particles.len() {
            let sigma = noise_scale / self.particles[i].mass.sqrt();
            let xi = Vec3::new(self.normal(), self.normal(), self.normal());
            self.particles[i].v = c * self.particles[i].v + sigma * xi;
        }

        // The O step re-thermalizes the center of mass as well. If the COM
        // degrees of freedom were removed (and are therefore excluded from the
        // reported temperature), keep removing them.
        if self.com_removed {
            self.remove_com_motion();
        }
    }

    /// BAOAB Langevin splitting step.
    fn step_baoab(&mut self) {
        let dt = self.dt;

        self.kick(0.5 * dt); // B
        self.drift(0.5 * dt); // A
        self.ornstein_uhlenbeck(dt); // O
        self.drift(0.5 * dt); // A
        self.compute_forces();
        self.kick(0.5 * dt); // B
    }

    /// Perform one NVE Velocity Verlet integration step.
    fn step_velocity_verlet(&mut self) {
        // Velocity Verlet algorithm:
        // x(t+dt) = x(t) + v(t) dt + 0.5 a(t) dt²
        // v(t+dt/2) = v(t) + 0.5 a(t) dt
        // Compute forces at t+dt
        // v(t+dt) = v(t+dt/2) + 0.5 a(t+dt) dt

        // Store old accelerations
        let mut old_accel = Vec::with_capacity(self.particles.len());
        for particle in &self.particles {
            old_accel.push(particle.f / particle.mass);
        }

        // Update positions and half-step velocities
        let dt = self.dt;
        for (i, particle) in self.particles.iter_mut().enumerate() {
            particle.x += particle.v * dt + 0.5 * old_accel[i] * dt * dt;
            particle.v += 0.5 * old_accel[i] * dt;
        }
        self.apply_pbc();

        // Compute new forces
        self.compute_forces();

        // Update velocities with new accelerations
        for particle in &mut self.particles {
            let new_accel = particle.f / particle.mass;
            particle.v += 0.5 * new_accel * dt;
        }
    }

    /// Compute total kinetic energy.
    pub fn kinetic_energy(&self) -> f64 {
        self.particles.iter().map(|p| p.kinetic_energy()).sum()
    }

    /// Compute total potential energy.
    pub fn potential_energy(&self) -> f64 {
        let mut pe = 0.0;

        // Non-bonded energy
        for &(i, j) in &self.neighbor_list.pairs {
            let mut r_ij = self.particles[j].x - self.particles[i].x;

            if let Some(box_size) = self.box_size {
                r_ij = crate::neighbor::minimum_image(r_ij, box_size);
            }

            let (_, pot) = self.force_field.compute_force(
                r_ij,
                self.particles[i].atom_type,
                self.particles[j].atom_type,
            );
            pe += pot;
        }

        // Bonded energy
        for bond in &self.bonds {
            let mut r_ij = self.particles[bond.j].x - self.particles[bond.i].x;

            if let Some(box_size) = self.box_size {
                r_ij = crate::neighbor::minimum_image(r_ij, box_size);
            }

            let (_, pot) = bond.potential.compute(r_ij);
            pe += pot;
        }

        pe
    }

    /// Number of kinetic degrees of freedom.
    ///
    /// `3N`, minus the 3 center-of-mass degrees of freedom when COM motion has
    /// been removed (which [`MdSystem::initialize_velocities`] does).
    pub fn n_dof(&self) -> usize {
        let n = 3 * self.particles.len();
        if self.com_removed { n.saturating_sub(3) } else { n }
    }

    /// Compute instantaneous temperature (K).
    pub fn temperature(&self, k_b: f64) -> f64 {
        let n_dof = self.n_dof();
        if n_dof == 0 {
            return 0.0;
        }
        2.0 * self.kinetic_energy() / (n_dof as f64 * k_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LennardJones;
    use approx::assert_relative_eq;

    /// Lennard-Jones fluid in reduced units (ε = σ = m = k_B = 1) on a simple
    /// cubic lattice, thermostatted at `temperature`.
    fn lj_fluid(seed: u64, n_side: usize, density: f64, temperature: f64, gamma: f64) -> MdSystem {
        let n = n_side * n_side * n_side;
        let l = (n as f64 / density).cbrt();
        // r_cut + skin must stay below L/2 for the minimum-image convention.
        let lj = Arc::new(LennardJones::new(1.0, 1.0, 2.0));
        let mut system = MdSystem::with_seed(lj, 0.005, seed);
        system.set_box_size(Vec3::new(l, l, l));

        let a = l / n_side as f64;
        for i in 0..n_side {
            for j in 0..n_side {
                for k in 0..n_side {
                    system.add_particle(Particle::new(
                        Vec3::new(i as f64 * a, j as f64 * a, k as f64 * a),
                        Vec3::zeros(),
                        1.0,
                        0,
                    ));
                }
            }
        }

        system.initialize_velocities(temperature, 1.0);
        system.set_thermostat(temperature, gamma, 1.0);
        system
    }

    /// Standard normal CDF (Abramowitz & Stegun 7.1.26 erf approximation).
    fn normal_cdf(x: f64) -> f64 {
        let z = x / std::f64::consts::SQRT_2;
        let sign = if z < 0.0 { -1.0 } else { 1.0 };
        let t = 1.0 / (1.0 + 0.3275911 * z.abs());
        let poly = t
            * (0.254829592
                + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        let erf = sign * (1.0 - poly * (-z * z).exp());
        0.5 * (1.0 + erf)
    }

    /// The time-averaged kinetic temperature of a thermostatted LJ fluid must
    /// match the target within statistical error.
    #[test]
    fn test_langevin_samples_target_temperature() {
        let target = 1.5;
        let mut system = lj_fluid(0xC0FFEE, 4, 0.4, target, 1.0);

        for _ in 0..4_000 {
            system.step();
        }

        let n_sample = 20_000;
        let mut sum = 0.0;
        for _ in 0..n_sample {
            system.step();
            sum += system.temperature(1.0);
        }
        let mean = sum / n_sample as f64;

        // 189 dof and a friction correlation time of 1/gamma = 200 steps leaves
        // roughly 100 independent samples, so the standard error is ~1%.
        let err = (mean - target).abs() / target;
        assert!(
            err < 0.05,
            "mean temperature {mean:.4} vs target {target:.4} ({:.2}% off)",
            err * 100.0
        );
    }

    /// The stationary velocity distribution must be Maxwell-Boltzmann.
    ///
    /// Uses non-interacting particles, for which the analytic marginal of each
    /// mass-scaled velocity component is exactly a standard normal.
    #[test]
    fn test_velocity_distribution_is_maxwell_boltzmann() {
        let target = 2.0;
        let k_b = 1.0;
        let lj = Arc::new(LennardJones::new(1.0, 1.0, 2.0));
        let mut system = MdSystem::with_seed(lj, 0.002, 7);
        // Particles spaced beyond the cutoff, so they never interact and the
        // thermostat alone determines the distribution.
        let masses = [1.0, 4.0];
        for i in 0..64 {
            system.add_particle(Particle::new(
                Vec3::new(i as f64 * 10.0, 0.0, 0.0),
                Vec3::zeros(),
                masses[i % 2],
                0,
            ));
        }
        system.rebuild_frequency = 1_000;
        system.set_thermostat(target, 5.0, k_b);

        // Burn in, then sample far enough apart to decorrelate (gamma * dt *
        // 500 = 5, so the OU autocorrelation is e^-5).
        for _ in 0..2_000 {
            system.step();
        }

        let edges = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
        let mut counts = [0usize; 10];
        let mut n = 0usize;
        let (mut m2, mut m4) = (0.0, 0.0);

        for _ in 0..200 {
            for _ in 0..500 {
                system.step();
            }
            for p in &system.particles {
                // v * sqrt(m / k_B T) ~ N(0, 1) per component.
                let scale = (p.mass / (k_b * target)).sqrt();
                for x in [p.v.x * scale, p.v.y * scale, p.v.z * scale] {
                    m2 += x * x;
                    m4 += x * x * x * x;
                    n += 1;
                    let bin = edges.iter().filter(|&&e| x >= e).count();
                    counts[bin] += 1;
                }
            }
        }

        let m2 = m2 / n as f64;
        let m4 = m4 / n as f64;
        assert!((m2 - 1.0).abs() < 0.05, "second moment {m2:.4}, expected 1");
        assert!((m4 - 3.0).abs() < 0.30, "fourth moment {m4:.4}, expected 3");

        // Pearson chi-squared goodness of fit, 9 dof.
        let mut chi2 = 0.0;
        for (bin, &count) in counts.iter().enumerate() {
            let lo = if bin == 0 {
                0.0
            } else {
                normal_cdf(edges[bin - 1])
            };
            let hi = if bin == counts.len() - 1 {
                1.0
            } else {
                normal_cdf(edges[bin])
            };
            let expected = n as f64 * (hi - lo);
            chi2 += (count as f64 - expected).powi(2) / expected;
        }
        // 0.1% critical value for 9 dof.
        assert!(chi2 < 27.88, "chi-squared {chi2:.2} over {n} samples");
    }

    /// Same seed => bit-identical trajectory; different seed => different one.
    #[test]
    fn test_trajectories_are_seed_reproducible() {
        let run = |seed: u64| {
            let mut system = lj_fluid(seed, 4, 0.4, 1.0, 2.0);
            for _ in 0..200 {
                system.step();
            }
            system
                .particles
                .iter()
                .map(|p| (p.x, p.v))
                .collect::<Vec<_>>()
        };

        let a = run(1);
        let b = run(1);
        let c = run(2);

        assert_eq!(a.len(), b.len());
        for (i, ((xa, va), (xb, vb))) in a.iter().zip(&b).enumerate() {
            assert_eq!(xa, xb, "position of particle {i} differs across equal seeds");
            assert_eq!(va, vb, "velocity of particle {i} differs across equal seeds");
        }

        let differs = a.iter().zip(&c).any(|((xa, _), (xc, _))| xa != xc);
        assert!(differs, "different seeds produced an identical trajectory");
    }

    #[test]
    fn test_temperature_dof_excludes_com() {
        let lj = Arc::new(LennardJones::argon());
        let mut system = MdSystem::with_seed(lj, 0.001, 3);
        for i in 0..4 {
            system.add_particle(Particle::new(
                Vec3::new(i as f64 * 5.0, 0.0, 0.0),
                Vec3::zeros(),
                1.0,
                0,
            ));
        }
        assert_eq!(system.n_dof(), 12);
        system.initialize_velocities(1.0, 1.0);
        assert_eq!(system.n_dof(), 9);
        assert_relative_eq!(
            system.temperature(1.0),
            2.0 * system.kinetic_energy() / 9.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_two_particle_lj() {
        let lj = Arc::new(LennardJones::argon());
        let mut system = MdSystem::new(lj, 0.001);

        // Two argon atoms at equilibrium distance
        let r_eq = 1.122 * 3.4;
        system.add_particle(Particle::new(Vec3::zeros(), Vec3::zeros(), 39.948, 0));
        system.add_particle(Particle::new(
            Vec3::new(r_eq, 0.0, 0.0),
            Vec3::zeros(),
            39.948,
            0,
        ));

        system
            .neighbor_list
            .build(&system.particles, system.box_size);

        let pe_initial = system.potential_energy();

        // Should be near minimum energy (-ε)
        assert_relative_eq!(pe_initial, -0.0103, epsilon = 1e-3);
    }

    #[test]
    fn test_energy_conservation() {
        let lj = Arc::new(LennardJones::argon());
        let mut system = MdSystem::new(lj, 0.001);

        // Two particles with initial velocities
        let mut p1 = Particle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::zeros(), 1.0, 0);
        p1.v = Vec3::new(1.0, 0.0, 0.0);
        system.add_particle(p1);

        let mut p2 = Particle::new(Vec3::new(5.0, 0.0, 0.0), Vec3::zeros(), 1.0, 0);
        p2.v = Vec3::new(-1.0, 0.0, 0.0);
        system.add_particle(p2);

        let e_initial = system.kinetic_energy() + system.potential_energy();

        // Run simulation
        for _ in 0..100 {
            system.step();
        }

        let e_final = system.kinetic_energy() + system.potential_energy();

        // Energy should be conserved (within numerical error)
        let drift = (e_final - e_initial).abs() / e_initial.abs().max(1e-10);
        assert!(drift < 0.01, "Energy drift: {:.2}%", drift * 100.0);
    }

    #[test]
    fn test_temperature() {
        let lj = Arc::new(LennardJones::argon());
        let mut system = MdSystem::with_seed(lj, 0.001, 42);

        for i in 0..10 {
            system.add_particle(Particle::new(
                Vec3::new(i as f64 * 5.0, 0.0, 0.0),
                Vec3::zeros(),
                1.0,
                0,
            ));
        }

        let k_b = 8.617e-5; // eV/K
        let target_temp = 300.0;
        system.initialize_velocities(target_temp, k_b);

        let temp = system.temperature(k_b);

        // Temperature should be close to target (statistical fluctuations)
        assert!(
            (temp - target_temp).abs() / target_temp < 0.5,
            "Temperature: {:.1} K (target: {:.1} K)",
            temp,
            target_temp
        );
    }
}
