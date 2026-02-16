//! Molecular dynamics system with Velocity Verlet integration.

use crate::{ForceField, HarmonicBond, NeighborList, Particle};
use std::sync::Arc;
use tau_math::Vec3;

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

impl MdSystem {
    /// Create a new MD system.
    pub fn new(force_field: Arc<dyn ForceField>, dt: f64) -> Self {
        let r_cut = force_field.cutoff();
        let neighbor_list = NeighborList::new(r_cut, 0.5);

        Self {
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

    /// Set Langevin thermostat.
    pub fn set_thermostat(&mut self, temperature: f64, gamma: f64, k_b: f64) {
        self.thermostat = Some(Thermostat {
            temperature,
            gamma,
            k_b,
        });
    }

    /// Initialize velocities from Maxwell-Boltzmann distribution.
    pub fn initialize_velocities(&mut self, temperature: f64, k_b: f64) {
        use std::f64::consts::PI;

        for particle in &mut self.particles {
            // Sample from normal distribution
            // Box-Muller transform for Gaussian sampling
            let sigma = (k_b * temperature / particle.mass).sqrt();

            for d in 0..3 {
                let u1: f64 = rand();
                let u2: f64 = rand();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                particle.v[d] = sigma * z;
            }
        }

        // Remove center-of-mass motion
        self.remove_com_motion();
    }

    /// Remove center-of-mass motion.
    fn remove_com_motion(&mut self) {
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

        // Thermostat forces (Langevin)
        if let Some(ref thermo) = self.thermostat {
            use std::f64::consts::PI;

            let prefactor = (2.0 * thermo.gamma * thermo.k_b * thermo.temperature).sqrt();

            for particle in &mut self.particles {
                // Friction force
                let friction = -thermo.gamma * particle.mass * particle.v;
                particle.add_force(friction);

                // Random force
                let sigma = prefactor * (particle.mass / self.dt).sqrt();
                for d in 0..3 {
                    let u1: f64 = rand();
                    let u2: f64 = rand();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                    particle.f[d] += sigma * z;
                }
            }
        }
    }

    /// Perform one Velocity Verlet integration step.
    pub fn step(&mut self) {
        // Rebuild neighbor list if needed
        if self.step.is_multiple_of(self.rebuild_frequency)
            || self
                .neighbor_list
                .needs_rebuild(&self.particles, self.box_size)
        {
            self.neighbor_list.build(&self.particles, self.box_size);
        }

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
        for (i, particle) in self.particles.iter_mut().enumerate() {
            particle.x += particle.v * self.dt + 0.5 * old_accel[i] * self.dt * self.dt;
            particle.v += 0.5 * old_accel[i] * self.dt;

            // Apply periodic boundary conditions
            if let Some(box_size) = self.box_size {
                for d in 0..3 {
                    if particle.x[d] < 0.0 {
                        particle.x[d] += box_size[d];
                    } else if particle.x[d] >= box_size[d] {
                        particle.x[d] -= box_size[d];
                    }
                }
            }
        }

        // Compute new forces
        self.compute_forces();

        // Update velocities with new accelerations
        for particle in &mut self.particles {
            let new_accel = particle.f / particle.mass;
            particle.v += 0.5 * new_accel * self.dt;
        }

        self.time += self.dt;
        self.step += 1;
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

    /// Compute instantaneous temperature (K).
    pub fn temperature(&self, k_b: f64) -> f64 {
        let ke = self.kinetic_energy();
        let n_dof = 3 * self.particles.len();
        2.0 * ke / (n_dof as f64 * k_b)
    }
}

/// Simple pseudo-random number generator (LCG).
fn rand() -> f64 {
    use std::cell::RefCell;
    thread_local! {
        static SEED: RefCell<u64> = const { RefCell::new(12345) };
    }

    SEED.with(|seed| {
        let mut s = seed.borrow_mut();
        *s = s.wrapping_mul(1103515245).wrapping_add(12345);
        // Return value in (0, 1) avoiding exactly 0 or 1
        let val = ((*s / 65536) % 32768) as f64 / 32768.0;
        val.clamp(1e-10, 1.0 - 1e-10)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LennardJones;
    use approx::assert_relative_eq;

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
        let mut system = MdSystem::new(lj, 0.001);

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
