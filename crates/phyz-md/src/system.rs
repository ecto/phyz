//! Molecular dynamics system: a stateful driver over the SoA [`crate::field`]
//! engine.
//!
//! `MdSystem` owns the structure-of-arrays state (positions, velocities,
//! forces, masses, charges, species) and the interaction terms acting on it,
//! and drives them with velocity Verlet. Every force evaluation goes through
//! the same [`crate::field`] kernels that standalone callers use — there is one
//! engine, not two.
//!
//! # Units
//!
//! Å, eV, amu, fs, e, K — see [`crate::field::units`]. Accelerations use the
//! [`crate::field::units::FORCE_TO_ACCEL`] conversion, so `dt` is a genuine
//! femtosecond count.

use phyz_math::Vec3;

use crate::field::cell::{Lattice, vec3};
use crate::field::dihedral::{DihedralTerm, HarmonicImpropers, ImproperTerm, PeriodicDihedrals};
use crate::field::ewald::{Ewald, Pme};
use crate::field::neighbor::NeighborList;
use crate::field::potentials::{Coulomb, HarmonicAngles, HarmonicBonds, LennardJones};
use crate::field::units::FORCE_TO_ACCEL;
use crate::field::verlet::{
    Barostat, Berendsen, NoseHoover, apply_berendsen, nose_hoover_half_step,
};
use crate::field::virial::{self, Contribution};
use crate::particle::Particle;

/// Bond definition between two particles.
#[derive(Clone, Debug, PartialEq)]
pub struct Bond {
    /// First atom index.
    pub i: usize,
    /// Second atom index.
    pub j: usize,
    /// Force constant in eV/Å².
    pub k: f64,
    /// Equilibrium length in Å.
    pub r0: f64,
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

/// How electrostatics are evaluated.
///
/// Under periodic boundaries only [`Electrostatics::Ewald`] and
/// [`Electrostatics::Pme`] are correct: a bare cutoff sum does not converge to
/// the Coulomb lattice energy.
#[derive(Clone, Debug, Default)]
pub enum Electrostatics {
    /// No charges.
    #[default]
    None,
    /// Direct cutoff sum. Valid for isolated (non-periodic) clusters only.
    Cutoff(Coulomb),
    /// Direct Ewald summation.
    Ewald(Ewald),
    /// Particle mesh Ewald.
    Pme(Pme),
}

impl Electrostatics {
    /// The real-space cutoff this scheme needs from the neighbor list.
    fn cutoff(&self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Cutoff(c) => c.cutoff,
            Self::Ewald(e) => e.r_cut,
            Self::Pme(p) => p.r_cut,
        }
    }
}

/// Molecular dynamics system.
///
/// State is structure-of-arrays: index `i` of every array refers to the same
/// atom.
#[derive(Debug, Default)]
pub struct MdSystem {
    /// Positions in Å.
    pub positions: Vec<[f64; 3]>,
    /// Velocities in Å/fs.
    pub velocities: Vec<[f64; 3]>,
    /// Forces in eV/Å, valid after the last force evaluation.
    pub forces: Vec<[f64; 3]>,
    /// Masses in amu.
    pub masses: Vec<f64>,
    /// Charges in units of the elementary charge.
    pub charges: Vec<f64>,
    /// Species / atom-type ids, used for force-field parameter lookup.
    pub species: Vec<u32>,

    /// Non-bonded van der Waals term.
    pub lj: Option<LennardJones>,
    /// Electrostatics scheme.
    pub electrostatics: Electrostatics,
    /// Harmonic bonds.
    pub bonds: Vec<Bond>,
    /// Harmonic angles.
    pub angles: HarmonicAngles,
    /// Proper torsions.
    pub dihedrals: PeriodicDihedrals,
    /// Improper torsions.
    pub impropers: HarmonicImpropers,
    /// Pairs excluded from the non-bonded sums (1-2 and 1-3 neighbors).
    pub exclusions: Vec<(usize, usize)>,

    /// Neighbor list for non-bonded interactions.
    pub neighbor_list: NeighborList,
    /// Periodic cell (`None` = isolated cluster).
    pub cell: Option<Lattice>,

    /// Current simulation time in fs.
    pub time: f64,
    /// Current step count.
    pub step: usize,
    /// Timestep in fs.
    pub dt: f64,
    /// Rebuild the neighbor list at least every N steps (0 = displacement
    /// heuristic only).
    pub rebuild_frequency: usize,

    /// Thermostat parameters (Langevin).
    pub thermostat: Option<Thermostat>,
    /// Berendsen thermostat, an alternative to Langevin.
    pub berendsen: Option<Berendsen>,
    /// Nosé-Hoover thermostat, the canonical-sampling option.
    pub nose_hoover: Option<NoseHoover>,
    /// Berendsen barostat for NPT.
    pub barostat: Option<Barostat>,

    /// Potential energy in eV from the last force evaluation.
    pub potential_energy: f64,
    /// Virial tensor in eV from the last force evaluation.
    pub virial: [[f64; 3]; 3],
}

impl MdSystem {
    /// An empty system with the given timestep (fs).
    pub fn new(dt: f64) -> Self {
        Self {
            dt,
            rebuild_frequency: 0,
            neighbor_list: NeighborList::new(0.0, 2.0),
            ..Default::default()
        }
    }

    /// A system with a Lennard-Jones non-bonded term.
    pub fn lennard_jones(lj: LennardJones, dt: f64) -> Self {
        let mut s = Self::new(dt);
        s.set_lennard_jones(lj);
        s
    }

    /// Number of atoms.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the system has no atoms.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Add a particle to the system.
    pub fn add_particle(&mut self, particle: Particle) {
        self.positions
            .push([particle.x.x, particle.x.y, particle.x.z]);
        self.velocities
            .push([particle.v.x, particle.v.y, particle.v.z]);
        self.forces.push([0.0; 3]);
        self.masses.push(particle.mass);
        self.charges.push(particle.charge);
        self.species.push(particle.atom_type);
    }

    /// Read back atom `i` as a [`Particle`].
    pub fn particle(&self, i: usize) -> Particle {
        Particle {
            x: Vec3::new(
                self.positions[i][0],
                self.positions[i][1],
                self.positions[i][2],
            ),
            v: Vec3::new(
                self.velocities[i][0],
                self.velocities[i][1],
                self.velocities[i][2],
            ),
            f: Vec3::new(self.forces[i][0], self.forces[i][1], self.forces[i][2]),
            mass: self.masses[i],
            charge: self.charges[i],
            atom_type: self.species[i],
        }
    }

    /// Position of atom `i` as a [`Vec3`].
    pub fn position(&self, i: usize) -> Vec3 {
        Vec3::new(
            self.positions[i][0],
            self.positions[i][1],
            self.positions[i][2],
        )
    }

    /// Velocity of atom `i` as a [`Vec3`].
    pub fn velocity(&self, i: usize) -> Vec3 {
        Vec3::new(
            self.velocities[i][0],
            self.velocities[i][1],
            self.velocities[i][2],
        )
    }

    /// Set the Lennard-Jones term.
    pub fn set_lennard_jones(&mut self, lj: LennardJones) {
        self.lj = Some(lj);
        self.sync_cutoff();
    }

    /// Use direct Ewald summation for electrostatics.
    pub fn set_ewald(&mut self, ewald: Ewald) {
        self.electrostatics = Electrostatics::Ewald(ewald);
        self.sync_cutoff();
    }

    /// Use particle mesh Ewald for electrostatics.
    pub fn set_pme(&mut self, pme: Pme) {
        self.electrostatics = Electrostatics::Pme(pme);
        self.sync_cutoff();
    }

    /// Use a plain cutoff Coulomb sum. Correct only for isolated clusters.
    pub fn set_cutoff_coulomb(&mut self, coulomb: Coulomb) {
        self.electrostatics = Electrostatics::Cutoff(coulomb);
        self.sync_cutoff();
    }

    /// Configure PME automatically for the current cell at the given real-space
    /// cutoff. Requires a periodic cell.
    pub fn use_pme(&mut self, r_cut: f64, accuracy: f64) {
        if let Some(cell) = self.cell {
            self.set_pme(Pme::tuned(&cell, r_cut, accuracy, 1.0));
        }
    }

    /// Add a bond between particles i and j.
    pub fn add_bond(&mut self, i: usize, j: usize, k: f64, r0: f64) {
        self.bonds.push(Bond { i, j, k, r0 });
        self.rebuild_exclusions();
    }

    /// Add a harmonic angle `i–j–k` with apex `j`.
    pub fn add_angle(&mut self, i: usize, j: usize, k: usize, force_k: f64, theta0: f64) {
        self.angles.triples.push((i, j, k));
        self.angles.per_angle.push((force_k, theta0));
        self.rebuild_exclusions();
    }

    /// Add a periodic proper torsion `E = k (1 + cos(n φ − δ))`.
    pub fn add_dihedral(
        &mut self,
        atoms: (usize, usize, usize, usize),
        k: f64,
        n: u32,
        delta: f64,
    ) {
        self.dihedrals
            .terms
            .push(DihedralTerm { atoms, k, n, delta });
    }

    /// Add a harmonic improper torsion `E = ½ k (φ − φ₀)²`.
    pub fn add_improper(&mut self, atoms: (usize, usize, usize, usize), k: f64, phi0: f64) {
        self.impropers.terms.push(ImproperTerm { atoms, k, phi0 });
    }

    /// Set periodic boundary conditions with an orthorhombic box.
    pub fn set_box_size(&mut self, box_size: Vec3) {
        self.cell = Some(Lattice::orthorhombic(box_size.x, box_size.y, box_size.z));
    }

    /// Set a general (possibly triclinic) periodic cell.
    pub fn set_cell(&mut self, cell: Lattice) {
        self.cell = Some(cell);
    }

    /// Cell volume in Å³, or 0 for a non-periodic system.
    pub fn volume(&self) -> f64 {
        self.cell.map(|c| c.volume().abs()).unwrap_or(0.0)
    }

    /// Set Langevin thermostat.
    pub fn set_thermostat(&mut self, temperature: f64, gamma: f64, k_b: f64) {
        self.thermostat = Some(Thermostat {
            temperature,
            gamma,
            k_b,
        });
    }

    /// Set a Nosé-Hoover thermostat at `temperature` K with coupling time
    /// `tau_fs`.
    pub fn set_nose_hoover(&mut self, temperature: f64, tau_fs: f64) {
        let dof = self.degrees_of_freedom();
        self.nose_hoover = Some(NoseHoover::new(temperature, tau_fs, dof));
    }

    /// Set a Berendsen barostat targeting `pressure` (eV/Å³).
    pub fn set_barostat(&mut self, pressure: f64, tau_fs: f64, compressibility: f64) {
        self.barostat = Some(Barostat::new(pressure, tau_fs, compressibility));
    }

    /// Degrees of freedom: `3N − 3` once center-of-mass motion is removed.
    pub fn degrees_of_freedom(&self) -> f64 {
        let n = self.len();
        if n == 0 {
            0.0
        } else {
            (3 * n).saturating_sub(3).max(1) as f64
        }
    }

    /// Recompute 1-2 and 1-3 exclusions from the bond and angle lists.
    ///
    /// Bonded partners must be excluded from the non-bonded sums, and for
    /// Ewald/PME the exclusion also has to be subtracted from the smooth
    /// reciprocal part — [`crate::field::ewald`] handles that half.
    pub fn rebuild_exclusions(&mut self) {
        let mut set: Vec<(usize, usize)> = Vec::new();
        let push = |a: usize, b: usize, set: &mut Vec<(usize, usize)>| {
            if a == b {
                return;
            }
            let p = (a.min(b), a.max(b));
            if !set.contains(&p) {
                set.push(p);
            }
        };
        for b in &self.bonds {
            push(b.i, b.j, &mut set);
        }
        for &(i, _, k) in &self.angles.triples {
            push(i, k, &mut set);
        }
        self.exclusions = set;
        self.neighbor_list
            .set_exclusions(self.exclusions.iter().copied());
    }

    /// Match the neighbor-list cutoff to the longest-ranged non-bonded term.
    fn sync_cutoff(&mut self) {
        let lj = self.lj.as_ref().map(|l| l.cutoff).unwrap_or(0.0);
        let r = lj.max(self.electrostatics.cutoff());
        if r > self.neighbor_list.cutoff {
            self.neighbor_list.cutoff = r;
        }
    }

    /// Initialize velocities from a Maxwell-Boltzmann distribution.
    pub fn initialize_velocities(&mut self, temperature: f64, k_b: f64) {
        use std::f64::consts::PI;

        for i in 0..self.len() {
            // <v²> per degree of freedom is k_B T · FORCE_TO_ACCEL / m in these
            // units, since KE = ½ m v² / FORCE_TO_ACCEL.
            let sigma = (k_b * temperature * FORCE_TO_ACCEL / self.masses[i]).sqrt();

            let rand_component = || -> f64 {
                let u1: f64 = rand();
                let u2: f64 = rand();
                (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
            };
            self.velocities[i] = [
                sigma * rand_component(),
                sigma * rand_component(),
                sigma * rand_component(),
            ];
        }

        // Remove center-of-mass motion
        self.remove_com_motion();
    }

    /// Remove center-of-mass motion.
    pub fn remove_com_motion(&mut self) {
        let mut total_momentum = [0.0f64; 3];
        let mut total_mass = 0.0;

        for (v, &m) in self.velocities.iter().zip(&self.masses) {
            vec3::add_assign(&mut total_momentum, vec3::scale(*v, m));
            total_mass += m;
        }
        if total_mass <= 0.0 {
            return;
        }
        let com = vec3::scale(total_momentum, 1.0 / total_mass);
        for v in &mut self.velocities {
            *v = vec3::sub(*v, com);
        }
    }

    /// Rebuild the neighbor list if the displacement heuristic (or the step
    /// counter) calls for it.
    pub fn update_neighbor_list(&mut self) {
        if self.neighbor_list.cutoff <= 0.0 {
            return;
        }
        let forced = self.rebuild_frequency > 0 && self.step.is_multiple_of(self.rebuild_frequency);
        if forced {
            self.neighbor_list
                .build(&self.positions, self.cell.as_ref());
        } else {
            self.neighbor_list
                .maybe_build(&self.positions, self.cell.as_ref());
        }
    }

    /// Evaluate every interaction term at the current positions.
    ///
    /// Populates [`Self::forces`], [`Self::potential_energy`], and
    /// [`Self::virial`]. Thermostat forces are *not* included in the virial —
    /// they are not part of the physical interaction.
    pub fn compute_forces(&mut self) {
        self.update_neighbor_list();
        let c = self.evaluate(&self.positions.clone());
        self.forces = c.forces;
        self.potential_energy = c.energy;
        self.virial = c.virial;
        self.apply_thermostat_forces();
    }

    /// Evaluate all conservative terms at the given positions.
    fn evaluate(&self, positions: &[[f64; 3]]) -> Contribution {
        let n = positions.len();
        let mut acc = Contribution::zeros(n);
        let cell = self.cell.as_ref();
        let pairs = self.neighbor_list.pairs();

        if let Some(lj) = &self.lj {
            acc.merge(&lj.compute_pairs(&self.species, positions, pairs, cell));
        }
        match &self.electrostatics {
            Electrostatics::None => {}
            Electrostatics::Cutoff(c) => {
                acc.merge(&c.compute_pairs(&self.charges, positions, pairs, cell));
            }
            Electrostatics::Ewald(e) => {
                acc.merge(&e.compute(
                    &self.charges,
                    positions,
                    cell,
                    Some(pairs),
                    &self.exclusions,
                ));
            }
            Electrostatics::Pme(p) => {
                acc.merge(&p.compute(
                    &self.charges,
                    positions,
                    cell,
                    Some(pairs),
                    &self.exclusions,
                ));
            }
        }
        if !self.bonds.is_empty() {
            let list: Vec<(usize, usize)> = self.bonds.iter().map(|b| (b.i, b.j)).collect();
            let params = HarmonicBonds {
                k: 0.0,
                r0: 0.0,
                per_bond: self.bonds.iter().map(|b| b.r0).collect(),
                per_bond_k: self.bonds.iter().map(|b| b.k).collect(),
            };
            acc.merge(&params.compute_all(&list, positions, cell));
        }
        if !self.angles.triples.is_empty() {
            acc.merge(&self.angles.compute_all(positions, cell));
        }
        if !self.dihedrals.terms.is_empty() {
            acc.merge(&self.dihedrals.compute_all(positions, cell));
        }
        if !self.impropers.terms.is_empty() {
            acc.merge(&self.impropers.compute_all(positions, cell));
        }
        acc
    }

    /// Add Langevin friction and random forces, if a Langevin thermostat is
    /// configured.
    fn apply_thermostat_forces(&mut self) {
        use std::f64::consts::PI;
        let Some(thermo) = self.thermostat.clone() else {
            return;
        };
        for i in 0..self.len() {
            let m = self.masses[i];
            // a_friction = −γ v, so F = −γ m v / FORCE_TO_ACCEL.
            let friction = vec3::scale(self.velocities[i], -thermo.gamma * m / FORCE_TO_ACCEL);
            vec3::add_assign(&mut self.forces[i], friction);

            // Fluctuation-dissipation: σ_F = sqrt(2 γ k_B T m / (c dt)).
            let sigma = (2.0 * thermo.gamma * thermo.k_b * thermo.temperature * m
                / (FORCE_TO_ACCEL * self.dt))
                .sqrt();
            let rand_component = || -> f64 {
                let u1: f64 = rand();
                let u2: f64 = rand();
                (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
            };
            vec3::add_assign(
                &mut self.forces[i],
                [
                    sigma * rand_component(),
                    sigma * rand_component(),
                    sigma * rand_component(),
                ],
            );
        }
    }

    /// Perform one velocity-Verlet integration step.
    pub fn step(&mut self) {
        let n = self.len();
        if n == 0 {
            self.step += 1;
            return;
        }
        if self.forces.len() != n {
            self.compute_forces();
        }
        let dt = self.dt;

        if let Some(mut nh) = self.nose_hoover {
            nose_hoover_half_step(dt, &mut nh, &mut self.velocities, &self.masses);
            self.nose_hoover = Some(nh);
        }

        // v(t+dt/2) = v(t) + ½ dt a(t);  x(t+dt) = x(t) + dt v(t+dt/2)
        for i in 0..n {
            let a = vec3::scale(self.forces[i], FORCE_TO_ACCEL / self.masses[i]);
            vec3::add_assign(&mut self.velocities[i], vec3::scale(a, 0.5 * dt));
            let dx = vec3::scale(self.velocities[i], dt);
            vec3::add_assign(&mut self.positions[i], dx);
        }

        // Wrap into the cell. `Lattice::wrap` uses rem_euclid, so an atom that
        // crossed several box lengths in one step still lands inside.
        if let Some(cell) = self.cell {
            for p in &mut self.positions {
                *p = cell.wrap(*p);
            }
        }

        self.compute_forces();

        // v(t+dt) = v(t+dt/2) + ½ dt a(t+dt)
        for i in 0..n {
            let a = vec3::scale(self.forces[i], FORCE_TO_ACCEL / self.masses[i]);
            vec3::add_assign(&mut self.velocities[i], vec3::scale(a, 0.5 * dt));
        }

        if let Some(mut nh) = self.nose_hoover {
            nose_hoover_half_step(dt, &mut nh, &mut self.velocities, &self.masses);
            self.nose_hoover = Some(nh);
        }
        if let Some(b) = self.berendsen {
            apply_berendsen(dt, b, &mut self.velocities, &self.masses);
        }
        if let (Some(baro), Some(mut cell)) = (self.barostat, self.cell) {
            let mu = baro.scale_factor(dt, self.pressure());
            baro.apply(mu, &mut self.positions, &mut cell);
            self.cell = Some(cell);
            // The box changed, so the cell decomposition is stale.
            self.neighbor_list
                .build(&self.positions, self.cell.as_ref());
        }

        self.time += dt;
        self.step += 1;
    }

    /// Total kinetic energy in eV.
    pub fn kinetic_energy(&self) -> f64 {
        crate::field::verlet::kinetic_energy(&self.velocities, &self.masses)
    }

    /// Recompute the total potential energy in eV at the current positions.
    ///
    /// Named distinctly from the [`Self::potential_energy`] *field*, which
    /// holds the value cached by the last force evaluation — having both under
    /// one name is a footgun.
    pub fn compute_potential_energy(&self) -> f64 {
        self.evaluate(&self.positions).energy
    }

    /// Total energy in eV (kinetic plus the cached potential).
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy
    }

    /// The conserved quantity for the active ensemble, in eV.
    ///
    /// In NVE this is `KE + PE`; under Nosé-Hoover it also includes the
    /// thermostat's extended-system terms, since `KE + PE` alone is not
    /// conserved there.
    pub fn conserved_energy(&self) -> f64 {
        let mut e = self.total_energy();
        if let Some(nh) = self.nose_hoover {
            e += nh.conserved_offset();
        }
        e
    }

    /// Instantaneous temperature (K).
    pub fn temperature(&self, k_b: f64) -> f64 {
        let dof = self.degrees_of_freedom();
        if dof <= 0.0 {
            return 0.0;
        }
        2.0 * self.kinetic_energy() / (dof * k_b)
    }

    /// Instantaneous pressure in eV/Å³ from the kinetic energy and the virial.
    ///
    /// Returns 0 for a non-periodic system, where pressure is not defined.
    pub fn pressure(&self) -> f64 {
        let v = self.volume();
        if v <= 0.0 {
            return 0.0;
        }
        let scalar = self.virial[0][0] + self.virial[1][1] + self.virial[2][2];
        virial::pressure(self.kinetic_energy(), scalar, v)
    }

    /// Instantaneous pressure in GPa.
    pub fn pressure_gpa(&self) -> f64 {
        virial::to_gpa(self.pressure())
    }

    /// Full pressure tensor in eV/Å³.
    pub fn pressure_tensor(&self) -> [[f64; 3]; 3] {
        virial::pressure_tensor(&self.velocities, &self.masses, &self.virial, self.volume())
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
    use crate::field::units::KB_EV_PER_K;
    use approx::assert_relative_eq;

    fn argon_system(dt: f64) -> MdSystem {
        MdSystem::lennard_jones(LennardJones::argon(), dt)
    }

    #[test]
    fn test_two_particle_lj() {
        let mut system = argon_system(1.0);

        // Two argon atoms at the LJ minimum.
        let r_eq = 1.122462 * 3.4;
        system.add_particle(Particle::new(Vec3::zeros(), Vec3::zeros(), 39.948, 0));
        system.add_particle(Particle::new(
            Vec3::new(r_eq, 0.0, 0.0),
            Vec3::zeros(),
            39.948,
            0,
        ));
        system.compute_forces();

        // The shifted potential sits slightly above −ε at the minimum.
        // The cutoff-shifted form sits ~1.7e-4 eV above the unshifted −ε.
        assert_relative_eq!(system.potential_energy, -0.0103, epsilon = 3e-4);
        // And the force there vanishes.
        assert!(system.forces[0][0].abs() < 1e-5);
    }

    #[test]
    fn nve_conserves_energy() {
        let mut system = argon_system(1.0);
        for i in 0..4 {
            let mut p = Particle::new(
                Vec3::new(i as f64 * 4.2, 0.0, 0.0),
                Vec3::zeros(),
                39.948,
                0,
            );
            p.v = Vec3::new(0.0, 1e-3 * (i as f64 - 1.5), 0.0);
            system.add_particle(p);
        }
        system.compute_forces();
        let e0 = system.total_energy();
        for _ in 0..2000 {
            system.step();
        }
        let drift = (system.total_energy() - e0).abs() / e0.abs().max(1e-9);
        assert!(drift < 1e-3, "energy drift {drift}");
    }

    /// Velocity Verlet is a second-order integrator, so the energy error of an
    /// NVE trajectory over fixed physical time must fall as dt². This is a
    /// sharper statement than "energy is roughly conserved" — it checks that
    /// the integrator is the order it claims to be.
    #[test]
    fn nve_energy_drift_scales_as_dt_squared() {
        // A bound argon dimer oscillating in its LJ well: periodic, not
        // chaotic, so the energy error is a clean function of dt rather than
        // something dominated by trajectory divergence.
        let run = |dt: f64, steps: usize| -> f64 {
            let mut s = argon_system(dt);
            s.add_particle(Particle::new(Vec3::zeros(), Vec3::zeros(), 39.948, 0));
            s.add_particle(Particle::new(
                Vec3::new(4.3, 0.0, 0.0), // stretched from r_min ≈ 3.816 Å
                Vec3::zeros(),
                39.948,
                0,
            ));
            s.compute_forces();
            let e0 = s.total_energy();
            let mut worst: f64 = 0.0;
            for _ in 0..steps {
                s.step();
                worst = worst.max((s.total_energy() - e0).abs());
            }
            worst
        };
        // Same total simulated time (4000 fs) at three timesteps.
        let e_coarse = run(4.0, 1000);
        let e_mid = run(2.0, 2000);
        let e_fine = run(1.0, 4000);

        // Halving dt should cut the error by ~4×.
        let r1 = e_coarse / e_mid;
        let r2 = e_mid / e_fine;
        assert!(
            (2.5..6.0).contains(&r1) && (2.5..6.0).contains(&r2),
            "drift ratios {r1:.2}, {r2:.2} (want ≈4 for dt² scaling); \
             errors {e_coarse:.3e} {e_mid:.3e} {e_fine:.3e}"
        );
    }

    #[test]
    fn test_temperature() {
        let mut system = argon_system(1.0);
        for i in 0..10 {
            system.add_particle(Particle::new(
                Vec3::new(i as f64 * 5.0, 0.0, 0.0),
                Vec3::zeros(),
                1.0,
                0,
            ));
        }

        let k_b = KB_EV_PER_K;
        let target_temp = 300.0;
        system.initialize_velocities(target_temp, k_b);

        let temp = system.temperature(k_b);
        assert!(
            (temp - target_temp).abs() / target_temp < 0.5,
            "Temperature: {temp:.1} K (target: {target_temp:.1} K)"
        );
    }

    /// A particle moving many box lengths in a single step must still land
    /// inside the cell. The previous single-crossing correction silently let it
    /// escape.
    #[test]
    fn pbc_wrap_survives_a_multi_box_jump() {
        let mut system = argon_system(1.0);
        system.set_box_size(Vec3::new(10.0, 10.0, 10.0));
        let mut p = Particle::new(Vec3::new(5.0, 5.0, 5.0), Vec3::zeros(), 39.948, 0);
        // ~7 box lengths per step.
        p.v = Vec3::new(70.0, -70.0, 0.0);
        system.add_particle(p);
        system.compute_forces();
        system.step();

        for k in 0..3 {
            assert!(
                (0.0..10.0).contains(&system.positions[0][k]),
                "axis {k} escaped the box: {:?}",
                system.positions[0]
            );
        }
    }

    #[test]
    fn neighbor_list_is_used_and_stays_linear() {
        // 10×10×10 argon at 4.2 Å spacing in a periodic box. The size matters:
        // with too few atoms per cell the stencil scan and the all-pairs loop
        // cost the same, and the test proves nothing.
        let mut system = MdSystem::lennard_jones(LennardJones::monatomic(0.0103, 3.4, 6.0), 1.0);
        system.neighbor_list.skin = 1.0;
        let spacing = 4.2;
        let n_side = 10;
        for i in 0..n_side {
            for j in 0..n_side {
                for k in 0..n_side {
                    system.add_particle(Particle::new(
                        Vec3::new(i as f64 * spacing, j as f64 * spacing, k as f64 * spacing),
                        Vec3::zeros(),
                        39.948,
                        0,
                    ));
                }
            }
        }
        let l = n_side as f64 * spacing;
        system.set_box_size(Vec3::new(l, l, l));
        system.compute_forces();

        assert!(
            !system.neighbor_list.used_fallback(),
            "cell lists should be active for {} atoms in a {l} Å box",
            system.len()
        );
        // An all-pairs build would be (N−1)/2 ≈ 500 checks per atom here.
        let per_atom = system.neighbor_list.checks() as f64 / system.len() as f64;
        assert!(per_atom < 150.0, "checks/atom = {per_atom}");
    }

    #[test]
    fn bonded_pairs_are_excluded_from_nonbonded_sums() {
        let mut system = argon_system(1.0);
        system.add_particle(Particle::new(Vec3::zeros(), Vec3::zeros(), 1.0, 0));
        system.add_particle(Particle::new(
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::zeros(),
            1.0,
            0,
        ));
        system.add_bond(0, 1, 10.0, 1.5);
        system.compute_forces();

        // At 1.5 Å the LJ term would be enormous (σ = 3.4 Å); the exclusion
        // must remove it, leaving only the bond at its equilibrium length.
        assert!(
            system.potential_energy.abs() < 1e-9,
            "expected only the (zero) bond energy, got {}",
            system.potential_energy
        );
        assert!(system.neighbor_list.pairs().is_empty());
    }

    #[test]
    fn pressure_of_an_ideal_gas_is_nkt_over_v() {
        // No interactions: P V = N k T exactly.
        let mut system = MdSystem::new(1.0);
        let n = 50;
        for i in 0..n {
            system.add_particle(Particle::new(
                Vec3::new(i as f64 * 0.4, 0.0, 0.0),
                Vec3::zeros(),
                39.948,
                0,
            ));
        }
        system.set_box_size(Vec3::new(20.0, 20.0, 20.0));
        system.initialize_velocities(300.0, KB_EV_PER_K);
        system.compute_forces();

        let want = 2.0 * system.kinetic_energy() / (3.0 * system.volume());
        assert_relative_eq!(system.pressure(), want, epsilon = 1e-12);
        // The virial must be identically zero with no interactions.
        assert!(system.virial.iter().flatten().all(|v| v.abs() < 1e-15));
    }

    #[test]
    fn barostat_drives_the_box_toward_the_target_pressure() {
        // A compressed LJ solid started at high pressure should expand.
        let mut system = MdSystem::lennard_jones(LennardJones::monatomic(0.0103, 3.4, 8.0), 2.0);
        let spacing = 3.4;
        let n_side = 5;
        for i in 0..n_side {
            for j in 0..n_side {
                for k in 0..n_side {
                    system.add_particle(Particle::new(
                        Vec3::new(i as f64 * spacing, j as f64 * spacing, k as f64 * spacing),
                        Vec3::zeros(),
                        39.948,
                        0,
                    ));
                }
            }
        }
        let l = n_side as f64 * spacing;
        system.set_box_size(Vec3::new(l, l, l));
        system.compute_forces();
        let p0 = system.pressure();
        assert!(p0 > 0.0, "expected a compressed start, got P = {p0}");

        let v0 = system.volume();
        system.set_barostat(0.0, 200.0, 5.0);
        system.berendsen = Some(Berendsen {
            target_k: 20.0,
            tau_fs: 100.0,
        });
        for _ in 0..400 {
            system.step();
        }
        assert!(
            system.volume() > v0,
            "box should have expanded: {v0} → {}",
            system.volume()
        );
        assert!(
            system.pressure() < p0,
            "pressure should have dropped: {p0} → {}",
            system.pressure()
        );
    }

    #[test]
    fn nose_hoover_holds_the_target_temperature() {
        let mut system = MdSystem::lennard_jones(LennardJones::monatomic(0.0103, 3.4, 8.0), 4.0);
        let spacing = 4.2;
        let n_side = 5;
        for i in 0..n_side {
            for j in 0..n_side {
                for k in 0..n_side {
                    system.add_particle(Particle::new(
                        Vec3::new(i as f64 * spacing, j as f64 * spacing, k as f64 * spacing),
                        Vec3::zeros(),
                        39.948,
                        0,
                    ));
                }
            }
        }
        let l = n_side as f64 * spacing;
        system.set_box_size(Vec3::new(l, l, l));
        system.initialize_velocities(120.0, KB_EV_PER_K);
        system.compute_forces();
        system.set_nose_hoover(80.0, 200.0);

        let mut avg = 0.0;
        let steps = 3000;
        for s in 0..steps {
            system.step();
            if s >= steps / 2 {
                avg += system.temperature(KB_EV_PER_K);
            }
        }
        avg /= (steps - steps / 2) as f64;
        assert!(
            (avg - 80.0).abs() / 80.0 < 0.25,
            "mean T = {avg:.1} K, target 80 K"
        );
    }
}
