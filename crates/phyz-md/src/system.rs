//! Molecular dynamics system: a stateful driver over the SoA [`crate::field`]
//! engine, with velocity-Verlet (NVE) and BAOAB Langevin (NVT) integration.
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
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

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

/// Time integration scheme.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Integrator {
    /// Plain NVE velocity Verlet (no thermostat coupling).
    #[default]
    VelocityVerlet,
    /// BAOAB Langevin splitting: the deterministic B/A pieces are velocity
    /// Verlet, and the O piece is an exact Ornstein-Uhlenbeck update that
    /// satisfies the fluctuation-dissipation relation.
    Baoab,
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
#[derive(Debug)]
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
    /// Integration scheme.
    pub integrator: Integrator,
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

    /// Seed of the random number generator (for reproducibility).
    seed: u64,
    /// Random number generator used for velocity initialization and the
    /// thermostat.
    rng: ChaCha8Rng,
    /// Whether the 3 center-of-mass degrees of freedom have been removed.
    com_removed: bool,
}

impl Default for MdSystem {
    /// An empty system with a 1 fs timestep and a fixed seed.
    ///
    /// Deliberately deterministic: `Default` is the base the constructors fill
    /// in, and a silently entropy-seeded default would make trajectories
    /// irreproducible by accident. [`MdSystem::new`] draws a random seed
    /// explicitly.
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            velocities: Vec::new(),
            forces: Vec::new(),
            masses: Vec::new(),
            charges: Vec::new(),
            species: Vec::new(),
            lj: None,
            electrostatics: Electrostatics::None,
            bonds: Vec::new(),
            angles: HarmonicAngles::default(),
            dihedrals: PeriodicDihedrals::default(),
            impropers: HarmonicImpropers::default(),
            exclusions: Vec::new(),
            neighbor_list: NeighborList::default(),
            cell: None,
            time: 0.0,
            step: 0,
            dt: 1.0,
            rebuild_frequency: 0,
            thermostat: None,
            integrator: Integrator::default(),
            berendsen: None,
            nose_hoover: None,
            barostat: None,
            potential_energy: 0.0,
            virial: [[0.0; 3]; 3],
            seed: 0,
            rng: ChaCha8Rng::seed_from_u64(0),
            com_removed: false,
        }
    }
}

impl MdSystem {
    /// An empty system with the given timestep (fs) and a randomly drawn seed.
    ///
    /// Use [`MdSystem::with_seed`] or [`MdSystem::set_seed`] when a
    /// reproducible trajectory is wanted.
    pub fn new(dt: f64) -> Self {
        Self::with_seed(dt, rand::random::<u64>())
    }

    /// An empty system with the given timestep (fs) and an explicit RNG seed.
    pub fn with_seed(dt: f64, seed: u64) -> Self {
        Self {
            dt,
            rebuild_frequency: 0,
            neighbor_list: NeighborList::new(0.0, 2.0),
            seed,
            rng: ChaCha8Rng::seed_from_u64(seed),
            ..Default::default()
        }
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

    /// Draw a standard normal variate from the system RNG.
    fn normal(&mut self) -> f64 {
        StandardNormal.sample(&mut self.rng)
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

    /// Degrees of freedom: `3N`, minus the 3 center-of-mass degrees of freedom
    /// once [`Self::remove_com_motion`] has taken them out.
    pub fn degrees_of_freedom(&self) -> f64 {
        let n = 3 * self.len();
        if n == 0 {
            return 0.0;
        }
        if self.com_removed {
            n.saturating_sub(3).max(1) as f64
        } else {
            n as f64
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
        for i in 0..self.len() {
            // <v²> per degree of freedom is k_B T · FORCE_TO_ACCEL / m in these
            // units, since KE = ½ m v² / FORCE_TO_ACCEL.
            let sigma = (k_b * temperature * FORCE_TO_ACCEL / self.masses[i]).sqrt();
            self.velocities[i] = [
                sigma * self.normal(),
                sigma * self.normal(),
                sigma * self.normal(),
            ];
        }

        // Remove center-of-mass motion
        self.remove_com_motion();
    }

    /// Remove center-of-mass motion.
    ///
    /// This eliminates 3 degrees of freedom, which
    /// [`Self::degrees_of_freedom`] accounts for afterwards.
    pub fn remove_com_motion(&mut self) {
        self.com_removed = true;
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
    /// [`Self::virial`]. Only conservative terms appear here; Langevin
    /// coupling is part of the integrator (see [`Integrator::Baoab`]), not a
    /// force term.
    pub fn compute_forces(&mut self) {
        self.update_neighbor_list();
        let c = self.evaluate(&self.positions.clone());
        self.forces = c.forces;
        self.potential_energy = c.energy;
        self.virial = c.virial;
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

    /// Perform one integration step with the configured [`Integrator`].
    pub fn step(&mut self) {
        let n = self.len();
        if n == 0 {
            self.step += 1;
            return;
        }
        // Both schemes assume the force accumulator holds F(x(t)) on entry.
        if self.step == 0 || self.forces.len() != n {
            self.compute_forces();
        }

        match self.integrator {
            Integrator::VelocityVerlet => self.step_velocity_verlet(),
            Integrator::Baoab => self.step_baoab(),
        }

        if let (Some(baro), Some(mut cell)) = (self.barostat, self.cell) {
            let mu = baro.scale_factor(self.dt, self.pressure());
            baro.apply(mu, &mut self.positions, &mut cell);
            self.cell = Some(cell);
            // The box changed, so the cell decomposition is stale.
            self.neighbor_list
                .build(&self.positions, self.cell.as_ref());
        }

        self.time += self.dt;
        self.step += 1;
    }

    /// One NVE velocity-Verlet step, plus whichever deterministic thermostat
    /// is configured.
    fn step_velocity_verlet(&mut self) {
        let dt = self.dt;

        if let Some(mut nh) = self.nose_hoover {
            nose_hoover_half_step(dt, &mut nh, &mut self.velocities, &self.masses);
            self.nose_hoover = Some(nh);
        }

        self.kick(0.5 * dt);
        self.drift(dt);
        self.compute_forces();
        self.kick(0.5 * dt);

        if let Some(mut nh) = self.nose_hoover {
            nose_hoover_half_step(dt, &mut nh, &mut self.velocities, &self.masses);
            self.nose_hoover = Some(nh);
        }
        if let Some(b) = self.berendsen {
            apply_berendsen(dt, b, &mut self.velocities, &self.masses);
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

    /// Velocity kick over `dt` with the current forces ("B" in BAOAB).
    fn kick(&mut self, dt: f64) {
        for i in 0..self.len() {
            let a = vec3::scale(self.forces[i], FORCE_TO_ACCEL / self.masses[i]);
            vec3::add_assign(&mut self.velocities[i], vec3::scale(a, dt));
        }
    }

    /// Position drift over `dt` ("A" in BAOAB), followed by the periodic wrap.
    fn drift(&mut self, dt: f64) {
        for i in 0..self.len() {
            let dx = vec3::scale(self.velocities[i], dt);
            vec3::add_assign(&mut self.positions[i], dx);
        }
        // `Lattice::wrap` uses rem_euclid, so an atom that crossed several box
        // lengths in one step still lands inside.
        if let Some(cell) = self.cell {
            for p in &mut self.positions {
                *p = cell.wrap(*p);
            }
        }
    }

    /// Ornstein-Uhlenbeck velocity update ("O" in BAOAB).
    ///
    /// `v <- c v + sqrt(k_B T c_f (1 - c²) / m) ξ`, with `c = exp(-γ dt)` and
    /// `c_f = FORCE_TO_ACCEL` carrying the unit convention (equilibrium
    /// `<v²>` per degree of freedom is `k_B T c_f / m`, since
    /// `KE = ½ m v² / c_f`). Friction and noise amplitude are tied together
    /// exactly as the fluctuation-dissipation relation requires, so the
    /// stationary distribution of `v` is Maxwell-Boltzmann at the target
    /// temperature for any `dt`.
    fn ornstein_uhlenbeck(&mut self, dt: f64) {
        let Some(thermo) = self.thermostat.clone() else {
            return;
        };

        let c = (-thermo.gamma * dt).exp();
        let noise_scale = (thermo.k_b * thermo.temperature * FORCE_TO_ACCEL * (1.0 - c * c)).sqrt();

        for i in 0..self.len() {
            let sigma = noise_scale / self.masses[i].sqrt();
            let xi = [self.normal(), self.normal(), self.normal()];
            self.velocities[i] =
                vec3::add(vec3::scale(self.velocities[i], c), vec3::scale(xi, sigma));
        }

        // The O step re-thermalizes the center of mass as well. If the COM
        // degrees of freedom were removed (and are therefore excluded from the
        // reported temperature), keep removing them.
        if self.com_removed {
            self.remove_com_motion();
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::units::KB_EV_PER_K;
    use approx::assert_relative_eq;

    /// A seeded system with the given Lennard-Jones term, so tests that draw
    /// random velocities are reproducible.
    fn seeded(lj: LennardJones, dt: f64) -> MdSystem {
        let mut s = MdSystem::with_seed(dt, 0xA1207);
        s.set_lennard_jones(lj);
        s
    }

    /// The reduced-unit time scale `τ = σ √(m/ε)` for `ε = σ = m = 1`.
    ///
    /// Accelerations here are `a = FORCE_TO_ACCEL · F/m`, so the natural time
    /// unit of a reduced-unit LJ system is `1/√FORCE_TO_ACCEL`. Reduced-unit
    /// timesteps and damping rates are scaled by it, which keeps the reference
    /// numbers (dt* = 0.005, γ* = 1) recognizable.
    const TAU: f64 = 10.180_929_990_432_5; // 1/sqrt(FORCE_TO_ACCEL)

    /// Lennard-Jones fluid in reduced units (ε = σ = m = k_B = 1) on a simple
    /// cubic lattice, thermostatted at `temperature`.
    fn lj_fluid(seed: u64, n_side: usize, density: f64, temperature: f64, gamma: f64) -> MdSystem {
        let n = n_side * n_side * n_side;
        let l = (n as f64 / density).cbrt();
        let mut system = MdSystem::with_seed(0.005 * TAU, seed);
        system.set_lennard_jones(LennardJones::monatomic(1.0, 1.0, 2.0));
        // r_cut + skin must stay below L/2 for the minimum-image convention.
        system.neighbor_list.skin = 0.5;
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
        system.set_thermostat(temperature, gamma / TAU, 1.0);
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
        let mut system = MdSystem::with_seed(0.002 * TAU, 7);
        system.set_lennard_jones(LennardJones::monatomic(1.0, 1.0, 2.0));
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
        system.set_thermostat(target, 5.0 / TAU, k_b);

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
            for i in 0..system.len() {
                // v √(m / (k_B T · FORCE_TO_ACCEL)) ~ N(0, 1) per component.
                let scale = (system.masses[i] / (k_b * target * FORCE_TO_ACCEL)).sqrt();
                for x in system.velocities[i].map(|v| v * scale) {
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

    /// Same seed => identical trajectory; different seed => different one.
    #[test]
    fn test_trajectories_are_seed_reproducible() {
        let run = |seed: u64| {
            let mut system = lj_fluid(seed, 4, 0.4, 1.0, 2.0);
            for _ in 0..200 {
                system.step();
            }
            system
                .positions
                .iter()
                .copied()
                .zip(system.velocities.iter().copied())
                .collect::<Vec<_>>()
        };

        let a = run(1);
        let b = run(1);
        let c = run(2);

        assert_eq!(a.len(), b.len());
        for (i, ((xa, va), (xb, vb))) in a.iter().zip(&b).enumerate() {
            // Without the `parallel` feature this is bit-identical; with it,
            // the pair-sum reduction order is unspecified, so equal seeds are
            // required only to track each other closely over 200 steps.
            for k in 0..3 {
                assert!(
                    (xa[k] - xb[k]).abs() < 1e-9,
                    "position of particle {i} differs across equal seeds"
                );
                assert!(
                    (va[k] - vb[k]).abs() < 1e-9,
                    "velocity of particle {i} differs across equal seeds"
                );
            }
        }

        let differs = a
            .iter()
            .zip(&c)
            .any(|((xa, _), (xc, _))| (0..3).any(|k| (xa[k] - xc[k]).abs() > 1e-6));
        assert!(differs, "different seeds produced an identical trajectory");
    }

    #[test]
    fn test_temperature_dof_excludes_com() {
        let mut system = MdSystem::with_seed(0.001, 3);
        system.set_lennard_jones(LennardJones::argon());
        for i in 0..4 {
            system.add_particle(Particle::new(
                Vec3::new(i as f64 * 5.0, 0.0, 0.0),
                Vec3::zeros(),
                1.0,
                0,
            ));
        }
        assert_eq!(system.degrees_of_freedom(), 12.0);
        system.initialize_velocities(1.0, 1.0);
        assert_eq!(system.degrees_of_freedom(), 9.0);
        assert_relative_eq!(
            system.temperature(1.0),
            2.0 * system.kinetic_energy() / 9.0,
            epsilon = 1e-12
        );
    }

    fn argon_system(dt: f64) -> MdSystem {
        let mut s = MdSystem::with_seed(dt, 0xA1207);
        s.set_lennard_jones(LennardJones::argon());
        s
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
        let mut system = seeded(LennardJones::monatomic(0.0103, 3.4, 6.0), 1.0);
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
        let mut system = MdSystem::with_seed(1.0, 0xF00D);
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
        let mut system = seeded(LennardJones::monatomic(0.0103, 3.4, 8.0), 2.0);
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
        let mut system = seeded(LennardJones::monatomic(0.0103, 3.4, 8.0), 4.0);
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
