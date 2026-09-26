//! The particle state XPBD operates on.

use phyz_math::Vec3;

/// A cloud of point masses: the only state XPBD integrates.
///
/// Everything the solver simulates — cloth, a tetrahedral soft body, a cable —
/// is this array plus a list of constraints over its indices. The solver never
/// stores per-object structure, so a single [`ParticleSystem`] can hold a
/// cloth, a rope and a soft body at once and they interact through whatever
/// constraints couple them.
///
/// # Mass is stored inverted
///
/// [`inv_mass`](Self::inv_mass) is the primitive, not mass. Position-based
/// projection distributes a correction in proportion to `w = 1/m`, so the
/// solver would divide by `m` on every constraint of every iteration of every
/// substep. Storing `w` also gives pinning for free: `w = 0` is a particle of
/// infinite mass that absorbs no correction, which is exactly a hard pin, with
/// no branch in the projection code and no special case in the solve order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParticleSystem {
    /// Current positions. This is what a renderer reads.
    pub positions: Vec<Vec3>,
    /// Current velocities, derived from the position change at the end of each
    /// substep rather than integrated independently.
    pub velocities: Vec<Vec3>,
    /// Inverse masses, `w = 1/m`. `0.0` pins the particle.
    pub inv_mass: Vec<f64>,
    /// Positions at the start of the current substep, before prediction.
    ///
    /// Public because it is genuinely part of the state — velocity at the end
    /// of a substep is `(x - prev)/h`, so a caller that teleports a particle
    /// must decide what `prev` means for it. The solver overwrites this at the
    /// start of every substep.
    pub prev_positions: Vec<Vec3>,
}

impl ParticleSystem {
    /// An empty system.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a particle of the given mass at rest.
    ///
    /// A `mass` of zero (or negative, which is meaningless) is stored as
    /// `w = 0`: a pinned particle. Returns the new particle's index, which is
    /// what constraints refer to.
    pub fn add(&mut self, position: Vec3, mass: f64) -> usize {
        let w = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        self.positions.push(position);
        self.velocities.push(Vec3::zeros());
        self.inv_mass.push(w);
        self.prev_positions.push(position);
        self.positions.len() - 1
    }

    /// Add a particle that never moves.
    pub fn add_pinned(&mut self, position: Vec3) -> usize {
        self.add(position, 0.0)
    }

    /// Number of particles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the system holds no particles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Pin particle `i` where it currently is, by zeroing its inverse mass.
    pub fn pin(&mut self, i: usize) {
        self.inv_mass[i] = 0.0;
        self.velocities[i] = Vec3::zeros();
    }

    /// Set particle `i`'s mass, un-pinning it if `mass > 0`.
    pub fn set_mass(&mut self, i: usize, mass: f64) {
        self.inv_mass[i] = if mass > 0.0 { 1.0 / mass } else { 0.0 };
    }

    /// Total kinetic energy, `Σ ½ m |v|²`. Pinned particles contribute nothing.
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        let mut e = 0.0;
        for i in 0..self.len() {
            if self.inv_mass[i] > 0.0 {
                e += 0.5 * self.velocities[i].dot(self.velocities[i]) / self.inv_mass[i];
            }
        }
        e
    }
}
