//! The XPBD time loop: predict, project, derive velocity.

use crate::constraint::{Constraint, project};
use crate::particles::ParticleSystem;
use phyz_math::Vec3;

/// XPBD solver settings and the time loop.
///
/// # The loop
///
/// For each of `substeps` substeps of duration `h = dt / substeps`:
///
/// 1. **Predict.** `prev = x`, `v += h·(g + a_ext)`, `x += h·v`. This is an
///    unconstrained symplectic-Euler step.
/// 2. **Project.** Zero every constraint's multiplier, then run `iterations`
///    passes over the constraint list in order, applying the XPBD update
///    ([`project`]) to each.
/// 3. **Derive velocity.** `v = (x − prev)/h`, then apply damping.
///
/// Deriving velocity from the position change rather than integrating it
/// separately is what makes the method stable: a constraint that removes a
/// position error also removes the velocity that produced it, in the same
/// step, with no force ever computed.
///
/// # Substeps versus iterations, and why the default is 10 × 1
///
/// XPBD's fixed point is the same whether you reach it with many iterations in
/// one big step or few iterations in many small ones, but the *error* is not.
/// Müller et al., "Small Steps in Physics Simulation" (2020), showed that for
/// a fixed total constraint-projection budget, spending it on substeps beats
/// spending it on iterations: substepping reduces the integration error too,
/// and it improves the conditioning of each individual solve because the
/// predicted positions are closer to the constrained manifold. So the default
/// is [`substeps`](Self::substeps) `= 10`, [`iterations`](Self::iterations)
/// `= 1`.
///
/// The exception is a system with long-range stiff coupling — a hanging chain
/// of `n` links, where information has to travel `n` constraints to reach the
/// pin. Gauss–Seidel propagates one constraint per iteration, so such a system
/// needs iterations (or substeps) on the order of `n` before it is anywhere
/// near converged, regardless of how the budget is split. This is a property
/// of the sweep, not of XPBD, and this crate does not fix it: no graph
/// colouring, no multigrid, no direct solve.
///
/// # Determinism
///
/// Same input, same output, bit for bit, on a given platform. The solve is
/// pure `f64`, single-threaded, and sweeps a `Vec` in index order — no hash
/// map iteration, no floating-point accumulation whose order depends on
/// anything but the constraint list you supplied. Reordering the constraint
/// list *does* change the answer (Gauss–Seidel is order-dependent), so the
/// order is part of your input, not an implementation detail.
#[derive(Debug, Clone, PartialEq)]
pub struct XpbdSolver {
    /// Frame duration in seconds. Split into `substeps` internally.
    pub dt: f64,
    /// Substeps per call to [`step`](Self::step). Default `10`.
    pub substeps: usize,
    /// Constraint sweeps per substep. Default `1`.
    pub iterations: usize,
    /// Uniform acceleration applied to every particle. Default is Earth
    /// gravity along `−Y`.
    pub gravity: Vec3,
    /// Linear velocity damping coefficient, s⁻¹, applied as
    /// `v *= max(0, 1 − damping·h)` at the end of each substep.
    ///
    /// Written in terms of `h` rather than as a bare per-substep multiplier so
    /// that changing `substeps` does not change how fast the system settles.
    /// A per-substep multiplier — the form most PBD code uses — would damp ten
    /// times as hard at ten substeps, silently coupling a numerical knob to a
    /// material one. Default `0.0`.
    pub damping: f64,
}

impl Default for XpbdSolver {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            substeps: 10,
            iterations: 1,
            gravity: Vec3::new(0.0, -phyz_math::GRAVITY, 0.0),
            damping: 0.0,
        }
    }
}

impl XpbdSolver {
    /// A solver with the given frame timestep and defaults elsewhere.
    #[must_use]
    pub fn new(dt: f64) -> Self {
        Self {
            dt,
            ..Self::default()
        }
    }

    /// Advance `particles` by [`dt`](Self::dt), subject to `constraints`.
    ///
    /// `constraints` is `&mut` because each constraint carries the Lagrange
    /// multiplier accumulated during the current substep. The multipliers are
    /// zeroed at the start of every substep, so the caller sees no state carried
    /// between `step` calls and reusing a constraint list across steps is safe.
    ///
    /// # Panics
    ///
    /// Panics if any constraint indexes a particle that does not exist.
    pub fn step(&self, particles: &mut ParticleSystem, constraints: &mut [Constraint]) {
        self.step_with_forces(particles, constraints, &[]);
    }

    /// [`step`](Self::step) with an additional per-particle external
    /// acceleration.
    ///
    /// `accelerations` may be empty (no extra acceleration) or exactly
    /// `particles.len()` long — anything else is a bug in the caller and
    /// panics, rather than silently applying wind to the first half of a
    /// cloth.
    ///
    /// Acceleration, not force, because the projection loop is already
    /// mass-weighted through `w`; taking a force here would mean dividing by
    /// mass at the one place a pinned particle has no mass to divide by.
    ///
    /// # Panics
    ///
    /// Panics if `accelerations` is neither empty nor `particles.len()` long,
    /// or if a constraint indexes a particle that does not exist.
    pub fn step_with_forces(
        &self,
        particles: &mut ParticleSystem,
        constraints: &mut [Constraint],
        accelerations: &[Vec3],
    ) {
        assert!(
            accelerations.is_empty() || accelerations.len() == particles.len(),
            "accelerations must be empty or one per particle ({} given, {} particles)",
            accelerations.len(),
            particles.len()
        );
        if self.substeps == 0 || particles.is_empty() {
            return;
        }
        let h = self.dt / self.substeps as f64;
        let n = particles.len();

        for _ in 0..self.substeps {
            // XPBD resets the multipliers once per substep, not once per
            // iteration. Resetting per iteration would erase the compliance
            // feedback term and turn the solver back into PBD; never resetting
            // would carry a stale force estimate from a configuration that no
            // longer exists.
            for c in constraints.iter_mut() {
                c.lambda = 0.0;
            }

            for i in 0..n {
                particles.prev_positions[i] = particles.positions[i];
                if particles.inv_mass[i] > 0.0 {
                    let mut a = self.gravity;
                    if !accelerations.is_empty() {
                        a += accelerations[i];
                    }
                    particles.velocities[i] += a * h;
                    let v = particles.velocities[i];
                    particles.positions[i] += v * h;
                }
            }

            for _ in 0..self.iterations {
                for c in constraints.iter_mut() {
                    project(c, particles, h);
                }
            }

            let damp = (1.0 - self.damping * h).max(0.0);
            for i in 0..n {
                if particles.inv_mass[i] > 0.0 {
                    let dx = particles.positions[i] - particles.prev_positions[i];
                    particles.velocities[i] = dx / h;
                    if self.damping != 0.0 {
                        particles.velocities[i] *= damp;
                    }
                } else {
                    // A pinned particle never moved, and a nonzero velocity on
                    // it would be a lie a caller could read.
                    particles.velocities[i] = Vec3::zeros();
                }
            }
        }
    }
}
