//! The `Solver` trait: the abstraction coupling drives.
//!
//! Every domain solver in the workspace exposes the same four capabilities to
//! the coupling layer:
//!
//! 1. **Advance** its own state by a requested `dt` ([`Solver::advance`]).
//! 2. **Expose state** for coupling queries — discrete carriers of mass/charge
//!    via [`Solver::sites`], continuum fields via [`Solver::field_at`].
//! 3. **Accept external input** from the other domain ([`Solver::apply_external`]).
//! 4. **Report its natural timestep** ([`Solver::natural_dt`]) so the driver can
//!    build a [`crate::SubcyclingSchedule`] instead of forcing a global `dt`.
//!
//! Only six methods are required; the two state-exposure hooks have defaults so
//! that a solver which is purely a force sink (or purely a field source) does
//! not have to implement the half it does not have.

use crate::coupling::SolverType;
use phyz_math::Vec3;

/// A discrete carrier of state that the coupling layer can see and push on.
///
/// A rigid body, an MD atom, an MPM particle, and an N-body mass all reduce to
/// this for the purpose of a handshake. `id` is the index the owning solver
/// uses to route an [`ExternalInput::Force`] back to the right internal object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingSite {
    /// Solver-local identifier. Must round-trip through `ExternalInput::Force`.
    pub id: usize,
    /// World-frame position (m).
    pub position: Vec3,
    /// World-frame linear velocity (m/s).
    pub velocity: Vec3,
    /// Mass (kg).
    pub mass: f64,
    /// Electric charge (C). Zero for uncharged sites.
    pub charge: f64,
}

impl CouplingSite {
    /// A neutral site with the given kinematics.
    pub fn neutral(id: usize, position: Vec3, velocity: Vec3, mass: f64) -> Self {
        Self {
            id,
            position,
            velocity,
            mass,
            charge: 0.0,
        }
    }
}

/// Continuum field values sampled at a point, as seen by another domain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldSample {
    /// Electric field (V/m).
    pub e_field: Vec3,
    /// Magnetic flux density (T).
    pub b_field: Vec3,
}

impl FieldSample {
    /// A sample with all components zero.
    pub fn zero() -> Self {
        Self {
            e_field: Vec3::zeros(),
            b_field: Vec3::zeros(),
        }
    }
}

impl Default for FieldSample {
    fn default() -> Self {
        Self::zero()
    }
}

/// Input handed to a solver by the coupling layer, consumed on the next
/// [`Solver::advance`].
///
/// The three variants are deliberately the three things a handshake can hand
/// across: a force on a discrete site, a source term for a continuum field, and
/// a bookkeeping entry for momentum that crossed the interface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExternalInput {
    /// A world-frame force and torque on the site with the given id.
    Force {
        /// Site id, as reported by [`Solver::sites`].
        site: usize,
        /// World-frame force (N).
        force: Vec3,
        /// World-frame torque (N·m).
        torque: Vec3,
    },
    /// A point current source (A·m) for a field solver — the physical
    /// back-reaction of a moving charge on the field it moves through.
    Current {
        /// World-frame position of the source (m).
        position: Vec3,
        /// Current moment `q·v` (A·m).
        moment: Vec3,
    },
    /// Momentum booked into this domain by the handshake (N·s).
    ///
    /// This is the accounting counterpart of a [`Force`](ExternalInput::Force)
    /// applied to the *other* domain. It lets a solver whose own momentum
    /// integral is expensive or ill-defined (a Yee grid, for instance) still
    /// close the momentum ledger exactly.
    Reaction {
        /// Impulse delivered into this domain (N·s).
        impulse: Vec3,
    },
}

/// A physics domain that can be coupled to other domains.
///
/// Implementations live next to the coupling layer (see [`crate::solvers`])
/// rather than inside each solver crate, so that `phyz-rigid`, `phyz-em`, and
/// friends keep zero knowledge of coupling. The trait is deliberately small: a
/// new domain needs `solver_type`, `natural_dt`, `time`, `apply_external`,
/// `advance`, and `energy`.
pub trait Solver {
    /// Which domain this is.
    fn solver_type(&self) -> SolverType;

    /// The largest timestep this solver is stable and accurate at, in seconds.
    ///
    /// For an FDTD grid this is the CFL limit; for a rigid-body integrator it
    /// is the configured `model.dt`. The coupling driver uses these to build a
    /// subcycling schedule rather than running every domain at the smallest dt.
    fn natural_dt(&self) -> f64;

    /// Current simulation time (s).
    fn time(&self) -> f64;

    /// Queue an external input, to be consumed by the next [`Self::advance`].
    fn apply_external(&mut self, input: ExternalInput);

    /// Advance the domain by `dt`, consuming and clearing queued external input.
    ///
    /// Implementations may subcycle internally at [`Self::natural_dt`]; the
    /// requested `dt` is the interval the coupling layer wants covered.
    fn advance(&mut self, dt: f64);

    /// Total energy currently held by this domain (J).
    fn energy(&self) -> f64;

    /// Discrete sites this domain exposes for coupling. Empty for pure fields.
    fn sites(&self) -> Vec<CouplingSite> {
        Vec::new()
    }

    /// Field values at a world-frame point. Zero for domains that carry no field.
    fn field_at(&self, _position: &Vec3) -> FieldSample {
        FieldSample::zero()
    }

    /// Total linear momentum held by this domain (kg·m/s).
    fn momentum(&self) -> Vec3 {
        Vec3::zeros()
    }
}
