//! Physics settings shared by every library under comparison.
//!
//! The cross-library numbers are only meaningful if both engines are asked to
//! do the same job. This module is the single source of truth for that job,
//! and it is serialised into every result so a reader can check the claim
//! rather than trust it.
//!
//! Where an exact match is impossible — and for phyz vs Rapier it is, because
//! one is a reduced-coordinate penalty-contact integrator and the other is an
//! impulse-based constraint solver — the mismatch is recorded in
//! [`Settings::caveats`] instead of being quietly averaged away.

use serde::{Deserialize, Serialize};

/// Standard gravity magnitude (m/s²) used by every scene.
pub const GRAVITY: f64 = 9.81;

/// The settings a scene is run under.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Integration timestep (s).
    pub dt: f64,
    /// Gravity vector (m/s²), world frame, z-up.
    pub gravity: [f64; 3],
    /// Velocity-solver iterations, where the engine exposes such a knob.
    pub solver_iterations: usize,
    /// Contact parameters, `None` for contact-free scenes.
    pub contact: Option<ContactSettings>,
    /// Known, unavoidable differences between engines under these settings.
    pub caveats: Vec<String>,
}

/// Contact model parameters, in the terms each engine actually accepts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactSettings {
    /// phyz penalty stiffness (N/m).
    pub stiffness: f64,
    /// phyz penalty damping (N·s/m).
    pub damping: f64,
    /// Coulomb friction coefficient — directly comparable between engines.
    pub friction: f64,
    /// Coefficient of restitution — directly comparable between engines.
    pub restitution: f64,
}

impl Settings {
    /// Contact-free articulated scenes (pendulums, ant): dt = 1 ms.
    pub fn articulated(dt: f64) -> Self {
        Self {
            dt,
            gravity: [0.0, 0.0, -GRAVITY],
            solver_iterations: DEFAULT_SOLVER_ITERATIONS,
            contact: None,
            caveats: vec![
                "phyz integrates reduced coordinates with ABA + semi-implicit Euler and has \
                 no velocity solver; `solver_iterations` is what we ask of engines that \
                 have such a knob, and is inert for phyz."
                    .into(),
            ],
        }
    }

    /// The box-stack contact scene.
    pub fn contact(dt: f64) -> Self {
        Self {
            dt,
            gravity: [0.0, 0.0, -GRAVITY],
            solver_iterations: DEFAULT_SOLVER_ITERATIONS,
            contact: Some(ContactSettings {
                stiffness: CONTACT_STIFFNESS,
                damping: CONTACT_DAMPING,
                friction: CONTACT_FRICTION,
                restitution: 0.0,
            }),
            caveats: vec![
                "phyz contact is a compliant penalty force parameterised by stiffness and \
                 damping. Only `friction` and `restitution` translate to a constraint-based \
                 solver; `stiffness`/`damping` have no analogue in one."
                    .into(),
                "The phyz box stack runs both contact paths — the ground plane query and \
                 sweep-and-prune + GJK/EPA between boxes — so the measured cost includes \
                 narrow-phase collision."
                    .into(),
                "phyz applies contact wrenches as world-frame forces on free bodies whose \
                 orientation starts at identity. For a settled stack this is exact; for a \
                 tumbling scene it would not be. The stack is initialised upright."
                    .into(),
            ],
        }
    }
}

/// Velocity-solver iterations requested of engines that have one.
pub const DEFAULT_SOLVER_ITERATIONS: usize = 4;
/// Penalty stiffness for the contact scenes (N/m).
pub const CONTACT_STIFFNESS: f64 = 1.0e5;
/// Penalty damping for the contact scenes (N·s/m).
pub const CONTACT_DAMPING: f64 = 1.0e3;
/// Coulomb friction coefficient for the contact scenes.
pub const CONTACT_FRICTION: f64 = 0.5;
/// Default timestep for articulated scenes (s).
pub const DT_ARTICULATED: f64 = 1.0e-3;
/// Default timestep for contact scenes (s).
pub const DT_CONTACT: f64 = 2.0e-3;
