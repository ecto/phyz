//! phyz — Multi-physics differentiable simulation engine.
//!
//! This crate is a thin umbrella: every type and function below lives in one
//! of the focused `phyz-*` crates and is re-exported here so that
//! `cargo add phyz` gives you the whole engine behind a single dependency.
//! There is no code of its own — the modules are aliases, not copies, so a
//! fix in `phyz-collision` is a fix for `phyz::collision` by construction.
//!
//! # Features
//!
//! - `collision` (default) — GJK/EPA narrow phase and broad-phase spatial hashing.
//! - `contact` (default) — soft/penalty contact dynamics; implies `collision`.
//! - `diff` (default) — analytical Jacobians and the trajectory adjoint.
//!
//! `math`, `model`, and `rigid` are always present.

// Sub-crates re-exported under their real names, so downstream code can name
// them either way (`phyz::phyz_rigid::aba` or `phyz::rigid::aba`).
pub use phyz_math;
pub use phyz_model;
pub use phyz_rigid;

/// Vectors, matrices, quaternions and spatial algebra ([`phyz_math`]).
pub use phyz_math as math;
/// Model and state types ([`phyz_model`]).
pub use phyz_model as model;
/// Featherstone articulated rigid body dynamics ([`phyz_rigid`]).
pub use phyz_rigid as rigid;

#[cfg(feature = "collision")]
pub use phyz_collision;
/// GJK/EPA collision detection and broad phase ([`phyz_collision`]).
#[cfg(feature = "collision")]
pub use phyz_collision as collision;

#[cfg(feature = "contact")]
pub use phyz_contact;
/// Soft contact dynamics and penalty-based contact resolution ([`phyz_contact`]).
#[cfg(feature = "contact")]
pub use phyz_contact as contact;

#[cfg(feature = "diff")]
pub use phyz_diff;

/// Differentiation: Jacobians of the dynamics and the trajectory adjoint.
///
/// Flattens [`phyz_diff`]'s `rollout` submodule to the module root so that
/// `phyz::diff::{adjoint, step}` and `phyz::diff::AdjointRollout` resolve
/// alongside `phyz::diff::finite_diff_jacobians`.
#[cfg(feature = "diff")]
pub mod diff {
    pub use phyz_diff::rollout::{
        AdjointGradients, AdjointRollout, CollisionMesh, ContactSetup, FinalStateObjective,
        GroundContact, N_INERTIA_PARAMS, adjoint_rollout_gradient, inertia_params,
        rollout_objective,
    };
    pub use phyz_diff::rollout::{adjoint, step};
    pub use phyz_diff::{
        StepJacobians, analytical_step_jacobians, finite_diff_jacobians, rollout, symbolic,
    };
}

// Re-export core types at crate root for convenience
#[cfg(feature = "collision")]
pub use collision::{
    AABB, Collision, epa_penetration, epa_penetration_rot, gjk_distance, gjk_distance_rot,
    sweep_and_prune,
};
#[cfg(feature = "contact")]
pub use contact::{
    ContactMaterial, compute_contact_force, compute_contact_force_implicit, contact_forces,
    contact_forces_implicit, find_contacts, find_ground_contacts,
};
pub use math::{DMat, DVec, GRAVITY, Mat3, Mat4, Vec3, skew};
pub use math::{Quat, SpatialInertia, SpatialMat, SpatialTransform, SpatialVec};
pub use model::{Actuator, Body, Geometry, Joint, JointType, Model, ModelBuilder, State};
pub use rigid::{aba, aba_with_external_forces, crba, forward_kinematics, rnea};
