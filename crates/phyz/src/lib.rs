//! phyz — Multi-physics differentiable simulation engine.
//!
//! This crate is a thin umbrella over the **rigid-body stack**: every type and
//! function below lives in one of the focused `phyz-*` crates and is
//! re-exported here, so `cargo add phyz` gives you that stack behind a single
//! dependency. The re-exported modules are aliases, not copies, so a fix in
//! `phyz-collision` is a fix for `phyz::collision` by construction. [`sim`] is
//! the one module with code of its own: the time loop, which none of the
//! focused crates own.
//!
//! It does **not** cover the whole workspace. `phyz-gpu`, `phyz-particle`,
//! `phyz-lbm`, `phyz-em`, `phyz-md` and the rest are separate crates you add
//! explicitly.
//!
//! # Features
//!
//! - `collision` (default) — GJK/EPA narrow phase and broad-phase spatial hashing.
//! - `contact` (default) — soft/penalty contact dynamics; implies `collision`.
//! - `diff` (default) — per-step Jacobians and the trajectory adjoint. Only
//!   the adjoint ([`diff::adjoint_rollout_gradient`]) and the symbolic path
//!   are derivative-exact; see [`phyz_diff`] for which is which.
//!
//! `math`, `model`, and `rigid` are always present.

#![warn(missing_docs)]

// Compile every ```rust block in the crate README and the repository README
// as doc-tests, so the documented API cannot drift from the real one.
// `cfg(doctest)` keeps both out of the rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct CrateReadmeDocTests;

#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
pub struct RepoReadmeDocTests;

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
    #[allow(deprecated)]
    pub use phyz_diff::analytical_step_jacobians;
    pub use phyz_diff::rollout::{
        AdjointGradients, AdjointRollout, CollisionMesh, ContactSetup, DofLayout,
        FinalStateObjective, GroundContact, N_INERTIA_PARAMS, adjoint_rollout_gradient,
        inertia_params, joint_nq, rollout_objective,
    };
    pub use phyz_diff::rollout::{adjoint, step};
    pub use phyz_diff::{
        StepJacobians, finite_diff_jacobians, rollout, semi_implicit_step_jacobians, symbolic,
    };
    // The unified adjoint through the convex contact solve — the same
    // contact model [`crate::Simulator::step_with_contacts`] integrates.
    pub use phyz_diff::{
        ConvexAdjointError, ConvexAdjointGradients, ConvexContactRollout, contact_adjoint,
        convex_adjoint_gradient, convex_rollout_objective,
    };
}

// Reproducibility: trajectory fingerprints, ulp perturbation, and the
// chaos-vs-bug measurement. Documented on the module itself — an outer doc
// comment here would be resolved in *this* module's scope, so its intra-doc
// links to `RolloutHasher` and friends would not resolve.
pub mod determinism;

/// Integrated time stepping: the [`Simulator`] driver and its solvers.
///
/// The only module here that isn't a re-export — it composes forward
/// dynamics, contact and integration, which no single sub-crate owns.
#[cfg(all(feature = "contact", feature = "diff"))]
pub mod sim;
#[cfg(all(feature = "contact", feature = "diff"))]
pub use sim::{Rk4Solver, SemiImplicitEulerSolver, Simulator, Solver};

// Re-export core types at crate root for convenience
#[cfg(feature = "collision")]
pub use collision::{
    AABB, Collision, Ray, RayHit, epa_penetration, epa_penetration_rot, gjk_distance,
    gjk_distance_rot, ray_cast, ray_intersect, sweep_and_prune,
};
#[cfg(feature = "contact")]
pub use contact::{
    ContactMaterial, ContactProblem, ContactSolverConfig, find_contacts, find_ground_contacts,
    solve_contacts,
};
pub use math::{DMat, DVec, GRAVITY, Mat3, Mat4, Vec3, skew};
pub use math::{Quat, SpatialInertia, SpatialMat, SpatialTransform, SpatialVec};
pub use model::{
    Actuator, Body, GeomInstance, Geometry, Joint, JointType, Model, ModelBuilder, State,
};
pub use rigid::{
    BodyKinematics, IkConfig, IkGoal, IkGoalKind, IkSolution, aba, aba_with_external_forces,
    body_wrenches, crba, forward_kinematics, forward_kinematics_acc, rnea, solve_ik,
};
