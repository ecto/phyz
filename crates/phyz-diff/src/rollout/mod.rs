//! Differentiable rollouts: exact parameter and contact-surface adjoints.
//!
//! Mirror of the umbrella crate's `phyz::diff` module for the split-crate
//! world (`phyz_model::Model` instead of `phyz::Model`), following the same
//! duplication convention as the rest of the tree — the two copies differ
//! only in import paths.
//!
//! One backward pass of the discrete trajectory adjoint yields, for a
//! semi-implicit Euler rollout with a final-state objective:
//!
//! 1. **`dJ/dπ`** per body — the 10 spatial-inertia scalars
//!    `[m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`, exact (dual-number
//!    lanes through a scalar-generic ABA; no finite differences). This is
//!    the parameter sensitivity `StepJacobians` never covered.
//! 2. **`∂J/∂x`** per collision-mesh vertex under the differentiable
//!    per-vertex ground-penalty contact model — the cotangent a CAD-side
//!    surface pullback (vcad's `surface_gradient`) consumes.
//!
//! See [`step`] and [`adjoint`] for the forward-model and contract details.

pub mod adjoint;
pub mod step;

pub use adjoint::{
    AdjointGradients, AdjointRollout, ContactSetup, FinalStateObjective, adjoint_rollout_gradient,
    rollout_objective,
};
pub use step::{CollisionMesh, GroundContact, N_INERTIA_PARAMS, inertia_params};
