//! Differentiable rollouts: exact parameter and contact-surface adjoints.
//!
//! This module closes the two sensitivity channels a CAD-side consumer (the
//! vcad differentiable seam) needs from the physics engine:
//!
//! 1. **Inertia-parameter adjoint** — `dJ/dπ` of a rollout objective with
//!    respect to every body's 10 spatial-inertia scalars
//!    `[m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`, exactly (dual-number
//!    lanes through a scalar-generic ABA, no finite differences). This
//!    replaces the ~20-rollouts-per-body central-FD factor in vcad's
//!    `rollout_gradient`.
//! 2. **Contact vertex adjoint** — `∂J/∂x` on the vertices of a body's
//!    collision mesh when ground-contact forces act *during* the rollout,
//!    under the differentiable per-vertex penalty contact model of
//!    [`step`]. This is the cotangent vcad's `surface_gradient` prices
//!    through the CAD seam.
//!
//! Both come out of **one** backward pass of the discrete trajectory adjoint
//! in [`adjoint`] (reverse over steps, exact dual-number Jacobians within
//! each step). See the module docs of [`step`] and [`adjoint`] for the
//! forward-model and contract details.

pub mod adjoint;
pub mod step;

pub use adjoint::{
    AdjointGradients, AdjointRollout, ContactSetup, FinalStateObjective, adjoint_rollout_gradient,
    rollout_objective,
};
pub use step::{CollisionMesh, GroundContact, N_INERTIA_PARAMS, inertia_params};
