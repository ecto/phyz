//! Differentiable rollouts: exact parameter and contact-surface adjoints.
//!
//! Re-exported by the umbrella crate as `phyz::diff`.
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
//!
//! # Which adjoint to use
//!
//! The contact model here is the module's **own** differentiable per-vertex
//! ground-penalty law — *not* the convex solve the forward simulator
//! (`phyz::Simulator::step_with_contacts`) integrates. It exists for the
//! vcad surface-gradient seam (`∂J/∂vertex`), which needs a smooth force per
//! vertex. For gradients of the physics the engine actually runs — state,
//! control and inertia channels through the convex contact solve — use
//! [`crate::contact_adjoint`] instead; its gradients match a finite
//! difference of the real forward path, and this module's do not (measured:
//! ~94% error on a box-drop mass gradient, `phyz`'s `diff_convex_contact.rs`
//! gate).

pub mod adjoint;
pub mod step;

pub use adjoint::{
    AdjointGradients, AdjointRollout, ContactSetup, FinalStateObjective, adjoint_rollout_gradient,
    rollout_objective,
};
pub use step::{
    CollisionMesh, DofLayout, GroundContact, N_INERTIA_PARAMS, inertia_params, joint_nq,
};
