//! Constrained multibody dynamics with **kinematic loops**.
//!
//! `phyz-rigid` is a reduced-coordinate Featherstone solver over a kinematic
//! *tree*. A tree has one path from the world to each body, so a closed chain
//! — four-bar, slider-crank, delta robot, any parallel manipulator — is not
//! slow to simulate there, it is **unrepresentable**. This crate closes that
//! gap the standard way: model the mechanism as a spanning tree, cut each
//! loop, and re-impose the cut joint as an explicit constraint solved
//! alongside the dynamics.
//!
//! The solve is **Proximal ADMM** on the equality-constrained least-constraint
//! problem, in the spirit of NVIDIA/Disney Newton's `kamino` solver
//! (arXiv:2504.19771). See [`solver`] for the derivation, the reason a direct
//! KKT factorization is not used, and the stabilization choice.
//!
//! ```
//! use phyz_loop::{Anchor, LoopConstraint, LoopConstraintSet, LoopSolverConfig, solve};
//! use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
//! use phyz_model::ModelBuilder;
//!
//! let link = SpatialInertia::new(
//!     1.0,
//!     Vec3::new(0.5, 0.0, 0.0),
//!     Mat3::from_diagonal(&Vec3::new(1e-3, 1e-3, 1e-3)),
//! );
//! let model = ModelBuilder::new()
//!     .gravity(Vec3::new(0.0, -9.81, 0.0))
//!     .dt(1e-3)
//!     .add_revolute_body("crank", -1, SpatialTransform::identity(), link)
//!     .build();
//! let state = model.default_state();
//!
//! // Pin the far end of the single link to where it currently is: a
//! // one-body "loop" that should hold the link completely still.
//! let mut set = LoopConstraintSet::new();
//! set.push(LoopConstraint::point(
//!     "pin",
//!     Anchor::body(0, Vec3::new(1.0, 0.0, 0.0)),
//!     Anchor::world(Vec3::new(1.0, 0.0, 0.0)),
//! ));
//!
//! let cfg = LoopSolverConfig::for_model(&model);
//! let sol = solve(&model, &state, &set, &cfg);
//! assert!(sol.converged, "residual {}", sol.residual);
//! assert!(sol.qdd[0].abs() < 1e-6);
//! ```
//!
//! # Scope
//!
//! In: loop-closure constraints (3-row point, 6-row weld), their Jacobians,
//! the constrained forward-dynamics solve, Baumgarte stabilization, mobility
//! analysis.
//!
//! Out, deliberately and not partially:
//!
//! - **Contact and friction.** `phyz-contact` owns the inequality/cone side of
//!   constrained dynamics. Solving frictional contact *inside* this loop
//!   solver would mean two crates with two different answers about the same
//!   physics, which this repo has been burned by before. A combined
//!   loop-plus-contact solve is a real thing to want; it is not here.
//! - **GPU.** CPU f64 only.
//! - **Actuation and joint limits.** No inequality constraints of any kind.
//!   Actuator forces enter only through ABA's `a_free`.
//!
//! # Determinism
//!
//! Pure `f64`, no `HashMap` anywhere in the solve, fixed iteration order over
//! `Vec`s, and the primal system is explicitly symmetrized before factoring so
//! the result cannot depend on which triangle Cholesky reads. `tests/determinism.rs`
//! asserts bit-identical output across two runs of a full rollout.

#![warn(missing_docs)]

pub mod constraint;
pub mod mobility;
pub mod solver;

pub use constraint::{
    Anchor, Attachment, LoopConstraint, LoopConstraintSet, LoopKind, LoopSystem, assemble,
};
pub use mobility::{MobilitySpace, grubler, mobility};
pub use solver::{
    LoopSolution, LoopSolverConfig, Stabilization, constraint_rank, project_velocity, solve, step,
};
