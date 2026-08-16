//! XPBD — Extended Position-Based Dynamics — for cloth, soft bodies and cables.
//!
//! Implements Macklin, Müller & Chentanez, *"XPBD: Position-Based Simulation of
//! Compliant Constrained Dynamics"* (MIG 2016), with the substepping of Müller
//! et al., *"Small Steps in Physics Simulation"* (SCA 2020).
//!
//! The rest of phyz simulates rigid bodies with forces and accelerations.
//! This crate simulates deformable things by moving positions directly: a
//! [`ParticleSystem`] of point masses, a list of scalar [`Constraint`]s over
//! them, and a time loop ([`XpbdSolver`]) that predicts positions, projects the
//! constraints, and reads velocity back out of the position change. That gets
//! you cloth, tetrahedral soft bodies and cables from one solver, and it stays
//! stable at timesteps where an explicit force-based spring model of the same
//! stiffness would need to be orders of magnitude smaller.
//!
//! # What XPBD adds over PBD, concretely
//!
//! Plain PBD projects each constraint to `C = 0` and scales the correction by
//! a "stiffness" in `[0, 1]`. That stiffness is not a material property: the
//! same value gives a stiffer cloth at more iterations and a stiffer cloth at
//! smaller timesteps, because each of the `n` passes removes a fixed fraction
//! of the remaining error. You cannot state what material you simulated.
//!
//! XPBD replaces it with a compliance `α = 1/k` — the physical inverse
//! stiffness, in metres per newton for a distance constraint — and carries a
//! Lagrange multiplier `λ` per constraint through the substep:
//!
//! ```text
//! α̃  = α / h²
//! Δλ = (−C − α̃ λ) / (Σ wᵢ |∇Cᵢ|² + α̃)
//! Δxᵢ = wᵢ ∇Cᵢ Δλ
//! ```
//!
//! The `−α̃ λ` feedback term is the entire difference. With it, `λ` converges
//! to the true constraint force multiplier, and the converged deformation is
//! `α · f` and nothing else — independent of iteration count, and consistent
//! across timesteps. This crate's tests pin exactly that. Measured: a hanging
//! mass stretches its constraint by `m g α` to a relative error of 5.3e-12,
//! and the same scene solved with 1, 2, 4, 8, 16 and 32 iterations agrees to
//! **zero** relative difference — the converged positions are bit-identical
//! across a 32× range of iteration count.
//!
//! # Quick start
//!
//! A 1 kg mass hanging from a compliant spring, run to rest:
//!
//! ```
//! use phyz_math::Vec3;
//! use phyz_xpbd::{Constraint, ParticleSystem, XpbdSolver};
//!
//! let mut p = ParticleSystem::new();
//! let anchor = p.add_pinned(Vec3::new(0.0, 0.0, 0.0));
//! let bob = p.add(Vec3::new(0.0, -1.0, 0.0), 1.0);
//! // Compliance 1e-3 m/N is a spring of k = 1000 N/m.
//! let mut cs = vec![Constraint::distance(anchor, bob, 1.0, 1.0e-3)];
//!
//! let solver = XpbdSolver {
//!     dt: 1.0 / 60.0,
//!     damping: 20.0, // settle quickly rather than oscillate forever
//!     ..XpbdSolver::default()
//! };
//! for _ in 0..600 {
//!     solver.step(&mut p, &mut cs);
//! }
//!
//! // Static equilibrium: stretch = α · m g.
//! let stretch = (p.positions[bob] - p.positions[anchor]).norm() - 1.0;
//! let expected = 1.0e-3 * 1.0 * 9.81;
//! assert!((stretch - expected).abs() < 1e-6, "stretch {stretch}, expected {expected}");
//! ```
//!
//! # Determinism
//!
//! Bit-reproducible: pure `f64`, a single thread, and a fixed sweep over the
//! constraint slice in index order. There is no `HashMap` anywhere in the
//! solve. Because Gauss–Seidel is order-dependent, the *order* of the
//! constraint list is part of the input — reorder it and you get a different
//! (equally valid) trajectory.
//!
//! # What this crate does not do
//!
//! Stated plainly, because each of these is a thing a reader might reasonably
//! assume is here:
//!
//! * **No rigid-body coupling.** XPBD extends to rigid bodies with orientation
//!   and generalised inverse mass, and Macklin et al. (2019) do exactly that.
//!   This crate is particles only. Nothing here talks to `phyz-rigid`.
//! * **No collision of any kind.** No self-collision, no particle-particle, no
//!   collision against the rigid-body world. There is no broad phase here. A
//!   cloth will pass through itself and through everything else.
//! * **No friction.** It follows from having no contacts.
//! * **No GPU.** Single-threaded CPU. The Gauss–Seidel sweep is inherently
//!   sequential; a parallel version needs graph colouring or a Jacobi variant
//!   with under-relaxation, neither of which is implemented.
//! * **Bending is skipped at exactly-flat configurations.** The dihedral
//!   gradient has a removable-in-theory, noisy-in-floating-point singularity
//!   when the two triangles are coplanar; the constraint is skipped there.
//!   See [`constraint`] for the full explanation.
//! * **Volume preservation is per-tetrahedron and local.** It constrains each
//!   tet's volume independently, which is not the same as enforcing global
//!   incompressibility; a mesh can shuffle volume between neighbouring tets
//!   within solver tolerance.
//! * **Long-range stiffness converges slowly.** An `n`-link chain needs
//!   `O(n)` projections for information to reach the pin. Measured on a
//!   20-link chain at rest: the worst node sits 5.2e-3 m from its analytic
//!   equilibrium depth at 10 substeps × 1 iteration, and 3.0e-9 m at
//!   10 × 20. See [`XpbdSolver`].

#![warn(missing_docs)]

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

pub mod constraint;
pub mod particles;
pub mod solver;

pub use constraint::{Constraint, ConstraintKind, project, tet_volume};
pub use particles::ParticleSystem;
pub use solver::XpbdSolver;
