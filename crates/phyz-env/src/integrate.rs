//! Re-export of the canonical joint-aware integrator.
//!
//! This lived here first, then moved to `phyz-rigid` so the GPU backend and the
//! CPU dynamics could share one definition of what "advance the state" means.
//! See [`phyz_rigid::integrate`] for the free-joint `q`/`v` layout discussion.

pub use phyz_rigid::integrate::semi_implicit_euler;
