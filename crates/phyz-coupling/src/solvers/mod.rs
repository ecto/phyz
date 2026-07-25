//! [`crate::Solver`] adapters for the workspace's real solver crates.
//!
//! Adapters live here rather than in the solver crates so the physics crates
//! stay free of any coupling dependency. Implemented so far:
//!
//! | Domain | Adapter | Backing crate |
//! |---|---|---|
//! | Rigid body | [`RigidSolver`] | `phyz-rigid` (ABA + semi-implicit Euler) |
//! | Electromagnetic | [`EmSolver`] | `phyz-em` (Yee-grid FDTD) |
//!
//! `phyz-particle`, `phyz-md`, `phyz-lbm`, and `phyz-gravity` do **not** have
//! adapters yet. The trait was sized to fit them — each has an advance step,
//! discrete carriers or a field, and a natural timestep — but nothing here has
//! been validated against them.

pub mod em;
pub mod rigid;

pub use em::EmSolver;
pub use rigid::{BodyCoupling, RigidSolver};
