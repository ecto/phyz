//! Harness crate for the repository-root `examples/` directory.
//!
//! This crate has no API of its own. It exists so that every file under
//! `examples/` is a real cargo target, which means
//! `cargo build --workspace --examples` type-checks all of them in CI and the
//! documented API cannot silently rot.
//!
//! Stepping lives in [`phyz::sim`] — `Simulator`, `Solver`,
//! `SemiImplicitEulerSolver` and `Rk4Solver`. The examples use those directly.

#![warn(missing_docs)]
