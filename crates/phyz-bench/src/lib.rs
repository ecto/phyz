//! Benchmark harness for phyz.
//!
//! Four dimensions, one report:
//!
//! 1. **Single-sim throughput** — steps/sec on pendulum, double pendulum, the
//!    MJCF ant, and a box-stack contact scene.
//! 2. **Batched throughput** — environment-steps/sec on the GPU path, swept
//!    across batch size.
//! 3. **Gradient throughput** — the cost of an adjoint gradient rollout
//!    relative to the forward rollout it differentiates.
//! 4. **Numerical quality** — energy drift over a long rollout at several
//!    timesteps, reported next to speed rather than instead of it.
//!
//! Plus a cross-library comparison against Rapier on equivalent scenes, with
//! the settings and the unavoidable model differences recorded in the output.
//!
//! Results are emitted as JSON ([`report::Report`]) for tracking over time and
//! as Markdown for `BENCHMARKS.md`.

pub mod meta;
pub mod report;
pub mod scenes;
pub mod settings;
pub mod suites;
pub mod timing;

pub use report::{Metric, Record, Report, Suite};
pub use scenes::{PhyzSim, Scene};
pub use settings::Settings;
pub use timing::{Budget, Timing, measure};
