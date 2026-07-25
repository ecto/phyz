//! Closed-form physics benchmarks for every phyz solver.
//!
//! `phyz-regge` and `phyz-quantum` validate analytic gradients against finite
//! differences across ~120 tests. This crate extends that standard to the
//! classical solvers: each benchmark integrates a phyz solver on a problem with
//! a known analytic solution or published reference data, reports a
//! **quantitative error metric**, and — wherever a discretization parameter
//! exists — demonstrates the **expected convergence order** under refinement.
//!
//! Tolerances are declared before the measurement and are never relaxed to make
//! a benchmark pass. A `FAIL` in the emitted report is a finding about the
//! solver.
//!
//! ```no_run
//! let report = phyz_validate::run_all();
//! print!("{}", report.to_console());
//! ```

pub mod em;
pub mod gravity;
pub mod lbm;
pub mod md;
pub mod report;
pub mod rigid;

pub use report::{Convergence, ErrorKind, Report, Status, Suite, Validation};

/// Run every validation suite and collect the results.
pub fn run_all() -> Report {
    let mut r = Report::new();
    r.push(rigid::run());
    r.push(gravity::run());
    r.push(em::run());
    r.push(lbm::run());
    r.push(md::run());
    r
}
