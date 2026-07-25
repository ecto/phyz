//! Regression guard over the validation suite.
//!
//! The suite is expected to contain failures — those are findings about the
//! solvers, recorded here as an explicit baseline. The test fails when a
//! benchmark that currently passes starts failing, or when a benchmark in the
//! baseline starts passing (in which case the baseline should shrink).
//!
//! The full suite takes ~30 s in release, so it is `#[ignore]`d by default:
//!
//! ```text
//! cargo test --release -p phyz-validate -- --ignored --nocapture
//! ```

use phyz_validate::Status;

/// Benchmarks that fail against the current solver implementations.
///
/// Every entry is a defect in the solver, documented in the `notes` of the
/// corresponding [`phyz_validate::Validation`]. Do not add to this list to make
/// a test green — fix the solver, or record why the benchmark is wrong.
const KNOWN_FAILURES: &[&str] = &[
    // 1PN acceleration disagrees with EIH; integrated precession is 14.3″/century
    // against the general-relativistic 42.998″/century, and does not converge.
    "gravity.pn.mercury_precession",
    "gravity.pn.mercury_convergence",
    // `PmlLayer` has no magnetic loss term, so it is a lossy slab, not a matched
    // layer; best measured reflection is ≈ −12 dB against ≈ −60 dB for a real PML.
    "em.absorbing_boundary_reflection",
    // Full-node bounce-back and equilibrium velocity boundaries are first order.
    "lbm.poiseuille.convergence",
    "lbm.cavity_re100.u.n33",
    "lbm.cavity_re100.v.n33",
    "lbm.cavity_re100.u.n65",
    "lbm.cavity_re100.v.n65",
    "lbm.cavity_re100.convergence",
    // `MdSystem::step` drops the first half-kick because forces are never
    // initialised; the start-up error is O(Δt) instead of O(Δt²).
    "md.startup_consistency",
];

#[test]
#[ignore = "full physics suite; run with --release -- --ignored"]
fn validation_suite_matches_the_recorded_baseline() {
    let report = phyz_validate::run_all();
    print!("{}", report.to_console());

    let mut unexpected_failures = Vec::new();
    let mut unexpected_passes = Vec::new();
    for e in report.entries() {
        let known = KNOWN_FAILURES.contains(&e.id.as_str());
        match (e.status, known) {
            (Status::Fail, false) => unexpected_failures.push(e.id.clone()),
            (Status::Pass, true) => unexpected_passes.push(e.id.clone()),
            _ => {}
        }
    }

    assert!(
        unexpected_failures.is_empty(),
        "benchmarks that used to pass now fail: {unexpected_failures:?}"
    );
    assert!(
        unexpected_passes.is_empty(),
        "benchmarks in KNOWN_FAILURES now pass — remove them from the baseline: \
         {unexpected_passes:?}"
    );
}

#[test]
fn report_renders_without_panicking() {
    let mut suite = phyz_validate::Suite::new("smoke");
    suite.push(
        phyz_validate::Validation::new(
            "smoke.a",
            "a",
            "c",
            "ref",
            "metric",
            1.0,
            2.0,
            phyz_validate::ErrorKind::Relative,
            0.1,
        )
        .with_convergence(phyz_validate::Convergence::fit(
            "dt",
            vec![(1.0, 1.0), (0.5, 0.25)],
            2.0,
            0.1,
        )),
    );
    let mut r = phyz_validate::Report::new();
    r.push(suite);
    assert!(r.to_markdown().contains("smoke.a"));
    assert!(r.to_json().contains("\"status\": \"FAIL\""));
    assert!(r.to_console().contains("FAIL"));
}
