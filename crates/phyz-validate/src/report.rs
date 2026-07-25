//! Reporting harness: quantitative error metrics, convergence-order fits,
//! and Markdown/JSON emission suitable for publishing on phyz.dev.
//!
//! Every benchmark returns a [`Validation`] carrying the *measured* number, the
//! *analytic* number, and the error between them. Tolerances are declared up
//! front and never tuned after the fact — a `Fail` in the emitted report is a
//! finding about the solver, not about the benchmark.

use std::fmt::Write as _;

/// How the error between measured and expected is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// `|measured - expected| / |expected|`.
    Relative,
    /// `|measured - expected|`.
    Absolute,
}

impl ErrorKind {
    fn label(self) -> &'static str {
        match self {
            ErrorKind::Relative => "rel",
            ErrorKind::Absolute => "abs",
        }
    }
}

/// Outcome of a single validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Error is within the declared tolerance.
    Pass,
    /// Error exceeds the declared tolerance.
    Fail,
    /// Measured and reported, but no pass/fail claim is made (diagnostics).
    Report,
}

impl Status {
    fn label(self) -> &'static str {
        match self {
            Status::Pass => "PASS",
            Status::Fail => "FAIL",
            Status::Report => "REPORT",
        }
    }
}

/// A convergence study: error as a function of a refinement parameter `h`
/// (timestep, grid spacing, ...), fitted to `error ~ C h^p`.
#[derive(Debug, Clone)]
pub struct Convergence {
    /// Name of the refinement parameter (e.g. `"dt"`, `"dx"`).
    pub parameter: String,
    /// `(h, error)` samples, coarse to fine.
    pub samples: Vec<(f64, f64)>,
    /// Least-squares slope of `log(error)` vs `log(h)`.
    pub measured_order: f64,
    /// Theoretically expected order.
    pub expected_order: f64,
    /// Absolute tolerance on `|measured_order - expected_order|`.
    pub order_tolerance: f64,
}

impl Convergence {
    /// Fit `error ~ C h^p` by least squares on the log-log samples.
    pub fn fit(
        parameter: &str,
        samples: Vec<(f64, f64)>,
        expected_order: f64,
        order_tolerance: f64,
    ) -> Self {
        let usable: Vec<(f64, f64)> = samples
            .iter()
            .copied()
            .filter(|&(h, e)| h > 0.0 && e > 0.0 && e.is_finite())
            .collect();

        let measured_order = if usable.len() < 2 {
            f64::NAN
        } else {
            let n = usable.len() as f64;
            let sx: f64 = usable.iter().map(|(h, _)| h.ln()).sum();
            let sy: f64 = usable.iter().map(|(_, e)| e.ln()).sum();
            let sxx: f64 = usable.iter().map(|(h, _)| h.ln() * h.ln()).sum();
            let sxy: f64 = usable.iter().map(|(h, e)| h.ln() * e.ln()).sum();
            (n * sxy - sx * sy) / (n * sxx - sx * sx)
        };

        Self {
            parameter: parameter.to_string(),
            samples,
            measured_order,
            expected_order,
            order_tolerance,
        }
    }

    /// Whether the fitted order matches theory within tolerance.
    pub fn is_ok(&self) -> bool {
        self.measured_order.is_finite()
            && (self.measured_order - self.expected_order).abs() <= self.order_tolerance
    }
}

/// A single validated quantity.
#[derive(Debug, Clone)]
pub struct Validation {
    /// Stable machine-readable id, e.g. `"em.yee_dispersion"`.
    pub id: String,
    /// Human-readable benchmark name.
    pub name: String,
    /// Solver crate under test.
    pub crate_name: String,
    /// Literature or closed-form source for the expected value.
    pub reference: String,
    /// Name of the quantity being compared, with units.
    pub metric: String,
    /// Value produced by the phyz solver.
    pub measured: f64,
    /// Closed-form / published value.
    pub expected: f64,
    /// How `error` is computed.
    pub error_kind: ErrorKind,
    /// The error metric.
    pub error: f64,
    /// Declared, pre-registered tolerance on `error`.
    pub tolerance: f64,
    /// Pass/fail/diagnostic.
    pub status: Status,
    /// Optional convergence study attached to this benchmark.
    pub convergence: Option<Convergence>,
    /// Free-form observations (including known implementation defects).
    pub notes: Vec<String>,
}

impl Validation {
    /// Build a validation, computing the error and status from the inputs.
    pub fn new(
        id: &str,
        name: &str,
        crate_name: &str,
        reference: &str,
        metric: &str,
        measured: f64,
        expected: f64,
        error_kind: ErrorKind,
        tolerance: f64,
    ) -> Self {
        let error = match error_kind {
            ErrorKind::Relative => {
                if expected == 0.0 {
                    (measured - expected).abs()
                } else {
                    (measured - expected).abs() / expected.abs()
                }
            }
            ErrorKind::Absolute => (measured - expected).abs(),
        };
        let status = if error.is_finite() && error <= tolerance {
            Status::Pass
        } else {
            Status::Fail
        };
        Self {
            id: id.to_string(),
            name: name.to_string(),
            crate_name: crate_name.to_string(),
            reference: reference.to_string(),
            metric: metric.to_string(),
            measured,
            expected,
            error_kind,
            error,
            tolerance,
            status,
            convergence: None,
            notes: Vec::new(),
        }
    }

    /// Mark this entry as a diagnostic measurement with no pass/fail claim.
    pub fn diagnostic(mut self) -> Self {
        self.status = Status::Report;
        self
    }

    /// Attach a convergence study. A failed order fit downgrades a `Pass`.
    pub fn with_convergence(mut self, c: Convergence) -> Self {
        if self.status == Status::Pass && !c.is_ok() {
            self.status = Status::Fail;
        }
        self.convergence = Some(c);
        self
    }

    /// Attach a note.
    pub fn note(mut self, s: impl Into<String>) -> Self {
        self.notes.push(s.into());
        self
    }

    /// Whether this entry counts as a failure.
    pub fn failed(&self) -> bool {
        self.status == Status::Fail
    }
}

/// A collection of validations from one solver domain.
#[derive(Debug, Clone, Default)]
pub struct Suite {
    /// Domain name (e.g. `"Electromagnetics (FDTD)"`).
    pub domain: String,
    /// Validations in this suite.
    pub entries: Vec<Validation>,
}

impl Suite {
    /// Create an empty suite for a domain.
    pub fn new(domain: &str) -> Self {
        Self {
            domain: domain.to_string(),
            entries: Vec::new(),
        }
    }

    /// Append a validation.
    pub fn push(&mut self, v: Validation) {
        self.entries.push(v);
    }

    /// Validations that failed.
    pub fn failures(&self) -> Vec<&Validation> {
        self.entries.iter().filter(|v| v.failed()).collect()
    }
}

/// The full validation report across all solver domains.
#[derive(Debug, Clone, Default)]
pub struct Report {
    /// Suites, in the order they were run.
    pub suites: Vec<Suite>,
}

impl Report {
    /// Create an empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a suite.
    pub fn push(&mut self, s: Suite) {
        self.suites.push(s);
    }

    /// All entries, flattened.
    pub fn entries(&self) -> impl Iterator<Item = &Validation> {
        self.suites.iter().flat_map(|s| s.entries.iter())
    }

    /// (passed, failed, reported) counts.
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut p = 0;
        let mut f = 0;
        let mut r = 0;
        for e in self.entries() {
            match e.status {
                Status::Pass => p += 1,
                Status::Fail => f += 1,
                Status::Report => r += 1,
            }
        }
        (p, f, r)
    }

    /// Render the report as Markdown, ready to drop onto phyz.dev.
    pub fn to_markdown(&self) -> String {
        /// Escape `|` so metric strings can never break a table row.
        fn cell(s: &str) -> String {
            s.replace('|', "\\|")
        }
        let mut s = String::new();
        let (pass, fail, rep) = self.counts();

        s.push_str("# phyz solver validation\n\n");
        s.push_str(
            "Every entry compares a phyz solver against a closed-form solution or published \
             reference data, and reports a quantitative error — not a pass/fail bit. Tolerances \
             are declared before the measurement is taken and are never relaxed to make a \
             benchmark pass.\n\n",
        );
        let _ = writeln!(
            s,
            "**{pass} passed · {fail} failed · {rep} reported (diagnostic)**\n"
        );

        if fail > 0 {
            s.push_str("## Failures\n\n");
            s.push_str("| Benchmark | Metric | Measured | Expected | Error | Tolerance |\n");
            s.push_str("|---|---|---:|---:|---:|---:|\n");
            for e in self.entries().filter(|e| e.failed()) {
                let _ = writeln!(
                    s,
                    "| `{}` | {} | {:.6e} | {:.6e} | {:.3e} ({}) | {:.3e} |",
                    cell(&e.id),
                    cell(&e.metric),
                    e.measured,
                    e.expected,
                    e.error,
                    e.error_kind.label(),
                    e.tolerance
                );
            }
            s.push('\n');
        }

        for suite in &self.suites {
            let _ = writeln!(s, "## {}\n", suite.domain);
            for e in &suite.entries {
                let _ = writeln!(s, "### {} — {}\n", e.name, e.status.label());
                let _ = writeln!(s, "- **crate**: `{}`", e.crate_name);
                let _ = writeln!(s, "- **id**: `{}`", e.id);
                let _ = writeln!(s, "- **reference**: {}", e.reference);
                let _ = writeln!(s, "- **metric**: {}", e.metric);
                let _ = writeln!(s, "- **measured**: {:.9e}", e.measured);
                let _ = writeln!(s, "- **expected**: {:.9e}", e.expected);
                let _ = writeln!(
                    s,
                    "- **error**: {:.4e} ({}) — tolerance {:.4e}",
                    e.error,
                    e.error_kind.label(),
                    e.tolerance
                );
                if let Some(c) = &e.convergence {
                    let _ = writeln!(
                        s,
                        "- **convergence in {}**: measured order p = {:.3} (expected {:.1} ± {:.1}) — {}",
                        c.parameter,
                        c.measured_order,
                        c.expected_order,
                        c.order_tolerance,
                        if c.is_ok() { "OK" } else { "MISMATCH" }
                    );
                    s.push_str("\n  | ");
                    let _ = write!(s, "{} | error | ratio |\n  |---:|---:|---:|\n", c.parameter);
                    let mut prev: Option<f64> = None;
                    for &(h, err) in &c.samples {
                        let ratio = prev.map(|p| p / err).unwrap_or(f64::NAN);
                        let _ = writeln!(s, "  | {h:.6e} | {err:.6e} | {ratio:.3} |");
                        prev = Some(err);
                    }
                }
                for n in &e.notes {
                    let _ = writeln!(s, "- _note_: {n}");
                }
                s.push('\n');
            }
        }
        s
    }

    /// Render the report as JSON (hand-rolled; the crate has no serde dependency).
    pub fn to_json(&self) -> String {
        fn esc(s: &str) -> String {
            s.replace('\\', "\\\\").replace('"', "\\\"")
        }
        fn num(x: f64) -> String {
            if x.is_finite() {
                format!("{x:.12e}")
            } else {
                "null".to_string()
            }
        }

        let mut s = String::from("{\n  \"suites\": [\n");
        for (si, suite) in self.suites.iter().enumerate() {
            let _ = write!(
                s,
                "    {{\n      \"domain\": \"{}\",\n      \"entries\": [\n",
                esc(&suite.domain)
            );
            for (ei, e) in suite.entries.iter().enumerate() {
                let _ = write!(
                    s,
                    "        {{\n          \"id\": \"{}\",\n          \"name\": \"{}\",\n          \"crate\": \"{}\",\n          \"reference\": \"{}\",\n          \"metric\": \"{}\",\n          \"measured\": {},\n          \"expected\": {},\n          \"error\": {},\n          \"error_kind\": \"{}\",\n          \"tolerance\": {},\n          \"status\": \"{}\"",
                    esc(&e.id),
                    esc(&e.name),
                    esc(&e.crate_name),
                    esc(&e.reference),
                    esc(&e.metric),
                    num(e.measured),
                    num(e.expected),
                    num(e.error),
                    e.error_kind.label(),
                    num(e.tolerance),
                    e.status.label()
                );
                if let Some(c) = &e.convergence {
                    let _ = write!(
                        s,
                        ",\n          \"convergence\": {{\n            \"parameter\": \"{}\",\n            \"measured_order\": {},\n            \"expected_order\": {},\n            \"order_tolerance\": {},\n            \"samples\": [",
                        esc(&c.parameter),
                        num(c.measured_order),
                        num(c.expected_order),
                        num(c.order_tolerance)
                    );
                    for (i, &(h, err)) in c.samples.iter().enumerate() {
                        let _ = write!(
                            s,
                            "{}[{}, {}]",
                            if i == 0 { "" } else { ", " },
                            num(h),
                            num(err)
                        );
                    }
                    s.push_str("]\n          }");
                }
                if !e.notes.is_empty() {
                    s.push_str(",\n          \"notes\": [");
                    for (i, n) in e.notes.iter().enumerate() {
                        let _ = write!(s, "{}\"{}\"", if i == 0 { "" } else { ", " }, esc(n));
                    }
                    s.push(']');
                }
                s.push_str("\n        }");
                if ei + 1 < suite.entries.len() {
                    s.push(',');
                }
                s.push('\n');
            }
            s.push_str("      ]\n    }");
            if si + 1 < self.suites.len() {
                s.push(',');
            }
            s.push('\n');
        }
        let (p, f, r) = self.counts();
        let _ = write!(
            s,
            "  ],\n  \"summary\": {{ \"passed\": {p}, \"failed\": {f}, \"reported\": {r} }}\n}}\n"
        );
        s
    }

    /// Compact one-line-per-entry console summary.
    pub fn to_console(&self) -> String {
        let mut s = String::new();
        for suite in &self.suites {
            let _ = writeln!(s, "\n=== {} ===", suite.domain);
            for e in &suite.entries {
                let _ = writeln!(
                    s,
                    "  [{:6}] {:<38} {:<26} measured={:>13.6e} expected={:>13.6e} err={:.3e} ({}) tol={:.3e}",
                    e.status.label(),
                    e.id,
                    e.metric,
                    e.measured,
                    e.expected,
                    e.error,
                    e.error_kind.label(),
                    e.tolerance
                );
                if let Some(c) = &e.convergence {
                    let _ = writeln!(
                        s,
                        "           order in {}: p = {:.3} (expected {:.1} ± {:.1}) {}",
                        c.parameter,
                        c.measured_order,
                        c.expected_order,
                        c.order_tolerance,
                        if c.is_ok() { "OK" } else { "MISMATCH" }
                    );
                }
                for n in &e.notes {
                    let _ = writeln!(s, "           note: {n}");
                }
            }
        }
        let (p, f, r) = self.counts();
        let _ = writeln!(s, "\n{p} passed, {f} failed, {r} reported");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convergence_fit_recovers_known_order() {
        // error = 3 h^2 exactly
        let samples: Vec<(f64, f64)> = [1.0, 0.5, 0.25, 0.125]
            .iter()
            .map(|&h| (h, 3.0 * h * h))
            .collect();
        let c = Convergence::fit("dt", samples, 2.0, 0.1);
        assert!(
            (c.measured_order - 2.0).abs() < 1e-9,
            "p = {}",
            c.measured_order
        );
        assert!(c.is_ok());
    }

    #[test]
    fn failed_order_downgrades_pass() {
        let v = Validation::new(
            "t.x",
            "t",
            "c",
            "r",
            "m",
            1.0,
            1.0,
            ErrorKind::Relative,
            1e-9,
        );
        assert_eq!(v.status, Status::Pass);
        let bad = Convergence::fit("dt", vec![(1.0, 1.0), (0.5, 0.9)], 2.0, 0.2);
        assert_eq!(v.with_convergence(bad).status, Status::Fail);
    }
}
