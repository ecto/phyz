//! Machine-readable results: the JSON schema published to phyz.dev, plus a
//! Markdown renderer for `BENCHMARKS.md`.
//!
//! The schema is versioned so that later tooling can track a number across
//! runs without guessing at what changed.

use serde::{Deserialize, Serialize};

use crate::meta::RunMeta;
use crate::settings::Settings;
use crate::timing::Timing;

/// Bump when the shape of [`Report`] changes incompatibly.
pub const SCHEMA_VERSION: u32 = 1;

/// A whole benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub schema_version: u32,
    pub meta: RunMeta,
    pub suites: Vec<Suite>,
}

/// One benchmark dimension (throughput, gradients, energy, batch, comparison).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suite {
    pub name: String,
    /// What this suite measures and how to read it.
    pub description: String,
    /// Present and populated when the suite could not run (no GPU, feature off).
    pub skipped: Option<String>,
    pub results: Vec<Record>,
}

impl Suite {
    /// A suite that did not run, with the reason preserved in the output.
    pub fn skipped(name: &str, description: &str, why: &str) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            skipped: Some(why.into()),
            results: Vec::new(),
        }
    }

    /// A suite that ran.
    pub fn new(name: &str, description: &str, results: Vec<Record>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            skipped: None,
            results,
        }
    }
}

/// A single measured data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    /// Engine that produced the number: `"phyz"`, `"rapier3d-f64"`, ...
    pub engine: String,
    /// Scene identifier, shared across engines so rows line up.
    pub scene: String,
    /// Human-readable scene description.
    pub description: String,
    /// Degrees of freedom, where meaningful.
    pub dof: Option<usize>,
    /// Batch size for the GPU suite; `1` for single-sim.
    pub batch: Option<usize>,
    /// The settings this point was measured under.
    pub settings: Settings,
    /// Timing, absent for pure-accuracy records.
    pub timing: Option<Timing>,
    /// Named scalar outcomes: energy drift, gradient ratios, error norms.
    pub metrics: Vec<Metric>,
    /// Anything a reader needs in order not to misread the number.
    pub notes: Vec<String>,
}

/// A named scalar with units.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub value: f64,
    pub unit: String,
}

impl Metric {
    /// Construct a metric.
    pub fn new(name: &str, value: f64, unit: &str) -> Self {
        Self {
            name: name.into(),
            value,
            unit: unit.into(),
        }
    }
}

/// Format a throughput for humans: `1.23 M` / `45.6 k` / `789`.
pub fn human(x: f64) -> String {
    if !x.is_finite() {
        return "—".into();
    }
    if x >= 1.0e6 {
        format!("{:.2} M", x / 1.0e6)
    } else if x >= 1.0e3 {
        format!("{:.1} k", x / 1.0e3)
    } else {
        format!("{x:.1}")
    }
}

impl Report {
    /// Render the run as Markdown tables, suitable for pasting into
    /// `BENCHMARKS.md` or rendering on the site.
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        let m = &self.meta;
        s.push_str("## Run metadata\n\n");
        s.push_str(&format!("- **Date:** {}\n", m.timestamp));
        s.push_str(&format!(
            "- **Hardware:** {} ({} logical cores{}{})\n",
            m.hardware.cpu,
            m.hardware.cores,
            m.hardware
                .memory_gib
                .map(|g| format!(", {g} GiB RAM"))
                .unwrap_or_default(),
            m.hardware
                .gpu
                .as_ref()
                .map(|g| format!(", GPU: {g}"))
                .unwrap_or_default(),
        ));
        s.push_str(&format!(
            "- **OS:** {} ({} {})\n",
            m.os.release, m.os.family, m.os.arch
        ));
        s.push_str(&format!(
            "- **Toolchain:** {} — profile `{}`{}\n",
            m.toolchain.rustc,
            m.toolchain.profile,
            m.toolchain
                .rustflags
                .as_ref()
                .map(|f| format!(", `RUSTFLAGS={f}`"))
                .unwrap_or_default(),
        ));
        for lib in &m.libraries {
            s.push_str(&format!(
                "- **{}:** {}{}\n",
                lib.name,
                lib.version,
                lib.note
                    .as_ref()
                    .map(|n| format!(" ({n})"))
                    .unwrap_or_default()
            ));
        }
        if let Some(c) = &m.git_commit {
            s.push_str(&format!("- **Commit:** `{}`\n", &c[..c.len().min(12)]));
        }
        s.push('\n');

        for suite in &self.suites {
            s.push_str(&format!("## {}\n\n{}\n\n", suite.name, suite.description));
            if let Some(why) = &suite.skipped {
                s.push_str(&format!("> **Skipped:** {why}\n\n"));
                continue;
            }
            s.push_str(
                "| Engine | Scene | DOF | Batch | dt (s) | Throughput | µs/step | Spread | Metrics |\n\
                 |---|---|---|---|---|---|---|---|---|\n",
            );
            for r in &suite.results {
                let (tput, us, spread) = match &r.timing {
                    Some(t) => (
                        format!("{}/s", human(t.throughput_per_sec)),
                        format!("{:.3}", t.median_sec_per_unit * 1.0e6),
                        // Flagged inline so a noisy number cannot be copied out
                        // of the table as if it were a clean one.
                        format!(
                            "{:.0}%{}",
                            t.spread * 100.0,
                            if t.is_noisy() { " ⚠" } else { "" }
                        ),
                    ),
                    None => ("—".into(), "—".into(), "—".into()),
                };
                let metrics = r
                    .metrics
                    .iter()
                    .map(|m| format!("{}={:.4} {}", m.name, m.value, m.unit))
                    .collect::<Vec<_>>()
                    .join("<br>");
                s.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                    r.engine,
                    r.scene,
                    r.dof.map(|d| d.to_string()).unwrap_or_else(|| "—".into()),
                    r.batch.map(|b| b.to_string()).unwrap_or_else(|| "—".into()),
                    r.settings.dt,
                    tput,
                    us,
                    spread,
                    metrics,
                ));
            }
            s.push('\n');

            let noisy = suite
                .results
                .iter()
                .filter(|r| r.timing.as_ref().is_some_and(|t| t.is_noisy()))
                .count();
            if noisy > 0 {
                s.push_str(&format!(
                    "> ⚠ **{noisy} of {} rows exceeded the {:.0}% spread threshold.** \
                     Those rows measured a busy machine as much as they measured physics. \
                     Re-run on an idle host before quoting them.\n\n",
                    suite.results.len(),
                    crate::timing::NOISE_THRESHOLD * 100.0,
                ));
            }

            let notes: Vec<&String> = suite
                .results
                .iter()
                .flat_map(|r| r.notes.iter())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            if !notes.is_empty() {
                s.push_str("**Notes**\n\n");
                for n in notes {
                    s.push_str(&format!("- {n}\n"));
                }
                s.push('\n');
            }

            let caveats: Vec<&String> = suite
                .results
                .iter()
                .flat_map(|r| r.settings.caveats.iter())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            if !caveats.is_empty() {
                s.push_str("**Settings caveats**\n\n");
                for c in caveats {
                    s.push_str(&format!("- {c}\n"));
                }
                s.push('\n');
            }
        }
        s
    }
}
