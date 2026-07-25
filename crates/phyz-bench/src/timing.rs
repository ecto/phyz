//! A small, explicit timing harness.
//!
//! Criterion is excellent for tracking a single library's regressions, but its
//! output is awkward to aggregate into one publishable document. This module
//! measures the same way for every library so the comparison is apples to
//! apples: fixed warmup, N repetitions of a fixed-size batch of work, and we
//! report the **median** (robust to a scheduler hiccup) alongside min/max so
//! the noise is visible rather than hidden.

use serde::{Deserialize, Serialize};
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Timing statistics over `reps` repetitions of a work batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timing {
    /// Number of timed repetitions (warmup excluded).
    pub reps: usize,
    /// Units of work per repetition (e.g. simulation steps).
    pub work_per_rep: u64,
    /// Median seconds per unit of work.
    pub median_sec_per_unit: f64,
    pub min_sec_per_unit: f64,
    pub max_sec_per_unit: f64,
    /// Relative spread `(max - min) / median`. Above ~0.2 the machine is noisy.
    pub spread: f64,
    /// Median units of work per second — the headline number.
    pub throughput_per_sec: f64,
}

/// Relative spread above which a measurement is treated as machine noise
/// rather than a result.
pub const NOISE_THRESHOLD: f64 = 0.25;

impl Timing {
    /// Whether the spread is wide enough that this number should be re-run on
    /// a quiet machine before anyone quotes it.
    pub fn is_noisy(&self) -> bool {
        self.spread > NOISE_THRESHOLD
    }

    fn from_samples(mut samples: Vec<Duration>, work_per_rep: u64) -> Self {
        assert!(!samples.is_empty(), "no timing samples");
        samples.sort();
        let per_unit = |d: Duration| d.as_secs_f64() / work_per_rep as f64;
        let median = per_unit(samples[samples.len() / 2]);
        let min = per_unit(samples[0]);
        let max = per_unit(samples[samples.len() - 1]);
        Self {
            reps: samples.len(),
            work_per_rep,
            median_sec_per_unit: median,
            min_sec_per_unit: min,
            max_sec_per_unit: max,
            spread: if median > 0.0 {
                (max - min) / median
            } else {
                0.0
            },
            throughput_per_sec: if median > 0.0 {
                1.0 / median
            } else {
                f64::INFINITY
            },
        }
    }
}

/// How hard to hit a benchmark. Kept in one place so every suite and every
/// library under comparison uses identical effort.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Repetitions run and discarded before timing starts.
    pub warmup_reps: usize,
    /// Timed repetitions.
    pub reps: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            warmup_reps: 2,
            reps: 7,
        }
    }
}

impl Budget {
    /// A quick pass for smoke-testing the harness (`--quick`).
    pub fn quick() -> Self {
        Self {
            warmup_reps: 1,
            reps: 3,
        }
    }
}

/// Time `f`, which performs `work_per_rep` units of work per call.
///
/// `f` is re-run from scratch each repetition, so it must reset any state it
/// mutates; a benchmark whose second repetition starts from a diverged state
/// is measuring something else.
pub fn measure<F, R>(budget: Budget, work_per_rep: u64, mut f: F) -> Timing
where
    F: FnMut() -> R,
{
    for _ in 0..budget.warmup_reps {
        black_box(f());
    }
    let mut samples = Vec::with_capacity(budget.reps);
    for _ in 0..budget.reps {
        let t0 = Instant::now();
        let out = f();
        let dt = t0.elapsed();
        black_box(out);
        samples.push(dt);
    }
    Timing::from_samples(samples, work_per_rep)
}
