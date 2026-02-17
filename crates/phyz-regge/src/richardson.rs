//! Richardson extrapolation for separating real symmetries from discretization artifacts.
//!
//! Runs the symmetry search at multiple mesh resolutions (n=2,3,...), collects
//! violation spectra, and extrapolates to h→0 assuming σ_k(h) = a + b·h².
//! Violations that extrapolate to ~0 are discretization artifacts; those that
//! remain finite are real symmetry-breaking effects.

use crate::action::{ActionParams, Fields};
use crate::complex::SimplicialComplex;
use crate::search::{search_symmetries, SearchConfig};
use crate::symmetry::{orthonormalize, Generator};

/// Data from a single resolution level.
#[derive(Debug, Clone)]
pub struct ResolutionData {
    /// Grid points per axis.
    pub n: usize,
    /// Effective spacing h = spacing / n.
    pub h: f64,
    /// Ordered violation spectrum (interesting range only: above exact-symmetry
    /// floor ~1e-14, below non-symmetry ceiling ~1.0).
    pub violations: Vec<f64>,
    /// Total degrees of freedom at this resolution.
    pub n_dof: usize,
    /// Number of known generators.
    pub n_known: usize,
}

/// Results from Richardson extrapolation across resolutions.
#[derive(Debug, Clone)]
pub struct RichardsonResults {
    /// Per-resolution data.
    pub resolutions: Vec<ResolutionData>,
    /// Extrapolated σ_k(h→0) for each spectral index k.
    pub extrapolated: Vec<f64>,
    /// Estimated convergence order p for each k (from 3-point fit).
    pub convergence_orders: Vec<f64>,
}

impl RichardsonResults {
    /// Human-readable summary table.
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Richardson Extrapolation ===\n");
        s.push_str(&format!("Resolutions: {:?}\n", self.resolutions.iter().map(|r| r.n).collect::<Vec<_>>()));
        s.push_str(&format!("Spacings h:  {:?}\n", self.resolutions.iter().map(|r| format!("{:.4}", r.h)).collect::<Vec<_>>()));
        s.push_str(&format!("Spectrum lengths: {:?}\n\n", self.resolutions.iter().map(|r| r.violations.len()).collect::<Vec<_>>()));

        let n_extrap = self.extrapolated.len();
        if n_extrap == 0 {
            s.push_str("No overlapping spectral indices to extrapolate.\n");
            return s;
        }

        s.push_str(&format!("{:<6} ", "k"));
        for rd in &self.resolutions {
            s.push_str(&format!("{:>12} ", format!("n={}", rd.n)));
        }
        s.push_str(&format!("{:>12} {:>8}\n", "extrap", "order"));
        s.push_str(&"-".repeat(6 + 13 * self.resolutions.len() + 22));
        s.push('\n');

        for k in 0..n_extrap {
            s.push_str(&format!("{:<6} ", k));
            for rd in &self.resolutions {
                if k < rd.violations.len() {
                    s.push_str(&format!("{:>12.2e} ", rd.violations[k]));
                } else {
                    s.push_str(&format!("{:>12} ", "---"));
                }
            }
            s.push_str(&format!("{:>12.2e} {:>8.1}\n", self.extrapolated[k], self.convergence_orders[k]));
        }

        // Highlight candidates.
        let candidates: Vec<usize> = (0..n_extrap)
            .filter(|&k| self.extrapolated[k].abs() < 1e-8)
            .collect();
        if !candidates.is_empty() {
            s.push_str(&format!(
                "\nCandidate real symmetries (extrapolated < 1e-8): {:?}\n",
                candidates
            ));
        } else {
            s.push_str("\nNo candidates with extrapolated violation < 1e-8.\n");
        }

        s
    }
}

/// Run symmetry search at multiple resolutions and Richardson-extrapolate.
///
/// # Arguments
/// * `ns` - Grid sizes to use (e.g., &[2, 3] or &[2, 3, 4])
/// * `background_builder` - Builds (complex, lengths) for a given n
/// * `known_builder` - Builds known generators for a given mesh
/// * `params` - Action parameters
/// * `base_seed` - RNG seed (varied per resolution to avoid correlation)
pub fn richardson_extrapolation(
    ns: &[usize],
    spacing: f64,
    background_builder: impl Fn(usize) -> (SimplicialComplex, Vec<f64>),
    known_builder: impl Fn(&SimplicialComplex, &Fields, usize) -> Vec<Generator>,
    params: &ActionParams,
    base_seed: u64,
) -> RichardsonResults {
    assert!(ns.len() >= 2, "need at least 2 resolutions");

    let mut resolutions = Vec::with_capacity(ns.len());

    for &n in ns {
        let (complex, lengths) = background_builder(n);
        let phases = vec![0.0; complex.n_edges()];
        let fields = Fields::new(lengths, phases);

        let known = known_builder(&complex, &fields, n);
        let n_known = orthonormalize(&known).len();
        let n_dof = fields.n_dof();

        // Auto-scale samples: at least 500, at least 2× (DOF - known).
        let n_samples = 500.max(2 * n_dof.saturating_sub(n_known));

        eprintln!(
            "Richardson n={n}: {n_dof} DOF, {n_known} known, {n_samples} samples...",
        );

        let config = SearchConfig {
            n_samples,
            perturbation_scale: 1e-3,
            seed: base_seed + n as u64,
        };

        let results = search_symmetries(&complex, &fields, &known, params, &config);

        // Extract interesting violations: above noise floor, below ceiling.
        let violations: Vec<f64> = results
            .candidates
            .iter()
            .map(|c| c.violation)
            .filter(|&v| v > 1e-14 && v < 1.0)
            .collect();

        let h = spacing / n as f64;

        resolutions.push(ResolutionData {
            n,
            h,
            violations,
            n_dof,
            n_known,
        });
    }

    // Extrapolate: for each spectral index k, fit σ_k(h) = a + b·h².
    let min_len = resolutions.iter().map(|r| r.violations.len()).min().unwrap_or(0);

    let mut extrapolated = Vec::with_capacity(min_len);
    let mut convergence_orders = Vec::with_capacity(min_len);

    for k in 0..min_len {
        let points: Vec<(f64, f64)> = resolutions
            .iter()
            .map(|r| (r.h, r.violations[k]))
            .collect();

        if points.len() >= 3 {
            // Three-point least-squares fit: σ = a + b·h^p
            // First estimate p from the three-point ratio, then fit a + b·h².
            let (a, p) = fit_three_point(&points);
            extrapolated.push(a);
            convergence_orders.push(p);
        } else {
            // Two-point Richardson: σ_extrap = (n₂²·σ₂ − n₁²·σ₁) / (n₂² − n₁²)
            let (h1, s1) = points[0];
            let (h2, s2) = points[1];
            let h1_sq = h1 * h1;
            let h2_sq = h2 * h2;
            let a = (h1_sq * s2 - h2_sq * s1) / (h1_sq - h2_sq);
            // Estimate order from ratio.
            let p = if (s1 - s2).abs() > 1e-15 {
                ((s1 / s2).abs().ln() / (h1 / h2).abs().ln()).max(0.0)
            } else {
                2.0
            };
            extrapolated.push(a);
            convergence_orders.push(p);
        }
    }

    RichardsonResults {
        resolutions,
        extrapolated,
        convergence_orders,
    }
}

/// Three-point fit: σ = a + b·h^p.
///
/// Estimates p from the three points, then computes a via least squares
/// with σ = a + b·h².
fn fit_three_point(points: &[(f64, f64)]) -> (f64, f64) {
    // Estimate convergence order p from ratio of differences.
    let (h0, s0) = points[0];
    let (h1, s1) = points[1];
    let (h2, s2) = points[2];

    // Try to estimate p from (s0-s1)/(s1-s2) = (h0^p - h1^p)/(h1^p - h2^p).
    // Fall back to p=2 if degenerate.
    let p = estimate_order(h0, s0, h1, s1, h2, s2);

    // Least-squares fit: σ_k = a + b·h^p.
    // Normal equations for [a, b] given data (h_i^p, σ_i).
    let mut sum_1 = 0.0;
    let mut sum_hp = 0.0;
    let mut sum_hp2 = 0.0;
    let mut sum_s = 0.0;
    let mut sum_s_hp = 0.0;

    for &(h, s) in points {
        let hp = h.powf(p);
        sum_1 += 1.0;
        sum_hp += hp;
        sum_hp2 += hp * hp;
        sum_s += s;
        sum_s_hp += s * hp;
    }

    let det = sum_1 * sum_hp2 - sum_hp * sum_hp;
    let a = if det.abs() > 1e-30 {
        (sum_s * sum_hp2 - sum_s_hp * sum_hp) / det
    } else {
        s2 // fallback: finest resolution value
    };

    (a, p)
}

/// Estimate convergence order from three (h, σ) pairs.
fn estimate_order(h0: f64, s0: f64, h1: f64, s1: f64, h2: f64, s2: f64) -> f64 {
    // Use the Aitken method: if σ = a + b·h^p, then
    //   (s0 - s1) / (s1 - s2) ≈ (h0^p - h1^p) / (h1^p - h2^p)
    // Solve for p numerically. Fall back to 2.0 if degenerate.
    let ds01 = s0 - s1;
    let ds12 = s1 - s2;

    if ds12.abs() < 1e-15 || ds01.abs() < 1e-15 {
        return 2.0;
    }

    let target_ratio = ds01 / ds12;

    // Binary search for p in [0.5, 6.0].
    let ratio_at = |p: f64| -> f64 {
        let num = h0.powf(p) - h1.powf(p);
        let den = h1.powf(p) - h2.powf(p);
        if den.abs() < 1e-30 { f64::MAX } else { num / den }
    };

    let mut lo = 0.5_f64;
    let mut hi = 6.0_f64;

    // Check monotonicity; if ratio_at is not bracketing target, return 2.0.
    let r_lo = ratio_at(lo);
    let r_hi = ratio_at(hi);
    if (r_lo - target_ratio) * (r_hi - target_ratio) > 0.0 {
        return 2.0;
    }

    for _ in 0..50 {
        let mid = (lo + hi) / 2.0;
        let r_mid = ratio_at(mid);
        if (r_mid - target_ratio) * (r_lo - target_ratio) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    (lo + hi) / 2.0
}
