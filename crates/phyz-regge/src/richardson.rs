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
    /// Wall-clock time for this resolution (seconds).
    pub elapsed_secs: f64,
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
    /// Fit residuals for each spectral index (only meaningful with ≥3 resolutions).
    pub fit_residuals: Vec<f64>,
    /// R² (coefficient of determination) for each fit.
    pub fit_r_squared: Vec<f64>,
}

impl RichardsonResults {
    /// Human-readable summary table.
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Richardson Extrapolation ===\n");
        s.push_str(&format!("Resolutions: {:?}\n", self.resolutions.iter().map(|r| r.n).collect::<Vec<_>>()));
        s.push_str(&format!("Spacings h:  {:?}\n", self.resolutions.iter().map(|r| format!("{:.4}", r.h)).collect::<Vec<_>>()));
        s.push_str(&format!("Spectrum lengths: {:?}\n", self.resolutions.iter().map(|r| r.violations.len()).collect::<Vec<_>>()));
        s.push_str(&format!("Elapsed (s): {:?}\n\n", self.resolutions.iter().map(|r| format!("{:.1}", r.elapsed_secs)).collect::<Vec<_>>()));

        let n_extrap = self.extrapolated.len();
        if n_extrap == 0 {
            s.push_str("No overlapping spectral indices to extrapolate.\n");
            return s;
        }

        let has_fit_quality = !self.fit_residuals.is_empty();

        s.push_str(&format!("{:<6} ", "k"));
        for rd in &self.resolutions {
            s.push_str(&format!("{:>12} ", format!("n={}", rd.n)));
        }
        s.push_str(&format!("{:>12} {:>8}", "extrap", "order"));
        if has_fit_quality {
            s.push_str(&format!("{:>12} {:>8}", "residual", "R²"));
        }
        s.push('\n');
        let width = 6 + 13 * self.resolutions.len() + 22 + if has_fit_quality { 22 } else { 0 };
        s.push_str(&"-".repeat(width));
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
            s.push_str(&format!("{:>12.2e} {:>8.1}", self.extrapolated[k], self.convergence_orders[k]));
            if has_fit_quality && k < self.fit_residuals.len() {
                s.push_str(&format!("{:>12.2e} {:>8.4}", self.fit_residuals[k], self.fit_r_squared[k]));
            }
            s.push('\n');
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
    richardson_extrapolation_with(ns, spacing, background_builder, known_builder, params, base_seed, None)
}

/// Like `richardson_extrapolation` with an optional sample cap.
pub fn richardson_extrapolation_with(
    ns: &[usize],
    spacing: f64,
    background_builder: impl Fn(usize) -> (SimplicialComplex, Vec<f64>),
    known_builder: impl Fn(&SimplicialComplex, &Fields, usize) -> Vec<Generator>,
    params: &ActionParams,
    base_seed: u64,
    max_samples: Option<usize>,
) -> RichardsonResults {
    assert!(ns.len() >= 2, "need at least 2 resolutions");

    let sample_cap = max_samples.unwrap_or(usize::MAX);
    let mut resolutions = Vec::with_capacity(ns.len());

    for &n in ns {
        let t0 = std::time::Instant::now();

        let (complex, lengths) = background_builder(n);
        let phases = vec![0.0; complex.n_edges()];
        let fields = Fields::new(lengths, phases);

        let known = known_builder(&complex, &fields, n);
        let n_known = orthonormalize(&known).len();
        let n_dof = fields.n_dof();

        // Auto-scale samples: at least 500, at least 2× (DOF - known), capped.
        let n_samples = 500.max(2 * n_dof.saturating_sub(n_known)).min(sample_cap);

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
        let elapsed_secs = t0.elapsed().as_secs_f64();

        resolutions.push(ResolutionData {
            n,
            h,
            violations,
            n_dof,
            n_known,
            elapsed_secs,
        });
    }

    // Extrapolate: for each spectral index k, fit σ_k(h) = a + b·h^p.
    let min_len = resolutions.iter().map(|r| r.violations.len()).min().unwrap_or(0);

    let mut extrapolated = Vec::with_capacity(min_len);
    let mut convergence_orders = Vec::with_capacity(min_len);
    let mut fit_residuals = Vec::with_capacity(min_len);
    let mut fit_r_squared = Vec::with_capacity(min_len);

    for k in 0..min_len {
        let points: Vec<(f64, f64)> = resolutions
            .iter()
            .map(|r| (r.h, r.violations[k]))
            .collect();

        if points.len() >= 3 {
            let (a, p, residual, r_sq) = fit_three_point(&points);
            extrapolated.push(a);
            convergence_orders.push(p);
            fit_residuals.push(residual);
            fit_r_squared.push(r_sq);
        } else {
            // Two-point Richardson: σ_extrap = (n₂²·σ₂ − n₁²·σ₁) / (n₂² − n₁²)
            let (h1, s1) = points[0];
            let (h2, s2) = points[1];
            let h1_sq = h1 * h1;
            let h2_sq = h2 * h2;
            let a = (h1_sq * s2 - h2_sq * s1) / (h1_sq - h2_sq);
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
        fit_residuals,
        fit_r_squared,
    }
}

/// Three-point fit: σ = a + b·h^p.
///
/// Returns (a, p, residual, R²).
fn fit_three_point(points: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let (h0, s0) = points[0];
    let (h1, s1) = points[1];
    let (h2, s2) = points[2];

    let p = estimate_order(h0, s0, h1, s1, h2, s2);

    // Least-squares fit: σ_k = a + b·h^p.
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
    let (a, b) = if det.abs() > 1e-30 {
        let a = (sum_s * sum_hp2 - sum_s_hp * sum_hp) / det;
        let b = (sum_1 * sum_s_hp - sum_hp * sum_s) / det;
        (a, b)
    } else {
        (s2, 0.0) // fallback
    };

    // Compute residual and R².
    let mean_s = sum_s / sum_1;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for &(h, s) in points {
        let predicted = a + b * h.powf(p);
        ss_res += (s - predicted) * (s - predicted);
        ss_tot += (s - mean_s) * (s - mean_s);
    }

    let residual = (ss_res / sum_1).sqrt();
    let r_squared = if ss_tot > 1e-30 { 1.0 - ss_res / ss_tot } else { 1.0 };

    (a, p, residual, r_squared)
}

/// Estimate convergence order from three (h, σ) pairs.
fn estimate_order(h0: f64, s0: f64, h1: f64, s1: f64, h2: f64, s2: f64) -> f64 {
    let ds01 = s0 - s1;
    let ds12 = s1 - s2;

    if ds12.abs() < 1e-15 || ds01.abs() < 1e-15 {
        return 2.0;
    }

    let target_ratio = ds01 / ds12;

    let ratio_at = |p: f64| -> f64 {
        let num = h0.powf(p) - h1.powf(p);
        let den = h1.powf(p) - h2.powf(p);
        if den.abs() < 1e-30 { f64::MAX } else { num / den }
    };

    let mut lo = 0.5_f64;
    let mut hi = 6.0_f64;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_richardson_synthetic_three_point() {
        // Synthetic data: σ = 0.001 + 2.5 * h^2.0
        let points = vec![
            (0.5, 0.001 + 2.5 * 0.5_f64.powf(2.0)),
            (1.0 / 3.0, 0.001 + 2.5 * (1.0 / 3.0_f64).powf(2.0)),
            (0.25, 0.001 + 2.5 * 0.25_f64.powf(2.0)),
        ];

        let (a, p, residual, r_sq) = fit_three_point(&points);

        assert!(
            (a - 0.001).abs() < 1e-4,
            "expected a ≈ 0.001, got {a}"
        );
        assert!(
            (p - 2.0).abs() < 0.5,
            "expected p ≈ 2.0, got {p}"
        );
        assert!(
            residual < 1e-4,
            "expected small residual, got {residual}"
        );
        assert!(
            r_sq > 0.99,
            "expected R² ≈ 1.0, got {r_sq}"
        );
    }
}
