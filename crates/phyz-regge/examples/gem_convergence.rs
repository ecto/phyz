//! GEM tidal tensor h-convergence test.
//!
//! Fixes the physical geometry (4×4×4 domain, 2×2 loops, separation=2) and
//! varies only lattice resolution N. If the Regge-extracted B_{ij} agrees
//! with the linearized GEM prediction, the ratio → 1.0 as h → 0.
//!
//! Configuration via env vars (all optional):
//!   CONV_DOMAIN    physical domain size        (default 4.0)
//!   CONV_AMP       amplitude                   (default 1e-4)
//!   CONV_NS        comma-separated N values    (default "4,8")
//!   CONV_DT_RATIO  dt/spacing ratio            (default 0.3)
//!
//! Run:
//!   cargo run --example gem_convergence -p phyz-regge --release
//!   CONV_NS=4,8,12 cargo run --example gem_convergence -p phyz-regge --release

use phyz_regge::foliation::foliated_hypercubic;
use phyz_regge::gem::{b_grav_tensor_frobenius, linearized_b_grav_tidal, vertex_spatial_coords};
use phyz_regge::transformer::{
    make_planar_winding, run_transformer_continuation, ResidualStats, TransformerConfig,
};

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Binary search for convergence order p given three (h, σ) pairs.
/// Solves: (s0 - s1) / (s1 - s2) = (h0^p - h1^p) / (h1^p - h2^p)
fn estimate_order(h0: f64, s0: f64, h1: f64, s1: f64, h2: f64, s2: f64) -> f64 {
    let ds01 = s0 - s1;
    let ds12 = s1 - s2;
    if ds12.abs() < 1e-15 || ds01.abs() < 1e-15 {
        return 2.0;
    }
    let target = ds01 / ds12;

    let ratio_at = |p: f64| -> f64 {
        let num = h0.powf(p) - h1.powf(p);
        let den = h1.powf(p) - h2.powf(p);
        if den.abs() < 1e-30 { f64::MAX } else { num / den }
    };

    let (mut lo, mut hi) = (0.5_f64, 6.0_f64);
    let r_lo = ratio_at(lo);
    let r_hi = ratio_at(hi);
    if (r_lo - target) * (r_hi - target) > 0.0 {
        return 2.0; // fallback
    }
    for _ in 0..50 {
        let mid = (lo + hi) / 2.0;
        if (ratio_at(mid) - target) * (ratio_at(lo) - target) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Least-squares fit σ = a + b·h^p, returns (a, b).
fn fit_linear(points: &[(f64, f64)], p: f64) -> (f64, f64) {
    let (mut s1, mut shp, mut shp2, mut ss, mut sshp) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(h, s) in points {
        let hp = h.powf(p);
        s1 += 1.0;
        shp += hp;
        shp2 += hp * hp;
        ss += s;
        sshp += s * hp;
    }
    let det = s1 * shp2 - shp * shp;
    if det.abs() > 1e-30 {
        ((ss * shp2 - sshp * shp) / det, (s1 * sshp - shp * ss) / det)
    } else {
        (ss / s1, 0.0)
    }
}

struct DataPoint {
    n: usize,
    h: f64,
    ratio: f64,
    regge_frob: f64,
    lin_frob: f64,
    residual_stats: ResidualStats,
    elapsed: f64,
}

fn main() {
    let domain: f64 = env_or("CONV_DOMAIN", 4.0);
    let amplitude: f64 = env_or("CONV_AMP", 1e-4);
    let dt_ratio: f64 = env_or("CONV_DT_RATIO", 0.3);
    let ns_str: String = env_or("CONV_NS", "4,8".to_string());
    let ns: Vec<usize> = ns_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    eprintln!("=== GEM Tidal Tensor h-Convergence ===");
    eprintln!("Domain: {domain}^3 (periodic), amplitude: {amplitude:.2e}");
    eprintln!("N values: {:?}, dt/h ratio: {dt_ratio}", ns);
    eprintln!();

    let mut data: Vec<DataPoint> = Vec::new();

    for &n in &ns {
        let spacing = domain / n as f64;
        let dt = dt_ratio * spacing;

        eprintln!("N={n}: spacing={spacing:.4}, dt={dt:.4} ...");

        let config = TransformerConfig {
            n_spatial: n,
            n_time: 3,
            spacing,
            dt,
            ..Default::default()
        };

        let t0 = std::time::Instant::now();
        let result = run_transformer_continuation(&config, &[amplitude], 4);
        let elapsed = t0.elapsed().as_secs_f64();
        let m = &result.measurements[0];

        // Build primary loop coordinates for linearized prediction
        let fc = foliated_hypercubic(3, n);
        let primary = make_planar_winding(&fc, 0, [0, n / 2], [0, n / 2], 0, "primary");
        let loop_coords: Vec<[f64; 3]> = primary
            .vertices
            .iter()
            .map(|&v| vertex_spatial_coords(v, &fc, spacing))
            .collect();

        // Secondary center
        let z_secondary = (n / 2).min(n - 1);
        let secondary =
            make_planar_winding(&fc, 0, [0, n / 2], [0, n / 2], z_secondary, "secondary");
        let center: [f64; 3] = {
            let mut c = [0.0; 3];
            for &v in &secondary.vertices {
                let coords = vertex_spatial_coords(v, &fc, spacing);
                c[0] += coords[0];
                c[1] += coords[1];
                c[2] += coords[2];
            }
            let nv = secondary.vertices.len() as f64;
            [c[0] / nv, c[1] / nv, c[2] / nv]
        };

        let fd_eps = spacing * 1e-4;
        let lin_tidal = linearized_b_grav_tidal(&loop_coords, amplitude, center, fd_eps);
        let lin_frob = b_grav_tensor_frobenius(&lin_tidal);

        let ratio = if lin_frob > 1e-30 {
            m.b_grav_frobenius / lin_frob
        } else {
            f64::NAN
        };

        let rs = &m.residual_stats;
        eprintln!(
            "  B_regge={:.6e}  B_lin={:.6e}  ratio={:.6}  res_max={:.2e}  ({:.1}s)",
            m.b_grav_frobenius, lin_frob, ratio, rs.max, elapsed,
        );
        eprintln!(
            "    residual: mean={:.2e}  median={:.2e}  p90={:.2e}  n_bad={}/{}",
            rs.mean, rs.median, rs.p90, rs.n_above_tol, rs.n_total,
        );

        data.push(DataPoint {
            n,
            h: spacing,
            ratio,
            regge_frob: m.b_grav_frobenius,
            lin_frob,
            residual_stats: m.residual_stats.clone(),
            elapsed,
        });
    }

    // Summary table
    eprintln!();
    eprintln!("--- Convergence Table ---");
    eprintln!(
        "{:>4} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>6} {:>8}",
        "N", "ratio", "max_res", "mean_res", "median_res", "p90_res", "B_regge", "n_bad", "time(s)"
    );
    for d in &data {
        let rs = &d.residual_stats;
        eprintln!(
            "{:>4} {:>8.6} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e} {:>6} {:>8.1}",
            d.n,
            d.ratio,
            rs.max,
            rs.mean,
            rs.median,
            rs.p90,
            d.regge_frob,
            rs.n_above_tol,
            d.elapsed,
        );
    }

    // Convergence fit (≥3 points)
    if data.len() >= 3 {
        let pts: Vec<(f64, f64)> = data.iter().map(|d| (d.h, d.ratio)).collect();
        let p = estimate_order(pts[0].0, pts[0].1, pts[1].0, pts[1].1, pts[2].0, pts[2].1);
        let (a, b) = fit_linear(&pts, p);

        eprintln!();
        eprintln!("--- Convergence Fit: ratio(h) = {a:.6} + {b:.4} * h^{p:.2} ---");
        eprintln!("  Extrapolated ratio (h→0): {a:.6}");
        eprintln!("  Convergence order:        {p:.2}");

        if (a - 1.0).abs() < 0.1 {
            eprintln!("  ✓ Extrapolated ratio ≈ 1.0 — Regge extraction validated");
        } else {
            eprintln!(
                "  Extrapolated ratio deviates from 1.0 by {:.1}%",
                (a - 1.0).abs() * 100.0
            );
        }
    }

    // TSV to stdout
    println!("N\th\tdt\tB_regge\tB_linear\tratio\tmax_res\tmean_res\tmedian_res\tp90_res\tn_bad\tn_total\ttime_s");
    for d in &data {
        let rs = &d.residual_stats;
        println!(
            "{}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{}\t{}\t{:.2}",
            d.n,
            d.h,
            dt_ratio * d.h,
            d.regge_frob,
            d.lin_frob,
            d.ratio,
            rs.max,
            rs.mean,
            rs.median,
            rs.p90,
            rs.n_above_tol,
            rs.n_total,
            d.elapsed,
        );
    }
}
