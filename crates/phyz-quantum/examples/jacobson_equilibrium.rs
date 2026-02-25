//! Jacobson's Entanglement Equilibrium on a Regge Lattice
//!
//! Tests whether ∂S_EE/∂l_e ∝ ∂S_R/∂l_e — entanglement gradients encode
//! the discrete Einstein equations.
//!
//! Runs on ∂Δ⁵ = S⁴ (boundary of 5-simplex), a closed 4-manifold with
//! V=6, E=15, T=20, P=6, dim=1024. Four experiments:
//! 1. On-shell equilibrium: equilateral S⁴ → conformal projection test
//! 2. Off-shell correlation: perturbed geometry → R² for ∂S_EE vs ∂S_R
//! 3. RT differential: ∂S_EE/∂l_e vs ∂Area_cut/∂l_e
//! 4. Coupling scan: R² vs g²
//!
//! Run:
//!   cargo run -p phyz-quantum --release --example jacobson_equilibrium

use phyz_quantum::jacobson::{
    boundary_5simplex, cut_area_gradient, entanglement_gradient_su2, project_out_conformal,
    EquilibriumConfig,
};
use phyz_quantum::ryu_takayanagi::*;
use phyz_regge::regge::regge_action_grad;
use std::time::Instant;

fn main() {
    eprintln!("=== Jacobson's Entanglement Equilibrium on a Regge Lattice ===\n");

    let complex = boundary_5simplex();
    let n_edges = complex.n_edges();
    eprintln!(
        "Complex: ∂Δ⁵ = S⁴, V={}, E={}, T={}, P={}",
        complex.n_vertices,
        n_edges,
        complex.n_triangles(),
        complex.n_pents()
    );

    let all_partitions = vertex_bipartitions(complex.n_vertices);
    // Singleton partitions produce zero ∂S_EE on complete-graph topology
    // (ground state entropy independent of geometry when |A|=1).
    // Filter to |A| ≥ 2 for regression sections.
    let partitions: Vec<&Vec<usize>> = all_partitions.iter().filter(|p| p.len() >= 2).collect();
    eprintln!(
        "Partitions: {} total, {} with |A|≥2",
        all_partitions.len(),
        partitions.len()
    );

    // ═══════════════════════════════════════════════════════════════════
    // Section 1: On-Shell Equilibrium (conformal projection on S⁴)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("\n── Section 1: On-Shell Equilibrium (equilateral S⁴) ──");

    let equilateral = vec![1.0; n_edges];
    let config = EquilibriumConfig::default();

    // On equilateral S⁴, Regge gradient is uniform (pure conformal) by S₆ symmetry.
    let regge_grad = regge_action_grad(&complex, &equilateral);
    let regge_g0 = regge_grad[0];
    let max_regge_var = regge_grad.iter().map(|g| (g - regge_g0).abs()).fold(0.0, f64::max);
    eprintln!("  ∂S_R/∂l (uniform) = {regge_g0:.4e}, max variation = {max_regge_var:.2e}");

    let (_, regge_conf) = project_out_conformal(&regge_grad);
    eprintln!("  Regge conformal component = {regge_conf:.4e}");

    println!("# Section 1: On-Shell Equilibrium (conformal projection)");
    println!("# Equilateral S⁴: all l_e = 1.0");
    println!("# S⁴ has positive curvature → ∂S_R ≠ 0, but is purely conformal");
    println!("# Test: ∂S_EE projected ⊥ conformal mode should be ~0");
    println!("partition\tmax_resid_perp\tconformal_ee\tconformal_regge\tratio_conf");

    let mut all_max_resid: Vec<f64> = Vec::new();

    let t0 = Instant::now();
    for part in &all_partitions {
        let ds_ee = entanglement_gradient_su2(&complex, &equilateral, part, &config);
        let (projected, ee_conf) = project_out_conformal(&ds_ee);
        let max_resid = projected.iter().map(|g| g.abs()).fold(0.0, f64::max);
        let ratio = if regge_conf.abs() > 1e-15 {
            ee_conf / regge_conf
        } else {
            f64::NAN
        };

        println!(
            "{}\t{max_resid:.6e}\t{ee_conf:.6e}\t{regge_conf:.6e}\t{ratio:.6e}",
            partition_label(part),
        );
        all_max_resid.push(max_resid);
    }
    let elapsed = t0.elapsed().as_secs_f64();

    let max_ds_ee_perp = all_max_resid.iter().copied().fold(0.0, f64::max);
    let mean_ds_ee_perp = all_max_resid.iter().sum::<f64>() / all_max_resid.len() as f64;
    println!();
    println!("# On-shell summary (conformal projection):");
    println!("# max|∂S_EE^⊥/∂l| = {max_ds_ee_perp:.6e} (want < 1e-3)");
    println!("# mean(max|∂S_EE^⊥|) = {mean_ds_ee_perp:.6e}");
    println!("# time = {elapsed:.1}s");
    eprintln!(
        "  max|∂S_EE^⊥/∂l| = {max_ds_ee_perp:.4e}, mean = {mean_ds_ee_perp:.4e} ({elapsed:.1}s)"
    );
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // Section 2: Off-Shell Correlation
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("\n── Section 2: Off-Shell Correlation ──");

    println!("# Section 2: Off-Shell Correlation");
    println!("# ∂S_EE/∂l_e vs ∂S_R/∂l_e across geometries and partitions");
    println!("geometry\tpartition\tedge\tds_ee_dl\tds_regge_dl");

    struct GeoSpec {
        name: String,
        lengths: Vec<f64>,
    }

    let geometries: Vec<GeoSpec> = {
        let mut g = Vec::new();

        // Perturbed geometries.
        for &seed in &[1u64, 2, 3] {
            let l = perturbed_edge_lengths(&complex, 0.15, seed);
            if geometry_valid(&complex, &l) {
                g.push(GeoSpec {
                    name: format!("perturbed_s{seed}"),
                    lengths: l,
                });
            }
        }

        // Schwarzschild.
        for &m in &[0.3, 0.7] {
            let l = schwarzschild_edge_lengths(&complex, m);
            if geometry_valid(&complex, &l) {
                g.push(GeoSpec {
                    name: format!("schwarz_M{m}"),
                    lengths: l,
                });
            }
        }

        // De Sitter.
        for &h in &[0.3, 0.7] {
            let l = de_sitter_edge_lengths(&complex, h);
            if geometry_valid(&complex, &l) {
                g.push(GeoSpec {
                    name: format!("deSitter_H{h}"),
                    lengths: l,
                });
            }
        }

        g
    };

    let mut pooled_ds_ee: Vec<f64> = Vec::new();
    let mut pooled_ds_regge: Vec<f64> = Vec::new();

    for geo in &geometries {
        let t0 = Instant::now();
        let regge_g = regge_action_grad(&complex, &geo.lengths);

        for part in &partitions {
            let ds_ee = entanglement_gradient_su2(&complex, &geo.lengths, part, &config);

            for ei in 0..n_edges {
                println!(
                    "{}\t{}\t{ei}\t{:.6e}\t{:.6e}",
                    geo.name,
                    partition_label(part),
                    ds_ee[ei],
                    regge_g[ei]
                );
                pooled_ds_ee.push(ds_ee[ei]);
                pooled_ds_regge.push(regge_g[ei]);
            }
        }
        let elapsed = t0.elapsed().as_secs_f64();
        eprintln!("  {} done ({elapsed:.1}s)", geo.name);
    }

    // Correlation.
    let (slope, intercept, r2) = linear_regression(&pooled_ds_regge, &pooled_ds_ee);
    println!();
    println!("# Off-shell correlation (pooled):");
    println!("# N = {} data points", pooled_ds_ee.len());
    println!("# slope = {slope:.6e}");
    println!("# intercept = {intercept:.6e}");
    println!("# R^2 = {r2:.6}");
    eprintln!("  Pooled: slope={slope:.4e}, R²={r2:.4}");
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // Section 3: RT Differential
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("\n── Section 3: RT Differential ──");

    println!("# Section 3: RT Differential");
    println!("# ∂S_EE/∂l_e vs ∂Area_cut/∂l_e across partitions");
    println!("geometry\tpartition\tedge\tds_ee_dl\tda_cut_dl");

    let mut rt_ds_ee: Vec<f64> = Vec::new();
    let mut rt_da_cut: Vec<f64> = Vec::new();

    // Use flat + a few off-shell geometries.
    let rt_geos: Vec<GeoSpec> = {
        let mut g = vec![GeoSpec {
            name: "equilateral".into(),
            lengths: equilateral.clone(),
        }];
        for &seed in &[1u64, 2] {
            let l = perturbed_edge_lengths(&complex, 0.1, seed);
            if geometry_valid(&complex, &l) {
                g.push(GeoSpec {
                    name: format!("perturbed_s{seed}"),
                    lengths: l,
                });
            }
        }
        g
    };

    for geo in &rt_geos {
        let t0 = Instant::now();

        for part in &partitions {
            let ds_ee = entanglement_gradient_su2(&complex, &geo.lengths, part, &config);
            let da_cut = cut_area_gradient(&complex, part, &geo.lengths);

            for ei in 0..n_edges {
                println!(
                    "{}\t{}\t{ei}\t{:.6e}\t{:.6e}",
                    geo.name,
                    partition_label(part),
                    ds_ee[ei],
                    da_cut[ei]
                );
                rt_ds_ee.push(ds_ee[ei]);
                rt_da_cut.push(da_cut[ei]);
            }
        }
        let elapsed = t0.elapsed().as_secs_f64();
        eprintln!("  RT {} done ({elapsed:.1}s)", geo.name);
    }

    let (rt_slope, _rt_int, rt_r2) = linear_regression(&rt_da_cut, &rt_ds_ee);
    let rt_gn = if rt_slope.abs() > 1e-15 {
        1.0 / (4.0 * rt_slope)
    } else {
        f64::INFINITY
    };
    println!();
    println!("# RT differential summary:");
    println!("# slope = {rt_slope:.6e} (≈ 1/4G_N)");
    println!("# R^2 = {rt_r2:.6}");
    println!("# G_N = {rt_gn:.6e}");
    eprintln!("  RT: slope={rt_slope:.4e}, R²={rt_r2:.4}, G_N={rt_gn:.4e}");
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // Section 4: Coupling Scan
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("\n── Section 4: Coupling Scan ──");

    println!("# Section 4: R^2 vs g^2 for off-shell correlation");
    println!("g_squared\tslope\tR^2");

    for &g_sq in &[0.5, 1.0, 2.0, 5.0] {
        let t0 = Instant::now();
        let scan_config = EquilibriumConfig {
            g_squared: g_sq,
            fd_eps: 1e-4,
        };

        let mut scan_ds_ee: Vec<f64> = Vec::new();
        let mut scan_ds_regge: Vec<f64> = Vec::new();

        // Use just 2 off-shell geometries for speed.
        for &seed in &[1u64, 2] {
            let l = perturbed_edge_lengths(&complex, 0.15, seed);
            if !geometry_valid(&complex, &l) {
                continue;
            }
            let regge_g = regge_action_grad(&complex, &l);
            for part in &partitions {
                let ds_ee = entanglement_gradient_su2(&complex, &l, part, &scan_config);
                for ei in 0..n_edges {
                    scan_ds_ee.push(ds_ee[ei]);
                    scan_ds_regge.push(regge_g[ei]);
                }
            }
        }

        let (s, _i, r2) = linear_regression(&scan_ds_regge, &scan_ds_ee);
        println!("{g_sq:.6e}\t{s:.6e}\t{r2:.6}");
        let elapsed = t0.elapsed().as_secs_f64();
        eprintln!("  g²={g_sq}: R²={r2:.4} ({elapsed:.1}s)");
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    println!("# Summary (∂Δ⁵ = S⁴, |A|≥2 partitions)");
    println!("# On-shell: max|∂S_EE^⊥/∂l| = {max_ds_ee_perp:.4e} (want < 1e-3)");
    println!("# Off-shell: R² = {r2:.4} (want > 0.5)");
    println!("# RT differential: G_N = {rt_gn:.4e}");
    println!("# Partitions: {} (filtered from {} total)", partitions.len(), all_partitions.len());

    eprintln!("\n=== Jacobson equilibrium analysis complete ===");
}
