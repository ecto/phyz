//! Extract Newton's Constant from First Principles
//!
//! Uses the Ryu-Takayanagi formula S_EE = A_cut / (4G_N) to extract G_N
//! numerically from lattice gauge entanglement on a Regge lattice.
//!
//! Five experiments:
//! 1. Absolute G_N: S_EE vs A_cut regression on equilateral S⁴
//! 2. Scaling test: G_N(λ)/λ² should be constant
//! 3. Coupling scan: G_N(g²) across gauge couplings
//! 4. Gradient RT: ∂S_EE/∂l_e vs ∂A_cut/∂l_e pooled regression
//! 5. Mesh refinement: G_N at level 0 and level 1
//!
//! Run:
//!   cargo run -p phyz-quantum --release --example jacobson_newton

use phyz_quantum::jacobson::{
    boundary_5simplex, cut_area_gradient, entanglement_gradient_su2, project_out_conformal,
    su2_ground_state, subdivided_s4, EquilibriumConfig,
};
use phyz_quantum::ryu_takayanagi::*;
use phyz_quantum::su2_quantum::{
    su2_entanglement_decomposed, su2_entanglement_for_partition, Su2HilbertSpace,
};
use std::time::Instant;

fn main() {
    eprintln!("=== Extract Newton's Constant from First Principles ===\n");

    // ═══════════════════════════════════════════════════════════════════
    // Section 1: Absolute G_N Extraction (equilateral S⁴, g²=1)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("── Section 1: Absolute G_N (equilateral S⁴, g²=1) ──");

    let complex = boundary_5simplex();
    let lengths = vec![1.0; complex.n_edges()];
    let g_squared = 1.0;
    let hs = Su2HilbertSpace::new(&complex);

    eprintln!(
        "  V={}, E={}, T={}, P={}, dim={}",
        complex.n_vertices,
        complex.n_edges(),
        complex.n_triangles(),
        complex.n_pents(),
        hs.dim()
    );

    let t0 = Instant::now();
    let gs = su2_ground_state(&hs, &complex, &lengths, g_squared);
    let solve_time = t0.elapsed().as_secs_f64();
    eprintln!("  Ground state solved in {solve_time:.2}s");

    // All non-trivial partitions on V=6 (|A|=1,2,3 → varied cut areas).
    // On equilateral S⁴, balanced partitions are all symmetry-equivalent,
    // so we need varied sizes to get area variation for regression.
    let all_partitions = vertex_bipartitions(complex.n_vertices);
    let balanced: Vec<&Vec<usize>> = all_partitions.iter().filter(|p| p.len() == 3).collect();
    eprintln!(
        "  {} total partitions, {} balanced (|A|=3)",
        all_partitions.len(),
        balanced.len()
    );

    println!("# Section 1: Absolute G_N Extraction");
    println!("# Equilateral S⁴ (∂Δ⁵), g²=1, all partitions |A|=1,2,3");
    println!("partition\tA_cut\tS_EE\tS_shannon\tS_distill\tn_bdy\ts_per_edge");

    let mut areas = Vec::new();
    let mut entropies = Vec::new();
    let mut shannons = Vec::new();
    let mut distillables = Vec::new();

    for part in &all_partitions {
        let a_cut = cut_area_triangles(&complex, part, &lengths);
        let s_ee = su2_entanglement_for_partition(&hs, &gs, &complex, part);
        let dec = su2_entanglement_decomposed(&hs, &gs, &complex, part);
        let (_, _, boundary) = classify_edges(&complex, part);
        let n_bdy = boundary.len();
        let s_per_edge = if n_bdy > 0 { s_ee / n_bdy as f64 } else { 0.0 };

        println!(
            "{}\t{a_cut:.6e}\t{s_ee:.6e}\t{:.6e}\t{:.6e}\t{n_bdy}\t{s_per_edge:.6e}",
            partition_label(part),
            dec.shannon,
            dec.distillable,
        );

        areas.push(a_cut);
        entropies.push(s_ee);
        shannons.push(dec.shannon);
        distillables.push(dec.distillable);
    }

    // Regression: S_EE vs A_cut.
    let (slope, intercept, r2) = linear_regression(&areas, &entropies);
    let g_n_abs = if slope.abs() > 1e-15 {
        1.0 / (4.0 * slope)
    } else {
        f64::INFINITY
    };

    // Shannon regression.
    let (slope_sh, _int_sh, r2_sh) = linear_regression(&areas, &shannons);
    let g_n_shannon = if slope_sh.abs() > 1e-15 {
        1.0 / (4.0 * slope_sh)
    } else {
        f64::INFINITY
    };

    // Distillable regression.
    let (slope_dist, _int_dist, r2_dist) = linear_regression(&areas, &distillables);

    // Per-edge entropy stats (balanced partitions only for this metric).
    let mean_s_per_edge: f64 = {
        let vals: Vec<f64> = balanced
            .iter()
            .map(|p| {
                let s = su2_entanglement_for_partition(&hs, &gs, &complex, p);
                let (_, _, bdy) = classify_edges(&complex, p);
                if bdy.is_empty() { 0.0 } else { s / bdy.len() as f64 }
            })
            .collect();
        vals.iter().sum::<f64>() / vals.len() as f64
    };
    let ln2 = 2.0_f64.ln();

    println!();
    println!("# Section 1 Results:");
    println!("# S_EE = κ·A_cut + c → G_N = 1/(4κ)");
    println!("# slope(S_total)    = {slope:.6e}, R² = {r2:.6}, G_N = {g_n_abs:.6e}");
    println!("# slope(S_shannon)  = {slope_sh:.6e}, R² = {r2_sh:.6}, G_N = {g_n_shannon:.6e}");
    println!("# slope(S_distill)  = {slope_dist:.6e}, R² = {r2_dist:.6}");
    println!("# intercept = {intercept:.6e}");
    println!("# mean s_per_edge = {mean_s_per_edge:.6e}, ln(2) = {ln2:.6e}, ratio = {:.4}", mean_s_per_edge / ln2);
    println!();

    eprintln!("  G_N(total) = {g_n_abs:.4e}, R² = {r2:.4}");
    eprintln!("  G_N(shannon) = {g_n_shannon:.4e}, R² = {r2_sh:.4}");
    eprintln!("  s/edge = {mean_s_per_edge:.4e}, ln(2) = {ln2:.4e}, ratio = {:.4}", mean_s_per_edge / ln2);

    // ═══════════════════════════════════════════════════════════════════
    // Section 2: Scaling Test — G_N(λ)/λ² should be constant
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("\n── Section 2: Scaling Test ──");

    println!("# Section 2: Scaling Test (G_N ~ length²)");
    println!("# Scale all edges by λ, verify G_N/λ² ≈ const");
    println!("lambda\tG_N\tG_N_over_lam2");

    let mut gn_over_lam2_vals = Vec::new();

    for &lam in &[0.5, 1.0, 2.0, 4.0] {
        let t0 = Instant::now();
        let scaled_lengths: Vec<f64> = lengths.iter().map(|&l| l * lam).collect();
        let gs_scaled = su2_ground_state(&hs, &complex, &scaled_lengths, g_squared);

        let mut sc_areas = Vec::new();
        let mut sc_ents = Vec::new();

        for part in &all_partitions {
            let a = cut_area_triangles(&complex, part, &scaled_lengths);
            let s = su2_entanglement_for_partition(&hs, &gs_scaled, &complex, part);
            sc_areas.push(a);
            sc_ents.push(s);
        }

        let (sc_slope, _, _) = linear_regression(&sc_areas, &sc_ents);
        let gn = if sc_slope.abs() > 1e-15 {
            1.0 / (4.0 * sc_slope)
        } else {
            f64::INFINITY
        };
        let ratio = gn / (lam * lam);
        gn_over_lam2_vals.push(ratio);

        let elapsed = t0.elapsed().as_secs_f64();
        println!("{lam:.2}\t{gn:.6e}\t{ratio:.6e}");
        eprintln!("  λ={lam:.1}: G_N={gn:.4e}, G_N/λ²={ratio:.4e} ({elapsed:.1}s)");
    }

    // Check constancy via CV.
    let mean_ratio = gn_over_lam2_vals.iter().sum::<f64>() / gn_over_lam2_vals.len() as f64;
    let var_ratio = gn_over_lam2_vals
        .iter()
        .map(|&r| (r - mean_ratio) * (r - mean_ratio))
        .sum::<f64>()
        / gn_over_lam2_vals.len() as f64;
    let cv_ratio = if mean_ratio.abs() > 1e-30 {
        var_ratio.sqrt() / mean_ratio
    } else {
        f64::INFINITY
    };
    println!();
    println!("# Scaling summary: mean(G_N/λ²) = {mean_ratio:.6e}, CV = {cv_ratio:.4}");
    println!("# (CV < 0.1 → good dimensional scaling)");
    println!();
    eprintln!("  mean(G_N/λ²) = {mean_ratio:.4e}, CV = {cv_ratio:.4}");

    // ═══════════════════════════════════════════════════════════════════
    // Section 3: Coupling Scan — G_N(g²)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("\n── Section 3: Coupling Scan ──");

    println!("# Section 3: G_N vs Gauge Coupling g²");
    println!("# Weak coupling → more entanglement → smaller G_N");
    println!("# Strong coupling → less entanglement → larger G_N");
    println!("g_squared\tslope\tR^2\tG_N\tslope_shannon\tG_N_shannon\tmean_s_per_edge");

    for &g_sq in &[0.1, 0.5, 1.0, 2.0, 5.0, 20.0] {
        let t0 = Instant::now();
        let gs_coup = su2_ground_state(&hs, &complex, &lengths, g_sq);

        let mut coup_areas = Vec::new();
        let mut coup_ents = Vec::new();
        let mut coup_shannon = Vec::new();
        let mut s_per_edges = Vec::new();

        for part in &all_partitions {
            let a = cut_area_triangles(&complex, part, &lengths);
            let s = su2_entanglement_for_partition(&hs, &gs_coup, &complex, part);
            let dec = su2_entanglement_decomposed(&hs, &gs_coup, &complex, part);
            let (_, _, bdy) = classify_edges(&complex, part);
            let n_bdy = bdy.len();

            coup_areas.push(a);
            coup_ents.push(s);
            coup_shannon.push(dec.shannon);
            if n_bdy > 0 {
                s_per_edges.push(s / n_bdy as f64);
            }
        }

        let (c_slope, _, c_r2) = linear_regression(&coup_areas, &coup_ents);
        let c_gn = if c_slope.abs() > 1e-15 {
            1.0 / (4.0 * c_slope)
        } else {
            f64::INFINITY
        };
        let (c_slope_sh, _, _) = linear_regression(&coup_areas, &coup_shannon);
        let c_gn_sh = if c_slope_sh.abs() > 1e-15 {
            1.0 / (4.0 * c_slope_sh)
        } else {
            f64::INFINITY
        };
        let mean_spe = if s_per_edges.is_empty() {
            0.0
        } else {
            s_per_edges.iter().sum::<f64>() / s_per_edges.len() as f64
        };

        let elapsed = t0.elapsed().as_secs_f64();
        println!(
            "{g_sq:.6e}\t{c_slope:.6e}\t{c_r2:.6}\t{c_gn:.6e}\t{c_slope_sh:.6e}\t{c_gn_sh:.6e}\t{mean_spe:.6e}"
        );
        eprintln!("  g²={g_sq}: G_N={c_gn:.4e}, R²={c_r2:.4}, s/edge={mean_spe:.4e} ({elapsed:.1}s)");
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // Section 4: Gradient-Based G_N (enhanced RT differential)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("\n── Section 4: Gradient-Based G_N ──");

    println!("# Section 4: Gradient RT Differential");
    println!("# ∂S_EE/∂l_e vs ∂A_cut/∂l_e pooled across geometries/partitions");
    println!("geometry\tpartition\tedge\tds_ee_dl\tda_cut_dl");

    let config = EquilibriumConfig::default();
    let n_edges = complex.n_edges();

    struct GeoSpec {
        name: String,
        lengths: Vec<f64>,
    }

    let rt_geos: Vec<GeoSpec> = {
        let mut g = vec![GeoSpec {
            name: "equilateral".into(),
            lengths: lengths.clone(),
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

    let mut rt_ds_ee_proj = Vec::new();
    let mut rt_da_cut_proj = Vec::new();

    for geo in &rt_geos {
        let t0 = Instant::now();

        for part in &balanced {
            let ds_ee = entanglement_gradient_su2(&complex, &geo.lengths, part, &config);
            let da_cut = cut_area_gradient(&complex, part, &geo.lengths);
            let (ee_proj, _) = project_out_conformal(&ds_ee);
            let (cut_proj, _) = project_out_conformal(&da_cut);

            for ei in 0..n_edges {
                println!(
                    "{}\t{}\t{ei}\t{:.6e}\t{:.6e}",
                    geo.name,
                    partition_label(part),
                    ds_ee[ei],
                    da_cut[ei]
                );
                rt_ds_ee_proj.push(ee_proj[ei]);
                rt_da_cut_proj.push(cut_proj[ei]);
            }
        }
        let elapsed = t0.elapsed().as_secs_f64();
        eprintln!("  RT {} done ({elapsed:.1}s)", geo.name);
    }

    let (rt_slope, _rt_int, rt_r2) = linear_regression(&rt_da_cut_proj, &rt_ds_ee_proj);
    let g_n_grad = if rt_slope.abs() > 1e-15 {
        1.0 / (4.0 * rt_slope)
    } else {
        f64::INFINITY
    };

    println!();
    println!("# Section 4 Results (conformal-projected):");
    println!("# slope = {rt_slope:.6e} (≈ 1/4G_N)");
    println!("# R² = {rt_r2:.6}");
    println!("# G_N_grad = {g_n_grad:.6e}");
    println!("# G_N_abs (Section 1) = {g_n_abs:.6e}");

    let consistency = if g_n_abs.is_finite() && g_n_grad.is_finite() && g_n_abs > 0.0 {
        (g_n_grad / g_n_abs - 1.0).abs()
    } else {
        f64::INFINITY
    };
    println!("# |G_N_grad/G_N_abs - 1| = {consistency:.4}");
    println!();

    eprintln!("  G_N_grad = {g_n_grad:.4e}, R² = {rt_r2:.4}");
    eprintln!("  G_N_abs = {g_n_abs:.4e}");
    eprintln!("  consistency = {consistency:.4}");

    // ═══════════════════════════════════════════════════════════════════
    // Section 5: Mesh Refinement (level 0, level 1)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("\n── Section 5: Mesh Refinement ──");

    println!("# Section 5: Mesh Refinement (level 0 vs level 1)");
    println!("level\tV\tE\tdim\tslope\tR^2\tG_N\tmean_edge_len\tG_N_over_l2");

    for level in 0..=1 {
        let t0 = Instant::now();
        let cmplx = subdivided_s4(level);
        let n_v = cmplx.n_vertices;
        let n_e = cmplx.n_edges();
        let hs_lv = Su2HilbertSpace::new(&cmplx);
        let dim = hs_lv.dim();

        // Equilateral lengths at this refinement level.
        // Level 1 has smaller edges — use 1.0 for all and report mean.
        let lv_lengths = vec![1.0; n_e];
        let mean_edge_len = 1.0;

        eprintln!(
            "  Level {level}: V={n_v}, E={n_e}, dim={dim}"
        );

        let gs_lv = su2_ground_state(&hs_lv, &cmplx, &lv_lengths, g_squared);
        let solve_time = t0.elapsed().as_secs_f64();
        eprintln!("  Ground state in {solve_time:.1}s");

        // Use all partitions to get cut-area variation for regression.
        let parts_lv = vertex_bipartitions(n_v);

        let mut lv_areas = Vec::new();
        let mut lv_ents = Vec::new();

        for part in &parts_lv {
            let a = cut_area_triangles(&cmplx, part, &lv_lengths);
            let s = su2_entanglement_for_partition(&hs_lv, &gs_lv, &cmplx, part);
            if s > 1e-12 && a > 1e-12 {
                lv_areas.push(a);
                lv_ents.push(s);
            }
        }

        if lv_areas.len() >= 2 {
            let (lv_slope, _, lv_r2) = linear_regression(&lv_areas, &lv_ents);
            let lv_gn = if lv_slope.abs() > 1e-15 {
                1.0 / (4.0 * lv_slope)
            } else {
                f64::INFINITY
            };
            let gn_over_l2 = lv_gn / (mean_edge_len * mean_edge_len);
            let elapsed = t0.elapsed().as_secs_f64();

            println!(
                "{level}\t{n_v}\t{n_e}\t{dim}\t{lv_slope:.6e}\t{lv_r2:.6}\t{lv_gn:.6e}\t{mean_edge_len:.4}\t{gn_over_l2:.6e}"
            );
            eprintln!(
                "  G_N = {lv_gn:.4e}, R² = {lv_r2:.4}, G_N/l² = {gn_over_l2:.4e} ({elapsed:.1}s)"
            );
        } else {
            let elapsed = t0.elapsed().as_secs_f64();
            println!("{level}\t{n_v}\t{n_e}\t{dim}\t—\t—\t—\t{mean_edge_len:.4}\t—");
            eprintln!("  Insufficient data points ({elapsed:.1}s)");
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    println!("# ═══════════════════════════════════════════════════════════");
    println!("# SUMMARY: Newton's Constant Extraction");
    println!("# ═══════════════════════════════════════════════════════════");
    println!("# G_N (absolute, S_total)   = {g_n_abs:.6e}");
    println!("# G_N (absolute, S_shannon) = {g_n_shannon:.6e}");
    println!("# G_N (gradient RT)         = {g_n_grad:.6e}");
    println!("# Scaling: mean(G_N/λ²)     = {mean_ratio:.6e}, CV = {cv_ratio:.4}");
    println!("# Abs vs Grad consistency    = {consistency:.4}");
    println!("# s_per_edge / ln(2)         = {:.4}", mean_s_per_edge / ln2);
    println!("# ═══════════════════════════════════════════════════════════");

    eprintln!("\n=== Newton's constant extraction complete ===");
}
