//! Continuum Limit: R² vs Mesh Refinement on S⁴
//!
//! Tests whether the ∂S_EE ∝ ∂S_R correlation improves on finer S⁴
//! triangulations. If R² → 1.0 as the mesh is refined, this establishes
//! that entanglement equilibrium becomes exact in the continuum limit.
//!
//! Uses stellar subdivision of ∂Δ⁵ to build a sequence of finer meshes:
//!
//! | Level | V | E | b₁ | dim = 2^b₁ | Method  |
//! |-------|---|---|----|------------|---------|
//! | 0     | 6 | 15| 10 | 1,024      | Dense   |
//! | 1     | 7 | 20| 14 | 16,384     | Lanczos |
//! | 2     | 8 | 25| 18 | 262,144    | Lanczos |
//!
//! Run:
//!   cargo run -p phyz-quantum --release --example jacobson_continuum
//!   cargo run -p phyz-quantum --release --example jacobson_continuum -- --level 1
//!   cargo run -p phyz-quantum --release --example jacobson_continuum -- --level 1 --scan-only

use phyz_quantum::jacobson::{
    EquilibriumConfig, cut_area_gradient, entanglement_gradient_su2, project_out_conformal,
    subdivided_s4,
};
use phyz_quantum::ryu_takayanagi::*;
use phyz_regge::regge::regge_action_grad;
use std::time::Instant;

fn main() {
    eprintln!("=== Continuum Limit: R² vs Mesh Refinement on S⁴ ===\n");

    // Parse optional --level argument (default: 1).
    let max_level = std::env::args()
        .position(|a| a == "--level")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);

    // Parse --scan-only flag: skip off-shell/RT, run only coupling scan.
    let scan_only = std::env::args().any(|a| a == "--scan-only");

    println!("# Continuum Limit: R² vs Mesh Refinement on S⁴");
    println!("# max_level = {max_level}");
    println!();

    // Geometry builders: use fewer for higher levels.
    let seed_list: Vec<u64> = vec![1, 2, 3];
    let mass_list = [0.3, 0.7];
    let hubble_list = [0.3, 0.7];

    if !scan_only {
        println!(
            "level\tV\tE\tb1\tdim\tN_part\tN_geom\tN_pts\tslope_raw\tR2_raw\tslope_proj\tR2_proj\tRT_R2_raw\tRT_R2_proj\ttime_s"
        );
    }

    for level in 0..=max_level {
        let t_level = Instant::now();
        let complex = subdivided_s4(level);
        let n_v = complex.n_vertices;
        let n_e = complex.n_edges();
        let b1 = n_e - n_v + 1;
        let dim = 1usize << b1;

        eprintln!(
            "── Level {level}: V={n_v}, E={n_e}, b₁={b1}, dim={dim}, P={} ──",
            complex.n_pents()
        );

        // Balanced partitions: |A| = floor(V/2).
        let half = n_v / 2;
        let all_partitions = vertex_bipartitions(n_v);
        let partitions: Vec<&Vec<usize>> =
            all_partitions.iter().filter(|p| p.len() == half).collect();

        // For large dims, limit the number of partitions and geometries.
        let max_parts = if dim > 100_000 { 10 } else { partitions.len() };
        let max_geos = if dim > 100_000 { 3 } else { 7 };
        let partitions = &partitions[..max_parts.min(partitions.len())];

        eprintln!(
            "  {} balanced partitions (|A|={}), using {}",
            all_partitions.iter().filter(|p| p.len() == half).count(),
            half,
            partitions.len()
        );

        let config = EquilibriumConfig::default();
        let equilateral = vec![1.0; n_e];

        // Build geometry list.
        struct GeoSpec {
            name: String,
            lengths: Vec<f64>,
        }

        if !scan_only {
            let mut geometries: Vec<GeoSpec> = Vec::new();

            for &seed in &seed_list {
                if geometries.len() >= max_geos {
                    break;
                }
                let l = perturbed_edge_lengths(&complex, 0.15, seed);
                if geometry_valid(&complex, &l) {
                    geometries.push(GeoSpec {
                        name: format!("perturbed_s{seed}"),
                        lengths: l,
                    });
                }
            }

            for &m in &mass_list {
                if geometries.len() >= max_geos {
                    break;
                }
                let l = schwarzschild_edge_lengths(&complex, m);
                if geometry_valid(&complex, &l) {
                    geometries.push(GeoSpec {
                        name: format!("schwarz_M{m}"),
                        lengths: l,
                    });
                }
            }

            for &h in &hubble_list {
                if geometries.len() >= max_geos {
                    break;
                }
                let l = de_sitter_edge_lengths(&complex, h);
                if geometry_valid(&complex, &l) {
                    geometries.push(GeoSpec {
                        name: format!("deSitter_H{h}"),
                        lengths: l,
                    });
                }
            }

            eprintln!("  {} geometries", geometries.len());

            // ── Off-shell correlation ──
            let mut pooled_ds_ee: Vec<f64> = Vec::new();
            let mut pooled_ds_regge: Vec<f64> = Vec::new();
            let mut pooled_ds_ee_proj: Vec<f64> = Vec::new();
            let mut pooled_ds_regge_proj: Vec<f64> = Vec::new();

            for geo in &geometries {
                let t0 = Instant::now();
                let regge_g = regge_action_grad(&complex, &geo.lengths);
                let (regge_proj, _) = project_out_conformal(&regge_g);

                for &part in partitions {
                    let ds_ee = entanglement_gradient_su2(&complex, &geo.lengths, part, &config);
                    let (ee_proj, _) = project_out_conformal(&ds_ee);

                    for ei in 0..n_e {
                        pooled_ds_ee.push(ds_ee[ei]);
                        pooled_ds_regge.push(regge_g[ei]);
                        pooled_ds_ee_proj.push(ee_proj[ei]);
                        pooled_ds_regge_proj.push(regge_proj[ei]);
                    }
                }
                let elapsed = t0.elapsed().as_secs_f64();
                eprintln!("  {} done ({elapsed:.1}s)", geo.name);
            }

            let (slope_raw, _, r2_raw) = linear_regression(&pooled_ds_regge, &pooled_ds_ee);
            let (slope_proj, _, r2_proj) =
                linear_regression(&pooled_ds_regge_proj, &pooled_ds_ee_proj);

            // ── RT differential ──
            let mut rt_ds_ee: Vec<f64> = Vec::new();
            let mut rt_da_cut: Vec<f64> = Vec::new();
            let mut rt_ds_ee_proj: Vec<f64> = Vec::new();
            let mut rt_da_cut_proj: Vec<f64> = Vec::new();

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
                for &part in partitions {
                    let ds_ee = entanglement_gradient_su2(&complex, &geo.lengths, part, &config);
                    let da_cut = cut_area_gradient(&complex, part, &geo.lengths);
                    let (ee_proj, _) = project_out_conformal(&ds_ee);
                    let (cut_proj, _) = project_out_conformal(&da_cut);

                    for ei in 0..n_e {
                        rt_ds_ee.push(ds_ee[ei]);
                        rt_da_cut.push(da_cut[ei]);
                        rt_ds_ee_proj.push(ee_proj[ei]);
                        rt_da_cut_proj.push(cut_proj[ei]);
                    }
                }
                let elapsed = t0.elapsed().as_secs_f64();
                eprintln!("  RT {} done ({elapsed:.1}s)", geo.name);
            }

            let (_, _, rt_r2_raw) = linear_regression(&rt_da_cut, &rt_ds_ee);
            let (_, _, rt_r2_proj) = linear_regression(&rt_da_cut_proj, &rt_ds_ee_proj);

            let elapsed_level = t_level.elapsed().as_secs_f64();

            println!(
                "{level}\t{n_v}\t{n_e}\t{b1}\t{dim}\t{}\t{}\t{}\t{slope_raw:.6e}\t{r2_raw:.6}\t{slope_proj:.6e}\t{r2_proj:.6}\t{rt_r2_raw:.6}\t{rt_r2_proj:.6}\t{elapsed_level:.1}",
                partitions.len(),
                geometries.len(),
                pooled_ds_ee.len(),
            );

            eprintln!(
                "  Level {level}: R²_raw={r2_raw:.4}, R²_proj={r2_proj:.4}, RT_R²_proj={rt_r2_proj:.4} ({elapsed_level:.1}s)"
            );
        }

        // ── Coupling scan ──
        eprintln!("  Coupling scan...");
        let coupling_values = [0.5, 1.0, 2.0, 5.0];

        if level == 0 && !scan_only {
            // Already have off-shell at g²=1 above; print coupling scan header.
        }

        for &g_sq in &coupling_values {
            let t_cs = Instant::now();
            let scan_config = EquilibriumConfig {
                g_squared: g_sq,
                fd_eps: 1e-4,
            };

            let mut cs_ee: Vec<f64> = Vec::new();
            let mut cs_regge: Vec<f64> = Vec::new();
            let mut cs_ee_proj: Vec<f64> = Vec::new();
            let mut cs_regge_proj: Vec<f64> = Vec::new();

            // Use 2 perturbed geometries (matches level-0 coupling scan).
            // Also track partition-averaged gradients per geometry.
            let mut avg_ee: Vec<f64> = Vec::new();
            let mut avg_regge: Vec<f64> = Vec::new();
            let mut avg_ee_proj: Vec<f64> = Vec::new();
            let mut avg_regge_proj: Vec<f64> = Vec::new();
            let mut per_geo_r2: Vec<(String, f64, f64)> = Vec::new();

            for &seed in &[1u64, 2] {
                let l = perturbed_edge_lengths(&complex, 0.15, seed);
                if !geometry_valid(&complex, &l) {
                    continue;
                }
                let regge_g = regge_action_grad(&complex, &l);
                let (regge_proj, _) = project_out_conformal(&regge_g);

                // Accumulate partition-averaged EE gradient.
                let mut geo_avg_ee = vec![0.0; n_e];
                let mut geo_ee: Vec<f64> = Vec::new();
                let mut geo_regge: Vec<f64> = Vec::new();

                for &part in partitions {
                    let ds_ee = entanglement_gradient_su2(&complex, &l, part, &scan_config);
                    let (ee_proj, _) = project_out_conformal(&ds_ee);
                    for ei in 0..n_e {
                        cs_ee.push(ds_ee[ei]);
                        cs_regge.push(regge_g[ei]);
                        cs_ee_proj.push(ee_proj[ei]);
                        cs_regge_proj.push(regge_proj[ei]);
                        geo_avg_ee[ei] += ds_ee[ei];
                        geo_ee.push(ds_ee[ei]);
                        geo_regge.push(regge_g[ei]);
                    }
                }

                // Per-geometry R².
                let (_, _, geo_r2) = linear_regression(&geo_regge, &geo_ee);
                let (_, _, geo_r2_proj) = {
                    let geo_ee_proj: Vec<f64> = geo_ee
                        .iter()
                        .enumerate()
                        .map(|(i, _)| cs_ee_proj[cs_ee_proj.len() - geo_ee.len() + i])
                        .collect();
                    let geo_regge_proj: Vec<f64> = geo_regge
                        .iter()
                        .enumerate()
                        .map(|(i, _)| cs_regge_proj[cs_regge_proj.len() - geo_regge.len() + i])
                        .collect();
                    linear_regression(&geo_regge_proj, &geo_ee_proj)
                };
                per_geo_r2.push((format!("s{seed}"), geo_r2, geo_r2_proj));

                // Partition-averaged gradient.
                let n_parts_f = partitions.len() as f64;
                for ei in 0..n_e {
                    geo_avg_ee[ei] /= n_parts_f;
                    avg_ee.push(geo_avg_ee[ei]);
                    avg_regge.push(regge_g[ei]);
                }
                let (avg_proj, _) = project_out_conformal(&geo_avg_ee);
                for ei in 0..n_e {
                    avg_ee_proj.push(avg_proj[ei]);
                    avg_regge_proj.push(regge_proj[ei]);
                }
            }

            let (s_raw, _, r2_raw) = linear_regression(&cs_regge, &cs_ee);
            let (s_proj, _, r2_proj) = linear_regression(&cs_regge_proj, &cs_ee_proj);
            let (s_avg, _, r2_avg) = linear_regression(&avg_regge, &avg_ee);
            let (s_avg_proj, _, r2_avg_proj) = linear_regression(&avg_regge_proj, &avg_ee_proj);
            let cs_elapsed = t_cs.elapsed().as_secs_f64();

            println!(
                "coupling\t{level}\t{g_sq:.1}\t{}\t{}\t{s_raw:.6e}\t{r2_raw:.6}\t{s_proj:.6e}\t{r2_proj:.6}\t{s_avg:.6e}\t{r2_avg:.6}\t{s_avg_proj:.6e}\t{r2_avg_proj:.6}\t{cs_elapsed:.1}",
                partitions.len(),
                cs_ee.len(),
            );
            for (name, gr2, gr2p) in &per_geo_r2 {
                eprintln!("    {name}: R²_raw={gr2:.4}, R²_proj={gr2p:.4}");
            }
            eprintln!(
                "  g²={g_sq}: R²_raw={r2_raw:.4}, R²_proj={r2_proj:.4}, R²_avg={r2_avg:.4}, R²_avg_proj={r2_avg_proj:.4} ({cs_elapsed:.1}s)"
            );
        }

        let elapsed_total = t_level.elapsed().as_secs_f64();
        eprintln!("  Level {level} total: {elapsed_total:.1}s");
    }

    println!();
    println!("# Summary: R²(level) — expect R² → 1.0 as mesh refines");
    println!("# If ∂S_EE ∝ ∂S_R becomes exact in continuum, this establishes");
    println!("# Einstein equations from entanglement alone.");
    println!(
        "# Coupling scan: coupling\\tlevel\\tg²\\tN_part\\tN_pts\\tslope_raw\\tR2_raw\\tslope_proj\\tR2_proj\\tslope_avg\\tR2_avg\\tslope_avg_proj\\tR2_avg_proj\\ttime_s"
    );

    eprintln!("\n=== Continuum limit analysis complete ===");
}
