//! SU(2) Ryu-Takayanagi with Variable Geometry
//!
//! Tests whether G_N = 1/(4·slope) from the RT formula S_EE = Area/4G_N is
//! universal across different background geometries on each complex size.
//!
//! Computes ~16 geometries × 4 complex sizes. If CV(G_N) < 10% at each size,
//! that's evidence for a discrete RT formula on dynamical backgrounds.
//!
//! Run:
//!   cargo run -p phyz-quantum --features gpu --release --example su2_rt_geometry

use phyz_quantum::diag;
use phyz_quantum::ryu_takayanagi::*;
use phyz_quantum::su2_quantum::{
    build_su2_hamiltonian, su2_entanglement_decomposed, su2_entanglement_for_partition,
    Su2HilbertSpace,
};
use phyz_regge::complex::SimplicialComplex;
use phyz_regge::gauge::metric_weights;
use std::time::Instant;

#[cfg(feature = "gpu")]
use phyz_quantum::gpu_lanczos::gpu_lanczos_diagonalize_su2;

/// A named geometry: edge lengths + description.
struct Geometry {
    name: String,
    lengths: Vec<f64>,
}

/// Build all candidate geometries for a given complex.
fn build_geometries(complex: &SimplicialComplex) -> Vec<Geometry> {
    let n_shared = if complex.n_vertices > 5 { 4 } else { 0 };
    let mut geos = Vec::new();

    // Flat
    geos.push(Geometry {
        name: "flat".into(),
        lengths: vec![1.0; complex.n_edges()],
    });

    // Schwarzschild (4 values)
    for &m in &[0.2, 0.5, 1.0, 2.0] {
        geos.push(Geometry {
            name: format!("schwarz_M{m}"),
            lengths: schwarzschild_edge_lengths(complex, m),
        });
    }

    // De Sitter (3 values)
    for &h in &[0.3, 0.7, 1.5] {
        geos.push(Geometry {
            name: format!("deSitter_H{h}"),
            lengths: de_sitter_edge_lengths(complex, h),
        });
    }

    // Perturbed (3 seeds)
    for &seed in &[1u64, 2, 3] {
        geos.push(Geometry {
            name: format!("perturbed_s{seed}"),
            lengths: perturbed_edge_lengths(complex, 0.15, seed),
        });
    }

    // Anisotropic (3 scale ratios) — only for multi-pentachoron
    if n_shared > 0 {
        for &(ls, lr) in &[(0.7, 1.3), (0.5, 1.5), (1.3, 0.7)] {
            geos.push(Geometry {
                name: format!("aniso_{ls}_{lr}"),
                lengths: anisotropic_edge_lengths(complex, ls, lr, n_shared),
            });
        }
    }

    // Conformal (2 values)
    for &alpha in &[0.3, 0.8] {
        geos.push(Geometry {
            name: format!("conformal_a{alpha}"),
            lengths: conformal_edge_lengths(complex, alpha),
        });
    }

    // Filter to valid geometries.
    geos.into_iter()
        .filter(|g| geometry_valid(complex, &g.lengths))
        .collect()
}

fn main() {
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("This example requires the 'gpu' feature.");
        eprintln!("Run with: cargo run -p phyz-quantum --features gpu --release --example su2_rt_geometry");
        return;
    }

    #[cfg(feature = "gpu")]
    run_analysis();
}

#[cfg(feature = "gpu")]
fn run_analysis() {
    eprintln!("=== SU(2) Ryu-Takayanagi with Variable Geometry ===\n");

    let g_squared = 1.0;

    let pentachoron_sets: &[&[[usize; 5]]] = &[
        &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]],
        &[
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 5],
            [0, 1, 2, 3, 6],
        ],
        &[
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 5],
            [0, 1, 2, 3, 6],
            [0, 1, 2, 3, 7],
        ],
        &[
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 5],
            [0, 1, 2, 3, 6],
            [0, 1, 2, 3, 7],
            [0, 1, 2, 3, 8],
        ],
    ];
    let n_verts = [6, 7, 8, 9];
    let n_pent_labels = [2, 3, 4, 5];

    // ── Section 1: Raw Data ──
    println!("# SU(2) j=1/2 RT with Variable Geometry");
    println!("# g^2 = {g_squared}");
    println!();

    println!("# Section 1: Raw Data");
    println!("n_pent\tgeometry\tpartition\t|A|\ttri_area\tS_EE");

    struct GeoResult {
        n_pent: usize,
        geometry: String,
        slope: f64,
        r_squared: f64,
        g_n: f64,
        n_points: usize,
    }

    let mut all_results: Vec<GeoResult> = Vec::new();
    let mut pooled_areas: Vec<f64> = Vec::new();
    let mut pooled_entropies: Vec<f64> = Vec::new();

    for (si, &pents) in pentachoron_sets.iter().enumerate() {
        let n_pent = n_pent_labels[si];
        let complex = SimplicialComplex::from_pentachorons(n_verts[si], pents);
        let hs = Su2HilbertSpace::new(&complex);
        let dim = hs.dim();
        let b1 = complex.n_edges() - complex.n_vertices + 1;

        eprintln!(
            "── {n_pent}-pent: V={}, E={}, b₁={b1}, dim={dim} ──",
            complex.n_vertices,
            complex.n_edges()
        );

        let geometries = build_geometries(&complex);
        eprintln!("  {} valid geometries", geometries.len());

        let use_dense = dim <= 4096;
        let parts = vertex_bipartitions(complex.n_vertices);

        for geo in &geometries {
            let t0 = Instant::now();

            // Compute metric weights from edge lengths.
            let mw = metric_weights(&complex, &geo.lengths);

            // Compute ground state.
            let gs = if use_dense {
                let h = build_su2_hamiltonian(&hs, &complex, g_squared, Some(&mw));
                let spec = diag::diagonalize(&h, Some(1));
                spec.states[0].clone()
            } else {
                match gpu_lanczos_diagonalize_su2(
                    &hs, &complex, g_squared, Some(&mw), 1, Some(300),
                ) {
                    Ok(spec) => spec.states[0].clone(),
                    Err(e) => {
                        eprintln!("  GPU failed for {}: {e}", geo.name);
                        continue;
                    }
                }
            };

            let elapsed = t0.elapsed().as_secs_f64();
            eprintln!("  {} ground state in {elapsed:.2}s", geo.name);

            let mut areas = Vec::new();
            let mut entropies = Vec::new();

            for part in &parts {
                let tri_area = cut_area_triangles(&complex, part, &geo.lengths);
                let s_ee = su2_entanglement_for_partition(&hs, &gs, &complex, part);

                println!(
                    "{n_pent}\t{}\t{}\t{}\t{tri_area:.6e}\t{s_ee:.6e}",
                    geo.name,
                    partition_label(part),
                    part.len()
                );

                if s_ee > 1e-12 && tri_area > 1e-12 {
                    areas.push(tri_area);
                    entropies.push(s_ee);
                    pooled_areas.push(tri_area);
                    pooled_entropies.push(s_ee);
                }
            }

            // Per-geometry regression.
            if areas.len() >= 2 {
                let (slope, _intercept, r2) = linear_regression(&areas, &entropies);
                let g_n = if slope.abs() > 1e-15 {
                    1.0 / (4.0 * slope)
                } else {
                    f64::INFINITY
                };
                all_results.push(GeoResult {
                    n_pent,
                    geometry: geo.name.clone(),
                    slope,
                    r_squared: r2,
                    g_n,
                    n_points: areas.len(),
                });
            }
        }
    }
    println!();

    // ── Section 2: Per-Complex G_N Comparison ──
    println!("# Section 2: Per-Complex G_N Comparison Across Geometries");
    println!("n_pent\tgeometry\tn_points\tslope\tR^2\tG_N");

    let mut gn_by_size: std::collections::BTreeMap<usize, Vec<f64>> =
        std::collections::BTreeMap::new();

    for r in &all_results {
        println!(
            "{}\t{}\t{}\t{:.6e}\t{:.6}\t{:.6e}",
            r.n_pent, r.geometry, r.n_points, r.slope, r.r_squared, r.g_n
        );
        if r.g_n.is_finite() && r.g_n > 0.0 {
            gn_by_size.entry(r.n_pent).or_default().push(r.g_n);
        }
    }
    println!();

    // G_N statistics per complex size.
    println!("# G_N Statistics per Complex Size");
    println!("n_pent\tn_geom\tmean_G_N\tstd_G_N\tCV\tuniversal?");
    for (&n_pent, gn_vals) in &gn_by_size {
        let n = gn_vals.len() as f64;
        let mean = gn_vals.iter().sum::<f64>() / n;
        let var = gn_vals.iter().map(|&g| (g - mean) * (g - mean)).sum::<f64>() / n;
        let std = var.sqrt();
        let cv = if mean.abs() > 1e-30 { std / mean } else { f64::INFINITY };
        let universal = if cv < 0.1 { "YES" } else { "no" };
        println!(
            "{n_pent}\t{}\t{mean:.6e}\t{std:.6e}\t{cv:.4}\t{universal}",
            gn_vals.len()
        );
    }
    println!();

    // ── Section 3: Pooled Cross-Geometry Regression ──
    println!("# Section 3: Pooled Cross-Geometry Regression");
    if pooled_areas.len() >= 2 {
        let (slope, intercept, r2) = linear_regression(&pooled_areas, &pooled_entropies);
        let g_n = if slope.abs() > 1e-15 {
            1.0 / (4.0 * slope)
        } else {
            f64::INFINITY
        };
        println!("# N = {} data points", pooled_areas.len());
        println!("# slope = {slope:.6e}");
        println!("# intercept = {intercept:.6e}");
        println!("# R^2 = {r2:.6}");
        println!("# G_N(pooled) = {g_n:.6e}");
    }
    println!();

    // ── Section 4: G_N Convergence with System Size ──
    println!("# Section 4: G_N Convergence with System Size");
    println!("n_pent\tmean_G_N\tCV");
    for (&n_pent, gn_vals) in &gn_by_size {
        let n = gn_vals.len() as f64;
        let mean = gn_vals.iter().sum::<f64>() / n;
        let var = gn_vals.iter().map(|&g| (g - mean) * (g - mean)).sum::<f64>() / n;
        let cv = if mean.abs() > 1e-30 {
            var.sqrt() / mean
        } else {
            f64::INFINITY
        };
        println!("{n_pent}\t{mean:.6e}\t{cv:.4}");
    }
    println!();

    // ── Section 5: Coupling Dependence ──
    println!("# Section 5: G_N vs Coupling g^2 (2-pentachoron, flat geometry)");
    println!("g_squared\tslope\tR^2\tG_N");

    let complex_2 = SimplicialComplex::from_pentachorons(6, &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]]);
    let hs_2 = Su2HilbertSpace::new(&complex_2);
    let parts_2 = vertex_bipartitions(complex_2.n_vertices);
    let flat_lengths = vec![1.0; complex_2.n_edges()];

    for &g_sq in &[0.2, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let h = build_su2_hamiltonian(&hs_2, &complex_2, g_sq, None);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        let mut areas = Vec::new();
        let mut ents = Vec::new();
        for part in &parts_2 {
            let tri_area = cut_area_triangles(&complex_2, part, &flat_lengths);
            let s_ee = su2_entanglement_for_partition(&hs_2, gs, &complex_2, part);
            if s_ee > 1e-12 && tri_area > 1e-12 {
                areas.push(tri_area);
                ents.push(s_ee);
            }
        }

        if areas.len() >= 2 {
            let (slope, _intercept, r2) = linear_regression(&areas, &ents);
            let g_n = if slope.abs() > 1e-15 {
                1.0 / (4.0 * slope)
            } else {
                f64::INFINITY
            };
            println!("{g_sq:.6e}\t{slope:.6e}\t{r2:.6}\t{g_n:.6e}");
        }
    }
    println!();

    // ── Section 6: Superselection Sector Decomposition ──
    println!("# Section 6: Superselection Sector Decomposition (2-pentachoron, flat, g^2=1)");
    println!("partition\t|A|\ttri_area\tS_total\tS_shannon\tS_distill\tn_sectors");

    let h_sec6 = build_su2_hamiltonian(&hs_2, &complex_2, 1.0, None);
    let spec_sec6 = diag::diagonalize(&h_sec6, Some(1));
    let gs_sec6 = spec_sec6.ground_state();

    let mut sec6_areas = Vec::new();
    let mut sec6_total = Vec::new();
    let mut sec6_shannon = Vec::new();
    let mut sec6_distill = Vec::new();

    for part in &parts_2 {
        let tri_area = cut_area_triangles(&complex_2, part, &flat_lengths);
        let dec = su2_entanglement_decomposed(&hs_2, gs_sec6, &complex_2, part);

        println!(
            "{}\t{}\t{tri_area:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{}",
            partition_label(part),
            part.len(),
            dec.total,
            dec.shannon,
            dec.distillable,
            dec.n_sectors
        );

        if dec.total > 1e-12 && tri_area > 1e-12 {
            sec6_areas.push(tri_area);
            sec6_total.push(dec.total);
            sec6_shannon.push(dec.shannon);
            sec6_distill.push(dec.distillable);
        }
    }
    println!();

    // Regressions: S_total vs area, S_shannon vs area, S_distill vs area.
    println!("# Section 6: Regression Comparison");
    println!("component\tslope\tR^2\tG_N");

    for (label, values) in [
        ("S_total", &sec6_total),
        ("S_shannon", &sec6_shannon),
        ("S_distill", &sec6_distill),
    ] {
        if sec6_areas.len() >= 2 {
            let (slope, _intercept, r2) = linear_regression(&sec6_areas, values);
            let g_n = if slope.abs() > 1e-15 {
                1.0 / (4.0 * slope)
            } else {
                f64::INFINITY
            };
            println!("{label}\t{slope:.6e}\t{r2:.6}\t{g_n:.6e}");
        }
    }
    println!();

    eprintln!("=== SU(2) RT geometry analysis complete ===");
}
