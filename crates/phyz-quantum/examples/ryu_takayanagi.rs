//! Ryu-Takayanagi formula check on simplicial gauge theory.
//!
//! Tests whether entanglement entropy S_EE correlates with the geometric area
//! of the entangling surface, as predicted by S_EE = Area(γ_A) / 4G_N.
//!
//! Phases:
//!   A. Flat background, topological area (1-pent, 2-pent, 3-pent)
//!   B. Varying edge lengths / curvature (2-pent, Schwarzschild-like)
//!   C. Coupling dependence (fixed partition, sweep g²)
//!   D. Linear regression: S_EE vs Area across all data
//!   E. Extended edge prescription comparison
//!   F. Mutual information analysis
//!   G. Finite-size scaling of G_N
//!   H. Triangulated torus RT
//!   I. SU(2) gauge theory (j_max = 1/2) RT comparison
//!
//! Run:
//!   cargo run --example ryu_takayanagi -p phyz-quantum --release

use phyz_quantum::diag;
use phyz_quantum::hamiltonian::{build_hamiltonian, KSParams};
use phyz_quantum::hilbert::U1HilbertSpace;
use phyz_quantum::lanczos::lanczos_diagonalize;
use phyz_quantum::observables;
use phyz_quantum::ryu_takayanagi::*;
use phyz_quantum::su2_quantum::{self, Su2HilbertSpace};
use phyz_quantum::triangulated_torus::{
    build_triangulated_torus_hamiltonian, TriangulatedTorus, TriangulatedTorusHilbert,
};
use phyz_regge::complex::SimplicialComplex;
use phyz_regge::gauge::metric_weights;

/// Helper: compute ground state for a simplicial complex.
fn ground_state_simplicial(
    hs: &U1HilbertSpace,
    complex: &SimplicialComplex,
    g_squared: f64,
    mw: Option<&[f64]>,
) -> nalgebra::DVector<f64> {
    if hs.dim() <= 5000 {
        let params = KSParams {
            g_squared,
            metric_weights: mw.map(|w| w.to_vec()),
        };
        let h = build_hamiltonian(hs, complex, &params);
        let spec = diag::diagonalize(&h, Some(1));
        spec.states[0].clone()
    } else {
        let spec = lanczos_diagonalize(hs, complex, g_squared, mw, 1);
        spec.states[0].clone()
    }
}

fn main() {
    eprintln!("=== Ryu-Takayanagi Formula Check on Simplicial Gauge Theory ===\n");

    let lambda = 1u32;
    let g_squared = 1.0;

    // Build test complexes.
    let pent1 = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);
    let pent2 = SimplicialComplex::from_pentachorons(6, &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]]);
    let pent3 = SimplicialComplex::from_pentachorons(
        7,
        &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 4, 6]],
    );

    let hs1 = U1HilbertSpace::new(&pent1, lambda);
    let hs2 = U1HilbertSpace::new(&pent2, lambda);
    let hs3 = U1HilbertSpace::new(&pent3, lambda);

    eprintln!("1-pent: V={}, E={}, dim={}", pent1.n_vertices, pent1.n_edges(), hs1.dim());
    eprintln!("2-pent: V={}, E={}, dim={}", pent2.n_vertices, pent2.n_edges(), hs2.dim());
    eprintln!("3-pent: V={}, E={}, dim={}", pent3.n_vertices, pent3.n_edges(), hs3.dim());

    // Ground states at g²=1, flat background.
    let gs1 = ground_state_simplicial(&hs1, &pent1, g_squared, None);
    let gs2 = ground_state_simplicial(&hs2, &pent2, g_squared, None);
    eprintln!("  Computing 3-pent ground state (Lanczos, dim={})...", hs3.dim());
    let gs3 = ground_state_simplicial(&hs3, &pent3, g_squared, None);

    // Collect (area, S_EE) for regression.
    let mut flat_areas: Vec<f64> = Vec::new();
    let mut flat_entropies: Vec<f64> = Vec::new();

    // ─────────────────────────────────────────────────────────────────
    // Phase A: Flat background, topological area (1-pent, 2-pent, 3-pent)
    // ─────────────────────────────────────────────────────────────────
    eprintln!("\n── Phase A: Flat Background, Topological Area ──\n");

    println!("# Phase A: Flat background (all lengths = 1.0), g^2={g_squared}, lambda={lambda}");

    for (name, complex, hs, gs) in [
        ("1-pent", &pent1, &hs1, &gs1),
        ("2-pent", &pent2, &hs2, &gs2),
        ("3-pent", &pent3, &hs3, &gs3),
    ] {
        println!("# {name}: V={}, E={}, dim={}", complex.n_vertices, complex.n_edges(), hs.dim());
        println!("partition\t|A|\t#cut_edges\tS_EE");

        let parts = vertex_bipartitions(complex.n_vertices);
        for part in &parts {
            let n_cut = cut_area_topological(complex, part);
            let s = entanglement_for_partition(hs, gs, complex, part);
            let label = partition_label(part);
            println!("{label}\t{}\t{n_cut}\t{s:.6e}", part.len());
            flat_areas.push(n_cut as f64);
            flat_entropies.push(s);
        }
        println!();
    }

    eprintln!("  Phase A complete ({} data points)\n", flat_areas.len());

    // ─────────────────────────────────────────────────────────────────
    // Phase B: Varying edge lengths (curvature) on 2-pentachoron
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── Phase B: Varying Edge Lengths (Schwarzschild-like) ──\n");

    println!("# Phase B: Schwarzschild-like edge lengths on 2-pentachoron");
    println!("M\tpartition\t|A|\tgeom_area\ttri_area\tS_EE");

    let mass_values = [0.0, 0.1, 0.5, 1.0];
    for &m in &mass_values {
        let lengths = schwarzschild_edge_lengths(&pent2, m);
        let mw = if m > 0.0 { Some(metric_weights(&pent2, &lengths)) } else { None };
        let gs_b = ground_state_simplicial(&hs2, &pent2, g_squared, mw.as_deref());

        for part in &vertex_bipartitions(pent2.n_vertices) {
            let geom = cut_area_geometric(&pent2, part, &lengths);
            let tri = cut_area_triangles(&pent2, part, &lengths);
            let s = entanglement_for_partition(&hs2, &gs_b, &pent2, part);
            println!("{m:.1}\t{}\t{}\t{geom:.6e}\t{tri:.6e}\t{s:.6e}", partition_label(part), part.len());
        }
    }
    println!();
    eprintln!("  Phase B complete\n");

    // ─────────────────────────────────────────────────────────────────
    // Phase C: Coupling dependence
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── Phase C: Coupling Dependence ──\n");

    let test_partition = vec![0usize, 1];
    let flat_lengths_1 = vec![1.0; pent1.n_edges()];
    let cut_area = cut_area_geometric(&pent1, &test_partition, &flat_lengths_1);

    println!("# Phase C: Coupling dependence (1-pentachoron, partition={{0,1}}, cut_area={cut_area:.1})");
    println!("g_squared\tS_EE\tS_EE/area\t1/(4*g^2)");

    let g_sq_sweep = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
    for &g_sq in &g_sq_sweep {
        let gs = ground_state_simplicial(&hs1, &pent1, g_sq, None);
        let s = entanglement_for_partition(&hs1, &gs, &pent1, &test_partition);
        println!(
            "{g_sq:.6e}\t{s:.6e}\t{:.6e}\t{:.6e}",
            s / cut_area,
            1.0 / (4.0 * g_sq)
        );
    }
    println!();
    eprintln!("  Phase C complete\n");

    // ─────────────────────────────────────────────────────────────────
    // Phase D: Linear regression (flat background only)
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── Phase D: Linear Regression ──\n");

    let filtered: Vec<(f64, f64)> = flat_areas.iter().zip(flat_entropies.iter())
        .filter(|&(&a, &s)| s > 1e-12 && a > 1e-12)
        .map(|(&a, &s)| (a, s))
        .collect();

    let fa: Vec<f64> = filtered.iter().map(|&(a, _)| a).collect();
    let fe: Vec<f64> = filtered.iter().map(|&(_, s)| s).collect();
    let (slope, intercept, r2) = linear_regression(&fa, &fe);

    println!("# Phase D: Linear regression S_EE = alpha * Area + beta (flat background, all complexes)");
    println!("# N = {} data points", filtered.len());
    println!("# slope (alpha) = {slope:.6e}");
    println!("# intercept (beta) = {intercept:.6e}");
    println!("# R^2 = {r2:.6}");
    println!("# Effective G_N = 1/(4*alpha) = {:.6e}", if slope.abs() > 1e-15 { 1.0 / (4.0 * slope) } else { f64::INFINITY });
    println!();

    // Summary: average S_EE by #cut_edges per complex.
    println!("# Summary: mean S_EE by #cut_edges");
    println!("# complex\tcut_edges\tmean_S_EE\tn_partitions");
    for (name, complex, hs, gs) in [
        ("1-pent", &pent1, &hs1, &gs1),
        ("2-pent", &pent2, &hs2, &gs2),
        ("3-pent", &pent3, &hs3, &gs3),
    ] {
        let mut by_cut: std::collections::BTreeMap<usize, Vec<f64>> = std::collections::BTreeMap::new();
        for part in &vertex_bipartitions(complex.n_vertices) {
            let n_cut = cut_area_topological(complex, part);
            let s = entanglement_for_partition(hs, gs, complex, part);
            by_cut.entry(n_cut).or_default().push(s);
        }
        for (n_cut, ents) in &by_cut {
            let mean: f64 = ents.iter().sum::<f64>() / ents.len() as f64;
            println!("{name}\t{n_cut}\t{mean:.6e}\t{}", ents.len());
        }
    }
    println!();
    eprintln!("  Phase D complete\n");

    // ─────────────────────────────────────────────────────────────────
    // Phase E: Extended edge prescription comparison
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── Phase E: Extended vs Algebraic Prescription ──\n");

    println!("# Phase E: Extended prescription (at least one endpoint in A) vs algebraic (both endpoints)");
    println!("# 1-pentachoron, g^2={g_squared}");
    println!("partition\t|A|\t#cut\tS_algebraic\tS_extended");

    for part in &vertex_bipartitions(pent1.n_vertices) {
        let n_cut = cut_area_topological(&pent1, part);
        let s_alg = entanglement_for_partition(&hs1, &gs1, &pent1, part);
        let s_ext = entanglement_for_partition_extended(&hs1, &gs1, &pent1, part);
        println!(
            "{}\t{}\t{n_cut}\t{s_alg:.6e}\t{s_ext:.6e}",
            partition_label(part), part.len()
        );
    }
    println!();

    // Extended prescription on 2-pent.
    println!("# 2-pentachoron, g^2={g_squared}");
    println!("partition\t|A|\t#cut\tS_algebraic\tS_extended");
    for part in &vertex_bipartitions(pent2.n_vertices) {
        let n_cut = cut_area_topological(&pent2, part);
        let s_alg = entanglement_for_partition(&hs2, &gs2, &pent2, part);
        let s_ext = entanglement_for_partition_extended(&hs2, &gs2, &pent2, part);
        println!(
            "{}\t{}\t{n_cut}\t{s_alg:.6e}\t{s_ext:.6e}",
            partition_label(part), part.len()
        );
    }
    println!();

    // Regression on extended prescription (skip 3-pent: too slow at dim=46K).
    let mut ext_areas: Vec<f64> = Vec::new();
    let mut ext_entropies: Vec<f64> = Vec::new();
    for (complex, hs, gs) in [(&pent1, &hs1, &gs1), (&pent2, &hs2, &gs2)] {
        for part in &vertex_bipartitions(complex.n_vertices) {
            let n_cut = cut_area_topological(complex, part) as f64;
            let s = entanglement_for_partition_extended(hs, gs, complex, part);
            ext_areas.push(n_cut);
            ext_entropies.push(s);
        }
    }
    let ext_f: Vec<(f64, f64)> = ext_areas.iter().zip(ext_entropies.iter())
        .filter(|&(&a, &s)| s > 1e-12 && a > 1e-12)
        .map(|(&a, &s)| (a, s))
        .collect();
    let exa: Vec<f64> = ext_f.iter().map(|&(a, _)| a).collect();
    let exe: Vec<f64> = ext_f.iter().map(|&(_, s)| s).collect();
    let (es, ei, er2) = linear_regression(&exa, &exe);
    println!("# Extended prescription regression (all complexes):");
    println!("# N = {}, slope = {es:.6e}, intercept = {ei:.6e}, R^2 = {er2:.6}", ext_f.len());
    println!("# G_N(ext) = {:.6e}", if es.abs() > 1e-15 { 1.0 / (4.0 * es) } else { f64::INFINITY });
    println!();

    eprintln!("  Phase E complete\n");

    // ─────────────────────────────────────────────────────────────────
    // Phase F: Mutual information analysis
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── Phase F: Mutual Information ──\n");

    println!("# Phase F: Mutual information I(A:B) = S_A + S_B - S_AB");
    println!("# I(A:B) is UV-finite and may show cleaner area-law scaling");

    for (name, complex, hs, gs) in [
        ("1-pent", &pent1, &hs1, &gs1),
        ("2-pent", &pent2, &hs2, &gs2),
    ] {
        println!("# {name}");
        println!("partition\t|A|\t#cut\tS_alg\tMI");

        let mut mi_areas: Vec<f64> = Vec::new();
        let mut mi_values: Vec<f64> = Vec::new();

        for part in &vertex_bipartitions(complex.n_vertices) {
            let n_cut = cut_area_topological(complex, part);
            let s = entanglement_for_partition(hs, gs, complex, part);
            let mi = mutual_information(hs, gs, complex, part);
            println!(
                "{}\t{}\t{n_cut}\t{s:.6e}\t{mi:.6e}",
                partition_label(part), part.len()
            );
            if mi > 1e-12 && n_cut > 0 {
                mi_areas.push(n_cut as f64);
                mi_values.push(mi);
            }
        }

        if mi_areas.len() >= 2 {
            let (ms, mi_i, mr2) = linear_regression(&mi_areas, &mi_values);
            println!("# MI regression: slope={ms:.6e}, intercept={mi_i:.6e}, R^2={mr2:.6}");
        }
        println!();
    }

    eprintln!("  Phase F complete\n");

    // ─────────────────────────────────────────────────────────────────
    // Phase G: Finite-size scaling of G_N
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── Phase G: Finite-Size Scaling of G_N ──\n");

    println!("# Phase G: Finite-size scaling of the effective G_N across 1/2/3-pentachoron");
    println!("# complex\tn_vertices\tn_edges\tdim\tslope_alpha\tR^2\tG_N_eff");

    for (name, complex, hs, gs) in [
        ("1-pent", &pent1, &hs1, &gs1),
        ("2-pent", &pent2, &hs2, &gs2),
        ("3-pent", &pent3, &hs3, &gs3),
    ] {
        let mut areas = Vec::new();
        let mut ents = Vec::new();
        for part in &vertex_bipartitions(complex.n_vertices) {
            let n_cut = cut_area_topological(complex, part) as f64;
            let s = entanglement_for_partition(hs, gs, complex, part);
            if s > 1e-12 && n_cut > 0.5 {
                areas.push(n_cut);
                ents.push(s);
            }
        }
        let (sl, _int, r2_local) = linear_regression(&areas, &ents);
        let g_n = if sl.abs() > 1e-15 { 1.0 / (4.0 * sl) } else { f64::INFINITY };
        println!(
            "{name}\t{}\t{}\t{}\t{sl:.6e}\t{r2_local:.6}\t{g_n:.6e}",
            complex.n_vertices, complex.n_edges(), hs.dim()
        );
    }
    println!();

    eprintln!("  Phase G complete\n");

    // ─────────────────────────────────────────────────────────────────
    // Phase H: Triangulated torus RT
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── Phase H: Triangulated Torus RT ──\n");

    let torus = TriangulatedTorus::new(2);
    let ths = TriangulatedTorusHilbert::new(&torus, 1);
    let th = build_triangulated_torus_hamiltonian(&ths, &torus, g_squared);
    let tspec = diag::diagonalize(&th, Some(1));
    let tgs = tspec.ground_state();

    eprintln!(
        "Triangulated 2x2 torus: V={}, E={}, b₁={}, dim={}",
        torus.n_vertices,
        torus.n_edges(),
        torus.n_edges() - torus.n_vertices + 1,
        ths.dim()
    );

    // Vertex bipartitions on the torus.
    let torus_parts = vertex_bipartitions(torus.n_vertices);

    println!("# Phase H: Triangulated 2x2 torus RT");
    println!("# V={}, E={}, dim={}, g^2={g_squared}", torus.n_vertices, torus.n_edges(), ths.dim());
    println!("partition\t|A|\t#cut\tS_alg\tS_ext");

    // Classify edges for torus (edges are (usize, usize) not [usize; 2]).
    let mut torus_areas: Vec<f64> = Vec::new();
    let mut torus_ents: Vec<f64> = Vec::new();

    for part in &torus_parts {
        let mut in_a = vec![false; torus.n_vertices];
        for &v in part.iter() {
            in_a[v] = true;
        }

        // Classify torus edges.
        let mut edges_a = Vec::new();
        let mut edges_ext = Vec::new();
        let mut n_boundary = 0usize;
        for (ei, &(u, v)) in torus.edges.iter().enumerate() {
            let au = in_a[u];
            let av = in_a[v];
            match (au, av) {
                (true, true) => {
                    edges_a.push(ei);
                    edges_ext.push(ei);
                }
                (false, false) => {}
                _ => {
                    n_boundary += 1;
                    edges_ext.push(ei);
                }
            }
        }

        // Compute entanglement entropy using raw basis.
        let s_alg = observables::entanglement_entropy_raw(
            &ths.basis, ths.n_edges, tgs, &edges_a,
        );
        let s_ext = observables::entanglement_entropy_raw(
            &ths.basis, ths.n_edges, tgs, &edges_ext,
        );

        println!(
            "{}\t{}\t{n_boundary}\t{s_alg:.6e}\t{s_ext:.6e}",
            partition_label(part), part.len()
        );

        if s_alg > 1e-12 && n_boundary > 0 {
            torus_areas.push(n_boundary as f64);
            torus_ents.push(s_alg);
        }
    }
    println!();

    if torus_areas.len() >= 2 {
        let (ts, ti, tr2) = linear_regression(&torus_areas, &torus_ents);
        let tgn = if ts.abs() > 1e-15 { 1.0 / (4.0 * ts) } else { f64::INFINITY };
        println!("# Torus RT regression: slope={ts:.6e}, intercept={ti:.6e}, R^2={tr2:.6}, G_N={tgn:.6e}");
        println!();
    }

    eprintln!("  Phase H complete\n");

    // ─────────────────────────────────────────────────────────────────
    // Phase I: SU(2) gauge theory (j_max = 1/2) RT comparison
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── Phase I: SU(2) / Z₂ Gauge Theory RT ──\n");

    println!("# Phase I: SU(2) at j_max = 1/2 (Z₂ reduction), g^2 = {g_squared}");
    println!("# Each edge carries j ∈ {{0, 1/2}}; Gauss law = even parity at each vertex");
    println!("# Hilbert space dimension = 2^b₁ (much smaller than U(1))");
    println!();

    for (name, complex) in [
        ("1-pent", &pent1),
        ("2-pent", &pent2),
        ("3-pent", &pent3),
    ] {
        let su2_hs = Su2HilbertSpace::new(complex);
        let su2_h = su2_quantum::build_su2_hamiltonian(&su2_hs, complex, g_squared);
        let su2_spec = diag::diagonalize(&su2_h, Some(1));
        let su2_gs = su2_spec.ground_state();

        eprintln!(
            "  {name}: SU(2) dim={}, E₀={:.6}",
            su2_hs.dim(),
            su2_spec.ground_energy()
        );

        println!(
            "# {name}: V={}, E={}, b₁={}, SU(2)_dim={}, U(1)_dim (Λ={}): {}",
            complex.n_vertices,
            complex.n_edges(),
            complex.n_edges() - complex.n_vertices + 1,
            su2_hs.dim(),
            lambda,
            if name == "1-pent" {
                hs1.dim()
            } else if name == "2-pent" {
                hs2.dim()
            } else {
                hs3.dim()
            }
        );
        println!("partition\t|A|\t#cut\tS_SU2\tS_U1");

        let mut su2_areas: Vec<f64> = Vec::new();
        let mut su2_ents: Vec<f64> = Vec::new();

        let parts = vertex_bipartitions(complex.n_vertices);
        for part in &parts {
            let n_cut = cut_area_topological(complex, part);
            let s_su2 =
                su2_quantum::su2_entanglement_for_partition(&su2_hs, su2_gs, complex, part);

            // Get corresponding U(1) entropy.
            let s_u1 = if name == "1-pent" {
                entanglement_for_partition(&hs1, &gs1, complex, part)
            } else if name == "2-pent" {
                entanglement_for_partition(&hs2, &gs2, complex, part)
            } else {
                entanglement_for_partition(&hs3, &gs3, complex, part)
            };

            println!(
                "{}\t{}\t{n_cut}\t{s_su2:.6e}\t{s_u1:.6e}",
                partition_label(part),
                part.len()
            );

            if s_su2 > 1e-12 && n_cut > 0 {
                su2_areas.push(n_cut as f64);
                su2_ents.push(s_su2);
            }
        }

        if su2_areas.len() >= 2 {
            let (su2_sl, su2_int, su2_r2) = linear_regression(&su2_areas, &su2_ents);
            let su2_gn = if su2_sl.abs() > 1e-15 {
                1.0 / (4.0 * su2_sl)
            } else {
                f64::INFINITY
            };
            println!(
                "# SU(2) regression: slope={su2_sl:.6e}, intercept={su2_int:.6e}, R²={su2_r2:.6}, G_N={su2_gn:.6e}"
            );
        }
        println!();
    }

    // SU(2) coupling sweep on 1-pentachoron.
    println!("# SU(2) coupling dependence (1-pent, partition={{0,1}})");
    println!("g_squared\tS_SU2\tS_U1");
    let su2_hs1 = Su2HilbertSpace::new(&pent1);
    for &g_sq in &g_sq_sweep {
        let su2_h = su2_quantum::build_su2_hamiltonian(&su2_hs1, &pent1, g_sq);
        let su2_spec = diag::diagonalize(&su2_h, Some(1));
        let su2_gs = su2_spec.ground_state();
        let s_su2 =
            su2_quantum::su2_entanglement_for_partition(&su2_hs1, &su2_gs, &pent1, &test_partition);

        let u1_gs = ground_state_simplicial(&hs1, &pent1, g_sq, None);
        let s_u1 = entanglement_for_partition(&hs1, &u1_gs, &pent1, &test_partition);
        println!("{g_sq:.6e}\t{s_su2:.6e}\t{s_u1:.6e}");
    }
    println!();

    eprintln!("  Phase I complete\n");

    eprintln!("=== Ryu-Takayanagi analysis complete ===");
}
