//! Full analysis on 2 pentachorons sharing a face (Λ=1, dim=3135).
//!
//! Extends the single-pentachoron results to a larger complex:
//!   1. Coupling sweep E₀(g²)
//!   2. Spectral gap — finite-size comparison with 1 pentachoron
//!   3. Wilson loops at multiple areas (different loop lengths)
//!   4. Entanglement entropy across the shared face
//!   5. Degeneracy / multiplet structure
//!   6. Stabilizer code parameters
//!
//! Run:
//!   cargo run --example two_pentachoron -p phyz-quantum --release

use phyz_quantum::diag;
use phyz_quantum::hamiltonian::{build_hamiltonian, KSParams};
use phyz_quantum::hilbert::U1HilbertSpace;
use phyz_quantum::observables;
use phyz_quantum::stabilizer;
use phyz_regge::complex::SimplicialComplex;

fn main() {
    let lambda = 1u32;

    // Two pentachorons sharing face [0,1,2,3]:
    // pent1 = [0,1,2,3,4], pent2 = [0,1,2,3,5]
    let complex = SimplicialComplex::from_pentachorons(6, &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]]);

    let hs = U1HilbertSpace::new(&complex, lambda);

    let n_v = complex.n_vertices;
    let n_e = complex.n_edges();
    let n_t = complex.n_triangles();
    let b1 = n_e - n_v + 1;

    eprintln!("=== 2-Pentachoron Analysis (Λ={lambda}) ===\n");
    eprintln!("  V={n_v}, E={n_e}, T={n_t}, b₁={b1}, dim={}\n", hs.dim());

    // ─────────────────────────────────────────────────────────────────
    // 1. Coupling sweep
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 1. E₀(g²) coupling sweep ──\n");

    println!("# Section 1: Coupling sweep (2 pentachorons, Λ={lambda}, dim={})", hs.dim());
    println!("# V={n_v} E={n_e} T={n_t} b1={b1}");
    println!("g_squared\tE0\tE1\tgap\tE2\tE3\tE4");

    let couplings = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0];
    for &g_sq in &couplings {
        let params = KSParams { g_squared: g_sq, metric_weights: None };
        let h = build_hamiltonian(&hs, &complex, &params);
        let spec = diag::diagonalize(&h, Some(5));

        let e: Vec<f64> = (0..5).map(|i| {
            if i < spec.energies.len() { spec.energies[i] } else { f64::NAN }
        }).collect();

        println!(
            "{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}",
            g_sq, e[0], e[1], e[1] - e[0], e[2], e[3], e[4]
        );
    }
    println!();

    // ─────────────────────────────────────────────────────────────────
    // 2. Spectral gap comparison: 1-pent vs 2-pent
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 2. Spectral gap: 1-pent vs 2-pent ──\n");

    let pent1 = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);
    let hs1 = U1HilbertSpace::new(&pent1, lambda);

    println!("# Section 2: Finite-size comparison at Λ={lambda}");
    println!("# 1-pent: V=5 E=10 b1=6 dim={}", hs1.dim());
    println!("# 2-pent: V={n_v} E={n_e} b1={b1} dim={}", hs.dim());
    println!("g_squared\tpent1_E0\tpent1_gap\tpent2_E0\tpent2_gap\tgap_ratio");

    let gap_couplings = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
    for &g_sq in &gap_couplings {
        let params = KSParams { g_squared: g_sq, metric_weights: None };

        let h1 = build_hamiltonian(&hs1, &pent1, &params);
        let spec1 = diag::diagonalize(&h1, Some(3));

        let h2 = build_hamiltonian(&hs, &complex, &params);
        let spec2 = diag::diagonalize(&h2, Some(3));

        let gap1 = spec1.gap();
        let gap2 = spec2.gap();
        let ratio = if gap1.abs() > 1e-15 { gap2 / gap1 } else { f64::NAN };

        println!(
            "{:.6e}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.4}",
            g_sq,
            spec1.ground_energy(), gap1,
            spec2.ground_energy(), gap2,
            ratio,
        );
    }
    println!();

    // ─────────────────────────────────────────────────────────────────
    // 3. Wilson loops at multiple areas
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 3. Wilson loops ──\n");

    let loops = observables::fundamental_loops(&complex);
    eprintln!("  {} fundamental loops:", loops.len());
    for (li, lp) in loops.iter().enumerate() {
        eprintln!("    loop {li}: {} edges", lp.len());
    }

    // Group loops by length (= area proxy).
    let mut loops_by_len: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (li, lp) in loops.iter().enumerate() {
        loops_by_len.entry(lp.len()).or_default().push(li);
    }

    println!("# Section 3: Wilson loops (2 pentachorons)");
    println!("# loops grouped by length (area proxy)");
    println!("g_squared\tloop_len\tn_loops\tavg_W\tneg_log_avg_W");

    let wilson_couplings = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0];
    for &g_sq in &wilson_couplings {
        let params = KSParams { g_squared: g_sq, metric_weights: None };
        let h = build_hamiltonian(&hs, &complex, &params);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        for (&len, indices) in &loops_by_len {
            let avg_w: f64 = indices.iter()
                .map(|&li| observables::wilson_loop(&hs, gs, &loops[li]))
                .sum::<f64>() / indices.len() as f64;
            let neg_log = if avg_w > 1e-15 { -avg_w.ln() } else { f64::INFINITY };

            println!(
                "{:.6e}\t{len}\t{}\t{:.6e}\t{:.4}",
                g_sq, indices.len(), avg_w, neg_log
            );
        }
    }
    println!();

    // ─────────────────────────────────────────────────────────────────
    // 4. Entanglement entropy
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 4. Entanglement entropy ──\n");

    // Natural bipartition: edges belonging to pent1 only vs shared + pent2.
    // The shared face [0,1,2,3] has edges [01,02,03,12,13,23] = 6 edges.
    // Total edges: the edges of pent1 + extra edges to vertex 5.
    // Pent1 edges are [01,02,03,04,12,13,14,23,24,34] = 10.
    // Extra edges: [05,15,25,35] = 4 new edges.
    // Shared: [01,02,03,12,13,23] = 6.
    // Pent1-only: [04,14,24,34] = 4.
    // Pent2-only: [05,15,25,35] = 4.
    //
    // Let's cut at the shared face: edges_A = edges within pent1 only.
    // We need to identify which edge indices belong to which set.

    // For a clean bipartition, split edges into those touching vertex 4 vs vertex 5.
    let edges_touching_4: Vec<usize> = (0..n_e)
        .filter(|&ei| {
            let e = &complex.edges[ei];
            e[0] == 4 || e[1] == 4
        })
        .collect();
    let edges_touching_5: Vec<usize> = (0..n_e)
        .filter(|&ei| {
            let e = &complex.edges[ei];
            e[0] == 5 || e[1] == 5
        })
        .collect();
    let shared_edges: Vec<usize> = (0..n_e)
        .filter(|&ei| {
            let e = &complex.edges[ei];
            e[0] < 4 && e[1] < 4
        })
        .collect();

    eprintln!("  edges touching v4: {:?} ({})", edges_touching_4, edges_touching_4.len());
    eprintln!("  edges touching v5: {:?} ({})", edges_touching_5, edges_touching_5.len());
    eprintln!("  shared edges [0-3]: {:?} ({})", shared_edges, shared_edges.len());

    // Bipartition A: all edges from pent1 (touching v4 + shared)
    let mut edges_a: Vec<usize> = shared_edges.clone();
    edges_a.extend_from_slice(&edges_touching_4);
    edges_a.sort();
    edges_a.dedup();

    eprintln!("  partition A (pent1): {} edges", edges_a.len());
    eprintln!("  partition B (pent2-only): {} edges", n_e - edges_a.len());

    println!("# Section 4: Entanglement entropy across shared face");
    println!("# partition A = pent1 edges ({} edges), B = pent2-only edges ({} edges)", edges_a.len(), n_e - edges_a.len());
    println!("g_squared\tS_A\tS_max");

    // Maximum entropy = log(min(dim_A, dim_B))
    for &g_sq in &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0] {
        let params = KSParams { g_squared: g_sq, metric_weights: None };
        let h = build_hamiltonian(&hs, &complex, &params);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        let s = observables::entanglement_entropy(&hs, gs, &edges_a);
        let s_max = (edges_a.len().min(n_e - edges_a.len()) as f64 * (2.0 * lambda as f64 + 1.0)).ln();

        println!("{:.6e}\t{:.6}\t{:.4}", g_sq, s, s_max);
    }
    println!();

    // Also try a symmetric cut: half the shared edges + one pent
    let edges_half: Vec<usize> = (0..n_e / 2).collect();
    println!("# Symmetric half-half cut ({} / {} edges)", edges_half.len(), n_e - edges_half.len());
    println!("g_squared\tS_half");

    for &g_sq in &[0.1, 1.0, 10.0] {
        let params = KSParams { g_squared: g_sq, metric_weights: None };
        let h = build_hamiltonian(&hs, &complex, &params);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        let s = observables::entanglement_entropy(&hs, gs, &edges_half);
        println!("{:.6e}\t{:.6}", g_sq, s);
    }
    println!();

    // ─────────────────────────────────────────────────────────────────
    // 5. Degeneracy / multiplet structure
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 5. Multiplet structure ──\n");

    let params = KSParams { g_squared: 1.0, metric_weights: None };
    let h = build_hamiltonian(&hs, &complex, &params);
    let spec = diag::diagonalize(&h, Some(30));

    println!("# Section 5: Degeneracy structure (2 pent, g^2=1.0, Λ={lambda})");
    println!("level\tenergy\tdegeneracy");

    let mut i = 0;
    let mut level = 0;
    while i < spec.energies.len() {
        let e = spec.energies[i];
        let mut deg = 1;
        while i + deg < spec.energies.len()
            && (spec.energies[i + deg] - e).abs() < 1e-6
        {
            deg += 1;
        }
        println!("{level}\t{e:.6}\t{deg}");
        i += deg;
        level += 1;
    }
    println!();

    // ─────────────────────────────────────────────────────────────────
    // 6. Stabilizer code
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 6. Stabilizer code ──\n");

    let code = stabilizer::stabilizer_code(&complex);
    println!("# Section 6: Z2 stabilizer code (2 pentachorons)");
    println!("n\tk\td\tn_stars\tn_plaquettes");
    println!("{}\t{}\t{}\t{}\t{}", code.n, code.k, code.d, code.n_stars, code.n_plaquettes);
    println!("star_weights\t{:?}", code.star_weights);
    println!("plaquette_weight\t{}", code.plaquette_weight);

    eprintln!("\n=== 2-Pentachoron analysis complete ===");
}
