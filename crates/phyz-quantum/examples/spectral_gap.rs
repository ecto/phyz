//! Deep spectral gap analysis: simplicial (triangular) vs hypercubic (square) plaquettes.
//!
//! Key question: the gap ratio Δ_simp/Δ_hyp ≈ 0.53 at Λ=1, g²=1.
//! Is this universal or Λ-dependent? What's the mechanism?
//!
//! Run:
//!   cargo run --example spectral_gap -p phyz-quantum --release

use phyz_quantum::diag;
use phyz_quantum::hamiltonian::{build_hamiltonian, KSParams};
use phyz_quantum::hilbert::U1HilbertSpace;
use phyz_quantum::hypercubic::{
    build_hypercubic_hamiltonian, HypercubicHilbert, HypercubicLattice,
};
use phyz_regge::complex::SimplicialComplex;

fn main() {
    eprintln!("=== Spectral Gap Analysis: Simplicial vs Hypercubic ===\n");

    let pent = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);
    let lat_2d = HypercubicLattice::new(2, 2);
    let lat_3d = HypercubicLattice::new(2, 3);

    // ─────────────────────────────────────────────────────────────────
    // 1. Gap vs Λ at fixed g² = 1.0
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 1. Gap vs Λ at g² = 1.0 ──\n");

    println!("# Section 1: Gap vs Lambda at g^2 = 1.0");
    println!("# Simplicial: single pentachoron (10 tri plaquettes, 10 edges)");
    println!("# Hypercubic 2D: 2x2 periodic torus (4 sq plaquettes, 4 edges)");
    println!("# Hypercubic 3D: 2^3 periodic torus (12 sq plaquettes, 12 edges)");
    println!("lambda\tsimp_dim\thyp2d_dim\thyp3d_dim\tsimp_E0\tsimp_gap\thyp2d_E0\thyp2d_gap\thyp3d_E0\thyp3d_gap\tratio_2d\tratio_3d");

    for lambda in 1..=2 {
        // Λ=3 has dim=30429 — too slow for dense diag in a quick analysis.
        let hs_s = U1HilbertSpace::new(&pent, lambda);
        let hh_2d = HypercubicHilbert::new(&lat_2d, lambda);
        let hh_3d = HypercubicHilbert::new(&lat_3d, lambda);

        eprintln!(
            "  Λ={lambda}: simp dim={}, hyp2d dim={}, hyp3d dim={}",
            hs_s.dim(), hh_2d.dim(), hh_3d.dim()
        );

        let g_sq = 1.0;
        let params = KSParams { g_squared: g_sq, metric_weights: None };

        let h_s = build_hamiltonian(&hs_s, &pent, &params);
        let spec_s = diag::diagonalize(&h_s, Some(5));

        let h_2 = build_hypercubic_hamiltonian(&hh_2d, &lat_2d, g_sq);
        let spec_2 = diag::diagonalize(&h_2, Some(5));

        let h_3 = build_hypercubic_hamiltonian(&hh_3d, &lat_3d, g_sq);
        let spec_3 = diag::diagonalize(&h_3, Some(5));

        let gap_s = spec_s.gap();
        let gap_2 = spec_2.gap();
        let gap_3 = spec_3.gap();

        println!(
            "{lambda}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.4}\t{:.4}",
            hs_s.dim(), hh_2d.dim(), hh_3d.dim(),
            spec_s.ground_energy(), gap_s,
            spec_2.ground_energy(), gap_2,
            spec_3.ground_energy(), gap_3,
            gap_s / gap_2, gap_s / gap_3,
        );
    }
    println!();

    // ─────────────────────────────────────────────────────────────────
    // 2. Gap vs g² at multiple Λ (both lattice types)
    // ─────────────────────────────────────────────────────────────────
    eprintln!("\n── 2. Gap ratio vs g² at Λ = 1, 2 ──\n");

    println!("# Section 2: Gap ratio vs g^2 at multiple Lambda");
    println!("lambda\tg_squared\tsimp_gap\thyp2d_gap\tratio");

    for lambda in [1, 2] {
        let hs_s = U1HilbertSpace::new(&pent, lambda);
        let hh_2d = HypercubicHilbert::new(&lat_2d, lambda);

        let couplings = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
        for &g_sq in &couplings {
            let params = KSParams { g_squared: g_sq, metric_weights: None };
            let h_s = build_hamiltonian(&hs_s, &pent, &params);
            let spec_s = diag::diagonalize(&h_s, Some(3));

            let h_2 = build_hypercubic_hamiltonian(&hh_2d, &lat_2d, g_sq);
            let spec_2 = diag::diagonalize(&h_2, Some(3));

            let gap_s = spec_s.gap();
            let gap_2 = spec_2.gap();
            let ratio = if gap_2.abs() > 1e-15 { gap_s / gap_2 } else { f64::NAN };

            println!("{lambda}\t{g_sq:.6e}\t{gap_s:.6}\t{gap_2:.6}\t{ratio:.4}");
        }
    }
    println!();

    // ─────────────────────────────────────────────────────────────────
    // 3. Hamiltonian connectivity analysis
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 3. Magnetic Hamiltonian connectivity ──\n");

    println!("# Section 3: Magnetic Hamiltonian connectivity");
    println!("# For each lattice type: count non-zero off-diagonal entries per row");
    println!("# in the magnetic Hamiltonian only (g^2 -> 0 limit)");
    println!("type\tlambda\tdim\tn_plaquettes\tn_edges\tavg_nnz_per_row\tmax_nnz_per_row\tmagnetic_bandwidth");

    for lambda in 1..=1 {
        // Connectivity analysis: O(dim²) scanning, keep at Λ=1 for speed.
        // Simplicial magnetic-only
        let hs_s = U1HilbertSpace::new(&pent, lambda);
        let params_mag = KSParams { g_squared: 1e-6, metric_weights: None };
        let h_s = build_hamiltonian(&hs_s, &pent, &params_mag);
        let dim_s = hs_s.dim();

        let (avg_nnz_s, max_nnz_s) = row_nnz_stats(&h_s, dim_s);
        let spec_s = diag::diagonalize(&h_s, None);
        let bw_s = spec_s.energies.last().unwrap() - spec_s.energies.first().unwrap();

        println!(
            "simplicial\t{lambda}\t{dim_s}\t{}\t{}\t{avg_nnz_s:.2}\t{max_nnz_s}\t{bw_s:.4}",
            pent.n_triangles(), pent.n_edges()
        );

        // Hypercubic 2D magnetic-only
        let hh_2d = HypercubicHilbert::new(&lat_2d, lambda);
        let h_2 = build_hypercubic_hamiltonian(&hh_2d, &lat_2d, 1e-6);
        let dim_2 = hh_2d.dim();

        let (avg_nnz_2, max_nnz_2) = row_nnz_stats(&h_2, dim_2);
        let spec_2 = diag::diagonalize(&h_2, None);
        let bw_2 = spec_2.energies.last().unwrap() - spec_2.energies.first().unwrap();

        println!(
            "hypercubic_2d\t{lambda}\t{dim_2}\t{}\t{}\t{avg_nnz_2:.2}\t{max_nnz_2}\t{bw_2:.4}",
            lat_2d.n_plaquettes(), lat_2d.n_edges()
        );

        // Hypercubic 3D magnetic-only
        let hh_3d = HypercubicHilbert::new(&lat_3d, lambda);
        let h_3 = build_hypercubic_hamiltonian(&hh_3d, &lat_3d, 1e-6);
        let dim_3 = hh_3d.dim();

        let (avg_nnz_3, max_nnz_3) = row_nnz_stats(&h_3, dim_3);
        let spec_3 = diag::diagonalize(&h_3, None);
        let bw_3 = spec_3.energies.last().unwrap() - spec_3.energies.first().unwrap();

        println!(
            "hypercubic_3d\t{lambda}\t{dim_3}\t{}\t{}\t{avg_nnz_3:.2}\t{max_nnz_3}\t{bw_3:.4}",
            lat_3d.n_plaquettes(), lat_3d.n_edges()
        );
    }
    println!();

    // ─────────────────────────────────────────────────────────────────
    // 4. Normalized gap comparisons
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 4. Normalized gap comparisons ──\n");

    println!("# Section 4: Normalized gap at g^2 = 1.0");
    println!("# Compare gap/n_plaquettes, gap/n_edges, gap/b1");
    println!("type\tlambda\tgap\tn_plaq\tn_edge\tb1\tgap_per_plaq\tgap_per_edge\tgap_per_b1");

    for lambda in 1..=2 {
        let g_sq = 1.0;

        // Simplicial
        let hs_s = U1HilbertSpace::new(&pent, lambda);
        let params = KSParams { g_squared: g_sq, metric_weights: None };
        let h_s = build_hamiltonian(&hs_s, &pent, &params);
        let spec_s = diag::diagonalize(&h_s, Some(3));
        let gap_s = spec_s.gap();
        let np_s = pent.n_triangles();
        let ne_s = pent.n_edges();
        let b1_s = ne_s - pent.n_vertices + 1;
        println!(
            "simplicial\t{lambda}\t{gap_s:.6}\t{np_s}\t{ne_s}\t{b1_s}\t{:.6}\t{:.6}\t{:.6}",
            gap_s / np_s as f64, gap_s / ne_s as f64, gap_s / b1_s as f64,
        );

        // Hypercubic 2D
        let hh_2d = HypercubicHilbert::new(&lat_2d, lambda);
        let h_2 = build_hypercubic_hamiltonian(&hh_2d, &lat_2d, g_sq);
        let spec_2 = diag::diagonalize(&h_2, Some(3));
        let gap_2 = spec_2.gap();
        let np_2 = lat_2d.n_plaquettes();
        let ne_2 = lat_2d.n_edges();
        let b1_2 = ne_2 - lat_2d.n_vertices + 1;
        println!(
            "hypercubic_2d\t{lambda}\t{gap_2:.6}\t{np_2}\t{ne_2}\t{b1_2}\t{:.6}\t{:.6}\t{:.6}",
            gap_2 / np_2 as f64, gap_2 / ne_2 as f64, gap_2 / b1_2 as f64,
        );

        // Hypercubic 3D
        let hh_3d = HypercubicHilbert::new(&lat_3d, lambda);
        let h_3 = build_hypercubic_hamiltonian(&hh_3d, &lat_3d, g_sq);
        let spec_3 = diag::diagonalize(&h_3, Some(3));
        let gap_3 = spec_3.gap();
        let np_3 = lat_3d.n_plaquettes();
        let ne_3 = lat_3d.n_edges();
        let b1_3 = ne_3 - lat_3d.n_vertices + 1;
        println!(
            "hypercubic_3d\t{lambda}\t{gap_3:.6}\t{np_3}\t{ne_3}\t{b1_3}\t{:.6}\t{:.6}\t{:.6}",
            gap_3 / np_3 as f64, gap_3 / ne_3 as f64, gap_3 / b1_3 as f64,
        );
    }
    println!();

    // ─────────────────────────────────────────────────────────────────
    // 5. Low-lying spectrum structure
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 5. Low-lying spectrum structure ──\n");

    println!("# Section 5: Low-lying eigenvalues at g^2 = 1.0");
    println!("type\tlambda\tE0\tE1\tE2\tE3\tE4\tE5\tE6\tE7\tE8\tE9");

    for lambda in 1..=2 {
        let g_sq = 1.0;
        let n_low = 10;

        let hs_s = U1HilbertSpace::new(&pent, lambda);
        let params = KSParams { g_squared: g_sq, metric_weights: None };
        let h_s = build_hamiltonian(&hs_s, &pent, &params);
        let spec_s = diag::diagonalize(&h_s, Some(n_low));
        print!("simplicial\t{lambda}");
        for i in 0..n_low.min(spec_s.energies.len()) {
            print!("\t{:.6}", spec_s.energies[i]);
        }
        println!();

        let hh_2d = HypercubicHilbert::new(&lat_2d, lambda);
        let h_2 = build_hypercubic_hamiltonian(&hh_2d, &lat_2d, g_sq);
        let spec_2 = diag::diagonalize(&h_2, Some(n_low));
        print!("hypercubic_2d\t{lambda}");
        for i in 0..n_low.min(spec_2.energies.len()) {
            print!("\t{:.6}", spec_2.energies[i]);
        }
        println!();

        let hh_3d = HypercubicHilbert::new(&lat_3d, lambda);
        let h_3 = build_hypercubic_hamiltonian(&hh_3d, &lat_3d, g_sq);
        let spec_3 = diag::diagonalize(&h_3, Some(n_low));
        print!("hypercubic_3d\t{lambda}");
        for i in 0..n_low.min(spec_3.energies.len()) {
            print!("\t{:.6}", spec_3.energies[i]);
        }
        println!();
    }
    println!();

    // ─────────────────────────────────────────────────────────────────
    // 6. Degeneracy-lifted multiplets
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 6. Ground state degeneracy and multiplet structure ──\n");

    println!("# Section 6: Degeneracy structure at g^2 = 1.0, Λ = 1");
    println!("# Energy levels grouped by degeneracy (threshold = 1e-6)");

    let hs1 = U1HilbertSpace::new(&pent, 1);
    let h1 = build_hamiltonian(&hs1, &pent, &KSParams { g_squared: 1.0, metric_weights: None });
    let spec1 = diag::diagonalize(&h1, Some(30));

    println!("type\tlevel\tenergy\tdegeneracy");
    let mut i = 0;
    let mut level = 0;
    while i < spec1.energies.len() {
        let e = spec1.energies[i];
        let mut deg = 1;
        while i + deg < spec1.energies.len()
            && (spec1.energies[i + deg] - e).abs() < 1e-6
        {
            deg += 1;
        }
        println!("simplicial\t{level}\t{e:.6}\t{deg}");
        i += deg;
        level += 1;
    }

    let hh2 = HypercubicHilbert::new(&lat_2d, 1);
    let h2 = build_hypercubic_hamiltonian(&hh2, &lat_2d, 1.0);
    let spec2 = diag::diagonalize(&h2, None);

    i = 0;
    level = 0;
    while i < spec2.energies.len() {
        let e = spec2.energies[i];
        let mut deg = 1;
        while i + deg < spec2.energies.len()
            && (spec2.energies[i + deg] - e).abs() < 1e-6
        {
            deg += 1;
        }
        println!("hypercubic_2d\t{level}\t{e:.6}\t{deg}");
        i += deg;
        level += 1;
    }
    println!();

    eprintln!("\n=== Spectral gap analysis complete ===");
}

/// Count off-diagonal non-zeros per row.
fn row_nnz_stats(h: &nalgebra::DMatrix<f64>, dim: usize) -> (f64, usize) {
    let mut total = 0usize;
    let mut max = 0usize;
    for i in 0..dim {
        let mut count = 0;
        for j in 0..dim {
            if i != j && h[(i, j)].abs() > 1e-15 {
                count += 1;
            }
        }
        total += count;
        if count > max {
            max = count;
        }
    }
    (total as f64 / dim as f64, max)
}
