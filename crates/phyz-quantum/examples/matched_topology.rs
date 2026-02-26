//! Matched-topology comparison: triangular vs square plaquettes on the same torus.
//!
//! Both lattice types live on the same n×n periodic 2D torus.
//! The only difference is plaquette shape:
//!   - Hypercubic: square plaquettes (4 edges each)
//!   - Triangulated: triangular plaquettes (3 edges each)
//!
//! This isolates the geometric effect of plaquette shape from topology effects.
//!
//! Run:
//!   cargo run --example matched_topology -p phyz-quantum --release

use phyz_quantum::diag;
use phyz_quantum::hypercubic::{
    HypercubicHilbert, HypercubicLattice, build_hypercubic_hamiltonian,
};
use phyz_quantum::triangulated_torus::{
    TriangulatedTorus, TriangulatedTorusHilbert, build_triangulated_torus_hamiltonian,
};

fn main() {
    eprintln!("=== Matched-Topology Comparison: Triangular vs Square Plaquettes ===\n");

    // ─────────────────────────────────────────────────────────────────
    // 1. Lattice parameters
    // ─────────────────────────────────────────────────────────────────
    println!("# Lattice parameters");
    println!("type\tn\tV\tE\tplaq\tplaq_size\tb1");

    for n in [2, 3] {
        let hyp = HypercubicLattice::new(n, 2);
        let tri = TriangulatedTorus::new(n);

        let hyp_b1 = hyp.n_edges() - hyp.n_vertices + 1;
        let tri_b1 = tri.n_edges() - tri.n_vertices + 1;

        println!(
            "hypercubic\t{n}\t{}\t{}\t{}\t4\t{hyp_b1}",
            hyp.n_vertices,
            hyp.n_edges(),
            hyp.n_plaquettes()
        );
        println!(
            "triangulated\t{n}\t{}\t{}\t{}\t3\t{tri_b1}",
            tri.n_vertices,
            tri.n_edges(),
            tri.n_plaquettes()
        );
    }
    println!();

    // ─────────────────────────────────────────────────────────────────
    // 2. Spectral comparison on 2×2 torus
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 2×2 torus comparison ──\n");

    let hyp2 = HypercubicLattice::new(2, 2);
    let tri2 = TriangulatedTorus::new(2);

    println!("# 2x2 torus spectral comparison");
    println!(
        "# hypercubic: V={} E={} plaq={} b1={}",
        hyp2.n_vertices,
        hyp2.n_edges(),
        hyp2.n_plaquettes(),
        hyp2.n_edges() - hyp2.n_vertices + 1
    );
    println!(
        "# triangulated: V={} E={} plaq={} b1={}",
        tri2.n_vertices,
        tri2.n_edges(),
        tri2.n_plaquettes(),
        tri2.n_edges() - tri2.n_vertices + 1
    );

    for lambda in [1u32, 2, 3] {
        let hh = HypercubicHilbert::new(&hyp2, lambda);
        let th = TriangulatedTorusHilbert::new(&tri2, lambda);

        eprintln!("  Λ={lambda}: hyp dim={}, tri dim={}", hh.dim(), th.dim());

        println!(
            "# lambda={lambda} hyp_dim={} tri_dim={}",
            hh.dim(),
            th.dim()
        );
        println!("lambda\tg_squared\thyp_E0\thyp_gap\ttri_E0\ttri_gap\tgap_ratio");

        let couplings = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
        for &g_sq in &couplings {
            let h_hyp = build_hypercubic_hamiltonian(&hh, &hyp2, g_sq);
            let spec_hyp = diag::diagonalize(&h_hyp, Some(5));

            let h_tri = build_triangulated_torus_hamiltonian(&th, &tri2, g_sq);
            let spec_tri = diag::diagonalize(&h_tri, Some(5));

            let gap_h = spec_hyp.gap();
            let gap_t = spec_tri.gap();
            let ratio = if gap_h.abs() > 1e-15 {
                gap_t / gap_h
            } else {
                f64::NAN
            };

            println!(
                "{lambda}\t{g_sq:.6e}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.4}",
                spec_hyp.ground_energy(),
                gap_h,
                spec_tri.ground_energy(),
                gap_t,
                ratio,
            );
        }
        println!();
    }

    // ─────────────────────────────────────────────────────────────────
    // 3. 3×3 torus (larger system)
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 3×3 torus comparison ──\n");

    let hyp3 = HypercubicLattice::new(3, 2);
    let tri3 = TriangulatedTorus::new(3);

    let hyp3_b1 = hyp3.n_edges() - hyp3.n_vertices + 1;
    let tri3_b1 = tri3.n_edges() - tri3.n_vertices + 1;

    eprintln!(
        "  hyp3: V={} E={} plaq={} b1={}",
        hyp3.n_vertices,
        hyp3.n_edges(),
        hyp3.n_plaquettes(),
        hyp3_b1
    );
    eprintln!(
        "  tri3: V={} E={} plaq={} b1={}",
        tri3.n_vertices,
        tri3.n_edges(),
        tri3.n_plaquettes(),
        tri3_b1
    );

    // b1 for tri3 = 27 - 9 + 1 = 19, so search space at Λ=1 is 3^19 ≈ 1.2 billion.
    // Too large to enumerate. Use Λ=1 only for the hypercubic (b1=10, 3^10=59049).

    {
        let lambda = 1u32;
        let hh3 = HypercubicHilbert::new(&hyp3, lambda);
        eprintln!("  hyp3 Λ={lambda}: dim={}", hh3.dim());

        // Check if triangulated is tractable
        let tri3_search = 3u64.pow(tri3_b1 as u32);
        if tri3_search <= 10_000_000 {
            let th3 = TriangulatedTorusHilbert::new(&tri3, lambda);
            eprintln!("  tri3 Λ={lambda}: dim={}", th3.dim());

            println!(
                "# 3x3 torus Λ={lambda}: hyp_dim={} tri_dim={}",
                hh3.dim(),
                th3.dim()
            );
            println!("lambda\tg_squared\thyp_E0\thyp_gap\ttri_E0\ttri_gap\tgap_ratio");

            for &g_sq in &[0.5, 1.0, 2.0, 5.0] {
                let h_hyp = build_hypercubic_hamiltonian(&hh3, &hyp3, g_sq);
                let spec_hyp = diag::diagonalize(&h_hyp, Some(5));

                let h_tri = build_triangulated_torus_hamiltonian(&th3, &tri3, g_sq);
                let spec_tri = diag::diagonalize(&h_tri, Some(5));

                let gap_h = spec_hyp.gap();
                let gap_t = spec_tri.gap();
                let ratio = if gap_h.abs() > 1e-15 {
                    gap_t / gap_h
                } else {
                    f64::NAN
                };

                println!(
                    "{lambda}\t{g_sq:.6e}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.4}",
                    spec_hyp.ground_energy(),
                    gap_h,
                    spec_tri.ground_energy(),
                    gap_t,
                    ratio,
                );
            }
        } else {
            eprintln!("  tri3 Λ={lambda}: search space {tri3_search} too large, skipping");
            println!(
                "# 3x3 triangulated torus at Λ={lambda}: search space too large ({tri3_search}), skipped"
            );

            // Still output hypercubic data
            println!("# 3x3 hypercubic only: dim={}", hh3.dim());
            println!("lambda\tg_squared\thyp_E0\thyp_gap");
            for &g_sq in &[0.5, 1.0, 2.0, 5.0] {
                let h_hyp = build_hypercubic_hamiltonian(&hh3, &hyp3, g_sq);
                let spec_hyp = diag::diagonalize(&h_hyp, Some(5));
                println!(
                    "{lambda}\t{g_sq:.6e}\t{:.6}\t{:.6}",
                    spec_hyp.ground_energy(),
                    spec_hyp.gap()
                );
            }
        }
    }
    println!();

    // ─────────────────────────────────────────────────────────────────
    // 4. Summary: gap ratio vs Λ on 2×2 torus at g²=1
    // ─────────────────────────────────────────────────────────────────
    println!("# Summary: gap ratio (triangulated/hypercubic) on same 2x2 torus at g^2=1");
    println!("lambda\thyp_dim\ttri_dim\thyp_gap\ttri_gap\tratio");

    for lambda in 1..=3 {
        let hh = HypercubicHilbert::new(&hyp2, lambda);
        let th = TriangulatedTorusHilbert::new(&tri2, lambda);

        let h_hyp = build_hypercubic_hamiltonian(&hh, &hyp2, 1.0);
        let spec_hyp = diag::diagonalize(&h_hyp, Some(3));

        let h_tri = build_triangulated_torus_hamiltonian(&th, &tri2, 1.0);
        let spec_tri = diag::diagonalize(&h_tri, Some(3));

        let ratio = spec_tri.gap() / spec_hyp.gap();
        println!(
            "{lambda}\t{}\t{}\t{:.6}\t{:.6}\t{:.4}",
            hh.dim(),
            th.dim(),
            spec_hyp.gap(),
            spec_tri.gap(),
            ratio,
        );
    }

    eprintln!("\n=== Matched-topology comparison complete ===");
}
