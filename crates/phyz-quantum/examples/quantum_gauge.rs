//! Comprehensive quantum gauge theory analysis on simplicial complexes.
//!
//! Produces results for Papers 1-2:
//!   1. E₀(g²) coupling sweep (single pentachoron)
//!   2. Simplicial vs hypercubic spectral comparison
//!   3. Curved background spectrum (Schwarzschild, de Sitter)
//!   4. Z₂ stabilizer code parameter tables
//!   5. Wilson loop area law verification
//!
//! Run:
//!   cargo run --example quantum_gauge -p phyz-quantum --release

use phyz_quantum::diag;
use phyz_quantum::hamiltonian::{KSParams, build_hamiltonian};
use phyz_quantum::hilbert::U1HilbertSpace;
use phyz_quantum::hypercubic::{
    HypercubicHilbert, HypercubicLattice, build_hypercubic_hamiltonian,
};
use phyz_quantum::observables;
use phyz_quantum::qubit_map;
use phyz_quantum::stabilizer;
use phyz_regge::complex::SimplicialComplex;

fn main() {
    let lambda = 1;

    eprintln!("=== Hamiltonian Lattice Gauge Theory on Simplicial Complexes ===\n");

    // Build test complexes.
    let pent1 = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);
    // ─────────────────────────────────────────────────────────────────
    // 1. E₀(g²) coupling sweep
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 1. E₀(g²) Coupling Sweep (single pentachoron, Λ={lambda}) ──\n");

    let hs1 = U1HilbertSpace::new(&pent1, lambda);
    eprintln!(
        "Complex: V={}, E={}, b₁={}, Hilbert dim={}",
        pent1.n_vertices,
        pent1.n_edges(),
        pent1.n_edges() - pent1.n_vertices + 1,
        hs1.dim()
    );

    println!("# Section 1: E0(g^2) coupling sweep");
    println!(
        "# complex=pentachoron V=5 E=10 b1=6 lambda={lambda} dim={}",
        hs1.dim()
    );
    println!("g_squared\tE0\tE1\tgap\tE2\tE3\tE4");

    let g_sq_values: Vec<f64> = {
        let mut v = Vec::new();
        // Log-spaced from 0.01 to 100
        let mut g = 0.01;
        while g <= 100.0 + 1e-9 {
            v.push(g);
            g *= 10.0_f64.powf(0.1); // ~25 points per decade
        }
        v
    };

    for &g_sq in &g_sq_values {
        let params = KSParams {
            g_squared: g_sq,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs1, &pent1, &params);
        let spec = diag::diagonalize(&h, Some(5));

        let e: Vec<f64> = (0..5)
            .map(|i| {
                if i < spec.energies.len() {
                    spec.energies[i]
                } else {
                    f64::NAN
                }
            })
            .collect();

        println!(
            "{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}",
            g_sq,
            e[0],
            e[1],
            e[1] - e[0],
            e[2],
            e[3],
            e[4]
        );
    }
    println!();

    eprintln!(
        "  ✓ Coupling sweep complete ({} points)\n",
        g_sq_values.len()
    );

    // ─────────────────────────────────────────────────────────────────
    // 2. Simplicial vs hypercubic comparison
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 2. Simplicial vs Hypercubic Spectral Comparison ──\n");

    let lat_2d = HypercubicLattice::new(2, 2);
    let hh_2d = HypercubicHilbert::new(&lat_2d, lambda);
    eprintln!(
        "Hypercubic 2D (2×2): V={}, E={}, b₁={}, dim={}",
        lat_2d.n_vertices,
        lat_2d.n_edges(),
        lat_2d.n_edges() - lat_2d.n_vertices + 1,
        hh_2d.dim()
    );

    let lat_3d = HypercubicLattice::new(2, 3);
    let hh_3d = HypercubicHilbert::new(&lat_3d, lambda);
    eprintln!(
        "Hypercubic 3D (2³):  V={}, E={}, b₁={}, dim={}",
        lat_3d.n_vertices,
        lat_3d.n_edges(),
        lat_3d.n_edges() - lat_3d.n_vertices + 1,
        hh_3d.dim()
    );

    println!("# Section 2: Simplicial vs hypercubic comparison");
    println!(
        "# simplicial: pentachoron (tri plaquettes), hypercubic_2d: 2x2 torus (sq plaquettes)"
    );
    println!("# hypercubic_3d: 2^3 torus (sq plaquettes)");
    println!("g_squared\tsimp_E0\tsimp_gap\thyp2d_E0\thyp2d_gap\thyp3d_E0\thyp3d_gap");

    let comparison_couplings = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0];
    for &g_sq in &comparison_couplings {
        // Simplicial
        let params = KSParams {
            g_squared: g_sq,
            metric_weights: None,
        };
        let h_s = build_hamiltonian(&hs1, &pent1, &params);
        let spec_s = diag::diagonalize(&h_s, Some(3));

        // Hypercubic 2D
        let h_h2 = build_hypercubic_hamiltonian(&hh_2d, &lat_2d, g_sq);
        let spec_h2 = diag::diagonalize(&h_h2, Some(3));

        // Hypercubic 3D
        let h_h3 = build_hypercubic_hamiltonian(&hh_3d, &lat_3d, g_sq);
        let spec_h3 = diag::diagonalize(&h_h3, Some(3));

        println!(
            "{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}",
            g_sq,
            spec_s.ground_energy(),
            spec_s.gap(),
            spec_h2.ground_energy(),
            spec_h2.gap(),
            spec_h3.ground_energy(),
            spec_h3.gap(),
        );
    }
    println!();

    eprintln!("  ✓ Spectral comparison complete\n");

    // ─────────────────────────────────────────────────────────────────
    // 3. Curved background spectrum
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 3. Curved Background Spectrum ──\n");

    // Curved backgrounds: apply metric weights to the SAME single pentachoron.
    // The phyz-regge mesh generators produce lattices too large for exact diag,
    // so we synthesize metric weights that mimic curvature effects.
    //
    // Approach: use a single pentachoron with unit edge lengths, then compute
    // metric weights as if the geometry were flat, Schwarzschild, or de Sitter.
    // For the pentachoron, "flat" means uniform weights; "curved" means
    // non-uniform weights encoding how curvature modifies the action density.

    // Flat: all triangles have equal weight.
    let n_tri_pent = pent1.n_triangles();
    let flat_weights = vec![1.0; n_tri_pent];

    // Schwarzschild-like: weights vary by triangle, simulating position-dependent
    // metric (triangles "closer to the mass" have larger dual volume / smaller area).
    // Use a simple radial model: weight_t = 1 + M/r_t where r_t is the average
    // vertex distance from vertex 0.
    let mass = 0.3;
    let schw_weights: Vec<f64> = (0..n_tri_pent)
        .map(|ti| {
            let tri = &pent1.triangles[ti];
            let avg_dist = tri.iter().map(|&v| v as f64 + 1.0).sum::<f64>() / 3.0;
            1.0 + mass / avg_dist
        })
        .collect();

    // de Sitter-like: conformal factor 1/(H²τ²) where τ depends on vertex.
    let h_cosmo = 0.3; // H = 1/L
    let ds_weights: Vec<f64> = (0..n_tri_pent)
        .map(|ti| {
            let tri = &pent1.triangles[ti];
            let avg_tau = tri.iter().map(|&v| v as f64 + 1.0).sum::<f64>() / 3.0;
            1.0 / (h_cosmo * h_cosmo * avg_tau * avg_tau)
        })
        .collect();

    println!("# Section 3: Curved background spectra (single pentachoron)");
    println!("# flat: uniform weights, schwarzschild: M={mass}, de_sitter: H={h_cosmo}");

    for (name, weights) in [
        ("flat", &flat_weights),
        ("schwarzschild", &schw_weights),
        ("de_sitter", &ds_weights),
    ] {
        eprintln!(
            "  {name}: weight range [{:.3}, {:.3}]",
            weights.iter().cloned().fold(f64::INFINITY, f64::min),
            weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );

        println!("# background={name}");
        println!("g_squared\tE0\tE1\tgap");

        for &g_sq in &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let params = KSParams {
                g_squared: g_sq,
                metric_weights: Some(weights.clone()),
            };
            let h = build_hamiltonian(&hs1, &pent1, &params);
            let spec = diag::diagonalize(&h, Some(3));

            println!(
                "{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}",
                g_sq,
                spec.ground_energy(),
                if spec.energies.len() > 1 {
                    spec.energies[1]
                } else {
                    f64::NAN
                },
                spec.gap(),
            );
        }
        println!();
    }

    eprintln!("  ✓ Curved background analysis complete\n");

    // ─────────────────────────────────────────────────────────────────
    // 4. Z₂ stabilizer code tables
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 4. Z₂ Stabilizer Code Parameters ──\n");

    println!("# Section 4: Z2 stabilizer code parameters");
    println!("# complex\tn\tk\td\tn_stars\tn_plaquettes\tmin_star_wt\tmax_star_wt");

    let complexes: Vec<(&str, SimplicialComplex)> = vec![
        (
            "1_pent",
            SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]),
        ),
        (
            "2_pent",
            SimplicialComplex::from_pentachorons(6, &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]]),
        ),
        (
            "3_pent",
            SimplicialComplex::from_pentachorons(
                7,
                &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 4, 6]],
            ),
        ),
        (
            "4_pent",
            SimplicialComplex::from_pentachorons(
                8,
                &[
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 5],
                    [0, 1, 2, 4, 6],
                    [0, 1, 3, 4, 7],
                ],
            ),
        ),
    ];

    for (name, complex) in &complexes {
        let code = stabilizer::stabilizer_code(complex);
        let min_w = code.star_weights.iter().min().unwrap_or(&0);
        let max_w = code.star_weights.iter().max().unwrap_or(&0);

        println!(
            "{name}\t{}\t{}\t{}\t{}\t{}\t{min_w}\t{max_w}",
            code.n, code.k, code.d, code.n_stars, code.n_plaquettes
        );
        eprintln!("  {name}: [[{}, {}, {}]]", code.n, code.k, code.d);
    }

    // Hypercubic comparison: 2D toric code
    let lat_toric = HypercubicLattice::new(3, 2);
    eprintln!(
        "  toric_3x3: n={}, k={}",
        lat_toric.n_edges(),
        lat_toric.n_edges() - lat_toric.n_vertices + 1
    );
    println!(
        "toric_3x3\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        lat_toric.n_edges(),
        lat_toric.n_edges() - lat_toric.n_vertices + 1,
        3, // toric code distance = n for n×n torus
        lat_toric.n_vertices,
        lat_toric.n_plaquettes(),
        4,
        4, // uniform weight 4
    );
    println!();

    eprintln!("  ✓ Stabilizer code tables complete\n");

    // ─────────────────────────────────────────────────────────────────
    // 5. Wilson loop area law
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── 5. Wilson Loop Area Law ──\n");

    println!("# Section 5: Wilson loop expectation values");
    println!("# complex=pentachoron lambda={lambda}");

    let loops = observables::fundamental_loops(&pent1);
    eprintln!("  {} fundamental loops found", loops.len());

    // Print loop weights (number of edges).
    for (li, lp) in loops.iter().enumerate() {
        eprintln!("    loop {li}: {} edges", lp.len());
    }

    println!("g_squared\tloop_idx\tloop_weight\tW\tneg_log_W");

    let wilson_couplings = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0];
    for &g_sq in &wilson_couplings {
        let params = KSParams {
            g_squared: g_sq,
            metric_weights: None,
        };
        let h = build_hamiltonian(&hs1, &pent1, &params);
        let spec = diag::diagonalize(&h, Some(1));
        let gs = spec.ground_state();

        for (li, lp) in loops.iter().enumerate() {
            let w = observables::wilson_loop(&hs1, gs, lp);
            let neg_log_w = if w > 1e-15 { -w.ln() } else { f64::INFINITY };
            println!(
                "{:.6e}\t{li}\t{}\t{:.6e}\t{:.6e}",
                g_sq,
                lp.len(),
                w,
                neg_log_w
            );
        }
    }
    println!();

    eprintln!("  ✓ Wilson loop analysis complete\n");

    // ─────────────────────────────────────────────────────────────────
    // Resource estimates
    // ─────────────────────────────────────────────────────────────────
    eprintln!("── Resource Estimates ──\n");

    println!("# Resource estimates");
    println!(
        "# complex\tlambda\tn_edges\tb1\tqubits_per_edge\ttotal_qubits\tgi_qubits\thilbert_dim\ttrotter_gates"
    );

    for (name, complex) in &complexes {
        let b1 = complex.n_edges() - complex.n_vertices + 1;
        for lam in [1u32, 2, 3] {
            // Skip if basis enumeration would be too large (> 10M configs to check).
            let base = 2 * lam + 1;
            let search_space = (base as u64).saturating_pow(b1 as u32);
            if search_space > 10_000_000 {
                // Print estimate without actual enumeration.
                let est = qubit_map::resource_estimate(complex, lam, 0);
                println!(
                    "{name}\t{lam}\t{}\t{}\t{}\t{}\t{}\t(skipped)\t{}",
                    est.n_edges,
                    est.b1,
                    est.qubits_per_edge,
                    est.total_qubits,
                    est.gauge_invariant_qubits,
                    est.trotter_gates_per_step,
                );
                continue;
            }
            let hs = U1HilbertSpace::new(complex, lam);
            let est = qubit_map::resource_estimate(complex, lam, hs.dim());
            println!(
                "{name}\t{lam}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                est.n_edges,
                est.b1,
                est.qubits_per_edge,
                est.total_qubits,
                est.gauge_invariant_qubits,
                est.hilbert_dim,
                est.trotter_gates_per_step,
            );
        }
    }

    eprintln!("\n=== All analyses complete ===");
}
