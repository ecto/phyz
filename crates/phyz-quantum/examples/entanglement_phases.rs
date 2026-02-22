//! Three entanglement experiments: phase transition, emergent metric, Page curve.
//!
//! Section 1: U(1) entanglement phase transition under λ scan
//! Section 2: Emergent metric from entanglement (invert RT formula)
//! Section 3: Page curve in SU(2) gauge theory

use phyz_quantum::diag;
use phyz_quantum::hamiltonian::{build_hamiltonian, KSParams};
use phyz_quantum::hilbert::U1HilbertSpace;
use phyz_quantum::lanczos::lanczos_diagonalize;
use phyz_quantum::ryu_takayanagi::*;
use phyz_quantum::su2_quantum::{
    build_su2_hamiltonian, su2_entanglement_for_partition, Su2HilbertSpace,
};
use phyz_regge::complex::SimplicialComplex;
use phyz_regge::gauge::metric_weights;
use std::time::Instant;

fn main() {
    let t0 = Instant::now();

    section1_phase_transition();
    section2_emergent_metric();
    section3_page_curve();

    eprintln!("\nTotal elapsed: {:.1}s", t0.elapsed().as_secs_f64());
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Entanglement Phase Transition (U(1) lambda scan)
// ─────────────────────────────────────────────────────────────────────────────

fn section1_phase_transition() {
    eprintln!("=== Section 1: Entanglement Phase Transition ===");

    let complex = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);
    let partition_a = vec![0, 1];

    println!("# Section 1: Entanglement Phase Transition (U(1) lambda scan)");
    println!("# Complex: single pentachoron, partition {{0,1}}");
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        "lambda", "g_squared", "dim", "S_total", "S_shannon", "S_distill", "n_sectors", "gap"
    );

    let g_squareds = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0];
    let dense_threshold = 1_000;

    for lambda in 1..=5u32 {
        let t_lam = Instant::now();
        let hs = U1HilbertSpace::new(&complex, lambda);
        let dim = hs.dim();
        let use_dense = dim <= dense_threshold;

        eprintln!(
            "  lambda={}: dim={}, method={}",
            lambda,
            dim,
            if use_dense { "dense" } else { "lanczos" }
        );

        for &g_sq in &g_squareds {
            let (ground_state, gap) = if use_dense {
                let params = KSParams {
                    g_squared: g_sq,
                    metric_weights: None,
                };
                let h = build_hamiltonian(&hs, &complex, &params);
                let spec = diag::diagonalize(&h, Some(2));
                (spec.states[0].clone(), spec.gap())
            } else {
                let spec = lanczos_diagonalize(&hs, &complex, g_sq, None, 2, Some(200));
                (spec.states[0].clone(), spec.gap())
            };

            let dec =
                entanglement_decomposed_for_partition(&hs, &ground_state, &complex, &partition_a);

            println!(
                "{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{:.6}",
                lambda, g_sq, dim, dec.total, dec.shannon, dec.distillable, dec.n_sectors, gap
            );
        }

        eprintln!("    done in {:.1}s", t_lam.elapsed().as_secs_f64());
    }

    println!();
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Emergent Metric from Entanglement
// ─────────────────────────────────────────────────────────────────────────────

fn section2_emergent_metric() {
    eprintln!("=== Section 2: Emergent Metric from Entanglement ===");

    let complex = SimplicialComplex::from_pentachorons(6, &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]]);
    let n_vertices = 6;
    let n_edges = complex.n_edges();
    let g_squared = 1.0;
    let mass = 0.5;

    // Target geometry
    let target_lengths = schwarzschild_edge_lengths(&complex, mass);
    let target_weights = metric_weights(&complex, &target_lengths);

    // Build SU(2) Hilbert space and compute ground state with target geometry
    let hs = Su2HilbertSpace::new(&complex);
    eprintln!("  2-pentachoron: V={}, E={}, dim={}", n_vertices, n_edges, hs.dim());

    let h = build_su2_hamiltonian(&hs, &complex, g_squared, Some(&target_weights));
    let spec = diag::diagonalize(&h, Some(1));
    let gs = spec.ground_state();

    // Compute S_EE for all bipartitions
    let partitions = vertex_bipartitions(n_vertices);
    let n_parts = partitions.len();

    let entropies: Vec<f64> = partitions
        .iter()
        .map(|p| su2_entanglement_for_partition(&hs, gs, &complex, p))
        .collect();

    eprintln!("  Bipartitions: {}", n_parts);

    println!("# Section 2: Emergent Metric from Entanglement");
    println!("# Complex: 2-pentachoron, g^2={}, target=schwarzschild_M{}", g_squared, mass);
    println!("# Bipartitions: {}", n_parts);

    // Gradient ascent on R² to recover edge lengths from entanglement
    let mut lengths = vec![1.0f64; n_edges];
    let step_size = 0.05;
    let delta = 0.05;
    let n_iter = 500;
    let l_min = 0.1;
    let l_max = 5.0;

    println!("{}\t{}\t{}", "iter", "loss", "R_squared");

    for iter in 0..n_iter {
        // Compute areas for current lengths
        let areas: Vec<f64> = partitions
            .iter()
            .map(|p| cut_area_triangles(&complex, p, &lengths))
            .collect();

        // Fit S = slope * area + intercept
        let (_, _, r_sq) = linear_regression(&areas, &entropies);
        let loss = 1.0 - r_sq;

        if iter % 50 == 0 || iter == n_iter - 1 {
            println!("{}\t{:.6}\t{:.6}", iter, loss, r_sq);
        }

        // Numerical gradient of R² w.r.t. each edge length
        let mut grad = vec![0.0; n_edges];
        for e in 0..n_edges {
            lengths[e] += delta;
            let areas_plus: Vec<f64> = partitions
                .iter()
                .map(|p| cut_area_triangles(&complex, p, &lengths))
                .collect();
            let (_, _, r_sq_plus) = linear_regression(&areas_plus, &entropies);

            lengths[e] -= 2.0 * delta;
            let areas_minus: Vec<f64> = partitions
                .iter()
                .map(|p| cut_area_triangles(&complex, p, &lengths))
                .collect();
            let (_, _, r_sq_minus) = linear_regression(&areas_minus, &entropies);

            lengths[e] += delta; // restore
            grad[e] = (r_sq_plus - r_sq_minus) / (2.0 * delta);
        }

        // Normalize gradient and take fixed-size step (gradient ascent on R²)
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm > 1e-15 {
            for e in 0..n_edges {
                lengths[e] += step_size * grad[e] / grad_norm;
                lengths[e] = lengths[e].clamp(l_min, l_max);
            }
        }
    }

    // Final comparison
    println!("# Final comparison");
    println!(
        "{}\t{}\t{}\t{}",
        "edge", "target_length", "recovered_length", "relative_error"
    );

    for e in 0..n_edges {
        let rel_err = (lengths[e] - target_lengths[e]).abs() / target_lengths[e];
        println!("{}\t{:.6}\t{:.6}\t{:.6}", e, target_lengths[e], lengths[e], rel_err);
    }

    // Correlation between recovered and target
    let (_, _, r_sq) = linear_regression(&target_lengths, &lengths);
    let r = r_sq.sqrt();
    println!("# Correlation: r = {:.4}", r);
    eprintln!("  Correlation r = {:.4}", r);

    println!();
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Page Curve in SU(2) Gauge Theory
// ─────────────────────────────────────────────────────────────────────────────

fn section3_page_curve() {
    eprintln!("=== Section 3: Page Curve ===");

    let complex = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);
    let partition_a = vec![0, 1];

    let hs = Su2HilbertSpace::new(&complex);
    let dim = hs.dim();
    eprintln!("  Single pentachoron: dim={}", dim);

    println!("# Section 3: Page Curve in SU(2) Gauge Theory");
    println!("# Complex: single pentachoron, partition {{0,1}}");
    println!("{}\t{}\t{}\t{}", "g_squared", "eigen_idx", "energy", "S_EE");

    let g_squareds = [0.5, 1.0, 2.0, 5.0];

    for &g_sq in &g_squareds {
        let h = build_su2_hamiltonian(&hs, &complex, g_sq, None);
        let spec = diag::diagonalize(&h, None); // all eigenstates

        for (i, (energy, state)) in spec.energies.iter().zip(spec.states.iter()).enumerate() {
            let s_ee = su2_entanglement_for_partition(&hs, state, &complex, &partition_a);
            println!("{}\t{}\t{:.6}\t{:.6}", g_sq, i, energy, s_ee);
        }
    }

    // Page curve reference: estimate effective dimensions
    // Count distinct A-configurations in the basis to estimate d_A
    let (edges_a, _, _) = classify_edges(&complex, &partition_a);
    let n_a_edges = edges_a.len();
    // For SU(2) j=1/2: each edge is 0 or 1, but gauge constraint restricts
    // d_A_eff ≈ 2^(n_a_free) where n_a_free = n_a_edges - n_a_vertices + 1
    // Count vertices entirely in A
    let a_verts: Vec<usize> = partition_a.clone();
    let n_a_free = if n_a_edges > a_verts.len() {
        n_a_edges - a_verts.len() + 1
    } else {
        1
    };
    let d_a = 1usize << n_a_free;
    let d_b = dim / d_a;
    let (d_small, d_big) = if d_a <= d_b { (d_a, d_b) } else { (d_b, d_a) };
    let s_page = (d_small as f64).ln() - (d_small as f64) / (2.0 * d_big as f64);

    println!("# Page curve reference");
    println!("# d_A_eff = {}, d_B_eff = {}", d_a, d_b);
    println!("# S_Page = {:.4}", s_page);

    println!();
}
