//! SU(2) j=1/2 Spectral Gap Scaling Study
//!
//! Finite-size scaling of SU(2) lattice gauge theory (Z₂ reduction) on
//! face-sharing pentachoron complexes. Dimensions are 2^b₁ (64→262K for
//! 1-5 pentachorons), well within GPU capacity.
//!
//! Run:
//!   cargo run -p phyz-quantum --features gpu --release --example su2_spectral_scaling

use phyz_quantum::diag;
use phyz_quantum::su2_quantum::{
    build_su2_hamiltonian, su2_entanglement_for_partition, su2_fundamental_loops, su2_wilson_loop,
    Su2HilbertSpace,
};
use phyz_regge::SimplicialComplex;
use std::panic;
use std::time::Instant;

#[cfg(feature = "gpu")]
use phyz_quantum::gpu_lanczos::gpu_lanczos_diagonalize_su2;

fn main() {
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("This example requires the 'gpu' feature.");
        eprintln!(
            "Run with: cargo run -p phyz-quantum --features gpu --release --example su2_spectral_scaling"
        );
        return;
    }

    #[cfg(feature = "gpu")]
    run_scaling_study();
}

#[cfg(feature = "gpu")]
fn run_scaling_study() {
    let gpu_precision = match phyz_gpu::sparse::request_device() {
        Ok((_, _, p)) => {
            eprintln!("GPU precision: {:?}", p);
            p
        }
        Err(e) => {
            eprintln!("No GPU adapter: {e}");
            return;
        }
    };

    let n_eigenvalues = 5;
    let couplings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0];

    let pentachoron_sets: &[&[[usize; 5]]] = &[
        &[[0, 1, 2, 3, 4]],
        &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]],
        &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 3, 6]],
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
        &[
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 5],
            [0, 1, 2, 3, 6],
            [0, 1, 2, 3, 7],
            [0, 1, 2, 3, 8],
            [0, 1, 2, 3, 9],
        ],
        &[
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 5],
            [0, 1, 2, 3, 6],
            [0, 1, 2, 3, 7],
            [0, 1, 2, 3, 8],
            [0, 1, 2, 3, 9],
            [0, 1, 2, 3, 10],
        ],
    ];
    let n_verts = [5, 6, 7, 8, 9, 10, 11];

    // ── Section 1: System Topology ──
    println!("# SU(2) j=1/2 Spectral Gap Scaling Study");
    println!("# Z₂ gauge theory on face-sharing pentachoron complexes");
    println!("# GPU precision: {gpu_precision:?}");
    println!();
    println!("# Section 1: System Topology");
    println!("# All pentachorons share tetrahedron [0,1,2,3]");
    println!("n_pent\tvertices\tedges\ttriangles\tb1\tdim");

    let mut complexes = Vec::new();
    let mut hilbert_spaces = Vec::new();
    let mut fund_loops: Vec<Vec<Vec<usize>>> = Vec::new();

    for (i, &pents) in pentachoron_sets.iter().enumerate() {
        let complex = SimplicialComplex::from_pentachorons(n_verts[i], pents);
        let hs = Su2HilbertSpace::new(&complex);

        let n_e = complex.n_edges();
        let b1 = n_e - complex.n_vertices + 1;

        println!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            i + 1,
            complex.n_vertices,
            n_e,
            complex.n_triangles(),
            b1,
            hs.dim(),
        );

        let loops = su2_fundamental_loops(&complex);
        fund_loops.push(loops);

        complexes.push(complex);
        hilbert_spaces.push(hs);
    }
    println!();

    // ── Section 2: Raw Data ──
    println!("# Section 2: Raw Data");
    print!("n_pent\tg_squared\tdim\tb1\tE0\tgap\tgap_per_b1\tS_EE\tavg_W");
    for k in 0..n_eigenvalues {
        print!("\tE[{k}]");
    }
    println!("\ttime_s");

    struct DataPoint {
        n_pent: usize,
        g_squared: f64,
        b1: usize,
        gap: f64,
        s_ee: f64,
        avg_w: f64,
    }
    let mut all_data: Vec<DataPoint> = Vec::new();

    for si in 0..pentachoron_sets.len() {
        let complex = &complexes[si];
        let hs = &hilbert_spaces[si];
        let n_pent = si + 1;
        let n_e = complex.n_edges();
        let b1 = n_e - complex.n_vertices + 1;
        // Dense for n≤2 (dim ≤ 512), GPU Lanczos for n≥3
        let use_gpu = n_pent >= 3;

        let iter_override = if n_pent >= 3 { Some(300) } else { None };

        for &g_sq in &couplings {
            eprintln!("  n_pent={n_pent}, g²={g_sq}, dim={} ...", hs.dim());
            let t0 = Instant::now();

            let spec = if use_gpu {
                let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    gpu_lanczos_diagonalize_su2(hs, complex, g_sq, None, n_eigenvalues, iter_override)
                }));
                match result {
                    Ok(Ok(s)) => s,
                    Ok(Err(e)) => {
                        eprintln!("  GPU failed: {e}");
                        continue;
                    }
                    Err(_) => {
                        eprintln!("  GPU panicked; skipping");
                        continue;
                    }
                }
            } else {
                let h = build_su2_hamiltonian(hs, complex, g_sq, None);
                diag::diagonalize(&h, Some(n_eigenvalues))
            };

            // Vertex partition: shared face vertices [0,1,2,3]
            let partition_a: Vec<usize> = (0..4).collect();
            let s_ee =
                su2_entanglement_for_partition(hs, spec.ground_state(), complex, &partition_a);

            let avg_w = if fund_loops[si].is_empty() {
                0.0
            } else {
                let sum: f64 = fund_loops[si]
                    .iter()
                    .map(|lp| su2_wilson_loop(hs, spec.ground_state(), lp))
                    .sum();
                sum / fund_loops[si].len() as f64
            };

            let total_time = t0.elapsed().as_secs_f64();

            let e0 = spec.ground_energy();
            let gap = spec.gap();

            print!(
                "{n_pent}\t{g_sq:.1}\t{}\t{b1}\t{e0:.8}\t{gap:.8}\t{:.8}\t{s_ee:.6}\t{avg_w:.6}",
                hs.dim(),
                gap / b1 as f64,
            );
            for k in 0..n_eigenvalues.min(spec.energies.len()) {
                print!("\t{:.8}", spec.energies[k]);
            }
            println!("\t{total_time:.3}");

            all_data.push(DataPoint {
                n_pent,
                g_squared: g_sq,
                b1,
                gap,
                s_ee,
                avg_w,
            });
        }
    }
    println!();

    // ── Section 3: Scaling Exponents ──
    println!("# Section 3: Scaling Exponents");
    println!("# Fit: log(gap) = -beta * log(b1) + const");
    println!("g_squared\tbeta\tR_squared\tintercept");

    for &g_sq in &couplings {
        let points: Vec<(f64, f64)> = all_data
            .iter()
            .filter(|d| (d.g_squared - g_sq).abs() < 1e-10 && d.gap > 0.0)
            .map(|d| (d.b1 as f64, d.gap))
            .collect();

        if points.len() < 2 {
            println!("{g_sq:.1}\tN/A\tN/A\tN/A");
            continue;
        }

        let (beta, r_sq, intercept) = log_log_fit(&points);
        println!("{g_sq:.1}\t{beta:.4}\t{r_sq:.6}\t{intercept:.4}");
    }
    println!();

    // ── Section 4: Wilson Loop Scaling ──
    println!("# Section 4: Wilson Loop Scaling");
    println!("# Average SU(2) Wilson loop ⟨W⟩ as order parameter");
    println!("# ⟨W⟩ → 1: deconfined, ⟨W⟩ → 0: confined");
    println!("# Fit: log(avg_W) = -alpha * log(b1) + const");
    println!("g_squared\talpha\tR_squared\tintercept");

    for &g_sq in &couplings {
        let points: Vec<(f64, f64)> = all_data
            .iter()
            .filter(|d| (d.g_squared - g_sq).abs() < 1e-10 && d.avg_w > 0.0)
            .map(|d| (d.b1 as f64, d.avg_w))
            .collect();

        if points.len() < 2 {
            println!("{g_sq:.1}\tN/A\tN/A\tN/A");
            continue;
        }

        let (alpha, r_sq, intercept) = log_log_fit(&points);
        println!("{g_sq:.1}\t{alpha:.4}\t{r_sq:.6}\t{intercept:.4}");
    }
    println!();

    // ── Section 5: Phase Structure Summary ──
    println!("# Section 5: Phase Structure Summary");
    println!("#");
    println!("# SU(2) j=1/2 scaling exponent beta from gap ~ b1^(-beta):");
    println!("#   beta > 0: gap closes with system size (gapless/critical)");
    println!("#   beta ~ 0: gap persists (gapped/confined)");
    println!("#");

    // Entanglement entropy table
    println!("# Entanglement entropy S_EE (shared face bipartition):");
    print!("# n_pent");
    for &g_sq in &couplings {
        print!("\tg2={g_sq:.1}");
    }
    println!();
    for n in 1..=pentachoron_sets.len() {
        print!("# {n}");
        for &g_sq in &couplings {
            if let Some(d) = all_data
                .iter()
                .find(|d| d.n_pent == n && (d.g_squared - g_sq).abs() < 1e-10)
            {
                print!("\t{:.4}", d.s_ee);
            } else {
                print!("\tN/A");
            }
        }
        println!();
    }
    println!();

    // Wilson loop table
    println!("# Average Wilson loop ⟨W⟩:");
    print!("# n_pent");
    for &g_sq in &couplings {
        print!("\tg2={g_sq:.1}");
    }
    println!();
    for n in 1..=pentachoron_sets.len() {
        print!("# {n}");
        for &g_sq in &couplings {
            if let Some(d) = all_data
                .iter()
                .find(|d| d.n_pent == n && (d.g_squared - g_sq).abs() < 1e-10)
            {
                print!("\t{:.4}", d.avg_w);
            } else {
                print!("\tN/A");
            }
        }
        println!();
    }

    eprintln!("\n=== SU(2) scaling study complete ===");
}

/// Log-log linear regression: y = A * x^(-beta).
/// Returns (beta, R², intercept).
fn log_log_fit(points: &[(f64, f64)]) -> (f64, f64, f64) {
    let n = points.len() as f64;
    let xs: Vec<f64> = points.iter().map(|(x, _)| x.ln()).collect();
    let ys: Vec<f64> = points.iter().map(|(_, y)| y.ln()).collect();

    let x_mean = xs.iter().sum::<f64>() / n;
    let y_mean = ys.iter().sum::<f64>() / n;

    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    for i in 0..points.len() {
        let dx = xs[i] - x_mean;
        let dy = ys[i] - y_mean;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }

    let slope = if sxx > 1e-30 { sxy / sxx } else { 0.0 };
    let intercept = y_mean - slope * x_mean;
    let r_sq = if syy > 1e-30 {
        (sxy * sxy) / (sxx * syy)
    } else {
        0.0
    };

    (-slope, r_sq, intercept)
}
