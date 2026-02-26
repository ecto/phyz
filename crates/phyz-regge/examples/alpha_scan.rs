//! α-scan: sweep gauge coupling for Einstein-Yang-Mills phase transitions.
//!
//! For each α value:
//! 1. Start from background + initial SU(2) field
//! 2. Run gradient descent solver → on-shell configuration
//! 3. Build field-dependent known generators at the solution
//! 4. Run symmetry search on the on-shell configuration
//! 5. Record symmetry count and gap ratio
//!
//! Configuration via env vars (all optional):
//!   SCAN_N          grid points per axis         (default 2)
//!   SCAN_SPACING    base grid spacing             (default 1.0)
//!   SCAN_BG         background: flat/rn/kerr      (default flat)
//!   SCAN_FIELD      field config: zero/random/monopole (default random)
//!   SCAN_ALPHA_MIN  minimum α                     (default 0.01)
//!   SCAN_ALPHA_MAX  maximum α                     (default 100.0)
//!   SCAN_ALPHA_N    number of α values (log-spaced) (default 15)
//!   SCAN_SAMPLES    search samples                (default 500)
//!   SCAN_MAX_ITER   solver max iterations         (default 3000)
//!   SCAN_SEED       RNG seed                      (default 42)
//!   SCAN_MASS       black hole mass M             (default 0.1)
//!   SCAN_CHARGE     RN charge Q                   (default 0.0)
//!   SCAN_SPIN       Kerr spin a                   (default 0.3)
//!   SCAN_RMIN       minimum isotropic r           (default 0.5)
//!   SCAN_STRENGTH   monopole field strength       (default 0.5)
//!
//! Run:
//!   cargo run --example alpha_scan -p phyz-regge --release
//!   SCAN_BG=rn SCAN_FIELD=monopole cargo run --example alpha_scan -p phyz-regge --release
//!   SCAN_ALPHA_N=5 cargo run --example alpha_scan -p phyz-regge --release

use std::env;
use std::time::Instant;

use phyz_regge::SimplicialComplex;
use phyz_regge::mesh;
use phyz_regge::search::{SearchConfig, search_symmetries_generic};
use phyz_regge::solver::{SolverConfig, minimize_einstein_yang_mills};
use phyz_regge::su2::Su2;
use phyz_regge::yang_mills::{
    all_su2_gauge_generators, su2_conformal_generator, su2_rotation_generator,
    su2_translation_generator,
};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_str(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Build a "monopole-like" SU(2) field on the lattice.
fn monopole_field(complex: &SimplicialComplex, n: usize, strength: f64) -> Vec<Su2> {
    use phyz_regge::symmetry::vertex_coords_4d;

    let center = (n as f64) / 2.0;
    let mut elements = Vec::with_capacity(complex.n_edges());

    for edge in &complex.edges {
        let c0 = vertex_coords_4d(edge[0], n);
        let c1 = vertex_coords_4d(edge[1], n);

        let mid = [
            (c0[1] as f64 + c1[1] as f64) / 2.0 - center,
            (c0[2] as f64 + c1[2] as f64) / 2.0 - center,
            (c0[3] as f64 + c1[3] as f64) / 2.0 - center,
        ];

        let dx = [
            (c1[1] as i64 - c0[1] as i64) as f64,
            (c1[2] as i64 - c0[2] as i64) as f64,
            (c1[3] as i64 - c0[3] as i64) as f64,
        ];

        let r = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();

        if r < 1e-10 {
            elements.push(Su2::identity());
            continue;
        }

        let r_hat = [mid[0] / r, mid[1] / r, mid[2] / r];
        let cross = [
            r_hat[1] * dx[2] - r_hat[2] * dx[1],
            r_hat[2] * dx[0] - r_hat[0] * dx[2],
            r_hat[0] * dx[1] - r_hat[1] * dx[0],
        ];

        let f = strength / (1.0 + r * r);
        let theta = [f * cross[0], f * cross[1], f * cross[2]];
        elements.push(Su2::exp(&theta));
    }

    elements
}

/// Build all known generators for SU(2) Einstein-Yang-Mills at a given configuration.
fn build_known_generators(
    complex: &SimplicialComplex,
    lengths: &[f64],
    elements: &[Su2],
    n: usize,
) -> Vec<phyz_regge::Generator> {
    let mut known = all_su2_gauge_generators(complex, elements);

    // Translations (4 axes).
    for axis in 0..4 {
        known.push(su2_translation_generator(
            complex, lengths, elements, axis, n,
        ));
    }

    // Spatial rotations (xy, xz, yz).
    for (a1, a2) in [(1, 2), (1, 3), (2, 3)] {
        known.push(su2_rotation_generator(
            complex, lengths, elements, a1, a2, n,
        ));
    }

    // Boosts (time-space rotations).
    for axis in 1..=3 {
        known.push(su2_rotation_generator(
            complex, lengths, elements, 0, axis, n,
        ));
    }

    // Conformal.
    known.push(su2_conformal_generator(complex, lengths));

    known
}

/// Result for one α value.
struct AlphaResult {
    _alpha: f64,
    converged: bool,
    n_exact: usize,
    gap_ratio: f64,
}

fn main() {
    let n: usize = env_or("SCAN_N", 2);
    let spacing: f64 = env_or("SCAN_SPACING", 1.0);
    let bg = env_str("SCAN_BG", "flat");
    let field_type = env_str("SCAN_FIELD", "random");
    let alpha_min: f64 = env_or("SCAN_ALPHA_MIN", 0.01);
    let alpha_max: f64 = env_or("SCAN_ALPHA_MAX", 100.0);
    let alpha_n: usize = env_or("SCAN_ALPHA_N", 15);
    let n_samples: usize = env_or("SCAN_SAMPLES", 500);
    let max_iter: usize = env_or("SCAN_MAX_ITER", 3000);
    let seed: u64 = env_or("SCAN_SEED", 42);
    let mass: f64 = env_or("SCAN_MASS", 0.1);
    let charge: f64 = env_or("SCAN_CHARGE", 0.0);
    let spin: f64 = env_or("SCAN_SPIN", 0.3);
    let r_min: f64 = env_or("SCAN_RMIN", 0.5);
    let strength: f64 = env_or("SCAN_STRENGTH", 0.5);

    println!("=== Alpha Scan: Einstein-Yang-Mills ===");
    println!("  bg={bg}, field={field_type}, n={n}");
    match bg.as_str() {
        "rn" => println!("  M={mass}, Q={charge}, r_min={r_min}"),
        "kerr" => println!("  M={mass}, a={spin}, r_min={r_min}"),
        _ => {}
    }
    println!("  alpha: {alpha_min:.2e} to {alpha_max:.2e} ({alpha_n} values, log-spaced)");
    println!("  solver: max_iter={max_iter}, samples={n_samples}, seed={seed}");
    println!();

    // --- mesh ---
    let t0 = Instant::now();
    let (complex, lengths) = match bg.as_str() {
        "rn" => mesh::reissner_nordstrom(n, spacing, mass, charge, r_min),
        "kerr" => mesh::kerr_schild(n, spacing, mass, spin, r_min),
        _ => mesh::flat_hypercubic(n, spacing),
    };
    let n_edges = complex.n_edges();
    let dim = n_edges + 3 * n_edges;
    println!(
        "Mesh: {} vertices, {} edges, {} DOF  ({:.1?})",
        complex.n_vertices,
        n_edges,
        dim,
        t0.elapsed()
    );
    println!();

    // --- initial SU(2) field ---
    let initial_elements: Vec<Su2> = match field_type.as_str() {
        "zero" => vec![Su2::identity(); n_edges],
        "monopole" => monopole_field(&complex, n, strength),
        _ => {
            let mut rng = StdRng::seed_from_u64(seed);
            (0..n_edges)
                .map(|_| {
                    Su2::exp(&[
                        rng.r#gen::<f64>() * 0.2 - 0.1,
                        rng.r#gen::<f64>() * 0.2 - 0.1,
                        rng.r#gen::<f64>() * 0.2 - 0.1,
                    ])
                })
                .collect()
        }
    };

    // --- generate α values (log-spaced) ---
    let alphas: Vec<f64> = if alpha_n <= 1 {
        vec![alpha_min]
    } else {
        let log_min = alpha_min.ln();
        let log_max = alpha_max.ln();
        (0..alpha_n)
            .map(|i| (log_min + (log_max - log_min) * i as f64 / (alpha_n - 1) as f64).exp())
            .collect()
    };

    // --- header ---
    println!(
        "{:<12} {:>5} {:>10} {:>12} {:>6} {:>10}",
        "alpha", "iter", "|grad|", "S_final", "exact", "gap"
    );
    println!("{}", "-".repeat(62));

    // --- scan ---
    let mut results: Vec<AlphaResult> = Vec::with_capacity(alpha_n);
    let total_t0 = Instant::now();

    for &alpha in &alphas {
        let iter_t0 = Instant::now();

        // 1. Solve for on-shell configuration.
        let solver_config = SolverConfig {
            max_iter,
            grad_tol: 1e-8,
            print_every: 0,
            ..SolverConfig::default()
        };
        let sol = minimize_einstein_yang_mills(
            &complex,
            &lengths,
            &initial_elements,
            alpha,
            &solver_config,
        );

        // 2. Build known generators at the solution.
        let known = build_known_generators(&complex, &sol.lengths, &sol.elements, n);

        // 3. On-shell symmetry search.
        let search_config = SearchConfig {
            n_samples,
            perturbation_scale: 1e-3,
            seed: seed + 1,
        };

        let sol_lengths = sol.lengths.clone();
        let sol_elements = sol.elements.clone();
        let pert_scale = search_config.perturbation_scale;
        let complex_ref = &complex;

        let search_results = search_symmetries_generic(dim, &known, &search_config, |rng| {
            let perturbed_lengths: Vec<f64> = sol_lengths
                .iter()
                .map(|&l| l * (1.0 + pert_scale * (2.0 * rng.r#gen::<f64>() - 1.0)))
                .collect();
            let perturbed_elements: Vec<Su2> = sol_elements
                .iter()
                .map(|u| {
                    let delta = [
                        pert_scale * (2.0 * rng.r#gen::<f64>() - 1.0),
                        pert_scale * (2.0 * rng.r#gen::<f64>() - 1.0),
                        pert_scale * (2.0 * rng.r#gen::<f64>() - 1.0),
                    ];
                    Su2::exp(&delta).mul(u)
                })
                .collect();

            phyz_regge::yang_mills::einstein_yang_mills_grad(
                complex_ref,
                &perturbed_lengths,
                &perturbed_elements,
                alpha,
            )
        });

        // 4. Analyze results.
        let n_exact = search_results
            .candidates
            .iter()
            .filter(|c| c.violation < 1e-10)
            .count();

        // Gap ratio: first violation above threshold / first violation below.
        let gap_ratio = if n_exact > 0 && n_exact < search_results.candidates.len() {
            let last_exact = search_results.candidates[n_exact - 1].violation.max(1e-16);
            let first_noise = search_results.candidates[n_exact].violation;
            first_noise / last_exact
        } else {
            0.0
        };

        let elapsed = iter_t0.elapsed();

        println!(
            "{:<12.2e} {:>5} {:>10.2e} {:>12.4e} {:>6} {:>10.2e}  ({:.1?})",
            alpha, sol.iterations, sol.grad_norm, sol.action, n_exact, gap_ratio, elapsed,
        );

        results.push(AlphaResult {
            _alpha: alpha,
            converged: sol.converged,
            n_exact,
            gap_ratio,
        });
    }

    println!("{}", "-".repeat(62));
    println!("Total time: {:.1?}", total_t0.elapsed());
    println!();

    // --- Phase transition detection ---
    let mut transitions = Vec::new();
    for i in 1..results.len() {
        let prev = &results[i - 1];
        let curr = &results[i];

        // Exact count change.
        let count_diff = (curr.n_exact as i64 - prev.n_exact as i64).unsigned_abs();
        if count_diff >= 2 {
            transitions.push(format!(
                "  alpha={:.2e}: exact count {} -> {} (delta={})",
                curr._alpha, prev.n_exact, curr.n_exact, count_diff,
            ));
        }

        // Gap ratio drop.
        if prev.gap_ratio > 0.0 && curr.gap_ratio > 0.0 && prev.gap_ratio / curr.gap_ratio > 10.0 {
            transitions.push(format!(
                "  alpha={:.2e}: gap ratio {:.2e} -> {:.2e} (10x drop)",
                curr._alpha, prev.gap_ratio, curr.gap_ratio,
            ));
        }
    }

    if transitions.is_empty() {
        println!("Phase transitions: none detected");
    } else {
        println!("Phase transitions detected:");
        for t in &transitions {
            println!("{t}");
        }
    }

    // --- Convergence summary ---
    let n_converged = results.iter().filter(|r| r.converged).count();
    println!();
    println!(
        "Solver convergence: {n_converged}/{} converged (grad < {:.0e})",
        results.len(),
        1e-8
    );

    let exact_range: Vec<usize> = results.iter().map(|r| r.n_exact).collect();
    let min_exact = exact_range.iter().copied().min().unwrap_or(0);
    let max_exact = exact_range.iter().copied().max().unwrap_or(0);
    println!("Exact symmetry range: {min_exact} - {max_exact}");
}
