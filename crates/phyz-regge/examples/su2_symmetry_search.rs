//! Symmetry search with SU(2) Yang-Mills gauge field.
//!
//! Searches for novel symmetries of the Einstein-Yang-Mills action
//! on various backgrounds with SU(2) gauge field configurations.
//!
//! Configuration via env vars (all optional):
//!   SU2_N        grid points per axis       (default 2)
//!   SU2_SPACING  base grid spacing           (default 1.0)
//!   SU2_SAMPLES  random samples              (default 500)
//!   SU2_SEED     RNG seed                    (default 42)
//!   SU2_PERT     perturbation scale          (default 1e-3)
//!   SU2_ALPHA    gauge coupling α            (default 1.0)
//!   SU2_BG       background: flat/rn/kerr    (default flat)
//!   SU2_FIELD    field config: zero/random/monopole (default random)
//!   SU2_MASS     black hole mass M           (default 0.1)
//!   SU2_CHARGE   RN charge Q                 (default 0.0)
//!   SU2_SPIN     Kerr spin a                 (default 0.3)
//!   SU2_RMIN     minimum isotropic r         (default 0.5)
//!
//! Run:
//!   cargo run --example su2_symmetry_search -p phyz-regge --release
//!   SU2_BG=rn SU2_FIELD=zero cargo run --example su2_symmetry_search -p phyz-regge --release
//!   SU2_BG=rn SU2_FIELD=monopole cargo run --example su2_symmetry_search -p phyz-regge --release

use std::env;
use std::time::Instant;

use phyz_regge::mesh;
use phyz_regge::search::{search_symmetries_generic, SearchConfig};
use phyz_regge::su2::Su2;
use phyz_regge::yang_mills::{
    all_su2_gauge_generators, einstein_yang_mills_grad,
    su2_conformal_generator, su2_rotation_generator, su2_translation_generator,
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
///
/// For each edge, the SU(2) element is exp(r̂ × ê · σ · f(r)), where r̂ is
/// the radial direction from center and ê is the edge direction. This gives
/// a hedgehog-like configuration respecting spherical symmetry (up to gauge).
fn monopole_field(
    complex: &phyz_regge::SimplicialComplex,
    n: usize,
    strength: f64,
) -> Vec<Su2> {
    use phyz_regge::symmetry::{vertex_coords_4d, vertex_index_4d};
    let _ = vertex_index_4d; // suppress warning

    let center = (n as f64) / 2.0;
    let mut elements = Vec::with_capacity(complex.n_edges());

    for edge in &complex.edges {
        let c0 = vertex_coords_4d(edge[0], n);
        let c1 = vertex_coords_4d(edge[1], n);

        // Midpoint position relative to center (spatial components only)
        let mid = [
            (c0[1] as f64 + c1[1] as f64) / 2.0 - center,
            (c0[2] as f64 + c1[2] as f64) / 2.0 - center,
            (c0[3] as f64 + c1[3] as f64) / 2.0 - center,
        ];

        // Edge direction (spatial components only)
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

        // Cross product: r̂ × ê
        let r_hat = [mid[0] / r, mid[1] / r, mid[2] / r];
        let cross = [
            r_hat[1] * dx[2] - r_hat[2] * dx[1],
            r_hat[2] * dx[0] - r_hat[0] * dx[2],
            r_hat[0] * dx[1] - r_hat[1] * dx[0],
        ];

        // f(r) = strength / (1 + r²) — smooth falloff
        let f = strength / (1.0 + r * r);

        let theta = [f * cross[0], f * cross[1], f * cross[2]];
        elements.push(Su2::exp(&theta));
    }

    elements
}

fn main() {
    let n: usize = env_or("SU2_N", 2);
    let spacing: f64 = env_or("SU2_SPACING", 1.0);
    let n_samples: usize = env_or("SU2_SAMPLES", 500);
    let seed: u64 = env_or("SU2_SEED", 42);
    let pert_scale: f64 = env_or("SU2_PERT", 1e-3);
    let alpha: f64 = env_or("SU2_ALPHA", 1.0);
    let bg = env_str("SU2_BG", "flat");
    let field_type = env_str("SU2_FIELD", "random");
    let mass: f64 = env_or("SU2_MASS", 0.1);
    let charge: f64 = env_or("SU2_CHARGE", 0.0);
    let spin: f64 = env_or("SU2_SPIN", 0.3);
    let r_min: f64 = env_or("SU2_RMIN", 0.5);

    println!("=== SU(2) Yang-Mills Symmetry Search ===");
    println!("  bg={bg}, field={field_type}, n={n}, spacing={spacing}");
    match bg.as_str() {
        "rn" => println!("  M={mass}, Q={charge}, r_min={r_min}"),
        "kerr" => println!("  M={mass}, a={spin}, r_min={r_min}"),
        _ => {}
    }
    println!("  samples={n_samples}, seed={seed}, pert={pert_scale:.0e}, alpha={alpha}");
    println!();

    // --- mesh ---
    let t0 = Instant::now();
    let (complex, lengths) = match bg.as_str() {
        "rn" => mesh::reissner_nordstrom(n, spacing, mass, charge, r_min),
        "kerr" => mesh::kerr_schild(n, spacing, mass, spin, r_min),
        _ => mesh::flat_hypercubic(n, spacing),
    };
    let mesh_time = t0.elapsed();
    let n_edges = complex.n_edges();
    let dim = n_edges + 3 * n_edges;

    println!(
        "Mesh: {} vertices, {} edges  ({:.1?})",
        complex.n_vertices, n_edges, mesh_time,
    );
    println!("DOF: {} ({} lengths + {} SU(2) field)", dim, n_edges, 3 * n_edges);
    println!();

    // --- SU(2) field ---
    let elements: Vec<Su2> = match field_type.as_str() {
        "zero" => vec![Su2::identity(); n_edges],
        "monopole" => monopole_field(&complex, n, 0.5),
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

    // Compute Wilson action for info.
    let s_ym = phyz_regge::yang_mills::wilson_action(&complex, &lengths, &elements);
    let s_r = phyz_regge::regge::regge_action(&complex, &lengths);
    println!("S_R = {s_r:.6e}, S_YM = {s_ym:.6e}, S_total = {:.6e}", s_r + alpha * s_ym);
    println!();

    // --- known generators ---
    let t0 = Instant::now();

    // SU(2) gauge generators (field-dependent via adjoint).
    let mut known = all_su2_gauge_generators(&complex, &elements);
    let n_gauge = known.len();

    // Geometric generators with proper SU(2) field transport.
    for axis in 0..4 {
        known.push(su2_translation_generator(&complex, &lengths, &elements, axis, n));
    }
    let n_trans = 4;

    // Spatial rotations (xy, xz, yz) with SU(2) field transport.
    let mut n_rot = 0;
    for (a1, a2) in [(1, 2), (1, 3), (2, 3)] {
        known.push(su2_rotation_generator(&complex, &lengths, &elements, a1, a2, n));
        n_rot += 1;
    }

    // Boosts (time-space rotations) with SU(2) field transport.
    let mut n_boost = 0;
    for axis in 1..=3 {
        known.push(su2_rotation_generator(&complex, &lengths, &elements, 0, axis, n));
        n_boost += 1;
    }

    // Conformal (pure length scaling, no gauge field effect).
    known.push(su2_conformal_generator(&complex, &lengths));

    let gen_time = t0.elapsed();
    println!(
        "Known generators: {} SU(2) gauge + {} translation + {} rotation + {} boost + 1 conformal = {}  ({:.1?})",
        n_gauge, n_trans, n_rot, n_boost, known.len(), gen_time,
    );
    println!();

    // Warn if samples too few.
    let n_known = known.len();
    if n_samples < dim.saturating_sub(n_known) {
        eprintln!(
            "WARNING: n_samples ({}) < DOF - n_known ({} - {} = {}). Consider increasing SU2_SAMPLES.",
            n_samples, dim, n_known, dim.saturating_sub(n_known),
        );
        eprintln!();
    }

    // --- search ---
    let config = SearchConfig {
        n_samples,
        perturbation_scale: pert_scale,
        seed: seed + 1,
    };

    let bg_lengths = lengths.clone();
    let bg_elements = elements.clone();
    let complex_ref = &complex;

    let t0 = Instant::now();
    let results = search_symmetries_generic(
        dim,
        &known,
        &config,
        |rng| {
            let perturbed_lengths: Vec<f64> = bg_lengths
                .iter()
                .map(|&l| l * (1.0 + pert_scale * (2.0 * rng.r#gen::<f64>() - 1.0)))
                .collect();

            let perturbed_elements: Vec<Su2> = bg_elements
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

            einstein_yang_mills_grad(complex_ref, &perturbed_lengths, &perturbed_elements, alpha)
        },
    );
    let search_time = t0.elapsed();

    // --- analysis ---
    // Don't print full candidate list (too long), show spectrum summary.
    println!("Search time: {search_time:.1?}");
    println!();

    // Spectrum bands.
    let thresholds = [1e-14, 1e-10, 1e-6, 1e-4, 1e-2, 1e-1, 1.0];
    let mut prev = 0.0_f64;
    println!("Violation spectrum:");
    for &t in &thresholds {
        let count = results.candidates.iter().filter(|c| c.violation >= prev && c.violation < t).count();
        if count > 0 {
            println!("  [{:.0e}, {:.0e}): {count}", prev, t);
        }
        prev = t;
    }
    let above = results.candidates.iter().filter(|c| c.violation >= prev).count();
    if above > 0 {
        println!("  [{:.0e}, ∞): {above}", prev);
    }
    println!();

    // Exact symmetries.
    let n_exact = results.candidates.iter().filter(|c| c.violation < 1e-10).count();
    println!("Exact symmetries (violation < 1e-10): {n_exact}");
    println!("Expected SU(2) gauge: {n_gauge}");

    // Novel candidates: low violation, low overlap with all known.
    let novel_threshold = 1e-4;
    let novel: Vec<_> = results
        .candidates
        .iter()
        .filter(|c| {
            c.violation < novel_threshold
                && c.overlaps.iter().all(|(_, o)| o.abs() < 0.3)
        })
        .collect();

    if novel.is_empty() {
        println!("\nNo novel symmetry candidates (violation < {novel_threshold:.0e}, overlap < 0.3).");
    } else {
        println!("\n=== NOVEL CANDIDATES (violation < {novel_threshold:.0e}) ===");
        for (i, c) in novel.iter().enumerate() {
            println!("  [{i}] violation = {:.2e}", c.violation);
            // Show top overlaps.
            let mut top_overlaps: Vec<_> = c.overlaps.iter()
                .filter(|(_, o)| o.abs() > 0.05)
                .collect();
            top_overlaps.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            for (name, overlap) in top_overlaps.iter().take(5) {
                println!("       overlap with {name}: {overlap:.4}");
            }
        }
    }

    // Show first few candidates near the gap.
    println!("\nSpectrum near gap (first 10 above gauge):");
    for (i, c) in results.candidates.iter().skip(n_exact).take(10).enumerate() {
        let idx = n_exact + i;
        print!("  [{idx}] violation={:.2e}", c.violation);
        // Show strongest overlap.
        if let Some((name, overlap)) = c.overlaps.iter().max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap()) {
            if overlap.abs() > 0.1 {
                print!("  strongest: {name}={overlap:.3}");
            }
        }
        println!();
    }

    println!("\nTotal: {:.1?}", mesh_time + gen_time + search_time);
}
