//! Symmetry search on a Reissner-Nordström background.
//!
//! Builds a discretised RN black hole geometry, constructs all known
//! generators (gauge + translations + conformal), then runs a random
//! search for novel symmetries with the known ones projected out.
//!
//! Configuration via env vars (all optional):
//!   RN_N        grid points per axis   (default 3)
//!   RN_MASS     black hole mass M      (default 0.1)
//!   RN_CHARGE   black hole charge Q    (default 0.0)
//!   RN_RMIN     minimum isotropic r    (default 0.5)
//!   RN_SPACING  base grid spacing      (default 1.0)
//!   RN_SAMPLES  random samples         (default 200)
//!   RN_SEED     RNG seed               (default 42)
//!
//! Run:
//!   cargo run --example rn_symmetry_search -p phyz-regge
//!   RN_N=2 RN_MASS=0.2 cargo run --example rn_symmetry_search -p phyz-regge

use std::env;
use std::time::Instant;

use phyz_regge::symmetry::{
    all_gauge_generators, all_rotation_generators, conformal_generator, translation_generator,
};
use phyz_regge::{
    search_symmetries, ActionParams, Fields, SearchConfig,
};
use phyz_regge::mesh;

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let n: usize = env_or("RN_N", 2);
    let mass: f64 = env_or("RN_MASS", 0.1);
    let charge: f64 = env_or("RN_CHARGE", 0.0);
    let r_min: f64 = env_or("RN_RMIN", 0.5);
    let spacing: f64 = env_or("RN_SPACING", 1.0);
    let n_samples: usize = env_or("RN_SAMPLES", 500);
    let seed: u64 = env_or("RN_SEED", 42);

    println!("=== Reissner-Nordström Symmetry Search ===");
    println!("  n={n}, M={mass}, Q={charge}, r_min={r_min}, spacing={spacing}");
    println!("  samples={n_samples}, seed={seed}");
    println!();

    // --- mesh ---
    let t0 = Instant::now();
    let (complex, lengths) = mesh::reissner_nordstrom(n, spacing, mass, charge, r_min);
    let mesh_time = t0.elapsed();
    println!(
        "Mesh: {} vertices, {} edges  ({:.1?})",
        complex.n_vertices,
        complex.n_edges(),
        mesh_time,
    );

    // --- background fields (zero gauge field — geometry carries the charge) ---
    let phases = vec![0.0; complex.n_edges()];
    let fields = Fields::new(lengths, phases);
    println!("DOF: {} ({} lengths + {} phases)", fields.n_dof(), complex.n_edges(), complex.n_edges());
    println!();

    // --- known generators ---
    let t0 = Instant::now();
    let mut known = all_gauge_generators(&complex);
    let n_gauge = known.len();

    for axis in 0..4 {
        known.push(translation_generator(&complex, &fields, axis, n));
    }

    let rotation_gens = all_rotation_generators(&complex, &fields, n);
    let n_rot = rotation_gens.len();
    known.extend(rotation_gens);

    known.push(conformal_generator(&complex, &fields));
    let gen_time = t0.elapsed();

    println!(
        "Known generators: {} gauge + 4 translation + {} rotation + 1 conformal = {}  ({:.1?})",
        n_gauge,
        n_rot,
        known.len(),
        gen_time,
    );
    println!();

    // Warn if samples are too few for the DOF.
    let dof = fields.n_dof();
    let n_known = known.len();
    if n_samples < dof.saturating_sub(n_known) {
        eprintln!(
            "WARNING: n_samples ({}) < DOF - n_known ({} - {} = {}). \
             Results may be dominated by rank-deficient artifacts. \
             Consider increasing RN_SAMPLES to at least {}.",
            n_samples,
            dof,
            n_known,
            dof.saturating_sub(n_known),
            2 * dof.saturating_sub(n_known),
        );
        eprintln!();
    }

    // --- search ---
    let config = SearchConfig {
        n_samples,
        perturbation_scale: 1e-3,
        seed,
    };
    let params = ActionParams::default();

    let t0 = Instant::now();
    let results = search_symmetries(&complex, &fields, &known, &params, &config);
    let search_time = t0.elapsed();

    println!("{}", results.report());
    println!("Search time: {search_time:.1?}");
    println!();

    // --- highlight novel candidates ---
    let novel = results.novel_candidates(1e-6);
    if novel.is_empty() {
        println!("No novel symmetry candidates (violation < 1e-6).");
    } else {
        println!("=== Novel candidates (violation < 1e-6) ===");
        for (i, c) in novel.iter().enumerate() {
            println!("  [{i}] violation = {:.2e}", c.violation);
            for (name, overlap) in &c.overlaps {
                if overlap.abs() > 1e-3 {
                    println!("       overlap with {name}: {overlap:.4}");
                }
            }
        }
    }

    println!();
    println!(
        "Total: {:.1?}",
        mesh_time + gen_time + search_time,
    );
}
