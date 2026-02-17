//! Symmetry search on a Kerr background.
//!
//! Supports three Kerr metric forms:
//! - slow: slow-rotation approximation (linear in a)
//! - ks: exact Kerr-Schild (Euclidean)
//! - bl: exact Boyer-Lindquist isotropic
//!
//! Configuration via env vars (all optional):
//!   KERR_N        grid points per axis   (default 2)
//!   KERR_MASS     black hole mass M      (default 0.1)
//!   KERR_SPIN     spin parameter a       (default 0.3)
//!   KERR_RMIN     minimum isotropic r    (default 0.5)
//!   KERR_SPACING  base grid spacing      (default 1.0)
//!   KERR_SAMPLES  random samples         (default 500)
//!   KERR_SEED     RNG seed               (default 42)
//!   KERR_FORM     slow|ks|bl             (default "slow")

use std::env;
use std::time::Instant;

use phyz_regge::mesh;
use phyz_regge::symmetry::{
    all_boost_generators, all_gauge_generators, all_rotation_generators, conformal_generator,
    translation_generator,
};
use phyz_regge::{search_symmetries, ActionParams, Fields, SearchConfig};

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let n: usize = env_or("KERR_N", 2);
    let mass: f64 = env_or("KERR_MASS", 0.1);
    let spin: f64 = env_or("KERR_SPIN", 0.3);
    let r_min: f64 = env_or("KERR_RMIN", 0.5);
    let spacing: f64 = env_or("KERR_SPACING", 1.0);
    let n_samples: usize = env_or("KERR_SAMPLES", 500);
    let seed: u64 = env_or("KERR_SEED", 42);
    let form: String = env_or("KERR_FORM", "slow".to_string());

    println!("=== Kerr Symmetry Search ({form}) ===");
    println!("  n={n}, M={mass}, a={spin}, r_min={r_min}, spacing={spacing}");
    println!("  samples={n_samples}, seed={seed}");
    println!();

    let t0 = Instant::now();
    let (complex, lengths) = match form.as_str() {
        "ks" => mesh::kerr_schild(n, spacing, mass, spin, r_min),
        "bl" => mesh::kerr_bl(n, spacing, mass, spin, r_min),
        _ => mesh::kerr(n, spacing, mass, spin, r_min),
    };
    let mesh_time = t0.elapsed();
    println!(
        "Mesh: {} vertices, {} edges  ({:.1?})",
        complex.n_vertices,
        complex.n_edges(),
        mesh_time,
    );

    let phases = vec![0.0; complex.n_edges()];
    let fields = Fields::new(lengths, phases);
    println!(
        "DOF: {} ({} lengths + {} phases)",
        fields.n_dof(),
        complex.n_edges(),
        complex.n_edges()
    );
    println!();

    let t0 = Instant::now();
    let mut known = all_gauge_generators(&complex);
    let n_gauge = known.len();

    for axis in 0..4 {
        known.push(translation_generator(&complex, &fields, axis, n));
    }

    let rotation_gens = all_rotation_generators(&complex, &fields, n);
    let n_rot = rotation_gens.len();
    known.extend(rotation_gens);

    let boost_gens = all_boost_generators(&complex, &fields, n);
    let n_boost = boost_gens.len();
    known.extend(boost_gens);

    known.push(conformal_generator(&complex, &fields));
    let gen_time = t0.elapsed();

    println!(
        "Known generators: {} gauge + 4 translation + {} rotation + {} boost + 1 conformal = {}  ({:.1?})",
        n_gauge, n_rot, n_boost, known.len(), gen_time,
    );
    println!();

    let dof = fields.n_dof();
    let n_known = known.len();
    if n_samples < dof.saturating_sub(n_known) {
        eprintln!(
            "WARNING: n_samples ({}) < DOF - n_known ({} - {} = {}). Consider increasing KERR_SAMPLES to at least {}.",
            n_samples, dof, n_known, dof.saturating_sub(n_known), 2 * dof.saturating_sub(n_known),
        );
        eprintln!();
    }

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
    println!("Total: {:.1?}", mesh_time + gen_time + search_time);
}
