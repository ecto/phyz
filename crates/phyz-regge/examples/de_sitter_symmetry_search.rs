//! Symmetry search on a de Sitter background.
//!
//! de Sitter in conformally flat coordinates is maximally symmetric:
//! all 6 "rotations" (3 spatial + 3 boost) should have comparable O(h²)
//! violation since f = g.
//!
//! Configuration via env vars (all optional):
//!   DS_N        grid points per axis     (default 2)
//!   DS_L        cosmological length L    (default 10.0)
//!   DS_SPACING  base grid spacing        (default 1.0)
//!   DS_SAMPLES  random samples           (default 500)
//!   DS_SEED     RNG seed                 (default 42)
//!   DS_FORM     conformal|static         (default "conformal")
//!   DS_RMIN     min r for static patch   (default 0.1)

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
    let n: usize = env_or("DS_N", 2);
    let cosmological_length: f64 = env_or("DS_L", 10.0);
    let spacing: f64 = env_or("DS_SPACING", 1.0);
    let n_samples: usize = env_or("DS_SAMPLES", 500);
    let seed: u64 = env_or("DS_SEED", 42);
    let form: String = env_or("DS_FORM", "conformal".to_string());
    let r_min: f64 = env_or("DS_RMIN", 0.1);

    println!("=== de Sitter Symmetry Search ({form}) ===");
    println!("  n={n}, L={cosmological_length}, spacing={spacing}");
    println!("  samples={n_samples}, seed={seed}");
    println!();

    let t0 = Instant::now();
    let (complex, lengths) = match form.as_str() {
        "static" => mesh::de_sitter_static(n, spacing, cosmological_length, r_min),
        _ => mesh::de_sitter(n, spacing, cosmological_length),
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
            "WARNING: n_samples ({}) < DOF - n_known ({} - {} = {}). Consider increasing DS_SAMPLES to at least {}.",
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
