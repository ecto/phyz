//! Richardson extrapolation to separate real symmetries from discretization artifacts.
//!
//! Runs the symmetry search at multiple mesh resolutions and extrapolates
//! violations to h→0. Violations that vanish are lattice artifacts; those
//! that persist indicate real symmetry breaking.
//!
//! Configuration via env vars (all optional):
//!   RICH_NS       comma-separated grid sizes  (default "2,3")
//!   RICH_BG       background: rn, ds, kerr    (default "rn")
//!   RICH_MASS     mass parameter               (default 0.1)
//!   RICH_CHARGE   charge parameter             (default 0.0)
//!   RICH_SPIN     spin parameter (kerr)        (default 0.3)
//!   RICH_RMIN     minimum r                    (default 0.5)
//!   RICH_L        cosmological length (ds)     (default 10.0)
//!   RICH_SPACING  base grid spacing            (default 1.0)
//!   RICH_SEED     RNG seed                     (default 42)
//!   RICH_MAX_SAMPLES  max samples per resolution (default 20000)
//!   RICH_METHOD   integration: midpoint|simpson (default "midpoint")
//!
//! Example:
//!   cargo run --example richardson_search -p phyz-regge --release
//!   RICH_NS=2,3,4 RICH_METHOD=simpson cargo run --example richardson_search -p phyz-regge --release

use std::env;
use std::time::Instant;

use phyz_regge::action::{ActionParams, Fields};
use phyz_regge::complex::SimplicialComplex;
use phyz_regge::mesh::{self, MetricIntegration};
use phyz_regge::richardson::richardson_extrapolation_with;
use phyz_regge::symmetry::{
    all_boost_generators, all_gauge_generators, all_rotation_generators, conformal_generator,
    translation_generator, Generator,
};

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn build_known(complex: &SimplicialComplex, fields: &Fields, n: usize) -> Vec<Generator> {
    let mut known = all_gauge_generators(complex);
    for axis in 0..4 {
        known.push(translation_generator(complex, fields, axis, n));
    }
    known.extend(all_rotation_generators(complex, fields, n));
    known.extend(all_boost_generators(complex, fields, n));
    known.push(conformal_generator(complex, fields));
    known
}

fn main() {
    let ns_str: String = env_or("RICH_NS", "2,3".to_string());
    let ns: Vec<usize> = ns_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let bg: String = env_or("RICH_BG", "rn".to_string());
    let mass: f64 = env_or("RICH_MASS", 0.1);
    let charge: f64 = env_or("RICH_CHARGE", 0.0);
    let spin: f64 = env_or("RICH_SPIN", 0.3);
    let r_min: f64 = env_or("RICH_RMIN", 0.5);
    let cosmo_l: f64 = env_or("RICH_L", 10.0);
    let spacing: f64 = env_or("RICH_SPACING", 1.0);
    let seed: u64 = env_or("RICH_SEED", 42);
    let max_samples: usize = env_or("RICH_MAX_SAMPLES", 20000);
    let method_str: String = env_or("RICH_METHOD", "midpoint".to_string());

    let integration = match method_str.as_str() {
        "simpson" => MetricIntegration::Simpson,
        _ => MetricIntegration::Midpoint,
    };

    println!("=== Richardson Extrapolation Search ===");
    println!("  background={bg}, ns={ns:?}, spacing={spacing}");
    println!("  method={method_str}, max_samples={max_samples}");
    match bg.as_str() {
        "rn" => println!("  M={mass}, Q={charge}, r_min={r_min}"),
        "ds" => println!("  L={cosmo_l}"),
        "kerr" => println!("  M={mass}, a={spin}, r_min={r_min}"),
        _ => {
            eprintln!("Unknown background '{bg}'. Use rn, ds, or kerr.");
            return;
        }
    }
    println!("  seed={seed}");
    println!();

    let params = ActionParams::default();
    let t0 = Instant::now();

    let results = richardson_extrapolation_with(
        &ns,
        spacing,
        |n| match bg.as_str() {
            "ds" => mesh::de_sitter_with(n, spacing, cosmo_l, integration),
            "kerr" => mesh::kerr_with(n, spacing, mass, spin, r_min, integration),
            _ => mesh::reissner_nordstrom_with(n, spacing, mass, charge, r_min, integration),
        },
        build_known,
        &params,
        seed,
        Some(max_samples),
    );

    let total = t0.elapsed();

    println!();
    println!("{}", results.report());
    println!("Total time: {total:.1?}");
}
