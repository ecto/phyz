//! Seed work units into Supabase via REST API.
//!
//! Usage:
//!   cargo run -p phyz-quantum --example seed_work_units -- <supabase_url> <anon_key>
//!
//! Or with env vars:
//!   SUPABASE_URL=https://xxx.supabase.co SUPABASE_KEY=sb_... cargo run -p phyz-quantum --example seed_work_units
//!
//! Work unit decomposition (per level):
//!   50 couplings × 2 geometry seeds × (1 base + 2×E_edges perturbations) = units
//!
//! Phase 1 (levels 0+1):
//!   L0: V=6, E=15 → 31 perturbations × 50 × 2 = 3,100
//!   L1: V=7, E=20 → 41 perturbations × 50 × 2 = 4,100
//!   Total: 7,200
//!
//! Phase 2 (level 3):
//!   L3: V=9, E=30 → 61 perturbations × 50 × 2 = 6,100
//!   Total: 6,100
//!
//! Grand total: 13,300

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let url = if args.len() > 1 {
        args[1].clone()
    } else {
        std::env::var("SUPABASE_URL").expect("pass <supabase_url> as arg or set SUPABASE_URL")
    };
    let key = if args.len() > 2 {
        args[2].clone()
    } else {
        std::env::var("SUPABASE_KEY").expect("pass <anon_key> as arg or set SUPABASE_KEY")
    };

    // Check for --delete flag
    let delete_first = args.iter().any(|a| a == "--delete");

    let client = ureq::Agent::new_with_defaults();
    let endpoint = format!("{}/rest/v1/work_units", url);

    if delete_first {
        eprintln!("Deleting all existing work units...");
        // Delete all rows (PostgREST needs a filter; use status != impossible)
        match client
            .delete(&format!("{endpoint}?status=neq.___impossible___"))
            .header("apikey", &key)
            .header("Authorization", &format!("Bearer {key}"))
            .header("Prefer", "return=minimal")
            .call()
        {
            Ok(_) => eprintln!("  deleted."),
            Err(e) => {
                eprintln!("  delete failed: {e}");
                std::process::exit(1);
            }
        }
    }

    let work_units = generate_work_units();
    eprintln!("Generated {} work units", work_units.len());

    // Batch insert via PostgREST
    let batch_size = 100;
    let mut inserted = 0;
    for chunk in work_units.chunks(batch_size) {
        let body = serde_json::to_string(chunk).unwrap();
        match client
            .post(&endpoint)
            .header("apikey", &key)
            .header("Authorization", &format!("Bearer {key}"))
            .header("Content-Type", "application/json")
            .header("Prefer", "return=minimal")
            .send(body.as_bytes())
        {
            Ok(_) => {
                inserted += chunk.len();
                eprint!("\r  inserted {inserted}/{}", work_units.len());
            }
            Err(e) => {
                eprintln!("\n  error at batch starting at {inserted}: {e}");
                std::process::exit(1);
            }
        }
    }
    eprintln!("\nDone. {inserted} work units seeded.");
}

/// Edge counts per level (from subdivided_s4 topology).
fn edges_for_level(level: u32) -> usize {
    match level {
        0 => 15,
        1 => 20,
        2 => 25,
        3 => 30,
        _ => panic!("unsupported level {level}"),
    }
}

fn generate_work_units() -> Vec<serde_json::Value> {
    let n_g2 = 50;
    let g2_min: f64 = 0.01;
    let g2_max: f64 = 100.0;
    let g2_values: Vec<f64> = (0..n_g2)
        .map(|i| {
            let t = i as f64 / (n_g2 - 1) as f64;
            10.0_f64.powf(g2_min.log10() + t * (g2_max.log10() - g2_min.log10()))
        })
        .collect();

    let geometry_seeds: Vec<u64> = vec![1, 2];

    // Phase 1: levels 0, 1. Phase 2: level 3.
    let levels: Vec<u32> = vec![0, 1, 3];

    let mut units = Vec::new();
    for &level in &levels {
        let n_edges = edges_for_level(level);
        for &g2 in &g2_values {
            for &seed in &geometry_seeds {
                // Base perturbation (unperturbed geometry)
                units.push(serde_json::json!({
                    "params": {
                        "level": level,
                        "coupling_g2": g2,
                        "geometry_seed": seed,
                        "perturbation": { "type": "base" },
                    }
                }));

                // Edge perturbations: +ε and -ε for each edge
                for ei in 0..n_edges {
                    for &dir in &[1.0_f64, -1.0] {
                        units.push(serde_json::json!({
                            "params": {
                                "level": level,
                                "coupling_g2": g2,
                                "geometry_seed": seed,
                                "perturbation": {
                                    "type": "edge",
                                    "index": ei,
                                    "direction": dir,
                                },
                            }
                        }));
                    }
                }
            }
        }
    }
    units
}
