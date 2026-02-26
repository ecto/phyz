//! Seed work units into Supabase via REST API.
//!
//! Usage:
//!   cargo run -p phyz-quantum --example seed_work_units -- <supabase_url> <anon_key>
//!
//! Or with env vars:
//!   SUPABASE_URL=https://xxx.supabase.co SUPABASE_KEY=sb_... cargo run -p phyz-quantum --example seed_work_units
//!
//! Inserts work units in batches of 100 via PostgREST bulk insert.

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

    let work_units = generate_work_units();
    eprintln!("Generated {} work units", work_units.len());

    // Batch insert via PostgREST
    let batch_size = 100;
    let endpoint = format!("{}/rest/v1/work_units", url);
    let client = ureq::Agent::new_with_defaults();

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
            Ok(resp) => {
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

    let triangulation = "s4_level0";
    let n_vertices = 6;
    let n_edges = 15;
    let edge_lengths: Vec<f64> = vec![1.0; n_edges];

    // All non-trivial vertex bipartitions up to 25
    let mut partitions: Vec<Vec<usize>> = Vec::new();

    // Size 1: C(6,1) = 6
    for a in 0..n_vertices {
        partitions.push(vec![a]);
    }

    // Size 2: C(6,2) = 15
    for a in 0..n_vertices {
        for b in (a + 1)..n_vertices {
            partitions.push(vec![a, b]);
        }
    }

    // Size 3: fill to 25
    'outer: for a in 0..n_vertices {
        for b in (a + 1)..n_vertices {
            for c in (b + 1)..n_vertices {
                partitions.push(vec![a, b, c]);
                if partitions.len() >= 25 {
                    break 'outer;
                }
            }
        }
    }

    let mut units = Vec::new();
    for &g2 in &g2_values {
        for partition in &partitions {
            units.push(serde_json::json!({
                "params": {
                    "triangulation": triangulation,
                    "edge_lengths": edge_lengths,
                    "g_squared": g2,
                    "partition": partition,
                }
            }));
        }
    }
    units
}
