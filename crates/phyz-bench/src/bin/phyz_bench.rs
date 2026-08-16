//! The `phyz-bench` harness.
//!
//! ```text
//! phyz-bench [--suite NAME]... [--quick] [--json PATH] [--markdown PATH]
//! ```
//!
//! Runs the benchmark suites and emits both machine-readable JSON (for
//! tracking over time and rendering on phyz.dev) and Markdown (for
//! `BENCHMARKS.md`). With no `--suite`, every suite runs.

use phyz_bench::report::{Report, SCHEMA_VERSION, Suite};
use phyz_bench::suites;
use phyz_bench::{Budget, meta::RunMeta};

const SUITE_NAMES: [&str; 7] = [
    "single-sim",
    "energy",
    "divergence",
    "gradient",
    "adjoint-scaling",
    "gpu",
    "rapier",
];

fn usage() -> ! {
    eprintln!(
        "usage: phyz-bench [--suite NAME]... [--quick] [--json PATH] [--markdown PATH]\n\
         \n\
         suites: {}\n\
         \n\
         --quick    fewer repetitions; for checking the harness, not for publishing\n\
         --json     write the machine-readable report here (default: bench-results.json)\n\
         --markdown write the rendered tables here (default: stdout only)",
        SUITE_NAMES.join(", ")
    );
    std::process::exit(2)
}

fn main() {
    let mut selected: Vec<String> = Vec::new();
    let mut quick = false;
    let mut json_path = "bench-results.json".to_string();
    let mut markdown_path: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--suite" => selected.push(args.next().unwrap_or_else(|| usage())),
            "--quick" => quick = true,
            "--json" => json_path = args.next().unwrap_or_else(|| usage()),
            "--markdown" => markdown_path = Some(args.next().unwrap_or_else(|| usage())),
            "-h" | "--help" => usage(),
            other => {
                eprintln!("unknown argument: {other}");
                usage()
            }
        }
    }

    for s in &selected {
        if !SUITE_NAMES.contains(&s.as_str()) {
            eprintln!("unknown suite: {s}");
            usage()
        }
    }
    let wants = |name: &str| selected.is_empty() || selected.iter().any(|s| s == name);

    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: this is a debug build. The numbers will be off by an order of \
             magnitude and must not be published. Use `cargo run --release`."
        );
    }

    let budget = if quick {
        Budget::quick()
    } else {
        Budget::default()
    };
    if quick {
        eprintln!("NOTE: --quick reduces repetitions; results are indicative only.");
    }

    let mut suites: Vec<Suite> = Vec::new();
    if wants("single-sim") {
        eprintln!("running: single-sim throughput...");
        suites.push(suites::single_sim::run(budget));
    }
    if wants("energy") {
        eprintln!("running: energy drift...");
        suites.push(suites::energy::run(budget));
    }
    if wants("divergence") {
        eprintln!("running: 1-ulp divergence...");
        suites.push(suites::divergence::run(budget));
    }
    if wants("gradient") {
        eprintln!("running: gradient throughput...");
        suites.push(suites::gradient::run(budget));
    }
    if wants("adjoint-scaling") {
        eprintln!("running: adjoint scaling...");
        suites.push(suites::adjoint_scaling::run(budget));
    }
    if wants("gpu") {
        eprintln!("running: GPU batch sweep...");
        suites.push(suites::gpu_batch::run(budget));
    }
    if wants("rapier") {
        eprintln!("running: cross-library comparison...");
        suites.push(suites::rapier::run(budget));
    }

    let report = Report {
        schema_version: SCHEMA_VERSION,
        meta: RunMeta::collect(),
        suites,
    };

    let json = serde_json::to_string_pretty(&report).expect("serialise report");
    if let Err(e) = std::fs::write(&json_path, &json) {
        eprintln!("failed to write {json_path}: {e}");
        std::process::exit(1);
    }
    eprintln!("wrote {json_path}");

    let markdown = report.to_markdown();
    if let Some(path) = markdown_path {
        if let Err(e) = std::fs::write(&path, &markdown) {
            eprintln!("failed to write {path}: {e}");
            std::process::exit(1);
        }
        eprintln!("wrote {path}");
    }
    println!("{markdown}");
}
