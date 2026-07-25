//! Run every solver validation and emit console, Markdown and JSON reports.
//!
//! ```text
//! cargo run --release -p phyz-validate -- target/validation
//! ```

use std::path::PathBuf;

fn main() {
    let out: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/validation".to_string())
        .into();

    let report = phyz_validate::run_all();
    print!("{}", report.to_console());

    if let Err(e) = std::fs::create_dir_all(&out) {
        eprintln!("could not create {}: {e}", out.display());
        std::process::exit(2);
    }
    let md = out.join("validation.md");
    let json = out.join("validation.json");
    std::fs::write(&md, report.to_markdown()).expect("write markdown");
    std::fs::write(&json, report.to_json()).expect("write json");
    println!("\nwrote {} and {}", md.display(), json.display());

    let (_, failed, _) = report.counts();
    if failed > 0 {
        std::process::exit(1);
    }
}
