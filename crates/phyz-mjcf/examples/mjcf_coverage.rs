//! Report how well the MJCF parser handles a set of models.
//!
//! Usage: `cargo run -p phyz-mjcf --example mjcf_coverage -- <file-or-dir>...`

use phyz_mjcf::MjcfLoader;
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: mjcf_coverage <file-or-dir>...");
        std::process::exit(2);
    }

    let mut files = Vec::new();
    for arg in &args {
        collect(Path::new(arg), &mut files);
    }
    files.sort();

    let mut ok = 0usize;
    for path in &files {
        match MjcfLoader::from_file(path) {
            Ok(loader) => {
                ok += 1;
                let model = loader.build_model();
                let shapes: usize = model.bodies.iter().map(|b| b.collisions.len()).sum();
                let offset: usize = model
                    .bodies
                    .iter()
                    .flat_map(|b| b.collisions.iter())
                    .filter(|g| !g.is_centered())
                    .count();
                println!(
                    "PASS {:<48} bodies={:<4} nv={:<4} actuators={:<4} shapes={:<4} (offset {:<4}) meshes={}",
                    short(path),
                    model.nbodies(),
                    model.nv,
                    model.actuators.len(),
                    shapes,
                    offset,
                    loader.meshes().len()
                );
                for note in loader.unsupported() {
                    println!("       ~ {}", note.detail);
                }
            }
            Err(e) => println!("FAIL {:<48} {e}", short(path)),
        }
    }
    println!("\n{ok}/{} models parsed", files.len());
}

fn collect(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            collect(&entry.path(), out);
        }
    } else if path.extension().map(|e| e == "xml").unwrap_or(false) {
        out.push(path.to_path_buf());
    }
}

fn short(path: &Path) -> String {
    let parts: Vec<_> = path.components().rev().take(2).collect();
    parts
        .into_iter()
        .rev()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect::<Vec<_>>()
        .join("/")
}
