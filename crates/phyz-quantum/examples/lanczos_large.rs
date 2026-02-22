//! Lanczos analysis on larger systems beyond dense diag reach.
//!
//! Systems:
//!   - 3-pentachoron at Λ=1 (dim ~47K)
//!   - 2-pentachoron at Λ=2 (dim ~269K) — if time permits
//!
//! Run:
//!   cargo run --example lanczos_large -p phyz-quantum --release

use phyz_quantum::lanczos::lanczos_diagonalize;
use phyz_quantum::hilbert::U1HilbertSpace;
use phyz_regge::complex::SimplicialComplex;

fn main() {
    eprintln!("=== Lanczos Large System Analysis ===\n");

    // ─────────────────────────────────────────────────────────────────
    // 1. 1-pentachoron at Λ=2 (dim=4175) — Lanczos vs dense baseline
    // ─────────────────────────────────────────────────────────────────
    {
        let complex = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);
        let hs = U1HilbertSpace::new(&complex, 2);
        eprintln!("── 1-pent Λ=2: dim={} ──", hs.dim());

        println!("# 1-pent Λ=2 (dim={}): Lanczos spectral gap", hs.dim());
        println!("g_squared\tE0\tgap");

        for &g_sq in &[0.5, 1.0, 2.0, 5.0] {
            let spec = lanczos_diagonalize(&hs, &complex, g_sq, None, 5, None);
            println!("{:.6e}\t{:.6}\t{:.6}", g_sq, spec.ground_energy(), spec.gap());
        }
        println!();
    }

    // ─────────────────────────────────────────────────────────────────
    // 2. 2-pentachoron at Λ=1 (dim=3135) — Lanczos gap scan
    // ─────────────────────────────────────────────────────────────────
    {
        let complex = SimplicialComplex::from_pentachorons(
            6, &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]],
        );
        let hs = U1HilbertSpace::new(&complex, 1);
        eprintln!("\n── 2-pent Λ=1: dim={} ──", hs.dim());

        println!("# 2-pent Λ=1 (dim={}): Lanczos spectral gap", hs.dim());
        println!("g_squared\tE0\tgap");

        for &g_sq in &[0.5, 1.0, 2.0, 5.0] {
            let spec = lanczos_diagonalize(&hs, &complex, g_sq, None, 5, None);
            println!("{:.6e}\t{:.6}\t{:.6}", g_sq, spec.ground_energy(), spec.gap());
        }
        println!();
    }

    // ─────────────────────────────────────────────────────────────────
    // 3. 3-pentachoron at Λ=1 — first result at this size
    // ─────────────────────────────────────────────────────────────────
    {
        let complex = SimplicialComplex::from_pentachorons(
            7, &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 4, 6]],
        );
        let hs = U1HilbertSpace::new(&complex, 1);
        let n_e = complex.n_edges();
        let b1 = n_e - complex.n_vertices + 1;
        eprintln!("\n── 3-pent Λ=1: V={}, E={}, b₁={}, dim={} ──",
            complex.n_vertices, n_e, b1, hs.dim());

        println!("# 3-pent Λ=1 (V={} E={} b1={} dim={}): Lanczos spectral gap",
            complex.n_vertices, n_e, b1, hs.dim());
        println!("g_squared\tE0\tgap");

        for &g_sq in &[0.5, 1.0, 2.0, 5.0] {
            let spec = lanczos_diagonalize(&hs, &complex, g_sq, None, 5, None);
            println!("{:.6e}\t{:.6}\t{:.6}", g_sq, spec.ground_energy(), spec.gap());
        }
        println!();
    }

    // ─────────────────────────────────────────────────────────────────
    // 4. Finite-size scaling: gap vs complex size at g²=1
    // ─────────────────────────────────────────────────────────────────
    eprintln!("\n── Finite-size scaling at g²=1 ──");

    println!("# Finite-size scaling: gap vs complex size at g^2=1");
    println!("n_pent\tV\tE\tT\tb1\tdim\tE0\tgap\tgap_per_plaq\tgap_per_b1");

    let configs: Vec<(&str, Vec<[usize; 5]>)> = vec![
        ("1", vec![[0, 1, 2, 3, 4]]),
        ("2", vec![[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]]),
        ("3", vec![[0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 4, 6]]),
    ];

    for (name, pents) in &configs {
        let n_v = pents.iter().flat_map(|p| p.iter()).copied().max().unwrap() + 1;
        let complex = SimplicialComplex::from_pentachorons(n_v, pents);
        let hs = U1HilbertSpace::new(&complex, 1);
        let n_e = complex.n_edges();
        let n_t = complex.n_triangles();
        let b1 = n_e - complex.n_vertices + 1;

        let spec = lanczos_diagonalize(&hs, &complex, 1.0, None, 5, None);
        let gap = spec.gap();

        println!(
            "{name}\t{}\t{n_e}\t{n_t}\t{b1}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            complex.n_vertices, hs.dim(),
            spec.ground_energy(), gap,
            gap / n_t as f64, gap / b1 as f64,
        );
    }

    eprintln!("\n=== Lanczos analysis complete ===");
}
