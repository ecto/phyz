//! Benchmark: GPU vs CPU Lanczos eigensolver.
//!
//! Run with:
//!   cargo run -p phyz-quantum --features gpu --release --example gpu_lanczos_bench

use phyz_quantum::hilbert::U1HilbertSpace;
use phyz_quantum::lanczos::lanczos_diagonalize;
use phyz_regge::SimplicialComplex;
use std::time::Instant;

#[cfg(feature = "gpu")]
use phyz_quantum::gpu_lanczos::gpu_lanczos_diagonalize;

fn main() {
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("This example requires the 'gpu' feature.");
        eprintln!(
            "Run with: cargo run -p phyz-quantum --features gpu --release --example gpu_lanczos_bench"
        );
        return;
    }

    #[cfg(feature = "gpu")]
    run_bench();
}

#[cfg(feature = "gpu")]
fn run_bench() {
    // Report GPU precision
    match phyz_gpu::sparse::request_device() {
        Ok((_, _, precision)) => {
            eprintln!("GPU precision: {:?}", precision);
        }
        Err(e) => {
            eprintln!("No GPU adapter: {e}");
            return;
        }
    }

    let n_eigenvalues = 5;

    // --- 1-pentachoron Lambda=1 (small, for validation) ---
    {
        let complex = SimplicialComplex::from_pentachorons(5, &[[0, 1, 2, 3, 4]]);
        let hs = U1HilbertSpace::new(&complex, 1);
        eprintln!("\n=== 1-pentachoron Lambda=1 (dim={}) ===", hs.dim());

        let t0 = Instant::now();
        let cpu = lanczos_diagonalize(&hs, &complex, 1.0, None, n_eigenvalues, None);
        let cpu_time = t0.elapsed();
        eprintln!("CPU Lanczos: {:.3}ms", cpu_time.as_secs_f64() * 1000.0);

        let t0 = Instant::now();
        let gpu = gpu_lanczos_diagonalize(&hs, &complex, 1.0, None, n_eigenvalues, None).unwrap();
        let gpu_time = t0.elapsed();
        eprintln!("GPU Lanczos: {:.3}ms", gpu_time.as_secs_f64() * 1000.0);

        eprintln!("Eigenvalue comparison:");
        for i in 0..n_eigenvalues {
            eprintln!(
                "  E[{i}]: cpu={:.8}, gpu={:.8}, diff={:.2e}",
                cpu.energies[i],
                gpu.energies[i],
                (cpu.energies[i] - gpu.energies[i]).abs()
            );
        }
    }

    // --- 3-pentachoron Lambda=1 (main benchmark) ---
    {
        let complex = SimplicialComplex::from_pentachorons(
            7,
            &[[0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 3, 6]],
        );
        let hs = U1HilbertSpace::new(&complex, 1);
        eprintln!("\n=== 3-pentachoron Lambda=1 (dim={}) ===", hs.dim());

        let t0 = Instant::now();
        let cpu = lanczos_diagonalize(&hs, &complex, 1.0, None, n_eigenvalues, None);
        let cpu_time = t0.elapsed();
        eprintln!("CPU Lanczos: {:.3}s", cpu_time.as_secs_f64());

        let t0 = Instant::now();
        let gpu = gpu_lanczos_diagonalize(&hs, &complex, 1.0, None, n_eigenvalues, None).unwrap();
        let gpu_time = t0.elapsed();
        eprintln!("GPU Lanczos: {:.3}s", gpu_time.as_secs_f64());

        eprintln!(
            "Speedup: {:.1}x",
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
        );

        eprintln!("Eigenvalue comparison:");
        for i in 0..n_eigenvalues {
            eprintln!(
                "  E[{i}]: cpu={:.8}, gpu={:.8}, diff={:.2e}",
                cpu.energies[i],
                gpu.energies[i],
                (cpu.energies[i] - gpu.energies[i]).abs()
            );
        }
    }

    // --- 4-pentachoron Lambda=1 (GPU only, too slow for CPU) ---
    {
        let complex = SimplicialComplex::from_pentachorons(
            8,
            &[
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 5],
                [0, 1, 2, 3, 6],
                [0, 1, 2, 3, 7],
            ],
        );
        let hs = U1HilbertSpace::new(&complex, 1);
        eprintln!(
            "\n=== 4-pentachoron Lambda=1 (dim={}) [GPU only] ===",
            hs.dim()
        );

        let t0 = Instant::now();
        match gpu_lanczos_diagonalize(&hs, &complex, 1.0, None, n_eigenvalues, None) {
            Ok(gpu) => {
                let gpu_time = t0.elapsed();
                eprintln!("GPU Lanczos: {:.3}s", gpu_time.as_secs_f64());
                eprintln!("Lowest eigenvalues:");
                for (i, e) in gpu.energies.iter().enumerate() {
                    eprintln!("  E[{i}] = {e:.8}");
                }
            }
            Err(e) => {
                eprintln!("GPU Lanczos failed: {e}");
            }
        }
    }
}
