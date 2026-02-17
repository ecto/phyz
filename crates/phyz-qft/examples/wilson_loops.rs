//! Wilson loop measurements to study confinement.
//!
//! Computes Wilson loops of various sizes to extract string tension.

use phyz_qft::{HmcParams, Lattice, U1};

fn main() {
    println!("Wilson Loop Measurements for U(1) Lattice");
    println!("{}", "=".repeat(60));

    let beta = 2.0;
    let lattice_size = 8; // Larger for better Wilson loops

    let mut lattice =
        Lattice::<U1>::new(lattice_size, lattice_size, lattice_size, lattice_size, beta);

    println!("Lattice: {}^4, β = {:.1}\n", lattice_size, beta);

    let hmc_params = HmcParams {
        n_md_steps: 10,
        dt: 0.1,
        metropolis: true,
    };

    // Hot start
    lattice.randomize();

    // Thermalize
    println!("Thermalizing (500 steps)...");
    for _ in 0..500 {
        lattice.hmc_step(&hmc_params);
    }

    // Measure Wilson loops of various sizes
    println!("\nMeasuring Wilson loops...\n");

    let sizes = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)];
    let n_measure = 50;

    println!("  R   T   ⟨W(R,T)⟩    -ln⟨W⟩   σ_eff");
    println!("{}", "-".repeat(50));

    for &(r, t) in &sizes {
        let mut wilson_sum = 0.0;

        for _ in 0..n_measure {
            lattice.hmc_step(&hmc_params);
            let wilson = lattice.measure_wilson_loop(r, t);
            wilson_sum += wilson.value;
        }

        let wilson_avg = wilson_sum / n_measure as f64;
        let neg_log_w = -wilson_avg.ln();
        let area = (r * t) as f64;
        let sigma_eff = neg_log_w / area;

        println!(
            "  {}   {}   {:.4}      {:.3}    {:.4}",
            r, t, wilson_avg, neg_log_w, sigma_eff
        );
    }

    println!("\n{}", "=".repeat(60));
    println!("Interpretation:");
    println!("  ⟨W(R,T)⟩ ~ exp(-σ RT) for large loops (area law)");
    println!("  σ_eff should converge for confined phase");
    println!("  Perimeter law (⟨W⟩ ~ exp(-cP)) for deconfined phase");
}
