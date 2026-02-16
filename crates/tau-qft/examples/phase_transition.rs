//! Phase transition example for U(1) lattice gauge theory.
//!
//! Demonstrates how plaquette expectation changes with β (coupling strength).

use tau_qft::{HmcParams, Lattice, U1};

fn main() {
    println!("U(1) Lattice Gauge Theory Phase Transition\n");
    println!("Scanning β to observe order-disorder transition");
    println!("{}", "=".repeat(60));

    // Scan different coupling strengths
    let betas = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0];
    let lattice_size = 4;

    let hmc_params = HmcParams {
        n_md_steps: 10,
        dt: 0.1,
        metropolis: true,
    };

    for &beta in &betas {
        let mut lattice =
            Lattice::<U1>::new(lattice_size, lattice_size, lattice_size, lattice_size, beta);

        // Hot start
        lattice.randomize();

        println!("\nβ = {:.1}", beta);
        println!("Thermalizing...");

        // Thermalization
        for step in 0..200 {
            let state = lattice.hmc_step(&hmc_params);

            if step % 50 == 0 {
                let plaq = lattice.measure_plaquette();
                let action = lattice.action();
                println!(
                    "  Step {:3}: ⟨Tr U_plaq⟩ = {:.4}, S = {:.2}, acc = {:.2}",
                    step,
                    plaq,
                    action,
                    state.acceptance_rate()
                );
            }
        }

        // Measurement phase
        println!("Measuring...");
        let mut plaq_sum = 0.0;
        let mut action_sum = 0.0;
        let n_measure = 100;

        for _ in 0..n_measure {
            lattice.hmc_step(&hmc_params);
            plaq_sum += lattice.measure_plaquette();
            action_sum += lattice.action();
        }

        let plaq_avg = plaq_sum / n_measure as f64;
        let action_avg = action_sum / n_measure as f64;

        println!("Final results:");
        println!("  ⟨Tr U_plaq⟩ = {:.4}", plaq_avg);
        println!("  ⟨S⟩         = {:.2}", action_avg);

        // Interpretation
        if plaq_avg > 0.25 {
            println!("  → Ordered phase (confined)");
        } else {
            println!("  → Disordered phase (deconfined)");
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("Expected behavior:");
    println!("  Small β → disordered (weak coupling, high temperature)");
    println!("  Large β → ordered (strong coupling, low temperature)");
    println!("  Critical β ~ 1-2 for 4D U(1) lattice");
}
