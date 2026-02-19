//! Gravitomagnetic transformer simulation.
//!
//! Simulates Swain's gravitational transformer: a circulating mass current
//! (primary) generates gravitomagnetic fields that couple to a secondary
//! loop via the GEM analog of Faraday's law.
//!
//! This is a first-principles simulation using Lorentzian Regge calculus —
//! the gravitomagnetic field *emerges* from solving the discrete Einstein
//! equations, not from the linearized GEM analogy.
//!
//! Configuration via env vars (all optional):
//!   GEM_N_SPATIAL    grid points per axis     (default 4)
//!   GEM_N_TIME       time slices              (default 3)
//!   GEM_SPACING      spatial spacing           (default 1.0)
//!   GEM_DT           time step                 (default 0.3)
//!   GEM_AMP_MIN      minimum amplitude         (default 0.0)
//!   GEM_AMP_MAX      maximum amplitude         (default 1e-4)
//!   GEM_AMP_N        number of amplitudes      (default 5)
//!   GEM_SEARCH       run permeability search    (default false)
//!
//! Run:
//!   cargo run --example gem_transformer -p phyz-regge --release

use phyz_regge::transformer::{
    permeability_search, run_transformer, TransformerConfig,
};

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let n_spatial: usize = env_or("GEM_N_SPATIAL", 4);
    let n_time: usize = env_or("GEM_N_TIME", 3);
    let spacing: f64 = env_or("GEM_SPACING", 1.0);
    let dt: f64 = env_or("GEM_DT", 0.3);
    let amp_min: f64 = env_or("GEM_AMP_MIN", 0.0);
    let amp_max: f64 = env_or("GEM_AMP_MAX", 1e-4);
    let amp_n: usize = env_or("GEM_AMP_N", 5);
    let do_search: bool = env_or("GEM_SEARCH", false);

    let config = TransformerConfig {
        n_spatial,
        n_time,
        spacing,
        dt,
        ..Default::default()
    };

    eprintln!("=== Gravitomagnetic Transformer Simulation ===");
    eprintln!(
        "Grid: {}^3 × {} slices, spacing={}, dt={}",
        n_spatial, n_time, spacing, dt
    );

    // Generate amplitude sweep
    let amplitudes: Vec<f64> = if amp_n <= 1 {
        vec![amp_max]
    } else {
        (0..amp_n)
            .map(|i| amp_min + (amp_max - amp_min) * i as f64 / (amp_n - 1) as f64)
            .collect()
    };

    eprintln!(
        "Amplitudes: {} values in [{:.2e}, {:.2e}]",
        amp_n, amp_min, amp_max
    );
    eprintln!();

    // Run the transformer experiment
    let result = run_transformer(&config, &amplitudes);

    // Print results
    eprintln!("--- Measurements ---");
    eprintln!("{:>12} {:>14} {:>14} {:>12}", "amplitude", "induced_emf", "max_B_g", "residual");
    for m in &result.measurements {
        eprintln!(
            "{:>12.4e} {:>14.6e} {:>14.6e} {:>12.2e}",
            m.amplitude, m.induced_emf, m.max_b_grav, m.residual
        );
    }

    eprintln!();
    eprintln!("--- Coupling Analysis ---");
    eprintln!("Linear coupling:        {:.6e}", result.coupling);
    eprintln!("GEM prediction:         {:.6e}", result.gem_prediction);
    eprintln!("Nonlinear correction:   {:.6e}", result.nonlinear_correction);

    // Permeability search
    if do_search {
        eprintln!();
        eprintln!("=== Permeability Search ===");

        let energy_densities: Vec<f64> = (0..5).map(|i| i as f64 * 1e-4).collect();
        let pressures: Vec<f64> = vec![-1e-4, 0.0, 1e-4];

        let perm = permeability_search(&config, amp_max, &energy_densities, &pressures);

        eprintln!("Vacuum coupling:    {:.6e}", perm.vacuum_coupling);
        eprintln!("Best coupling:      {:.6e}", perm.best_coupling);
        eprintln!(
            "Best core: ρ={:.2e}, p={:.2e}",
            perm.best_params.energy_density, perm.best_params.pressure
        );
        eprintln!("Enhancement factor: {:.4}", perm.enhancement);
    }

    // Output TSV for plotting
    println!("amplitude\tinduced_emf\tmax_B_g\tresidual");
    for m in &result.measurements {
        println!(
            "{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}",
            m.amplitude, m.induced_emf, m.max_b_grav, m.residual
        );
    }
}
