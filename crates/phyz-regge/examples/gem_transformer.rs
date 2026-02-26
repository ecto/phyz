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
//!   GEM_CONTINUATION use continuation solver    (default true)
//!   GEM_SUBSTEPS     continuation sub-steps     (default 4)
//!   GEM_COMPARE      print linearized comparison (default true)
//!   GEM_SEARCH       run permeability search    (default false)
//!
//! Run:
//!   cargo run --example gem_transformer -p phyz-regge --release

use phyz_regge::foliation::foliated_hypercubic;
use phyz_regge::gem::{b_grav_tensor_frobenius, linearized_b_grav_tidal, vertex_spatial_coords};
use phyz_regge::transformer::{
    TransformerConfig, make_planar_winding, permeability_search, run_transformer,
    run_transformer_continuation,
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
    let use_continuation: bool = env_or("GEM_CONTINUATION", true);
    let substeps: usize = env_or("GEM_SUBSTEPS", 4);
    let do_compare: bool = env_or("GEM_COMPARE", true);
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
        "Grid: {}^3 x {} slices, spacing={}, dt={}",
        n_spatial, n_time, spacing, dt
    );
    eprintln!(
        "Solver: {}",
        if use_continuation {
            format!("continuation ({substeps} sub-steps)")
        } else {
            "independent".to_string()
        }
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
    let result = if use_continuation {
        run_transformer_continuation(&config, &amplitudes, substeps)
    } else {
        run_transformer(&config, &amplitudes)
    };

    // Print results
    eprintln!("--- Measurements ---");
    eprintln!(
        "{:>12} {:>14} {:>14} {:>12}",
        "amplitude", "induced_emf", "max_B_g", "residual"
    );
    for m in &result.measurements {
        eprintln!(
            "{:>12.4e} {:>14.6e} {:>14.6e} {:>12.2e}",
            m.amplitude,
            m.induced_emf,
            m.max_b_grav,
            m.residual()
        );
    }

    eprintln!();
    eprintln!("--- Coupling Analysis ---");
    eprintln!("Linear coupling:        {:.6e}", result.coupling);
    eprintln!("GEM prediction:         {:.6e}", result.gem_prediction);
    eprintln!(
        "Nonlinear correction:   {:.6e}",
        result.nonlinear_correction
    );

    // Linearized GEM comparison
    if do_compare && !result.measurements.is_empty() {
        eprintln!();
        eprintln!("--- Linearized GEM Comparison ---");

        let fc = foliated_hypercubic(n_time, n_spatial);
        let n = n_spatial;

        // Build primary loop coordinates
        let primary = make_planar_winding(&fc, 0, [0, n / 2], [0, n / 2], 0, "primary");
        let loop_coords: Vec<[f64; 3]> = primary
            .vertices
            .iter()
            .map(|&v| vertex_spatial_coords(v, &fc, spacing))
            .collect();

        // Secondary loop center as the field point
        let z_secondary = (n / 2).min(n - 1);
        let secondary =
            make_planar_winding(&fc, 0, [0, n / 2], [0, n / 2], z_secondary, "secondary");
        let secondary_center: [f64; 3] = {
            let mut c = [0.0; 3];
            for &v in &secondary.vertices {
                let coords = vertex_spatial_coords(v, &fc, spacing);
                c[0] += coords[0];
                c[1] += coords[1];
                c[2] += coords[2];
            }
            let n_v = secondary.vertices.len() as f64;
            [c[0] / n_v, c[1] / n_v, c[2] / n_v]
        };

        let fd_eps = spacing * 1e-4;
        eprintln!(
            "{:>12} {:>14} {:>14} {:>10}",
            "amplitude", "B_ij_regge", "B_ij_linear", "ratio"
        );

        for m in &result.measurements {
            // Linearized tidal tensor B_{ij} = -∂_j B^i at secondary center
            let lin_tidal =
                linearized_b_grav_tidal(&loop_coords, m.amplitude, secondary_center, fd_eps);
            let lin_frob = b_grav_tensor_frobenius(&lin_tidal);

            let ratio = if lin_frob > 1e-30 {
                m.b_grav_frobenius / lin_frob
            } else {
                f64::NAN
            };

            eprintln!(
                "{:>12.4e} {:>14.6e} {:>14.6e} {:>10.4}",
                m.amplitude, m.b_grav_frobenius, lin_frob, ratio
            );
        }
    }

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
            "Best core: rho={:.2e}, p={:.2e}",
            perm.best_params.energy_density, perm.best_params.pressure
        );
        eprintln!("Enhancement factor: {:.4}", perm.enhancement);
    }

    // Output TSV for plotting
    println!("amplitude\tinduced_emf\tmax_B_g\tB_ij_frob\tresidual");
    for m in &result.measurements {
        println!(
            "{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}",
            m.amplitude,
            m.induced_emf,
            m.max_b_grav,
            m.b_grav_frobenius,
            m.residual()
        );
    }
}
