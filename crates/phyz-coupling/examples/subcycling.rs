//! Subcycling and multi-timescale integration example.
//!
//! Demonstrates r-RESPA-like subcycling where fast forces are integrated
//! with smaller timesteps than slow forces.

use phyz_coupling::{SubcyclingSchedule, TimeScale};

fn main() {
    println!("Multi-Timescale Subcycling Example");
    println!("===================================\n");

    // Create a schedule with three timescales
    let dt_base = 1e-6; // 1 microsecond base timestep
    let scales = vec![TimeScale::Fast, TimeScale::Medium, TimeScale::Slow];
    let schedule = SubcyclingSchedule::from_timescales(dt_base, &scales);

    println!("Subcycling schedule:");
    println!("  Base timestep: {:.2e} s", dt_base);
    for (level, scale) in scales.iter().enumerate() {
        println!(
            "  Level {}: {:?}, dt = {:.2e} s, ratio = {}",
            level,
            scale,
            schedule.dt_for_level(level),
            schedule.num_substeps(level)
        );
    }
    println!();

    // Simulate the stepping pattern
    let total_steps = 1000;
    println!("Stepping pattern (first 20 base steps):");
    println!("Step   Fast   Medium   Slow");
    println!("-----  -----  -------  -----");

    let mut fast_count = 0;
    let mut medium_count = 0;
    let mut slow_count = 0;

    for step in 0..total_steps {
        let steps_fast = schedule.should_step(0, step);
        let steps_medium = schedule.should_step(1, step);
        let steps_slow = schedule.should_step(2, step);

        if steps_fast {
            fast_count += 1;
        }
        if steps_medium {
            medium_count += 1;
        }
        if steps_slow {
            slow_count += 1;
        }

        if step < 20 {
            println!(
                "{:4}   {:5}  {:7}  {:5}",
                step,
                if steps_fast { "✓" } else { " " },
                if steps_medium { "✓" } else { " " },
                if steps_slow { "✓" } else { " " }
            );
        }
    }

    println!("\nTotal steps executed:");
    println!("  Fast (level 0): {} steps", fast_count);
    println!("  Medium (level 1): {} steps", medium_count);
    println!("  Slow (level 2): {} steps", slow_count);

    let fast_ratio = schedule.num_substeps(0);
    let medium_ratio = schedule.num_substeps(1);
    let slow_ratio = schedule.num_substeps(2);

    println!("\nExpected ratios:");
    println!(
        "  Fast:Medium:Slow = 1:{:.0}:{:.0}",
        medium_ratio as f64 / fast_ratio as f64,
        slow_ratio as f64 / fast_ratio as f64
    );

    println!("\nActual ratios:");
    println!(
        "  Fast:Medium:Slow = {:.1}:{:.1}:{:.1}",
        fast_count as f64 / fast_count as f64,
        fast_count as f64 / medium_count as f64,
        fast_count as f64 / slow_count as f64
    );

    println!("\nThis demonstrates how subcycling reduces computational cost");
    println!("by evaluating slow forces less frequently than fast forces.");
}
