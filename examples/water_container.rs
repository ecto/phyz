//! Water filling a container simulation.
//!
//! Demonstrates fluid MPM with water-like equation of state.

use tau::{Material, MpmSolver, Particle, Vec3, material::EquationOfState};

fn main() {
    println!("Water filling container simulation");
    println!("==================================\n");

    // Setup
    let h = 0.02; // 2cm grid spacing
    let dt = 0.001; // 1ms timestep
    let bounds = (Vec3::zeros(), Vec3::new(1.0, 1.0, 1.0)); // 1m cube

    let mut solver = MpmSolver::new(h, dt, bounds);

    // Material: water with ideal gas EOS
    let mat = Material::Fluid {
        viscosity: 1e-3, // 1 mPa·s (water viscosity)
        eos: EquationOfState::IdealGas {
            rho0: 1000.0, // 1000 kg/m³
            cs: 100.0,    // 100 m/s (artificial sound speed for stability)
        },
    };

    // Create water particles in a falling stream
    let mut particles = Vec::new();
    let particle_mass = 0.01; // 10g per particle
    let particle_volume = 0.00001; // 10 mm³

    // Drop location: center of container, high up
    let nx = 10;
    let ny = 20;
    let nz = 10;
    let spacing = 0.015;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = Vec3::new(
                    0.4 + i as f64 * spacing,
                    0.5 + j as f64 * spacing,
                    0.4 + k as f64 * spacing,
                );
                particles.push(Particle::new(
                    x,
                    Vec3::zeros(),
                    particle_mass,
                    particle_volume,
                    mat,
                ));
            }
        }
    }

    println!("Particle count: {}", particles.len());
    println!(
        "Total water mass: {:.2}kg\n",
        particles.len() as f64 * particle_mass
    );

    // Simulate for 1 second
    let n_steps = 1000;
    let print_interval = 100;

    for step in 0..n_steps {
        solver.step(&mut particles);

        if step % print_interval == 0 {
            let avg_y: f64 = particles.iter().map(|p| p.x.y).sum::<f64>() / particles.len() as f64;
            let ke = MpmSolver::kinetic_energy(&particles);

            // Find lowest particle
            let min_y = particles
                .iter()
                .map(|p| p.x.y)
                .fold(f64::INFINITY, f64::min);

            println!(
                "Step {}: avg_y={:.3}m, min_y={:.3}m, KE={:.3}J",
                step, avg_y, min_y, ke
            );
        }
    }

    // Final statistics
    let avg_y: f64 = particles.iter().map(|p| p.x.y).sum::<f64>() / particles.len() as f64;
    let min_y = particles
        .iter()
        .map(|p| p.x.y)
        .fold(f64::INFINITY, f64::min);
    let max_y = particles
        .iter()
        .map(|p| p.x.y)
        .fold(f64::NEG_INFINITY, f64::max);

    println!("\nFinal state:");
    println!("  Average height: {:.3}m", avg_y);
    println!(
        "  Water depth: {:.3}m (from y={:.3} to y={:.3})",
        max_y - min_y,
        min_y,
        max_y
    );
    println!("  Final KE: {:.3}J", MpmSolver::kinetic_energy(&particles));
    println!("\nSimulation complete!");
}
