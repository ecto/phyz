//! Granular column collapse simulation.
//!
//! Creates a tall column of granular particles that collapses under gravity.
//! Measures runout distance and demonstrates Drucker-Prager yield behavior.

use phyz::{Material, MpmSolver, Particle, Vec3};

fn main() {
    println!("Granular column collapse simulation");
    println!("===================================\n");

    // Setup: 10k particles in a tall column
    let h = 0.05; // 5cm grid spacing
    let dt = 0.001; // 1ms timestep
    let bounds = (Vec3::zeros(), Vec3::new(2.0, 3.0, 2.0));

    let mut solver = MpmSolver::new(h, dt, bounds);

    // Material: granular with 30° friction angle
    let mat = Material::Granular {
        phi: 30.0_f64.to_radians(),
    };

    // Create column: 0.2m x 1.0m x 0.2m (width x height x depth)
    let mut particles = Vec::new();
    let particle_mass = 0.01; // 10g per particle
    let particle_volume = 0.001; // 1cm³

    let nx = 20;
    let ny = 100;
    let nz = 20;
    let spacing = 0.01;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = Vec3::new(
                    0.9 + i as f64 * spacing,
                    0.1 + j as f64 * spacing,
                    0.9 + k as f64 * spacing,
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

    println!("Initial particle count: {}", particles.len());
    println!("Column height: {:.2}m", ny as f64 * spacing);
    println!(
        "Total mass: {:.2}kg\n",
        particles.len() as f64 * particle_mass
    );

    // Initial center of mass
    let initial_com: Vec3 = particles.iter().map(|p| p.x * p.mass).sum::<Vec3>()
        / particles.iter().map(|p| p.mass).sum::<f64>();

    println!(
        "Initial COM: ({:.3}, {:.3}, {:.3})",
        initial_com.x, initial_com.y, initial_com.z
    );

    // Simulate for 2 seconds (2000 steps)
    let n_steps = 2000;
    let print_interval = 200;

    for step in 0..n_steps {
        solver.step(&mut particles);

        if step % print_interval == 0 {
            let com: Vec3 = particles.iter().map(|p| p.x * p.mass).sum::<Vec3>()
                / particles.iter().map(|p| p.mass).sum::<f64>();
            let ke = MpmSolver::kinetic_energy(&particles);
            let momentum = MpmSolver::total_momentum(&particles);

            println!(
                "Step {}: COM=({:.3}, {:.3}, {:.3}), KE={:.3}J, p=({:.3}, {:.3}, {:.3})",
                step, com.x, com.y, com.z, ke, momentum.x, momentum.y, momentum.z
            );
        }
    }

    // Final measurements
    let final_com: Vec3 = particles.iter().map(|p| p.x * p.mass).sum::<Vec3>()
        / particles.iter().map(|p| p.mass).sum::<f64>();

    println!(
        "\nFinal COM: ({:.3}, {:.3}, {:.3})",
        final_com.x, final_com.y, final_com.z
    );

    // Runout distance (horizontal spread)
    let max_x = particles
        .iter()
        .map(|p| p.x.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_x = particles
        .iter()
        .map(|p| p.x.x)
        .fold(f64::INFINITY, f64::min);
    let runout = max_x - min_x;

    println!("Runout distance: {:.3}m", runout);
    println!("Height drop: {:.3}m", initial_com.y - final_com.y);
    println!("\nSimulation complete!");
}
