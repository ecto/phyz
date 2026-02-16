//! Rigid sphere falling into granular material.
//!
//! Demonstrates basic rigid-particle coupling concept.
//! (Note: Full coupling requires collision detection between rigid bodies and particles,
//! which will be implemented in a future phase.)

use tau::{Material, MpmSolver, Particle, Vec3};

fn main() {
    println!("Sphere falling into sand simulation");
    println!("===================================\n");

    // Setup
    let h = 0.03; // 3cm grid spacing
    let dt = 0.001; // 1ms timestep
    let bounds = (Vec3::zeros(), Vec3::new(2.0, 2.0, 2.0));

    let mut solver = MpmSolver::new(h, dt, bounds);

    // Material: granular sand
    let mat = Material::Granular {
        phi: 35.0_f64.to_radians(), // 35° friction angle (typical sand)
    };

    // Create sand pile: flat layer at bottom
    let mut particles = Vec::new();
    let particle_mass = 0.01; // 10g per particle
    let particle_volume = 0.001; // 1cm³

    let nx = 50;
    let ny = 20; // 20cm tall pile
    let nz = 50;
    let spacing = 0.02;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = Vec3::new(
                    0.2 + i as f64 * spacing,
                    0.05 + j as f64 * spacing,
                    0.2 + k as f64 * spacing,
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

    println!("Sand particle count: {}", particles.len());
    println!(
        "Sand pile dimensions: {:.2}m x {:.2}m x {:.2}m\n",
        nx as f64 * spacing,
        ny as f64 * spacing,
        nz as f64 * spacing
    );

    // Add "rigid sphere" as a dense cluster of particles
    // (This simulates a rigid body falling into the sand)
    let sphere_mat = Material::Elastic { e: 1e9, nu: 0.3 }; // Very stiff
    let sphere_radius = 0.1; // 10cm radius
    let sphere_center = Vec3::new(1.0, 1.0, 1.0); // Start high above sand
    let sphere_particle_mass = 0.1; // Heavier particles
    let sphere_particle_volume = 0.0001;

    // Create sphere as dense particle cluster
    let sphere_resolution = 10;
    let sphere_spacing = sphere_radius * 2.0 / sphere_resolution as f64;

    for i in 0..sphere_resolution {
        for j in 0..sphere_resolution {
            for k in 0..sphere_resolution {
                let offset = Vec3::new(
                    (i as f64 - sphere_resolution as f64 / 2.0) * sphere_spacing,
                    (j as f64 - sphere_resolution as f64 / 2.0) * sphere_spacing,
                    (k as f64 - sphere_resolution as f64 / 2.0) * sphere_spacing,
                );

                // Only add if inside sphere
                if offset.norm() <= sphere_radius {
                    let x = sphere_center + offset;
                    particles.push(Particle::new(
                        x,
                        Vec3::zeros(),
                        sphere_particle_mass,
                        sphere_particle_volume,
                        sphere_mat,
                    ));
                }
            }
        }
    }

    let sphere_particle_count = particles.len() - (nx * ny * nz);
    println!("Sphere particle count: {}", sphere_particle_count);
    println!("Total particles: {}\n", particles.len());

    // Track sphere center of mass
    let sphere_start_idx = nx * ny * nz;

    // Simulate
    let n_steps = 2000;
    let print_interval = 200;

    for step in 0..n_steps {
        solver.step(&mut particles);

        if step % print_interval == 0 {
            // Compute sphere COM
            let sphere_particles = &particles[sphere_start_idx..];
            if !sphere_particles.is_empty() {
                let sphere_com: Vec3 = sphere_particles.iter().map(|p| p.x * p.mass).sum::<Vec3>()
                    / sphere_particles.iter().map(|p| p.mass).sum::<f64>();

                let sphere_velocity: Vec3 =
                    sphere_particles.iter().map(|p| p.v * p.mass).sum::<Vec3>()
                        / sphere_particles.iter().map(|p| p.mass).sum::<f64>();

                println!(
                    "Step {}: sphere_COM=({:.3}, {:.3}, {:.3}), v=({:.2}, {:.2}, {:.2})",
                    step,
                    sphere_com.x,
                    sphere_com.y,
                    sphere_com.z,
                    sphere_velocity.x,
                    sphere_velocity.y,
                    sphere_velocity.z
                );
            }
        }
    }

    // Final measurements
    let sphere_particles = &particles[sphere_start_idx..];
    if !sphere_particles.is_empty() {
        let final_com: Vec3 = sphere_particles.iter().map(|p| p.x * p.mass).sum::<Vec3>()
            / sphere_particles.iter().map(|p| p.mass).sum::<f64>();

        println!(
            "\nFinal sphere COM: ({:.3}, {:.3}, {:.3})",
            final_com.x, final_com.y, final_com.z
        );
        println!("Height drop: {:.3}m", sphere_center.y - final_com.y);
        println!(
            "Penetration depth: {:.3}m",
            final_com.y - (ny as f64 * spacing)
        );
    }

    println!("\nSimulation complete!");
}
