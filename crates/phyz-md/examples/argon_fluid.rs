//! Argon fluid simulation using Lennard-Jones potential.

use phyz_math::Vec3;
use phyz_md::{LennardJones, MdSystem, Particle};
use std::sync::Arc;

fn main() {
    // Constants
    let k_b = 8.617e-5; // Boltzmann constant (eV/K)
    let mass_ar = 39.948; // Argon mass (amu)
    let sigma = 3.4; // LJ σ (Å)
    let temperature = 300.0; // K

    // Create Lennard-Jones force field for argon
    let lj = Arc::new(LennardJones::argon());

    // Timestep (fs)
    let dt = 0.5;
    let mut system = MdSystem::new(lj, dt);

    // Create FCC lattice of argon atoms
    let n_cells: i32 = 4;
    let lattice_constant = 1.5 * sigma; // Slightly larger than σ

    println!("Creating {} argon atoms in FCC lattice", n_cells.pow(3) * 4);

    for ix in 0..n_cells {
        for iy in 0..n_cells {
            for iz in 0..n_cells {
                let x0 = ix as f64 * lattice_constant;
                let y0 = iy as f64 * lattice_constant;
                let z0 = iz as f64 * lattice_constant;

                // FCC basis positions
                let positions = [
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.5, 0.5, 0.0) * lattice_constant,
                    Vec3::new(0.5, 0.0, 0.5) * lattice_constant,
                    Vec3::new(0.0, 0.5, 0.5) * lattice_constant,
                ];

                for pos in &positions {
                    system.add_particle(Particle::new(
                        Vec3::new(x0, y0, z0) + pos,
                        Vec3::zeros(),
                        mass_ar,
                        0,
                    ));
                }
            }
        }
    }

    // Set periodic boundary conditions
    let box_size = Vec3::new(
        n_cells as f64 * lattice_constant,
        n_cells as f64 * lattice_constant,
        n_cells as f64 * lattice_constant,
    );
    system.set_box_size(box_size);

    println!(
        "Box size: {:.2} x {:.2} x {:.2} Å",
        box_size.x, box_size.y, box_size.z
    );

    // Initialize velocities from Maxwell-Boltzmann distribution
    system.initialize_velocities(temperature, k_b);

    println!("Initial temperature: {:.1} K", system.temperature(k_b));
    println!(
        "Initial energy: {:.6} eV\n",
        system.kinetic_energy() + system.potential_energy()
    );

    // Optionally add thermostat for equilibration
    system.set_thermostat(temperature, 0.01, k_b); // γ = 0.01 1/fs

    println!("Running MD simulation (with thermostat)...");
    println!(
        "{:>8} {:>10} {:>12} {:>12} {:>12}",
        "Step", "Time(ps)", "T(K)", "KE(eV)", "PE(eV)"
    );

    // Run equilibration with thermostat
    for _ in 0..5000 {
        system.step();

        if system.step % 500 == 0 {
            let ke = system.kinetic_energy();
            let pe = system.potential_energy();
            let temp = system.temperature(k_b);
            println!(
                "{:8} {:10.3} {:12.2} {:12.6} {:12.6}",
                system.step,
                system.time / 1000.0,
                temp,
                ke,
                pe
            );
        }
    }

    println!("\nTurning off thermostat for production run...");

    // Turn off thermostat
    system.thermostat = None;

    let e_initial = system.kinetic_energy() + system.potential_energy();

    // Production run
    for _ in 0..5000 {
        system.step();

        if system.step % 500 == 0 {
            let ke = system.kinetic_energy();
            let pe = system.potential_energy();
            let temp = system.temperature(k_b);
            let e_total = ke + pe;
            let drift = (e_total - e_initial).abs() / e_initial.abs() * 100.0;

            println!(
                "{:8} {:10.3} {:12.2} {:12.6} {:12.6} ({:.3}% drift)",
                system.step,
                system.time / 1000.0,
                temp,
                ke,
                pe,
                drift
            );
        }
    }

    println!("\nFinal statistics:");
    let e_final = system.kinetic_energy() + system.potential_energy();
    let energy_drift = (e_final - e_initial).abs() / e_initial.abs() * 100.0;
    println!("Energy conservation: {:.3}% drift", energy_drift);
    println!("Final temperature: {:.1} K", system.temperature(k_b));

    if energy_drift < 1.0 {
        println!("✓ Energy conservation test passed!");
    } else {
        println!("⚠ Energy drift exceeds 1%");
    }
}
