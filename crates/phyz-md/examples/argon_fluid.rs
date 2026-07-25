//! Argon fluid simulation using the Lennard-Jones potential.
//!
//! Equilibrates an FCC argon lattice with a Nosé-Hoover thermostat, then runs
//! NVE production and reports energy conservation, temperature, and pressure.

use phyz_math::Vec3;
use phyz_md::field::units::KB_EV_PER_K;
use phyz_md::{LennardJones, MdSystem, Particle};

fn main() {
    let k_b = KB_EV_PER_K;
    let mass_ar = 39.948; // Argon mass (amu)
    let sigma = 3.4; // LJ σ (Å)
    let temperature = 120.0; // K — argon is a liquid near here

    let dt = 5.0; // fs
    let mut system = MdSystem::lennard_jones(LennardJones::argon(), dt);

    // FCC lattice of argon atoms.
    let n_cells: i32 = 4;
    let lattice_constant = 1.6 * sigma;

    println!("Creating {} argon atoms in FCC lattice", n_cells.pow(3) * 4);

    for ix in 0..n_cells {
        for iy in 0..n_cells {
            for iz in 0..n_cells {
                let origin = Vec3::new(ix as f64, iy as f64, iz as f64) * lattice_constant;
                let basis = [
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.5, 0.5, 0.0) * lattice_constant,
                    Vec3::new(0.5, 0.0, 0.5) * lattice_constant,
                    Vec3::new(0.0, 0.5, 0.5) * lattice_constant,
                ];
                for pos in basis {
                    system.add_particle(Particle::new(origin + pos, Vec3::zeros(), mass_ar, 0));
                }
            }
        }
    }

    let l = n_cells as f64 * lattice_constant;
    system.set_box_size(Vec3::new(l, l, l));
    println!("Box size: {l:.2} x {l:.2} x {l:.2} Å");

    system.initialize_velocities(temperature, k_b);
    system.compute_forces();

    println!("Initial temperature: {:.1} K", system.temperature(k_b));
    println!("Initial energy: {:.6} eV\n", system.total_energy());
    println!(
        "Neighbor list: {} pairs, cell lists {}",
        system.neighbor_list.pairs().len(),
        if system.neighbor_list.used_fallback() {
            "inactive (box too small for binning)"
        } else {
            "active"
        }
    );

    // Equilibrate with a Nosé-Hoover thermostat.
    system.set_nose_hoover(temperature, 500.0);

    println!("\nRunning equilibration (Nosé-Hoover)...");
    println!(
        "{:>8} {:>10} {:>12} {:>12} {:>12}",
        "Step", "Time(ps)", "T(K)", "KE(eV)", "PE(eV)"
    );

    for _ in 0..4000 {
        system.step();
        if system.step.is_multiple_of(500) {
            println!(
                "{:8} {:10.3} {:12.2} {:12.6} {:12.6}",
                system.step,
                system.time / 1000.0,
                system.temperature(k_b),
                system.kinetic_energy(),
                system.potential_energy,
            );
        }
    }

    println!("\nTurning off the thermostat for NVE production...");
    system.nose_hoover = None;

    let e_initial = system.total_energy();

    for _ in 0..4000 {
        system.step();

        if system.step.is_multiple_of(500) {
            let e_total = system.total_energy();
            let drift = (e_total - e_initial).abs() / e_initial.abs() * 100.0;
            println!(
                "{:8} {:10.3} {:12.2} {:12.6} {:12.6} ({:.4}% drift)",
                system.step,
                system.time / 1000.0,
                system.temperature(k_b),
                system.kinetic_energy(),
                system.potential_energy,
                drift
            );
        }
    }

    println!("\nFinal statistics:");
    let energy_drift = (system.total_energy() - e_initial).abs() / e_initial.abs() * 100.0;
    println!("Energy conservation: {energy_drift:.4}% drift");
    println!("Final temperature:   {:.1} K", system.temperature(k_b));
    println!("Pressure:            {:.4} GPa", system.pressure_gpa());
    println!(
        "Neighbor rebuilds:   {} over {} steps",
        system.neighbor_list.builds(),
        system.step
    );

    if energy_drift < 1.0 {
        println!("✓ Energy conservation test passed!");
    } else {
        println!("⚠ Energy drift exceeds 1%");
    }
}
