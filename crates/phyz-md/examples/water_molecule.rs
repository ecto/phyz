//! Simple water molecule model with harmonic bonds.

use std::sync::Arc;
use phyz_math::Vec3;
use phyz_md::{Coulomb, MdSystem, Particle};

fn main() {
    // Constants
    let k_b = 8.617e-5; // eV/K
    let temperature = 300.0; // K

    // Coulomb force field
    // Charges: O = -0.8e, H = +0.4e
    let charges = vec![-0.8, 0.4];
    let coulomb = Arc::new(Coulomb::new(14.4, charges, 10.0)); // k in eV·Å/e²

    let dt = 0.2; // fs
    let mut system = MdSystem::new(coulomb, dt);

    // Water molecule geometry (simplified)
    // O at origin, two H atoms at ±104.5°/2 angle, 0.96 Å bond length
    let r_oh = 0.96; // Å
    let angle = 104.5_f64.to_radians() / 2.0;

    // Oxygen (atom type 0)
    system.add_particle(Particle::new(Vec3::zeros(), Vec3::zeros(), 15.999, 0));

    // Hydrogen 1 (atom type 1)
    system.add_particle(Particle::new(
        Vec3::new(r_oh * angle.sin(), r_oh * angle.cos(), 0.0),
        Vec3::zeros(),
        1.008,
        1,
    ));

    // Hydrogen 2 (atom type 1)
    system.add_particle(Particle::new(
        Vec3::new(-r_oh * angle.sin(), r_oh * angle.cos(), 0.0),
        Vec3::zeros(),
        1.008,
        1,
    ));

    // Add O-H bonds with harmonic potential
    let k_bond = 100.0; // eV/Å²
    system.add_bond(0, 1, k_bond, r_oh);
    system.add_bond(0, 2, k_bond, r_oh);

    // Add H-H angle constraint (optional, as harmonic bond)
    let r_hh = 2.0 * r_oh * angle.sin();
    system.add_bond(1, 2, 50.0, r_hh);

    println!("Water molecule MD simulation");
    println!("O-H bond length: {:.3} Å", r_oh);
    println!("H-H distance: {:.3} Å", r_hh);

    // Initialize velocities
    system.initialize_velocities(temperature, k_b);

    println!("\nInitial positions:");
    for (i, p) in system.particles.iter().enumerate() {
        println!(
            "  Particle {}: ({:.3}, {:.3}, {:.3}) Å",
            i, p.x.x, p.x.y, p.x.z
        );
    }

    println!(
        "\n{:>8} {:>10} {:>12} {:>12}",
        "Step", "Time(ps)", "O-H1(Å)", "O-H2(Å)"
    );

    // Run simulation
    for _ in 0..10000 {
        system.step();

        if system.step % 1000 == 0 {
            let r_oh1 = (system.particles[1].x - system.particles[0].x).norm();
            let r_oh2 = (system.particles[2].x - system.particles[0].x).norm();

            println!(
                "{:8} {:10.3} {:12.6} {:12.6}",
                system.step,
                system.time / 1000.0,
                r_oh1,
                r_oh2
            );
        }
    }

    println!("\nFinal positions:");
    for (i, p) in system.particles.iter().enumerate() {
        println!(
            "  Particle {}: ({:.3}, {:.3}, {:.3}) Å",
            i, p.x.x, p.x.y, p.x.z
        );
    }

    // Check bond lengths
    let r_oh1 = (system.particles[1].x - system.particles[0].x).norm();
    let r_oh2 = (system.particles[2].x - system.particles[0].x).norm();

    println!("\nFinal bond lengths:");
    println!("  O-H1: {:.6} Å (should be ~{:.6} Å)", r_oh1, r_oh);
    println!("  O-H2: {:.6} Å (should be ~{:.6} Å)", r_oh2, r_oh);

    let drift1 = (r_oh1 - r_oh).abs() / r_oh * 100.0;
    let drift2 = (r_oh2 - r_oh).abs() / r_oh * 100.0;

    if drift1 < 1.0 && drift2 < 1.0 {
        println!("✓ Bond lengths maintained within 1%");
    } else {
        println!(
            "⚠ Bond length drift: O-H1={:.2}%, O-H2={:.2}%",
            drift1, drift2
        );
    }
}
