//! A flexible water molecule: harmonic O-H bonds, an H-O-H angle, and
//! intramolecular electrostatics excluded by the bonded topology.

use phyz_math::Vec3;
use phyz_md::field::units::KB_EV_PER_K;
use phyz_md::{Coulomb, MdSystem, Particle};

fn main() {
    let k_b = KB_EV_PER_K;
    let temperature = 300.0; // K

    let dt = 0.2; // fs — O-H stretches are fast
    let mut system = MdSystem::new(dt);
    // An isolated molecule, so a direct cutoff sum is the right choice; a
    // periodic box of water would need PME instead.
    system.set_cutoff_coulomb(Coulomb { cutoff: 10.0 });

    // Water geometry: O at the origin, two H at ±104.5°/2, 0.96 Å bonds.
    let r_oh = 0.96;
    let half_angle = 104.5_f64.to_radians() / 2.0;

    system.add_particle(Particle::new(Vec3::zeros(), Vec3::zeros(), 15.999, 0).with_charge(-0.8));
    system.add_particle(
        Particle::new(
            Vec3::new(r_oh * half_angle.sin(), r_oh * half_angle.cos(), 0.0),
            Vec3::zeros(),
            1.008,
            1,
        )
        .with_charge(0.4),
    );
    system.add_particle(
        Particle::new(
            Vec3::new(-r_oh * half_angle.sin(), r_oh * half_angle.cos(), 0.0),
            Vec3::zeros(),
            1.008,
            1,
        )
        .with_charge(0.4),
    );

    // Bonds and a real angle term (rather than a fake H-H bond).
    let k_bond = 45.0; // eV/Å²
    system.add_bond(0, 1, k_bond, r_oh);
    system.add_bond(0, 2, k_bond, r_oh);
    system.add_angle(1, 0, 2, 4.5, 104.5_f64.to_radians());

    println!("Water molecule MD simulation");
    println!("O-H bond length: {r_oh:.3} Å");
    println!("H-O-H angle:     104.5°");
    println!(
        "Excluded pairs:  {:?} (1-2 and 1-3 neighbors)",
        system.exclusions
    );

    system.initialize_velocities(temperature, k_b);
    system.remove_com_motion();
    system.compute_forces();

    let e0 = system.total_energy();

    println!("\nInitial positions:");
    for i in 0..system.len() {
        let x = system.position(i);
        println!("  Particle {i}: ({:.3}, {:.3}, {:.3}) Å", x.x, x.y, x.z);
    }

    println!(
        "\n{:>8} {:>10} {:>12} {:>12} {:>10} {:>12}",
        "Step", "Time(ps)", "O-H1(Å)", "O-H2(Å)", "angle(°)", "drift(%)"
    );

    for _ in 0..10000 {
        system.step();

        if system.step.is_multiple_of(1000) {
            let (r1, r2, theta) = geometry(&system);
            let drift = (system.total_energy() - e0).abs() / e0.abs().max(1e-12) * 100.0;
            println!(
                "{:8} {:10.3} {:12.6} {:12.6} {:10.2} {:12.5}",
                system.step,
                system.time / 1000.0,
                r1,
                r2,
                theta.to_degrees(),
                drift
            );
        }
    }

    let (r1, r2, theta) = geometry(&system);
    println!("\nFinal geometry:");
    println!("  O-H1:  {r1:.6} Å (equilibrium {r_oh:.6} Å)");
    println!("  O-H2:  {r2:.6} Å");
    println!("  H-O-H: {:.2}° (equilibrium 104.50°)", theta.to_degrees());
    println!(
        "  Energy drift: {:.5}%",
        (system.total_energy() - e0).abs() / e0.abs().max(1e-12) * 100.0
    );

    // A vibrating bond oscillates around r0; check the excursion is bounded.
    let d1 = (r1 - r_oh).abs() / r_oh * 100.0;
    let d2 = (r2 - r_oh).abs() / r_oh * 100.0;
    if d1 < 10.0 && d2 < 10.0 {
        println!("✓ Bond lengths stayed within 10% of equilibrium");
    } else {
        println!("⚠ Bond excursion: O-H1={d1:.2}%, O-H2={d2:.2}%");
    }
}

fn geometry(system: &MdSystem) -> (f64, f64, f64) {
    let o = system.position(0);
    let h1 = system.position(1);
    let h2 = system.position(2);
    let a = h1 - o;
    let b = h2 - o;
    let cos = (a.dot(b) / (a.norm() * b.norm())).clamp(-1.0, 1.0);
    (a.norm(), b.norm(), cos.acos())
}
