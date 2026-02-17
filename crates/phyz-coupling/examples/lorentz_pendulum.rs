//! Lorentz-forced pendulum example.
//!
//! Demonstrates EM→rigid body coupling where an oscillating electromagnetic
//! field exerts Lorentz forces on a charged pendulum bob.

use phyz_coupling::lorentz_force;
use phyz_math::Vec3;

fn main() {
    println!("Lorentz-Forced Pendulum Simulation");
    println!("===================================\n");

    // Physical parameters
    let charge = 1e-6; // 1 μC charge on pendulum bob
    let mass = 0.1; // 100g bob
    let length = 1.0; // 1m pendulum

    // Initial conditions
    let mut theta: f64 = 0.1; // 0.1 rad initial angle
    let mut omega: f64 = 0.0; // initially at rest

    // Simulation parameters
    let dt = 0.001; // 1 ms timestep
    let n_steps = 10000;
    let em_freq = 2.0; // EM field oscillates at 2 Hz

    // Time-varying electromagnetic field
    let b_amplitude = 0.1; // 0.1 T magnetic field
    let e_amplitude = 1000.0; // 1 kV/m electric field

    println!("Parameters:");
    println!("  Charge: {:.1e} C", charge);
    println!("  Mass: {:.3} kg", mass);
    println!("  Length: {:.1} m", length);
    println!("  EM frequency: {:.1} Hz", em_freq);
    println!("  B field amplitude: {:.2} T", b_amplitude);
    println!("  E field amplitude: {:.1} V/m\n", e_amplitude);

    println!("Time (s)  Angle (deg)  Velocity (rad/s)  Energy (J)");
    println!("-------------------------------------------------------");

    for step in 0..n_steps {
        let time = step as f64 * dt;

        // Pendulum position and velocity
        let x = length * theta.sin();
        let y = -length * theta.cos();
        let position = Vec3::new(x, y, 0.0);

        let vx = length * omega * theta.cos();
        let vy = length * omega * theta.sin();
        let velocity = Vec3::new(vx, vy, 0.0);

        // Time-varying EM fields
        let phase = 2.0 * std::f64::consts::PI * em_freq * time;
        let e_field = Vec3::new(e_amplitude * phase.sin(), 0.0, 0.0);
        let b_field = Vec3::new(0.0, 0.0, b_amplitude * phase.cos());

        // Lorentz force
        let f_lorentz = lorentz_force(charge, position, velocity, &e_field, &b_field);

        // Gravitational force
        let g = 9.81;
        let f_gravity = Vec3::new(-mass * g * theta.sin(), 0.0, 0.0);

        // Total tangential force
        let f_tangent = f_lorentz.x + f_gravity.x;

        // Angular acceleration: α = F_tangent / (m L)
        let alpha = f_tangent / (mass * length);

        // Semi-implicit Euler integration
        omega += alpha * dt;
        theta += omega * dt;

        // Compute energy
        let ke = 0.5 * mass * length * length * omega * omega;
        let pe = mass * g * length * (1.0 - theta.cos());
        let energy = ke + pe;

        // Print every 100 steps
        if step % 1000 == 0 {
            println!(
                "{:8.3}  {:11.2}  {:16.3}  {:10.6}",
                time,
                theta.to_degrees(),
                omega,
                energy
            );
        }
    }

    println!("\nExample demonstrating EM→rigid body coupling via Lorentz force.");
}
