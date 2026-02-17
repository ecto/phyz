//! Ball drop example — demonstrates contact physics with ground plane.
//! Simplified particle simulation (no full rigid body dynamics).

use phyz::{
    ContactMaterial,
    phyz_math::{GRAVITY, Vec3},
};

fn main() {
    let radius = 0.1; // 10 cm ball
    let mass = 1.0; // 1 kg
    let dt = 0.001;

    // Simple particle simulation
    let mut pos = Vec3::new(0.0, 0.0, 2.0); // Start at 2m height
    let mut vel = Vec3::zeros();

    let material = ContactMaterial::bouncy();

    println!("Ball drop simulation");
    println!("Initial height: {:.3} m", pos.z);
    println!("Radius: {:.3} m", radius);
    println!("Mass: {:.3} kg", mass);
    println!("Bounce coefficient: {:.2}\n", material.bounce);

    let ground_height = 0.0;
    let mut max_bounce_height = 0.0;
    let mut t = 0.0;

    for i in 0..5000 {
        let z = pos.z;

        // Track maximum height after first bounce
        if t > 0.5 && z > max_bounce_height {
            max_bounce_height = z;
        }

        // Check ground contact
        let min_z = z - radius;
        let mut in_contact = false;

        if min_z < ground_height {
            in_contact = true;
            let penetration = ground_height - min_z;

            // Apply penalty force
            let normal_vel = vel.z;
            let force_mag = material.stiffness * penetration - material.damping * normal_vel;
            let accel = force_mag / mass;
            vel.z += accel * dt;

            // Bounce if hitting ground
            if normal_vel < -0.1 && penetration > 0.001 {
                vel.z = -normal_vel * material.bounce;
            }
        }

        // Apply gravity
        vel.z -= GRAVITY * dt;

        // Integrate
        pos += vel * dt;
        t += dt;

        // Print every 0.2 seconds
        if i % 200 == 0 {
            println!(
                "t={:.3}s  z={:.3}m  vz={:.3}m/s  contact={}",
                t, z, vel.z, in_contact
            );
        }

        // Stop after 5 seconds
        if t > 5.0 {
            break;
        }
    }

    println!("\nMaximum bounce height: {:.3} m", max_bounce_height);
    println!(
        "Energy retention: {:.1}%",
        (max_bounce_height / 2.0) * 100.0
    );
}
