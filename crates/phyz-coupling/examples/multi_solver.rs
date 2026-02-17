//! Multi-solver coupling example.
//!
//! Demonstrates how to couple different physics solvers using handshake regions
//! and force transfer mechanisms.

use phyz_coupling::{BoundingBox, Coupling, ForceTransfer, SolverType};
use phyz_math::Vec3;

/// Mock rigid body state for demonstration.
#[derive(Clone, Debug)]
struct RigidBody {
    position: Vec3,
    velocity: Vec3,
    mass: f64,
}

/// Mock particle state for demonstration.
#[derive(Clone, Debug)]
struct Particle {
    position: Vec3,
    velocity: Vec3,
    mass: f64,
}

/// Mock EM field point for demonstration.
#[allow(dead_code)]
#[derive(Clone, Debug)]
struct EmField {
    e_field: Vec3,
    b_field: Vec3,
}

fn main() {
    println!("Multi-Solver Coupling Example");
    println!("==============================\n");

    // Define coupling regions
    let coupling_region = BoundingBox::new(Vec3::new(-2.0, -2.0, -2.0), Vec3::new(2.0, 2.0, 2.0));

    // Create coupling between rigid body and particles
    let coupling = Coupling::new(
        SolverType::RigidBody,
        SolverType::Particle,
        coupling_region.clone(),
        ForceTransfer::Direct { damping: 1.0 },
    );

    println!("Coupling configuration:");
    println!("  Solver A: {:?}", coupling.solver_a);
    println!("  Solver B: {:?}", coupling.solver_b);
    println!(
        "  Overlap region: [{:.1}, {:.1}, {:.1}] to [{:.1}, {:.1}, {:.1}]",
        coupling.overlap_region.min.x,
        coupling.overlap_region.min.y,
        coupling.overlap_region.min.z,
        coupling.overlap_region.max.x,
        coupling.overlap_region.max.y,
        coupling.overlap_region.max.z,
    );
    println!("  Force transfer: Direct damping\n");

    // Create mock objects
    let mut rigid = RigidBody {
        position: Vec3::new(0.0, 0.0, 0.0),
        velocity: Vec3::new(1.0, 0.0, 0.0),
        mass: 10.0,
    };

    let mut particle = Particle {
        position: Vec3::new(1.0, 0.0, 0.0),
        velocity: Vec3::new(-0.5, 0.0, 0.0),
        mass: 1.0,
    };

    // Simulation parameters
    let dt = 0.01;
    let n_steps = 100;

    println!("Initial state:");
    println!(
        "  Rigid body: pos={:.2?}, vel={:.2?}",
        rigid.position, rigid.velocity
    );
    println!(
        "  Particle: pos={:.2?}, vel={:.2?}",
        particle.position, particle.velocity
    );
    println!();

    println!("Time (s)  RigidPos  RigidVel  PartPos  PartVel  Coupling");
    println!("--------------------------------------------------------------");

    for step in 0..n_steps {
        let time = step as f64 * dt;

        // Check if coupling force should be applied
        let force_opt = coupling.compute_coupling_force(
            &rigid.position,
            &rigid.velocity,
            &particle.position,
            &particle.velocity,
        );

        let in_coupling = force_opt.is_some();
        let force = force_opt.unwrap_or(Vec3::zeros());

        // Apply coupling force (Newton's 3rd law)
        let accel_rigid = force / rigid.mass;
        let accel_particle = -force / particle.mass;

        // Simple Euler integration
        rigid.velocity += accel_rigid * dt;
        rigid.position += rigid.velocity * dt;

        particle.velocity += accel_particle * dt;
        particle.position += particle.velocity * dt;

        // Print every 10 steps
        if step % 10 == 0 {
            println!(
                "{:8.2}  {:8.2}  {:8.3}  {:7.2}  {:7.3}  {}",
                time,
                rigid.position.x,
                rigid.velocity.x,
                particle.position.x,
                particle.velocity.x,
                if in_coupling { "Active" } else { "Inactive" }
            );
        }
    }

    println!("\nFinal state:");
    println!(
        "  Rigid body: pos={:.2?}, vel={:.2?}",
        rigid.position, rigid.velocity
    );
    println!(
        "  Particle: pos={:.2?}, vel={:.2?}",
        particle.position, particle.velocity
    );

    // Compute momentum
    let initial_momentum = 10.0 * 1.0 + 1.0 * (-0.5);
    let final_momentum = rigid.mass * rigid.velocity.x + particle.mass * particle.velocity.x;

    println!("\nMomentum conservation:");
    println!("  Initial: {:.3} kg·m/s", initial_momentum);
    println!("  Final: {:.3} kg·m/s", final_momentum);
    println!("  Error: {:.2e}", (final_momentum - initial_momentum).abs());

    println!("\nThis demonstrates handshake region coupling with direct damping.");
    println!("Momentum is conserved as forces obey Newton's 3rd law.");
}
