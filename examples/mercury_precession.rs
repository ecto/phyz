//! Mercury perihelion precession demonstration.
//!
//! Simulates Mercury's orbit around the Sun with 1PN post-Newtonian corrections.
//! Demonstrates the famous 43 arcsec/century perihelion precession predicted by GR.

use tau_gravity::{GravitySolver, GravityParticle, PostNewtonianSolver, orbital_elements, perihelion_precession_rate};
use tau_math::Vec3;

fn main() {
    println!("=== Mercury Perihelion Precession (1PN) ===\n");

    // Physical constants
    let m_sun = 1.989e30; // kg
    let m_mercury = 3.285e23; // kg
    let au = 1.496e11; // m

    // Mercury orbital parameters (approximate)
    let a = 0.387 * au; // semi-major axis
    let e = 0.2056; // eccentricity
    let period = 87.969 * 24.0 * 3600.0; // orbital period (seconds)

    // Initial conditions: perihelion (closest approach)
    let r_perihelion = a * (1.0 - e);
    let v_perihelion = ((tau_gravity::G * m_sun / a) * ((1.0 + e) / (1.0 - e))).sqrt();

    println!("Initial orbital elements:");
    println!("  Semi-major axis: {:.3e} m ({:.3} AU)", a, a / au);
    println!("  Eccentricity: {:.4}", e);
    println!("  Period: {:.2} days", period / 86400.0);
    println!("  Perihelion distance: {:.3e} m ({:.3} AU)", r_perihelion, r_perihelion / au);
    println!("  Perihelion velocity: {:.3} km/s\n", v_perihelion / 1e3);

    // Predicted perihelion precession (GR)
    let predicted_precession = perihelion_precession_rate(a, e, m_sun, period);
    println!("Predicted precession (1PN): {:.2} arcsec/century", predicted_precession);
    println!("Observed (historical): 43.11 ± 0.45 arcsec/century\n");

    // Create particles
    let mut particles = vec![
        GravityParticle::new(Vec3::zeros(), Vec3::zeros(), m_sun),
        GravityParticle::new(
            Vec3::new(r_perihelion, 0.0, 0.0),
            Vec3::new(0.0, v_perihelion, 0.0),
            m_mercury,
        ),
    ];

    // Create 1PN solver
    let mut solver = PostNewtonianSolver::new(1.0);
    solver.softening = 0.0; // No softening for two-body

    // Simulation parameters
    let dt = 3600.0; // 1 hour timestep
    let n_orbits = 10;
    let steps_per_orbit = (period / dt) as usize;
    let total_steps = n_orbits * steps_per_orbit;

    println!("Simulation:");
    println!("  Timestep: {:.0} s ({:.1} min)", dt, dt / 60.0);
    println!("  Steps per orbit: {}", steps_per_orbit);
    println!("  Total orbits: {}\n", n_orbits);

    // Track perihelion longitude
    let mut perihelion_longitudes = Vec::new();
    let mut last_r = f64::INFINITY;
    let mut found_perihelion = false;

    // Simulation loop
    for step in 0..total_steps {
        // Velocity Verlet integration
        solver.compute_forces(&mut particles);

        for p in &mut particles {
            p.velocity_verlet_step(dt);
        }

        solver.compute_forces(&mut particles);

        for p in &mut particles {
            p.velocity_verlet_complete(dt);
        }

        // Check for perihelion passage (minimum distance)
        let r_mercury = (particles[1].x - particles[0].x).norm();

        if r_mercury < last_r {
            found_perihelion = true;
        } else if found_perihelion {
            // Just passed perihelion
            found_perihelion = false;

            // Compute orbital elements
            let (a_cur, e_cur, _i, _omega, omega_bar, _nu) = orbital_elements(
                particles[1].x - particles[0].x,
                particles[1].v - particles[0].v,
                m_sun,
            );

            perihelion_longitudes.push(omega_bar);

            let orbit_num = perihelion_longitudes.len();
            println!(
                "Orbit {}: a={:.4} AU, e={:.4}, ω={:.6} rad ({:.2}°)",
                orbit_num,
                a_cur / au,
                e_cur,
                omega_bar,
                omega_bar.to_degrees()
            );
        }

        last_r = r_mercury;
    }

    // Compute precession from first and last perihelion
    if perihelion_longitudes.len() >= 2 {
        let delta_omega = perihelion_longitudes.last().unwrap()
            - perihelion_longitudes.first().unwrap();
        let n_completed = (perihelion_longitudes.len() - 1) as f64;

        let precession_per_orbit = delta_omega / n_completed;

        // Convert to arcsec/century
        let orbits_per_century = 3.15576e9 / period;
        let precession_arcsec_century = precession_per_orbit * orbits_per_century * 206265.0;

        println!("\nSimulated precession:");
        println!("  Δω over {} orbits: {:.6} rad ({:.4}°)", n_completed, delta_omega, delta_omega.to_degrees());
        println!("  Per orbit: {:.6} rad", precession_per_orbit);
        println!("  Extrapolated: {:.2} arcsec/century", precession_arcsec_century);
        println!("\nComparison:");
        println!("  Predicted: {:.2} arcsec/century", predicted_precession);
        println!("  Simulated: {:.2} arcsec/century", precession_arcsec_century);
        println!("  Difference: {:.2}%", (precession_arcsec_century - predicted_precession).abs() / predicted_precession * 100.0);
    }
}
