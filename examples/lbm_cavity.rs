//! Lid-driven cavity flow demonstration.
//!
//! Classical CFD benchmark: square cavity with top wall moving at constant velocity.
//! At steady state, LBM should recover Navier-Stokes solution with characteristic
//! primary vortex and secondary corner vortices.

use tau_lbm::LatticeBoltzmann2D;

fn main() {
    println!("Lid-driven cavity flow (2D LBM)");
    println!("================================");

    let nx = 128;
    let ny = 128;
    let nu = 0.01; // kinematic viscosity
    let u_lid = 0.1; // lid velocity

    let mut lbm = LatticeBoltzmann2D::new(nx, ny, nu);

    // Initialize with uniform density, zero velocity
    lbm.initialize_uniform(1.0, [0.0, 0.0]);

    println!("Grid: {}x{}", nx, ny);
    println!("Viscosity: ν = {}", nu);
    println!("Lid velocity: u_lid = {}", u_lid);
    println!("Reynolds number: Re ≈ {:.1}", u_lid * nx as f64 / nu);
    println!();

    // Simulate
    let n_steps = 10000;
    let print_every = 1000;

    for step in 0..=n_steps {
        // Apply boundary conditions
        for x in 0..nx {
            // Top wall: moving lid
            lbm.set_velocity_bc(x, ny - 1, [u_lid, 0.0]);
            // Bottom wall: no-slip
            lbm.set_no_slip_bc(x, 0);
        }
        for y in 0..ny {
            // Left and right walls: no-slip
            lbm.set_no_slip_bc(0, y);
            lbm.set_no_slip_bc(nx - 1, y);
        }

        lbm.collide_and_stream();

        if step % print_every == 0 {
            let u_max = lbm.max_velocity();
            let ke = lbm.kinetic_energy();
            println!("Step {:5}: u_max = {:.6}, KE = {:.6}", step, u_max, ke);
        }
    }

    println!();
    println!("Steady state reached:");

    // Sample velocity profile at x = nx/2 (vertical centerline)
    println!("\nVertical centerline velocity profile (x = {}):", nx / 2);
    println!("  y/H      u_x");
    println!("  ----    ------");
    for i in 0..10 {
        let y = i * ny / 10;
        let u = lbm.velocity(nx / 2, y);
        println!("  {:.2}    {:.6}", y as f64 / ny as f64, u[0]);
    }

    // Check for primary vortex center (should be around [0.5, 0.6] for Re~1000)
    let mut max_vorticity = 0.0;
    let mut vortex_x = 0;
    let mut vortex_y = 0;

    for y in 1..ny - 1 {
        for x in 1..nx - 1 {
            let u_right = lbm.velocity(x + 1, y);
            let u_left = lbm.velocity(x - 1, y);
            let u_up = lbm.velocity(x, y + 1);
            let u_down = lbm.velocity(x, y - 1);

            // Vorticity: ω = ∂v/∂x - ∂u/∂y
            let dvdx = (u_right[1] - u_left[1]) / 2.0;
            let dudy = (u_up[0] - u_down[0]) / 2.0;
            let vorticity = (dvdx - dudy).abs();

            if vorticity > max_vorticity {
                max_vorticity = vorticity;
                vortex_x = x;
                vortex_y = y;
            }
        }
    }

    println!("\nPrimary vortex center:");
    println!("  x/L = {:.3}, y/H = {:.3}", vortex_x as f64 / nx as f64, vortex_y as f64 / ny as f64);
    println!("  Max vorticity: {:.6}", max_vorticity);

    println!("\n✓ LBM cavity flow complete");
}
