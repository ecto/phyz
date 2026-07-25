//! Two real solvers, coupled end to end.
//!
//! A charged rigid body (`phyz-rigid`, ABA) orbits in a magnetic field held by
//! an FDTD grid (`phyz-em`, Yee). Each step reads the field out of the grid the
//! FDTD solver just advanced, applies the Lorentz force to the body, and books
//! the equal-and-opposite impulse into the field domain.
//!
//! Run with: `cargo run -p phyz-coupling --example coupled_cyclotron`

use phyz_coupling::{BoundingBox, CoupledSystem, EmSolver, RigidSolver, Solver};
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::ModelBuilder;

fn main() {
    // ω = qB/m = 2π·10⁶ rad/s  →  T = 1 µs,  r = v/ω ≈ 15.9 mm
    let charge = 6.283_185_307_179_586e-3; // C
    let mass = 1e-9; // kg
    let b_mag = 1.0; // T
    let speed = 1e5; // m/s

    let omega = charge * b_mag / mass;
    let period = 2.0 * std::f64::consts::PI / omega;
    let radius = speed / omega;

    let steps = 20_000;
    let dt = period / steps as f64;

    let grid_n = 8;
    let dx = 0.1;
    let center = 0.5 * grid_n as f64 * dx;

    let model = ModelBuilder::new()
        .gravity(Vec3::zeros())
        .dt(dt)
        .add_free_body(
            "charged_bob",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                mass,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(1e-12, 1e-12, 1e-12)),
            ),
        )
        .build();

    let mut state = model.default_state();
    state.q[0] = center;
    state.q[1] = center;
    state.q[2] = center;
    state.v[3] = speed; // free joint: v = [ω(3), v(3)]

    let mut matter = RigidSolver::new(model, state);
    matter.couple_body(0, charge);

    let field = EmSolver::uniform_b_field(grid_n, dx, Vec3::new(0.0, 0.0, b_mag));

    let extent = grid_n as f64 * dx;
    let region = BoundingBox::new(Vec3::zeros(), Vec3::new(extent, extent, extent));

    println!("Coupled rigid-body ↔ electromagnetic simulation");
    println!("===============================================\n");
    println!("Matter domain: {:?}", matter.solver_type());
    println!("  natural dt : {:.3e} s", matter.natural_dt());
    println!("Field domain : {:?}", field.solver_type());
    println!("  natural dt : {:.3e} s (CFL)", field.natural_dt());

    let mut sys = CoupledSystem::new(matter, field, region);
    let schedule = sys.schedule();
    println!(
        "\nSubcycling: base dt = {:.3e} s, ratios = {:?}\n",
        schedule.dt_base, schedule.ratios
    );

    println!("Analytic reference:");
    println!("  ω = {omega:.6e} rad/s");
    println!("  T = {period:.6e} s");
    println!("  r = {radius:.6} m\n");

    let guiding_center = Vec3::new(center, center - radius, center);
    let ke0 = sys.matter.energy();

    println!("  t/T     x (m)      y (m)     |v|/v0     r/r_analytic");
    println!("  -----  ---------  ---------  ---------  ------------");

    for step in 0..steps {
        sys.step(dt);

        if step % (steps / 10) == 0 || step == steps - 1 {
            let site = sys.matter.sites()[0];
            let r = (site.position - guiding_center).norm();
            println!(
                "  {:.3}  {:9.6}  {:9.6}  {:9.6}  {:12.6}",
                (step + 1) as f64 / steps as f64,
                site.position.x,
                site.position.y,
                site.velocity.norm() / speed,
                r / radius,
            );
        }
    }

    let (err_matter, err_field) = sys.absorption_error();
    let ke = sys.matter.energy();

    println!("\nConservation across the handshake:");
    println!(
        "  net impulse booked      : {:.3e} N·s",
        sys.ledger.net_impulse().norm()
    );
    println!(
        "  net work booked         : {:.3e} J",
        sys.ledger.net_work()
    );
    println!(
        "  |impulse| into matter   : {:.3e} N·s",
        sys.ledger.impulse_a.norm()
    );
    println!("  matter absorption error : {:.3e} N·s", err_matter.norm());
    println!("  field  absorption error : {:.3e} N·s", err_field.norm());
    println!("  kinetic energy drift    : {:.3e}", (ke - ke0).abs() / ke0);
    println!("\n  (energy drift is semi-implicit Euler's ~2·2π²/N, not the coupling's)");
}
