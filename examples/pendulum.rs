//! Single pendulum example — validates period and demonstrates gradients.

use phyz::{
    ModelBuilder, Simulator,
    phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3},
    phyz_rigid::{kinetic_energy, potential_energy, total_energy},
};

fn main() {
    let length = 1.0;
    let mass = 1.0;

    // Pendulum: revolute about Z, gravity along -Y, rod hangs in -Y at q=0.
    // CoM at [0, -L/2, 0]. Inertia of rod about CoM: I_xx = I_zz = mL²/12.
    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001)
        .add_revolute_body(
            "pendulum",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                mass,
                Vec3::new(0.0, -length / 2.0, 0.0),
                Mat3::from_diagonal(&Vec3::new(
                    mass * length * length / 12.0,
                    0.0,
                    mass * length * length / 12.0,
                )),
            ),
        )
        .build();

    let mut state = model.default_state();
    // Start at small angle for simple harmonic approximation
    state.q[0] = 0.1; // radians

    let sim = Simulator::new();

    // Expected period for compound pendulum: T = 2π√(I/(m*g*d))
    // I = mL²/3 (about pivot), d = L/2 (distance pivot to CoM)
    let i_pivot = mass * length * length / 3.0;
    let d = length / 2.0;
    let expected_period = 2.0 * std::f64::consts::PI * (i_pivot / (mass * GRAVITY * d)).sqrt();
    println!("Expected period: {expected_period:.4} s");

    let e0 = total_energy(&model, &state);
    println!("Initial energy: {e0:.6} J");
    println!("Initial angle:  {:.4} rad\n", state.q[0]);

    // Simulate and find period by detecting zero crossings
    let mut prev_q = state.q[0];
    let mut zero_crossings: Vec<f64> = Vec::new();
    let total_steps = 20_000; // 20 seconds

    print!("time(s)    q(rad)     v(rad/s)   KE         PE         Total E\n");
    print!("─────────────────────────────────────────────────────────────────\n");

    for step in 0..total_steps {
        sim.step(&model, &mut state);

        // Detect zero crossing (positive → negative)
        if prev_q > 0.0 && state.q[0] <= 0.0 {
            let frac = prev_q / (prev_q - state.q[0]);
            let t_cross = (step as f64 + frac) * model.dt;
            zero_crossings.push(t_cross);
        }
        prev_q = state.q[0];

        if step % 2000 == 0 {
            let ke = kinetic_energy(&model, &state);
            let pe = potential_energy(&model, &state);
            let te = total_energy(&model, &state);
            println!(
                "{:8.3}   {:+8.5}   {:+8.5}   {:8.6}   {:8.6}   {:8.6}",
                state.time, state.q[0], state.v[0], ke, pe, te
            );
        }
    }

    let e_final = total_energy(&model, &state);
    let energy_drift = ((e_final - e0) / e0).abs();
    println!("\nFinal energy:  {e_final:.6} J");
    println!("Energy drift:  {:.2e} (relative)", energy_drift);

    // Compute period from zero crossings (each positive→negative crossing = one period)
    if zero_crossings.len() >= 2 {
        let mut periods = Vec::new();
        for i in 0..zero_crossings.len() - 1 {
            periods.push(zero_crossings[i + 1] - zero_crossings[i]);
        }
        let avg_period: f64 = periods.iter().sum::<f64>() / periods.len() as f64;
        let period_error = ((avg_period - expected_period) / expected_period).abs();
        println!("\nMeasured period: {avg_period:.4} s");
        println!("Expected period: {expected_period:.4} s");
        println!("Period error:    {:.2}%", period_error * 100.0);
    }

    // Demonstrate gradient computation
    println!("\n── Gradient Demo ──");
    let mut grad_state = model.default_state();
    grad_state.q[0] = 0.3;
    grad_state.v[0] = 0.0;

    let sim = Simulator::new();
    let jac = sim.step_with_jacobians(&model, &mut grad_state);

    println!("At q=0.3, v=0:");
    println!("  dq'/dq = {:.6}", jac.dqnext_dq[(0, 0)]);
    println!("  dq'/dv = {:.6}", jac.dqnext_dv[(0, 0)]);
    println!("  dv'/dq = {:.6}", jac.dvnext_dq[(0, 0)]);
    println!("  dv'/dv = {:.6}", jac.dvnext_dv[(0, 0)]);
}
