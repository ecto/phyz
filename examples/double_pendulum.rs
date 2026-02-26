//! Double pendulum — energy conservation and gradient validation.

use phyz::{
    ModelBuilder, Simulator, phyz_diff,
    phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3},
    phyz_rigid::total_energy,
};

fn main() {
    let length = 1.0;
    let mass = 1.0;

    // Double pendulum: revolute Z joints, gravity along -Y, links hang in -Y.
    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001)
        .add_revolute_body(
            "link1",
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
        .add_revolute_body(
            "link2",
            0,
            SpatialTransform::from_translation(Vec3::new(0.0, -length, 0.0)), // joint at end of link1
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
    state.q[0] = std::f64::consts::FRAC_PI_4; // 45 deg
    state.q[1] = std::f64::consts::FRAC_PI_4;

    let sim = Simulator::rk4();
    let e0 = total_energy(&model, &state);

    println!("Double Pendulum Simulation");
    println!("==========================");
    println!("Initial angles: [{:.3}, {:.3}] rad", state.q[0], state.q[1]);
    println!("Initial energy: {e0:.8} J\n");

    let total_steps = 10_000; // 10 seconds
    let mut max_drift: f64 = 0.0;

    println!("time(s)    q1(rad)    q2(rad)    energy     drift");
    println!("----------------------------------------------------");

    for step in 0..total_steps {
        sim.step(&model, &mut state);

        let e = total_energy(&model, &state);
        let drift = ((e - e0) / e0).abs();
        max_drift = max_drift.max(drift);

        if step % 1000 == 0 {
            println!(
                "{:8.3}   {:+7.4}    {:+7.4}    {:10.8}  {:.2e}",
                state.time, state.q[0], state.q[1], e, drift
            );
        }
    }

    let e_final = total_energy(&model, &state);
    let final_drift = ((e_final - e0) / e0).abs();

    println!("\n-- Energy Conservation --");
    println!("Initial energy: {e0:.8} J");
    println!("Final energy:   {e_final:.8} J");
    println!("Max drift:      {max_drift:.2e}");
    println!("Final drift:    {final_drift:.2e}");

    if final_drift < 0.01 {
        println!("PASS: Energy conserved within 1% over 10s");
    } else {
        println!("FAIL: Energy drift exceeds 1%!");
    }

    // ── Gradient Validation ──
    println!("\n-- Gradient Validation --");
    let mut state_grad = model.default_state();
    state_grad.q[0] = 0.3;
    state_grad.q[1] = 0.2;
    state_grad.v[0] = 0.1;
    state_grad.v[1] = -0.1;

    let fd = phyz_diff::finite_diff_jacobians(&model, &state_grad, 1e-6);
    let an = phyz_diff::analytical_step_jacobians(&model, &state_grad);

    let checks = [
        ("dq'/dq", &fd.dqnext_dq, &an.dqnext_dq),
        ("dq'/dv", &fd.dqnext_dv, &an.dqnext_dv),
        ("dv'/dq", &fd.dvnext_dq, &an.dvnext_dq),
        ("dv'/dv", &fd.dvnext_dv, &an.dvnext_dv),
        ("dv'/dctrl", &fd.dvnext_dctrl, &an.dvnext_dctrl),
    ];

    let mut all_pass = true;
    for (name, fd_mat, an_mat) in &checks {
        let diff = (*fd_mat - *an_mat).norm();
        let scale = fd_mat.norm().max(1.0);
        let rel_err = diff / scale;
        let pass = rel_err < 1e-4;
        println!(
            "  {name:12} rel_error = {rel_err:.2e}  {}",
            if pass { "PASS" } else { "FAIL" }
        );
        if !pass {
            all_pass = false;
        }
    }

    if all_pass {
        println!("\nPASS: All gradients match within 1e-4 relative error");
    } else {
        println!("\nFAIL: Some gradients failed validation!");
    }
}
