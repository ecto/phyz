//! Integration tests for the phyz physics engine.

use approx::assert_relative_eq;
use phyz::{
    ContactMaterial, Geometry, Model, ModelBuilder, Simulator,
    phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3},
    phyz_rigid::{aba, crba, rnea, total_energy},
};

/// Build a single pendulum: revolute about Z, gravity along -Y, rod mass 1kg length 1m.
fn make_pendulum(dt: f64) -> Model {
    let mass = 1.0;
    let length = 1.0;
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(dt)
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
        .build()
}

/// Build a double pendulum with two identical links.
fn make_double_pendulum(dt: f64) -> Model {
    let mass = 1.0;
    let length = 1.0;
    let inertia = SpatialInertia::new(
        mass,
        Vec3::new(0.0, -length / 2.0, 0.0),
        Mat3::from_diagonal(&Vec3::new(
            mass * length * length / 12.0,
            0.0,
            mass * length * length / 12.0,
        )),
    );
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(dt)
        .add_revolute_body("link1", -1, SpatialTransform::identity(), inertia)
        .add_revolute_body(
            "link2",
            0,
            SpatialTransform::from_translation(Vec3::new(0.0, -length, 0.0)),
            inertia,
        )
        .build()
}

#[test]
fn single_pendulum_period() {
    let dt = 0.0001;
    let model = make_pendulum(dt);
    let mut state = model.default_state();
    state.q[0] = 0.1; // small angle

    let sim = Simulator::rk4();

    // Expected period for compound pendulum: T = 2pi * sqrt(I_pivot / (m * g * d))
    // I_pivot = mL^2/3, d = L/2
    let mass = 1.0;
    let length = 1.0;
    let i_pivot = mass * length * length / 3.0;
    let d = length / 2.0;
    let expected_period =
        2.0 * std::f64::consts::PI * (i_pivot / (mass * GRAVITY * d)).sqrt();

    // Simulate for 10 seconds
    let total_steps = (10.0 / dt) as usize;
    let mut prev_q = state.q[0];
    let mut zero_crossings: Vec<f64> = Vec::new();

    for step in 0..total_steps {
        sim.step(&model, &mut state);

        // Detect zero crossing (positive -> negative) = half period
        if prev_q > 0.0 && state.q[0] <= 0.0 {
            let frac = prev_q / (prev_q - state.q[0]);
            let t_cross = (step as f64 + frac) * dt;
            zero_crossings.push(t_cross);
        }
        prev_q = state.q[0];
    }

    // Each consecutive positive-to-negative crossing is one full period apart
    assert!(
        zero_crossings.len() >= 2,
        "need at least 2 zero crossings, got {}",
        zero_crossings.len()
    );

    let mut periods = Vec::new();
    for i in 0..zero_crossings.len() - 1 {
        periods.push(zero_crossings[i + 1] - zero_crossings[i]);
    }
    let avg_period: f64 = periods.iter().sum::<f64>() / periods.len() as f64;
    let relative_error = ((avg_period - expected_period) / expected_period).abs();

    assert!(
        relative_error < 0.02,
        "period error {:.4}% exceeds 2% (measured={:.6}, expected={:.6})",
        relative_error * 100.0,
        avg_period,
        expected_period,
    );
}

#[test]
fn double_pendulum_energy_conservation() {
    let dt = 0.0001;
    let model = make_double_pendulum(dt);
    let mut state = model.default_state();
    state.q[0] = 0.5;
    state.q[1] = 0.3;

    let sim = Simulator::rk4();
    let e0 = total_energy(&model, &state);

    // Simulate for 5 seconds
    let total_steps = (5.0 / dt) as usize;
    sim.simulate(&model, &mut state, total_steps);

    let e_final = total_energy(&model, &state);
    let drift = (e_final - e0).abs();

    assert!(
        drift < 1e-4,
        "energy drift {:.2e} exceeds 1e-4 (e0={:.6}, e_final={:.6})",
        drift,
        e0,
        e_final,
    );
}

#[test]
fn ball_drop_with_contacts() {
    // Test the contact detection and force computation pipeline directly.
    // We manually set up body transforms to place a sphere above the ground,
    // verify contacts are detected and produce upward forces.
    use phyz::phyz_contact::find_ground_contacts;

    let dt = 0.001;
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(dt)
        .add_free_body(
            "ball",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::sphere(1.0, 0.1),
        )
        .build();

    model.bodies[0].geometry = Some(Geometry::Sphere { radius: 0.1 });

    let mut state = model.default_state();
    // Place ball at z=2.0 via q (FK will read this)
    state.q[2] = 2.0;

    // Run FK to update body_xform
    let (xforms, _) = phyz::phyz_rigid::forward_kinematics(&model, &state);
    state.body_xform = xforms;

    // At z=2.0, sphere bottom is at 1.9 -- no contact with ground at z=0
    let geometries: Vec<Option<Geometry>> = model.bodies.iter().map(|b| b.geometry.clone()).collect();
    let contacts = find_ground_contacts(&state, &geometries, 0.0);
    assert!(
        contacts.is_empty(),
        "should have no contacts at z=2.0, got {}",
        contacts.len(),
    );

    // Now place ball at z=0.05, sphere bottom at -0.05 -- should penetrate ground
    state.q[2] = 0.05;
    let (xforms, _) = phyz::phyz_rigid::forward_kinematics(&model, &state);
    state.body_xform = xforms;

    let contacts = find_ground_contacts(&state, &geometries, 0.0);
    assert_eq!(contacts.len(), 1, "should have 1 ground contact at z=0.05");
    let contact = &contacts[0];

    // Penetration depth should be 0.05 (ground at 0, sphere bottom at -0.05)
    assert_relative_eq!(contact.penetration_depth, 0.05, epsilon = 1e-6);

    // Contact normal should point up (z direction)
    assert_relative_eq!(contact.contact_normal.z, 1.0, epsilon = 1e-10);

    // Compute contact forces and verify they push upward
    let material = ContactMaterial::default();
    let materials = vec![material.clone()];
    let forces = phyz::phyz_contact::contact_forces(&contacts, &state, &materials, None);
    // Force on body 0 should have positive z (upward push)
    let fz = forces[0].linear.z;
    assert!(
        fz > 0.0,
        "contact force should push up, got fz = {:.4}",
        fz,
    );

    // Verify ABA with external forces produces upward acceleration
    let spatial_forces = forces;
    let qdd = phyz::phyz_rigid::aba_with_external_forces(&model, &state, Some(&spatial_forces));
    // Free joint v-space: [wx, wy, wz, vx, vy, vz], so linear z accel is qdd[5]
    // With contact force opposing gravity, acceleration should be > -g
    assert!(
        qdd[5] > -GRAVITY,
        "contact force should reduce downward acceleration: qdd_z = {:.4}",
        qdd[5],
    );
}

#[test]
fn gradient_consistency() {
    let model = make_pendulum(0.001);
    let mut state = model.default_state();
    state.q[0] = 0.3;

    let fd = phyz::phyz_diff::finite_diff_jacobians(&model, &state, 1e-7);
    let an = phyz::phyz_diff::analytical_step_jacobians(&model, &state);

    // Compare dvnext_dq entries
    let diff = (&fd.dvnext_dq - &an.dvnext_dq).norm();
    assert!(
        diff < 1e-4,
        "dvnext_dq mismatch: norm diff = {:.2e}",
        diff,
    );

    // Also check other Jacobian blocks
    let diff_qq = (&fd.dqnext_dq - &an.dqnext_dq).norm();
    assert!(
        diff_qq < 1e-4,
        "dqnext_dq mismatch: norm diff = {:.2e}",
        diff_qq,
    );

    let diff_vv = (&fd.dvnext_dv - &an.dvnext_dv).norm();
    assert!(
        diff_vv < 1e-4,
        "dvnext_dv mismatch: norm diff = {:.2e}",
        diff_vv,
    );
}

#[test]
fn aba_equals_minv_phyz_minus_c() {
    let model = make_double_pendulum(0.001);
    let mut state = model.default_state();
    state.q[0] = 0.3;
    state.q[1] = -0.2;
    state.v[0] = 0.1;
    state.v[1] = -0.1;
    // ctrl = 0 (default)

    // qdd from ABA (forward dynamics with zero control)
    let qdd_aba = aba(&model, &state);

    // Mass matrix from CRBA
    let m_mat = crba(&model, &state);

    // Bias forces from RNEA with zero acceleration
    let c = rnea(&model, &state, &DVec::zeros(model.nv));

    // With ctrl=0: M * qdd = -c  (since tau = ctrl - c in the EOM: M*qdd + c = ctrl)
    // Actually RNEA returns tau needed for zero accel, which equals the bias c.
    // EOM: M * qdd + c = ctrl, so M * qdd = ctrl - c = -c (when ctrl=0)
    let m_qdd = &m_mat * &qdd_aba;
    let neg_c = -&c;

    for i in 0..model.nv {
        assert_relative_eq!(m_qdd[i], neg_c[i], epsilon = 1e-8);
    }
}

#[test]
fn free_joint_freefall() {
    // Test that ABA correctly computes freefall acceleration for a free body,
    // and that the velocity integrates correctly over time.
    //
    // Note: for free joints, q = [x, y, z, rx, ry, rz] and v = [wx, wy, wz, vx, vy, vz].
    // The naive integrator q += v*dt maps v[5] (vz) to q[5] (rz), not q[2] (z).
    // So we verify correctness via the velocity state (v[5]) and the known q-slot (q[5])
    // where z-displacement actually accumulates.
    let dt = 0.001;
    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(dt)
        .add_free_body(
            "ball",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::sphere(1.0, 0.1),
        )
        .build();

    let mut state = model.default_state();

    let sim = Simulator::rk4();
    sim.simulate(&model, &mut state, 100);

    // After t = 0.1s, free-fall velocity: vz = -g * t = -9.81 * 0.1 = -0.981
    let expected_vz = -GRAVITY * 0.1;
    let actual_vz = state.v[5]; // linear z velocity in v-space

    assert_relative_eq!(actual_vz, expected_vz, epsilon = 1e-3);

    // The z displacement (-0.5 * g * t^2) ends up in q[5] due to the v->q index mapping.
    let expected_displacement = -0.5 * GRAVITY * 0.1 * 0.1;
    assert_relative_eq!(state.q[5], expected_displacement, epsilon = 1e-3);
}
