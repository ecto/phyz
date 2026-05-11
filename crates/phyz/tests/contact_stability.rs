//! Body-to-body contact stability tests.
//!
//! Exercises the surfaces that the mecheval `fit_physics` grader relies on:
//! a fixed host body and a free accessory body, body-pair contact detection,
//! and penalty-based contact forces under gravity.

use approx::assert_relative_eq;
use phyz::{
    ContactMaterial, Geometry, Mat3, ModelBuilder, SpatialInertia, SpatialTransform, SpatialVec,
    Vec3,
    collision::sweep_and_prune,
    contact::{contact_forces, find_contacts, find_ground_contacts},
};

/// Goal 1 — body-pair contact force pushes the free body AWAY from the fixed body.
///
/// Mirrors `ball_drop_with_contacts` in `integration.rs` but uses two real bodies
/// (one fixed host below, one free accessory above) instead of a ground plane.
#[test]
fn body_drop_on_fixed_body_with_contacts() {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        // Body 0: fixed host (a small box) at the origin.
        .add_fixed_body(
            "host",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                1.0,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(0.1, 0.1, 0.1)),
            ),
        )
        // Body 1: free accessory (a sphere) parented to world so its free-joint
        // q is the world transform.
        .add_free_body(
            "accessory",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::sphere(0.1, 0.1),
        )
        .build();

    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: Vec3::new(0.5, 0.5, 0.1),
    });
    model.bodies[1].geometry = Some(Geometry::Sphere { radius: 0.1 });

    let mut state = model.default_state();

    // Place the accessory just slightly overlapping the host on the +z side.
    // Free joint q layout = [x, y, z, wx, wy, wz]; we only set z so the
    // sphere centre sits at z = 0.05, penetrating the box's top face (z = 0.1).
    let q_offset = model.q_offsets[model.bodies[1].joint_idx];
    state.q[q_offset + 2] = 0.05;

    // Manually populate body transforms instead of going through FK so we have
    // an unambiguous contact geometry independent of joint conventions.
    state.body_xform[0] = SpatialTransform::identity();
    state.body_xform[1] = SpatialTransform::new(Mat3::identity(), Vec3::new(0.0, 0.0, 0.05));

    let geometries: Vec<Option<Geometry>> =
        model.bodies.iter().map(|b| b.geometry.clone()).collect();

    let contacts = find_contacts(&model, &state, &geometries);
    assert_eq!(
        contacts.len(),
        1,
        "expected exactly one body-body contact, got {}",
        contacts.len()
    );
    let contact = &contacts[0];
    assert_eq!(contact.body_i, 0);
    assert_eq!(contact.body_j, 1);

    // body_j is above body_i, so the normal (pos_j - pos_i).normalize() points +z.
    assert_relative_eq!(contact.contact_normal.z, 1.0, epsilon = 1e-10);

    // No ground contacts in this scenario.
    let ground = find_ground_contacts(&state, &geometries, -10.0);
    assert!(ground.is_empty());

    let materials = vec![ContactMaterial::default()];
    let forces = contact_forces(&contacts, &state, &materials, None);

    // Force on the free accessory (body 1, sitting ABOVE the host) should point
    // upward (+z) so it separates from the host. Before the sign fix this would
    // be negative.
    let fz_free = forces[1].linear.z;
    assert!(
        fz_free > 0.0,
        "contact force on accessory should push UP (+z), got fz = {:.4}",
        fz_free,
    );

    // Body 0 receives the opposite force (-z). It's a fixed host so this won't
    // move it, but the sign must still be correct for Newton's 3rd law.
    let fz_host = forces[0].linear.z;
    assert!(
        fz_host < 0.0,
        "reaction force on host should push DOWN (-z), got fz = {:.4}",
        fz_host,
    );
    assert_relative_eq!(fz_free, -fz_host, epsilon = 1e-12);

    // Sanity: force magnitude is non-zero.
    assert!(forces[1].linear.norm() > 0.0);
    // No spurious torque component for an axis-aligned offset along the normal
    // (Goal 4 changes this for general offsets, but for now the wrench is
    // purely linear).
    let _ = SpatialVec::zero();
}

/// Goal 2 — `find_contacts` must not panic or seed NaN into the world when a
/// body's transform contains NaN, and the broad phase must totally-order
/// itself even in the presence of NaN AABB endpoints.
#[test]
fn nan_body_xform_does_not_panic_broad_phase() {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_free_body(
            "a",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::sphere(1.0, 0.1),
        )
        .add_free_body(
            "b",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::sphere(1.0, 0.1),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Sphere { radius: 0.1 });
    model.bodies[1].geometry = Some(Geometry::Sphere { radius: 0.1 });

    let mut state = model.default_state();
    // Body 0 stays at the origin; body 1's transform is poisoned with NaN.
    state.body_xform[0] = SpatialTransform::identity();
    state.body_xform[1] = SpatialTransform::new(
        Mat3::identity(),
        Vec3::new(f64::NAN, f64::NAN, f64::NAN),
    );

    let geometries: Vec<Option<Geometry>> =
        model.bodies.iter().map(|b| b.geometry.clone()).collect();

    // Must not panic.
    let contacts = find_contacts(&model, &state, &geometries);
    for c in &contacts {
        assert!(
            c.contact_normal.x.is_finite()
                && c.contact_normal.y.is_finite()
                && c.contact_normal.z.is_finite(),
            "NaN snuck into a contact normal: {:?}",
            c.contact_normal,
        );
    }
}

/// Goal 2 — `find_contacts` must derive a finite normal even when the two
/// bodies are exactly co-located (so `(pos_j - pos_i).normalize()` would NaN).
#[test]
fn coincident_bodies_produce_finite_contact_normal() {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_free_body(
            "a",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::sphere(1.0, 0.1),
        )
        .add_free_body(
            "b",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::sphere(1.0, 0.1),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Sphere { radius: 0.1 });
    model.bodies[1].geometry = Some(Geometry::Sphere { radius: 0.1 });

    let mut state = model.default_state();
    // Both spheres centred at the origin — fully overlapping.
    state.body_xform[0] = SpatialTransform::identity();
    state.body_xform[1] = SpatialTransform::identity();

    let geometries: Vec<Option<Geometry>> =
        model.bodies.iter().map(|b| b.geometry.clone()).collect();

    let contacts = find_contacts(&model, &state, &geometries);
    for c in &contacts {
        assert!(
            c.contact_normal.x.is_finite()
                && c.contact_normal.y.is_finite()
                && c.contact_normal.z.is_finite(),
            "got non-finite normal {:?} for coincident bodies",
            c.contact_normal,
        );
        assert!(
            (c.contact_normal.norm() - 1.0).abs() < 1e-6,
            "contact normal should be unit length, got |n|={}",
            c.contact_normal.norm(),
        );
    }
}

/// Goal 3 — a 30mm plastic cube on a flat plate, default `ContactMaterial`,
/// `dt = 1/2000`. The mecheval `cube_on_plate_simulation_runs` fixture
/// reproduces this exact scenario. With the explicit penalty force the cube
/// launches off the plate; with implicit damping it must:
///
/// * drift < 5mm in z after 1s, and
/// * never penetrate the plate by more than 0.5mm at any sub-step.
///
/// We drive the simulation with a hand-rolled 1D semi-implicit Euler step on
/// the cube's z-axis instead of going through the full ABA pipeline: this
/// isolates the contact-force change and keeps the test independent of the
/// free-joint integration conventions and of `find_contacts`'s broad/narrow
/// phase (GJK returns -1 instead of the true penetration depth for
/// overlapping boxes today — orthogonal to Goal 3). The geometry / depth
/// computation is done analytically below.
#[test]
fn low_mass_cube_settles_on_plate() {
    use phyz::collision::Collision;

    let m = 0.032_f64; // 32g plastic cube
    let half = 0.015_f64; // 30mm cube => 15mm half-extent
    let dt = 1.0 / 2000.0;
    let g = 9.81_f64;
    let plate_top = 0.0_f64;

    let material = ContactMaterial::default();
    let k = material.stiffness;

    // Cube starts 5mm above the plate top.
    let mut z = half + 0.005;
    let mut vz = 0.0_f64;

    let mut max_penetration: f64 = 0.0;
    let total_steps = (1.0 / dt) as usize;
    for _ in 0..total_steps {
        let depth = (plate_top - (z - half)).max(0.0);
        let fz_contact = if depth > 0.0 {
            max_penetration = max_penetration.max(depth);
            // Body-pair convention: plate is body i (below), cube is body j
            // (above). Normal = +z (from i to j). The cube has vel_j = (0,0,vz).
            let collision = Collision {
                body_i: 0,
                body_j: 1,
                contact_point: Vec3::new(0.0, 0.0, plate_top),
                contact_normal: Vec3::z(),
                penetration_depth: depth,
            };
            let wrench = phyz::compute_contact_force_implicit(
                &collision,
                &material,
                &Vec3::zeros(),
                &Vec3::new(0.0, 0.0, vz),
                f64::INFINITY,
                m,
                dt,
            );
            // Force on the cube is along +normal (Goal 1 convention).
            wrench.linear.z
        } else {
            0.0
        };

        let az = fz_contact / m - g;
        vz += az * dt;
        z += vz * dt;

        // Sanity: never NaN.
        assert!(z.is_finite(), "cube z became non-finite: {z}");
        assert!(vz.is_finite(), "cube vz became non-finite: {vz}");
        // Never launches off the plate.
        assert!(
            z < 1.0,
            "cube launched: z={z}, vz={vz}; implicit damping is broken",
        );
    }

    // Equilibrium under implicit-stiffness scheme: F_c · denom = m · k · x_eq
    // and F_c = m · g at rest, giving x_eq = g · denom / k where
    // denom = m + dt·c + dt²·k. This is the discrete-system steady state,
    // which is slightly larger than the continuous m·g/k.
    let denom = m + dt * material.damping + dt * dt * k;
    let x_eq_discrete = g * denom / k;
    let z_eq = half - x_eq_discrete;
    let drift = (z - z_eq).abs();
    assert!(
        drift < 5e-3,
        "cube drifted {:.4}mm after 1s, want < 5mm; z={z}, z_eq={z_eq}",
        drift * 1000.0,
    );
    assert!(
        max_penetration < 5e-4,
        "max penetration {:.4}mm exceeded 0.5mm",
        max_penetration * 1000.0,
    );
    // Cube has effectively stopped moving.
    assert!(
        vz.abs() < 1e-2,
        "cube still moving at v_z = {vz} after 1s",
    );
}

/// Goal 3 sanity: compare implicit vs. explicit contact force at impact for a
/// low-mass body. The implicit form must produce a strictly smaller (and
/// always-stable, m·(...)/(m+dt·c+dt²·k)-scaled) force, so the cube can't
/// flip sign in a single step.
#[test]
fn implicit_force_is_smaller_than_explicit_at_impact() {
    use phyz::collision::Collision;
    use phyz::{compute_contact_force_implicit, contact::compute_contact_force};

    let material = ContactMaterial::default();
    // Body-pair convention: body_i is the (stationary) plate at the origin,
    // body_j is the cube above it, normal = +z (points from i to j).
    let collision = Collision {
        body_i: 0,
        body_j: 1,
        contact_point: Vec3::zeros(),
        contact_normal: Vec3::z(),
        penetration_depth: 1e-5,
    };
    // Cube (body j) is falling at ~1 m/s INTO the contact.
    let vel_i = Vec3::zeros();
    let vel_j = Vec3::new(0.0, 0.0, -1.0);
    let dt = 1.0 / 2000.0;
    let m = 0.032;

    let f_exp = compute_contact_force(&collision, &material, &vel_i, &vel_j).linear.z;
    let f_imp =
        compute_contact_force_implicit(&collision, &material, &vel_i, &vel_j, f64::INFINITY, m, dt)
            .linear
            .z;

    assert!(f_exp > 0.0);
    assert!(f_imp > 0.0);
    assert!(
        f_imp < f_exp,
        "implicit force {f_imp} should be < explicit {f_exp}",
    );
    // The expected ratio is m/(m+dt·c+dt²·k) ≈ 0.38 for default material.
    let expected_ratio_upper_bound = 0.5_f64;
    assert!(
        f_imp / f_exp < expected_ratio_upper_bound,
        "implicit/explicit ratio = {}, want < {}",
        f_imp / f_exp,
        expected_ratio_upper_bound,
    );
}

/// Goal 2 — the broad phase used to panic in `partial_cmp(...).unwrap()` when
/// any AABB endpoint was NaN. After the fix it should produce an empty pair
/// list rather than aborting.
#[test]
fn broad_phase_nan_endpoints_do_not_panic() {
    use phyz::collision::AABB;
    let nan = f64::NAN;
    let aabbs = vec![
        AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
        AABB::new(Vec3::new(nan, 0.0, 0.0), Vec3::new(2.0, 1.0, 1.0)),
        AABB::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5)),
    ];
    // Should not panic; the NaN body is skipped, but bodies 0 and 2 still
    // overlap and produce a pair.
    let pairs = sweep_and_prune(&aabbs);
    assert!(
        pairs.iter().all(|&(a, b)| a != 1 && b != 1),
        "expected NaN body to be excluded, got pairs {pairs:?}",
    );
}
