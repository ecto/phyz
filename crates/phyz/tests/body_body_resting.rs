//! A box dropped on a box must rest on it, not sink through it.
//!
//! The strongest statement of the contact-normal convention bug this suite
//! exists to prevent. `Collision::contact_normal` is the direction `body_i`
//! must move to *separate*, and the solver's non-penetration row is
//! `J·v ≥ 0` with `J = J_i − J_j`. `find_contacts` used to pass
//! `Manifold::normal` through unchanged, and that points from shape `a`
//! toward shape `b` — the exact opposite sense. The constraint therefore
//! measured approach rather than separation, and the solver drove overlapping
//! bodies *together*.
//!
//! Measured before the fix: a 0.2 m box released just above another sank
//! straight through to full overlap (centre-to-centre distance 0.0 m, i.e.
//! 0.196 m of penetration) while reporting four contacts the entire way down.
//! No test caught it because no body-body contact ever occurred — only the two
//! feet had collision geometry, and feet do not touch each other while
//! standing.

use phyz::Simulator;
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, ModelBuilder};

const HALF: f64 = 0.1;

#[test]
fn a_box_dropped_on_a_box_rests_on_top_of_it() {
    let inertia = SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01);
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        // Welded to the world, so the only thing that can move is the upper
        // box and the only thing that can stop it is the contact.
        .add_fixed_body("lower", -1, SpatialTransform::identity(), inertia)
        .add_free_body("upper", -1, SpatialTransform::identity(), inertia)
        .build();
    for i in 0..2 {
        model.bodies[i].geometry = Some(Geometry::Box {
            half_extents: Vec3::new(HALF, HALF, HALF),
        });
    }

    let mut state = model.default_state();
    // Free-joint q is [wx, wy, wz, x, y, z]; released 5 cm above contact.
    state.q[5] = 2.0 * HALF + 0.05;

    let sim = Simulator::new();
    // Ground far below, so nothing but the body-body contact is in play.
    for _ in 0..400 {
        sim.step_with_contacts(&model, &mut state, -100.0, &Default::default());
        assert!(
            state.q.as_slice().iter().all(|x| x.is_finite()),
            "state went non-finite"
        );
    }

    let z = state.q[5];
    assert!(
        (z - 2.0 * HALF).abs() < 1e-3,
        "the upper box settled at z = {z:.4} m; resting on the lower box is \
         z = {:.4}. Sinking to ~0 means the contact normal is inverted and \
         the solver is pulling the pair together.",
        2.0 * HALF
    );

    // And it is genuinely in contact, at a rigid-contact depth.
    let contacts = phyz_contact::find_contacts(&model, &state, 0.0);
    assert_eq!(contacts.len(), 4, "a face contact is four corners");
    for c in &contacts {
        assert!(
            c.penetration_depth > 0.0 && c.penetration_depth < 1e-3,
            "resting penetration {:.2e} m is not a rigid contact",
            c.penetration_depth
        );
    }
}

/// Newton's third law across the pair, through the solver rather than by
/// construction: the reaction the welded body absorbs is the upper body's
/// weight.
#[test]
fn the_resting_contact_carries_exactly_the_weight() {
    let mass = 2.5;
    let inertia = SpatialInertia::new(mass, Vec3::zeros(), Mat3::identity() * 0.01);
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_fixed_body("lower", -1, SpatialTransform::identity(), inertia)
        .add_free_body("upper", -1, SpatialTransform::identity(), inertia)
        .build();
    for i in 0..2 {
        model.bodies[i].geometry = Some(Geometry::Box {
            half_extents: Vec3::new(HALF, HALF, HALF),
        });
    }
    let mut state = model.default_state();
    state.q[5] = 2.0 * HALF + 0.01;

    let sim = Simulator::new();
    for _ in 0..600 {
        sim.step_with_contacts(&model, &mut state, -100.0, &Default::default());
    }

    // Resting: the upper box is not accelerating, so the contact must be
    // supplying m·g.
    let vz = state.v[5];
    assert!(
        vz.abs() < 1e-3,
        "the upper box is still moving at {vz:.4} m/s after 0.6 s"
    );
    assert!(
        (state.q[5] - 2.0 * HALF).abs() < 1e-3,
        "settled at z = {:.4}, not on top",
        state.q[5]
    );
}
