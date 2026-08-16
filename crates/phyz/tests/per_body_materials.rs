//! Per-body contact materials: a body's own material reaches the solve, and
//! the `max`-friction combine rule makes it reach the solve from either side
//! of a contact pair.
//!
//! These are the two facts the API's ergonomics rest on. The first is that
//! `Body::material` is not decoration — setting it, and nothing else, changes
//! whether a block slides. The second is the hazard: because friction
//! combines by `max`, it does not matter *which* member of a pair carries the
//! grippy material, which is exactly why it matters which *other* bodies that
//! member touches. See `phyz-contact`'s README.

use phyz::{
    ContactMaterial, Geometry, Mat3, Model, ModelBuilder, Simulator, SpatialInertia,
    SpatialTransform, State, Vec3,
};

/// Slope angle. `tan(25°) = 0.466`, which sits between the two frictions the
/// tests use, so the same geometry both slides and sticks.
const SLOPE: f64 = 25.0_f64.to_radians();

/// Below `tan(SLOPE)`: a block on this alone slides.
const SLIPPERY: f64 = 0.2;

/// Above `tan(SLOPE)`: a block on this sticks.
const GRIPPY: f64 = 1.2;

const DT: f64 = 0.001;
const STEPS: usize = 600;

/// Gravity rotated by `SLOPE` about `+y`, so the ground stays a horizontal
/// plane and the *field* is what tilts. A tilted gravity and a tilted floor
/// are the same physics, and this way the contact normal stays exactly `+z`
/// and any sliding shows up as pure `+x` displacement.
fn tilted_gravity() -> Vec3 {
    Vec3::new(9.81 * SLOPE.sin(), 0.0, -9.81 * SLOPE.cos())
}

fn box_inertia(mass: f64, half: Vec3) -> SpatialInertia {
    let (x, y, z) = (half.x * 2.0, half.y * 2.0, half.z * 2.0);
    let c = mass / 12.0;
    SpatialInertia::new(
        mass,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(
            c * (y * y + z * z),
            c * (x * x + z * z),
            c * (x * x + y * y),
        )),
    )
}

/// Free-joint `q` layout is `[wx, wy, wz, x, y, z]`.
fn set_position(model: &Model, state: &mut State, body: usize, p: Vec3) {
    let off = model.q_offsets[model.bodies[body].joint_idx];
    state.q[off + 3] = p.x;
    state.q[off + 4] = p.y;
    state.q[off + 5] = p.z;
}

fn position(model: &Model, state: &State, body: usize) -> Vec3 {
    let off = model.q_offsets[model.bodies[body].joint_idx];
    Vec3::new(state.q[off + 3], state.q[off + 4], state.q[off + 5])
}

// ---------------------------------------------------------------------------
// A block on the ground: its own material is the only thing that holds it.
// ---------------------------------------------------------------------------

/// A single free box resting on the ground plane at `z = 0`.
fn block_on_ground() -> Model {
    let half = Vec3::new(0.05, 0.05, 0.05);
    let mut model = ModelBuilder::new()
        .gravity(tilted_gravity())
        .dt(DT)
        .add_free_body(
            "block",
            -1,
            SpatialTransform::identity(),
            box_inertia(1.0, half),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box { half_extents: half });
    model
}

/// How far the block travels down the slope in `STEPS`.
fn slide_distance(model: &Model, scene: &ContactMaterial) -> f64 {
    let sim = Simulator::new();
    let mut state = model.default_state();
    // Resting exactly on the plane: half-extent above it.
    set_position(model, &mut state, 0, Vec3::new(0.0, 0.0, 0.05));
    for _ in 0..STEPS {
        sim.step_with_contacts(model, &mut state, 0.0, scene);
    }
    position(model, &state, 0).x
}

/// The headline claim: the *only* difference between these two runs is
/// `Body::material`, and it is the difference between sliding and sticking.
///
/// Both runs pass the same slippery scene material. In the first the block
/// takes it (`material == None`, the pre-existing behaviour) and slides. In
/// the second the block brings `mu = 1.2` of its own, and stops.
#[test]
fn block_sticks_on_incline_only_because_of_its_own_material() {
    let slippery = ContactMaterial {
        friction: SLIPPERY,
        ..Default::default()
    };

    // Sanity: the slope really is steep enough to slide a mu = 0.2 block.
    assert!(SLIPPERY < SLOPE.tan() && SLOPE.tan() < GRIPPY);

    let bare = block_on_ground();
    assert!(bare.bodies[0].material.is_none(), "default must be None");
    let slid = slide_distance(&bare, &slippery);

    let mut gripped = block_on_ground();
    gripped.bodies[0].material = Some(ContactMaterial {
        friction: GRIPPY,
        ..Default::default()
    });
    let held = slide_distance(&gripped, &slippery);

    // Free fall along the slope for 0.6 s would be ~0.75 m; even with friction
    // eating most of it, a mu = 0.2 block on a 25° slope has a net downslope
    // acceleration of g*(sin - mu*cos) = 2.4 m/s^2, so ~0.43 m.
    assert!(
        slid > 0.05,
        "with the scene's slippery material the block should slide; went {slid:.4} m"
    );
    assert!(
        held.abs() < 1e-3,
        "with its own mu = {GRIPPY} the block should hold; went {held:.4} m"
    );
    assert!(
        held.abs() < 0.05 * slid,
        "sticking run ({held:.4} m) should be far shorter than the sliding one ({slid:.4} m)"
    );
}

/// The per-body material must not leak onto bodies that did not ask for one.
/// A second, bare block in the same scene still slides.
#[test]
fn a_bodys_material_does_not_leak_to_its_neighbours() {
    let half = Vec3::new(0.05, 0.05, 0.05);
    let mut model = ModelBuilder::new()
        .gravity(tilted_gravity())
        .dt(DT)
        .add_free_body(
            "gripped",
            -1,
            SpatialTransform::identity(),
            box_inertia(1.0, half),
        )
        .add_free_body(
            "bare",
            -1,
            SpatialTransform::identity(),
            box_inertia(1.0, half),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box { half_extents: half });
    model.bodies[1].geometry = Some(Geometry::Box { half_extents: half });
    model.bodies[0].material = Some(ContactMaterial {
        friction: GRIPPY,
        ..Default::default()
    });

    let scene = ContactMaterial {
        friction: SLIPPERY,
        ..Default::default()
    };
    let sim = Simulator::new();
    let mut state = model.default_state();
    set_position(&model, &mut state, 0, Vec3::new(0.0, 0.0, 0.05));
    // Well clear in y so the two never touch each other.
    set_position(&model, &mut state, 1, Vec3::new(0.0, 1.0, 0.05));
    for _ in 0..STEPS {
        sim.step_with_contacts(&model, &mut state, 0.0, &scene);
    }

    let gripped = position(&model, &state, 0).x;
    let bare = position(&model, &state, 1).x;
    assert!(gripped.abs() < 1e-3, "gripped block moved {gripped:.4} m");
    assert!(
        bare > 0.05,
        "bare block should still slide; went {bare:.4} m"
    );
}

// ---------------------------------------------------------------------------
// A pair: max-combine means either side can supply the grip.
// ---------------------------------------------------------------------------

/// A fixed slab with a free block resting on its top face. Ground is placed
/// far below so the only contact in the problem is the body-body one.
fn block_on_slab() -> Model {
    let slab = Vec3::new(2.0, 2.0, 0.5);
    let block = Vec3::new(0.05, 0.05, 0.05);
    let mut model = ModelBuilder::new()
        .gravity(tilted_gravity())
        .dt(DT)
        .add_fixed_body(
            "slab",
            -1,
            SpatialTransform::identity(),
            box_inertia(1000.0, slab),
        )
        .add_free_body(
            "block",
            -1,
            SpatialTransform::identity(),
            box_inertia(1.0, block),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box { half_extents: slab });
    model.bodies[1].geometry = Some(Geometry::Box {
        half_extents: block,
    });
    model
}

/// Downslope travel of the block on the slab, with `grippy` attached to
/// whichever body index is named (or to neither, for `None`).
fn pair_slide(grippy_on: Option<usize>) -> f64 {
    let mut model = block_on_slab();
    if let Some(b) = grippy_on {
        model.bodies[b].material = Some(ContactMaterial {
            friction: GRIPPY,
            ..Default::default()
        });
    }
    let scene = ContactMaterial {
        friction: SLIPPERY,
        ..Default::default()
    };

    let sim = Simulator::new();
    let mut state = model.default_state();
    // Slab top face is z = +0.5; the block sits on it.
    set_position(&model, &mut state, 1, Vec3::new(0.0, 0.0, 0.55));
    for _ in 0..STEPS {
        // Ground far below: body-body is the only contact.
        sim.step_with_contacts(&model, &mut state, -10.0, &scene);
    }
    position(&model, &state, 1).x
}

/// The combine rule is `max`, so it does not matter which member of the pair
/// carries the grip: the moving block and the static slab produce the same
/// contact, and both hold.
///
/// This is the symmetry that makes the API a hazard rather than a
/// convenience. "Either side works" for *this* pair is the same fact as
/// "a grippy body grips everything else it touches too" — the reason grip
/// tape belongs on the shoes and not on the skateboard deck, whose underside
/// also meets the road.
#[test]
fn max_combine_lets_either_body_supply_the_grip() {
    let neither = pair_slide(None);
    let on_slab = pair_slide(Some(0));
    let on_block = pair_slide(Some(1));

    assert!(
        neither > 0.05,
        "control: with neither body grippy the block should slide; went {neither:.4} m"
    );
    assert!(
        on_slab.abs() < 1e-3,
        "grip on the static slab should hold the block; went {on_slab:.4} m"
    );
    assert!(
        on_block.abs() < 1e-3,
        "grip on the moving block should hold it just the same; went {on_block:.4} m"
    );
    // And not merely "both small": `combine` is commutative, so the two runs
    // are solving an identical problem and must agree to solver tolerance.
    assert!(
        (on_slab - on_block).abs() < 1e-9,
        "max-combine is commutative: {on_slab:.12} vs {on_block:.12}"
    );
}

// ---------------------------------------------------------------------------
// Backwards compatibility.
// ---------------------------------------------------------------------------

/// `Model::contact_materials` on a model whose bodies all carry `None` is
/// exactly the `vec![material; n_bodies]` every caller used to write. This is
/// the claim that lets existing scenes be declared unchanged.
#[test]
fn contact_materials_reproduces_the_old_uniform_vector() {
    let model = block_on_slab();
    let scene = ContactMaterial {
        friction: 0.37,
        restitution: 0.11,
        ..Default::default()
    };
    let got = model.contact_materials(&scene);
    let old = vec![scene.clone(); model.bodies.len().max(1)];
    assert_eq!(got.len(), old.len());
    for (g, o) in got.iter().zip(&old) {
        assert_eq!(g.friction, o.friction);
        assert_eq!(g.restitution, o.restitution);
        assert_eq!(g.stiffness, o.stiffness);
        assert_eq!(g.margin, o.margin);
    }
}

/// Restitution is a per-body property now, and the solver honours it: a
/// bouncy ball on a dead floor bounces, and the dead scene material it was
/// dropped into does not suppress it (restitution also combines by `max`).
#[test]
fn per_body_restitution_reaches_the_solve() {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(DT)
        .add_free_body(
            "ball",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::sphere(0.2, 0.05),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Sphere { radius: 0.05 });

    // Fully inelastic scene: without a per-body material nothing bounces.
    let dead = ContactMaterial {
        restitution: 0.0,
        ..Default::default()
    };

    let peak_after_impact = |m: &Model| {
        let sim = Simulator::new();
        let mut state = m.default_state();
        set_position(m, &mut state, 0, Vec3::new(0.0, 0.0, 0.30));
        let mut touched = false;
        let mut peak = 0.0f64;
        for _ in 0..1200 {
            sim.step_with_contacts(m, &mut state, 0.0, &dead);
            let z = position(m, &state, 0).z;
            if !touched && z <= 0.051 {
                touched = true;
            } else if touched {
                peak = peak.max(z);
            }
        }
        peak
    };

    let inelastic = peak_after_impact(&model);
    model.bodies[0].material = Some(ContactMaterial {
        restitution: 0.7,
        ..Default::default()
    });
    let bouncy = peak_after_impact(&model);

    assert!(
        bouncy > inelastic + 0.01,
        "a restitution-0.7 body must rebound higher than an inelastic one: \
         {bouncy:.4} m vs {inelastic:.4} m"
    );
}
