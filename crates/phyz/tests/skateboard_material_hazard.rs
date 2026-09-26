//! The deck-vs-shoe hazard, as a regression test.
//!
//! `phyz-contact`'s README justifies per-body materials with a measurement
//! from a skateboarding robot: grip tape belongs on the *shoes*, and putting
//! the same friction on the *deck* silently brakes the board against the
//! road. That is the single fact the API's ergonomics turn on, so it is
//! pinned here rather than left as prose.
//!
//! The scene is the mechanism stripped to its bones — a plank rolling flat on
//! the ground with a weight riding on it — because the mechanism is what the
//! claim is about. Friction combines by `max`, so:
//!
//! - **foot-on-deck** is `max(shoe, deck)`, which is `1.5` in *both*
//!   arrangements. Moving the grip to the deck looks like it works.
//! - **deck-on-ground** is the deck's own material, which is `0.6` (wood) in
//!   one arrangement and `1.5` (grip tape) in the other. That is the cost,
//!   and it is invisible at the contact you were trying to fix.
//!
//! The real robot lost 0.6 -> 0.39 m/s and 2 ms of air to this. Here it shows
//! up as the board coasting 0.0124 m instead of 0.0311 m — a factor of 2.51,
//! which is `1.5 / 0.6` to three digits. Coasting distance goes as `1/mu` for
//! a body braked by Coulomb friction, so the measured ratio being the friction
//! ratio is the check that this is the documented mechanism and not a solver
//! artefact that happens to point the right way.

use phyz::{
    ContactMaterial, Geometry, Mat3, Model, ModelBuilder, Simulator, SpatialInertia,
    SpatialTransform, State, Vec3,
};

/// Grip tape on rubber soles.
const GRIP: f64 = 1.5;
/// The deck's bare underside: wood or plastic on asphalt.
const WOOD: f64 = 0.6;

const DT: f64 = 0.0005;
/// 0.4 s — long enough for both arrangements to come to rest, so the measured
/// quantity is total coast distance rather than an arbitrary sample time.
const STEPS: usize = 800;

/// Initial roll speed, m/s. The board is already moving, as a rolling ollie's
/// would be; nothing is jerked to speed at t = 0.
const ROLL: f64 = 0.6;

fn deck_half() -> Vec3 {
    Vec3::new(0.40, 0.10, 0.010)
}
fn foot_half() -> Vec3 {
    Vec3::new(0.06, 0.05, 0.030)
}
const DECK_MASS: f64 = 2.0;
/// Stands in for the rider: what presses the deck onto the road.
const RIDER_MASS: f64 = 10.0;

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

/// Free-joint layout is angular-first for both `q` and `v`:
/// `[wx, wy, wz, x, y, z]`. With the body unrotated the linear half is world
/// position / world linear velocity.
fn place(model: &Model, state: &mut State, body: usize, z: f64, vx: f64) {
    let q = model.q_offsets[model.bodies[body].joint_idx];
    let v = model.v_offsets[model.bodies[body].joint_idx];
    state.q[q + 5] = z;
    state.v[v + 3] = vx;
}

fn x_of(model: &Model, state: &State, body: usize) -> f64 {
    state.q[model.q_offsets[model.bodies[body].joint_idx] + 3]
}

/// Body 0 is the deck, body 1 the rider's foot resting on it.
fn skateboard(deck: ContactMaterial, shoe: ContactMaterial) -> Model {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(DT)
        .add_free_body(
            "deck",
            -1,
            SpatialTransform::identity(),
            box_inertia(DECK_MASS, deck_half()),
        )
        .add_free_body(
            "foot",
            -1,
            SpatialTransform::identity(),
            box_inertia(RIDER_MASS, foot_half()),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: deck_half(),
    });
    model.bodies[1].geometry = Some(Geometry::Box {
        half_extents: foot_half(),
    });
    model.bodies[0].material = Some(deck);
    model.bodies[1].material = Some(shoe);
    model
}

struct Run {
    /// How far the deck coasted before friction stopped it, m.
    deck_travel: f64,
    /// Worst foot-relative-to-deck slip over the run, m. This is the quantity
    /// the grip tape exists to keep at zero.
    foot_slip: f64,
}

fn coast(deck: ContactMaterial, shoe: ContactMaterial) -> Run {
    let model = skateboard(deck, shoe);
    let sim = Simulator::new();
    let mut state = model.default_state();
    place(&model, &mut state, 0, deck_half().z, ROLL);
    // The foot starts on the deck's top face, moving with it.
    place(
        &model,
        &mut state,
        1,
        2.0 * deck_half().z + foot_half().z,
        ROLL,
    );

    let x0_deck = x_of(&model, &state, 0);
    let x0_foot = x_of(&model, &state, 1);
    let mut foot_slip = 0.0f64;

    // The scene material is never what decides anything here: both bodies
    // carry their own. It is present because `step_with_contacts` needs one.
    let asphalt = ContactMaterial {
        friction: 0.8,
        ..Default::default()
    };
    for _ in 0..STEPS {
        sim.step_with_contacts(&model, &mut state, 0.0, &asphalt);
        let slip = (x_of(&model, &state, 1) - x0_foot) - (x_of(&model, &state, 0) - x0_deck);
        foot_slip = foot_slip.max(slip.abs());
    }

    Run {
        deck_travel: x_of(&model, &state, 0) - x0_deck,
        foot_slip,
    }
}

/// Grip on the shoes and grip on the deck fix the foot equally — and only one
/// of them leaves the board rolling.
///
/// If this test ever fails because the two arrangements have become
/// equivalent, the `max`-friction combine rule has changed, and every "which
/// body do I attach this to?" answer in the docs is wrong.
#[test]
fn grip_on_the_deck_brakes_the_board_that_grip_on_the_shoe_does_not() {
    let grip = ContactMaterial {
        friction: GRIP,
        ..Default::default()
    };
    let wood = ContactMaterial {
        friction: WOOD,
        ..Default::default()
    };

    // Right: grip tape on the soles, bare wood underneath.
    let on_shoe = coast(wood.clone(), grip.clone());
    // Wrong: the same number, moved to the deck body.
    let on_deck = coast(grip, wood);

    // 1. The foot is held in both. This is why the mistake is invisible at the
    //    contact you were trying to fix: `max(1.5, 0.6)` either way.
    assert!(
        on_shoe.foot_slip < 2e-3,
        "grip on the shoe must hold the foot; slipped {:.4} m",
        on_shoe.foot_slip
    );
    assert!(
        on_deck.foot_slip < 2e-3,
        "grip on the deck holds the foot just as well — that is the trap; \
         slipped {:.4} m",
        on_deck.foot_slip
    );

    // 2. And the board pays for it against the road, where nobody was looking.
    assert!(
        on_shoe.deck_travel > 2.0 * on_deck.deck_travel,
        "grip on the deck must brake the board against the ground: \
         shoe {:.4} m vs deck {:.4} m (expect ~2.51x, the friction ratio)",
        on_shoe.deck_travel,
        on_deck.deck_travel
    );

    // 3. Both must actually roll — a test where nothing moves would pass (1)
    //    and (2) is a ratio, so this is what keeps the scene honest.
    assert!(
        on_deck.deck_travel > 1e-3,
        "the board should roll at least a little; went {:.4} m",
        on_deck.deck_travel
    );
}

/// The control the claim needs: the difference above is caused by the deck's
/// material and nothing else. Holding the shoe at `GRIP` and sweeping only the
/// deck reproduces it monotonically — a grippier deck always coasts less.
#[test]
fn deck_travel_is_monotone_in_the_decks_own_friction() {
    let grip = ContactMaterial {
        friction: GRIP,
        ..Default::default()
    };
    let travels: Vec<f64> = [0.4, 0.6, 1.0, 1.5]
        .iter()
        .map(|&mu| {
            coast(
                ContactMaterial {
                    friction: mu,
                    ..Default::default()
                },
                grip.clone(),
            )
            .deck_travel
        })
        .collect();

    for w in travels.windows(2) {
        assert!(
            w[0] > w[1],
            "a grippier deck must coast less far: {travels:?}"
        );
    }
    // The span is large, not marginal — this is a design-relevant effect, not
    // solver noise.
    assert!(
        travels[0] > 2.0 * travels[3],
        "mu 0.4 vs 1.5 should differ by more than 2x: {travels:?}"
    );
}
