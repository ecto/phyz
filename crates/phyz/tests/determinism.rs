//! Reproducibility gates for the rollout path.
//!
//! These are the tests that make "the same experiment run twice is the same
//! experiment" a checked property rather than an assumption. They are all
//! about **bits**, not tolerances: a tolerance test cannot distinguish an
//! engine that is reproducible from one that is merely close, and "merely
//! close" is what a contact-rich rollout amplifies into a different answer.
//!
//! The scenes are deliberately contact-rich and deliberately unstable — a box
//! tipping over an edge, a wheel rolling, a chain of links collapsing onto the
//! ground. A pendulum swinging in vacuum would pass every test here without
//! exercising a single line of the code that has ever broken.
//!
//! See `docs/determinism.md` for what these gates do and do not promise.

use phyz::determinism::{RolloutHasher, divergence, hash_rollout, state_distance, ulp_offset};
use phyz::{
    ContactMaterial, Geometry, Mat3, Model, ModelBuilder, Simulator, SpatialInertia,
    SpatialTransform, State, Vec3,
};

// ---------------------------------------------------------------- the scenes

/// Ground plane height and the material every scene contacts it with.
const GROUND: f64 = 0.0;

fn material() -> ContactMaterial {
    ContactMaterial::default()
}

/// A named scene: a model, an initial state, and how many steps to run.
struct Scene {
    name: &'static str,
    model: Model,
    state: State,
    steps: usize,
}

fn box_inertia(mass: f64, h: Vec3) -> SpatialInertia {
    let ix = mass * (h.y * h.y + h.z * h.z) / 3.0;
    let iy = mass * (h.x * h.x + h.z * h.z) / 3.0;
    let iz = mass * (h.x * h.x + h.y * h.y) / 3.0;
    SpatialInertia::new(
        mass,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(ix, iy, iz)),
    )
}

/// A box dropped a few millimetres with a sideways shove and a little spin, so
/// it lands on a corner and tips. The interesting part is not the drop: it is
/// the moment the contact set goes from one corner to two to four, which is
/// where an order-dependent solve produces a different answer.
fn box_tipping() -> Scene {
    let half = Vec3::new(0.05, 0.05, 0.05);
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(1e-3)
        .add_free_body(
            "box",
            -1,
            SpatialTransform::identity(),
            box_inertia(1.0, half),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box { half_extents: half });

    let mut state = model.default_state();
    // q = [wx, wy, wz, x, y, z]. Tilt about y so it lands on an edge, and give
    // it lateral velocity so it keeps going once it does.
    state.q[1] = 0.35;
    state.q[5] = 0.08;
    state.v[4] = 0.9;
    state.v[1] = -1.5;

    Scene {
        name: "box_tipping",
        model,
        state,
        steps: 1_500,
    }
}

/// A cylinder on its side, spun up, rolling along the ground. Rolling contact
/// is the case where the contact *point* moves every step even though the
/// contact is physically the same feature, so it exercises the warm-start
/// cache's keying and the manifold's point ordering at the same time.
fn wheel_rolling() -> Scene {
    let (radius, height, mass) = (0.1, 0.04, 2.0);
    let i_axial = 0.5 * mass * radius * radius;
    let i_diam = mass * (3.0 * radius * radius + height * height) / 12.0;
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(1e-3)
        .add_free_body(
            "wheel",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                mass,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(i_diam, i_axial, i_diam)),
            ),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Cylinder { radius, height });

    let mut state = model.default_state();
    // Lay the cylinder's axis along -y (rotate 90 deg about x) so it rolls in x.
    state.q[0] = std::f64::consts::FRAC_PI_2;
    state.q[5] = radius + 2e-3;
    // Spin about the AXLE, which is the shape's own z — `v[1]` is the body's
    // y, and after the 90 deg roll that is world *up*, so the fixture used to
    // yaw the disc like a spun coin and never rolled it at all. Nothing
    // noticed, because the four-rim-sample cylinder could not roll either.
    // With the axle right and `v_x = omega r`, this is rolling without
    // slipping: the contact point is stationary on the ground.
    state.v[2] = -12.0;
    state.v[3] = 12.0 * radius;

    Scene {
        name: "wheel_rolling",
        model,
        state,
        steps: 1_500,
    }
}

/// A free base with three hinged links below it, released above the ground and
/// allowed to collapse: the humanoid-falling case in miniature. Multiple
/// bodies means multiple simultaneous ground contacts *and* body-body pairs,
/// which is the configuration where the broad phase's output order used to
/// depend on where the bodies happened to be.
fn chain_falling() -> Scene {
    let link = SpatialInertia::new(
        0.5,
        Vec3::new(0.0, 0.0, -0.1),
        Mat3::from_diagonal(&Vec3::new(4e-3, 4e-3, 1e-3)),
    );
    let mut builder = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(1e-3)
        .add_free_body(
            "torso",
            -1,
            SpatialTransform::identity(),
            box_inertia(3.0, Vec3::new(0.08, 0.05, 0.12)),
        );
    for k in 0..3 {
        let drop = if k == 0 { 0.12 } else { 0.2 };
        builder = builder.add_revolute_body(
            "link",
            k,
            SpatialTransform::new(Mat3::identity(), Vec3::new(0.0, 0.0, -drop)),
            link,
        );
    }
    let mut model = builder.build();
    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: Vec3::new(0.08, 0.05, 0.12),
    });
    for b in model.bodies.iter_mut().skip(1) {
        b.geometry = Some(Geometry::Capsule {
            radius: 0.03,
            length: 0.16,
        });
    }

    let mut state = model.default_state();
    state.q[1] = 0.5; // pitch the torso so it lands on a corner
    state.q[5] = 0.55;
    state.v[3] = 0.4;
    // Kink the chain so the links swing into each other on the way down.
    state.q[6] = 0.4;
    state.q[7] = -0.7;
    state.q[8] = 0.5;

    Scene {
        name: "chain_falling",
        model,
        state,
        steps: 1_200,
    }
}

fn scenes() -> Vec<Scene> {
    vec![box_tipping(), wheel_rolling(), chain_falling()]
}

/// Fingerprint one scene's rollout on a freshly constructed simulator.
///
/// Fresh is the point: the contract these tests pin is that a rollout is
/// reproducible given `(model, initial state, dt)` *and a simulator in a known
/// initial state*, which `Simulator::new()` is.
fn fingerprint(scene: &Scene) -> u64 {
    let sim = Simulator::new();
    let mut state = scene.state.clone();
    let mat = material();
    hash_rollout(&mut state, scene.steps, |s| {
        sim.step_with_contacts(&scene.model, s, GROUND, &mat);
    })
}

// --------------------------------------------------------------- the golden

/// Golden fingerprints for the three scenes, `(name, hash)`.
///
/// **If this test fails, the bits moved.** That is not automatically wrong —
/// a deliberate physics change moves them, and should. What it must never be
/// is a surprise. The workflow when it fires:
///
/// 1. Work out *which* change moved them. If you cannot name one, you have
///    found a bug, and it is almost certainly non-determinism rather than
///    physics: check anything newly iterating a `HashMap`, sorting by address
///    or discovery order, reducing in parallel, or calling `f64::sin` instead
///    of `phyz_math::fp::sin`.
/// 2. If the change was intended, check the physics gates still pass
///    (`analytic_benchmarks`, `contact_stability`, the energy suites), then
///    update the constants below **in the same commit as the change**, so the
///    history says which commit moved the numbers.
///
/// Do not update these to make a red test green without doing step 1.
///
/// `wheel_rolling` moved on the commit that gave the cylinder an analytic
/// ground contact (its lowest generator line, instead of eight body-frame rim
/// samples). Two changes in that commit reach this scene: the contact geometry
/// itself, and the fixture's spin axis, which was `v[1]` — the body's *y*,
/// which after the 90 deg roll is world up — so the disc was being spun like a
/// coin rather than rolled. The other two scenes carry no cylinder and their
/// hashes are unchanged, which is the check that the new code is confined to
/// the shape it claims.
const GOLDEN: &[(&str, u64)] = &[
    ("box_tipping", 0xdf32_a0df_8c7f_0577),
    ("wheel_rolling", 0x0843_1b92_eb20_3d15),
    ("chain_falling", 0x6c5b_4ac0_1d83_af9f),
];

#[test]
fn golden_rollout_hashes() {
    let mut got = Vec::new();
    for scene in scenes() {
        got.push((scene.name, fingerprint(&scene)));
    }

    let mismatched: Vec<String> = got
        .iter()
        .zip(GOLDEN)
        .filter(|((n, h), (gn, gh))| {
            assert_eq!(n, gn, "scene order changed");
            h != gh
        })
        .map(|((n, h), (_, gh))| format!("  {n}: expected {gh:016x}, got {h:016x}"))
        .collect();

    assert!(
        mismatched.is_empty(),
        "rollout fingerprints moved:\n{}\n\n\
         Read the doc comment on GOLDEN before touching it. In short: name the \
         change that moved them, or you have found a determinism bug.",
        mismatched.join("\n")
    );
}

// ------------------------------------------------------- reproducibility

/// The same rollout run twice in the same process must be bit-identical.
///
/// The weakest possible form of the guarantee, and the one that catches
/// address-order iteration and uninitialised reads.
#[test]
fn a_rollout_replays_bit_for_bit() {
    for scene in scenes() {
        assert_eq!(
            fingerprint(&scene),
            fingerprint(&scene),
            "{} did not replay",
            scene.name
        );
    }
}

/// Stepping the same state on eight threads at once must give eight identical
/// answers.
///
/// The rigid rollout path is single-threaded — `rayon` appears nowhere in
/// `phyz-rigid`, `phyz-collision`, `phyz-contact` or `phyz`, so there is no
/// parallel reduction whose order could vary. This is the test that keeps that
/// true: it fails the moment a step starts depending on process-global mutable
/// state, a thread-local scratch buffer, or a floating-point mode set per
/// thread. It is the "parallel equals serial" gate for a serial engine.
#[test]
fn concurrent_rollouts_agree_with_the_serial_one() {
    for scene in scenes() {
        let serial = fingerprint(&scene);
        let scene = &scene;
        let parallel: Vec<u64> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|_| s.spawn(move || fingerprint(scene)))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (k, h) in parallel.iter().enumerate() {
            assert_eq!(
                *h, serial,
                "{} thread {k} disagreed with the serial rollout",
                scene.name
            );
        }
    }
}

/// A simulator's warm-start cache must not leak one rollout into the next.
///
/// This is the concrete form of the trap: the contact cache is hidden mutable
/// state on `Simulator`, and warm starting *does* move the solver's answer
/// within its tolerance — the seed changes where PGS stops and which active
/// set Newton is handed. So reusing one simulator across trials makes a
/// candidate's score depend on what was evaluated before it. Both documented
/// escapes must actually work.
#[test]
fn a_reused_simulator_does_not_contaminate_the_next_rollout() {
    let mat = material();
    for scene in scenes() {
        let run = |sim: &Simulator| {
            let mut s = scene.state.clone();
            hash_rollout(&mut s, scene.steps, |s| {
                sim.step_with_contacts(&scene.model, s, GROUND, &mat);
            })
        };

        // Escape 1: reset the cache between trials.
        let sim = Simulator::new();
        let first = run(&sim);
        sim.reset_contact_cache();
        assert_eq!(
            run(&sim),
            first,
            "{}: reset_contact_cache did not restore the initial state",
            scene.name
        );

        // Escape 2: turn warm starting off, and the simulator carries nothing
        // between trials at all.
        let cold = Simulator::new().with_warm_start(false);
        let a = run(&cold);
        let b = run(&cold);
        assert_eq!(a, b, "{}: cold simulator was not stateless", scene.name);

        // And a *dirty* simulator must reproduce the cold answer exactly,
        // which is the property that makes `with_warm_start(false)` a real
        // escape rather than a differently-flavoured one.
        let dirty = Simulator::new().with_warm_start(false);
        let mut throwaway = scene.state.clone();
        for _ in 0..97 {
            dirty.step_with_contacts(&scene.model, &mut throwaway, GROUND, &mat);
        }
        assert_eq!(
            run(&dirty),
            a,
            "{}: history leaked through a cold simulator",
            scene.name
        );
    }
}

/// Two independently constructed simulators must agree.
///
/// Distinct from the replay test: this one would catch state that lives on the
/// `Simulator` and is *initialised* differently — a capacity-dependent
/// allocation pattern, say — rather than state carried across steps.
#[test]
fn independent_simulators_agree() {
    let mat = material();
    for scene in scenes() {
        let run = || {
            let sim = Simulator::new();
            let mut s = scene.state.clone();
            hash_rollout(&mut s, scene.steps, |s| {
                sim.step_with_contacts(&scene.model, s, GROUND, &mat);
            })
        };
        assert_eq!(run(), run(), "{}", scene.name);
    }
}

// ----------------------------------------------------- chaos, calibrated

/// A one-ulp perturbation must start at one ulp — not at zero, and not
/// already large.
///
/// This is the sanity check under the divergence report: if the first sample
/// were already far apart, the perturbation would be measuring a discontinuity
/// at step 0 rather than the growth rate of the dynamics.
#[test]
fn one_ulp_diverges_smoothly_from_one_ulp() {
    let mat = material();
    for scene in scenes() {
        // Each rollout gets its own simulator so the contact caches cannot
        // couple them — otherwise this measures the cache, not the physics.
        let sims = [Simulator::new(), Simulator::new()];
        let d = divergence(
            scene.model.nq,
            &scene.state,
            5, // the free base's z coordinate in every scene
            1,
            scene.steps,
            10,
            |which, s| {
                sims[which].step_with_contacts(&scene.model, s, GROUND, &mat);
            },
        );

        assert!(
            d.initial > 0.0 && d.initial < 1e-12,
            "{}: a 1-ulp perturbation should start at ~1e-17, got {}",
            scene.name,
            d.initial
        );
        assert!(
            d.samples.iter().all(|s| s.distance.is_finite()),
            "{}: rollout went non-finite",
            scene.name
        );
        // The separation is monotone in the large: it may dip, but it must not
        // return to exactly zero, because that would mean the perturbation was
        // silently quantized away — the state snapping to a grid somewhere.
        assert!(
            d.final_distance() > 0.0,
            "{}: perturbation vanished entirely",
            scene.name
        );
    }
}

/// Perturbing by zero ulps is the identity, so the two rollouts must stay
/// bit-identical for the whole horizon.
///
/// Together with the test above this is what separates "the engine is
/// deterministic" from "the engine is stable": here the inputs are identical
/// and the output separation must be exactly zero at every sample, no
/// tolerance.
#[test]
fn a_zero_ulp_perturbation_never_separates() {
    let mat = material();
    for scene in scenes() {
        let sims = [Simulator::new(), Simulator::new()];
        let d = divergence(
            scene.model.nq,
            &scene.state,
            5,
            0,
            scene.steps,
            10,
            |which, s| {
                sims[which].step_with_contacts(&scene.model, s, GROUND, &mat);
            },
        );
        assert_eq!(d.initial, 0.0);
        for s in &d.samples {
            assert_eq!(
                s.distance, 0.0,
                "{}: identical inputs separated by {} at t = {}",
                scene.name, s.distance, s.time
            );
        }
    }
}

/// Calibration: what a 1-ulp difference actually does to these scenes.
///
/// The point of this test is not that the numbers are large. Measured on this
/// engine at `dt = 1 ms`, over ~1.5 s of simulated time, a one-ulp change to
/// the base height gives:
///
/// | scene           | growth  | fitted lambda | doubling time |
/// |-----------------|---------|---------------|---------------|
/// | `box_tipping`   | ~85x    | ~-1.1 / s     | — (settles)   |
/// | `wheel_rolling` | ~1.4e4x | ~+1.4 / s     | ~0.48 s       |
/// | `chain_falling` | ~8x     | ~+1.1 / s     | ~0.65 s       |
///
/// Which is worth stating plainly, because it contradicts the intuition that
/// sends people looking for chaos first: the box *contracts*. Contact is
/// dissipative, and a box that settles forgets its initial perturbation rather
/// than amplifying it. The scenes that keep moving — the articulated chain,
/// and the wheel once it could actually roll — show sustained exponential
/// growth. Even there the separation is still `1e-12` after 1.5 s, because
/// exponential growth from `1e-16` takes tens of seconds to reach anything you
/// could see.
///
/// `wheel_rolling` used to sit in the contracting column at ~1.4e2x. That was
/// not a property of rolling: the cylinder's ground contact was eight rim
/// samples, so the wheel clunked over its own corners and stopped after 23 cm.
/// A wheel that rolls does not settle, and a rolling wheel's lean is genuinely
/// unstable, so the sign flipped when the contact became analytic.
///
/// The practical reading: if two rollouts of a scene like these disagree in
/// the *third* digit after one second, chaos is not a sufficient explanation.
/// Look for a bug. That is the distinction this whole module exists to make
/// cheap, and it is the one that cost this project several hours.
#[test]
fn a_one_ulp_perturbation_is_calibrated_for_every_scene() {
    let mat = material();
    for scene in scenes() {
        // Each rollout gets its own simulator so the contact caches cannot
        // couple them — otherwise this measures the cache, not the physics.
        let sims = [Simulator::new(), Simulator::new()];
        let d = divergence(
            scene.model.nq,
            &scene.state,
            5,
            1,
            scene.steps,
            5,
            |which, s| {
                sims[which].step_with_contacts(&scene.model, s, GROUND, &mat);
            },
        );

        // Every scene amplifies — a contact-rich rollout is never a contraction
        // all the way down, because the contact set itself is a discrete
        // function of the state. If this ever reads ~1.0, the perturbation is
        // being quantized away somewhere and the calibration means nothing.
        assert!(
            d.final_distance() > 5.0 * d.initial,
            "{}: a 1-ulp difference should amplify; {} -> {}",
            scene.name,
            d.initial,
            d.final_distance()
        );
        // ...but none of them reaches a scale you could confuse with a bug
        // over this horizon. This is the bound a researcher can quote.
        assert!(
            d.final_distance() < 1e-10,
            "{}: separation reached {} after {} steps, which is far past what \
             1-ulp chaos produces here — suspect a determinism bug",
            scene.name,
            d.final_distance(),
            scene.steps
        );
        assert!(d.time_to_threshold.is_none(), "{}", scene.name);
    }
}

/// The articulated scene is the one with a genuinely positive Lyapunov
/// exponent, and the fit must find it.
///
/// Pinned qualitatively, not to a value: the exponent depends on the scene and
/// asserting a number would make this a change-detector rather than a gate.
/// What must hold is that the report can tell a growing separation from a
/// shrinking one, because that is the judgement it exists to support.
#[test]
fn the_articulated_scene_has_a_positive_lyapunov_exponent() {
    let mat = material();
    let scene = chain_falling();
    let sims = [Simulator::new(), Simulator::new()];
    let d = divergence(
        scene.model.nq,
        &scene.state,
        5,
        1,
        scene.steps,
        5,
        |which, s| {
            sims[which].step_with_contacts(&scene.model, s, GROUND, &mat);
        },
    );
    let lambda = d.lyapunov.expect("separation should fit an exponential");
    assert!(lambda > 0.0, "expected growth, fitted {lambda}");
    let doubling = d
        .doubling_time()
        .expect("positive lambda has a doubling time");
    assert!(
        doubling > 0.0 && doubling < 10.0,
        "doubling time {doubling} s is not a usable calibration"
    );
    // The number a publication would cite: how long before a f64 rollout from
    // identical inputs has lost every bit of agreement.
    let horizon = 52.0 * doubling;
    assert!(
        (1.0..1e3).contains(&horizon),
        "full-mantissa horizon {horizon} s looks wrong"
    );

    // A settling scene must fit the *other* sign, so the report is not simply
    // reporting "positive" for everything. `box_tipping` is the one that
    // settles: it tips onto a face and stops. `wheel_rolling` used to be here
    // too, back when the wheel could not roll — see the calibration table.
    #[allow(clippy::single_element_loop)]
    for settling in [box_tipping()] {
        let sims = [Simulator::new(), Simulator::new()];
        let d = divergence(
            settling.model.nq,
            &settling.state,
            5,
            1,
            settling.steps,
            5,
            |which, s| {
                sims[which].step_with_contacts(&settling.model, s, GROUND, &mat);
            },
        );
        assert!(
            d.lyapunov.is_some_and(|l| l < 0.0),
            "{} should contract once it settles, fitted {:?}",
            settling.name,
            d.lyapunov
        );
    }
}

// -------------------------------------------------- the hasher's own gates

/// The fingerprint must be sensitive to the last bit of the trajectory,
/// otherwise the golden test above is checking nothing.
#[test]
fn the_fingerprint_notices_a_single_ulp() {
    let mat = material();
    let scene = box_tipping();
    let base = fingerprint(&scene);

    let mut nudged = scene.state.clone();
    nudged.q[5] = ulp_offset(nudged.q[5], 1);
    let sim = Simulator::new();
    let other = hash_rollout(&mut nudged, scene.steps, |s| {
        sim.step_with_contacts(&scene.model, s, GROUND, &mat);
    });
    assert_ne!(base, other, "one ulp of initial height left the hash alone");
}

/// `state_distance` and the hasher must agree about what "identical" means:
/// zero distance and equal hashes are the same condition.
#[test]
fn distance_zero_and_equal_hashes_coincide() {
    let scene = box_tipping();
    let a = scene.state.clone();
    let mut b = scene.state.clone();
    assert_eq!(state_distance(&a, &b), 0.0);

    let mut ha = RolloutHasher::new();
    ha.absorb_state(&a);
    let mut hb = RolloutHasher::new();
    hb.absorb_state(&b);
    assert_eq!(ha.finish(), hb.finish());

    b.q[5] = ulp_offset(b.q[5], 1);
    assert!(state_distance(&a, &b) > 0.0);
    let mut hb = RolloutHasher::new();
    hb.absorb_state(&b);
    assert_ne!(ha.finish(), hb.finish());
}
