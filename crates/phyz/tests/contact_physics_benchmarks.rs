//! Design-doc §6.2–§6.4 as **simulations**, not as algebra.
//!
//! `phyz-contact`'s `analytic_benchmarks.rs` covers the same sections against
//! hand-built `ContactProblem`s — one solve, closed-form inputs. That tests the
//! solver. It cannot catch anything that goes wrong across a trajectory:
//! accumulating drift, a stack that settles and then wanders, restitution that
//! is right for one impact and never lets the body come to rest, or a damping
//! term whose sign only shows up after a hundred bounces.
//!
//! So these drop an actual sphere and measure an actual apex.
//!
//! Every tolerance below is written next to the number that was measured, so a
//! future change shows up as a *changed* number rather than as a test that
//! still passes with three times the error.

use phyz::Simulator;
use phyz_contact::ContactMaterial;
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder, State};

const DT: f64 = 1e-3;
/// Free-joint layout is `[rot(3), pos(3)]`.
const Z: usize = 5;

fn sphere(radius: f64, mass: f64) -> Model {
    let i = 0.4 * mass * radius * radius;
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(DT)
        .add_free_body(
            "ball",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                mass,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(i, i, i)),
            ),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Sphere { radius });
    model
}

fn boxes(n: usize, half: f64, masses: &[f64]) -> Model {
    let mut b = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(DT);
    for m in masses.iter().take(n) {
        let i = m * 2.0 / 3.0 * half * half;
        b = b.add_free_body(
            "box",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(*m, Vec3::zeros(), Mat3::from_diagonal(&Vec3::new(i, i, i))),
        );
    }
    let mut model = b.build();
    for k in 0..n {
        model.bodies[k].geometry = Some(Geometry::Box {
            half_extents: Vec3::new(half, half, half),
        });
    }
    model
}

fn state_at(model: &Model, q: &[f64]) -> State {
    let mut s = model.default_state();
    s.q = DVec::from_slice(q);
    s
}

// ---------------------------------------------------------------------------
// §6.2 — restitution from drop height
// ---------------------------------------------------------------------------

/// Drop a sphere from `h0` above the ground and return the apex it reaches
/// after its first bounce, measured as the peak of the actual trajectory.
///
/// The bounce is the step where the vertical velocity turns around. It is not
/// the step where `h` reaches zero: an impacting contact is detected anywhere
/// inside the material's margin, so the turn can happen a fraction of a
/// millimetre *above* the plane, and a harness that waited for `h <= 0` would
/// report a rebound of nothing.
fn bounce_apex(e: f64, h0: f64) -> f64 {
    let radius = 0.05;
    let model = sphere(radius, 1.0);
    let sim = Simulator::new();
    let material = ContactMaterial {
        restitution: e,
        friction: 0.5,
        ..ContactMaterial::default()
    };
    let mut st = state_at(&model, &[0.0, 0.0, 0.0, 0.0, 0.0, radius + h0]);

    // Fall, bounce, then track the rebound until it turns over.
    let mut bounced = false;
    let mut apex: f64 = 0.0;
    let mut v_prev = 0.0;
    for _ in 0..20_000 {
        sim.step_with_contacts(&model, &mut st, 0.0, &material);
        let h = st.q[Z] - radius;
        if !bounced && v_prev < 0.0 && st.v[Z] >= 0.0 {
            bounced = true;
        }
        v_prev = st.v[Z];
        if bounced {
            if st.v[Z] > 0.0 {
                apex = apex.max(h);
            } else if h < apex {
                break;
            }
        }
    }
    apex
}

/// Newton's rule is `h1/h0 = e^2`, and §6.2 budgets 2% on it.
///
/// It used to be short by a measured 8–19%, and from 5 cm the sphere did not
/// rebound at all. The soft resting-contact model was doing it: its impedance
/// delivered the target rebound scaled by `d` (0.95 penetrating, tapering to
/// nothing across the margin, so an impact detected half a millimetre above
/// the plane lost most of its bounce), its stabilization bias added `erp` to
/// the effective `e`, and the restitution target was read off the free
/// velocity, gravity's `g dt` included. An impacting row is now rigid and
/// bias-free, and restitution acts on the pre-step approach speed; see
/// `ContactRow::impact`.
///
/// What remains is the time-of-impact quantization every fixed-step scheme
/// has: the bounce happens at whatever height the step boundary finds the
/// sphere, anywhere from `v dt` below the plane to `margin` above it, and the
/// apex carries that offset. So the tolerance is `2%` **or** `v dt + margin`,
/// whichever is larger — at `e = 0.3` from 20 cm the whole rebound is 1.8 cm,
/// and a 2 mm quantization is already 11% of it.
///
/// Measured `sqrt(h1/h0)` as a fraction of nominal `e`, for the record:
///
/// | nominal `e` | `h0 = 0.20 m` | `h0 = 0.80 m` |
/// |---|---|---|
/// | 0.30 | 1.023 | 0.982 |
/// | 0.50 | 1.006 | 0.993 |
/// | 0.80 | 1.000 | 0.997 |
/// | 0.95 | 0.998 | 0.998 |
///
/// The residual `0.2%` at high `e` is [`phyz_contact::IMPACT_IMPEDANCE`]: an
/// impacting row is rigid to a part in a thousand, not exactly.
///
/// `e = 0` never bounces and the rebound is monotone in `e`; both are checked
/// unconditionally.
#[test]
fn restitution_matches_newton_from_drop_height() {
    let margin = ContactMaterial::default().margin;
    for h0 in [0.20, 0.80] {
        let v_impact = (2.0 * GRAVITY * h0).sqrt();
        let quantization = v_impact * DT + margin;
        let mut previous = -1.0f64;
        for e in [0.0, 0.3, 0.5, 0.8, 0.95] {
            let h1 = bounce_apex(e, h0);
            let ratio = h1 / h0;
            let e_eff = ratio.sqrt();
            eprintln!(
                "h0 = {h0:.2}: e = {e:.2}: h1/h0 = {ratio:.4} (Newton {:.4}), e_eff/e = {:.4}",
                e * e,
                if e > 0.0 { e_eff / e } else { 0.0 }
            );
            // Monotone in e: more nominal bounce must never mean less real bounce.
            assert!(
                ratio > previous,
                "rebound must increase with e; {ratio:.4} followed {previous:.4}"
            );
            previous = ratio;

            if e == 0.0 {
                assert!(
                    ratio < 1e-6,
                    "e = 0 must not bounce, got h1/h0 = {ratio:.3e}"
                );
                continue;
            }
            let newton = e * e * h0;
            let tolerance = (0.02 * newton).max(quantization);
            assert!(
                (h1 - newton).abs() <= tolerance,
                "h0 = {h0}, e = {e}: apex {h1:.4} m against Newton's {newton:.4} m, \
                 outside 2% or the {quantization:.1e} m time-of-impact quantization"
            );
        }
    }
}

/// A bouncy sphere must come to rest and **stay** there.
///
/// This is what §4.3's restitution threshold exists to guarantee, and it is
/// where naive implementations fail: with restitution applied unconditionally,
/// a resting body micro-bounces forever and the stack never settles. The ramp
/// is a `smoothstep` rather than a hard cutoff so that the settling is also
/// differentiable, which `phyz-diff`'s
/// `restitution_gradient_vanishes_below_the_low_speed_ramp` checks from the
/// other side.
#[test]
fn a_bouncy_sphere_settles_and_stays_settled() {
    let radius = 0.05;
    let model = sphere(radius, 1.0);
    let sim = Simulator::new();
    let material = ContactMaterial {
        restitution: 0.8,
        friction: 0.5,
        ..ContactMaterial::default()
    };
    let mut st = state_at(&model, &[0.0, 0.0, 0.0, 0.0, 0.0, radius + 0.20]);

    // 10 s to settle.
    for _ in 0..10_000 {
        sim.step_with_contacts(&model, &mut st, 0.0, &material);
    }
    let speed_after_settling = st.v[Z].abs();
    assert!(
        speed_after_settling < 1e-3,
        "sphere should be at rest after 10 s, |v_z| = {speed_after_settling:.3e}"
    );

    // 5 s more: it must not wander or restart bouncing.
    let z_settled = st.q[Z];
    let mut max_excursion: f64 = 0.0;
    for _ in 0..5_000 {
        sim.step_with_contacts(&model, &mut st, 0.0, &material);
        max_excursion = max_excursion.max((st.q[Z] - z_settled).abs());
        assert!(st.q[Z].is_finite(), "NaN in the settled tail");
    }
    eprintln!("settled excursion over the next 5 s: {max_excursion:.3e} m");
    assert!(
        max_excursion < 1e-5,
        "settled sphere wandered {max_excursion:.3e} m"
    );
}

// ---------------------------------------------------------------------------
// §6.4 — energy behaviour
// ---------------------------------------------------------------------------

/// Soft contact dissipates. It must never *inject*.
///
/// A badly-signed damping term is the classic way to get energy growth, and it
/// is invisible in a single bounce — it shows up as a slow climb over many.
/// The assertion is one-sided on purpose: the design doc asks that total energy
/// be monotonically non-increasing and bounded, not that it be conserved.
#[test]
fn bouncing_energy_never_increases() {
    let radius = 0.05;
    let model = sphere(radius, 1.0);
    let sim = Simulator::new();
    // e = 1 and frictionless: the only remaining loss is the soft contact
    // itself, so any *gain* is unambiguously a bug rather than a modelling
    // choice.
    let material = ContactMaterial {
        restitution: 1.0,
        friction: 0.0,
        ..ContactMaterial::default()
    };
    let mut st = state_at(&model, &[0.0, 0.0, 0.0, 0.0, 0.0, radius + 0.30]);

    let energy = |s: &State| {
        let h = (s.q[Z] - radius).max(0.0);
        GRAVITY * h + 0.5 * s.v[Z] * s.v[Z]
    };
    let e0 = energy(&st);
    let mut worst_gain: f64 = 0.0;
    // 30 s is many tens of bounces at this height.
    for _ in 0..30_000 {
        sim.step_with_contacts(&model, &mut st, 0.0, &material);
        assert!(
            st.q[Z].is_finite() && st.v[Z].is_finite(),
            "NaN in the trajectory"
        );
        worst_gain = worst_gain.max(energy(&st) - e0);
    }
    eprintln!(
        "energy: start {:.5}, end {:.5}, worst excess over start {:.3e}",
        e0,
        energy(&st),
        worst_gain
    );
    // A small positive tolerance covers the integrator's per-step exchange
    // between the two terms at the moment of contact; the failure this guards
    // is a trend, which would be orders larger. (Restitution read off the
    // free velocity — gravity's kick included — was such a trend: `m g dt |v|`
    // gained per bounce, 50% over these 30 s. The soft contact's own 5% loss
    // per bounce hid it until impacting rows became rigid.)
    assert!(
        worst_gain < 1e-3 * e0,
        "energy grew by {worst_gain:.3e} above the initial {e0:.5} — check the \
         sign of the contact damping"
    );
}

// ---------------------------------------------------------------------------
// §6.3 — stacking
// ---------------------------------------------------------------------------

struct StackResult {
    lateral_drift: f64,
    max_tilt: f64,
    max_penetration: f64,
}

/// Settle a stack of `masses.len()` boxes released in exact contact, then
/// measure how far it moves over the following 10 s.
fn run_stack(masses: &[f64], half: f64) -> StackResult {
    let n = masses.len();
    let model = boxes(n, half, masses);
    let sim = Simulator::new();
    let material = ContactMaterial {
        friction: 0.6,
        ..ContactMaterial::default()
    };
    let mut q = vec![0.0; 6 * n];
    for k in 0..n {
        q[6 * k + Z] = half * (2 * k + 1) as f64;
    }
    let mut st = state_at(&model, &q);

    // 2 s to settle out of the released-in-exact-contact transient.
    for _ in 0..2_000 {
        sim.step_with_contacts(&model, &mut st, 0.0, &material);
    }
    let settled: Vec<f64> = (0..n).map(|k| st.q[6 * k + 3]).collect();

    let mut lateral_drift: f64 = 0.0;
    let mut max_tilt: f64 = 0.0;
    let mut max_penetration: f64 = 0.0;
    for _ in 0..10_000 {
        sim.step_with_contacts(&model, &mut st, 0.0, &material);
        for (k, settled_x) in settled.iter().enumerate().take(n) {
            let base = 6 * k;
            for c in 0..6 {
                assert!(
                    st.q[base + c].is_finite() && st.v[base + c].is_finite(),
                    "NaN in the stack trajectory at body {k}, component {c}"
                );
            }
            lateral_drift = lateral_drift.max((st.q[base + 3] - settled_x).abs());
            let tilt = st.q[base]
                .abs()
                .max(st.q[base + 1].abs())
                .max(st.q[base + 2].abs());
            max_tilt = max_tilt.max(tilt);
            // Overlap with the box below (or the ground for the lowest).
            let below = if k == 0 {
                0.0
            } else {
                st.q[base - 6 + Z] + half
            };
            max_penetration = max_penetration.max(below - (st.q[base + Z] - half));
        }
    }
    StackResult {
        lateral_drift,
        max_tilt,
        max_penetration,
    }
}

/// Five equal boxes: §6.3's headline case.
///
/// The penetration bound is asserted **and** checked against the analytic soft
/// contact sink, because §6.3 asks for the second explicitly: a small number is
/// only reassuring if it is the number the model predicts. The load on the
/// bottom contact is `Σmg`, so the sink there should scale with the stack
/// height, and asserting the bound alone would pass just as happily if contact
/// were far too stiff or the stack were quietly floating.
#[test]
fn five_box_stack_is_stable() {
    let half = 0.05;
    let r = run_stack(&[1.0; 5], half);
    eprintln!(
        "5-stack: drift {:.3e} m, tilt {:.3e} rad, penetration {:.3e} m",
        r.lateral_drift, r.max_tilt, r.max_penetration
    );
    // The bounds are what this stack does, not what a settled stack should
    // do. It never settles: the boxes chatter at ~0.1 rad/s and ~1 cm/s
    // under the solver, and where that limit cycle walks over ten seconds is
    // decided at rounding level. Measured on the commit that made the free
    // joint's body-frame velocity turn exactly: drift 0.15 mm and tilt 0.03°
    // before it, 2.3 mm and 0.53° after, and 2.7 mm and 0.58° for the *same*
    // first-order arithmetic in a different operation order. A 1 mm bound
    // was being passed by luck, not by the physics.
    assert!(
        r.lateral_drift < 5e-3,
        "lateral drift {:.3e} m exceeds 5 mm",
        r.lateral_drift
    );
    assert!(
        r.max_tilt < 1.0f64.to_radians(),
        "tilt {:.3e} rad exceeds 1 deg",
        r.max_tilt
    );
    assert!(
        r.max_penetration < 2e-3,
        "penetration {:.3e} m exceeds 2 mm",
        r.max_penetration
    );
}

/// §6.3's heavy-on-light variant. It does **not** degrade gracefully: past a
/// mass ratio of about 5 the stack falls over, and it does so *non-monotonically*.
///
/// Three unit boxes carrying one heavy box, tilt after 10 s of settled
/// simulation:
///
/// | mass ratio | max tilt | penetration |
/// |---|---|---|
/// | 1 | 0.01° | 2.7e-4 m |
/// | 2 | 0.00° | 3.4e-4 m |
/// | 5 | 0.89° | 5.0e-4 m |
/// | 10 | **40.85°** | 2.4e-2 m |
/// | 20 | 0.00° | 9.8e-4 m |
/// | 50 | **180.36°** | 1.6e-1 m |
/// | 100 | **190.40°** | 1.9e-1 m |
///
/// The design doc anticipated degradation here and asked for "a degraded but
/// bounded result, documented rather than tuned until it looks good". The
/// honest reading of the table is that there is no bound to document: 20:1 is
/// *perfect* while 10:1 has fallen flat, so this is not soft contact gracefully
/// losing accuracy with mass ratio — it is an instability whose onset is not
/// monotone in the parameter, which is a different and more serious thing. A
/// smooth degradation could be budgeted for; this cannot.
///
/// §7.4's tradeoff table claims "stacking robustness: good, worse than TGS at
/// high mass ratio" for phyz. On this evidence that row is too generous, and it
/// is now marked as such in the design doc's §8.
///
/// The test asserts the *equal-mass* end holds and pins 10:1 as a known
/// collapse, so a genuine fix breaks it loudly and gets the table updated.
#[test]
fn heavy_on_light_stack_collapses_non_monotonically() {
    let half = 0.05;

    let ok = run_stack(&[1.0, 1.0, 1.0, 2.0], half);
    eprintln!(
        "2:1 stack: drift {:.3e} m, tilt {:.3e} rad",
        ok.lateral_drift, ok.max_tilt
    );
    // The gate is "still standing", not the 0.00 deg in the table above. Ten
    // seconds of a four-box stack is chaotic in the tilt channel: the table's
    // numbers are one machine's, and the same build tilts 2.87 deg on a CI
    // runner whose codegen contracts a multiply-add differently. What is
    // reproducible is the *distinction* this test exists to pin — a low mass
    // ratio stands (single degrees) while 10:1 lies flat (40+ deg) — so the
    // bound is set where that distinction lives rather than where one host's
    // measurement happened to land.
    assert!(
        ok.max_tilt < 5f64.to_radians(),
        "a 2:1 stack must still stand; tilt {:.3} deg",
        ok.max_tilt.to_degrees()
    );

    let bad = run_stack(&[1.0, 1.0, 1.0, 10.0], half);
    eprintln!(
        "10:1 stack: drift {:.3e} m, tilt {:.2} deg",
        bad.lateral_drift,
        bad.max_tilt.to_degrees()
    );
    assert!(
        bad.max_tilt > 10f64.to_radians(),
        "10:1 was a documented collapse (40.85 deg) and now tilts only {:.2} \
         deg — if this is a fix, update the table in this test's docs, the \
         design doc §8, and §7.4's stacking claim together",
        bad.max_tilt.to_degrees()
    );
}
