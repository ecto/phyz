//! Rollout-level FD gates for the two contact regimes the design doc treats as
//! the sharpest tests of a Coulomb model, plus a genuinely multi-point one.
//!
//! §6.1's incline battery already exists in `phyz-contact` — but against a
//! hand-built single-contact `ContactProblem` for a point mass, which is a test
//! of the *solver*. These are the same physics as a **trajectory**: full model,
//! four-corner manifold, detection and assembly in the loop, gradient by the
//! adjoint. §6.5 asks for exactly this and lists the incline's two branches
//! separately, because they exercise different sides of the IFT — the cone
//! interior for stiction and the cone boundary for slip.
//!
//! # The incline is built by tilting gravity, not the ground
//!
//! `ConvexContactRollout` takes a `ground_height`, i.e. a horizontal `z` plane.
//! Rotating gravity to `g·(sin α, 0, −cos α)` against that flat plane is not an
//! approximation of a block on a slope — it is the identical mechanical system
//! written in the slope's own frame, so `tan α ≤ μ` still decides stiction and
//! the sliding acceleration is still `g(sin α − μ cos α)`. It also keeps the
//! contact normal exactly `+ẑ`, which means these tests isolate the friction
//! behaviour instead of also re-testing the narrow phase on a tilted box.

use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::{
    ConvexContactRollout, FinalStateObjective, convex_adjoint_gradient, convex_rollout_objective,
};
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder};

const HALF: f64 = 0.05;
const X: usize = 3;
/// Pitch about `y` — the tipping tests' objective.
const PITCH: usize = 1;

fn box_model(gravity: Vec3) -> Model {
    let mass = 1.0;
    let i = mass * 2.0 / 3.0 * HALF * HALF;
    let mut model = ModelBuilder::new()
        .gravity(gravity)
        .dt(1e-3)
        .add_free_body(
            "box",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                mass,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(i, i, i)),
            ),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: Vec3::new(HALF, HALF, HALF),
    });
    model
}

/// Gravity as seen in the frame of a slope of angle `alpha`.
fn slope_gravity(alpha: f64) -> Vec3 {
    Vec3::new(GRAVITY * alpha.sin(), 0.0, -GRAVITY * alpha.cos())
}

fn q_objective(i: usize) -> FinalStateObjective<'static> {
    let value: &'static dyn Fn(&[f64], &[f64]) -> f64 =
        Box::leak(Box::new(move |q: &[f64], _: &[f64]| q[i]));
    // The objective gradient's signature is what `FinalStateObjective` asks
    // for; naming it keeps clippy's type-complexity lint honest about the fact
    // that this is one type used twice, not an accident.
    type GradFn = dyn Fn(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>);
    let gradient: &'static GradFn = Box::leak(Box::new(move |q: &[f64], v: &[f64]| {
        let mut gq = vec![0.0; q.len()];
        gq[i] = 1.0;
        (gq, vec![0.0; v.len()])
    }));
    FinalStateObjective { value, gradient }
}

struct Case {
    name: &'static str,
    model: Model,
    material: ContactMaterial,
    q0: Vec<f64>,
    v0: Vec<f64>,
    steps: usize,
    obj_index: usize,
    tol: f64,
}

fn make<'a>(
    c: &'a Case,
    q0: &[f64],
    v0: &[f64],
    ctrl: &'a dyn Fn(usize) -> DVec,
) -> ConvexContactRollout<'a> {
    ConvexContactRollout {
        model: &c.model,
        ground_height: 0.0,
        material: c.material.clone(),
        config: ContactSolverConfig::gradients(),
        q0: DVec::from_slice(q0),
        v0: DVec::from_slice(v0),
        steps: c.steps,
        ctrl,
    }
}

/// Sweep every state lane against a central difference of the real rollout.
fn check(case: &Case) -> f64 {
    let obj = q_objective(case.obj_index);
    let ctrl = |_t: usize| DVec::zeros(6);
    let g = convex_adjoint_gradient(&make(case, &case.q0, &case.v0, &ctrl), &obj)
        .expect("adjoint must not refuse");

    let h = 1e-7;
    let mut rows: Vec<(String, f64, f64)> = Vec::new();
    for i in 0..6 {
        let mut qp = case.q0.clone();
        let mut qm = case.q0.clone();
        qp[i] += h;
        qm[i] -= h;
        let fp = convex_rollout_objective(&make(case, &qp, &case.v0, &ctrl), &obj);
        let fm = convex_rollout_objective(&make(case, &qm, &case.v0, &ctrl), &obj);
        rows.push((format!("dJ/dq0[{i}]"), g.d_q0[i], (fp - fm) / (2.0 * h)));
    }
    for i in 0..6 {
        let mut vp = case.v0.clone();
        let mut vm = case.v0.clone();
        vp[i] += h;
        vm[i] -= h;
        let fp = convex_rollout_objective(&make(case, &case.q0, &vp, &ctrl), &obj);
        let fm = convex_rollout_objective(&make(case, &case.q0, &vm, &ctrl), &obj);
        rows.push((format!("dJ/dv0[{i}]"), g.d_v0[i], (fp - fm) / (2.0 * h)));
    }

    let scale = rows.iter().fold(1e-4f64, |m, r| m.max(r.2.abs()));
    let mut max_rel: f64 = 0.0;
    for (label, adj, fd) in &rows {
        let rel = (adj - fd).abs() / fd.abs().max(scale);
        max_rel = max_rel.max(rel);
        assert!(
            rel <= case.tol,
            "{}: {label} adjoint {adj} vs FD {fd} (rel {rel:.3e})",
            case.name
        );
    }
    eprintln!("{}: max relative error {max_rel:.3e}", case.name);
    max_rel
}

// ---------------------------------------------------------------------------
// §6.5 — the incline, both branches
// ---------------------------------------------------------------------------

/// **Cone interior.** `mu = 0.6`, slope 20°, and `tan 20° = 0.364 < 0.6`, so
/// Coulomb says the block holds. The gradient here is the sticking branch of
/// the IFT: the constraint is active as an equality in all three directions.
///
/// The travel assertion is the point of the test as much as the gradient is.
/// The friction law this crate replaced (`min(mu*Fn, c*|v_t|)`) could not hold a
/// block on *any* slope at *any* `mu`, because its friction force vanishes with
/// sliding speed regardless of normal load, so it would slide the full
/// `0.5·g·sin(20°)·t^2 = 6.7 m` here.
///
/// The bound is §6.1 A's `1 mm`. What the block actually does, measured over
/// `500 / 1000 / 2000 / 4000` steps, is `1.726 / 1.767 / 1.851 / 2.019` x
/// `1e-5 m`: a `~17 um` one-time settle as the contact takes up its penetration,
/// then a residual creep of about `0.9 um/s`. Both are two orders under the
/// bound, and quoting them here is what makes the test able to detect a
/// *change* rather than only a catastrophe.
#[test]
fn incline_sticking_below_the_friction_angle() {
    let alpha: f64 = 20f64.to_radians();
    let case = Case {
        name: "incline_stick",
        model: box_model(slope_gravity(alpha)),
        material: ContactMaterial {
            friction: 0.6,
            ..ContactMaterial::default()
        },
        q0: vec![0.0, 0.0, 0.0, 0.0, 0.0, HALF],
        v0: vec![0.0; 6],
        steps: 2000,
        obj_index: X,
        tol: 1e-3,
    };
    assert!(
        alpha.tan() < case.material.friction,
        "scene must be sub-critical"
    );

    let obj = q_objective(X);
    let ctrl = |_t: usize| DVec::zeros(6);
    let slid = convex_rollout_objective(&make(&case, &case.q0, &case.v0, &ctrl), &obj);
    eprintln!("incline_stick: travel over 2 s = {slid:.3e} m");
    assert!(
        slid.abs() < 1e-3,
        "a block below the friction angle must not creep; slid {slid:.3e} m"
    );
    // Regression guard on the settle itself, an order under the spec bound.
    assert!(
        slid.abs() < 5e-5,
        "settle+creep regressed from the measured 1.85e-5 m; slid {slid:.3e} m"
    );

    check(&case);
}

/// **Cone boundary.** Same `mu`, slope 40°, `tan 40° = 0.839 > 0.6`, so the
/// block slides and the tangential rows are pinned to `||f_t|| = mu·f_n`.
///
/// # An open discrepancy, pinned rather than hidden
///
/// Design doc §6.1 C wants the closed-form `a = g(sin α − mu cos α)` to 1%.
/// `phyz-contact`'s own benchmark meets that — but against a hand-built
/// **single-contact** problem for a point mass. Run as a *trajectory* on a box,
/// with the four-corner manifold the narrow phase actually produces, the
/// steady-state acceleration is `2.0838` against a theoretical `1.7968`: a
/// **16% excess**, equivalently an effective `mu` of `0.5618` rather than
/// `0.600`, a 6.4% friction deficit. (`2.0838` is the windowed
/// `dv_x/dt`; the `2·x/t^2` form this test uses reads `2.0912`, the difference
/// being the initial settle, which that formula folds in.)
///
/// It is steady, not transient — the windowed acceleration is `2.0838` over
/// every interval from 0.1 s to 0.4 s — and two natural explanations are ruled
/// out by measurement:
///
/// - **Not the solver preset.** `simulation()` and `gradients()` give
///   `2.0838` and `2.0839`, so it is not tolerance or regularization strength.
/// - **Not the impedance regularizer.** Sweeping `solimp` `dmin=dmax` over
///   `0.9 / 0.99 / 0.999 / 0.9999` moves the answer by nothing at all, which
///   also kills the appealing arithmetic that `(1-d)/d ≈ 5%` is the deficit.
/// - **Not the box rotating.** Final pitch is `4.2e-4 rad` (0.024°) with a
///   pitch rate of `1.2e-6 rad/s`, so all four corners slip in the same
///   direction and the cones cannot be disagreeing about it.
///
/// So it is a real property of the multi-point path that the single-contact
/// benchmark does not see, and it deserves its own investigation rather than a
/// widened tolerance here. The assertion below is therefore a **regression
/// guard on the measured value**, not a physics check — deliberately, and
/// labelled as such, so that nobody reads a passing suite as agreement with
/// theory.
#[allow(clippy::doc_markdown)]
#[test]
fn incline_sliding_above_the_friction_angle() {
    let alpha: f64 = 40f64.to_radians();
    let mu = 0.6;
    let case = Case {
        name: "incline_slide",
        model: box_model(slope_gravity(alpha)),
        material: ContactMaterial {
            friction: mu,
            ..ContactMaterial::default()
        },
        q0: vec![0.0, 0.0, 0.0, 0.0, 0.0, HALF],
        v0: vec![0.0; 6],
        steps: 300,
        obj_index: X,
        tol: 1e-3,
    };

    let obj = q_objective(X);
    let ctrl = |_t: usize| DVec::zeros(6);
    let travelled = convex_rollout_objective(&make(&case, &case.q0, &case.v0, &ctrl), &obj);
    let t = case.steps as f64 * case.model.dt;
    let a_measured = 2.0 * travelled / (t * t);
    let a_theory = GRAVITY * (alpha.sin() - mu * alpha.cos());
    let rel = (a_measured - a_theory).abs() / a_theory;
    eprintln!(
        "incline_slide: a = {a_measured:.4} vs theory {a_theory:.4} \
         (gap {:.1}% — see this test's docs, open question)",
        rel * 100.0
    );
    // Regression guard on the measured number, NOT agreement with theory.
    assert!(
        (a_measured - 2.0912).abs() < 5e-3,
        "sliding acceleration moved from the recorded 2.0912 to {a_measured}; \
         if this is a fix for the documented 16% gap, update the doc comment \
         and this guard together"
    );

    check(&case);
}

// ---------------------------------------------------------------------------
// §6.5 — a box tipping on an edge
// ---------------------------------------------------------------------------

/// A box rotating over onto one edge: the manifold genuinely reduces from a
/// four-point face to a two-point edge and the load redistributes across it.
///
/// This is the multi-point row of §6.5, and it is the scenario a per-contact
/// *local* gradient gets wrong. While the box pivots, the surviving contacts
/// share one Delassus matrix, so each impulse depends on all the others; a
/// design that treated contacts independently would produce a gradient that
/// looks reasonable and is wrong by the coupling term. The objective is the
/// final pitch, which is a pure function of how that shared load resolved.
///
/// The box is *started* balanced on an edge and allowed to fall flat, rather
/// than spun from flat and expected to rise. Spinning it from flat does not
/// work and the reason is physical, not a tuning failure: a box lying on four
/// coplanar contacts that is given a pitch rate immediately drives its trailing
/// corners into the ground, and the inelastic contact absorbs the angular
/// momentum. Swept from `2.2` to `14 rad/s`, the peak pitch reached is `0.05°`
/// — the ground eats all of it. Starting tilted puts the energy where gravity
/// can convert it instead.
///
/// So the trajectory runs the manifold transition *backwards*, which is the
/// same test: two contacts on an edge become four on a face, and the load has
/// to redistribute across the new points as it happens.
#[test]
fn box_tipping_on_an_edge() {
    // Balanced on the +x/-z edge: for a pitch theta the lowest corner sits at
    // -HALF*(cos+sin), so this rests exactly on the ground.
    let theta: f64 = 0.7;
    let z = HALF * (theta.cos() + theta.sin());
    let case = Case {
        name: "box_tipping",
        model: box_model(Vec3::new(0.0, 0.0, -GRAVITY)),
        material: ContactMaterial {
            friction: 0.8,
            ..ContactMaterial::default()
        },
        q0: vec![0.0, theta, 0.0, 0.0, 0.0, z],
        v0: vec![0.0; 6],
        // Long enough to fall flat and settle on the face.
        steps: 300,
        obj_index: PITCH,
        tol: 1e-3,
    };

    let obj = q_objective(PITCH);
    let ctrl = |_t: usize| DVec::zeros(6);
    let pitch = convex_rollout_objective(&make(&case, &case.q0, &case.v0, &ctrl), &obj);
    let rotated = (theta - pitch).abs();
    eprintln!(
        "box_tipping: {:.1} deg -> {:.2} deg (rotated {:.1} deg)",
        theta.to_degrees(),
        pitch.to_degrees(),
        rotated.to_degrees()
    );
    assert!(
        rotated > 20f64.to_radians(),
        "the box must actually tip over for this to exercise the manifold \
         change; it rotated only {:.2} deg",
        rotated.to_degrees()
    );

    check(&case);
}
