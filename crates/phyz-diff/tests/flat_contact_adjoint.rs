//! Flat (box) contact adjoint gates at the K1 rig's solver settings.
//!
//! ipse's K1 `fd_check --flat` reported `dJ/dπ ~ 1e10–1e17` through box feet
//! and finite differences that never converged. The root cause turned out not
//! to be the contact at all (see `docs/design/flat-contact-adjoint-rootcause.md`):
//! it is the explicit `tanh` dry joint friction on light joints, which makes
//! the step map locally expanding. These tests pin both halves:
//!
//! - A single box face on the plane — four coplanar corner contacts, rank-
//!   deficient normal block — at `dt = 2 ms`, `timeconst = 0.02`,
//!   `friction = 0.8`, `tol = 1e-10`: resting, pressed-and-rocked by an off-
//!   centre load, and driven into a sustained slide. Adjoint vs central FD on
//!   mass and the COM offset, `rel < 1e-3`.
//! - The same box carrying a light hinged link with `friction_loss`: under
//!   [`FrictionLossIntegration::Explicit`] a `1e-9` perturbation blows up by
//!   orders of magnitude (no gradient exists at FD resolution); under
//!   [`FrictionLossIntegration::Implicit`] the adjoint matches FD.

use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::{
    ConvexContactRollout, FinalStateObjective, convex_adjoint_gradient, convex_rollout_objective,
    inertia_params,
};
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{FrictionLossIntegration, Geometry, Model, ModelBuilder};

/// Foot-like box: long in x, low in z.
const HX: f64 = 0.10;
const HY: f64 = 0.05;
const HZ: f64 = 0.02;
const DT: f64 = 2e-3;

fn box_inertia(mass: f64, com: Vec3) -> SpatialInertia {
    let (a, b, c) = (2.0 * HX, 2.0 * HY, 2.0 * HZ);
    let ixx = mass / 12.0 * (b * b + c * c);
    let iyy = mass / 12.0 * (a * a + c * c);
    let izz = mass / 12.0 * (a * a + b * b);
    SpatialInertia::new(mass, com, Mat3::from_diagonal(&Vec3::new(ixx, iyy, izz)))
}

/// A free box (body 0), optionally carrying a light hinged link (body 1) on
/// its top face — the "hand on an arm" whose dry friction breaks explicit
/// integration.
fn model(with_link: Option<FrictionLossIntegration>) -> Model {
    let mut b = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(DT)
        .add_free_body(
            "foot",
            -1,
            SpatialTransform::identity(),
            box_inertia(1.0, Vec3::new(0.02, 0.0, 0.0)),
        );
    if with_link.is_some() {
        b = b.add_revolute_body(
            "link",
            0,
            SpatialTransform::from_translation(Vec3::new(0.0, 0.0, HZ)),
            SpatialInertia::new(
                0.1,
                Vec3::new(0.03, 0.0, 0.04),
                Mat3::from_diagonal(&Vec3::new(1e-4, 1e-4, 1e-5)),
            ),
        );
    }
    let mut m = b.build();
    m.bodies[0].geometry = Some(Geometry::Box {
        half_extents: Vec3::new(HX, HY, HZ),
    });
    if let Some(mode) = with_link {
        let j = &mut m.joints[1];
        j.axis = Vec3::new(0.0, 1.0, 0.0);
        j.damping = 0.05;
        j.armature = 1e-3;
        j.friction_loss = 0.05;
        j.friction_loss_integration = mode;
    }
    m
}

fn material() -> ContactMaterial {
    let mut m = ContactMaterial {
        friction: 0.8,
        ..ContactMaterial::default()
    };
    m.solref.timeconst = 0.02;
    m
}

fn with_param(model: &Model, body: usize, k: usize, value: f64) -> Model {
    let mut p = inertia_params(&model.bodies[body].inertia);
    p[k] = value;
    let mut m = model.clone();
    m.bodies[body].inertia = SpatialInertia::new(
        p[0],
        Vec3::new(p[1], p[2], p[3]),
        Mat3::new(p[4], p[7], p[8], p[7], p[5], p[9], p[8], p[9], p[6]),
    );
    m
}

struct Case<'a> {
    model: &'a Model,
    q0: Vec<f64>,
    v0: Vec<f64>,
    steps: usize,
    ctrl: &'a dyn Fn(usize) -> DVec,
    config: ContactSolverConfig,
}

fn rollout<'a>(c: &Case<'a>, m: &'a Model) -> ConvexContactRollout<'a> {
    ConvexContactRollout {
        model: m,
        ground_height: 0.0,
        material: material(),
        config: c.config,
        q0: DVec::from_slice(&c.q0),
        v0: DVec::from_slice(&c.v0),
        steps: c.steps,
        ctrl: c.ctrl,
    }
}

/// A fixed, dense final-state objective, so every channel reaches `J`.
fn with_objective<R>(nq: usize, nv: usize, f: impl FnOnce(&FinalStateObjective) -> R) -> R {
    let wq: Vec<f64> = (0..nq).map(|i| 1.0 + 0.37 * i as f64).collect();
    let wv: Vec<f64> = (0..nv).map(|i| 0.1 - 0.013 * i as f64).collect();
    let (wq2, wv2) = (wq.clone(), wv.clone());
    let value = move |q: &[f64], v: &[f64]| -> f64 {
        q.iter().zip(&wq).map(|(a, b)| a * b).sum::<f64>()
            + v.iter().zip(&wv).map(|(a, b)| a * b).sum::<f64>()
    };
    let gradient = move |_q: &[f64], _v: &[f64]| (wq2.clone(), wv2.clone());
    f(&FinalStateObjective {
        value: &value,
        gradient: &gradient,
    })
}

/// Adjoint vs central FD on `(body, k)` inertia parameters; returns max rel.
fn check(name: &str, c: &Case, params: &[(usize, usize, &str)], tol: f64) -> f64 {
    with_objective(c.model.nq, c.model.nv, |obj| {
        let g = convex_adjoint_gradient(&rollout(c, c.model), obj).expect("adjoint refused");
        let mut worst: f64 = 0.0;
        // Denominator floored at the case's largest gradient (this crate's
        // convention, see convex_contact_adjoint.rs): a lane that is ~0 by
        // symmetry (com_z at rest) is judged against the case's scale, not
        // against FD round-off a thousand times below it.
        let floor = params
            .iter()
            .map(|&(b, k, _)| g.d_inertia[b][k].abs())
            .fold(1e-9, f64::max);
        for &(body, k, label) in params {
            let p0 = inertia_params(&c.model.bodies[body].inertia)[k];
            // h = 1e-3 of the parameter's scale (COM offsets on a 0.1 m
            // scale). The contact solve stops at its 1e-10 tolerance, so the
            // primal is piecewise smooth with ~3e-11 steps in J wherever an
            // iteration count changes (measured on the resting box: dJ/dm on
            // a 1e-6 grid alternates between -3.46e-4, -3.74e-4, -4.03e-4).
            // A step must be large enough to average those out.
            let h = 1e-3 * p0.abs().max(0.1);
            let mp = with_param(c.model, body, k, p0 + h);
            let mm = with_param(c.model, body, k, p0 - h);
            let fd = (convex_rollout_objective(&rollout(c, &mp), obj)
                - convex_rollout_objective(&rollout(c, &mm), obj))
                / (2.0 * h);
            let adj = g.d_inertia[body][k];
            let rel = (adj - fd).abs() / fd.abs().max(adj.abs()).max(floor);
            eprintln!("{name}: d/d{label} adjoint {adj:.9e} fd {fd:.9e} rel {rel:.2e}");
            worst = worst.max(rel);
            assert!(
                rel < tol,
                "{name}: d/d{label} adjoint {adj} vs FD {fd} (rel {rel:.3e})"
            );
        }
        worst
    })
}

fn rest_q(nq: usize) -> Vec<f64> {
    let mut q = vec![0.0; nq];
    q[5] = HZ;
    q
}

const BOX_PARAMS: [(usize, usize, &str); 3] = [(0, 0, "mass"), (0, 1, "com_x"), (0, 3, "com_z")];

#[test]
fn box_face_at_rest() {
    let m = model(None);
    let ctrl = |_t: usize| DVec::zeros(6);
    let c = Case {
        model: &m,
        q0: rest_q(m.nq),
        v0: vec![0.0; 6],
        steps: 150,
        ctrl: &ctrl,
        config: ContactSolverConfig::simulation(),
    };
    check("rest", &c, &BOX_PARAMS, 1e-3);
}

/// Pressed down and rocked by an off-centre pitching load: the four corners
/// carry very different normal impulses, the toe pair near the edge of
/// unloading.
#[test]
fn box_face_pressed_and_rocked() {
    let m = model(None);
    let ctrl = |t: usize| {
        let mut u = DVec::zeros(6);
        u[5] = -20.0;
        u[1] = 0.15 * (t as f64 * DT * 2.0 * std::f64::consts::PI * 3.0).sin();
        u
    };
    let c = Case {
        model: &m,
        q0: rest_q(m.nq),
        v0: vec![0.0; 6],
        steps: 150,
        ctrl: &ctrl,
        // A loaded four-corner face has 8 tangential rows on 3 in-plane DOFs;
        // at the default `regularization = 1e-6` that 5-dim null space is
        // regularized by 1e-6 alone, the solve stalls at ~1e-8 against the
        // 1e-10 tolerance (spending the full 1000-iteration budget does not
        // help), and the adjoint refuses — or, at a loosened 1e-7, is 0.4%
        // off. `1e-4` of tangential compliance (an existing config knob, a
        // forward-physics setting the caller chooses) makes the block well
        // posed: converged at 1e-10, adjoint = FD to 4e-7.
        config: ContactSolverConfig {
            tolerance: 1e-10,
            regularization: 1e-4,
            ..ContactSolverConfig::simulation()
        },
    };
    check("press", &c, &BOX_PARAMS, 1e-3);
}

/// Driven slide: a tangential force above `μ·m·g` keeps all four corners
/// sliding for the whole window (no stick/slide transition inside it).
#[test]
fn box_face_driven_slide() {
    let m = model(None);
    let ctrl = |_t: usize| {
        let mut u = DVec::zeros(6);
        u[3] = 12.0;
        u
    };
    let c = Case {
        model: &m,
        q0: rest_q(m.nq),
        v0: vec![0.0, 0.0, 0.0, 0.3, 0.0, 0.0],
        steps: 100,
        ctrl: &ctrl,
        config: ContactSolverConfig::simulation(),
    };
    check("slide", &c, &BOX_PARAMS, 1e-3);
}

fn link_case_ctrl(t: usize) -> DVec {
    let mut u = DVec::zeros(7);
    // A small swing torque on the link, then nothing: the joint has to come
    // to rest against its own dry friction.
    if t < 20 {
        u[6] = 0.02;
    }
    u
}

/// One-step velocity gain `∂v'/∂v` of a fixed-base light hinge with dry
/// friction, at rest, under semi-implicit Euler with `phyz_rigid::aba` (the
/// integration every rollout in this crate performs).
fn hinge_step_gain(mode: FrictionLossIntegration) -> f64 {
    let mut m = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(DT)
        .add_revolute_body(
            "hand",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                0.1,
                Vec3::new(0.03, 0.0, 0.04),
                Mat3::from_diagonal(&Vec3::new(1e-4, 1e-4, 1e-5)),
            ),
        )
        .build();
    let j = &mut m.joints[0];
    j.axis = Vec3::new(0.0, 1.0, 0.0);
    j.damping = 0.05;
    j.armature = 1e-3;
    j.friction_loss = 0.05;
    j.friction_loss_integration = mode;
    let step = |v0: f64| {
        let mut s = m.default_state();
        s.v[0] = v0;
        s.v[0] + DT * phyz_rigid::aba(&m, &s)[0]
    };
    let h = 1e-7;
    (step(h) - step(-h)) / (2.0 * h)
}

/// The failure mode, pinned. Near rest the `tanh` friction is a viscous
/// damper of `friction_loss/s = 50 N·m·s`; on a joint with `~1.4e-3 kg·m²` of
/// effective inertia at `dt = 2 ms` an explicit damper's step gain is
/// `1 − dt·c/M ≈ −70`, so the step map is locally expanding around rest —
/// coupled across the K1's arms, hands and head, that is the chaos that
/// made its trajectories non-differentiable. The implicit treatment's gain
/// is `1/(1 + dt·c/M) ∈ (0, 1)`.
#[test]
fn explicit_friction_loss_on_light_joint_is_expanding() {
    let explicit = hinge_step_gain(FrictionLossIntegration::Explicit);
    let implicit = hinge_step_gain(FrictionLossIntegration::Implicit);
    eprintln!("one-step dv'/dv at rest: explicit {explicit:.4}, implicit {implicit:.4}");
    assert!(explicit < -10.0, "explicit gain {explicit}");
    assert!(implicit > 0.0 && implicit < 0.1, "implicit gain {implicit}");
}

/// Under the opt-in implicit treatment the same articulated case — box face
/// on the plane plus the light frictional link — has a gradient, and the
/// adjoint matches it.
#[test]
fn implicit_friction_loss_box_with_link_matches_fd() {
    let m = model(Some(FrictionLossIntegration::Implicit));
    let c = Case {
        model: &m,
        q0: rest_q(m.nq),
        v0: vec![0.0; m.nv],
        steps: 150,
        ctrl: &link_case_ctrl,
        config: ContactSolverConfig::simulation(),
    };
    check(
        "link",
        &c,
        &[
            (0, 0, "foot mass"),
            (0, 1, "foot com_x"),
            (1, 0, "link mass"),
            (1, 3, "link com_z"),
        ],
        1e-3,
    );
}

/// Default physics is untouched: nothing opts in unless asked.
#[test]
fn friction_loss_integration_defaults_to_explicit() {
    assert_eq!(
        phyz_model::Joint::default().friction_loss_integration,
        FrictionLossIntegration::Explicit
    );
}
