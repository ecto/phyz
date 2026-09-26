//! FD gates for the **material-parameter** channels of the convex adjoint:
//! `dJ/dmu` and `dJ/de`.
//!
//! These are design-doc §6.5 rows that could not exist before, because the
//! sensitivities were computed at solver level and never plumbed out to the
//! trajectory. They answer questions of a different kind from the state and
//! control channels: not "where should the robot push" but "how much grip do I
//! need", which is a *design* gradient rather than a control one.
//!
//! Each channel gets both a positive gate (it matches a central difference of
//! the real forward rollout) and a structural gate (it is exactly zero where
//! the physics says it must be). The structural ones matter as much as the
//! positive ones: a plausible-looking wrong implementation of `dJ/dmu` would
//! most likely report something small-but-nonzero for a block that never slips,
//! and only the zero test catches that.

use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::{
    ConvexContactRollout, FinalStateObjective, convex_adjoint_gradient, convex_rollout_objective,
};
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder};

const HALF: f64 = 0.05;
const RADIUS: f64 = 0.05;

/// Free-joint layout is `[rot(3), pos(3)]`.
const X: usize = 3;
const Z: usize = 5;

fn free_body(geometry: Geometry, mass: f64, inertia_scale: f64) -> Model {
    let i = mass * inertia_scale;
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_free_body(
            "body",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                mass,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(i, i, i)),
            ),
        )
        .build();
    model.bodies[0].geometry = Some(geometry);
    model
}

fn box_model() -> Model {
    free_body(
        Geometry::Box {
            half_extents: Vec3::new(HALF, HALF, HALF),
        },
        1.0,
        2.0 / 3.0 * HALF * HALF,
    )
}

fn sphere_model() -> Model {
    free_body(
        Geometry::Sphere { radius: RADIUS },
        1.0,
        0.4 * RADIUS * RADIUS,
    )
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

struct Scene {
    model: Model,
    material: ContactMaterial,
    q0: Vec<f64>,
    v0: Vec<f64>,
    steps: usize,
    obj_index: usize,
}

fn rollout<'a>(
    sc: &'a Scene,
    material: &ContactMaterial,
    ctrl: &'a dyn Fn(usize) -> DVec,
) -> ConvexContactRollout<'a> {
    ConvexContactRollout {
        model: &sc.model,
        ground_height: 0.0,
        material: material.clone(),
        config: ContactSolverConfig::gradients(),
        q0: DVec::from_slice(&sc.q0),
        v0: DVec::from_slice(&sc.v0),
        steps: sc.steps,
        ctrl,
    }
}

/// Central difference of the true forward rollout along one material scalar.
fn fd_material(sc: &Scene, set: impl Fn(&mut ContactMaterial, f64), h: f64) -> f64 {
    let obj = q_objective(sc.obj_index);
    let ctrl = |_t: usize| DVec::zeros(6);
    let mut mp = sc.material.clone();
    let mut mm = sc.material.clone();
    set(&mut mp, h);
    set(&mut mm, -h);
    let fp = convex_rollout_objective(&rollout(sc, &mp, &ctrl), &obj);
    let fm = convex_rollout_objective(&rollout(sc, &mm, &ctrl), &obj);
    (fp - fm) / (2.0 * h)
}

fn adjoint(sc: &Scene) -> phyz_diff::ConvexAdjointGradients {
    let obj = q_objective(sc.obj_index);
    let ctrl = |_t: usize| DVec::zeros(6);
    convex_adjoint_gradient(&rollout(sc, &sc.material, &ctrl), &obj)
        .expect("the material gates run on converging trajectories")
}

/// A box sliding on the ground, well inside the slip regime: `dJ/dmu` is the
/// derivative of how far it travels with respect to how much grip the floor has.
fn sliding_box() -> Scene {
    let mut v0 = vec![0.0; 6];
    v0[X] = 1.0;
    Scene {
        model: box_model(),
        material: ContactMaterial {
            friction: 0.4,
            ..ContactMaterial::default()
        },
        q0: vec![0.0, 0.0, 0.0, 0.0, 0.0, HALF],
        v0,
        // v0/(mu*g) = 0.255 s; 150 steps stops short of the stick transition.
        steps: 150,
        obj_index: X,
    }
}

#[test]
fn friction_gradient_matches_finite_difference_on_a_slide() {
    let sc = sliding_box();
    let g = adjoint(&sc);
    // h = 1e-6 on mu: large enough to clear the solver's 1e-12 tolerance floor
    // in the objective, small enough that the slide stays in one regime.
    let fd = fd_material(&sc, |m, d| m.friction += d, 1e-6);
    let rel = (g.d_friction - fd).abs() / fd.abs().max(1e-4);
    eprintln!(
        "dJ/dmu: adjoint {:.6e}  FD {:.6e}  rel {rel:.3e}",
        g.d_friction, fd
    );
    // More friction must shorten the slide.
    assert!(
        fd < 0.0,
        "sanity: more grip should mean less travel, got {fd:.3e}"
    );
    assert!(
        rel <= 1e-3,
        "dJ/dmu adjoint {} vs FD {fd} (rel {rel:.3e})",
        g.d_friction
    );
}

#[test]
fn friction_gradient_is_exactly_zero_when_nothing_slips() {
    // Same box, dropped and left to settle. Every contact ends up sticking, so
    // the cone boundary never binds and `mu` is structurally irrelevant.
    //
    // This is the test that separates a correct `dJ/dmu` from a plausible one.
    // Friction only enters through the cone constraint, so a contact strictly
    // inside the cone has *exactly* zero sensitivity to mu — not merely a small
    // one. An implementation that, say, differenced the assembled problem in mu
    // would return round-off here instead of zero and would look fine.
    let sc = Scene {
        model: box_model(),
        material: ContactMaterial {
            friction: 0.4,
            ..ContactMaterial::default()
        },
        q0: vec![0.0, 0.0, 0.0, 0.0, 0.0, HALF + 0.01],
        v0: vec![0.0; 6],
        steps: 200,
        obj_index: X,
    };
    let g = adjoint(&sc);
    assert_eq!(
        g.d_friction, 0.0,
        "a never-slipping trajectory must report structurally zero friction \
         sensitivity, got {}",
        g.d_friction
    );
}

/// A sphere dropped onto the plane hard enough to bounce: `dJ/de` is the
/// derivative of where it ends up with respect to how bouncy the floor is.
///
/// This is the design doc's §6.5 restitution row, and it discriminates the way
/// §2.4 says it should. Restitution is a term in `b` (assembly scales the
/// normal row of the free velocity by `1 + e_eff`), so it is differentiable in
/// `e` and the lane machinery prices it. Had restitution been implemented as a
/// post-solve velocity reset — the obvious alternative — it would be a branch
/// on the primal with no derivative in `e` at all, and this test would report a
/// flat zero against a clearly nonzero FD.
fn bouncing_sphere(e: f64) -> Scene {
    Scene {
        model: sphere_model(),
        material: ContactMaterial {
            restitution: e,
            friction: 0.5,
            ..ContactMaterial::default()
        },
        // Dropped from 20 cm: impact at ~1.9 m/s, far above the 0.05 m/s
        // restitution threshold, so the low-speed ramp is saturated and `e` is
        // fully in play.
        q0: vec![0.0, 0.0, 0.0, 0.0, 0.0, RADIUS + 0.20],
        v0: vec![0.0; 6],
        // Long enough to impact and rise well into the rebound.
        steps: 260,
        obj_index: Z,
    }
}

#[test]
fn restitution_gradient_matches_finite_difference_through_a_bounce() {
    let sc = bouncing_sphere(0.5);
    let g = adjoint(&sc);
    let fd = fd_material(&sc, |m, d| m.restitution += d, 1e-6);
    let rel = (g.d_restitution - fd).abs() / fd.abs().max(1e-4);
    eprintln!(
        "dJ/de: adjoint {:.6e}  FD {:.6e}  rel {rel:.3e}",
        g.d_restitution, fd
    );
    // A bouncier floor must leave the sphere higher at a fixed time in the
    // rebound. If this sign flips, the channel is wired backwards.
    assert!(
        fd > 0.0,
        "sanity: more bounce should mean more height, got {fd:.3e}"
    );
    assert!(
        rel <= 1e-3,
        "dJ/de adjoint {} vs FD {fd} (rel {rel:.3e})",
        g.d_restitution
    );
}

#[test]
fn restitution_gradient_vanishes_below_the_low_speed_ramp() {
    // A sphere resting on the plane never approaches faster than the
    // restitution threshold, so `effective_restitution` has smoothstepped to
    // zero and stays there. The gradient must follow it to zero rather than
    // reporting the nominal `e`.
    //
    // This is the ramp of §4.3 doing its job in the *derivative* as well as the
    // value, which is the entire reason it is a smoothstep and not an
    // `if |v_n| < eps { 0 }`: a hard cutoff would be non-differentiable exactly
    // at rest, which is where a settled stack spends all its time.
    let sc = Scene {
        model: sphere_model(),
        material: ContactMaterial {
            restitution: 0.8,
            friction: 0.5,
            ..ContactMaterial::default()
        },
        q0: vec![0.0, 0.0, 0.0, 0.0, 0.0, RADIUS],
        v0: vec![0.0; 6],
        steps: 200,
        obj_index: Z,
    };
    let g = adjoint(&sc);
    let fd = fd_material(&sc, |m, d| m.restitution += d, 1e-6);
    eprintln!(
        "settled dJ/de: adjoint {:.3e}  FD {:.3e}",
        g.d_restitution, fd
    );
    assert!(
        g.d_restitution.abs() < 1e-9,
        "a contact below the restitution ramp must report ~0, got {}",
        g.d_restitution
    );
}
