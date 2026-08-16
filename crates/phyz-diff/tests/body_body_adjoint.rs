//! FD gates for the trajectory adjoint through **body-body** contacts.
//!
//! The companion file `convex_contact_adjoint.rs` covers a single box on the
//! ground plane. Everything here needs two *moving* bodies in contact, which
//! until now the adjoint refused outright.
//!
//! The scene is a block riding on a driven plank that itself rests on the
//! ground — a deliberate reduction of the case this work exists for: a foot on
//! a skateboard's grip tape. It exercises what a ground-only adjoint structurally
//! cannot:
//!
//! - **tangential friction between two moving bodies.** The block is carried
//!   only by stiction against the plank's top face. `dJ/d(plank control)` is
//!   nonzero *only* through the friction channel, so a gradient that dropped
//!   body-body coupling would report ~0 here and the FD would not.
//! - **a rotating contact normal.** The plank pitches under load, so the
//!   contact frame turns with it. This is the channel a world-frozen normal
//!   would silently lose.
//! - **coupled multi-point manifolds.** Four block-plank points and four
//!   plank-ground points share one Delassus matrix, so every contact's impulse
//!   depends on all the others — the coupling a per-contact-local gradient
//!   misses (design doc §6.5, the two-box-stack row).

use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::{
    ConvexAdjointError, ConvexContactRollout, FinalStateObjective, convex_adjoint_gradient,
    convex_rollout_objective,
};
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder};

/// Plank half-extents: long and thin, so it pitches visibly under the block.
fn plank_half() -> Vec3 {
    Vec3::new(0.30, 0.12, 0.02)
}
/// Block half-extents.
fn block_half() -> Vec3 {
    Vec3::new(0.04, 0.04, 0.04)
}

/// Free-joint layout is `[rot(3), pos(3)]` per body, so body `b` owns DOFs
/// `6b..6b+6` and its z coordinate is `6b + 5`.
#[allow(dead_code)]
const PLANK_X: usize = 3;
const BLOCK_X: usize = 9;

fn inertia(mass: f64, h: Vec3) -> SpatialInertia {
    let c = mass / 3.0;
    SpatialInertia::new(
        mass,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(
            c * (h.y * h.y + h.z * h.z),
            c * (h.x * h.x + h.z * h.z),
            c * (h.x * h.x + h.y * h.y),
        )),
    )
}

/// Body 0 = plank (resting on the ground), body 1 = block (riding the plank).
fn plank_and_block(plank_mass: f64, block_mass: f64) -> Model {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_free_body(
            "plank",
            -1,
            SpatialTransform::identity(),
            inertia(plank_mass, plank_half()),
        )
        .add_free_body(
            "block",
            -1,
            SpatialTransform::identity(),
            inertia(block_mass, block_half()),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: plank_half(),
    });
    model.bodies[1].geometry = Some(Geometry::Box {
        half_extents: block_half(),
    });
    model
}

/// Resting configuration: plank on the ground, block on the plank, both flat.
fn stacked_q0() -> Vec<f64> {
    vec![
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        plank_half().z, // plank
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0 * plank_half().z + block_half().z, // block
    ]
}

/// `J = q_T[i]`, with the index baked in at run time.
fn q_objective(i: usize) -> FinalStateObjective<'static> {
    // `FinalStateObjective` borrows `&'static dyn Fn`, so the index cannot be
    // captured; a leaked closure is the cheapest way to keep the tests generic
    // over the component without writing one objective per index by hand.
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
    q0: Vec<f64>,
    v0: Vec<f64>,
    steps: usize,
    /// Constant world-x force applied to the plank every step.
    plank_push: f64,
    obj_index: usize,
    /// Relative tolerance for |adjoint − FD|.
    tol: f64,
}

fn check(case: &Case) {
    let model = plank_and_block(2.0, 1.0);
    // mu = 0.9 matches ipse's `board_material()` — the grip-tape coefficient
    // this scene is a reduction of.
    let material = ContactMaterial {
        friction: 0.9,
        ..ContactMaterial::default()
    };
    let config = ContactSolverConfig::gradients();
    let obj = q_objective(case.obj_index);
    let push = case.plank_push;
    let ctrl = move |_t: usize| {
        let mut u = DVec::zeros(12);
        u[PLANK_X] = push;
        u
    };

    fn make<'a>(
        m: &'a Model,
        material: &ContactMaterial,
        config: ContactSolverConfig,
        q0: &[f64],
        v0: &[f64],
        steps: usize,
        c: &'a dyn Fn(usize) -> DVec,
    ) -> ConvexContactRollout<'a> {
        ConvexContactRollout {
            model: m,
            ground_height: 0.0,
            material: material.clone(),
            config,
            q0: DVec::from_slice(q0),
            v0: DVec::from_slice(v0),
            steps,
            ctrl: c,
        }
    }

    let rollout = make(
        &model, &material, config, &case.q0, &case.v0, case.steps, &ctrl,
    );
    let g = convex_adjoint_gradient(&rollout, &obj)
        .expect("body-body contacts must no longer be refused");

    let h = 1e-7;
    let mut rows: Vec<(String, f64, f64)> = Vec::new();
    for i in 0..12 {
        let mut qp = case.q0.clone();
        let mut qm = case.q0.clone();
        qp[i] += h;
        qm[i] -= h;
        let fp = convex_rollout_objective(
            &make(&model, &material, config, &qp, &case.v0, case.steps, &ctrl),
            &obj,
        );
        let fm = convex_rollout_objective(
            &make(&model, &material, config, &qm, &case.v0, case.steps, &ctrl),
            &obj,
        );
        rows.push((format!("dJ/dq0[{i}]"), g.d_q0[i], (fp - fm) / (2.0 * h)));
    }
    for i in 0..12 {
        let mut vp = case.v0.clone();
        let mut vm = case.v0.clone();
        vp[i] += h;
        vm[i] -= h;
        let fp = convex_rollout_objective(
            &make(&model, &material, config, &case.q0, &vp, case.steps, &ctrl),
            &obj,
        );
        let fm = convex_rollout_objective(
            &make(&model, &material, config, &case.q0, &vm, case.steps, &ctrl),
            &obj,
        );
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
}

/// The block rides a driven plank: every bit of its motion arrives through
/// body-body friction.
#[test]
fn block_carried_by_friction_on_a_driven_plank() {
    check(&Case {
        name: "block_carried_by_friction",
        q0: stacked_q0(),
        v0: vec![0.0; 12],
        steps: 120,
        plank_push: 6.0,
        obj_index: BLOCK_X,
        tol: 1e-3,
    });
}

/// The block slides on the plank, decelerated by body-body Coulomb friction —
/// the sliding branch of the cone, between two moving bodies.
///
/// 80 steps is not a tuned number: at `v0 = 0.8 m/s` and `mu = 0.9` the block
/// decelerates at `mu*g = 8.83 m/s^2` and reaches zero slip at `t = 91 ms`, so
/// this rollout ends while the contact is unambiguously *still sliding*, which
/// is the branch under test. Running it through the stop is a different test
/// with a different expectation — see
/// [`slip_to_stick_transition_refuses_rather_than_guessing`].
///
/// The 1 mrad initial roll is load-bearing and is *not* cosmetic jitter. With
/// the plank exactly level, all four block-plank depths and all four
/// plank-ground depths tie at zero, and the objective has a genuine kink in the
/// roll lane: the measured one-sided derivatives there are `-5.457e-3` (roll
/// up) and `-1.391e-2` (roll down), each stable across `h` from `1e-6` to
/// `1e-8`, so a *central* difference returns their average `-9.686e-3` — the
/// midpoint of a corner, which is the derivative of nothing. The adjoint holds
/// the nominal symmetric regime and reports `0` for that lane, which is not
/// merely a different branch but sits outside the Clarke interval
/// `[-1.391e-2, -5.457e-3]` altogether. Tilting a millirad off the symmetry set
/// makes the manifold non-degenerate, and the roll lane then agrees.
///
/// It holds to the same `1e-3` as the flat-ground scenarios in
/// `convex_contact_adjoint.rs` — measured worst lane `4.3e-4` — so an
/// eight-contact two-body manifold costs no tolerance over a single box on a
/// plane.
/// **Negative test.** On an exactly symmetric manifold the adjoint's
/// symmetry-breaking lane is not a subgradient, and that is worth pinning.
///
/// Design doc §6.5 asks for the known limits to be mechanically checked rather
/// than described, so that a regression is visible and a competitor's benchmark
/// finds nothing we have not already published. This is that test for the
/// body-body extension.
///
/// With the plank exactly level, all eight contacts tie at zero depth. The
/// objective genuinely kinks in the plank-roll lane: the one-sided derivatives
/// are `-5.457e-3` (roll up) and `-1.391e-2` (roll down), each stable across
/// `h` from `1e-6` to `1e-8`. The adjoint holds the nominal symmetric regime
/// and returns `0`, which is not one of the two branches and does not lie in
/// the Clarke interval between them — an optimizer stepping on it would see a
/// flat direction where the true function has a corner.
///
/// It is a measure-zero configuration: the companion
/// [`block_sliding_on_the_plank_decelerates`] tilts 1 mrad off symmetry and the
/// same lane then agrees to `4.3e-4`. Real scenes are not exactly symmetric,
/// but hand-built initial conditions frequently are, which is exactly when a
/// caller would hit this.
#[test]
fn exact_symmetry_gives_a_lane_outside_the_clarke_set() {
    let model = plank_and_block(2.0, 1.0);
    let material = ContactMaterial {
        friction: 0.9,
        ..ContactMaterial::default()
    };
    let config = ContactSolverConfig::gradients();
    let obj = q_objective(BLOCK_X);
    let ctrl = |_t: usize| DVec::zeros(12);
    let mut v0 = vec![0.0; 12];
    v0[BLOCK_X] = 0.8;
    let rollout = ConvexContactRollout {
        model: &model,
        ground_height: 0.0,
        material,
        config,
        q0: DVec::from_slice(&stacked_q0()),
        v0: DVec::from_slice(&v0),
        steps: 80,
        ctrl: &ctrl,
    };
    let g = convex_adjoint_gradient(&rollout, &obj).expect("the solve itself converges");
    let roll = g.d_q0[0];
    assert!(
        roll.abs() < 1e-9,
        "the symmetric regime should report a flat roll lane, got {roll:.3e}"
    );
    // Both one-sided derivatives are strictly negative, so zero is outside
    // their hull. Stated as the assertion rather than as a comment.
    assert!(
        !(-1.391e-2..=-5.457e-3).contains(&roll),
        "0 must sit outside the Clarke interval — that is the documented defect"
    );
}

#[test]
fn block_sliding_on_the_plank_decelerates() {
    let mut v0 = vec![0.0; 12];
    v0[BLOCK_X] = 0.8;
    let mut q0 = stacked_q0();
    q0[0] = 1e-3;
    check(&Case {
        name: "block_sliding_on_plank",
        q0,
        v0,
        steps: 80,
        plank_push: 0.0,
        obj_index: BLOCK_X,
        tol: 1e-3,
    });
}

/// Carrying the same slide *through* the moment the block stops does not
/// produce a gradient — it produces a refusal, and that is the designed
/// behaviour.
///
/// At the slip-to-stick transition the eight-contact manifold is redundant
/// (four coplanar block-plank points and four plank-ground points) *and* the
/// tangential rows are switching branch, so the active-set Newton stalls
/// around a `1e-7` residual instead of reaching the `1e-12` the gradient preset
/// asks for. The solution is not a KKT point, the IFT does not apply, and
/// `convex_adjoint_gradient` reports `ConvexAdjointError::Unconverged` rather
/// than differentiating a point that is not a fixed point.
///
/// This test exists because the failure is worth *owning*: it is the one place
/// in the body-body extension where a caller doing contact-implicit trajectory
/// optimization is actually stopped, so it should be a pinned, named behaviour
/// rather than a surprise at the call site. If a future solver change makes
/// this converge, this test failing is the correct signal to promote the case
/// into [`check`].
#[test]
fn slip_to_stick_transition_refuses_rather_than_guessing() {
    let model = plank_and_block(2.0, 1.0);
    let material = ContactMaterial {
        friction: 0.9,
        ..ContactMaterial::default()
    };
    let mut v0 = vec![0.0; 12];
    v0[BLOCK_X] = 0.8;
    let ctrl = |_t: usize| DVec::zeros(12);
    let rollout = ConvexContactRollout {
        model: &model,
        ground_height: 0.0,
        material,
        config: ContactSolverConfig::gradients(),
        q0: DVec::from_slice(&stacked_q0()),
        v0: DVec::from_slice(&v0),
        // 150 steps carries the block well past the 91 ms stop.
        steps: 150,
        ctrl: &ctrl,
    };
    // `ConvexAdjointGradients` is not `Debug`, so `expect_err` is unavailable;
    // match the result directly.
    match convex_adjoint_gradient(&rollout, &q_objective(BLOCK_X)) {
        Err(ConvexAdjointError::Unconverged { step, residual, .. }) => {
            eprintln!("slip_to_stick: refused at step {step}, residual {residual:.3e}");
        }
        Err(other) => panic!("expected a loud non-convergence, got {other:?}"),
        Ok(_) => panic!("the transition must not silently yield a gradient"),
    }
}
