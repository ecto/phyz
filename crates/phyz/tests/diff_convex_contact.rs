//! Gates tying the unified convex-contact adjoint to the forward simulator.
//!
//! 1. The adjoint's forward pass must be **bit-identical** to
//!    `Simulator::step_with_contacts` — same detection, assembly, warm-started
//!    solve, integration. Anything less re-opens the forward/backward
//!    contact-model split this adjoint exists to close.
//! 2. The gradient must match a central finite difference through that same
//!    forward path.
//! 3. The mismatch demonstration: the old per-vertex penalty adjoint
//!    (`phyz::diff::AdjointRollout`), pointed at the same physical scenario,
//!    reports a gradient for *different physics*. Its disagreement with the
//!    FD of the real forward pass is measured and printed — the size of the
//!    error the unified adjoint removes.

use std::cell::RefCell;

use phyz::contact::{ContactMaterial, ContactSolverConfig};
use phyz::diff::{
    AdjointRollout, CollisionMesh, ContactSetup, ConvexContactRollout, FinalStateObjective,
    GroundContact, adjoint_rollout_gradient, convex_adjoint_gradient, convex_rollout_objective,
};
use phyz::math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz::model::{Geometry, Model, ModelBuilder};
use phyz::sim::Simulator;

const HALF: f64 = 0.05;
const MASS: f64 = 1.0;
const DT: f64 = 1e-3;
const STEPS: usize = 200;
const DROP_Z: f64 = HALF + 0.01;

fn box_model(mass: f64) -> Model {
    let h = HALF;
    let ix = mass / 12.0 * (2.0 * h) * (2.0 * h) * 2.0;
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(DT)
        .add_free_body(
            "box",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                mass,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(ix, ix, ix)),
            ),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box {
        half_extents: Vec3::new(h, h, h),
    });
    model
}

fn rollout<'a>(model: &'a Model, ctrl: &'a dyn Fn(usize) -> DVec) -> ConvexContactRollout<'a> {
    let mut q0 = DVec::zeros(6);
    q0[5] = DROP_Z;
    ConvexContactRollout {
        model,
        ground_height: 0.0,
        material: ContactMaterial::default(),
        config: ContactSolverConfig::simulation(),
        q0,
        v0: DVec::zeros(6),
        steps: STEPS,
        ctrl,
    }
}

/// Gate 1 — the adjoint's forward pass reproduces the simulator bit for bit.
#[test]
fn forward_pass_is_bit_identical_to_simulator() {
    let model = box_model(MASS);
    let material = ContactMaterial::default();

    // Reference: the shipping simulator path.
    let sim = Simulator::new();
    let mut state = model.default_state();
    state.q[5] = DROP_Z;
    for _ in 0..STEPS {
        sim.step_with_contacts(&model, &mut state, 0.0, &material);
    }

    // The adjoint's forward pass, observed through the objective closure.
    let captured: RefCell<(Vec<f64>, Vec<f64>)> = RefCell::new((Vec::new(), Vec::new()));
    let obj = FinalStateObjective {
        value: &|q: &[f64], v: &[f64]| {
            *captured.borrow_mut() = (q.to_vec(), v.to_vec());
            q[5]
        },
        gradient: &|q: &[f64], v: &[f64]| (vec![0.0; q.len()], vec![0.0; v.len()]),
    };
    let ctrl = |_t: usize| DVec::zeros(6);
    let _ = convex_rollout_objective(&rollout(&model, &ctrl), &obj);
    let (qf, vf) = captured.borrow().clone();

    for i in 0..6 {
        assert!(
            qf[i] == state.q[i] && vf[i] == state.v[i],
            "trajectory diverged at dof {i}: adjoint ({}, {}) vs simulator ({}, {})",
            qf[i],
            vf[i],
            state.q[i],
            state.v[i]
        );
    }
}

fn height_objective<'a>() -> FinalStateObjective<'a> {
    FinalStateObjective {
        value: &|q: &[f64], _v: &[f64]| q[5],
        gradient: &|q: &[f64], v: &[f64]| {
            let mut gq = vec![0.0; q.len()];
            gq[5] = 1.0;
            (gq, vec![0.0; v.len()])
        },
    }
}

/// FD of the real forward path w.r.t. the body mass (inertia scaled with it,
/// as a solid box's is — the same physical parameter the penalty adjoint's
/// `d_inertia[0][0] + I-scaling` cannot see; kept to the pure mass scalar so
/// all three numbers speak about the same derivative).
fn fd_dj_dmass() -> f64 {
    let obj = height_objective();
    let ctrl = |_t: usize| DVec::zeros(6);
    let h = 1e-6;
    let mut grads = [0.0; 2];
    for (i, m) in [MASS + h, MASS - h].into_iter().enumerate() {
        // Perturb the mass scalar only, holding the inertia tensor: that is
        // the `d_inertia[0][0]` component both adjoints report.
        let mut model = box_model(MASS);
        model.bodies[0].inertia = SpatialInertia::new(
            m,
            model.bodies[0].inertia.com,
            model.bodies[0].inertia.inertia,
        );
        grads[i] = convex_rollout_objective(&rollout(&model, &ctrl), &obj);
    }
    (grads[0] - grads[1]) / (2.0 * h)
}

/// Gate 2 — the unified adjoint matches FD through the real forward pass.
/// Gate 3 (measured, printed) — the old penalty adjoint does not: it
/// differentiates a different contact model than the simulator integrates.
#[test]
fn unified_adjoint_matches_forward_physics_and_penalty_adjoint_does_not() {
    let model = box_model(MASS);
    let obj = height_objective();
    let ctrl = |_t: usize| DVec::zeros(6);

    let fd = fd_dj_dmass();

    // Unified adjoint.
    let g = convex_adjoint_gradient(&rollout(&model, &ctrl), &obj).expect("adjoint refused");
    let unified = g.d_inertia[0][0];
    let unified_rel = (unified - fd).abs() / fd.abs();

    // Old penalty adjoint on the same physical scenario (its own contact
    // model: per-vertex spring-damper, its own quaternion layout).
    let h = HALF;
    let mut corners = Vec::with_capacity(8);
    for x in [-h, h] {
        for y in [-h, h] {
            for z in [-h, h] {
                corners.push(Vec3::new(x, y, z));
            }
        }
    }
    let meshes = vec![CollisionMesh {
        body: 0,
        vertices: corners,
    }];
    let ground = GroundContact {
        height: 0.0,
        stiffness: ContactMaterial::default().stiffness,
        damping: ContactMaterial::default().damping,
    };
    // Free-joint quaternion layout: [x, y, z, w, qx, qy, qz].
    let mut q0 = vec![0.0; 7];
    q0[2] = DROP_Z;
    q0[3] = 1.0;
    let penalty_obj = FinalStateObjective {
        value: &|q: &[f64], _v: &[f64]| q[2],
        gradient: &|q: &[f64], v: &[f64]| {
            let mut gq = vec![0.0; q.len()];
            gq[2] = 1.0;
            (gq, vec![0.0; v.len()])
        },
    };
    let ctrl_pen = |_t: usize| DVec::zeros(6);
    let penalty = adjoint_rollout_gradient(
        &AdjointRollout {
            model: &model,
            contact: Some(ContactSetup {
                ground,
                meshes: &meshes,
            }),
            q0,
            v0: vec![0.0; 6],
            steps: STEPS,
            ctrl: &ctrl_pen,
        },
        &penalty_obj,
    );
    let penalty_dmass = penalty.d_inertia[0][0];
    let penalty_rel = (penalty_dmass - fd).abs() / fd.abs();

    println!("dJ/dmass  FD(real forward)   = {fd:+.9e}");
    println!("dJ/dmass  unified adjoint    = {unified:+.9e}  (rel err {unified_rel:.3e})");
    println!("dJ/dmass  penalty adjoint    = {penalty_dmass:+.9e}  (rel err {penalty_rel:.3e})");

    assert!(
        unified_rel < 1e-3,
        "unified adjoint dJ/dmass {unified} vs FD {fd} (rel {unified_rel:.3e})"
    );
    // The demonstration, kept as an assertion so it cannot silently rot: the
    // penalty adjoint's gradient of the same scenario is order-of-magnitude
    // wrong about the physics the simulator integrates. If this ever starts
    // agreeing, the two models were changed to coincide and this gate (plus
    // the docs that cite it) should be revisited.
    assert!(
        penalty_rel > 10.0 * unified_rel,
        "penalty adjoint unexpectedly agrees with the convex forward physics: \
         rel err {penalty_rel:.3e} vs unified {unified_rel:.3e}"
    );
}
