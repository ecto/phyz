//! The same reverse-mode smooth-adjoint gates as `smooth_adjoint.rs`, but
//! through the **solver-level** contact channel (`PHYZ_SOLVER_ADJOINT=1`):
//! the covectors come from the transposed re-execution of the recorded solve
//! (`bar_apr`/`bar_c` against the generic `apr`/`cvec` pieces) instead of the
//! IFT map linearization. Same FD oracle, separate process because the env
//! gates are `OnceLock`-cached.

use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::{
    ConvexContactRollout, FinalStateObjective, convex_adjoint_gradient, convex_rollout_objective,
};
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder};

fn enable() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| unsafe {
        std::env::set_var("PHYZ_SMOOTH_ADJOINT", "1");
        std::env::set_var("PHYZ_SOLVER_ADJOINT", "1");
    });
}

const HALF: f64 = 0.05;

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

fn q_objective(i: usize) -> FinalStateObjective<'static> {
    let value: &'static dyn Fn(&[f64], &[f64]) -> f64 =
        Box::leak(Box::new(move |q: &[f64], _: &[f64]| q[i]));
    type GradFn = dyn Fn(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>);
    let gradient: &'static GradFn = Box::leak(Box::new(move |q: &[f64], v: &[f64]| {
        let mut gq = vec![0.0; q.len()];
        gq[i] = 1.0;
        (gq, vec![0.0; v.len()])
    }));
    FinalStateObjective { value, gradient }
}

fn rollout<'a>(
    model: &'a Model,
    material: &ContactMaterial,
    q0: &[f64],
    v0: &[f64],
    steps: usize,
    ctrl: &'a dyn Fn(usize) -> DVec,
) -> ConvexContactRollout<'a> {
    ConvexContactRollout {
        model,
        ground_height: 0.0,
        material: material.clone(),
        config: ContactSolverConfig::gradients(),
        q0: DVec::from_slice(q0),
        v0: DVec::from_slice(v0),
        steps,
        ctrl,
    }
}

fn slope_gravity(alpha: f64) -> Vec3 {
    Vec3::new(GRAVITY * alpha.sin(), 0.0, -GRAVITY * alpha.cos())
}

fn fd_gate(alpha: f64, mu: f64, obj_index: usize, tol: f64) {
    enable();
    let model = box_model(slope_gravity(alpha));
    let material = ContactMaterial {
        friction: mu,
        ..Default::default()
    };
    let q0 = [0.0, 0.0, 0.0, 0.0, 0.0, HALF - 1e-5];
    let v0 = [0.0; 6];
    let steps = 10;
    let ctrl = |_t: usize| DVec::zeros(6);
    let obj = q_objective(obj_index);

    let g = convex_adjoint_gradient(&rollout(&model, &material, &q0, &v0, steps, &ctrl), &obj)
        .expect("adjoint must not refuse");

    let h = 1e-6;
    let f = |q: &[f64], v: &[f64]| {
        convex_rollout_objective(&rollout(&model, &material, q, v, steps, &ctrl), &obj)
    };
    let mut worst: f64 = 0.0;
    let mut scale: f64 = 0.0;
    for i in 0..6 {
        let (mut qp, mut qm) = (q0, q0);
        qp[i] += h;
        qm[i] -= h;
        let fd = (f(&qp, &v0) - f(&qm, &v0)) / (2.0 * h);
        worst = worst.max((g.d_q0[i] - fd).abs());
        scale = scale.max(fd.abs());

        let (mut vp, mut vm) = (v0, v0);
        vp[i] += h;
        vm[i] -= h;
        let fd = (f(&q0, &vp) - f(&q0, &vm)) / (2.0 * h);
        worst = worst.max((g.d_v0[i] - fd).abs());
        scale = scale.max(fd.abs());
    }
    assert!(
        worst <= tol * scale.max(1.0),
        "alpha={alpha} mu={mu}: worst state-lane error {worst} (scale {scale})"
    );

    // Mass lane through the solver-level covectors.
    let fd_at = |dm: f64| {
        let mut m2 = model.clone();
        let si = m2.bodies[0].inertia;
        m2.bodies[0].inertia = SpatialInertia::new(si.mass + dm, si.com, si.inertia);
        convex_rollout_objective(&rollout(&m2, &material, &q0, &v0, steps, &ctrl), &obj)
    };
    let fd = (fd_at(1e-6) - fd_at(-1e-6)) / 2e-6;
    assert!(
        (g.d_inertia[0][0] - fd).abs() <= tol * fd.abs().max(1e-6),
        "mass lane: adjoint {} vs fd {fd}",
        g.d_inertia[0][0]
    );
}

#[test]
fn incline_stiction_matches_fd() {
    fd_gate(0.15, 0.5, 3, 2e-5);
}

#[test]
fn incline_slip_matches_fd() {
    fd_gate(0.35, 0.2, 3, 2e-5);
}

#[test]
fn resting_normal_channel_matches_fd() {
    fd_gate(0.0, 0.5, 5, 2e-5);
}
