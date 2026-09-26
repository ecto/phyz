//! Gates for the reverse-mode smooth adjoint (`PHYZ_SMOOTH_ADJOINT=1`), IFT
//! contact channel (the default mode).
//!
//! The env gates are `OnceLock`-cached, so this file owns its process: every
//! test sets the variable before the first gate read, and no test here ever
//! wants it unset. The lane-machinery behaviour with the gate off is covered
//! by every pre-existing adjoint test in this crate, which run in their own
//! processes without the variable.
//!
//! Three kinds of pin:
//! - **contact-free exactness**: free fall has closed-form gradients, and the
//!   reverse path must hit them to rounding — the lane machinery's `~1e-8`
//!   difference floor is exactly what this path removes;
//! - **contacted FD gates**: stiction and slip on the tilted-gravity incline,
//!   every state/control lane against a central difference of the real
//!   rollout — the same oracle `incline_and_tipping.rs` holds the lane
//!   machinery to;
//! - **parameter channels**: inertia and restitution against FD.

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

/// Contact-free free fall: `q_z(T)` is linear in `v_z0` with coefficient
/// `steps·dt` (identity orientation, no angular velocity, so the free joint's
/// `world_lin` rotation is the identity), and independent of every inertia
/// scalar. The reverse path must reproduce both **to rounding** — this is the
/// pin that the `~1e-8` finite-difference accuracy floor is gone.
#[test]
fn contact_free_gradients_are_exact() {
    enable();
    let model = box_model(Vec3::new(0.0, 0.0, -GRAVITY));
    let material = ContactMaterial::default();
    // Well above the ground: 20 steps of fall from 1 m never touch it.
    let q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let v0 = [0.0; 6];
    let steps = 20;
    let ctrl = |_t: usize| DVec::zeros(6);
    let obj = q_objective(5); // world z
    let g = convex_adjoint_gradient(&rollout(&model, &material, &q0, &v0, steps, &ctrl), &obj)
        .expect("contact-free adjoint must not refuse");

    let want = steps as f64 * model.dt;
    assert!(
        (g.d_v0[5] - want).abs() < 1e-14,
        "dq_z/dv_z0 = {} vs exact {want}",
        g.d_v0[5]
    );
    assert!(
        (g.d_q0[5] - 1.0).abs() < 1e-14,
        "dq_z/dq_z0 = {}",
        g.d_q0[5]
    );
    for (b, di) in g.d_inertia.iter().enumerate() {
        // Free fall is mass-independent; the COM/inertia channels are exactly
        // zero too at identity orientation with zero angular velocity.
        for (k, v) in di.iter().enumerate() {
            assert!(
                v.abs() < 1e-12,
                "d_inertia[{b}][{k}] = {v} should vanish in free fall"
            );
        }
    }
    // Control on the z DOF: q_z picks up dt per remaining step per unit force
    // over mass 1 → sum_{t} (steps - t)·dt² · gear(=1).
    let want_u: f64 = (0..steps)
        .map(|t| (steps - t) as f64 * model.dt * model.dt)
        .sum();
    let du: f64 = g.d_ctrl.iter().map(|d| d[5]).sum();
    assert!(
        (du - want_u).abs() < 1e-13,
        "summed control gradient {du} vs exact {want_u}"
    );
}

/// Gravity as seen in the frame of a slope of angle `alpha`.
fn slope_gravity(alpha: f64) -> Vec3 {
    Vec3::new(GRAVITY * alpha.sin(), 0.0, -GRAVITY * alpha.cos())
}

/// Sweep q0/v0/ctrl/restitution/inertia-mass lanes against central FD of the
/// real rollout on a contacted trajectory.
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

    // First-step control lane along the slope (x force).
    {
        let h = 1e-4;
        let mk = |s: f64| {
            move |t: usize| {
                let mut u = DVec::zeros(6);
                if t == 0 {
                    u[3] = s;
                }
                u
            }
        };
        let (cp, cm) = (mk(h), mk(-h));
        let fp = convex_rollout_objective(&rollout(&model, &material, &q0, &v0, steps, &cp), &obj);
        let fm = convex_rollout_objective(&rollout(&model, &material, &q0, &v0, steps, &cm), &obj);
        let fd = (fp - fm) / (2.0 * h);
        assert!(
            (g.d_ctrl[0][3] - fd).abs() <= tol * fd.abs().max(1e-4),
            "ctrl lane: adjoint {} vs fd {fd}",
            g.d_ctrl[0][3]
        );
    }

    // Mass lane.
    {
        let h = 1e-6;
        let fd_at = |dm: f64| {
            let mut m2 = model.clone();
            let si = m2.bodies[0].inertia;
            m2.bodies[0].inertia = SpatialInertia::new(si.mass + dm, si.com, si.inertia);
            convex_rollout_objective(&rollout(&m2, &material, &q0, &v0, steps, &ctrl), &obj)
        };
        let fd = (fd_at(h) - fd_at(-h)) / (2.0 * h);
        assert!(
            (g.d_inertia[0][0] - fd).abs() <= tol * fd.abs().max(1e-6),
            "mass lane: adjoint {} vs fd {fd}",
            g.d_inertia[0][0]
        );
    }

    // Restitution lane (zero in these resting cases is fine — FD sees the
    // same flat region; the assert is agreement, not non-triviality).
    {
        let h = 1e-5;
        let fd_at = |de: f64| {
            let mut m2 = material.clone();
            m2.restitution += de;
            convex_rollout_objective(&rollout(&model, &m2, &q0, &v0, steps, &ctrl), &obj)
        };
        let fd = (fd_at(h) - fd_at(-h)) / (2.0 * h);
        assert!(
            (g.d_restitution - fd).abs() <= tol * fd.abs().max(1e-6),
            "restitution lane: adjoint {} vs fd {fd}",
            g.d_restitution
        );
    }
}

/// Stiction: `tan α < μ`, cone interior.
#[test]
fn incline_stiction_matches_fd() {
    fd_gate(0.15, 0.5, 3, 2e-5);
}

/// Slip: `tan α > μ`, cone boundary — the sliding IFT branch plus the
/// `mt_rel` channel.
#[test]
fn incline_slip_matches_fd() {
    fd_gate(0.35, 0.2, 3, 2e-5);
}

/// Flat resting box, vertical objective — the pure normal/impedance channel.
#[test]
fn resting_normal_channel_matches_fd() {
    fd_gate(0.0, 0.5, 5, 2e-5);
}
