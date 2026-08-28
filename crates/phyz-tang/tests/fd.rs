//! The FD discipline: every gradient this crate reports is checked against a
//! central finite difference of the *executed* forward — the same
//! [`PhysicsStep::forward`] a caller would run, not a reimplementation of it.
//!
//! Three rungs, in order of what they can prove:
//!
//! 1. **Free flight.** No contact, smooth dynamics, agreement to FD noise.
//! 2. **One contacted step.** A box on a plane, resting and impacting.
//! 3. **A 32-step chained rollout.** The point of the whole exercise: N ops
//!    composed, one cotangent pulled back through all of them, checked
//!    against FD of the full 32-step forward.

use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder};
use phyz_tang::{PhysicsStep, PhysicsTape};
use tang_tensor::{Shape, Tensor};

const HALF: f64 = 0.05;
const DT: f64 = 1e-3;

/// The canonical free box with a box geometry — the scene phyz's own convex
/// adjoint tests use, so a disagreement here is this crate's, not physics'.
fn box_model(mass: f64) -> Model {
    let ix = mass / 12.0 * (2.0 * HALF) * (2.0 * HALF) * 2.0;
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
        half_extents: Vec3::new(HALF, HALF, HALF),
    });
    model
}

fn vec(v: &[f64]) -> Tensor<f64> {
    Tensor::new(v.to_vec(), Shape::from_slice(&[v.len()]))
}

/// A deterministic, non-axis-aligned cotangent: a diagonal seed would let a
/// diagonal error pass.
fn seed(n: usize) -> Tensor<f64> {
    Tensor::new(
        (0..n).map(|i| 0.3 + 0.17 * (i as f64) % 1.1).collect(),
        Shape::from_slice(&[n]),
    )
}

fn dot(a: &Tensor<f64>, b: &Tensor<f64>) -> f64 {
    a.data().iter().zip(b.data()).map(|(x, y)| x * y).sum()
}

/// Relative error, with the denominator floored at `scale` so a genuinely
/// tiny component is judged against the size of the vector it lives in.
fn worst_rel(analytic: &[f64], fd: &[f64]) -> (f64, usize) {
    let scale = fd.iter().fold(0.0_f64, |m, x| m.max(x.abs())).max(1e-9) * 1e-3;
    let mut worst = 0.0_f64;
    let mut at = 0;
    for (i, (a, f)) in analytic.iter().zip(fd).enumerate() {
        let r = (a - f).abs() / f.abs().max(scale);
        if r > worst {
            worst = r;
            at = i;
        }
    }
    (worst, at)
}

/// FD of `w · forward(state, ctrl)` w.r.t. each component of state and ctrl.
fn fd_step(
    op: &PhysicsStep,
    state: &Tensor<f64>,
    ctrl: &Tensor<f64>,
    w: &Tensor<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let probe = |s: &Tensor<f64>, c: &Tensor<f64>| -> f64 { dot(w, &op.forward(s, c).unwrap()) };
    let h = 1e-7;
    let mut ds = vec![0.0; state.numel()];
    for (i, out) in ds.iter_mut().enumerate() {
        let mut up = state.clone();
        up.data_mut()[i] += h;
        let mut dn = state.clone();
        dn.data_mut()[i] -= h;
        *out = (probe(&up, ctrl) - probe(&dn, ctrl)) / (2.0 * h);
    }
    let hu = 1e-4;
    let mut du = vec![0.0; ctrl.numel()];
    for (i, out) in du.iter_mut().enumerate() {
        let mut up = ctrl.clone();
        up.data_mut()[i] += hu;
        let mut dn = ctrl.clone();
        dn.data_mut()[i] -= hu;
        *out = (probe(state, &up) - probe(state, &dn)) / (2.0 * hu);
    }
    (ds, du)
}

fn check_step(label: &str, op: &PhysicsStep, state: &Tensor<f64>, ctrl: &Tensor<f64>, tol: f64) {
    let w = seed(op.state_dim());
    let g = op.vjp(state, ctrl, &w).expect("adjoint refused");
    let (fd_s, fd_u) = fd_step(op, state, ctrl, &w);

    let (rs, is) = worst_rel(g.d_state_in.data(), &fd_s);
    let (ru, iu) = worst_rel(g.d_ctrl.data(), &fd_u);
    println!("{label}: d_state worst rel {rs:.3e} @ {is}, d_ctrl worst rel {ru:.3e} @ {iu}");
    println!("  adjoint d_state {:?}", g.d_state_in.data());
    println!("  fd      d_state {fd_s:?}");
    assert!(
        rs < tol,
        "{label}: d_state_in rel {rs:.3e} at {is} exceeds {tol:.1e}"
    );
    assert!(
        ru < tol,
        "{label}: d_ctrl rel {ru:.3e} at {iu} exceeds {tol:.1e}"
    );
}

/// Rung 1 — contact-free. The free-flight path must differentiate too, or a
/// chained rollout breaks the moment the body leaves the ground.
#[test]
fn free_flight_step_matches_fd() {
    let model = box_model(1.0);
    let op = PhysicsStep::new(&model);
    let state = vec(&[
        0.1, -0.2, 0.05, 0.0, 0.0, 1.0, 0.3, -0.1, 0.2, 0.4, 0.5, -0.6,
    ]);
    let ctrl = vec(&[0.2, -0.3, 0.1, 1.0, -2.0, 0.5]);
    check_step("free flight", &op, &state, &ctrl, 1e-5);
}

/// Rung 1b — no contact and no control either: pure ballistic.
#[test]
fn free_flight_zero_ctrl_matches_fd() {
    let model = box_model(1.0);
    let op = PhysicsStep::new(&model);
    let state = vec(&[0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let ctrl = vec(&[0.0; 6]);
    check_step("ballistic", &op, &state, &ctrl, 1e-5);
}

/// Rung 2 — one contacted step, box resting on the plane.
#[test]
fn resting_contact_step_matches_fd() {
    let model = box_model(1.0);
    let op = PhysicsStep::new(&model);
    let state = vec(&[0.0, 0.0, 0.0, 0.0, 0.0, HALF, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let ctrl = vec(&[0.0; 6]);
    check_step("resting contact", &op, &state, &ctrl, 5e-3);
}

/// Rung 2b — a contacted step with the box sliding and driven, so friction is
/// live and the control channel is not trivially zero.
#[test]
fn sliding_contact_step_matches_fd() {
    let model = box_model(1.0);
    let op = PhysicsStep::new(&model);
    let state = vec(&[0.0, 0.0, 0.0, 0.0, 0.0, HALF, 0.0, 0.0, 0.0, 0.8, 0.0, -0.1]);
    let ctrl = vec(&[0.0, 0.0, 0.0, 4.0, 0.0, 0.0]);
    check_step("sliding contact", &op, &state, &ctrl, 5e-3);
}

/// Rung 3 — the whole point. 32 ops composed; one cotangent on the final
/// state pulled back through every one of them; checked against FD of the
/// full 32-step forward.
#[test]
fn chained_32_step_rollout_matches_fd() {
    let model = box_model(1.0);
    let op = PhysicsStep::new(&model);
    let n = 32;

    // Drop from just above the plane, driven sideways: the window contains
    // free flight, the impact, and the contacted slide after it.
    let state0 = vec(&[
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        HALF + 0.004,
        0.0,
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
    ]);
    let ctrls: Vec<Tensor<f64>> = (0..n)
        .map(|t| vec(&[0.0, 0.0, 0.0, 3.0 + 0.1 * t as f64, 0.5, 0.0]))
        .collect();

    let roll = |s0: &Tensor<f64>, cs: &[Tensor<f64>]| -> Tensor<f64> {
        let mut tape = PhysicsTape::new(&op, s0.clone());
        for c in cs {
            tape.step(c).unwrap();
        }
        tape.state().clone()
    };

    let mut tape = PhysicsTape::new(&op, state0.clone());
    for c in &ctrls {
        tape.step(c).unwrap();
    }
    let w = seed(op.state_dim());
    let grads = tape.backward(&w).expect("chained adjoint refused");

    // FD w.r.t. state0.
    let h = 1e-7;
    let mut fd_s0 = vec![0.0; state0.numel()];
    for (i, out) in fd_s0.iter_mut().enumerate() {
        let mut up = state0.clone();
        up.data_mut()[i] += h;
        let mut dn = state0.clone();
        dn.data_mut()[i] -= h;
        *out = (dot(&w, &roll(&up, &ctrls)) - dot(&w, &roll(&dn, &ctrls))) / (2.0 * h);
    }
    let (rs, is) = worst_rel(grads.d_state0.data(), &fd_s0);
    println!("chained/32: d_state0 worst rel {rs:.3e} @ {is}");
    println!("  adjoint {:?}", grads.d_state0.data());
    println!("  fd      {fd_s0:?}");

    // FD w.r.t. three probe controls spread across the window.
    let hu = 1e-4;
    let mut worst_u = 0.0_f64;
    for &t in &[0_usize, n / 2, n - 1] {
        let mut fd_u = vec![0.0; op.ctrl_dim()];
        for (i, out) in fd_u.iter_mut().enumerate() {
            let mut up = ctrls.clone();
            up[t].data_mut()[i] += hu;
            let mut dn = ctrls.clone();
            dn[t].data_mut()[i] -= hu;
            *out = (dot(&w, &roll(&state0, &up)) - dot(&w, &roll(&state0, &dn))) / (2.0 * hu);
        }
        let (ru, iu) = worst_rel(grads.d_ctrl[t].data(), &fd_u);
        println!("chained/32: d_ctrl[{t}] worst rel {ru:.3e} @ {iu}");
        println!("  adjoint {:?}", grads.d_ctrl[t].data());
        println!("  fd      {fd_u:?}");
        worst_u = worst_u.max(ru);
    }

    assert!(rs < 1e-2, "chained d_state0 rel {rs:.3e} at {is}");
    assert!(worst_u < 1e-2, "chained d_ctrl rel {worst_u:.3e}");
}

/// The one-forward-caching hazard, pinned as a test: a second forward on the
/// tape must *extend* the trajectory, not silently replace the first step's
/// record. tang-train's `Module` cache would fail this by construction.
#[test]
fn second_forward_extends_rather_than_invalidates() {
    let model = box_model(1.0);
    let op = PhysicsStep::new(&model);
    let s0 = vec(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let u = vec(&[0.0; 6]);

    let mut tape = PhysicsTape::new(&op, s0.clone());
    tape.step(&u).unwrap();
    tape.step(&u).unwrap();
    assert_eq!(tape.len(), 2);

    let w = seed(op.state_dim());
    // Backward takes &self and is idempotent: two calls, same answer.
    let a = tape.backward(&w).unwrap();
    let b = tape.backward(&w).unwrap();
    assert_eq!(a.d_state0.data(), b.d_state0.data());
    assert_eq!(a.d_ctrl.len(), 2);

    // And two chained ballistic steps give dz_N/dvz_0 = 2*dt. Not bit-exact:
    // `contact_adjoint` still finite-differences its own lanes internally
    // (`PHYZ_ADJOINT_FD_EPS`, default 1e-8), so even the contact-free path
    // carries that instrument's noise — ~1e-7 relative, visible here and in
    // the free-flight rungs above.
    let mut wz = Tensor::<f64>::zeros(Shape::from_slice(&[op.state_dim()]));
    wz.data_mut()[5] = 1.0;
    let g = tape.backward(&wz).unwrap();
    assert!(
        (g.d_state0.data()[11] - 2.0 * DT).abs() / (2.0 * DT) < 1e-6,
        "dz2/dvz0 = {}, want {}",
        g.d_state0.data()[11],
        2.0 * DT
    );
}
