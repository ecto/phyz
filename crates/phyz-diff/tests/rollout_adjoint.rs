//! Smoke gate for the split-crate mirror of the trajectory adjoint. The full
//! gate suite (pendulum/chain FD sweeps, contact vertex gradients) lives in
//! the umbrella crate's `diff_adjoint.rs`; the two copies differ only in
//! import paths, so one exact closed-form gate proves the mirror is wired.

use phyz_diff::rollout::{adjoint_rollout_gradient, AdjointRollout, FinalStateObjective};
use phyz_math::{DVec, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::ModelBuilder;

/// Flywheel: `v_T = steps·dt·τ/I_zz` exactly under semi-implicit Euler, so
/// `dJ/dI_zz = −steps·dt·τ/I_zz²` must match to machine precision.
#[test]
fn flywheel_closed_form() {
    const TAU: f64 = 0.02;
    const IZZ: f64 = 0.05;
    const DT: f64 = 1.0 / 480.0;
    const STEPS: usize = 96;

    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(DT)
        .add_revolute_body(
            "disc",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                1.4,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(0.03, 0.03, IZZ)),
            ),
        )
        .build();

    let ctrl = |_t: usize| DVec::from_slice(&[TAU]);
    let obj = FinalStateObjective {
        value: &|_q: &[f64], v: &[f64]| v[0],
        gradient: &|q: &[f64], v: &[f64]| {
            let mut gv = vec![0.0; v.len()];
            gv[0] = 1.0;
            (vec![0.0; q.len()], gv)
        },
    };
    let rollout = AdjointRollout {
        model: &model,
        contact: None,
        q0: vec![0.0],
        v0: vec![0.0],
        steps: STEPS,
        ctrl: &ctrl,
    };

    let g = adjoint_rollout_gradient(&rollout, &obj);
    let t_total = STEPS as f64 * DT;
    let expected = -TAU * t_total / (IZZ * IZZ);
    let rel = (g.d_inertia[0][6] - expected).abs() / expected.abs();
    assert!(
        rel <= 1e-12,
        "dJ/dIzz = {} vs closed form {expected} (rel {rel:.3e})",
        g.d_inertia[0][6]
    );
}
