//! Articulated inertia across a translated joint, against the CPU and the
//! analytic double pendulum.
//!
//! The ABA shader's `build_motion_transform` held skew(p)ᵀ where it meant
//! skew(p), so the articulated inertia a child passes up through a joint with
//! a non-zero `parent_to_joint` offset carried the wrong sign on its
//! translation block. Every single-step test tolerated the per-step error;
//! over a 200-step swing the trajectory left the CPU's by 0.2 rad. This test
//! is the one that would have caught it: two uniform rods, the second hung
//! one metre below the first, checked against both the closed-form equations
//! of motion (one step) and the f64 CPU rollout.

use phyz_gpu::GpuBatchSimulator;
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Model, ModelBuilder};
use phyz_rigid::{aba, semi_implicit_euler};

fn rod(m: f64, half_len: f64) -> SpatialInertia {
    let len = 2.0 * half_len;
    let i = m * len * len / 12.0;
    SpatialInertia::new(
        m,
        Vec3::new(0.0, -half_len, 0.0),
        Mat3::from_diagonal(&Vec3::new(i, 0.0, i)),
    )
}

/// Two 1 kg, 1 m rods; the lower joint sits at the upper rod's tip.
fn offset_double_pendulum() -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001)
        .add_revolute_body("upper", -1, SpatialTransform::identity(), rod(1.0, 0.5))
        .add_revolute_body(
            "lower",
            0,
            SpatialTransform::from_translation(Vec3::new(0.0, -1.0, 0.0)),
            rod(1.0, 0.5),
        )
        .build()
}

/// Closed-form joint accelerations for the equal-rod planar double pendulum
/// at rest with both links tilted by `theta` (relative elbow angle zero):
/// M = [[4/3, 1/2], [1/2, 1/3]], tau = -g sin(theta) [3/2, 1/2] in absolute
/// angles; the elbow acceleration is the difference.
fn analytic_qdd(theta: f64) -> [f64; 2] {
    let (m11, m12, m22) = (4.0 / 3.0, 0.5, 1.0 / 3.0);
    let (t1, t2) = (-1.5 * GRAVITY * theta.sin(), -0.5 * GRAVITY * theta.sin());
    let det = m11 * m22 - m12 * m12;
    let a1 = (m22 * t1 - m12 * t2) / det;
    let a2 = (m11 * t2 - m12 * t1) / det;
    [a1, a2 - a1]
}

#[test]
fn analytic_double_pendulum_matches_gpu() {
    let model = offset_double_pendulum();
    let Ok(sim) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let theta = 0.4;
    let mut s = model.default_state();
    s.q[0] = theta;

    // The CPU agrees with the closed form; that is the reference's warrant.
    let cpu = aba(&model, &s);
    let want = analytic_qdd(theta);
    assert!((cpu[0] - want[0]).abs() < 1e-9 && (cpu[1] - want[1]).abs() < 1e-9);

    sim.load_states(std::slice::from_ref(&s));
    sim.step();
    let out = &sim.readback_states()[0];
    let got = [out.v[0] / model.dt, out.v[1] / model.dt];
    for j in 0..2 {
        assert!(
            (got[j] - want[j]).abs() < 1e-3,
            "qdd[{j}]: gpu {:.5} vs analytic {:.5}",
            got[j],
            want[j]
        );
    }
}

#[test]
fn offset_double_pendulum_trajectory_matches_cpu() {
    let model = offset_double_pendulum();
    let Ok(sim) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut cpu = model.default_state();
    cpu.q[0] = 0.4;
    cpu.q[1] = -0.3;
    sim.load_states(std::slice::from_ref(&cpu));
    for _ in 0..200 {
        sim.step();
        let qdd = aba(&model, &cpu);
        semi_implicit_euler(&model, &mut cpu, qdd.as_slice(), model.dt);
    }
    let out = &sim.readback_states()[0];
    for j in 0..2 {
        let d = (out.q[j] - cpu.q[j]).abs();
        assert!(
            d < 1e-3,
            "q[{j}] after 200 steps: gpu {} cpu {} diff {d:.2e}",
            out.q[j],
            cpu.q[j]
        );
    }
}
