//! Passive joint springs on the GPU, checked against the f64 CPU reference.
//!
//! `Joint::stiffness`/`spring_ref` is what a skateboard truck's bushing is —
//! and until this landed, the GPU path silently dropped it: `pack_bodies`
//! carried only `damping`, so a sprung model trained on the GPU was a
//! different mechanism than the one the CPU (and the deploy stack) would
//! integrate. Same class of bug as the weld-to-fixed one `multidof_vs_cpu.rs`
//! guards against, same style of guard.

use phyz_gpu::GpuBatchSimulator;
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Joint, JointType, Model, ModelBuilder, State};
use phyz_rigid::{aba, semi_implicit_euler};

/// A horizontal-axis pendulum sprung toward `spring_ref`, damped enough to
/// settle inside the test horizon.
fn sprung_pendulum(stiffness: f64, spring_ref: f64) -> Model {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_body(
            "arm",
            -1,
            Joint {
                joint_type: JointType::Revolute,
                parent_to_joint: SpatialTransform::identity(),
                axis: Vec3::new(0.0, 1.0, 0.0),
                stiffness,
                spring_ref,
                damping: 0.4,
                name: "hinge".into(),
                ..Joint::default()
            },
            SpatialInertia::new(
                1.0,
                Vec3::new(0.0, 0.0, -0.3),
                Mat3::from_diagonal(&Vec3::new(0.05, 0.05, 0.02)),
            ),
        )
        .build();
    model.dt = 0.001;
    model
}

fn cpu_rollout(model: &Model, state: &State, steps: usize) -> State {
    let mut s = state.clone();
    for _ in 0..steps {
        let qdd = aba(model, &s);
        semi_implicit_euler(model, &mut s, qdd.as_slice(), model.dt);
    }
    s
}

/// Multi-step trajectory agreement, f32-vs-f64 tolerance.
#[test]
fn a_sprung_pendulum_tracks_the_cpu() {
    let model = sprung_pendulum(8.0, 0.4);
    let mut s = model.default_state();
    s.q[0] = -0.9;

    let Ok(sim) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.load_states(std::slice::from_ref(&s));
    for _ in 0..500 {
        sim.step();
    }
    let gpu = sim.readback_states();
    let cpu = cpu_rollout(&model, &s, 500);

    let d = (gpu[0].q[0] - cpu.q[0]).abs();
    assert!(
        d < 1e-3,
        "sprung pendulum diverged: gpu q={:.6} cpu q={:.6} diff={d:.2e}",
        gpu[0].q[0],
        cpu.q[0]
    );
}

/// The behavioral claim, independent of the CPU: a damped sprung joint
/// settles at `spring_ref`, not at the gravity minimum. This is the test
/// that fails if the packed words are read but ignored, or the sign flips.
#[test]
fn the_spring_holds_the_arm_off_gravity_minimum() {
    let spring_ref = 0.5;
    let model = sprung_pendulum(60.0, spring_ref);
    let mut s = model.default_state();
    s.q[0] = spring_ref; // start at the sprung rest to isolate holding

    let Ok(sim) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.load_states(std::slice::from_ref(&s));
    for _ in 0..3000 {
        sim.step();
    }
    let out = sim.readback_states();

    // Equilibrium: k(q - ref) = m g l cos-ish term; with k=60 the offset from
    // ref is small but nonzero. Without the spring the arm hangs at -pi/2.
    let q = out[0].q[0];
    assert!(
        (q - spring_ref).abs() < 0.1,
        "arm settled at {q:.3}, spring_ref is {spring_ref} — the spring is not acting"
    );

    // And the CPU agrees about where it settles.
    let cpu = cpu_rollout(&model, &s, 3000);
    assert!(
        (q - cpu.q[0]).abs() < 5e-3,
        "settle point disagrees: gpu {q:.4} cpu {:.4}",
        cpu.q[0]
    );
}
