//! GPU multi-DOF joint support, checked against the f64 CPU reference.
//!
//! Before this landed, `pack_bodies` mapped ball and free joints to *fixed*,
//! so every floating-base model — which is all four RL benchmarks — silently
//! simulated a different robot on the GPU than on the CPU. These tests exist
//! so that cannot regress.

use phyz_gpu::GpuBatchSimulator;
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Joint, Model, ModelBuilder, State};
use phyz_rigid::{aba, semi_implicit_euler};

fn inertia(mass: f64, i: f64) -> SpatialInertia {
    SpatialInertia::new(
        mass,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(i, i, i)),
    )
}

/// A single free-floating body — the simplest model the GPU used to get wrong.
fn free_body() -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_body(
            "root",
            -1,
            Joint::free(SpatialTransform::identity()),
            inertia(2.0, 0.1),
        )
        .build()
}

/// Floating base with a revolute limb: exercises the multi-DOF backward pass
/// propagating an articulated inertia into a 6-DOF parent.
fn free_base_with_limb() -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_body(
            "torso",
            -1,
            Joint::free(SpatialTransform::identity()),
            inertia(5.0, 0.3),
        )
        .add_revolute_body(
            "limb",
            0,
            SpatialTransform::from_translation(Vec3::new(0.2, 0.0, 0.0)),
            inertia(1.0, 0.05),
        )
        .build()
}

fn ball_joint_model() -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_body(
            "ball",
            -1,
            Joint::spherical(SpatialTransform::identity()),
            SpatialInertia::new(
                1.0,
                Vec3::new(0.0, 0.0, -0.3),
                Mat3::from_diagonal(&Vec3::new(0.05, 0.05, 0.02)),
            ),
        )
        .build()
}

/// Step both backends once from `state` and compare.
///
/// The tolerance is loose because the GPU runs f32 against the CPU's f64 —
/// that gap is the documented precision policy, not a bug.
fn compare(model: &Model, state: &State, tol: f64, label: &str) {
    let Ok(sim) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping {label}: no GPU adapter");
        return;
    };

    sim.load_states(&[state.clone()]);
    sim.step();
    let gpu = sim.readback_states();

    let mut cpu = state.clone();
    let qdd = aba(model, &cpu);
    semi_implicit_euler(model, &mut cpu, qdd.as_slice(), model.dt);

    for j in 0..model.nq {
        let d = (gpu[0].q[j] - cpu.q[j]).abs();
        assert!(
            d < tol,
            "{label}: q[{j}] gpu={:.9} cpu={:.9} diff={d:.2e}",
            gpu[0].q[j],
            cpu.q[j]
        );
    }
    for j in 0..model.nv {
        let d = (gpu[0].v[j] - cpu.v[j]).abs();
        assert!(
            d < tol,
            "{label}: v[{j}] gpu={:.9} cpu={:.9} diff={d:.2e}",
            gpu[0].v[j],
            cpu.v[j]
        );
    }
}

#[test]
fn free_joint_matches_cpu() {
    let model = free_body();
    let mut s = model.default_state();
    s.q[2] = 1.0; // 1 m up
    s.v[0] = 0.3; // spin about x
    s.v[5] = -0.2; // moving along body z
    compare(&model, &s, 1e-4, "free_joint");
}

/// A free body under gravity alone must accelerate downward. This is the check
/// that would have caught the weld-to-fixed bug: a welded root does not move.
#[test]
fn free_body_actually_falls_on_gpu() {
    let model = free_body();
    let mut s = model.default_state();
    s.q[2] = 5.0;

    let Ok(sim) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.load_states(&[s.clone()]);
    for _ in 0..200 {
        sim.step();
    }
    let out = sim.readback_states();

    let dropped = 5.0 - out[0].q[2];
    // 0.2 s of free fall ≈ 0.196 m.
    assert!(
        dropped > 0.15 && dropped < 0.25,
        "expected ~0.196 m of free fall, got {dropped:.4}"
    );
}

#[test]
fn ball_joint_matches_cpu() {
    let model = ball_joint_model();
    let mut s = model.default_state();
    s.q[0] = 0.2;
    s.q[1] = -0.1;
    s.v[2] = 0.5;
    compare(&model, &s, 1e-4, "ball_joint");
}

#[test]
fn floating_base_with_limb_matches_cpu() {
    let model = free_base_with_limb();
    let mut s = model.default_state();
    s.q[2] = 0.8;
    s.q[6] = 0.4; // limb angle
    s.v[1] = 0.2;
    s.v[3] = 0.1;
    s.v[6] = -0.3;
    // Articulated inertia propagation through a 6x6 solve loses a little more
    // f32 precision than a scalar joint does.
    compare(&model, &s, 5e-4, "floating_base_with_limb");
}

/// The whole point of the batch backend: environments must stay independent.
#[test]
fn floating_base_batch_stays_independent() {
    let model = free_base_with_limb();
    let n = 8;
    let Ok(sim) = GpuBatchSimulator::new(model.clone(), n) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let states: Vec<State> = (0..n)
        .map(|i| {
            let mut s = model.default_state();
            s.q[2] = 1.0 + i as f64 * 0.1;
            s.q[6] = i as f64 * 0.05;
            s
        })
        .collect();

    sim.load_states(&states);
    for _ in 0..50 {
        sim.step();
    }
    let out = sim.readback_states();

    for i in 0..n {
        assert!(out[i].q.as_slice().iter().all(|x| x.is_finite()));
        if i > 0 {
            assert!(
                (out[i].q[2] - out[i - 1].q[2]).abs() > 1e-5,
                "envs {} and {i} collapsed to the same state",
                i - 1
            );
        }
    }
}

/// Ant is the credibility model; it must run on the GPU without NaN and with
/// its root actually moving.
#[test]
fn ant_runs_on_gpu() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/ant.xml");
    let Ok(loader) = phyz_mjcf::MjcfLoader::from_file(path) else {
        eprintln!("skipping: {path} not found");
        return;
    };
    let model = loader.build_model();

    let Ok(sim) = GpuBatchSimulator::new(model.clone(), 64) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut s = model.default_state();
    s.q[2] = 0.75;
    let states = vec![s; 64];
    sim.load_states(&states);
    for _ in 0..100 {
        sim.step();
    }
    let out = sim.readback_states();

    assert!(
        out.iter()
            .all(|s| s.q.as_slice().iter().all(|x| x.is_finite())),
        "ant diverged on GPU"
    );
    assert!(
        out[0].q[2] < 0.75,
        "ant torso should fall without contact; z = {}",
        out[0].q[2]
    );
}
