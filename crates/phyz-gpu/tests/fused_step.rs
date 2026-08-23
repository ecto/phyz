//! The fused impulse step is the unfused sequence, bit for bit.
//!
//! `phyz_step_impulse` runs PD, the leading ABA, `sweeps` x [contact, ABA]
//! and integrate in one thread per world, carrying the ABA factorisation
//! across the sweeps of a step. The factorisation is a function of `q`, `v`
//! and `dt`, none of which move within a step — so reusing it is not an
//! approximation and the states must agree to the bit, not to a tolerance.
#![cfg(feature = "cuda-host")]

use phyz_gpu::cuda::HostBatchSimulator;
use phyz_gpu::{BodyContactGains, PdDof};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Joint, JointType, Model, ModelBuilder, State};

fn box_inertia(mass: f64, h: Vec3) -> SpatialInertia {
    let (lx, ly, lz) = (2.0 * h.x, 2.0 * h.y, 2.0 * h.z);
    SpatialInertia::new(
        mass,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(
            mass / 12.0 * (ly * ly + lz * lz),
            mass / 12.0 * (lx * lx + lz * lz),
            mass / 12.0 * (lx * lx + ly * ly),
        )),
    )
}

fn hinge(offset_z: f64) -> Joint {
    Joint {
        joint_type: JointType::Revolute,
        parent_to_joint: SpatialTransform::new(Mat3::identity(), Vec3::new(0.0, 0.0, offset_z)),
        axis: Vec3::new(0.0, 1.0, 0.0),
        ..Joint::default()
    }
}

/// A two-link arm on a free base with a box foot: a free joint, revolutes,
/// PD servos and a real contact manifold, which is the shape RL runs.
fn rig() -> Model {
    let half = Vec3::new(0.12, 0.1, 0.06);
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(0.001)
        .add_body(
            "base",
            -1,
            Joint::free(SpatialTransform::identity()),
            box_inertia(3.0, half),
        )
        .add_body(
            "link",
            0,
            hinge(-0.2),
            box_inertia(1.0, half),
        )
        .add_body(
            "foot",
            1,
            hinge(-0.2),
            box_inertia(0.5, half),
        )
        .build();
    for b in &mut model.bodies {
        b.geometry = Some(phyz_model::Geometry::Box { half_extents: half });
    }
    model
}

fn states(model: &Model, n: usize) -> Vec<State> {
    (0..n)
        .map(|i| {
            let mut s = model.default_state();
            s.q[5] = 0.6 + 0.05 * i as f64;
            s.q[1] = 0.2;
            s.q[6] = 0.3;
            s.q[7] = -0.4;
            s.v[3] = 0.5 - 0.2 * i as f64;
            s.v[1] = 1.0;
            s
        })
        .collect()
}

fn pd_dofs(model: &Model) -> Vec<PdDof> {
    (6..model.nv)
        .map(|i| PdDof {
            q_index: i,
            v_index: i,
            kp: 30.0,
            kd: 1.5,
            max_force: 40.0,
        })
        .collect()
}

fn run(fused: bool, model: &Model, init: &[State], steps: usize, chunk: usize) -> Vec<State> {
    let mut sim = HostBatchSimulator::new(model.clone(), init.len()).expect("simulator");
    sim.set_fused_step_enabled(fused);
    assert_eq!(sim.fused_step_enabled(), false, "no contact enabled yet");
    let g = BodyContactGains::uniform_frequency(model, 60.0, 1.0);
    sim.enable_contact_impulse(0.0, 0.7, &g, &[], None).unwrap();
    sim.enable_pd_control(&pd_dofs(model)).unwrap();
    sim.set_position_targets(&vec![vec![0.1; pd_dofs(model).len()]; init.len()])
        .unwrap();
    assert_eq!(sim.fused_step_enabled(), fused);
    sim.load_states(init);
    for _ in 0..(steps / chunk) {
        sim.step_many(chunk).unwrap();
    }
    sim.readback_states()
}

#[test]
fn fused_impulse_step_is_bit_identical() {
    let model = rig();
    let init = states(&model, 3);
    let plain = run(false, &model, &init, 600, 1);
    for (label, chunk) in [("fused(1)", 1usize), ("fused(20)", 20)] {
        let got = run(true, &model, &init, 600, chunk);
        for (w, (a, b)) in plain.iter().zip(&got).enumerate() {
            for j in 0..model.nq {
                assert_eq!(
                    a.q[j].to_bits(),
                    b.q[j].to_bits(),
                    "{label} world {w} q[{j}]: {} vs {}",
                    a.q[j],
                    b.q[j]
                );
            }
            for j in 0..model.nv {
                assert_eq!(
                    a.v[j].to_bits(),
                    b.v[j].to_bits(),
                    "{label} world {w} v[{j}]: {} vs {}",
                    a.v[j],
                    b.v[j]
                );
            }
        }
    }
}

/// The rig has to actually be in contact, or the test above compares two
/// free-fall trajectories and proves nothing about the solver.
#[test]
fn the_rig_touches_the_ground() {
    let model = rig();
    let init = states(&model, 1);
    let mut sim = HostBatchSimulator::new(model.clone(), 1).expect("simulator");
    let g = BodyContactGains::uniform_frequency(&model, 60.0, 1.0);
    sim.enable_contact_impulse(0.0, 0.7, &g, &[], None).unwrap();
    sim.load_states(&init);
    let mut touched = false;
    for _ in 0..600 {
        sim.step();
        touched |= sim.readback_contacts().unwrap()[0]
            .iter()
            .any(|c| c.touching);
    }
    assert!(touched, "no contact in the fused-step fixture");
}
