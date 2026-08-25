//! The fused and FISSIONED impulse steps are the unfused sequence, bit for
//! bit.
//!
//! `phyz_step_impulse` runs PD, the leading ABA, `sweeps` x [contact, ABA]
//! and integrate in one thread per world, carrying the ABA factorisation
//! across the sweeps of a step. The factorisation is a function of `q`, `v`
//! and `dt`, none of which move within a step — so reusing it is not an
//! approximation and the states must agree to the bit, not to a tolerance.
//!
//! [`StepMode::Fission`] carries the same two caches in global
//! structure-of-arrays buffers instead of the thread's local stack, one stage
//! kernel per pass. Same expressions, same order, same f32 bits through
//! memory — so it is held to the same bit-identity, and to it with the
//! compound deck-and-rider surface in the loop.
#![cfg(feature = "cuda-host")]

use phyz_gpu::contact_pipeline::BodyPlane;
use phyz_gpu::cuda::{HostBatchSimulator, StepMode};
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
        .add_body("link", 0, hinge(-0.2), box_inertia(1.0, half))
        .add_body("foot", 1, hinge(-0.2), box_inertia(0.5, half))
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

/// A deck with two faces — one flat, one tilted up at 10 deg like a
/// kicktail — and a box resting on them. The compound-surface shape whose
/// clipped manifold the fused step now carries across the sweeps.
fn deck_rig() -> Model {
    let deck = Vec3::new(0.4, 0.1, 0.02);
    let rider = Vec3::new(0.08, 0.08, 0.06);
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(0.001)
        .add_body(
            "deck",
            -1,
            Joint::free(SpatialTransform::identity()),
            box_inertia(2.0, deck),
        )
        .add_body(
            "rider",
            -1,
            Joint::free(SpatialTransform::identity()),
            box_inertia(4.0, rider),
        )
        .build();
    model.bodies[0].geometry = Some(phyz_model::Geometry::Box { half_extents: deck });
    model.bodies[1].geometry = Some(phyz_model::Geometry::Box {
        half_extents: rider,
    });
    model
}

fn deck_faces() -> Vec<BodyPlane> {
    let deck = Vec3::new(0.4, 0.1, 0.02);
    let c = 0.9848f64;
    let s = 0.1736f64;
    let tilt = Mat3::from_cols(
        Vec3::new(c, 0.0, s),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(-s, 0.0, c),
    );
    vec![
        BodyPlane {
            body: 0,
            offset: deck.z,
            max_depth: 0.05,
            half_x: 0.7 * deck.x,
            half_y: deck.y,
            exclude: vec![0],
            tilt: Mat3::identity(),
            center: Vec3::zeros(),
        },
        BodyPlane {
            body: 0,
            offset: deck.z,
            max_depth: 0.05,
            half_x: 0.3 * deck.x,
            half_y: deck.y,
            exclude: vec![0],
            tilt,
            center: Vec3::new(-0.7 * deck.x, 0.0, 0.0),
        },
    ]
}

/// The rider starts just above the flat face, drifting toward the tilted one,
/// so both faces enter the pooled manifold over the run.
fn deck_states(model: &Model, n: usize) -> Vec<State> {
    (0..n)
        .map(|i| {
            let mut st = model.default_state();
            st.q[5] = 0.02;
            st.q[11] = 0.14 + 0.005 * i as f64;
            st.q[9] = 0.05 * i as f64;
            st.v[9] = -0.4 - 0.05 * i as f64;
            st
        })
        .collect()
}

fn run(mode: StepMode, model: &Model, init: &[State], steps: usize, chunk: usize) -> Vec<State> {
    run_with(mode, model, init, steps, chunk, &[])
}

fn run_with(
    mode: StepMode,
    model: &Model,
    init: &[State],
    steps: usize,
    chunk: usize,
    planes: &[BodyPlane],
) -> Vec<State> {
    let mut sim = HostBatchSimulator::new(model.clone(), init.len()).expect("simulator");
    sim.set_step_mode(mode);
    assert_eq!(sim.fused_step_enabled(), false, "no contact enabled yet");
    let g = BodyContactGains::uniform_frequency(model, 60.0, 1.0);
    sim.enable_contact_impulse(0.0, 0.7, &g, planes, None)
        .unwrap();
    sim.enable_pd_control(&pd_dofs(model)).unwrap();
    sim.set_position_targets(&vec![vec![0.1; pd_dofs(model).len()]; init.len()])
        .unwrap();
    assert_eq!(sim.fused_step_enabled(), mode == StepMode::Fused);
    assert_eq!(sim.fission_enabled(), mode == StepMode::Fission);
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
    let plain = run(StepMode::Unfused, &model, &init, 600, 1);
    for (label, mode, chunk) in [
        ("fused(1)", StepMode::Fused, 1usize),
        ("fused(20)", StepMode::Fused, 20),
        ("fission(1)", StepMode::Fission, 1),
        ("fission(20)", StepMode::Fission, 20),
    ] {
        let got = run(mode, &model, &init, 600, chunk);
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

/// The same bit-identity, with a compound BODY-ATTACHED SURFACE in the loop.
///
/// The face pass is where a rig like ipse's K1 spends its step — 45 deck and
/// kicktail faces, Sutherland-Hodgman-clipped per body — and the fused step
/// carries that clipped manifold across the sweeps rather than re-searching
/// it 16 times. The search reads `w_rot`, `w_pos` and the geometry table, all
/// of them `q`-only, so this is bit-identity and not a tolerance either.
#[test]
fn fused_step_with_body_faces_is_bit_identical() {
    let model = deck_rig();
    let init = deck_states(&model, 3);
    let planes = deck_faces();
    let plain = run_with(StepMode::Unfused, &model, &init, 600, 1, &planes);
    for (label, mode, chunk) in [
        ("fused(1)", StepMode::Fused, 1usize),
        ("fused(20)", StepMode::Fused, 20),
        ("fission(1)", StepMode::Fission, 1),
        ("fission(20)", StepMode::Fission, 20),
    ] {
        let got = run_with(mode, &model, &init, 600, chunk, &planes);
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

/// ...and the faces have to actually carry something, or the test above is
/// two more free-fall trajectories.
#[test]
fn the_rig_touches_a_face() {
    let model = deck_rig();
    let init = deck_states(&model, 3);
    let mut sim = HostBatchSimulator::new(model.clone(), init.len()).expect("simulator");
    let g = BodyContactGains::uniform_frequency(&model, 60.0, 1.0);
    sim.enable_contact_impulse(0.0, 0.7, &g, &deck_faces(), None)
        .unwrap();
    sim.load_states(&init);
    let mut touched = 0usize;
    for _ in 0..600 {
        sim.step();
        touched += sim
            .readback_contacts()
            .unwrap()
            .iter()
            .flatten()
            .filter(|c| c.plane.touching && c.plane.points > 0)
            .count();
    }
    assert!(touched > 0, "no face contact in the fused-step fixture");
}
