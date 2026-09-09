//! The launch bill, pinned.
//!
//! The device collector is launch-bound, so the number of host calls a step
//! costs is a contract, not an implementation detail: a change that quietly
//! turns one launch per control period back into twenty is a regression even
//! though every state it produces is bit-identical. These tests pin the
//! counts, including the one that is the whole diagnosis — that the bill does
//! not shrink when the batch widens.
//!
//! Counts are what `BatchSim` issues, which is backend-independent, so the
//! host mirror is a valid witness for the CUDA path's launch sequence.

#![cfg(feature = "cuda-host")]

use phyz_gpu::cuda::census;
use phyz_gpu::cuda::{BatchSim, HostBackend, StepMode};
use phyz_gpu::{BodyContactGains, PdDof};
use phyz_math::{GRAVITY, Vec3};
use phyz_model::Model;

fn ant_or_skip() -> Option<Model> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/ant.xml");
    match phyz_mjcf::MjcfLoader::from_file(path) {
        Ok(l) => {
            let mut m = l.build_model();
            m.gravity = Vec3::new(0.0, 0.0, -GRAVITY);
            Some(m)
        }
        Err(_) => {
            eprintln!("skipping launch census: {path} not found");
            None
        }
    }
}

fn sim_for(model: &Model, nworld: usize) -> BatchSim<HostBackend> {
    let mut sim = phyz_gpu::HostBatchSimulator::new(model.clone(), nworld).expect("host sim");
    let gains = BodyContactGains::uniform_frequency(model, 100.0, 1.0);
    sim.enable_contact_impulse(0.0, 0.8, &gains, &[], None)
        .expect("impulse contact");
    let pd: Vec<PdDof> = (6..model.nv)
        .map(|i| PdDof {
            q_index: i,
            v_index: i,
            kp: 40.0,
            kd: 2.0,
            max_force: 60.0,
        })
        .collect();
    sim.enable_pd_control(&pd).expect("pd");
    sim.set_position_targets(&vec![vec![0.0; pd.len()]; nworld])
        .expect("targets");
    sim.set_controls(&vec![vec![0.0; model.nv]; nworld]);
    let mut start = model.default_state();
    start.q[5] = 0.75;
    sim.load_states(&vec![start; nworld]);
    // The first step allocates the per-step caches; keep that out of the
    // counts, which are about the steady state.
    sim.step();
    sim
}

fn count(sim: &mut BatchSim<HostBackend>, f: impl FnOnce(&mut BatchSim<HostBackend>)) -> u64 {
    census::reset();
    f(sim);
    census::snapshot().kernel_launches
}

/// The default mode issues a control period as ONE launch — but only if the
/// caller asks for the period. Twenty `step()`s cost twenty launches for the
/// identical physics, which is the caller-side cliff the collector sits on.
#[test]
fn fused_period_is_one_launch_only_via_step_many() {
    let Some(model) = ant_or_skip() else { return };
    let mut sim = sim_for(&model, 8);
    sim.set_step_mode(StepMode::Fused);

    assert_eq!(count(&mut sim, |s| s.step()), 1, "a fused step is one launch");
    assert_eq!(
        count(&mut sim, |s| s.step_many(20).unwrap()),
        1,
        "a fused control period is one launch"
    );
    assert_eq!(
        count(&mut sim, |s| {
            for _ in 0..20 {
                s.step();
            }
        }),
        20,
        "twenty step()s cost twenty launches — the same physics, 20x the bill"
    );
}

/// The unfused sequence is `PD + ABA + sweeps x [contact, ABA] + integrate`.
/// Pinning it is what makes the fused row above mean something.
#[test]
fn unfused_step_is_three_plus_two_per_sweep() {
    let Some(model) = ant_or_skip() else { return };
    let mut sim = sim_for(&model, 8);
    let want = 3 + 2 * sim.contact_sweeps as u64;
    for mode in [StepMode::Unfused, StepMode::Fission] {
        sim.set_step_mode(mode);
        sim.step();
        assert_eq!(
            count(&mut sim, |s| s.step()),
            want,
            "{mode:?}: PD + ABA + {} x [contact, ABA] + integrate",
            sim.contact_sweeps
        );
    }
}

/// The finding: widening the batch adds THREADS to a launch, never launches.
/// The host bill for a control period is the same at 8 worlds as at 256, so
/// no amount of extra width amortises it — which is exactly what "launch
/// bound" means, said as a count instead of a duration.
#[test]
fn the_launch_bill_is_flat_in_the_world_count() {
    let Some(model) = ant_or_skip() else { return };
    let mut narrow = sim_for(&model, 8);
    let mut wide = sim_for(&model, 256);
    for mode in [StepMode::Fused, StepMode::Unfused] {
        narrow.set_step_mode(mode);
        wide.set_step_mode(mode);
        narrow.step();
        wide.step();
        let a = count(&mut narrow, |s| {
            for _ in 0..20 {
                s.step();
            }
        });
        let b = count(&mut wide, |s| {
            for _ in 0..20 {
                s.step();
            }
        });
        assert_eq!(a, b, "{mode:?}: 32x the worlds must cost the same launches");
        assert!(a > 0, "{mode:?}: a control period must launch something");
    }
}
