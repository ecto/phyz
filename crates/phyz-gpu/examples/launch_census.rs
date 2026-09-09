//! How many host-issued calls a physics step and a control period cost.
//!
//! ```text
//! cargo run --release -p phyz-gpu --features cuda-host      --example launch_census -- [worlds..]
//! cargo run --release -p phyz-gpu --features cuda,cuda-host --example launch_census -- [worlds..]
//! ```
//!
//! The example is gated on `cuda-host` because that is the backend it can
//! always fall back to; enable `cuda` alongside it on a machine with a device
//! and it runs on the GPU instead, which is the only way to get the graph
//! rows.
//!
//! The device collector this crate feeds is launch-bound: 16x the worlds
//! costs 1.6x the time, and the host's issue time dominates. The census is
//! the exact statement of why. It counts CALLS — kernel launches, graph
//! replays, transfers, syncs — for one physics step and for one control
//! period (`control_every` physics steps), at several world counts.
//!
//! Two things the table is meant to show:
//!
//! * the launch bill is **flat in the world count**. Widening the batch adds
//!   threads to a launch, never launches, so every column at 4096 worlds is
//!   the column at 512. That is the launch-bound diagnosis stated as a count
//!   rather than as a duration.
//! * what capture removes. `step()` under graphs replays a ONE-step
//!   recording, so a 20-step control period is still 20 host calls;
//!   `step_many(20)` records the period once and replays it with ONE. The
//!   ratio between those two rows is the multiplier a caller gets for free
//!   by switching call shape — no kernel changes at all.
//!
//! Counts are backend-independent: `BatchSim` issues the same sequence to
//! every backend, so a census taken on the host mirror is the count the CUDA
//! path issues. Only the graph rows need a real device, since the host mirror
//! does not capture.

use phyz_gpu::cuda::census::{self, LaunchCensus};
use phyz_gpu::cuda::{BatchSim, KernelBackend, StepMode};
use phyz_gpu::{BodyContactGains, PdDof};
use phyz_math::{GRAVITY, Vec3};
use phyz_model::Model;

fn ant() -> Result<Model, String> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/ant.xml");
    let loader = phyz_mjcf::MjcfLoader::from_file(path).map_err(|e| format!("{path}: {e}"))?;
    let mut m = loader.build_model();
    m.gravity = Vec3::new(0.0, 0.0, -GRAVITY);
    Ok(m)
}

fn pd_dofs(model: &Model) -> Vec<PdDof> {
    (0..model.nv)
        .skip(6.min(model.nv))
        .map(|i| PdDof {
            q_index: i,
            v_index: i,
            kp: 40.0,
            kd: 2.0,
            max_force: 60.0,
        })
        .collect()
}

/// Run `f` and report what it issued.
fn measure<B: KernelBackend>(sim: &mut BatchSim<B>, f: impl FnOnce(&mut BatchSim<B>)) -> LaunchCensus {
    census::reset();
    f(sim);
    census::snapshot()
}

fn row(label: &str, c: &LaunchCensus) {
    println!(
        "  {:<26} {:>7} {:>7} {:>8} {:>8} {:>7} {:>7}",
        label,
        c.kernel_launches,
        c.graph_launches,
        c.graph_captures,
        c.uploads,
        c.downloads,
        c.syncs,
    );
}

fn header() {
    println!(
        "  {:<26} {:>7} {:>7} {:>8} {:>8} {:>7} {:>7}",
        "", "kernel", "graph", "captures", "uploads", "dnloads", "syncs"
    );
}

fn census_at<B: KernelBackend>(
    mut sim: BatchSim<B>,
    model: &Model,
    nworld: usize,
    control_every: usize,
) -> Result<(), String> {
    let gains = BodyContactGains::uniform_frequency(model, 100.0, 1.0);
    sim.enable_contact_impulse(0.0, 0.8, &gains, &[], None)?;
    let dofs = pd_dofs(model);
    sim.enable_pd_control(&dofs)?;
    sim.set_position_targets(&vec![vec![0.0; dofs.len()]; nworld])?;
    sim.set_controls(&vec![vec![0.0; model.nv]; nworld]);

    let mut start = model.default_state();
    if model.nq >= 6 {
        start.q[5] = 0.75;
    }
    let states = vec![start; nworld];
    sim.load_states(&states);

    let sweeps = sim.contact_sweeps;
    let can_capture = sim.graphs_enabled();
    println!(
        "\n{nworld} worlds | sweeps {sweeps} | control_every {control_every} | capture {}",
        if can_capture { "available" } else { "UNAVAILABLE (host mirror)" }
    );

    // Warm up so first-touch allocation is not counted as steady-state work.
    sim.set_graphs_enabled(false);
    for _ in 0..4 {
        sim.step();
    }

    // All three issue modes. They run the same arithmetic in the same order
    // and are bit-identical to each other; they differ only in how many host
    // calls that costs, which is the whole subject here.
    for mode in [StepMode::Fused, StepMode::Fission, StepMode::Unfused] {
        sim.set_step_mode(mode);
        println!("\n  -- {mode:?} --");
        header();
        sim.set_graphs_enabled(false);
        sim.step();

        let c = measure(&mut sim, |s| s.step());
        row("step() no graphs", &c);
        let c = measure(&mut sim, |s| {
            for _ in 0..control_every {
                s.step();
            }
        });
        row("period, step() no graphs", &c);
        let c = measure(&mut sim, |s| {
            s.step_many(control_every).expect("step_many");
        });
        row("period, step_many no graphs", &c);

        if can_capture {
            sim.set_graphs_enabled(true);
            // Pay each capture once outside the measured span.
            sim.step();
            let c = measure(&mut sim, |s| s.step());
            row("step() captured", &c);
            let c = measure(&mut sim, |s| {
                for _ in 0..control_every {
                    s.step();
                }
            });
            row("period, step() captured", &c);
            sim.step_many(control_every).expect("warm capture");
            let c = measure(&mut sim, |s| {
                s.step_many(control_every).expect("step_many");
            });
            row("period, step_many captured", &c);
        }
    }
    sim.set_step_mode(StepMode::Fused);
    println!();
    header();

    // The readback the collector pays once per control period, for scale.
    let c = measure(&mut sim, |s| {
        let _ = s.readback_states();
    });
    row("readback_states()", &c);
    Ok(())
}

fn main() {
    let worlds: Vec<usize> = {
        let v: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
        if v.is_empty() { vec![512, 4096] } else { v }
    };
    let control_every: usize = std::env::var("CONTROL_EVERY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);

    let model = match ant() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    println!("model: ant ({} bodies, nv {})", model.nbodies(), model.nv);

    for nworld in worlds {
        #[cfg(feature = "cuda")]
        let r = phyz_gpu::CudaBatchSimulator::new(model.clone(), nworld)
            .and_then(|sim| census_at(sim, &model, nworld, control_every));
        #[cfg(all(not(feature = "cuda"), feature = "cuda-host"))]
        let r = phyz_gpu::HostBatchSimulator::new(model.clone(), nworld)
            .and_then(|sim| census_at(sim, &model, nworld, control_every));
        #[cfg(not(any(feature = "cuda", feature = "cuda-host")))]
        let r: Result<(), String> = Err("build with --features cuda or cuda-host".into());

        if let Err(e) = r {
            eprintln!("launch_census: {e}");
            std::process::exit(1);
        }
    }
}
