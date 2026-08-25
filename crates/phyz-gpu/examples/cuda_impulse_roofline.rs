//! Where the fused impulse step's time actually goes on an NVIDIA GPU.
//!
//! ```text
//! cargo run --release -p phyz-gpu --features cuda --example cuda_impulse_roofline -- [model] [maxworlds]
//! ```
//!
//! The fused step (`phyz_step_impulse`) is one thread per world. That makes
//! the world count the *only* source of parallelism, so the shape of
//! `ms/step` against the world count says which wall the kernel is against:
//!
//! * flat, then linear   — the flat part is unsaturated parallelism (the
//!                         device has more slots than we gave it) and the
//!                         knee is where the machine fills;
//! * linear from world 1 — there is no slack at all: one thread per world is
//!                         already too little work-per-launch to hide any
//!                         latency, and per-thread serial arithmetic is the
//!                         whole cost.
//!
//! The per-world-step microsecond column is the same number normalised: a
//! device with spare capacity gets *cheaper* per world as worlds are added.
//! Where that column goes flat is the saturation point.

use std::time::Instant;

use phyz_gpu::cuda::{BatchSim, KernelBackend};
use phyz_gpu::CudaBatchSimulator;
use phyz_gpu::{BodyContactGains, PdDof};
use phyz_math::{GRAVITY, Vec3};
use phyz_model::{Model, State};

fn model_from(name: &str) -> Result<Model, String> {
    let path = format!("{}/../../models/{name}.xml", env!("CARGO_MANIFEST_DIR"));
    let loader = phyz_mjcf::MjcfLoader::from_file(&path).map_err(|e| format!("{path}: {e}"))?;
    let mut m = loader.build_model();
    m.gravity = Vec3::new(0.0, 0.0, -GRAVITY);
    Ok(m)
}

fn initial(model: &Model, i: usize, n: usize) -> State {
    let t = i as f64 / n.max(1) as f64;
    let mut s = model.default_state();
    if model.nq >= 6 {
        s.q[5] = 0.75;
        s.q[6 % model.nq] += 0.3 * t;
    }
    s
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

/// Minimum per-step wall over `reps` synchronised chunks — see
/// `cuda_graph_bench::time_min` for why the minimum and not the mean.
fn time_min<B: KernelBackend>(
    sim: &mut BatchSim<B>,
    states: &[State],
    chunk: usize,
    reps: usize,
) -> Result<f64, String> {
    sim.load_states(states);
    for _ in 0..2 {
        sim.step_many(chunk)?;
    }
    sim.backend().synchronize()?;
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        sim.step_many(chunk)?;
        sim.backend().synchronize()?;
        let dt = t0.elapsed().as_secs_f64() / chunk as f64;
        if dt < best {
            best = dt;
        }
    }
    Ok(best)
}

fn row(model: &Model, nworld: usize, sweeps: usize, reps: usize) -> Result<f64, String> {
    let mut sim = CudaBatchSimulator::new(model.clone(), nworld)?;
    sim.contact_sweeps = sweeps;
    let gains = BodyContactGains::uniform_frequency(model, 100.0, 1.0);
    sim.enable_contact_impulse(0.0, 0.8, &gains, &[], None)?;
    sim.enable_pd_control(&pd_dofs(model))?;
    sim.set_position_targets(&vec![vec![0.0; pd_dofs(model).len()]; nworld])?;
    sim.set_controls(&vec![vec![0.0; model.nv]; nworld]);
    let states: Vec<State> = (0..nworld).map(|i| initial(model, i, nworld)).collect();
    time_min(&mut sim, &states, 10, reps)
}

fn main() {
    let arg = |i: usize| std::env::args().nth(i);
    let name = arg(1).unwrap_or_else(|| "humanoid".into());
    let maxworlds: usize = arg(2).and_then(|s| s.parse().ok()).unwrap_or(16384);
    let reps: usize = std::env::var("ROOFLINE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let model = match model_from(&name) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    println!(
        "model {name}: {} bodies, nv {}, dt {}",
        model.nbodies(),
        model.nv,
        model.dt
    );

    for sweeps in [16usize, 8, 4, 1] {
        println!("\ncontact_sweeps = {sweeps}");
        println!("  {:>7}  {:>10}  {:>12}  {:>12}", "worlds", "ms/step", "us/world-st", "env-steps/s");
        let mut n = 32usize;
        while n <= maxworlds {
            match row(&model, n, sweeps, reps) {
                Ok(s) => println!(
                    "  {n:>7}  {:>10.4}  {:>12.3}  {:>12.3e}",
                    s * 1e3,
                    s * 1e6 / n as f64,
                    n as f64 / s
                ),
                Err(e) => println!("  {n:>7}  error: {e}"),
            }
            n *= 4;
        }
    }
}
