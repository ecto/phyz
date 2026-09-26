//! CUDA smoke test: step N parallel worlds on the CUDA backend, check them
//! against the CPU, and report environment-steps per second.
//!
//! ```text
//! cargo run --release -p phyz-gpu --features cuda --example cuda_batch -- [worlds] [steps] [model]
//! ```
//!
//! `model` is `ant` (default; `models/ant.xml`, floating base + 8 revolute)
//! or `pendulum` (offset double pendulum). With `--features cuda-host` and
//! `PHYZ_KERNELS=host` the same run goes through the CUDA C compiled as host
//! C++ — the no-GPU dry run of exactly this program.
//!
//! Exits non-zero if the CUDA path is unavailable or disagrees with the CPU;
//! prints why.

use std::time::Instant;

use phyz_gpu::cuda::{BatchSim, KernelBackend};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Model, ModelBuilder, State};
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

fn pendulum() -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001)
        .add_revolute_body("upper", -1, SpatialTransform::identity(), rod(1.0, 0.5))
        .add_revolute_body(
            "lower",
            0,
            SpatialTransform::from_translation(Vec3::new(0.0, -1.0, 0.0)),
            rod(0.8, 0.4),
        )
        .build()
}

fn ant() -> Result<Model, String> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/ant.xml");
    let loader = phyz_mjcf::MjcfLoader::from_file(path).map_err(|e| format!("{path}: {e}"))?;
    Ok(loader.build_model())
}

fn initial(model: &Model, i: usize, n: usize) -> State {
    let t = i as f64 / n.max(1) as f64;
    let mut s = model.default_state();
    if model.nq >= 6 {
        s.q[5] = 0.75; // floating base height
        s.q[6 % model.nq] += 0.3 * t;
    } else {
        s.q[0] = 0.2 + 1.2 * t;
        s.q[1] = -0.4 + 0.8 * t;
    }
    s
}

fn run<B: KernelBackend>(
    mut sim: BatchSim<B>,
    model: &Model,
    nworld: usize,
    steps: usize,
) -> Result<(), String> {
    println!("kernels on: {}", sim.backend().device_name());
    let states: Vec<State> = (0..nworld).map(|i| initial(model, i, nworld)).collect();
    sim.load_states(&states);
    sim.set_controls(&vec![vec![0.0; model.nv]; nworld]);

    // Warm-up step so JIT/first-launch cost is not in the number.
    sim.try_step()?;
    sim.readback_states();
    sim.load_states(&states);

    let t0 = Instant::now();
    for _ in 0..steps {
        sim.try_step()?;
    }
    let out = sim.readback_states();
    let dev = t0.elapsed().as_secs_f64();
    let dev_rate = (nworld * steps) as f64 / dev;
    println!("device: {nworld} worlds x {steps} steps in {dev:.3} s = {dev_rate:.3e} env-steps/s",);

    // CPU: a handful of worlds, timed, and used as the reference.
    let ncpu = nworld.min(8);
    let mut cpu: Vec<State> = states[..ncpu].to_vec();
    let t0 = Instant::now();
    for s in cpu.iter_mut() {
        for _ in 0..steps {
            let qdd = aba(model, s);
            semi_implicit_euler(model, s, qdd.as_slice(), model.dt);
        }
    }
    let cpu_t = t0.elapsed().as_secs_f64();
    let cpu_rate = (ncpu * steps) as f64 / cpu_t;
    println!(
        "cpu:    {ncpu} worlds x {steps} steps in {cpu_t:.3} s = {cpu_rate:.3e} env-steps/s (1 thread, f64)",
    );
    println!("speedup: {:.1}x", dev_rate / cpu_rate);

    let mut worst = 0.0f64;
    for (w, (d, c)) in out.iter().zip(&cpu).enumerate() {
        for j in 0..model.nq {
            let e = (d.q[j] - c.q[j]).abs();
            if e > worst {
                worst = e;
            }
            if !d.q[j].is_finite() {
                return Err(format!("world {w} q[{j}] is not finite"));
            }
        }
    }
    println!("max |q_device - q_cpu| over {ncpu} worlds after {steps} steps: {worst:.2e}");
    // f32 vs f64 over `steps` steps of a falling ant / swinging pendulum: the
    // documented precision gap is ~1e-3; an indexing or sign error is 1e-1.
    if worst > 1e-2 {
        return Err(format!("device disagrees with CPU by {worst:.2e}"));
    }
    println!("parity: ok");
    Ok(())
}

fn main() {
    let arg = |i: usize| std::env::args().nth(i);
    let nworld: usize = arg(1).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let steps: usize = arg(2).and_then(|s| s.parse().ok()).unwrap_or(500);
    let which = arg(3).unwrap_or_else(|| "ant".into());
    let model = match which.as_str() {
        "pendulum" => pendulum(),
        _ => match ant() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(2);
            }
        },
    };
    println!(
        "model: {which} ({} bodies, nv {})",
        model.nbodies(),
        model.nv
    );

    let host = std::env::var("PHYZ_KERNELS").is_ok_and(|v| v == "host");
    let result = if host {
        #[cfg(feature = "cuda-host")]
        {
            phyz_gpu::HostBatchSimulator::new(model.clone(), nworld)
                .and_then(|sim| run(sim, &model, nworld, steps))
        }
        #[cfg(not(feature = "cuda-host"))]
        {
            Err("PHYZ_KERNELS=host needs --features cuda-host".to_string())
        }
    } else {
        #[cfg(feature = "cuda")]
        {
            phyz_gpu::CudaBatchSimulator::new(model.clone(), nworld)
                .and_then(|sim| run(sim, &model, nworld, steps))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err("built without --features cuda".to_string())
        }
    };
    if let Err(e) = result {
        eprintln!("cuda_batch: {e}");
        std::process::exit(1);
    }
}
