//! What the fused impulse step saves in arithmetic, on the host backend.
//!
//! ```text
//! cargo run --release -p phyz-gpu --features cuda-host --example fused_step_bench -- [worlds] [steps]
//! ```
//!
//! The host backend walks the SAME kernel text serially, one world at a time
//! — which is what a CUDA thread does too, since every pass here is one
//! thread per world with no cross-thread coupling. So the ratio it reports
//! is the per-thread work removed, and it is the part of the CUDA number
//! that is not launch overhead. Launch count per step (2 + 2*sweeps -> 1) has
//! to be measured on the device.
fn main() {
    #[cfg(not(feature = "cuda-host"))]
    eprintln!("build with --features cuda-host");
    #[cfg(feature = "cuda-host")]
    run();
}

#[cfg(feature = "cuda-host")]
fn run() {
    use phyz_gpu::cuda::HostBatchSimulator;
    use phyz_gpu::{BodyContactGains, PdDof};
    use phyz_math::{GRAVITY, Vec3};
    use phyz_model::{Model, State};
    use std::time::Instant;

    let arg = |i: usize| std::env::args().nth(i);
    let nworld: usize = arg(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    let steps: usize = arg(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let name = arg(3).unwrap_or_else(|| "humanoid".into());
    let path = format!("{}/../../models/{name}.xml", env!("CARGO_MANIFEST_DIR"));
    let mut model: Model = phyz_mjcf::MjcfLoader::from_file(&path)
        .expect("load")
        .build_model();
    model.gravity = Vec3::new(0.0, 0.0, -GRAVITY);
    let pd: Vec<PdDof> = (6..model.nv)
        .map(|i| PdDof { q_index: i, v_index: i, kp: 40.0, kd: 2.0, max_force: 60.0 })
        .collect();
    let gains = BodyContactGains::uniform_frequency(&model, 100.0, 1.0);
    let states: Vec<State> = (0..nworld)
        .map(|i| {
            let mut s = model.default_state();
            s.q[5] = 0.55;
            s.q[6 % model.nq] += 0.3 * (i as f64 / nworld as f64);
            s
        })
        .collect();
    println!(
        "model: {name} ({} bodies, nv {}), {nworld} worlds x {steps} steps",
        model.nbodies(),
        model.nv
    );
    for sweeps in [1usize, 4, 16] {
        let mut row = Vec::new();
        for fused in [false, true] {
            let mut sim = HostBatchSimulator::new(model.clone(), nworld).expect("sim");
            sim.set_fused_step_enabled(fused);
            sim.enable_contact_impulse(0.0, 0.8, &gains, &[], None).unwrap();
            sim.contact_sweeps = sweeps;
            sim.enable_pd_control(&pd).unwrap();
            sim.set_position_targets(&vec![vec![0.0; pd.len()]; nworld]).unwrap();
            sim.load_states(&states);
            sim.step_many(4).unwrap();
            sim.load_states(&states);
            let t0 = Instant::now();
            for _ in 0..(steps / 20) {
                sim.step_many(20).unwrap();
            }
            row.push(t0.elapsed().as_secs_f64() / (nworld * steps) as f64 * 1e6);
        }
        println!(
            "  sweeps={sweeps:<3} unfused {:>8.2} us/world-step   fused {:>8.2}   {:.2}x",
            row[0],
            row[1],
            row[0] / row[1]
        );
    }
}
