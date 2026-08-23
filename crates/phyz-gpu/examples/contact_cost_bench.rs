//! Where an impulse-mode step's cost goes, on wgpu.
//!
//! ```text
//! cargo run --release -p phyz-gpu --example contact_cost_bench -- [worlds] [steps]
//! ```
//!
//! Times penalty mode against impulse mode at several sweep counts. The step
//! is `2 + 2*sweeps` dispatches, so a linear fit in `sweeps` separates the
//! fixed part (PD + integrate + leading ABA) from the marginal cost of one
//! `[contact, ABA]` pair — which is what the pre-tip collector pays 16 times.
use std::time::Instant;

use phyz_gpu::{BodyContactGains, GpuBatchSimulator, PdDof};
use phyz_math::{GRAVITY, Vec3};
use phyz_model::{Model, State};

fn model_from(path: &str) -> Result<Model, String> {
    let loader = phyz_mjcf::MjcfLoader::from_file(path).map_err(|e| format!("{path}: {e}"))?;
    let mut m = loader.build_model();
    m.gravity = Vec3::new(0.0, 0.0, -GRAVITY);
    Ok(m)
}

fn pd_dofs(model: &Model) -> Vec<PdDof> {
    (0..model.nv)
        .skip(6.min(model.nv))
        .map(|i| PdDof { q_index: i, v_index: i, kp: 40.0, kd: 2.0, max_force: 60.0 })
        .collect()
}

fn initial(model: &Model, i: usize, n: usize) -> State {
    let t = i as f64 / n.max(1) as f64;
    let mut s = model.default_state();
    if model.nq >= 6 {
        s.q[5] = 0.6;
        s.q[6 % model.nq] += 0.3 * t;
    }
    s
}

fn time_min(sim: &mut GpuBatchSimulator, states: &[State], chunk: usize, reps: usize) -> f64 {
    sim.load_states(states);
    for _ in 0..chunk { sim.step(); }
    let _ = sim.readback_states();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        for _ in 0..chunk { sim.step(); }
        let _ = sim.readback_states();
        let dt = t0.elapsed().as_secs_f64() / chunk as f64;
        if dt < best { best = dt; }
    }
    best
}

fn main() {
    let arg = |i: usize| std::env::args().nth(i);
    let nworld: usize = arg(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let reps: usize = arg(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let name = arg(3).unwrap_or_else(|| "humanoid".into());
    let path = format!("{}/../../models/{name}.xml", env!("CARGO_MANIFEST_DIR"));
    let model = model_from(&path).unwrap();
    println!("model: {name} ({} bodies, nv {})  worlds {nworld}", model.nbodies(), model.nv);
    let gains = BodyContactGains::uniform_frequency(&model, 100.0, 1.0);
    let states: Vec<State> = (0..nworld).map(|i| initial(&model, i, nworld)).collect();

    // No contact at all: PD + ABA + integrate.
    {
        let mut sim = GpuBatchSimulator::new(model.clone(), nworld).unwrap();
        sim.enable_pd_control(&pd_dofs(&model)).unwrap();
        sim.set_position_targets(&vec![vec![0.0; pd_dofs(&model).len()]; nworld]).unwrap();
        sim.set_controls(&vec![vec![0.0; model.nv]; nworld]);
        let ms = 1e3 * time_min(&mut sim, &states, 20, reps);
        println!("  no-contact         {ms:>9.4} ms/step  (PD+ABA+integrate)");
    }
    // Penalty reference.
    {
        let mut sim = GpuBatchSimulator::new(model.clone(), nworld).unwrap();
        sim.enable_ground_contact_per_body(0.0, 0.8, &gains).unwrap();
        sim.enable_pd_control(&pd_dofs(&model)).unwrap();
        sim.set_position_targets(&vec![vec![0.0; pd_dofs(&model).len()]; nworld]).unwrap();
        sim.set_controls(&vec![vec![0.0; model.nv]; nworld]);
        let ms = 1e3 * time_min(&mut sim, &states, 20, reps);
        println!("  penalty            {ms:>9.4} ms/step  (2 dispatches)");
    }
    for sweeps in [1usize, 2, 4, 8, 16] {
        let mut sim = GpuBatchSimulator::new(model.clone(), nworld).unwrap();
        sim.enable_contact_impulse(0.0, 0.8, &gains, &[], None).unwrap();
        sim.contact_sweeps = sweeps;
        sim.enable_pd_control(&pd_dofs(&model)).unwrap();
        sim.set_position_targets(&vec![vec![0.0; pd_dofs(&model).len()]; nworld]).unwrap();
        sim.set_controls(&vec![vec![0.0; model.nv]; nworld]);
        let ms = 1e3 * time_min(&mut sim, &states, 20, reps);
        println!("  impulse sweeps={sweeps:<3}    {ms:>9.4} ms/step  ({} dispatches)", 2 + 2 * sweeps);
    }
}
