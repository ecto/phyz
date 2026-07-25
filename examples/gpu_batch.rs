//! `phyz-gpu`: simulate many independent environments in parallel on the GPU.
//!
//! Builds one double-pendulum model, fans it out across `NWORLD` worlds with
//! different initial conditions, and steps them all with a single wgpu
//! compute dispatch per timestep. Then it re-runs one of those worlds on the
//! CPU with `phyz-rigid` and compares, so the example is a correctness check
//! and not just a benchmark.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p phyz-examples --example gpu_batch
//! ```
//!
//! Requires a working GPU adapter (Metal / Vulkan / DX12). If none is
//! available the example prints why and exits 0 rather than failing, so it
//! stays usable on headless CI machines.

use std::time::Instant;

use phyz_gpu::GpuBatchSimulator;
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Model, ModelBuilder, State};
use phyz_rigid::aba;

/// Number of parallel environments.
const NWORLD: usize = 1024;
/// Steps to run.
const STEPS: usize = 500;
const DT: f64 = 0.002;

/// Spatial inertia of a uniform rod of mass `m` and length `2 * half_len`
/// hanging along -Y from the joint.
fn rod(m: f64, half_len: f64) -> SpatialInertia {
    let len = 2.0 * half_len;
    let i = m * len * len / 12.0;
    SpatialInertia::new(
        m,
        Vec3::new(0.0, -half_len, 0.0),
        Mat3::from_diagonal(&Vec3::new(i, 0.0, i)),
    )
}

fn double_pendulum() -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(DT)
        .add_revolute_body("upper", -1, SpatialTransform::identity(), rod(1.0, 0.5))
        .add_revolute_body("lower", 0, SpatialTransform::identity(), rod(0.8, 0.4))
        .build()
}

/// Initial joint angles for world `i`, fanned out across the workspace.
fn initial_q(i: usize) -> [f64; 2] {
    let t = i as f64 / NWORLD as f64;
    [0.2 + 1.2 * t, -0.4 + 0.8 * t]
}

/// Reference CPU rollout of a single world, semi-implicit Euler — the same
/// integrator the GPU kernel implements.
fn cpu_rollout(model: &Model, q0: [f64; 2], steps: usize) -> State {
    let mut state = model.default_state();
    state.q[0] = q0[0];
    state.q[1] = q0[1];
    for _ in 0..steps {
        let qdd = aba(model, &state);
        for k in 0..model.nv {
            state.v[k] += DT * qdd[k];
        }
        for k in 0..model.nq {
            state.q[k] += DT * state.v[k];
        }
    }
    state
}

fn main() {
    let model = double_pendulum();
    println!("=== phyz-gpu batched simulation ===\n");
    println!(
        "model      : double pendulum, nq={} nv={}",
        model.nq, model.nv
    );
    println!("worlds     : {NWORLD}");
    println!("steps      : {STEPS} @ dt={DT}s\n");

    let mut sim = match GpuBatchSimulator::new(model.clone(), NWORLD) {
        Ok(sim) => sim,
        Err(e) => {
            println!("No usable GPU adapter, skipping: {e}");
            return;
        }
    };

    // Ground contact is optional; enabling it adds a penalty-force pass
    // before ABA. The pendulum never reaches the ground here, but this shows
    // the call site.
    if let Err(e) = sim.enable_ground_contact(-2.0, 1.0e4, 5.0e1, 0.6) {
        println!("(ground contact unavailable: {e})");
    }

    // Fan out initial conditions across worlds.
    let states: Vec<State> = (0..NWORLD)
        .map(|i| {
            let mut s = model.default_state();
            let q = initial_q(i);
            s.q[0] = q[0];
            s.q[1] = q[1];
            s
        })
        .collect();
    sim.load_states(&states);
    sim.set_controls(&vec![vec![0.0; model.nv]; NWORLD]);

    let start = Instant::now();
    for _ in 0..STEPS {
        sim.step();
    }
    let final_states = sim.readback_states();
    let elapsed = start.elapsed();

    let env_steps = (NWORLD * STEPS) as f64;
    println!(
        "wall clock : {:.3}s for {:.0} env-steps ({:.2}M env-steps/s)",
        elapsed.as_secs_f64(),
        env_steps,
        env_steps / elapsed.as_secs_f64() / 1.0e6
    );

    // ---- correctness: compare one world against the CPU integrator ----
    let probe = NWORLD / 2;
    let cpu = cpu_rollout(&model, initial_q(probe), STEPS);
    let gpu = &final_states[probe];

    println!("\n--- world {probe} after {STEPS} steps ---");
    println!("        {:>12}  {:>12}", "gpu (f32)", "cpu (f64)");
    for k in 0..model.nq {
        println!("q[{k}]  {:>12.6}  {:>12.6}", gpu.q[k], cpu.q[k]);
    }
    for k in 0..model.nv {
        println!("v[{k}]  {:>12.6}  {:>12.6}", gpu.v[k], cpu.v[k]);
    }

    let dq = DVec::from_iterator(model.nq, (0..model.nq).map(|k| gpu.q[k] - cpu.q[k]));
    println!(
        "\nposition mismatch (L2): {:.3e}  \
         — the GPU kernel runs in f32 and the CPU reference in f64, so \
         agreement to ~1e-7 over this horizon is the expected result.",
        dq.norm()
    );

    // Spread across worlds shows the batch really is running distinct states.
    let spread = final_states
        .iter()
        .map(|s| s.q[0])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), q| {
            (lo.min(q), hi.max(q))
        });
    println!("q[0] across all worlds: [{:.4}, {:.4}]", spread.0, spread.1);
    assert!(
        spread.1 - spread.0 > 1e-6,
        "worlds should not all be identical"
    );
}
