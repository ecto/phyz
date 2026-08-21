//! What a captured step span costs against the same launches issued by hand.
//!
//! ```text
//! cargo run --release -p phyz-gpu --features cuda --example cuda_graph_bench -- [worlds] [steps] [control_every]
//! ```
//!
//! A step in impulse mode is `2 + 2 * contact_sweeps` launches of a few
//! microseconds each, so the step time is flat in the world count over the
//! range RL cares about — it is the per-launch cost, not the arithmetic. The
//! bench separates the two halves of that cost by timing the issue loop
//! before synchronising as well as after: `issue` is what the host spent
//! pushing launches, `total` is what the device took to retire them. When
//! `issue` is close to `total`, the host is the bottleneck and a graph is
//! the fix.
//!
//! Three configurations are timed over the same span, from the same initial
//! states, and their final states are compared for bit-identity:
//!
//! * `off`      — `PHYZ_CUDA_GRAPHS=0`, every launch issued individually;
//! * `step`     — a one-step graph replayed per step (what an unmodified
//!                caller of `step()` now gets);
//! * `span(n)`  — one graph over a whole control period, replayed once per
//!                period (what a caller gets from `step_many`).

use std::time::Instant;

use phyz_gpu::cuda::{BatchSim, KernelBackend};
use phyz_gpu::{BodyContactGains, PdDof};
use phyz_math::{GRAVITY, Vec3};
use phyz_model::{Model, State};

fn ant() -> Result<Model, String> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/ant.xml");
    let loader = phyz_mjcf::MjcfLoader::from_file(path).map_err(|e| format!("{path}: {e}"))?;
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

/// Every revolute DOF servoed, which is the PD pass RL runs.
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

struct Timing {
    label: String,
    issue: f64,
    total: f64,
    states: Vec<State>,
}

fn time_run<B: KernelBackend>(
    sim: &mut BatchSim<B>,
    label: &str,
    states: &[State],
    steps: usize,
    chunk: usize,
) -> Result<Timing, String> {
    sim.load_states(states);
    // Warm-up: pays the capture, the module load and the first-touch faults
    // so none of them land in the number.
    for _ in 0..(chunk * 2) {
        sim.step_many(chunk)?;
    }
    sim.backend().synchronize()?;
    sim.load_states(states);

    let t0 = Instant::now();
    for _ in 0..(steps / chunk) {
        sim.step_many(chunk)?;
    }
    let issue = t0.elapsed().as_secs_f64();
    sim.backend().synchronize()?;
    let total = t0.elapsed().as_secs_f64();
    Ok(Timing {
        label: label.into(),
        issue,
        total,
        states: sim.readback_states(),
    })
}

fn report(t: &Timing, nworld: usize, steps: usize) {
    let ms = 1e3 * t.total / steps as f64;
    let issue_ms = 1e3 * t.issue / steps as f64;
    let rate = (nworld * steps) as f64 / t.total;
    println!(
        "  {:<10} {ms:>8.4} ms/step   (issue {issue_ms:>7.4})   {rate:>10.3e} env-steps/s",
        t.label
    );
}

fn run<B: KernelBackend>(
    mut sim: BatchSim<B>,
    model: &Model,
    nworld: usize,
    steps: usize,
    control_every: usize,
) -> Result<(), String> {
    println!("kernels on: {}", sim.backend().device_name());
    println!(
        "graphs: {}   sweeps: {}   launches/step: {}",
        sim.graphs_enabled(),
        sim.contact_sweeps,
        2 + 2 * sim.contact_sweeps
    );
    let gains = BodyContactGains::uniform_frequency(model, 100.0, 1.0);
    sim.enable_contact_impulse(0.0, 0.8, &gains, None, None)?;
    sim.enable_pd_control(&pd_dofs(model))?;
    sim.set_position_targets(&vec![vec![0.0; pd_dofs(model).len()]; nworld])?;
    sim.set_controls(&vec![vec![0.0; model.nv]; nworld]);

    let states: Vec<State> = (0..nworld).map(|i| initial(model, i, nworld)).collect();
    let steps = steps - steps % control_every;

    let graphs = sim.graphs_enabled();
    let mut runs = Vec::new();
    sim.set_graphs_enabled(false);
    runs.push(time_run(&mut sim, "off", &states, steps, 1)?);
    if graphs {
        sim.set_graphs_enabled(true);
        runs.push(time_run(&mut sim, "step", &states, steps, 1)?);
        runs.push(time_run(
            &mut sim,
            &format!("span({control_every})"),
            &states,
            steps,
            control_every,
        )?);
    }

    println!("\n{nworld} worlds x {steps} steps:");
    for t in &runs {
        report(t, nworld, steps);
    }
    if runs.len() > 1 {
        println!();
        for t in &runs[1..] {
            println!(
                "  speedup {:<10} {:.2}x wall   {:.2}x issue",
                t.label,
                runs[0].total / t.total,
                runs[0].issue / t.issue
            );
        }
    }

    // Bit-identity: a replay runs the same kernels on the same addresses
    // with the same arguments, so it is not "close to" the uncaptured
    // sequence — it is the same bits.
    for t in &runs[1..] {
        for (w, (a, b)) in runs[0].states.iter().zip(&t.states).enumerate() {
            for j in 0..model.nq {
                if a.q[j].to_bits() != b.q[j].to_bits() {
                    return Err(format!(
                        "{} differs from uncaptured at world {w} q[{j}]: {:?} vs {:?}",
                        t.label, a.q[j], b.q[j]
                    ));
                }
            }
            for j in 0..model.nv {
                if a.v[j].to_bits() != b.v[j].to_bits() {
                    return Err(format!(
                        "{} differs from uncaptured at world {w} v[{j}]: {:?} vs {:?}",
                        t.label, a.v[j], b.v[j]
                    ));
                }
            }
        }
        println!("  bit-identical to uncaptured: {}", t.label);
    }
    Ok(())
}

fn main() {
    let arg = |i: usize| std::env::args().nth(i);
    let nworld: usize = arg(1).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let steps: usize = arg(2).and_then(|s| s.parse().ok()).unwrap_or(400);
    let control_every: usize = arg(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let model = match ant() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    println!("model: ant ({} bodies, nv {})", model.nbodies(), model.nv);

    #[cfg(feature = "cuda")]
    let result = phyz_gpu::CudaBatchSimulator::new(model.clone(), nworld)
        .and_then(|sim| run(sim, &model, nworld, steps, control_every));
    #[cfg(not(feature = "cuda"))]
    let result: Result<(), String> = Err("built without --features cuda".into());

    if let Err(e) = result {
        eprintln!("cuda_graph_bench: {e}");
        std::process::exit(1);
    }
}
