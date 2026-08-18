//! Dimension 2: batched throughput on the GPU path (environment-steps/sec).
//!
//! The README claims "thousands of parallel environments". This suite sweeps
//! batch size and reports environment-steps per second, against the CPU
//! single-sim number for the same scene as the baseline — because a GPU batch
//! that never beats one CPU core is not a win no matter how large the batch.
//!
//! **Measurement honesty.** GPU work is submitted asynchronously. Timing
//! `step()` alone would measure command encoding, not physics. Each timed
//! repetition therefore issues `STEPS_PER_REP` steps and then blocks on a
//! readback of the full state, which forces the queue to drain. That folds one
//! readback into each repetition; the readback cost is reported separately so
//! it can be subtracted mentally.
//!
//! Compiled only with `--features gpu`, and reports `skipped` (rather than
//! failing) when no adapter is available. `--features gpu-cuda` adds the same
//! sweep on the CUDA path (`run_cuda`), for hosts where Vulkan is unavailable.

use crate::report::Suite;
use crate::timing::Budget;

/// Batch sizes swept.
pub const BATCH_SIZES: [usize; 6] = [1, 16, 128, 1024, 4096, 16384];

/// Steps issued per timed repetition, amortising the forced readback.
pub const STEPS_PER_REP: usize = 200;

const SUITE_NAME: &str = "batched GPU throughput";
const SUITE_DESC: &str = "Environment-steps per second on the phyz-gpu wgpu path, swept across \
     batch size. The `speedup_vs_cpu` metric compares against one CPU thread \
     running the same scene, which is the number that decides whether the GPU \
     path is worth using at all.";

const CUDA_SUITE_NAME: &str = "batched GPU throughput (CUDA)";
const CUDA_SUITE_DESC: &str = "Environment-steps per second on the phyz-gpu CUDA path, swept across \
     batch size — the same sweep as the wgpu suite, on the backend a rented \
     cloud GPU can actually open.";

#[cfg(not(feature = "gpu"))]
/// Run the GPU batch suite (disabled build: always reports skipped).
pub fn run(_budget: Budget) -> Suite {
    Suite::skipped(
        SUITE_NAME,
        SUITE_DESC,
        "built without `--features gpu`; rebuild with `cargo run --release -p phyz-bench \
         --features gpu` on a machine with a working wgpu adapter",
    )
}

#[cfg(feature = "gpu")]
pub use enabled::run;

#[cfg(not(feature = "gpu-cuda"))]
/// Run the CUDA batch suite (disabled build: always reports skipped).
pub fn run_cuda(_budget: Budget) -> Suite {
    Suite::skipped(
        CUDA_SUITE_NAME,
        CUDA_SUITE_DESC,
        "built without `--features gpu-cuda`; rebuild with `cargo run --release -p phyz-bench \
         --features gpu-cuda` on a machine with an NVIDIA driver",
    )
}

#[cfg(feature = "gpu-cuda")]
pub use enabled::run_cuda;

#[cfg(feature = "gpu")]
mod enabled {
    use super::*;
    use crate::report::{Metric, Record};
    use crate::scenes::{Scene, build_model, initial_state};
    use crate::settings::{DT_ARTICULATED, Settings};
    use crate::suites::single_sim;
    use crate::timing::measure;
    use phyz_gpu::GpuBatchSimulator;
    use phyz_model::{Model, State};

    /// The two batch simulators, seen through the four calls the sweep makes.
    trait Batch: Sized {
        const ENGINE: &'static str;
        fn create(model: &Model, batch: usize) -> Result<Self, String>;
        fn load_states(&mut self, states: &[State]);
        fn set_controls(&mut self, controls: &[Vec<f64>]);
        fn step(&mut self);
        fn readback_states(&mut self) -> Vec<State>;
    }

    impl Batch for GpuBatchSimulator {
        const ENGINE: &'static str = "phyz-gpu";
        fn create(model: &Model, batch: usize) -> Result<Self, String> {
            GpuBatchSimulator::new(model.clone(), batch)
        }
        fn load_states(&mut self, states: &[State]) {
            GpuBatchSimulator::load_states(self, states)
        }
        fn set_controls(&mut self, controls: &[Vec<f64>]) {
            GpuBatchSimulator::set_controls(self, controls)
        }
        fn step(&mut self) {
            GpuBatchSimulator::step(self)
        }
        fn readback_states(&mut self) -> Vec<State> {
            GpuBatchSimulator::readback_states(self)
        }
    }

    #[cfg(feature = "gpu-cuda")]
    impl Batch for phyz_gpu::CudaBatchSimulator {
        const ENGINE: &'static str = "phyz-gpu (CUDA)";
        fn create(model: &Model, batch: usize) -> Result<Self, String> {
            phyz_gpu::CudaBatchSimulator::new(model.clone(), batch)
        }
        fn load_states(&mut self, states: &[State]) {
            phyz_gpu::CudaBatchSimulator::load_states(self, states)
        }
        fn set_controls(&mut self, controls: &[Vec<f64>]) {
            phyz_gpu::CudaBatchSimulator::set_controls(self, controls)
        }
        fn step(&mut self) {
            phyz_gpu::CudaBatchSimulator::step(self)
        }
        fn readback_states(&mut self) -> Vec<State> {
            phyz_gpu::CudaBatchSimulator::readback_states(self)
        }
    }

    /// The scene batched on the GPU. The ant is the interesting case: enough
    /// DOF that the per-environment work is nontrivial.
    const SCENE: Scene = Scene::Ant;

    /// Run the GPU batch sweep on the wgpu path, or report why it could not run.
    pub fn run(budget: Budget) -> Suite {
        run_on::<GpuBatchSimulator>(budget, SUITE_NAME, SUITE_DESC)
    }

    /// Run the GPU batch sweep on the CUDA path, or report why it could not run.
    #[cfg(feature = "gpu-cuda")]
    pub fn run_cuda(budget: Budget) -> Suite {
        run_on::<phyz_gpu::CudaBatchSimulator>(budget, CUDA_SUITE_NAME, CUDA_SUITE_DESC)
    }

    fn run_on<B: Batch>(budget: Budget, name: &'static str, desc: &'static str) -> Suite {
        let settings = Settings::articulated(DT_ARTICULATED);
        let model = build_model(SCENE, settings.dt);

        // Probe with the smallest batch: if the adapter or the pipeline is
        // unavailable we want to say so once, clearly.
        if let Err(e) = B::create(&model, 1) {
            return Suite::skipped(
                name,
                desc,
                &format!("no usable GPU device or pipeline: {e}"),
            );
        }

        // CPU baseline for the same scene, measured here rather than assumed.
        let cpu = single_sim::run_scene(SCENE, budget);
        let cpu_steps_per_sec = cpu
            .timing
            .as_ref()
            .map(|t| t.throughput_per_sec)
            .unwrap_or(f64::NAN);
        // Every `speedup_vs_cpu` below divides by this one measurement.
        let cpu_noisy = cpu.timing.as_ref().is_some_and(|t| t.is_noisy());

        let mut results = vec![Record {
            engine: "phyz (CPU, 1 thread)".into(),
            scene: SCENE.name(),
            description: format!("{} — CPU baseline for the GPU sweep", SCENE.description()),
            dof: Some(model.nv),
            batch: Some(1),
            settings: settings.clone(),
            timing: cpu.timing.clone(),
            metrics: vec![Metric::new("speedup_vs_cpu", 1.0, "×")],
            notes: vec!["Baseline row: the same scene on one CPU thread.".into()],
        }];

        for &batch in &BATCH_SIZES {
            match run_batch::<B>(
                &model,
                batch,
                &settings,
                cpu_steps_per_sec,
                cpu_noisy,
                budget,
            ) {
                Ok(r) => results.push(r),
                Err(e) => {
                    // A batch that will not allocate is a real limit; record it
                    // and stop climbing rather than silently truncating.
                    results.push(failed_record::<B>(batch, &settings, model.nv, &e));
                    break;
                }
            }
        }

        Suite::new(name, desc, results)
    }

    fn failed_record<B: Batch>(batch: usize, settings: &Settings, nv: usize, err: &str) -> Record {
        Record {
            engine: B::ENGINE.into(),
            scene: SCENE.name(),
            description: format!("{} — batch {batch} FAILED", SCENE.description()),
            dof: Some(nv),
            batch: Some(batch),
            settings: settings.clone(),
            timing: None,
            metrics: Vec::new(),
            notes: vec![format!(
                "Batch {batch} could not be created ({err}); the sweep stops here. Larger \
                 batch sizes were not attempted."
            )],
        }
    }

    fn run_batch<B: Batch>(
        model: &phyz_model::Model,
        batch: usize,
        settings: &Settings,
        cpu_steps_per_sec: f64,
        cpu_baseline_noisy: bool,
        budget: Budget,
    ) -> Result<Record, String> {
        let mut sim = B::create(model, batch)?;
        let init = initial_state(SCENE, model);
        let states = vec![init; batch];
        sim.load_states(&states);
        sim.set_controls(&vec![vec![0.0; model.nv]; batch]);

        // Work unit = one environment-step.
        let work = (STEPS_PER_REP * batch) as u64;
        let timing = measure(budget, work, || {
            sim.load_states(&states);
            for _ in 0..STEPS_PER_REP {
                sim.step();
            }
            // Forces the queue to drain — without this we would be timing
            // command encoding.
            sim.readback_states()
        });

        // Isolate the readback so its share of the number is visible.
        let readback = measure(budget, 1, || sim.readback_states());
        let readback_share = readback.median_sec_per_unit
            / (timing.median_sec_per_unit * work as f64).max(f64::MIN_POSITIVE);

        let speedup = timing.throughput_per_sec / cpu_steps_per_sec;

        // `speedup_vs_cpu` is a ratio of two measurements, and the two do not
        // degrade together. GPU throughput here is largely insensitive to CPU
        // contention; the CPU baseline is not. On a busy host the denominator
        // collapses and the speedup inflates — a bias that flatters the GPU
        // path. Flag it, because it points the wrong way to catch by eye.
        let mut notes = vec![
            format!(
                "Each timed repetition issues {STEPS_PER_REP} steps then blocks on one \
                 full state readback, which is included in the reported time."
            ),
            "phyz-gpu computes in f32; the CPU path is f64. The GPU numbers are \
             therefore not comparable at equal precision, and this is a real \
             difference in what is being computed, not just how fast."
                .into(),
            "GPU contact is not enabled in this sweep — the ant scene is contact-free, \
             so this measures the ABA + integration pipeline only."
                .into(),
        ];
        if cpu_baseline_noisy {
            notes.push(
                "SPEEDUP INFLATED: the CPU baseline row was itself noisy, so \
                 `speedup_vs_cpu` divides by a depressed denominator and overstates the \
                 GPU's advantage. GPU throughput is comparatively insensitive to host \
                 load, so this bias runs in phyz's favour and will not look wrong. \
                 Re-run on an idle host."
                    .into(),
            );
        }

        Ok(Record {
            engine: B::ENGINE.into(),
            scene: SCENE.name(),
            description: format!("{} — batch {batch}", SCENE.description()),
            dof: Some(model.nv),
            batch: Some(batch),
            settings: settings.clone(),
            timing: Some(timing),
            metrics: vec![
                Metric::new("speedup_vs_cpu", speedup, "×"),
                Metric::new("readback_ms", readback.median_sec_per_unit * 1.0e3, "ms"),
                Metric::new("readback_share", readback_share, "fraction"),
            ],
            notes,
        })
    }
}
