//! Tick time at a fixed world count, for a model of a given width.
//!
//! Used to price the MAX_BODIES removal: the same 32-body rig must tick at
//! the same speed before and after, and a 34-body rig must be merely wider,
//! not catastrophically slower.
//!
//!   cargo run -p phyz-gpu --release --example tick_bench -- <nbodies> [nworld]
//!   cargo run -p phyz-gpu --release --features cuda --example tick_bench -- 34
//!
//! Runs on CUDA when the `cuda` feature is on and a device is present, and on
//! the wgpu backend otherwise.

use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Joint, Model, ModelBuilder};
use std::time::Instant;

fn inertia(mass: f64, i: f64) -> SpatialInertia {
    SpatialInertia::new(
        mass,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(i, i, i)),
    )
}

fn chain(n: usize) -> Model {
    let mut b = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(0.001)
        .add_body(
            "root",
            -1,
            Joint::fixed(SpatialTransform::identity()),
            inertia(1.0, 0.01),
        );
    for i in 1..n {
        b = b.add_revolute_body(
            &format!("link{i}"),
            i as i32 - 1,
            SpatialTransform::from_translation(Vec3::new(0.1, 0.0, 0.0)),
            inertia(0.5, 0.005),
        );
    }
    b.build()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let nb: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(32);
    let nworld: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(4096);
    let warmup = 50;
    let steps = 500;
    let model = chain(nb);
    assert_eq!(model.nbodies(), nb);

    #[cfg(feature = "cuda")]
    {
        match phyz_gpu::cuda::CudaBatchSimulator::new(model.clone(), nworld) {
            Ok(mut sim) => {
                for _ in 0..warmup {
                    sim.step();
                }
                // readback_states() forces a sync. Without it this times
                // the ENQUEUE of the launches, not the work: CUDA step() is
                // asynchronous, and the loop alone reports ~2us/step, which
                // is the driver accepting launches, not the GPU doing ABA.
                let _ = sim.readback_states();
                let t = Instant::now();
                for _ in 0..steps {
                    sim.step();
                }
                let sunk = sim.readback_states();
                std::hint::black_box(&sunk);
                let us = t.elapsed().as_secs_f64() * 1e6 / steps as f64;
                println!("CUDA nbodies={nb} nworld={nworld} tick={us:.1}us");
                return;
            }
            Err(e) => eprintln!("no CUDA ({e}), falling back to wgpu"),
        }
    }

    let mut sim = phyz_gpu::GpuBatchSimulator::new(model.clone(), nworld)
        .expect("no GPU adapter and no CUDA device");
    for _ in 0..warmup {
        sim.step();
    }
    let _ = sim.readback_states();
    let t = Instant::now();
    for _ in 0..steps {
        sim.step();
    }
    let sunk = sim.readback_states();
    std::hint::black_box(&sunk);
    let us = t.elapsed().as_secs_f64() * 1e6 / steps as f64;
    println!("wgpu nbodies={nb} nworld={nworld} tick={us:.1}us");
}
