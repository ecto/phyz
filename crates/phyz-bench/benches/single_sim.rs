//! Criterion view of the single-sim scenes.
//!
//! The published numbers come from the `phyz-bench` binary, which measures
//! every engine the same way and emits JSON. This file exists for the other
//! job criterion is good at: catching a regression between two commits on the
//! same machine, with confidence intervals and change detection.
//!
//! Run with `cargo bench -p phyz-bench`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use phyz_bench::scenes::PhyzSim;
use phyz_bench::suites::single_sim::settings_for;

/// Steps per criterion iteration. Small, because criterion runs many of them.
const STEPS: usize = 100;

fn scenes(c: &mut Criterion) {
    let mut group = c.benchmark_group("step");
    group.throughput(Throughput::Elements(STEPS as u64));

    for scene in phyz_bench::suites::standard_scenes() {
        let dt = settings_for(scene).dt;
        let mut sim = PhyzSim::new(scene, dt);
        group.bench_with_input(
            BenchmarkId::from_parameter(scene.name()),
            &STEPS,
            |b, &steps| {
                b.iter(|| {
                    // Reset inside the timed closure so every iteration
                    // simulates the same trajectory; the reset is a memcpy of
                    // a handful of vectors next to `steps` dynamics solves.
                    sim.reset();
                    sim.steps(steps);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, scenes);
criterion_main!(benches);
