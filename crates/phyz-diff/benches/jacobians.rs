//! Criterion benchmarks comparing FD, analytical, and symbolic Jacobians.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use phyz_diff::symbolic::CompiledStepJacobians;
use phyz_diff::{analytical_step_jacobians, finite_diff_jacobians};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Model, ModelBuilder};

// ---------------------------------------------------------------------------
// Model builders
// ---------------------------------------------------------------------------

fn make_pendulum() -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001)
        .add_revolute_body(
            "link1",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                1.0,
                Vec3::new(0.0, -0.5, 0.0),
                Mat3::from_diagonal(&Vec3::new(1.0 / 12.0, 0.0, 1.0 / 12.0)),
            ),
        )
        .build()
}

fn make_double_pendulum() -> Model {
    let length = 1.0;
    let mass = 1.0;
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001)
        .add_revolute_body(
            "link1",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                mass,
                Vec3::new(0.0, -length / 2.0, 0.0),
                Mat3::from_diagonal(&Vec3::new(
                    mass * length * length / 12.0,
                    0.0,
                    mass * length * length / 12.0,
                )),
            ),
        )
        .add_revolute_body(
            "link2",
            0,
            SpatialTransform::from_translation(Vec3::new(0.0, -length, 0.0)),
            SpatialInertia::new(
                mass,
                Vec3::new(0.0, -length / 2.0, 0.0),
                Mat3::from_diagonal(&Vec3::new(
                    mass * length * length / 12.0,
                    0.0,
                    mass * length * length / 12.0,
                )),
            ),
        )
        .build()
}

/// Build a chain of N revolute links hanging vertically.
fn make_chain(n: usize) -> Model {
    let length = 1.0;
    let mass = 1.0;
    let inertia = SpatialInertia::new(
        mass,
        Vec3::new(0.0, -length / 2.0, 0.0),
        Mat3::from_diagonal(&Vec3::new(
            mass * length * length / 12.0,
            0.0,
            mass * length * length / 12.0,
        )),
    );

    let mut builder = ModelBuilder::new()
        .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
        .dt(0.001);

    for i in 0..n {
        let parent = if i == 0 { -1 } else { (i - 1) as i32 };
        let xf = if i == 0 {
            SpatialTransform::identity()
        } else {
            SpatialTransform::from_translation(Vec3::new(0.0, -length, 0.0))
        };
        builder = builder.add_revolute_body(&format!("link{}", i + 1), parent, xf, inertia);
    }

    builder.build()
}

/// Create a non-zero state for a model (sets q and v to small values).
fn random_state(model: &Model) -> phyz_model::State {
    let mut state = model.default_state();
    for i in 0..model.nq {
        state.q[i] = 0.3 + 0.1 * i as f64;
    }
    for i in 0..model.nv {
        state.v[i] = 0.1 - 0.05 * i as f64;
    }
    state
}

/// Pack state into the flat input vector [q, v, ctrl] for compiled Jacobians.
fn pack_inputs(state: &phyz_model::State, nq: usize, nv: usize) -> Vec<f64> {
    let mut inputs = Vec::with_capacity(nq + nv + nv);
    inputs.extend_from_slice(state.q.as_slice());
    inputs.extend_from_slice(state.v.as_slice());
    inputs.extend_from_slice(state.ctrl.as_slice());
    inputs
}

// ---------------------------------------------------------------------------
// Benchmark 1: Single pendulum (n=1)
// ---------------------------------------------------------------------------

fn bench_single_pendulum(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_pendulum");
    let model = make_pendulum();
    let state = random_state(&model);
    let inputs = pack_inputs(&state, model.nq, model.nv);

    group.bench_function("fd_jacobians", |b| {
        b.iter(|| finite_diff_jacobians(&model, &state, 1e-6));
    });

    group.bench_function("analytical_jacobians", |b| {
        b.iter(|| analytical_step_jacobians(&model, &state));
    });

    group.bench_function("symbolic_build", |b| {
        b.iter(|| CompiledStepJacobians::build(&model));
    });

    let compiled = CompiledStepJacobians::build(&model);
    group.bench_function("symbolic_eval", |b| {
        b.iter(|| compiled.eval(&inputs));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 2: Double pendulum (n=2)
// ---------------------------------------------------------------------------

fn bench_double_pendulum(c: &mut Criterion) {
    let mut group = c.benchmark_group("double_pendulum");
    let model = make_double_pendulum();
    let state = random_state(&model);
    let inputs = pack_inputs(&state, model.nq, model.nv);

    group.bench_function("fd_jacobians", |b| {
        b.iter(|| finite_diff_jacobians(&model, &state, 1e-6));
    });

    group.bench_function("analytical_jacobians", |b| {
        b.iter(|| analytical_step_jacobians(&model, &state));
    });

    group.bench_function("symbolic_build", |b| {
        b.iter(|| CompiledStepJacobians::build(&model));
    });

    let compiled = CompiledStepJacobians::build(&model);
    group.bench_function("symbolic_eval", |b| {
        b.iter(|| compiled.eval(&inputs));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 3: Chain scaling (n = 1, 2, 4, 8, 16)
// ---------------------------------------------------------------------------

fn bench_chain_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain_scaling");
    group.sample_size(20);

    // N=8+ causes combinatorial explosion in the symbolic expression graph
    // (each joint's expressions depend on all ancestors, so graph size grows
    // quadratically with chain length). Cap at N=4 to keep total bench time
    // under 60 seconds.
    for &n in &[1, 2, 4] {
        let model = make_chain(n);
        let state = random_state(&model);
        let inputs = pack_inputs(&state, model.nq, model.nv);

        group.bench_with_input(BenchmarkId::new("fd_jacobians", n), &n, |b, _| {
            b.iter(|| finite_diff_jacobians(&model, &state, 1e-6));
        });

        group.bench_with_input(BenchmarkId::new("analytical_jacobians", n), &n, |b, _| {
            b.iter(|| analytical_step_jacobians(&model, &state));
        });

        group.bench_with_input(BenchmarkId::new("symbolic_build", n), &n, |b, _| {
            b.iter(|| CompiledStepJacobians::build(&model));
        });

        let compiled = CompiledStepJacobians::build(&model);
        group.bench_with_input(BenchmarkId::new("symbolic_eval", n), &n, |b, _| {
            b.iter(|| compiled.eval(&inputs));
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 4: Compiled system reuse (build once, eval 100 times)
// ---------------------------------------------------------------------------

fn bench_compiled_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_reuse");
    group.sample_size(10);

    for &n in &[1, 2, 4] {
        let model = make_chain(n);

        // Generate 100 different states
        let states: Vec<Vec<f64>> = (0..100)
            .map(|seed| {
                let mut state = model.default_state();
                for i in 0..model.nq {
                    state.q[i] = 0.1 * ((seed * 7 + i * 13) % 31) as f64 / 31.0;
                }
                for i in 0..model.nv {
                    state.v[i] = 0.05 * ((seed * 11 + i * 17) % 31) as f64 / 31.0;
                }
                pack_inputs(&state, model.nq, model.nv)
            })
            .collect();

        // FD: compute all 100 from scratch
        let fd_states: Vec<phyz_model::State> = (0..100)
            .map(|seed| {
                let mut state = model.default_state();
                for i in 0..model.nq {
                    state.q[i] = 0.1 * ((seed * 7 + i * 13) % 31) as f64 / 31.0;
                }
                for i in 0..model.nv {
                    state.v[i] = 0.05 * ((seed * 11 + i * 17) % 31) as f64 / 31.0;
                }
                state
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("fd_100_evals", n), &n, |b, _| {
            b.iter(|| {
                for s in &fd_states {
                    std::hint::black_box(finite_diff_jacobians(&model, s, 1e-6));
                }
            });
        });

        group.bench_with_input(
            BenchmarkId::new("symbolic_build_plus_100_evals", n),
            &n,
            |b, _| {
                b.iter(|| {
                    let compiled = CompiledStepJacobians::build(&model);
                    for inputs in &states {
                        std::hint::black_box(compiled.eval(inputs));
                    }
                });
            },
        );

        // Amortized: pre-built, just the 100 evals
        let compiled = CompiledStepJacobians::build(&model);
        group.bench_with_input(
            BenchmarkId::new("symbolic_100_evals_prebuilt", n),
            &n,
            |b, _| {
                b.iter(|| {
                    for inputs in &states {
                        std::hint::black_box(compiled.eval(inputs));
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_pendulum,
    bench_double_pendulum,
    bench_chain_scaling,
    bench_compiled_reuse,
);
criterion_main!(benches);
