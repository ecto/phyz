# phyz

Open-source differentiable multi-physics simulation. Pure Rust.

**[Try the demos →](https://phyz.dev)**

Simulate a robot, compute gradients through the entire trajectory, and optimize a controller — in one loop. `phyz` gives you a reverse-mode adjoint over a rollout with exact dual-number Jacobians inside each step, so you can fit physical parameters, reshape a contact surface, or solve inverse problems without finite-differencing the whole simulation. It's the simulation backend for when you need physics to be a differentiable function.

## Installation

```bash
cargo add phyz
```

The `phyz` crate is an umbrella over the **rigid-body stack**: spatial math,
articulated models, Featherstone dynamics, collision, contact, and the
differentiable rollout. Those modules are re-exports of the focused `phyz-*`
crates, so `phyz::collision` and `phyz_collision` are the same thing. It also
adds `phyz::sim` — `Simulator` and its solvers — which is the one piece of code
it owns.

Collision, contact and diff are default features and can be turned off:

```bash
cargo add phyz --no-default-features            # math + model + rigid only
cargo add phyz --no-default-features -F diff    # ...plus gradients
```

It does **not** cover the whole workspace. Everything else lives in its own
crate and must be added explicitly:

```bash
cargo add phyz-gpu        # batched simulation on wgpu
cargo add phyz-particle   # MPM / SPH / granular
cargo add phyz-lbm        # lattice Boltzmann fluids
# ...and so on, see the crate table below
```

## Quick start

```rust
use phyz::{DVec, Mat3, ModelBuilder, SpatialInertia, SpatialTransform, Vec3};
use phyz::diff::{AdjointRollout, FinalStateObjective, adjoint_rollout_gradient};

// Double pendulum: two revolute joints, rods hanging along -Y.
let rod = |m: f64, half_len: f64| {
    let i = m * (2.0 * half_len) * (2.0 * half_len) / 12.0;
    SpatialInertia::new(
        m,
        Vec3::new(0.0, -half_len, 0.0),
        Mat3::from_diagonal(&Vec3::new(i, 0.0, i)),
    )
};

let model = ModelBuilder::new()
    .gravity(Vec3::new(0.0, -9.81, 0.0))
    .dt(0.002)
    .add_revolute_body("upper", -1, SpatialTransform::identity(), rod(1.0, 0.5))
    .add_revolute_body("lower", 0, SpatialTransform::identity(), rod(0.8, 0.4))
    .build();

// Open-loop rollout: no applied torque.
let zero = DVec::zeros(model.nv);
let ctrl = |_step: usize| zero.clone();

let rollout = AdjointRollout {
    model: &model,
    contact: None,
    q0: vec![0.5, -0.3],
    v0: vec![0.0, 0.0],
    steps: 500,
    ctrl: &ctrl,
};

// Objective: squared joint angles at the final step.
let objective = FinalStateObjective {
    value: &|q: &[f64], _v: &[f64]| q.iter().map(|x| x * x).sum(),
    gradient: &|q: &[f64], v: &[f64]| {
        (q.iter().map(|x| 2.0 * x).collect(), vec![0.0; v.len()])
    },
};

// One forward rollout, one backward pass.
let grad = adjoint_rollout_gradient(&rollout, &objective);

// grad.objective    — J at the nominal trajectory
// grad.d_inertia[i] — exact dJ/dπ for body i's 10 spatial-inertia scalars
// grad.d_vertices   — exact ∂J/∂x per collision-mesh vertex (when contact is on)
assert_eq!(grad.d_inertia.len(), model.nbodies());
```

This example is compiled and run as a doc-test in CI (see
`crates/phyz/src/lib.rs`), so it cannot silently rot.

## Differentiability: what is actually exact

The word "analytical" is easy to over-claim, so here is the honest breakdown.

| API | What it differentiates | Method | Exact? |
| --- | --- | --- | --- |
| `phyz_diff::rollout::adjoint_rollout_gradient` (`phyz::diff::…`) | whole rollout w.r.t. inertia params and contact-mesh vertices | reverse-mode adjoint, dual numbers within each step | **yes** |
| `phyz_diff::symbolic::symbolic_step_jacobians` | one step w.r.t. `(q, v, ctrl)` | symbolic differentiation | **yes** |
| `phyz_diff::semi_implicit_step_jacobians` | one step w.r.t. `(q, v, ctrl)` | chain rule through the integrator, **finite differences on ABA** | no |
| `phyz_diff::finite_diff_jacobians` | one step w.r.t. `(q, v, ctrl)` | central differences over the whole step | no |

`phyz_diff::analytical_step_jacobians` was a misnomer — it is finite-difference
based. It is now deprecated in favour of `semi_implicit_step_jacobians`.

## Crates

`phyz` is the only crate a rigid-body user needs. The rest are independent and
opt-in.

| Crate | What it does |
| --- | --- |
| [`phyz`](crates/phyz) | Umbrella over the rigid-body stack: re-exports math, model, rigid, collision, contact, diff, plus the `sim` time loop |
| [`phyz-math`](crates/phyz-math) | Spatial algebra: vectors, matrices, quaternions, spatial transforms and inertias |
| [`phyz-model`](crates/phyz-model) | Articulated body models, joints, actuators, state |
| [`phyz-rigid`](crates/phyz-rigid) | Featherstone ABA, RNEA, CRBA, forward kinematics, energy |
| [`phyz-diff`](crates/phyz-diff) | Per-step Jacobians (finite-difference, chain-rule, symbolic) and the exact trajectory adjoint |
| [`phyz-collision`](crates/phyz-collision) | GJK/EPA narrow phase, ray casting, sweep-and-prune broad phase |
| [`phyz-contact`](crates/phyz-contact) | Contact resolution, friction, implicit penalty forces |
| [`phyz-mjcf`](crates/phyz-mjcf) | MuJoCo MJCF model loading |
| [`phyz-urdf`](crates/phyz-urdf) | URDF (ROS) robot description import |
| [`phyz-gpu`](crates/phyz-gpu) | wgpu compute: batched simulation of many independent worlds |
| [`phyz-compile`](crates/phyz-compile) | Physics IR → WGSL compute shaders, with kernel fusion |
| [`phyz-particle`](crates/phyz-particle) | MPM solver, SPH fluids, granular media |
| [`phyz-md`](crates/phyz-md) | Molecular dynamics, Lennard-Jones, field engine |
| [`phyz-em`](crates/phyz-em) | Maxwell's equations on a Yee lattice |
| [`phyz-lbm`](crates/phyz-lbm) | Lattice Boltzmann fluids: BGK/TRT/MRT, declarative boundaries, LES |
| [`phyz-gravity`](crates/phyz-gravity) | N-body gravitational dynamics |
| [`phyz-qft`](crates/phyz-qft) | Lattice QFT, Wilson action |
| [`phyz-regge`](crates/phyz-regge) | Regge calculus (discrete GR + EM) |
| [`phyz-prob`](crates/phyz-prob) | Probabilistic inference over physics (SVGD, HMC) |
| [`phyz-coupling`](crates/phyz-coupling) | Multi-physics coupling: `Solver` trait, coupled systems, subcycling, flux accounting |
| [`phyz-guardian`](crates/phyz-guardian) | Conservation monitoring, adaptive time-stepping, solver auto-switching |
| [`phyz-world`](crates/phyz-world) | Scene graph, world assembly, collision-backed sensors |
| [`phyz-real2sim`](crates/phyz-real2sim) | Inverse problems, system identification, parameter fitting |
| [`phyz-format`](crates/phyz-format) | `.phyz` scene serialization, MJCF and URDF import |
| [`phyz-dream`](crates/phyz-dream) | Learned latent dynamics on top of the simulator |

Not published: `phyz-wasm` (browser demo bindings), `phyz-home` (the site
backend), and `phyz-quantum` (Hamiltonian lattice gauge theory — it depends on
the unpublished `tang-mesh`). `phyz-py` (Python bindings) lives outside the
workspace.

Every workspace crate shares one version (`workspace.package.version`), so
`phyz` and the crates it re-exports never disagree. `phyz-py`, which lives
outside the workspace, versions independently.

## Examples

```bash
cargo run --release -p phyz-examples --example pendulum
cargo run --release -p phyz-examples --example gpu_batch        # phyz-gpu
cargo run --release -p phyz-examples --example kernel_fusion    # phyz-compile
cargo run --release -p phyz-coupling --example coupled_cyclotron # phyz-coupling
```

All files under `examples/` are registered as targets of the `phyz-examples`
dev crate and built in CI, so they compile against the current API. Some crates
also carry their own `examples/` directory (`phyz-coupling`, `phyz-guardian`),
picked up by cargo directly.
```text
phyz/
├── crates/
│   ├── phyz/             # Umbrella — re-exports everything
│   ├── phyz-math/        # Spatial algebra, vectors, matrices
│   ├── phyz-model/       # Articulated body models, joints, inertia
│   ├── phyz-rigid/       # Featherstone ABA, forward/inverse dynamics
│   ├── phyz-diff/        # Analytical Jacobians, differentiable stepping
│   ├── phyz-mjcf/        # MuJoCo MJCF model loading
│   ├── phyz-urdf/        # URDF (ROS) robot description import
│   ├── phyz-collision/   # GJK/EPA collision, ray casting
│   ├── phyz-contact/     # Contact resolution, friction
│   ├── phyz-gpu/         # WGPU compute, batched simulation
│   ├── phyz-compile/     # Physics kernel compiler, op fusion
│   ├── phyz-particle/    # SPH fluids, granular media
│   ├── phyz-em/          # Maxwell's equations on Yee lattice
│   ├── phyz-md/          # Molecular dynamics: Lennard-Jones, PME electrostatics, cell lists
│   ├── phyz-qft/         # Lattice QFT, Wilson action
│   ├── phyz-gravity/     # N-body gravitational dynamics
│   ├── phyz-lbm/         # Lattice Boltzmann fluid method
│   ├── phyz-regge/       # Regge calculus (discrete GR + EM)
│   ├── phyz-prob/        # Probabilistic inference (SVGD, HMC)
│   ├── phyz-coupling/    # Multi-physics coupling layer
│   ├── phyz-guardian/    # Energy/momentum conservation guards
│   ├── phyz-world/       # Scene graph, multi-physics world
│   ├── phyz-real2sim/    # Inverse problems, parameter fitting
│   ├── phyz-format/      # Serialization, model I/O (.phyz, MJCF, URDF)
│   ├── phyz-validate/    # Closed-form physics benchmarks for every solver
│   └── phyz-wasm/        # WASM bindings for browser demos
├── examples/             # Runnable examples
└── site/                 # Landing page + interactive demos
```

## Validation

Every solver is benchmarked against a closed-form solution or published reference
data, with a quantitative error and — where a discretization parameter exists — a
demonstrated convergence order:

```bash
cargo run --release -p phyz-validate
```

That writes `target/validation/validation.md` and `validation.json`. See
[VALIDATION.md](VALIDATION.md) for the current results, including the benchmarks
that fail.

## Development

```bash
cargo test --workspace
cargo build --workspace --examples
cargo clippy --workspace -- -D warnings
cargo doc --workspace --no-deps
```

`tang`, `tang-la` and `tang-expr` come from crates.io. The one exception is
`tang-mesh`, which is unpublished and backs `phyz-quantum`'s default `mesh`
feature, so a full workspace build still wants a sibling checkout:

```bash
git clone https://github.com/ecto/tang ../tang
```

To develop against a local `tang`, uncomment the `[patch.crates-io]` block at
the bottom of the workspace `Cargo.toml`.

Build the WASM demos:

```bash
wasm-pack build crates/phyz-wasm --target web --out-dir ../../site/pkg
```

## Status

- No crate in this workspace is `#![no_std]`. Everything depends on `std`,
  including `phyz-wasm`.
- GPU support means `phyz-gpu` and `phyz-compile`; the rigid-body core in
  `phyz` runs on the CPU.

## Benchmarks

[**BENCHMARKS.md**](BENCHMARKS.md) — throughput, gradient cost, energy drift,
and a like-for-like comparison against Rapier, including the cases where phyz
loses. Reproduce with `make bench`.

Two numbers worth knowing before you adopt phyz:

- The GPU path does not break even against a single CPU thread until roughly
  **batch 128**. Below that it is slower, substantially so at batch 1.
- Gradient rollouts cost 18–54× a forward rollout, and that ratio grows with
  parameter count. The derivatives are *exact*, which is the real benefit —
  they are not asymptotically cheaper than finite differences today.

## License

[MIT](LICENSE)
