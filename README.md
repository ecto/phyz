# phyz

Open-source differentiable multi-physics simulation. Pure Rust.

**[Try the demos →](https://phyz.dev)**

Simulate a robot, compute gradients through the entire trajectory, and optimize a controller — in one loop. `phyz` gives you a reverse-mode adjoint over a rollout with exact dual-number Jacobians inside each step, so you can fit physical parameters, reshape a contact surface, or solve inverse problems without finite-differencing the whole simulation. It's the simulation backend for when you need physics to be a differentiable function.

## Installation

```bash
cargo add phyz
```

The `phyz` crate is the **rigid-body core**: spatial math, articulated models,
Featherstone dynamics, collision, contact, and the differentiable rollout. It is
self-contained and does not pull in the rest of the workspace.

Everything else lives in its own crate and must be added explicitly:

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
| `phyz::diff::adjoint_rollout_gradient` | whole rollout w.r.t. inertia params and contact-mesh vertices | reverse-mode adjoint, dual numbers within each step | **yes** |
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
| [`phyz`](crates/phyz) | Rigid-body core: spatial math, models, ABA/RNEA/CRBA, collision, contact, differentiable rollout |
| [`phyz-math`](crates/phyz-math) | Spatial algebra: vectors, matrices, quaternions, spatial transforms and inertias |
| [`phyz-model`](crates/phyz-model) | Articulated body models, joints, actuators, state |
| [`phyz-rigid`](crates/phyz-rigid) | Featherstone ABA, RNEA, CRBA, forward kinematics, energy |
| [`phyz-diff`](crates/phyz-diff) | Per-step Jacobians: finite-difference, chain-rule, and symbolic |
| [`phyz-collision`](crates/phyz-collision) | GJK/EPA narrow phase, sweep-and-prune broad phase |
| [`phyz-contact`](crates/phyz-contact) | Contact resolution, friction, implicit penalty forces |
| [`phyz-mjcf`](crates/phyz-mjcf) | MuJoCo MJCF model loading |
| [`phyz-gpu`](crates/phyz-gpu) | wgpu compute: batched simulation of many independent worlds |
| [`phyz-compile`](crates/phyz-compile) | Physics IR → WGSL compute shaders, with kernel fusion |
| [`phyz-particle`](crates/phyz-particle) | MPM solver, SPH fluids, granular media |
| [`phyz-md`](crates/phyz-md) | Molecular dynamics, Lennard-Jones, field engine |
| [`phyz-em`](crates/phyz-em) | Maxwell's equations on a Yee lattice |
| [`phyz-lbm`](crates/phyz-lbm) | Lattice Boltzmann fluids |
| [`phyz-gravity`](crates/phyz-gravity) | N-body gravitational dynamics |
| [`phyz-qft`](crates/phyz-qft) | Lattice QFT, Wilson action |
| [`phyz-regge`](crates/phyz-regge) | Regge calculus (discrete GR + EM) |
| [`phyz-quantum`](crates/phyz-quantum) | Quantum state evolution |
| [`phyz-prob`](crates/phyz-prob) | Probabilistic inference over physics (SVGD, HMC) |
| [`phyz-coupling`](crates/phyz-coupling) | Multi-physics coupling: handshake regions, subcycling, Lorentz transfer |
| [`phyz-guardian`](crates/phyz-guardian) | Conservation monitoring, adaptive time-stepping, solver auto-switching |
| [`phyz-world`](crates/phyz-world) | Scene graph and multi-physics world assembly |
| [`phyz-real2sim`](crates/phyz-real2sim) | Inverse problems, system identification, parameter fitting |
| [`phyz-format`](crates/phyz-format) | `.phyz` scene serialization and model I/O |
| [`phyz-dream`](crates/phyz-dream) | Learned latent dynamics on top of the simulator |

Not published: `phyz-wasm` (browser demo bindings) and `phyz-home` (the site
backend). `phyz-py` (Python bindings) lives outside the workspace.

> **Versioning note:** `phyz` is at 0.3.x; every other crate is at 0.1.0. That
> split is being addressed separately.

## Examples

```bash
cargo run --release -p phyz-examples --example pendulum
cargo run --release -p phyz-examples --example gpu_batch          # phyz-gpu
cargo run --release -p phyz-examples --example coupled_em_rigid   # phyz-coupling
cargo run --release -p phyz-examples --example kernel_fusion      # phyz-compile
```

All files under `examples/` are registered as targets of the `phyz-examples`
dev crate and built in CI, so they compile against the current API.

## Development

```bash
cargo test --workspace
cargo build --workspace --examples
cargo clippy --workspace -- -D warnings
cargo doc --workspace --no-deps
```

The workspace resolves `tang` through a path dependency on a sibling checkout,
so `../tang` must exist next to this repo:

```bash
git clone https://github.com/ecto/tang ../tang
```

Build the WASM demos:

```bash
wasm-pack build crates/phyz-wasm --target web --out-dir ../../site/pkg
```

## Status

- No crate in this workspace is `#![no_std]`. Everything depends on `std`,
  including `phyz-wasm`.
- GPU support means `phyz-gpu` and `phyz-compile`; the rigid-body core in
  `phyz` runs on the CPU.

## License

[MIT](LICENSE)
