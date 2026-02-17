# phyz

Open-source differentiable multi-physics simulation. Pure Rust. GPU-accelerated.

**[Try the demos →](https://phyz.dev)**

Simulate a robot, compute gradients through the entire trajectory, and optimize a controller — in one loop. phyz gives you analytical derivatives for free, so you can train policies, fit physical parameters from video, or solve inverse problems without finite differencing. It's the simulation backend for when you need physics to be a differentiable function.

## Features

- **Rigid Bodies** — Featherstone ABA, O(n) forward dynamics, articulated joints
- **Differentiable** — Analytical Jacobians at every timestep for optimization, control, and learning
- **Particles** — SPH fluids, granular media, molecular dynamics
- **Electromagnetism** — Maxwell's equations on Yee lattice
- **General Relativity** — Regge calculus with Einstein-Maxwell action
- **Probabilistic** — SVGD, HMC for Bayesian inference over physics
- **MJCF Loading** — Import MuJoCo XML models directly
- **GPU Batching** — WGPU compute for thousands of parallel environments
- **Kernel Compiler** — Fuse physics ops into optimized GPU kernels

## Quick start

```rust
use phyz::{ModelBuilder, Simulator, Vec3};
use phyz_diff::analytical_step_jacobians;

// Double pendulum with analytical gradients
let model = ModelBuilder::new()
    .add_revolute_body("upper", -1, 1.0, Vec3::new(0., 0., -1.))
    .add_revolute_body("lower",  0, 0.8, Vec3::new(0., 0., -1.))
    .build();

let mut state = model.default_state();
state.q  = vec![0.5, -0.3];
state.qd = vec![0.0,  0.0];

let sim = Simulator::rk4();
for _ in 0..500 {
    let jac = analytical_step_jacobians(&model, &state, 0.002);
    sim.step(&model, &mut state, 0.002);
    // jac.dq_dq0: sensitivity of next state to current state
    // jac.dq_dtau: sensitivity of next state to applied torques
}
```

## Architecture

```
phyz/
├── crates/
│   ├── phyz/             # Umbrella — re-exports everything
│   ├── phyz-math/        # Spatial algebra, vectors, matrices
│   ├── phyz-model/       # Articulated body models, joints, inertia
│   ├── phyz-rigid/       # Featherstone ABA, forward/inverse dynamics
│   ├── phyz-diff/        # Analytical Jacobians, differentiable stepping
│   ├── phyz-mjcf/        # MuJoCo MJCF model loading
│   ├── phyz-collision/   # GJK/EPA collision detection
│   ├── phyz-contact/     # Contact resolution, friction
│   ├── phyz-gpu/         # WGPU compute, batched simulation
│   ├── phyz-compile/     # Physics kernel compiler, op fusion
│   ├── phyz-particle/    # SPH fluids, granular media
│   ├── phyz-em/          # Maxwell's equations on Yee lattice
│   ├── phyz-md/          # Molecular dynamics, Lennard-Jones
│   ├── phyz-qft/         # Lattice QFT, Wilson action
│   ├── phyz-gravity/     # N-body gravitational dynamics
│   ├── phyz-lbm/         # Lattice Boltzmann fluid method
│   ├── phyz-regge/       # Regge calculus (discrete GR + EM)
│   ├── phyz-prob/        # Probabilistic inference (SVGD, HMC)
│   ├── phyz-coupling/    # Multi-physics coupling layer
│   ├── phyz-guardian/    # Energy/momentum conservation guards
│   ├── phyz-world/       # Scene graph, multi-physics world
│   ├── phyz-real2sim/    # Inverse problems, parameter fitting
│   ├── phyz-format/      # Serialization, model I/O
│   └── phyz-wasm/        # WASM bindings for browser demos
├── examples/             # Runnable examples
└── site/                 # Landing page + interactive demos
```

## Development

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings

# Build WASM demos
wasm-pack build crates/phyz-wasm --target web --out-dir ../../site/pkg
```

## License

[MIT](LICENSE)
