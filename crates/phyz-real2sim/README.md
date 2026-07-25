# phyz-real2sim

Inverse problems and system identification for phyz.

Fit simulation parameters so a model reproduces measured behaviour — the
real-to-sim half of the loop.

| Type | Purpose |
| --- | --- |
| `TrajectoryMatcher` | objective comparing simulated and observed trajectories |
| `Optimizer`, `GradientDescentOptimizer`, `OptimizerConfig` | the fitting loop |
| `ObservationAdapter` | map raw measurements onto model quantities |
| `JointAngleObserver`, `EndEffectorPoseObserver` | concrete adapters |

The observation adapters matter because measurements rarely line up with state
variables: you get joint encoder counts or a tracked end-effector pose, not
`q` and `v`.

## Example

```bash
cargo run --release -p phyz-examples --example real2sim_pendulum
```

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
