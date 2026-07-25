# phyz-guardian

Conservation monitoring and adaptive stepping for phyz.

Watches a simulation for the failure modes that fixed-step integrators hide:
energy drift, momentum leakage, and silently wrong results at too large a `dt`.

| Type | Purpose |
| --- | --- |
| `ConservationMonitor`, `ConservationState` | track energy/momentum against a baseline |
| `AdaptiveTimeStep`, `PiController`, `EmbeddedRkError` | error-controlled `dt` |
| `AutoSwitchController`, `SolverQuality`, `DegradationStrategy` | escalate solver quality when accuracy drops |
| `RRespaIntegrator`, `split_forces_stiffness`, `split_forces_gravity` | multi-rate integration |

## Example

```bash
cargo run --release -p phyz-guardian --example adaptive_pendulum
cargo run --release -p phyz-guardian --example conservation_monitor
cargo run --release -p phyz-guardian --example multi_rate
```

`AdaptiveTimeStep::step` returns `(accepted, new_dt)`, so a rejected step is
visible to the caller rather than being silently retried.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
