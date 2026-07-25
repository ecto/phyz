# phyz-world

Scene assembly and procedural world generation for phyz.

The layer above a single `Model`: whole scenes, actuation routing, sensing, and
recording.

| Type | Purpose |
| --- | --- |
| `World` | a scene of models plus its physics settings |
| `WorldGenerator` | procedural terrain and scene generation |
| `Tendon` | routed tendon actuation across joints |
| `Sensor`, `SensorOutput` | sensor definitions and readings |
| `TrajectoryRecorder` | capture rollouts for logging or training |

## Example

```bash
cargo run --release -p phyz-examples --example world_generation
cargo run --release -p phyz-examples --example tendon_demo
```

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
