# phyz-dream

Learned surrogate dynamics for phyz models.

Trains a neural surrogate on rollouts of a phyz model, so downstream search or
control can query a cheap approximate dynamics function instead of the full
simulator.

| Type | Purpose |
| --- | --- |
| `generate_dataset`, `SampleConfig`, `input_dim` | sample rollouts from a `Model` |
| `DreamModel`, `DreamMeta`, `NormStats` | the surrogate and its normalization |
| `train`, `TrainConfig` | training loop |
| `validate`, `ValidationResult` | held-out error against the true dynamics |
| `GpuDreamModel` | GPU inference |

Validation is not optional here: a surrogate that has drifted from the
simulator is worse than no surrogate, and `validate` is what tells you which
you have.

## Status

Experimental. The API is expected to move.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
