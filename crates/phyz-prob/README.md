# phyz-prob

Probabilistic simulation and uncertainty quantification.

Propagate uncertainty through physics instead of simulating a single
trajectory: ensembles of states with uncertain parameters, and gradient-based
inference over them.

| Type | Purpose |
| --- | --- |
| `ProbabilisticState` | a weighted particle ensemble over `(q, v)` |
| `Distribution` | priors over physical parameters |
| `ensemble::EnsembleSolver` | the per-particle stepping trait |
| `svgd_step` | Stein variational gradient descent |

`ProbabilisticState::perturbed_samples` builds an ensemble from a nominal
state; `sample_parameters` draws uncertain physical parameters per particle;
`mean_and_std` reports the propagated distribution and effective sample size.

## Example

See `examples/ensemble_pendulum.rs`:

```bash
cargo run --release -p phyz-examples --example ensemble_pendulum
```

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
