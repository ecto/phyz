# phyz-qft

Lattice gauge theory with Hybrid Monte Carlo.

Euclidean lattice QFT with the Wilson gauge action and HMC sampling.

| Type | Purpose |
| --- | --- |
| `Lattice` | the gauge-link configuration |
| `U1`, `SU2`, `SU3` | gauge groups, via the `Group` trait |
| `HmcState`, `HmcParams` | Hybrid Monte Carlo sampler |
| `WilsonLoop`, `PolyakovLoop`, `Observables` | measurements |

HMC is what makes the sampling tractable: molecular-dynamics trajectories in
the gauge field's fictitious momentum, with a Metropolis accept/reject that
corrects for integrator error, so the chain decorrelates far faster than local
updates.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
