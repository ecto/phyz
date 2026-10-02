# phyz-qft

Lattice gauge theory with Hybrid Monte Carlo.

Euclidean lattice QFT with the Wilson gauge action and HMC sampling.

| Type | Purpose |
| --- | --- |
| `Lattice` | the gauge-link configuration |
| `U1`, `SU2`, `SU3` | gauge groups, via the `Group` trait |
| `HmcState`, `HmcParams` | Hybrid Monte Carlo sampler |
| `WilsonLoop`, `PolyakovLoop`, `Observables` | measurements |
| `su3::Su3Lattice` | concrete SU(3): heatbath + overrelaxation, stout smearing, clover field strength |
| `su3::FluxAccumulator` | static baryon flux tube: three-quark Wilson loops correlated with field densities |
| `su3::FieldFile` | `.qcdf` export for viewers such as kosm's `kosm-qcd` |

HMC is what makes the sampling tractable: molecular-dynamics trajectories in
the gauge field's fictitious momentum, with a Metropolis accept/reject that
corrects for integrator error, so the chain decorrelates far faster than local
updates.

## SU(3) examples

```sh
# QCD vacuum: action and topological charge densities ("lava lamp")
cargo run -p phyz-qft --release --example qcd_vacuum -- vacuum.qcdf
# static-baryon flux tube (vacuum suppression between three quarks)
cargo run -p phyz-qft --release --example baryon_flux -- baryon.qcdf
```

Validation: `cargo test -p phyz-qft --release -- --include-ignored` checks
⟨P⟩ = 0.5937 at β = 6.0 and near-integer topological charge after smoothing.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
