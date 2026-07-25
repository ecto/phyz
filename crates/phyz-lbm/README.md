# phyz-lbm

Lattice Boltzmann fluid dynamics for phyz.

Mesoscopic fluid simulation: the macroscopic flow emerges from a discrete
velocity distribution rather than being solved for directly. At macroscopic
scales this recovers the incompressible Navier-Stokes equations.

| Type | Purpose |
| --- | --- |
| `LatticeBoltzmann2D` | D2Q9 lattice |
| `LatticeBoltzmann3D` | D3Q19 lattice |
| `CollisionModel` | BGK, TRT, and (in 2D) full MRT |
| `Turbulence` | Smagorinsky LES from the non-equilibrium stress |
| `Boundaries`, `Boundary`, `Side` | boundaries declared once, applied by `step()` |
| `analytic` | closed-form and published reference solutions |
| `EquationFreeWrapper`, `CoarseProjector`, `FineSolver` | projective integration |

TRT with `Λ = 3/16` ([`MAGIC_BOUNCE_BACK`]) is the default, because it removes
the viscosity-dependent bounce-back wall error that makes plain BGK unusable
for wall-bounded flow.

Boundaries are declarative — periodic, no-slip, moving wall, Zou-He velocity
inlet, pressure outlet, symmetry — declared once and applied automatically,
rather than re-imposed by hand each step.

## Example: lid-driven cavity

```rust
use phyz_lbm::{LatticeBoltzmann2D, boundary};

let mut lbm = LatticeBoltzmann2D::new(64, 64, 0.1)
    .with_boundaries(boundary::cavity_2d([0.1, 0.0]));
lbm.initialize_uniform(1.0, [0.0, 0.0]);
lbm.run(1000);

let u = lbm.velocity(32, 32);
println!("centre velocity: [{:.4}, {:.4}]", u[0], u[1]);
```

## Validation

`analytic` carries force-driven Poiseuille flow, Taylor-Green decay, and a
`convergence_order` helper; the benchmark suite lives in `tests/validation.rs`.
Results are checked against closed-form solutions, not eyeballed.

## Equation-free integration

The `equation_free` layer runs short bursts of the fine LBM solver, projects
the coarse observable forward over a much longer interval, and lifts back — so
macroscopic timescales are reachable without paying for every fine step.

```bash
cargo run --release -p phyz-examples --example lbm_cavity
cargo run --release -p phyz-examples --example equation_free_demo
```

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
