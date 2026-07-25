# phyz-gravity

Gravity for phyz, from constant-g to post-Newtonian.

A layered gravity solver: pick the cheapest model that is accurate enough.

| Type | Regime |
| --- | --- |
| `ConstantGravity` | uniform field — terrestrial robotics |
| `NBodySolver`, `BarnesHutTree` | direct and O(n log n) N-body |
| `PoissonSolver` | grid-based potential from a density field |
| `PostNewtonianSolver` | 1PN corrections — precession, strong fields |
| `orbital_elements`, `perihelion_precession_rate` | orbital diagnostics |

## Example

`examples/mercury_precession.rs` in the repository reproduces Mercury's
43″/century perihelion advance from the post-Newtonian solver:

```bash
cargo run --release -p phyz-examples --example mercury_precession
```

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
