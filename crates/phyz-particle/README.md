# phyz-particle

Material Point Method particle simulation for phyz.

MPM for continua that do not fit rigid bodies: granular media, sand, snow,
weakly compressible fluids.

| Type | Purpose |
| --- | --- |
| `MpmSolver` | the particle-to-grid / grid-to-particle step |
| `Particle` | position, velocity, deformation gradient, mass |
| `Material` | constitutive model and its parameters |

`Material` covers elastic, plastic and fluid responses (including an equation
of state for weakly compressible flow), which is what lets the same solver run
sand and water.

## Example

```rust
use phyz_math::Vec3;
use phyz_particle::MpmSolver;

// Grid spacing, timestep, and the simulation bounds.
let mut solver = MpmSolver::new(
    0.05,
    1.0e-4,
    (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
);

let mut particles = Vec::new();
// ...seed `particles` with positions, velocities and a `Material`...
solver.step(&mut particles);
```

See `examples/granular_column.rs`, `examples/sphere_in_sand.rs` and
`examples/water_container.rs` in the repository for runnable setups.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
