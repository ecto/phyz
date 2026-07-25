# phyz-coupling

Multi-physics coupling between phyz solvers.

The layer that lets different solvers, running at different timescales, act on
each other — and that accounts for what crossed between them.

## The abstraction

| Piece | Purpose |
| --- | --- |
| `Solver` | a domain that advances by `dt`, exposes state for coupling queries, accepts external input, and reports its natural timestep |
| `CoupledSystem` | pairs two `Solver`s, evaluates the handshake over an overlap region, applies it antisymmetrically |
| `CouplingSite`, `FieldSample`, `ExternalInput` | what the two sides exchange |
| `FluxLedger` | books the momentum and energy that crossed, so conservation is checkable |
| `ReactionMode` | whether and how the back-reaction is applied |
| `BoundingBox` | the spatial overlap region |
| `SubcyclingSchedule`, `TimeScale` | r-RESPA multi-rate stepping |
| `lorentz_force`, `magnetic_torque` | the EM to rigid-body transfer law |

Two solvers implement the trait today: `RigidSolver` over `phyz-rigid`'s ABA,
and `EmSolver` over `phyz-em`'s Yee-grid FDTD. `phyz-particle`, `phyz-md`,
`phyz-lbm` and `phyz-gravity` have no adapters yet.

Subcycling is what makes this practical: without it, coupling a picosecond EM
solver to a millisecond rigid-body solver forces everything to the EM timestep.

## Coupled simulation

```rust,no_run
use phyz_coupling::{BoundingBox, CoupledSystem, EmSolver, RigidSolver};
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::ModelBuilder;

let model = ModelBuilder::new()
    .gravity(Vec3::zeros())
    .dt(5e-11)
    .add_free_body(
        "bob",
        -1,
        SpatialTransform::identity(),
        SpatialInertia::new(
            1e-9,
            Vec3::zeros(),
            Mat3::from_diagonal(&Vec3::new(1e-12, 1e-12, 1e-12)),
        ),
    )
    .build();

let mut state = model.default_state();
state.q[0] = 0.4;
state.v[3] = 1e5; // free joint: v = [angular(3), linear(3)]

let matter = RigidSolver::new(model, state);
// ...build an `EmSolver` over a Yee grid, then pair them:
// let mut system = CoupledSystem::new(matter, field, region);
// system.run(20_000, 5e-11);
// assert!(system.relative_energy_drift() < 1e-3);
```

`tests/cyclotron.rs` runs the pair against the closed-form cyclotron solution,
so the coupling is validated rather than merely plausible.

## Force transfer only

If you are driving the loop yourself, the transfer laws stand alone:

```rust
use phyz_coupling::lorentz_force;
use phyz_math::Vec3;

let force = lorentz_force(
    1e-6,                       // charge (C)
    Vec3::zeros(),              // position
    Vec3::new(1.0, 0.0, 0.0),   // velocity (m/s)
    &Vec3::new(0.0, 0.0, 1e3),  // E (V/m)
    &Vec3::new(0.0, 1.0, 0.0),  // B (T)
);
```

## Examples

```bash
cargo run --release -p phyz-coupling --example coupled_cyclotron
cargo run --release -p phyz-coupling --example lorentz_pendulum
cargo run --release -p phyz-coupling --example multi_solver
cargo run --release -p phyz-coupling --example subcycling
```

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
