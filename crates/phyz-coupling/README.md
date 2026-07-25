# phyz-coupling

Multi-physics coupling between phyz solvers.

The layer that lets different solvers, running at different timescales, act on
each other.

| Piece | Purpose |
| --- | --- |
| `BoundingBox` | the handshake region where two solvers overlap |
| `Coupling`, `SolverType`, `ForceTransfer` | declarative description of a coupled pair |
| `SubcyclingSchedule`, `TimeScale` | r-RESPA multi-rate stepping |
| `RespaPartition` | short-range/long-range force splitting |
| `lorentz_force`, `magnetic_torque` | EM → rigid-body transfer |
| `electric_dipole_force`, `radiation_pressure_force` | further EM transfers |

Subcycling is what makes this practical: without it, coupling a 1 µs EM solver
to a 1 ms rigid-body solver forces everything to the EM timestep.

## Example

```rust
use phyz_coupling::{BoundingBox, Coupling, ForceTransfer, SolverType, lorentz_force};
use phyz_math::Vec3;

let coupling = Coupling {
    solver_a: SolverType::Electromagnetic,
    solver_b: SolverType::RigidBody,
    overlap_region: BoundingBox::new(
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ),
    force_transfer: ForceTransfer::Direct { damping: 0.1 },
};

let force = lorentz_force(
    1e-6,                       // charge (C)
    Vec3::zeros(),              // position
    Vec3::new(1.0, 0.0, 0.0),   // velocity (m/s)
    &Vec3::new(0.0, 0.0, 1e3),  // E (V/m)
    &Vec3::new(0.0, 1.0, 0.0),  // B (T)
);
```

For a full coupled loop:

```bash
cargo run --release -p phyz-examples --example coupled_em_rigid
```

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
