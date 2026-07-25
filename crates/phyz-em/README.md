# phyz-em

FDTD electromagnetics on a Yee grid.

Maxwell's equations solved by finite-difference time-domain on a staggered Yee
lattice.

| Type | Purpose |
| --- | --- |
| `YeeGrid` | staggered E/H grid with spacing and timestep |
| `FdtdSolver` | the update loop |
| `BoundaryCondition` | PML, periodic, PEC |
| `Source` | point dipole, plane wave, current loop |
| `Probe` | field sampling, energy and Poynting flux |

## Example

```rust
use phyz_em::{BoundaryCondition, FdtdSolver, Source, YeeGrid};
use phyz_math::Vec3;

let dx = 1e-9;                                  // 1 nm
let c = 3e8;
let dt = dx / (c * 3_f64.sqrt() * 1.1);         // CFL-stable
let grid = YeeGrid::new(32, 32, 32, dx, dt);

let mut solver = FdtdSolver::new(grid);
solver.add_source(Source::PointDipole {
    pos: Vec3::new(16.0 * dx, 16.0 * dx, 16.0 * dx),
    freq: 1e9,
    amplitude: 1.0,
    direction: Vec3::new(1.0, 0.0, 0.0),
});
```

Coupling EM fields to rigid bodies (Lorentz force, magnetic torque) lives in
[`phyz-coupling`](https://docs.rs/phyz-coupling).

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
