# phyz-lbm

Lattice Boltzmann fluid dynamics for phyz.

Mesoscopic fluid simulation where the macroscopic flow emerges from a discrete
velocity distribution.

| Type | Purpose |
| --- | --- |
| `LatticeBoltzmann2D` | D2Q9 lattice |
| `LatticeBoltzmann3D` | D3Q19 lattice |
| `EquationFreeWrapper` | coarse-timestep projective integration |
| `CoarseProjector`, `FineSolver` | the lift/restrict pair it needs |
| `effective_information` | a coarse-graining diagnostic |

The equation-free layer is the interesting part: it runs short bursts of the
fine LBM solver, projects the coarse observable forward over a much longer
interval, and lifts back — so you can reach macroscopic timescales without
paying for every fine step.

## Example

```rust,no_run
use phyz_lbm::LatticeBoltzmann2D;

// nx, ny, kinematic viscosity
let mut lbm = LatticeBoltzmann2D::new(128, 128, 0.02);
lbm.initialize_uniform(1.0, [0.0, 0.0]);

// Lid-driven cavity: drive the top wall, no-slip elsewhere.
for x in 0..128 {
    lbm.set_velocity_bc(x, 127, [0.1, 0.0]);
    lbm.set_no_slip_bc(x, 0);
}

for _ in 0..1000 {
    lbm.collide_and_stream();
}

let ke = lbm.kinetic_energy();
```

See `examples/lbm_cavity.rs` and `examples/equation_free_demo.rs`.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
