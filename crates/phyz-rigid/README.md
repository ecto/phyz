# phyz-rigid

Featherstone articulated rigid-body dynamics for phyz.

The classical O(n) algorithms, operating on
[`phyz-model`](https://docs.rs/phyz-model)'s `Model` and `State`.

| Function | Computes | Cost |
| --- | --- | --- |
| `aba` | forward dynamics `qdd = f(q, v, τ)` | O(n) |
| `aba_with_external_forces` | forward dynamics with per-body wrenches | O(n) |
| `rnea` | inverse dynamics `τ = f(q, v, qdd)` | O(n) |
| `crba` | joint-space mass matrix `M(q)` | O(n²) |
| `forward_kinematics` | body transforms into `state.body_xform` | O(n) |
| `kinetic_energy`, `potential_energy`, `total_energy` | conserved quantities | O(n) |

## Example

```rust
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::ModelBuilder;
use phyz_rigid::{aba, total_energy};

let model = ModelBuilder::new()
    .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
    .dt(0.002)
    .add_revolute_body(
        "pendulum",
        -1,
        SpatialTransform::identity(),
        SpatialInertia::new(
            1.0,
            Vec3::new(0.0, -0.5, 0.0),
            Mat3::from_diagonal(&Vec3::new(0.083, 0.0, 0.083)),
        ),
    )
    .build();

let mut state = model.default_state();
state.q[0] = 0.3;

// Semi-implicit Euler: this crate provides the dynamics, you write the loop.
for _ in 0..500 {
    let qdd = aba(&model, &state);
    for k in 0..model.nv {
        state.v[k] += model.dt * qdd[k];
    }
    for k in 0..model.nq {
        state.q[k] += model.dt * state.v[k];
    }
}

let e = total_energy(&model, &state);
```

There is deliberately no `Simulator` type: integration policy belongs to the
caller. See [`phyz-guardian`](https://docs.rs/phyz-guardian) for adaptive
time-stepping and conservation monitoring on top of these primitives.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
