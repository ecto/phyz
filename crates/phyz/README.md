# phyz

The differentiable rigid-body core of phyz.

`phyz` holds no code of its own. It re-exports the focused `phyz-*` crates that
make up the rigid-body stack, so `cargo add phyz` gets you all of them behind
one dependency and `phyz::collision` is literally `phyz_collision`.

It does **not** cover the whole workspace — `phyz-gpu`, `phyz-particle`,
`phyz-lbm` and friends are separate crates you add explicitly.

## Features

`collision`, `contact` and `diff` are on by default; `math`, `model` and
`rigid` are always present.

```bash
cargo add phyz                                  # everything below
cargo add phyz --no-default-features            # math + model + rigid only
cargo add phyz --no-default-features -F diff    # ...plus gradients
```

## Modules

| Module | Contents |
| --- | --- |
| `math` | [`phyz-math`](https://docs.rs/phyz-math): `Vec3`, `Mat3`, `Quat`, `SpatialVec`, `SpatialTransform`, `SpatialInertia` |
| `model` | [`phyz-model`](https://docs.rs/phyz-model): `Model`, `ModelBuilder`, `Body`, `Joint`, `JointType`, `Actuator`, `State` |
| `rigid` | [`phyz-rigid`](https://docs.rs/phyz-rigid): `aba`, `rnea`, `crba`, `forward_kinematics`, `body_wrenches` |
| `collision` | [`phyz-collision`](https://docs.rs/phyz-collision): `gjk_distance`, `epa_penetration`, `ray_cast`, `sweep_and_prune`, `AABB` |
| `contact` | [`phyz-contact`](https://docs.rs/phyz-contact): `find_contacts`, `find_ground_contacts`, `contact_forces`, `ContactMaterial` |
| `diff` | [`phyz-diff`](https://docs.rs/phyz-diff): `adjoint_rollout_gradient`, plus the per-step Jacobians |

## Forward dynamics

```rust
use phyz::{Mat3, ModelBuilder, SpatialInertia, SpatialTransform, Vec3};
use phyz::rigid::aba;

let model = ModelBuilder::new()
    .gravity(Vec3::new(0.0, -9.81, 0.0))
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

let qdd = aba(&model, &state);   // joint accelerations, O(n)
assert_eq!(qdd.len(), model.nv);
```

## Gradients

`diff::adjoint_rollout_gradient` runs one forward rollout and one backward
adjoint pass, returning:

* `d_inertia[i]` — `dJ/dπ` for body `i`'s ten spatial-inertia scalars
  `[m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`
* `d_vertices[i]` — `∂J/∂x` per collision-mesh vertex, when ground contact is
  configured

Both are **exact**: the per-step Jacobians come from dual numbers through a
scalar-generic ABA, not finite differences. See the repository README for a
worked example.

For per-step Jacobians with respect to `(q, v, ctrl)`, use
[`phyz-diff`](https://docs.rs/phyz-diff) instead.

## Naming

Each sub-crate is reachable under both names: `phyz::rigid::aba` and
`phyz::phyz_rigid::aba` are the same function, and both are
`phyz_rigid::aba`. Types cross crate boundaries freely because there is only
ever one of them.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
