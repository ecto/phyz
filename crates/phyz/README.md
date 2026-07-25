# phyz

The differentiable rigid-body core of phyz.

`phyz` is self-contained. It bundles spatial algebra, articulated models,
Featherstone dynamics, collision, contact, and a reverse-mode differentiable
rollout — and it does **not** re-export the other crates in the workspace.
`phyz-gpu`, `phyz-particle`, `phyz-lbm` and friends are added separately.

## Modules

| Module | Contents |
| --- | --- |
| `math` | Re-export of [`phyz-math`](https://docs.rs/phyz-math): `Vec3`, `Mat3`, `Quat`, `SpatialVec`, `SpatialTransform`, `SpatialInertia` |
| `model` | `Model`, `ModelBuilder`, `Body`, `Joint`, `JointType`, `Actuator`, `State` |
| `rigid` | `aba`, `aba_with_external_forces`, `rnea`, `crba`, `forward_kinematics` |
| `collision` | `gjk_distance`, `epa_penetration`, `sweep_and_prune`, `AABB` |
| `contact` | `find_contacts`, `find_ground_contacts`, `contact_forces`, `ContactMaterial` |
| `diff` | `adjoint_rollout_gradient` — exact inertia-parameter and contact-vertex adjoints |

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

## Note

This crate currently vendors its own copies of the model/rigid/collision/contact
code rather than depending on `phyz-model`, `phyz-rigid`, `phyz-collision` and
`phyz-contact`. That means `phyz::model::Model` and `phyz_model::Model` are
distinct types. De-duplicating them is tracked separately.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
