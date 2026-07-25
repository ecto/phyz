# phyz-model

Articulated model and state types for phyz.

The static description of a mechanism (`Model`) and its time-varying
configuration (`State`). Built with a fluent `ModelBuilder`; consumed by
[`phyz-rigid`](https://docs.rs/phyz-rigid), [`phyz-diff`](https://docs.rs/phyz-diff),
[`phyz-gpu`](https://docs.rs/phyz-gpu) and the rest of the workspace.

## Types

* `Model` — bodies, joints, actuators, gravity, `dt`, `nq`, `nv`
* `ModelBuilder` — `add_revolute_body`, `add_prismatic_body`,
  `add_spherical_body`, `add_free_body`, `add_fixed_body`, `add_body`
* `Body`, `Joint`, `JointType`, `Geometry`, `Actuator`
* `State` — `q`, `v`, `ctrl`, `time`, `body_xform`, `qfrc_external`

## Example

```rust
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::ModelBuilder;

let model = ModelBuilder::new()
    .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
    .dt(0.002)
    .add_revolute_body(
        "upper",
        -1,                              // parent: -1 is the world
        SpatialTransform::identity(),    // parent frame -> joint frame
        SpatialInertia::new(
            1.0,
            Vec3::new(0.0, -0.5, 0.0),
            Mat3::from_diagonal(&Vec3::new(0.083, 0.0, 0.083)),
        ),
    )
    .add_revolute_body("lower", 0, SpatialTransform::identity(), /* ... */
        SpatialInertia::new(0.8, Vec3::new(0.0, -0.4, 0.0),
            Mat3::from_diagonal(&Vec3::new(0.043, 0.0, 0.043))))
    .build();

let state = model.default_state();
assert_eq!(model.nq, 2);
```

Bodies form a kinematic tree: each body names its parent by index, `-1` meaning
the world. Joint transforms are Plücker transforms from the parent body frame
to the joint frame.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
