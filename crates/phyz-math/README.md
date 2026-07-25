# phyz-math

Spatial algebra for the phyz physics engine.

Featherstone-style 6D spatial vectors and the linear algebra they need. This is
the type foundation every other phyz crate is built on.

## Types

| Type | What it is |
| --- | --- |
| `Vec3`, `Mat3`, `Mat4`, `Quat` | 3D primitives |
| `DVec`, `DMat` | Dynamically sized vectors and matrices |
| `SpatialVec` | 6D motion/force vector (`angular`, `linear`) |
| `SpatialMat` | 6×6 spatial matrix |
| `SpatialTransform` | Plücker transform between body frames |
| `SpatialInertia` | Rigid-body spatial inertia |
| `GRAVITY`, `skew` | Standard gravity, skew-symmetric cross-product matrix |

## Example

```rust
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, SpatialVec, Vec3};

// A 1 kg rod, centre of mass 0.5 m below the joint.
let inertia = SpatialInertia::new(
    1.0,
    Vec3::new(0.0, -0.5, 0.0),
    Mat3::from_diagonal(&Vec3::new(0.083, 0.0, 0.083)),
);

// Spatial velocity: 1 rad/s about Z.
let v = SpatialVec::new(Vec3::new(0.0, 0.0, 1.0), Vec3::zeros());

let xform = SpatialTransform::identity();
```

The scalar types are generic where differentiation needs them, which is what
lets `phyz`'s adjoint run dual numbers through the same dynamics code.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
