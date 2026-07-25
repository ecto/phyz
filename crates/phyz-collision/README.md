# phyz-collision

GJK/EPA collision detection for phyz.

Narrow-phase distance and penetration between convex shapes, plus a
sweep-and-prune broad phase.

| Function | Purpose |
| --- | --- |
| `gjk_distance`, `gjk_distance_rot` | separating distance between disjoint convex shapes |
| `epa_penetration`, `epa_penetration_rot` | penetration depth and normal when they overlap |
| `sweep_and_prune` | broad-phase AABB pair pruning |

Shapes are described by `Geometry` (`Sphere`, `Capsule`, `Box`, `Cylinder`,
`Mesh`, `Plane`) and bounded by `AABB`. Results come back as `Collision`
records that [`phyz-contact`](https://docs.rs/phyz-contact) turns into forces.

## Example

```rust
use phyz_collision::{AABB, Geometry, gjk_distance, sweep_and_prune};
use phyz_math::Vec3;

let a = Geometry::Sphere { radius: 0.5 };
let b = Geometry::Sphere { radius: 0.5 };

let d = gjk_distance(&a, &b, &Vec3::zeros(), &Vec3::new(2.0, 0.0, 0.0));

// Broad phase over a set of AABBs.
let boxes = vec![
    AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(0.0, 0.0, 0.0)),
    AABB::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5)),
];
let pairs = sweep_and_prune(&boxes);
```

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
