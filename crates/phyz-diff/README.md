# phyz-diff

Per-step Jacobians of the phyz equations of motion.

Three routines compute the same object — the Jacobians of one semi-implicit
Euler step — by different means:

| Function | Method | Exact? |
| --- | --- | --- |
| `finite_diff_jacobians` | central differences over the whole step | no |
| `semi_implicit_step_jacobians` | chain rule through the integrator, central differences on ABA | no |
| `symbolic::symbolic_step_jacobians` | symbolic differentiation of the dynamics | **yes** |

Only the symbolic path is derivative-exact. The other two are finite
differences and carry the usual step-size versus round-off tradeoff.

> `analytical_step_jacobians` was a misnomer — its ABA block has always been
> finite differences. It is deprecated in favour of
> `semi_implicit_step_jacobians`.

For **exact** gradients of a whole trajectory with respect to inertia
parameters and contact-surface geometry, use the dual-number adjoint in
[`phyz`](https://docs.rs/phyz)'s `diff` module instead.

## Example

```rust
use phyz_diff::{finite_diff_jacobians, semi_implicit_step_jacobians};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::ModelBuilder;

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

let jac = semi_implicit_step_jacobians(&model, &state);
let fd = finite_diff_jacobians(&model, &state, 1e-7);

// dq'/dq, dq'/dv, dv'/dq, dv'/dv, dv'/dctrl
assert_eq!(jac.dvnext_dq.nrows(), model.nv);
```

The `rollout` module lifts these to multi-step trajectories.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
