# phyz-contact

Soft contact dynamics and friction for phyz.

Penalty-based contact resolution on top of
[`phyz-collision`](https://docs.rs/phyz-collision)'s `Collision` records.

| Function | Purpose |
| --- | --- |
| `find_contacts` | body-body contacts from model geometry |
| `find_ground_contacts` | contacts against a horizontal ground plane |
| `contact_forces` | explicit penalty wrenches |
| `contact_forces_implicit` | semi-implicit variant, stable at larger `dt` |
| `compute_contact_force`, `compute_contact_force_implicit` | single-contact kernels |

The convex solve (`assemble` + `solve_contacts`) is what the stepper uses, and
what supersedes the penalty functions above. It carries MuJoCo-style
`solref`/`solimp` position stabilization, so penetration is repaid rather than
frozen in place, combines the two contacting bodies' materials
(`ContactMaterial::combine`), and warm starts from the previous step's impulses
via `ContactCache`.

`ContactMaterial` carries stiffness, damping and friction. The implicit form
solves for the normal impulse that the next step will produce, rather than the
one the current penetration implies, which is what keeps stacks from exploding
at practical timesteps.

## Example

```rust
use phyz_contact::{ContactMaterial, contact_forces, find_ground_contacts};

// `state.body_xform` must be up to date — call
// `phyz_rigid::forward_kinematics` first.
# fn demo(model: &phyz_model::Model, state: &phyz_model::State) {
let geometries: Vec<_> = model.bodies.iter().map(|b| b.geometry.clone()).collect();
let contacts = find_ground_contacts(state, &geometries, 0.0);
let materials = vec![ContactMaterial::default(); contacts.len()];
let wrenches = contact_forces(&contacts, state, &materials, None);
// feed `wrenches` to phyz_rigid::aba_with_external_forces
# }
```

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
