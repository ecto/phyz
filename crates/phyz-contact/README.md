# phyz-contact

Soft contact dynamics and friction for phyz.

Penalty-based contact resolution on top of
[`phyz-collision`](https://docs.rs/phyz-collision)'s `Collision` records.

| Function | Purpose |
| --- | --- |
| `find_contacts` | body-body contacts from model geometry |
| `find_ground_contacts` | contacts against a horizontal ground plane (one centred shape per body) |
| `find_ground_contacts_model` | ground contacts over `Body::collisions` — every shape, offsets and orientations included |
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

## Per-body materials, and which body to put one on

A material belongs to a **body**, via `phyz_model::Body::material`. `None` —
the default — means "the scene material", so a model that sets nothing behaves
exactly as it did when every caller passed `vec![one; n_bodies]`.

```rust
# use phyz_model::ContactMaterial;
# fn demo(model: &mut phyz_model::Model) {
let asphalt = ContactMaterial { friction: 0.8, ..Default::default() };

// Grip tape on rubber soles: the one contact that must never slip.
let grip = ContactMaterial { friction: 1.5, ..Default::default() };
model.set_body_material("left_foot", grip.clone());
model.set_body_material("right_foot", grip);
// 95A urethane: the one that is meant to roll.
model.set_body_material(
    "wheel_fl",
    ContactMaterial {
        friction: 0.75,
        restitution: 0.15,
        ..Default::default()
    },
);

// What you hand to `assemble`.
let materials = model.contact_materials(&asphalt);
# let _ = materials;
# }
```

**Friction combines by `max`, and that decides which body you attach a
material to.** `ContactMaterial::combine` takes the elementwise maximum of the
pair's friction (MuJoCo's rule — see its docs for why), so a grippy body grips
*everything it touches*, not just the surface you had in mind. A body's
material applies to all of its contacts; there is no per-pair override.

The worked example, from a skateboarding robot where it was measured:

- Grip tape on rubber soles is about `mu = 1.5`, while 95A urethane wheels
  want `0.75`. With both on one shared `0.9`, the feet **slid on the deck**: a
  searched "ollie" turned out to be a foot hooking the deck's edge and
  dragging the board up by friction.
- Putting that `1.5` on the **deck** body instead of the shoes fixes the
  foot-on-deck contact — `max(1.5, 0.9) = 1.5` either way — but it also
  raises the *deck-on-ground* contact, and the deck's underside is bare wood
  that scrapes the asphalt during the pop. Measured, that braked the board
  from 0.6 to 0.39 m/s and collapsed the trick to 2 ms of air.
- Putting it on the **shoes** raises only foot-on-deck and leaves
  deck-on-ground at wood's `0.6`, which is the physics.

The rule of thumb: attach the material to the part whose *surface* it
describes, then check what else that part touches. Two bodies that must grip
each other but slide on everything else cannot be expressed by per-body
materials at all — split the geometry into separate bodies, or lower the scene
material so `max` lands where you want it.

## Example

```rust
use phyz_contact::{ContactMaterial, contact_forces, find_ground_contacts};

// `state.body_xform` must be up to date — call
// `phyz_rigid::forward_kinematics` first.
# fn demo(model: &phyz_model::Model, state: &phyz_model::State) {
let geometries: Vec<_> = model.bodies.iter().map(|b| b.geometry.clone()).collect();
let material = ContactMaterial::default();
// The last argument is the contact margin: candidates within it of the plane
// are kept with a negative depth, so a support point's normal force ramps to
// zero instead of being cut off while it is still carrying load.
let contacts = find_ground_contacts(state, &geometries, 0.0, material.margin);
let materials = vec![material; contacts.len()];
let wrenches = contact_forces(&contacts, state, &materials, None);
// feed `wrenches` to phyz_rigid::aba_with_external_forces
# }
```

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
