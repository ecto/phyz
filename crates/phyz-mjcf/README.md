# phyz-mjcf

MuJoCo MJCF model loading for phyz.

Parses MuJoCo's XML model format into a
[`phyz-model`](https://docs.rs/phyz-model) `Model`, so existing MuJoCo assets
run against phyz's dynamics.

## Example

```rust
use phyz_mjcf::MjcfLoader;

let xml = r#"
<mujoco>
  <worldbody>
    <body name="pole">
      <joint name="hinge" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"#;

let loader = MjcfLoader::from_xml_str(xml).unwrap();
let model = loader.build_model();
println!("{} bodies, nq={} nv={}", model.nbodies(), model.nq, model.nv);
```

Or straight from disk with `MjcfLoader::from_file("models/ant.xml")`.

Anything the parser understood but the `Model` cannot represent is reported
rather than dropped silently:

```rust
use phyz_mjcf::MjcfLoader;

let loader = MjcfLoader::from_xml_str("<mujoco><worldbody/></mujoco>").unwrap();
for note in loader.unsupported() {
    eprintln!("{}: {}", note.element, note.detail);
}
```

Malformed input is always an error, never a panic and never a silently
substituted default. Errors name the element and attribute they came from:

```text
<body> attribute 'pos' has invalid value "1 2": expected 3 numbers, found 2
<geom> references undefined default class 'leg_'
```

## Coverage

MJCF is large; this parser targets the subset needed for articulated
rigid-body models.

| Area | Support |
| --- | --- |
| `<compiler>` | `angle`, `coordinate` (local only), `eulerseq`, `meshdir`, `assetdir` |
| `<option>` | `gravity`, `timestep` |
| `<default>` | named classes, arbitrary nesting with inheritance, `class` and `childclass` |
| Bodies | `<body>`, `<inertial>` (`diaginertia` and `fullinertia`), `<site>` |
| Joints | `hinge`, `slide`, `ball`, `free`, `<freejoint>`, `range`/`limited`, `damping`, `armature`, `stiffness`, `springref`, `frictionloss` |
| Orientation | `quat`, `euler` (with `eulerseq` case rules), `axisangle`, `xyaxes`, `zaxis`, `fromto` |
| Geoms | `sphere`, `capsule`, `box`, `cylinder`, `plane`, `mesh` |
| Actuators | `motor`, `position`, `velocity`, `general` (joint transmissions) |
| Assets | `<mesh>` (STL binary/ASCII, OBJ), `<texture>`, `<material>`, `<hfield>` |
| Files | `<include>`, with cycle detection |

`DefaultsManager` implements MJCF's `<default>` class inheritance, so attributes
resolve the way MuJoCo resolves them rather than being read literally off each
element.

Every geom on a body is carried into `Body::collisions` as a `GeomInstance` with
its own body-relative pose. `Body::geometry` mirrors the first shape that is
actually centred on the body frame, so the single-shape contact path is
unchanged.

Actuators use MuJoCo's affine model, so `position` and `velocity` servos are
special cases of `general` rather than separate code paths:

```text
force = gear * (gain * ctrl + bias[0] + bias[1] * q + bias[2] * qdot)
```

## Known gaps

Each of these is reported through `MjcfLoader::unsupported()` when a model uses
it.

**Blocked on `phyz-collision`:**

- `<hfield>`: parsed and exposed via `MjcfLoader::hfields()`, but there is no
  heightfield collision shape.
- Meshes are loaded and become `Geometry::Mesh`, which phyz-collision treats as
  a convex hull. Non-convex meshes will collide as their hull.
- `ellipsoid` geoms have no phyz equivalent.

**Parsed and recorded, but not simulated** — phyz has no representation for
these, so they are noted and skipped: `<equality>`, `<tendon>`, `<sensor>`,
`<contact>` (pairs and excludes), `<keyframe>`.

**Not modelled:**

- Actuators with tendon/site/body transmissions (only joint transmissions are
  carried into the `Model`).
- `general` actuators with a non-`fixed` gain type, non-`affine` bias type, or
  any `dyntype` — actuator state integration does not exist in phyz.
- `compiler coordinate="global"`, which is rejected outright rather than
  misinterpreted as local coordinates.
- `inertiafromgeom`: a body with geoms but no `<inertial>` gets a 1 kg
  placeholder rather than an inertia computed from its geometry.

## Compound joints

MJCF allows several `<joint>` elements on one body, which together form a single
compound joint. These are expanded into a serial chain of links, with the body's
inertia on the last one so its mass is counted once. A body with three hinges
therefore appears as three entries in `Model::bodies` contributing 3 DOFs.

## Measuring coverage

`cargo run -p phyz-mjcf --example mjcf_coverage -- <file-or-dir>...` reports
bodies, DOFs, actuators, collision shapes, and gap notes for a set of models.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
