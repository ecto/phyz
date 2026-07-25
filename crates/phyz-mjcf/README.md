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
    eprintln!("{}: {}", note.tag, note.detail);
}
```

Malformed input is always an error, never a panic and never a silently
substituted default. Errors name the element and attribute they came from:

```text
<body> attribute 'euler' has invalid value "1 2": expected 3 numbers, found 2
<include file="scene.xml"> could not be read: No such file or directory (os error 2)
```

## Coverage

MJCF is large; this parser targets the subset needed for articulated
rigid-body models.

| Area | Support |
| --- | --- |
| `<compiler>` | `angle`, `coordinate` (local only), `eulerseq`, `meshdir`, `assetdir` |
| `<option>` | `gravity`, `timestep` |
| `<default>` | named classes, arbitrary nesting with inheritance, `class` and `childclass` |
| Bodies | `<body>`, `<inertial>` (`diaginertia` and `fullinertia`); inertia is derived from geoms + `density` when `<inertial>` is absent |
| Joints | `hinge`, `slide`, `ball`, `free`, `<freejoint>`, `range`/`limited`, `damping`, `armature`, `stiffness`, `springref`, `frictionloss` |
| Orientation | `quat`, `euler` (with `eulerseq` case rules), `axisangle`, `xyaxes`, `zaxis`, `fromto` |
| Geoms | `sphere`, `capsule`, `box`, `cylinder`, `plane`, `mesh` |
| Actuators | `motor`, `position`, `velocity`, `general` (joint transmissions) |
| Sensors | `<sensor>` elements recorded as `SensorElement` |
| Assets | `<mesh>` (STL binary/ASCII, OBJ) |
| Files | `<include>`, with cycle detection |

`DefaultsManager` implements MJCF's `<default>` class inheritance, so attributes
resolve the way MuJoCo resolves them rather than being read literally off each
element.

Every geom is carried into `Body::collisions` (or `Body::visuals`, for
`contype="0" conaffinity="0"`) as a `GeomInstance` with its own body-relative
pose, since a `fromto` capsule sits at its midpoint rather than the body origin.
`Body::geometry` mirrors the first centred collision shape so single-shape
consumers keep working.

Actuators use MuJoCo's affine model, so `position` and `velocity` servos are
special cases of `general` rather than separate code paths:

```text
force = gear * (gain * ctrl + bias_q * q + bias_v * v)
```

## Known gaps

Each of these is reported through `MjcfLoader::unsupported()` when a model uses
it.

**Blocked on `phyz-collision`:**

- `<hfield>`: there is no heightfield collision shape, so heightfield geoms are
  dropped.
- Meshes are loaded and become `Geometry::Mesh`, which phyz-collision treats as
  a convex hull. Non-convex meshes will collide as their hull.
- `ellipsoid` geoms have no phyz equivalent.

**Parsed and recorded, but not simulated** — phyz has no representation for
these, so they are noted and skipped: `<equality>`, `<tendon>`, `<contact>`,
`<keyframe>`.

**Not modelled:**

- Actuators with tendon/site/body transmissions (only joint transmissions are
  carried into the `Model`).
- `<texture>` and `<material>`, since phyz has no renderer to hand them to.

## Measuring coverage

`cargo run -p phyz-mjcf --example mjcf_coverage -- <file-or-dir>...` reports
bodies, DOFs, actuators, collision shapes, and gap notes for a set of models.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
