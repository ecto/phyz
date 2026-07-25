# phyz-urdf

URDF (ROS robot description) import for phyz.

Reads plain, non-xacro URDF and produces a [`phyz-model`](https://docs.rs/phyz-model)
`Model` with the kinematic tree, inertials, joint limits and dynamics, and
visual/collision primitives.

| Item | Purpose |
| --- | --- |
| `load_file`, `load_str` | parse a URDF from disk or from a string |
| `UrdfModel` | the resulting `Model`, plus robot name, mesh refs and warnings |
| `UrdfOptions` | how the root link attaches, timestep, gravity |
| `BaseKind` | fixed, floating, or free root attachment |
| `MeshRef` | a mesh the URDF referenced but that was not loaded |
| `robot_to_model`, `actuated_dofs`, `rpy_to_matrix` | lower-level conversion helpers |
| `UrdfError`, `Result` | error handling |

## Example

```rust,no_run
let robot = phyz_urdf::load_file("panda.urdf", &Default::default()).unwrap();
println!("{} bodies, {} DOF", robot.model.nbodies(), robot.model.nv);

// Import is lossy at the edges, and says so rather than guessing.
for w in &robot.warnings {
    eprintln!("warning: {w}");
}
```

## What it will not silently guess

Two things importers commonly get wrong, handled explicitly here:

- **Meshes are not fabricated.** phyz's `Geometry::Mesh` needs real vertices;
  URDF only gives a file path, often `package://…` that resolves through a ROS
  workspace. Rather than substitute a bounding box and pass it off as the
  robot's collision shape, unloaded meshes are surfaced as `MeshRef` for the
  caller to resolve.
- **Lossy conversions are reported.** `UrdfModel::warnings` being non-empty is
  not an error, but anything safety-relevant should inspect it.

## Not supported

`.xacro` files need macro expansion before they are URDF at all. Preprocess
them first:

```bash
xacro model.xacro > model.urdf
```

## Parser

Parsing is delegated to [`urdf-rs`](https://crates.io/crates/urdf-rs), the
parser behind the OpenRR robotics stack — it already handles the fiddly schema
details (whitespace-separated numeric attributes, optional-element defaults,
`<mimic>`, `<safety_controller>`, `package://` expansion) and is maintained
against real robot descriptions. The *physics* conventions, which are the part
that actually goes wrong, are handled explicitly in this crate's `convert`
module rather than by the parser.

For MuJoCo models, see [`phyz-mjcf`](https://docs.rs/phyz-mjcf).

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
