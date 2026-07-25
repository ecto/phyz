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

`ElementDefaults` implements MJCF's `<default>` class inheritance, so
attributes resolve the way MuJoCo resolves them rather than being read
literally off each element.

## Coverage

MJCF is large; this parser targets the subset needed for articulated
rigid-body models — bodies, joints, geoms, inertials, actuators and defaults.
Unsupported elements are reported rather than silently ignored.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
