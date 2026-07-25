# phyz-format

The `.phyz` scene format, with MJCF and URDF import.

A multi-domain scene description: not just rigid bodies, but particle, field
and coupled domains in one file.

| Type | Purpose |
| --- | --- |
| `PhyzSpec`, `WorldConfig` | the schema root |
| `Domain`, `DomainType`, `RigidBodyDomain` | per-domain descriptions |
| `Coupling`, `CouplingType`, `ForceTransfer` | declared couplings between domains |
| `export_phyz`, `load_phyz_model` | serialization |
| `from_mjcf`, `from_urdf` | importers |
| `TauFormatError`, `Result` | error handling |

Multi-domain is the point: MJCF and URDF describe articulated rigid bodies and
nothing else, so a scene that also contains a fluid or a field has nowhere to
live in those formats.

## Example

```rust,no_run
use phyz_format::{export_phyz, load_phyz_model};

# fn demo(spec: &phyz_format::PhyzSpec) -> phyz_format::Result<()> {
let json = export_phyz(spec)?;
# Ok(())
# }
```

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
