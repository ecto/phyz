# phyz-md

Molecular dynamics for phyz.

Classical MD with the standard potentials, plus a structure-of-arrays field
engine for larger systems.

| Type | Purpose |
| --- | --- |
| `MdSystem` | particles, bonds, box, thermostat |
| `LennardJones`, `Coulomb`, `HarmonicBond` | force-field terms |
| `ForceField` | the trait these implement |
| `NeighborList`, `minimum_image` | cell lists and periodic wrapping |
| `Thermostat` | Berendsen temperature control |

The `field` module holds the SoA engine: slice-based potentials,
velocity-Verlet integration, and FIRE energy minimization. `field::units`
defines the shared Å / eV / amu / fs / e / K constants — every potential in
this crate agrees on them.

## Example

```rust
use std::sync::Arc;

use phyz_md::{LennardJones, MdSystem};

// Argon-like LJ fluid: epsilon (eV), sigma (Å), cutoff (Å).
let ff = Arc::new(LennardJones::new(0.0103, 3.4, 8.5));
let mut system = MdSystem::new(ff, 1.0); // dt in fs
```

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
