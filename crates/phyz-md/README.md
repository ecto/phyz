# phyz-md

Molecular dynamics for phyz.

Classical MD with the standard potentials, plus a structure-of-arrays field
engine for larger systems.

| Type | Purpose |
| --- | --- |
| `MdSystem` | particles, bonds, box, thermostat, integrator |
| `LennardJones`, `Coulomb`, `HarmonicBonds`, `HarmonicAngles` | non-bonded and bonded terms |
| `PeriodicDihedrals`, `HarmonicImpropers` | torsions |
| `Ewald`, `Pme` | periodic electrostatics |
| `NeighborList`, `Lattice`, `min_image` | cell lists and periodic wrapping |
| `Berendsen`, `NoseHoover`, `Barostat` | temperature and pressure control |
| `Rdf` | radial distribution function |

The `field` module holds the SoA engine: slice-based potentials,
velocity-Verlet integration, virial/pressure bundles, and FIRE energy
minimization. `field::units` defines the shared Å / eV / amu / fs / e / K
constants — every potential in this crate agrees on them, and
`field::units::FORCE_TO_ACCEL` is the one conversion that ties them together.

## Example

```rust
use phyz_math::Vec3;
use phyz_md::{LennardJones, MdSystem, Particle};

// Argon fluid, 1 fs timestep.
let mut system = MdSystem::lennard_jones(LennardJones::argon(), 1.0);

for i in 0..10 {
    system.add_particle(Particle::new(
        Vec3::new(i as f64 * 3.4, 0.0, 0.0),
        Vec3::zeros(),
        39.948, // argon mass (amu)
        0,      // species
    ));
}
```

`LennardJones::monatomic(epsilon, sigma, cutoff)` takes explicit parameters;
per-species tables with Lorentz-Berthelot mixing are also supported.

## Part of phyz

[`phyz`](https://github.com/ecto/phyz) is an open-source differentiable
multi-physics simulation workspace in pure Rust. Each crate is independent —
adding this one does not pull in the rest.

Licensed under [MIT](https://github.com/ecto/phyz/blob/main/LICENSE).
