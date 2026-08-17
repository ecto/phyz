# phyz-xpbd

Extended Position-Based Dynamics for phyz: cloth, tetrahedral soft bodies and
cables from one solver.

Implements Macklin, Müller & Chentanez, *"XPBD: Position-Based Simulation of
Compliant Constrained Dynamics"* (MIG 2016), with the substepping of Müller et
al., *"Small Steps in Physics Simulation"* (SCA 2020).

| Type | Purpose |
| --- | --- |
| `ParticleSystem` | positions, velocities, inverse masses (`0` = pinned) |
| `Constraint` | a scalar constraint plus its compliance and Lagrange multiplier |
| `ConstraintKind` | distance, dihedral bending, tetrahedron volume, attachment |
| `XpbdSolver` | the predict / project / derive-velocity time loop |

## Why compliance, not "stiffness in [0, 1]"

Plain PBD scales each constraint correction by a factor in `[0, 1]`. That
factor is not a material property: the same value produces a stiffer cloth at
more iterations and at smaller timesteps, so you cannot say what you simulated.

XPBD takes a compliance `α = 1/k` — the physical inverse stiffness, m/N for a
distance constraint — and carries a Lagrange multiplier through each substep:

```text
α̃  = α / h²
Δλ = (−C − α̃ λ) / (Σ wᵢ |∇Cᵢ|² + α̃)
Δxᵢ = wᵢ ∇Cᵢ Δλ
```

The `−α̃ λ` term makes the converged deformation `α · f` and nothing else. The
crate's tests measure it: a hanging mass stretches by `α·m·g` to a relative
error of 5.3e-12.

Iteration independence is measured on a **coupled** system, since a single
constraint cannot show it — one sweep solves a one-constraint problem exactly,
so any iteration count agrees bit-for-bit whether the compliance feedback works
or not. On a 20-link chain, deviation from the 32-iteration answer:

| iterations | 1 | 2 | 3 | 4 | 8, 16, 32 |
|---|---|---|---|---|---|
| deviation | 2.6e-3 | 1.4e-5 | 7.1e-8 | 3.7e-10 | bit-identical |

Iterations buy convergence speed; `α` alone decides where it converges to.

## Example

```rust
use phyz_math::Vec3;
use phyz_xpbd::{Constraint, ParticleSystem, XpbdSolver};

// A 12-link cable, pinned at one end, swinging under gravity.
let mut p = ParticleSystem::new();
let pin = p.add_pinned(Vec3::zeros());
let mut idx = vec![pin];
for k in 1..=12 {
    idx.push(p.add(Vec3::new(0.1 * k as f64, 0.0, 0.0), 0.25));
}

// Compliance 0.0 is an inextensible cable; raise it for a stretchy one.
let mut constraints: Vec<Constraint> = (0..12)
    .map(|j| Constraint::distance(idx[j], idx[j + 1], 0.1, 0.0))
    .collect();

let solver = XpbdSolver::new(1.0 / 60.0);
for _ in 0..120 {
    solver.step(&mut p, &mut constraints);
}

// Rigid links keep their length — to the accuracy the projection budget buys.
// At the default 10 substeps x 1 iteration a swinging 12-link chain holds each
// link to ~1.4e-4 m of its rest length, not to machine precision: information
// takes one iteration per link to reach the pin. Raise `iterations` (or
// `substeps`) to tighten it.
let len = (p.positions[idx[1]] - p.positions[idx[0]]).norm();
assert!((len - 0.1).abs() < 5e-4, "link length {len}");
```

## Substeps versus iterations

The default is 10 substeps × 1 iteration. For a fixed projection budget,
substepping beats iterating: it reduces integration error as well as constraint
error, and each solve starts closer to the constraint manifold. Systems with
long stiff chains are the exception — Gauss–Seidel propagates one constraint
per pass, so an `n`-link chain needs `O(n)` projections either way. Measured on
a 20-link chain at rest: worst node depth error 5.2e-3 m at 10 substeps × 1
iteration, 3.0e-9 m at 10 × 20.

## Determinism

Pure `f64`, one thread, a fixed sweep in constraint-list order, no hash maps in
the solve. Two identical runs are bit-identical, and a test asserts it on raw
bit patterns. Because Gauss–Seidel is order-dependent, the order of your
constraint list is part of the input, not an implementation detail.

## Not implemented

Deliberately, and stated here so no one assumes otherwise:

- **No rigid-body coupling.** Particles only; nothing here talks to
  `phyz-rigid`.
- **No collision at all** — no self-collision, no particle-particle, no
  collision against the rigid world, and therefore no friction.
- **No GPU and no parallelism.** The Gauss–Seidel sweep is sequential.
- **Bending is skipped at coplanar configurations**, where the dihedral
  gradient is numerically singular.
- **Volume preservation is per-tetrahedron**, which is not global
  incompressibility.

## Part of phyz

See the [workspace README](../../README.md) for the full crate list.
