# phyz-vbd

Vertex Block Descent for tetrahedral deformable bodies.

Implements Chen et al., *"Vertex Block Descent"* (SIGGRAPH 2024): a backward-Euler
step is the minimiser of a variational energy, and VBD minimises it by
Gauss–Seidel block coordinate descent over **vertices** — one guarded 3×3 Newton
step per vertex, no global matrix, no factorisation.

| Type | Purpose |
| --- | --- |
| `SoftBody` / `SoftBodyBuilder` | mesh, material, pins, lumped masses, colouring |
| `VbdSolver` / `VbdConfig` | the time loop: predict, sweep, update velocity |
| `TetElement` | stable Neo-Hookean FEM energy on a tetrahedron |
| `Spring` | mass-spring energy, as an *additional* term |
| `mesh::tet_box` | a deterministic hex-to-tet box mesher |
| `spd::spd_solve` | the eigenvalue-clamped local solve |
| `coloring::color_vertices` | deterministic greedy colouring |

## Example

```rust
use phyz_math::Vec3;
use phyz_vbd::{Material, SoftBody, VbdConfig, VbdSolver, mesh};

let (rest, tets) = mesh::tet_box(8, 1, 1, Vec3::new(1.0, 0.1, 0.1));
let clamped: Vec<usize> = rest
    .iter()
    .enumerate()
    .filter(|(_, p)| p.x == 0.0)
    .map(|(i, _)| i)
    .collect();

let mut body = SoftBody::builder(rest, Material::default())
    .tets(&tets)
    .pin(&clamped)
    .build()
    .unwrap();

let mut solver = VbdSolver::new(VbdConfig {
    dt: 1.0 / 30.0,
    iterations: 20,
    ..VbdConfig::default()
});
for _ in 0..30 {
    solver.step(&mut body);
}
```

## What is here, and what is not

**Here:** the outer loop; the per-vertex local solve with an eigenvalue-clamped
Hessian; stable Neo-Hookean FEM on tetrahedra; mass-spring as an extra energy;
deterministic greedy graph colouring; pinned vertices; Rayleigh stiffness
damping.

**Not here, deliberately:**

* **Collision and self-collision.** No contact energy of any kind. A body passes
  through itself and through everything else.
* **GPU.** The colouring is what a GPU backend needs and it is implemented and
  tested, but the sweep runs serially on the CPU.
* **Coupling to the rigid-body engine.** Pinned vertices are the only kinematic
  input; there is no two-way constraint against `phyz-rigid`.
* **The paper's accelerated initialisation** and its multi-level extension. The
  warm start is the plain inertial prediction.

## On "unconditionally stable"

The claim holds here in a specific sense, and the crate docs spell out the
assumptions. Briefly: every block step is a descent direction and the energy is
bounded below, so the iteration cannot blow up within a step at any timestep.
`tests/validation.rs` finds no divergence up to `h = 256 s` — about 1.6 × 10⁵
times the explicit CFL limit — on a scene where symplectic Euler goes
non-finite at `h = 1/60 s`. Stability is not accuracy: at those timesteps the
trajectory is heavily damped and wrong, just bounded. And the published stability results include contact in the energy,
which this crate does not implement, so the claim here is narrower than the
paper's.

## Validation

`cargo test -p phyz-vbd -- --nocapture` prints the measured numbers: static
equilibrium residual for a single tet, cantilever tip deflection against the
Euler–Bernoulli closed form, an axial-bar control against its exact solution,
energy drift over free vibration, and the largest bounded timestep.

Headline measurements, all printed by the test run:

| Check | Measured |
| --- | --- |
| Single-tet static equilibrium residual | 6.1e-15 N, i.e. 1.5e-14 of the vertex weight |
| Axial bar extension vs exact `ρgL²/2E` | 0.29% relative error |
| Cantilever tip deflection vs Euler–Bernoulli (8×1×1) | −2.97e-3 m vs −1.47e-2 m — **80% low** |
| Energy drift, free vibration, 0.2 s at `h = 1e-4 s` | −5.3% (dissipative, no gain) |
| Largest bounded timestep | ≥ 256 s; explicit diverges above ~1.6 ms |

The cantilever disagreement is real and is reported rather than tuned away. It
is a discretisation error, not a solver error: constant-strain tetrahedra lock
in bending, the error shrinks when the cross-section is refined (0.20× → 0.26×
of the reference going from 8×1×1 to 8×2×2, the finer mesh measured at a
slightly looser residual), and the axial control — same
material, same mesh, same solve, in the mode linear tets handle well — lands
within 0.3% of exact.

## Part of phyz

See the [workspace README](../../README.md).
