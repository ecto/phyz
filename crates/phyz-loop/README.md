# phyz-loop

Kinematic-loop closure constraints and Proximal-ADMM constrained forward
dynamics for phyz.

`phyz-rigid` is a reduced-coordinate Featherstone solver over a kinematic
**tree**. A tree has exactly one path from the world to every body, so a closed
chain — four-bar linkage, slider-crank, delta robot, any parallel manipulator —
is not *slow* there, it is **unrepresentable**. This crate closes that gap: model
the mechanism as a spanning tree, cut each loop, and re-impose the cut joint as
an explicit constraint solved alongside the dynamics.

The formulation follows the Proximal-ADMM forward dynamics of NVIDIA/Disney
Newton's `kamino` solver (arXiv:2504.19771).

## What it does

| Item | Detail |
| --- | --- |
| `LoopConstraint::point` | 3 rows: two anchor points coincide (a ball joint closing the loop) |
| `LoopConstraint::weld` | 6 rows: coincident *and* relative orientation held at the captured value |
| `assemble` | `J`, `c(q)`, `J v` and `Jdot v` at a state |
| `solve` | constrained forward dynamics — Proximal ADMM |
| `project_velocity` | least-squares projection of `v` onto `J v = 0`, for consistent initial conditions |
| `step` | `solve` plus the unmodified `phyz_rigid::semi_implicit_euler` |
| `grubler` / `mobility` | closed-form and numerical (`nv - rank J`) mobility |

Jacobian rows come from `phyz_rigid::point_jacobian` and
`phyz_rigid::body_angular_jacobian` — the same rows the contact solver uses, not
a second implementation.

## The solve

```text
minimize_a  1/2 (a - a_free)^T M (a - a_free)   subject to   J a = b
```

Gauss's principle of least constraint. Solved by Proximal ADMM rather than by
factoring the KKT matrix, because `J M^-1 J^T` is singular whenever the
constraints are redundant — and redundancy is *normal* in mechanisms. Even the
plain planar four-bar is redundant: its 3-row point closure has an identically
zero out-of-plane row, so `rank(J) = 2`.

Two proximal terms do the work. `sigma` on the primal keeps the factored system
SPD even with a massless link. `epsilon` on the dual bounds the multiplier when
the constraint set is redundant *and* inconsistent — which is what a consistent
redundant set becomes after one step of drift. Full derivation, including why
`epsilon > 0` costs a small non-zero residual, is in the `solver` module docs.

## Stabilization

**Baumgarte**, folded into the acceleration target as
`b = -Jdot v - 2 alpha (J v) - beta^2 c`. Chosen over post-step position
projection because it keeps the step to one linear solve, needs no change to
the existing integrator, and stays differentiable. Position projection is *not*
implemented.

Baumgarte does not eliminate drift; it damps it. Measured, spinning the crank of
the test four-bar at 4 rad/s for the full duration:

| Setting | max \|c\|_inf over the rollout |
| --- | --- |
| default gains, `dt = 1e-3`, 10 s | **8.7e-4 m** |
| default gains, `dt = 2e-4`, 2 s | **3.8e-5 m** |
| **no** stabilization, `dt = 1e-3`, 2 s | **1.0e-1 m** |

The residual falls with `dt` rather than sitting on a floor, which is what says
the remaining error is the integrator's and not a modelling error in the
constraint. Velocity residual `|J v|_inf` over the same 10 s run: 3.7e-2 m/s at
`dt = 1e-3`, 9.3e-3 m/s at `dt = 2e-4`.

Cost: 2.8 ADMM iterations per step on average (max 4) over 10 000 steps of the
four-bar, zero non-converged steps.

## Validation

- **Four-bar mobility.** `nv - rank(J) = 1`, agreeing with the planar Grübler
  count `3(4 - 1 - 4) + 4 = 1`.
- **Slider-crank against closed-form kinematics.**
  `x = r cos(theta) + sqrt(l^2 - r^2 sin^2(theta))`. Over 5 s at `dt = 1e-3`
  with the crank at 6 rad/s: max `|x - x_analytic|` = 6.8e-4 m, max
  `|xdot - xdot_analytic|` = 3.0e-2 m/s against a peak slider speed of 4.1 m/s.
- **Redundant / over-constrained loops.** Duplicating the closure leaves the
  accelerations unchanged to 1.5e-7 rad/s². Making the two copies disagree by
  1 µm — an unsatisfiable set — keeps the accelerations within 3.7e-3 rad/s² of
  the nearby well-posed problem and reports `converged: false`, which it will
  always do, because the residual floor of an unsatisfiable set is the
  inconsistency itself. The multiplier reaches 1.0e3 at the default
  `epsilon = 1e-9` — there it is the iteration cap doing the bounding, not
  `epsilon`, whose cap sits far above the budget; at `epsilon = 1e-4`, a
  compliance sized to this problem, it is 5.0e1 with the accelerations
  unchanged. With `epsilon = 0` there is no fixed point at all: 10× the
  iterations gives 10× the multiplier, measured.
- **Determinism.** Two independent 500-step rollouts of both mechanisms agree in
  every bit, including the solver's own residual and iteration count.

## Not implemented

Stated rather than half-done:

- **Contact and friction.** `phyz-contact` owns the inequality/cone side. A
  combined loop-plus-contact solve is a real thing to want; it is not here.
- **GPU.** CPU `f64` only.
- **Actuation and joint limits**, or any other inequality constraint. Actuator
  forces enter only through ABA's free acceleration.
- **Post-step position projection.** Baumgarte only.
- **Automatic loop detection.** You state the closures; nothing scans the model
  for them.
