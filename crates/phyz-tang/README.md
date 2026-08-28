# phyz-tang

One phyz simulation step as a differentiable op, so short-horizon analytic
policy gradients, gradient-based sysid, and trajectory optimisation become
ordinary tang training loops instead of bespoke programs.

```rust
let op = PhysicsStep::new(&model);              // the scene
let mut tape = PhysicsTape::new(&op, state0);   // the trajectory
for u in &controls {
    tape.step(u)?;                              // forward
}
let g = tape.backward(&cotangent)?;             // reverse sweep
// g.d_state0, g.d_ctrl[t] — straight into tang_train::Parameter
```

## Why the crate lives phyz-side

`phyz-diff` already depends on `tang`. A `tang-phyz` crate in the tang
workspace would close a dependency cycle, so the bridge lives here and tang
stays the leaf.

## Where the derivative comes from

`phyz_diff::convex_adjoint_gradient` is trajectory-level and
objective-scoped. A VJP is that with `steps = 1` and a *linear* objective:
set `J = w·(q_out, v_out)` and the reported `d_q0`, `d_v0`, `d_ctrl[0]` are
`wᵀ ∂state_out/∂·` by definition — one call, not one per output component.

Set `PHYZ_SOLVER_ADJOINT=1` to close the contact channel through the solver's
own executed sweeps rather than the implicit function theorem, so a truncated
solve yields a gradient instead of a refusal. Both modes are tested here.

## The one-forward-caching answer

tang-train's `Module` caches its forward's input in a single `Option<Tensor>`
that the next forward overwrites, so forward-forward-backward silently
returns the *second* forward's gradient. `PhysicsTape` has no such slot: each
`step` **appends** `(state_in, ctrl)`, `backward` takes `&self` and is
re-runnable with a different seed, and a new trajectory is the explicit
`reset`. Nothing can go stale because every step's VJP is re-derived from the
inputs recorded for that step — which is affordable only because the contact
solve is deterministic and the backward can re-execute it.

## Determinism, and the price of a pure step

The op's forward is a fresh-cache one-step rollout, because a
`(state_in, ctrl)` signature does not name `Simulator`'s warm-start cache. An
N-step tape is therefore identical to `Simulator` with
`with_warm_start(false)`, not to the warm-started default. Cold-starting
every step costs solver iterations and buys a step function whose gradient is
the gradient of what ran.

## Validation

`cargo test -p phyz-tang` — every gradient checked against a central finite
difference of the executed forward. Worst relative error, non-axis-aligned
cotangent:

| rung | d_state_in | d_ctrl |
| --- | --- | --- |
| ballistic (no contact, no ctrl) | 7.7e-9 | 8.3e-8 |
| free flight (no contact, driven) | 1.5e-8 | 8.2e-8 |
| resting contact | 1.4e-4 | 8.2e-8 |
| sliding + driven contact | 3.9e-3 | 2.6e-7 |
| **32-step chained rollout** | **1.9e-4** | **1.1e-5** |

The contact-free rungs are not exact because `contact_adjoint` still
finite-differences its own lanes internally (`PHYZ_ADJOINT_FD_EPS`, default
1e-8); that instrument's noise is the ~1e-8 floor. The contacted residual is
the known frozen-contact-anchor gap.

## Demo

`cargo run --release -p phyz-tang --example trajopt_box` — 12 control
scalars optimised through 40 chained ops to put a box on a target, with
`tang_train::ModuleAdam` closing the loop. Converges in ~60 iterations,
~0.6 s.
