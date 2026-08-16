# Differentiable Contact: Design

**Status:** largely implemented — see [§8](#8-implementation-status) for what landed, what diverged, and what is still open
**Scope:** replaces `crates/phyz-contact` and the vendored `crates/phyz/src/contact/`
**Author:** design phase, 2026-07
**Target integration point:** `crates/phyz-diff/src/rollout/`, `crates/phyz-rigid/src/aba.rs`

---

## 0. Why this rewrite exists

phyz's positioning is *physics as a differentiable function* in a compiled language.
Rapier and Avian have no gradients at all. MuJoCo/MJX has excellent gradients but only
inside a Python/JAX stack — you cannot embed MJX in a Rust robot controller, a WASM
build, or a `no_std` target. Dojo has the best contact gradients in the literature but
lives in Julia. The gap phyz can occupy is: *contact-rich differentiable simulation
that compiles to a static library.*

Contact is where that positioning is won or lost. Everything else in a rigid-body
engine (ABA, FK, integration) is smooth and trivially differentiable — `phyz-diff`
already differentiates all of it exactly. Contact is the only place where the
derivative is hard, and it is therefore the only place where the differentiator is
real.

### 0.1 What is actually wrong today

Read in order:

| Location | Problem |
|---|---|
| `crates/phyz-contact/src/lib.rs:45`, `:114` | `friction = min(mu*Fn, c*tangent_speed)`. This is not Coulomb friction. It is a viscous damper clipped by a Coulomb bound. As `‖v_t‖ → 0` the friction force → 0 **regardless of normal load**, so nothing ever sticks: a block on a level plane creeps forever, and a block on any incline slides no matter the friction coefficient. There is no cone, no stiction, no anchor. |
| `crates/phyz-contact/src/material.rs:13-17` | `bounce`, `soft_cfm`, `soft_erp` are public fields that no solver reads. `ContactMaterial::bouncy()` is a lie — restitution is entirely unimplemented. |
| `crates/phyz-contact/src/solver.rs:102-126` | Contact normal defaults to `normalize(pos_j - pos_i)` — the vector between *body centers* — with EPA only as a degenerate fallback. Contact point is `(pos_i + pos_j) * 0.5`, the midpoint of the two centers, which is generally inside both bodies and not on either surface. |
| `crates/phyz-contact/src/solver.rs` (whole file) | One contact per pair. A box resting on a plane gets a single point, so it has no resistance to tipping and the stack jitters. |
| `crates/phyz-contact/src/solver.rs:44-64`, `:80-83`, `:122-124` | Extensive NaN plumbing: NaN-tagged AABBs, finiteness filters, contact-dropping guards. This is not defensive rigor, it is a symptom. A stiff explicit penalty spring (`stiffness: 50000.0` in `ContactMaterial::rigid()`) with `dt` chosen independently is unconditionally unstable; the NaNs are the blowups arriving. |
| `crates/phyz-diff/src/rollout/step.rs:31-33` | The diff rollout has its **own, separate** contact model — per-vertex penalty, no friction at all, with an honest module comment saying friction was omitted because `‖v_t‖` is kinked exactly at the sticking state. So today phyz has two unrelated contact models, and the differentiable one has no friction. |

The last row is the most important. The simulation contact model and the
differentiable contact model are different code. That is the thing this design
eliminates: **one scalar-generic contact solver, instantiated at `f64` for simulation
and at dual/adjoint scalars for gradients.**

---

## 1. Solver choice

### 1.1 The three candidates

#### (a) Projected Gauss-Seidel LCP / NCP (Bullet, ODE, classic)

Formulate the discrete-time contact problem as a linear complementarity problem in
the normal impulses, with friction handled by a pyramidal linearization of the cone
and an outer iteration:

```
find λ_n ≥ 0  s.t.  (A λ + b)_n ≥ 0,  λ_n ⊙ (A λ + b)_n = 0
```

where `A = J M⁻¹ Jᵀ` is the Delassus operator and `b` the free-velocity residual.

**Robustness:** good. Well understood, decades of production use.

**Differentiability:** poor, and poor in a way that is not fixable by tuning. The
complementarity condition `λ ⊥ (Aλ + b)` is exactly a statement that the solution
lies on the boundary of an orthant, and the active set is a *combinatorial* object.
The map from parameters to solution is piecewise-smooth with kinks at every active-set
change. Worse, PGS is typically run to a fixed low iteration count and is
*not converged*, so the implicit function theorem does not apply to what you actually
computed — you would have to differentiate the iteration itself, which means unrolling
a sweep whose per-contact projection `λ ← max(0, λ - r·(Aλ+b))` injects a hard `max`
kink per contact per iteration. The gradient of a truncated PGS is a gradient of the
*algorithm*, not of the physics.

**Verdict: rejected.** The complementarity condition is the enemy.

#### (b) TGS-Soft (Rapier, Avian, PhysX)

Temporal Gauss-Seidel with soft constraints: substepped position-level solve with
per-substep relaxation, a soft (CFM/ERP-like) constraint law, and warm-starting.

**Robustness:** excellent — best-in-class for stacking and large mass ratios, which is
precisely why Rapier and Avian use it.

**Differentiability:** awkward. TGS derives its robustness from (i) substepping, (ii)
warm-starting from the previous frame's impulses, and (iii) a fixed small iteration
budget. All three are hostile to gradients:

- Substepping multiplies the unroll depth by the substep count.
- Warm-starting makes the step a function of *solver state carried across frames*,
  not just `(q, v, u)`. The map you want to differentiate is no longer the map the
  code implements; you would need to carry the impulse cache in the adjoint state.
- The fixed iteration budget again means you are differentiating a non-converged
  iterate, so IFT does not apply, and unrolling has the same per-iteration `max`
  problem as PGS.

Rapier chose this because Rapier does not need gradients. If phyz's goal were to match
Rapier's robustness, TGS-Soft would be the right answer. It is not our goal.

**Verdict: rejected as the primary solver.** Noted as a possible future
`Solver::Robust` backend for non-differentiable use (§7.3).

#### (c) MuJoCo-style convex soft contact — **selected**

MuJoCo drops the strict complementarity condition and replaces the contact problem
with a *convex optimization problem* whose solution is a soft contact force. This is
not an approximation MuJoCo tolerates for speed — it is the design decision that makes
MuJoCo's contact differentiable, and it is why MuJoCo contact is soft
([Todorov 2011][todorov2011], [Todorov 2014][todorov2014]). The contact force grows
with constraint violation rather than being an unbounded multiplier enforcing an
inequality exactly.

Concretely, per timestep we solve a strictly convex problem in the constraint-space
velocity or impulse:

```
minimize_λ   ½ λᵀ (A + R) λ + λᵀ b + Σ_c I_{K_c}(λ_c)
```

where
- `A = J M⁻¹ Jᵀ` — the Delassus / inverse inertia in constraint space,
- `R ≻ 0` — the *regularizer*, MuJoCo's constraint impedance (the generalization of
  CFM); this is what makes the problem strictly convex,
- `b` — the free velocity plus the reference-acceleration bias `a_ref` (MuJoCo's
  `solref` spring-damper that drives penetration to zero; the generalization of ERP),
- `K_c` — the second-order friction cone for contact `c`, and `I_K` its indicator.

**Why this is the right choice for phyz:**

1. **Strict convexity ⇒ unique solution ⇒ well-defined implicit derivative.** With
   `R ≻ 0` the objective is strongly convex, so the solution `λ*(θ)` is a *single-valued
   function* of parameters, and it is locally Lipschitz. There is exactly one thing to
   differentiate.
2. **The regularizer `R` is a knob that trades physical fidelity for gradient quality
   in a *principled* direction.** Large `R` ⇒ softer contact ⇒ smoother `λ*(θ)` ⇒
   better-conditioned gradients. Small `R` ⇒ crisper contact ⇒ sharper gradients. This
   is the same lever Dojo exposes as the interior-point central-path parameter κ
   ([Howell et al. 2022][dojo]), and it is the single most valuable feature of the
   design: **the user can dial gradient smoothness independently of the rest of the
   model.** §2.5.
3. **The friction cone is a genuine second-order cone**, so we get real Coulomb
   friction with stiction as the interior of the cone, not a viscous approximation.
   §4.
4. **It converges.** Unlike PGS/TGS we run to a residual tolerance, so IFT is applied
   to a point that actually satisfies the KKT conditions to `~1e-10`. This is the
   difference between a correct gradient and a gradient of an unconverged iterate.
5. **It is what MJX does**, so phyz's numerics are comparable to the reference
   differentiable simulator, and MJCF models (`crates/phyz-mjcf` already exists) map
   onto it with the same `solref`/`solimp` semantics
   ([MuJoCo computation docs][mjdoc]).

**The honest cost:** contact is *soft*. There is always some penetration; a resting
box sinks by `O(mg/k)`. Contact forces are not exact complementarity multipliers, so
the simulation is not a rigid-body simulation in the strict sense. Very stiff contact
(`R → 0`) recovers the rigid limit but degrades the solver conditioning and the
gradient quality simultaneously. We do not claim to dominate Rapier on stacking
robustness at equal timestep, and §6 includes benchmarks specifically designed to
measure the shortfall rather than hide it.

### 1.2 Concrete solver algorithm

We solve the convex problem with a **primal-dual interior-point method on the
second-order cone program**, following Dojo's structure ([Howell et al. 2022][dojo]),
rather than MuJoCo's Newton-with-pyramidal-cone. Rationale:

- The interior-point central-path parameter κ *is* the smoothing parameter, and it is
  a first-class quantity in the IFT derivation (§2). Getting smoothing for free from
  the solver is worth more than the constant factor MuJoCo's specialized Newton buys.
- SOC constraints are handled natively — no pyramidal linearization, so no
  direction-dependent friction artifacts (a box sliding at 45° does not get `√2` more
  friction than one sliding along an axis).
- The residual is a smooth function of the iterate everywhere on the central path, so
  the *linearization we already need for the Newton step* is exactly the matrix the
  IFT needs. The gradient is nearly free given the forward solve.

Fallback: for small problems (`n_contacts ≤ 4`) a direct SOCP is overkill; we keep a
specialized path. See §7.2 (perf).

---

## 2. How gradients flow

### 2.1 The two options, and the choice

**Unrolling** (differentiate every solver iteration) is what DiffTaichi-style systems
do ([Hu et al. 2020][difftaichi]). It is simple and always "works", but:
- memory is `O(iterations × contacts)` per step, and the trajectory adjoint already
  multiplies by trajectory length;
- the gradient is the derivative of the *algorithm*, so it depends on the iteration
  count and the initial guess — change the tolerance and the gradient changes;
- interior-point iterations near the boundary have enormous local derivatives that
  cancel only in exact arithmetic.

**Implicit differentiation** (differentiate the converged KKT solution via the
implicit function theorem) is the standard approach and is what Dojo uses. It is
`O(1)` in the iteration count, memory-free w.r.t. solver internals, and independent of
the initial guess.

**Decision: implicit function theorem on the converged relaxed-KKT system.** The
solver iterations are *not* on the tape. `phyz-diff`'s adjoint driver will receive an
explicit, closed-form contact-force Jacobian.

### 2.2 The derivation

Let `θ` collect everything the contact solve depends on: `(q, v, u, π)` — configuration,
velocity, controls, and model parameters (masses, geometry, `μ`, `solref`/`solimp`).
Let `z = (λ, s, y)` be the primal-dual iterate (impulse, slack, dual). The
central-path residual is

```
F_κ(z; θ) = 0
```

with `κ > 0` the central-path / smoothing parameter, and `F_κ` **smooth in both
arguments for κ > 0** (this is the whole point of the interior-point relaxation: the
complementarity condition `s ∘ y = 0` is replaced by `s ∘ y = κ e`, which is a smooth
equation).

At the converged `z*(θ)`, IFT gives

```
∂z*/∂θ = − (∂F_κ/∂z)⁻¹ · (∂F_κ/∂θ)
```

The key facts:

- `∂F_κ/∂z` is exactly the matrix the last Newton step already factorized. The
  factorization is cached from the forward solve, so a reverse-mode contraction
  `wᵀ ∂z*/∂θ` costs **one back-substitution**, not a new solve.
- `∂F_κ/∂θ` is a sparse, structured block that we write by hand and evaluate
  generically over `T: Scalar` (§3), so it is exact, not finite-differenced.

For the adjoint driver, we never form `∂z*/∂θ`. Given the incoming cotangent `χ` on
the contact impulses (the driver already prices an analogous wrench cotangent — see
`crates/phyz-diff/src/rollout/adjoint.rs:236-252`), we compute

```
ν  = −(∂F_κ/∂z)⁻ᵀ · χ         // one back-substitution with the cached factor
∂J/∂θ += νᵀ · (∂F_κ/∂θ)
```

This slots into the existing pattern in `adjoint.rs` cleanly: the contact block becomes
another channel alongside the inertia-parameter channel and the vertex channel, and the
per-step cost stays `O(nq + nv + 10·nb + 6·nb)` dual lanes plus one contact
back-substitution.

### 2.3 Non-smooth points — the honest part

The true contact dynamics map is **not** differentiable at:

| Event | What breaks |
|---|---|
| **Making/breaking contact** | The contact set changes. The true `∂(next state)/∂(state)` jumps. Even with soft contact, the gap function `d(q)` entering the constraint is only `C⁰` where the closest-feature pair changes (box corner → edge → face). |
| **Stick↔slip transition** | The friction force leaves the cone interior for its boundary. `‖f_t‖ = μ f_n` on the boundary and `f_t` is determined by the stick constraint in the interior; the map is `C⁰` but not `C¹` across the transition. |
| **Sliding direction reversal** | `f_t = −μ f_n · v_t/‖v_t‖` has a `‖·‖` kink at `v_t = 0`. This is precisely the kink `rollout/step.rs:33` cites as the reason friction was omitted from the diff rollout. |
| **Impact with restitution** | A discrete velocity jump. `∂v⁺/∂v⁻` involves `−e`, and the *time* of impact depends on the state, giving the classic time-of-impact discontinuity ([Hu et al. 2020][difftaichi]). |

What our design produces at these points, honestly stated:

**(a) For κ > 0 the computed gradient is smooth, and it is *not* the true gradient.**
It is the exact derivative of the *κ-relaxed* dynamics, which is a smooth surrogate for
the true dynamics. As `κ → 0` the surrogate converges to the true dynamics and the
gradient converges to a Clarke subgradient element where one exists — but at a
genuine discontinuity, no limit exists and the gradient becomes arbitrarily large as
`κ → 0` rather than converging. **This is a bias/variance tradeoff, not a bug, and we
document it as such.**

**(b) The bias is real and we name it.** [Suh, Simchowitz, Zhang & Tedrake
(2022)][suh2022] showed that first-order gradient estimators from differentiable
simulators are not automatically better than zeroth-order ones: near stiff or
discontinuous dynamics the first-order estimator has *low empirical variance but high
bias*, which is the worst combination because the optimizer confidently walks in the
wrong direction. Their diagnosis is that the exact gradient of a stiff dynamics is a
poor descent direction for the *smoothed* objective an RL/trajopt problem actually
cares about. The fix is to estimate the gradient of a smoothed objective directly.

**(c) Our mitigations, in order of preference:**

1. **Solver-level smoothing via κ (analytic, cheap, default).** Because we relax
   complementarity rather than clip a penalty, increasing κ smooths the *dynamics*, not
   just the gradient — forward and backward stay consistent. This is Dojo's approach
   and is the single most important design property. Exposed as
   `ContactSolverConfig::smoothing`.
2. **Randomized smoothing / bundled gradients (opt-in, expensive, correct).** For
   users who need an unbiased estimate of the smoothed objective's gradient, we expose
   a bundling wrapper: sample `N` perturbed rollouts, average the per-sample analytic
   gradients. This is exactly the *bundled gradient* of [Suh, Pang & Tedrake
   (2022)][bundled], and it is the theoretically correct object when the objective
   itself is stochastic. Cost is `N×`, so it is not the default.
3. **α-order interpolation (opt-in).** [Suh et al.][suh2022] propose blending the
   first-order (analytic) and zeroth-order (REINFORCE-style) estimators with a weight
   chosen from the empirical bias diagnostic. We expose the analytic gradient in a form
   that lets a caller do this; we do not implement the estimator itself, since it
   belongs at the optimizer level, not the simulator level.
4. **We do NOT implement time-of-impact correction in v1.** DiffTaichi's TOI trick
   ([Hu et al. 2020][difftaichi]) materially improves gradient quality for
   impact-dominated tasks, but it interacts badly with a converged implicit solve
   (the impact time becomes another implicit variable). Recorded as future work (§7.3).

**(d) What we will state in the public docs, verbatim in spirit:**

> phyz returns the exact gradient of the contact model it simulates. That model is a
> smooth relaxation of rigid contact controlled by `smoothing`. It is not the gradient
> of ideal rigid-body contact, because ideal rigid-body contact is not differentiable.
> If your objective is contact-mode-sensitive, expect bias, and either raise
> `smoothing` or use bundled gradients.

An engine that claims exact gradients through hard contact is either wrong or is
silently smoothing; we would rather say which.

### 2.4 Restitution and the gradient

Restitution enters as a modification of `b` (the target normal velocity becomes
`−e · v_n⁻` instead of `0`) rather than as a discrete post-solve velocity reset. This
matters for gradients: as a term in `b` it is differentiable w.r.t. `e` and w.r.t.
`v⁻`, and `∂b/∂θ` is a term we already need for the IFT. A post-solve reset would be a
branch on the primal, non-differentiable in `e` at `v_n⁻ = 0`, and would fight the
solver.

### 2.5 The smoothing knob

```rust
pub struct ContactSolverConfig {
    /// Central-path / relaxation parameter κ. Larger = softer contact,
    /// smoother and better-conditioned gradients, more penetration.
    /// Smaller = crisper physics, sharper and noisier gradients.
    /// Default 1e-4 (sim-fidelity biased); 1e-2 is a good gradient default.
    pub smoothing: f64,
    /// Residual tolerance for the forward solve. IFT validity assumes this
    /// is tight; gradients degrade gracefully but are not exact if loose.
    pub tolerance: f64,
    pub max_iterations: usize,
    pub cone: FrictionCone,   // Elliptic (default) | Pyramidal
}
```

We ship two presets: `ContactSolverConfig::simulation()` and
`ContactSolverConfig::gradients()`. Documenting that these differ — and that a user
optimizing through the simulator should probably use the latter for the optimization
and validate with the former — is more useful than pretending one setting serves both.

---

## 3. Generic-over-scalar design

The single hard requirement: **one code path, `f64` for simulation and dual/adjoint
scalars for gradients**, matching how `crates/phyz-diff/src/rollout/step.rs` is
already written (generic over `tang::Scalar`, with `T::select` for branchless
comparisons and `to_f64()` only for primal-side control flow).

### 3.1 Trait bounds

`tang::Scalar` (see `tang/crates/tang/src/scalar.rs:8`) already provides everything
the contact math needs: field ops, `sqrt`/`abs`/`min`/`max`/`clamp`/`recip`,
`from_f64`/`to_f64`, and crucially `select(cond, a, b)` — branchless select on the
primal sign. No new trait is required for the *math*.

What is new is the **linear solve**. The IFT contraction needs a factorization of
`∂F_κ/∂z`, and factorizing a dual-valued matrix is wasteful (we only ever need the
`f64` factor — see §3.3). So we split the API:

```rust
/// Everything in the contact model that is pure arithmetic — gap functions,
/// Jacobian rows, cone projection, the residual F_κ — is generic.
pub trait ContactScalar: tang::Scalar {}
impl<T: tang::Scalar> ContactScalar for T {}   // blanket; a named alias for clarity

/// The residual, generic. This is the function IFT differentiates.
pub fn contact_residual<T: ContactScalar>(
    problem: &ContactProblem<T>,   // A, b, cones, R — all in T
    z: &ConstraintVec<T>,          // (λ, s, y)
    kappa: T,
) -> ConstraintVec<T>;

/// Contact impulses for one step. Generic in T for the *evaluation*;
/// the inner Newton iteration is always driven on the f64 primal.
pub fn solve_contact<T: ContactScalar>(
    problem: &ContactProblem<T>,
    config: &ContactSolverConfig,
) -> ContactSolution<T>;
```

and the per-contact kernels, which are what `phyz-diff` will actually seed:

```rust
/// Gap function and its constraint-space Jacobian row block for one contact.
/// Generic so that seeding a geometry vertex or a body configuration gives an
/// exact tangent, exactly as `vertex_wrench` does today.
pub fn contact_jacobian<T: ContactScalar>(
    manifold: &ContactPoint<T>,
    xform_i: &SpatialTransform<T>,
    xform_j: &SpatialTransform<T>,
) -> ContactJacobianBlock<T>;   // 3 rows: [normal, tangent_u, tangent_w]

/// Second-order cone projection, smoothed by κ. Uses only Scalar ops:
/// `sqrt`, `max`, `select`. No branches on T.
pub fn project_cone<T: ContactScalar>(f: Vec3<T>, mu: T, kappa: T) -> Vec3<T>;
```

### 3.2 The branch discipline

Same rule the existing `step.rs` follows and which we make explicit and enforced:

- **Every comparison branches on `.to_f64()` of the primal.** Never on `T` ordering
  in a way that could differ between primal and tangent.
- **Every clamp/max/min goes through `T::select` or `T::max`**, whose tangent is the
  derivative of the branch the primal took. Documented consequence: at exactly the
  kink the tangent is one-sided; measure-zero, and the smoothing κ pushes the actual
  operating point off the kink.
- **No `is_finite` guards.** The whole NaN-plumbing layer in the current
  `solver.rs` gets deleted, not ported. A converged convex solve with `R ≻ 0` cannot
  produce a NaN from a finite input; if it does, that is a bug to fix at the source,
  not a value to filter. Removing this is a stated goal of the rewrite, and the
  NaN-freedom is asserted in tests rather than defended in the hot loop.

### 3.3 The factorization is `f64`-only, and that is correct

`∂F_κ/∂z` at the solution is needed only to *invert*. In forward mode we would need
its dual part; but the standard trick applies — for `A(θ) x(θ) = b(θ)`,

```
ẋ = A⁻¹ (ḃ − Ȧ x)
```

i.e. the derivative needs `A⁻¹` (the `f64` factor, already computed) applied to a
dual-valued right-hand side. So:

- the forward Newton solve factorizes an `f64` matrix once per step (cached),
- the tangent/adjoint pass reuses that factor and only the RHS carries dual parts.

This keeps the expensive linear algebra out of the generic code, and means the
gradient cost per contact step is one triangular solve. Concretely the generic surface
is: residual, Jacobian blocks, cone projection, manifold generation — all cheap
pointwise math — plus an `f64` `LDLᵀ` of the KKT matrix that both modes share.

### 3.4 Integration with `phyz-diff/src/rollout/`

`step.rs`'s per-vertex penalty contact (`vertex_wrench`, `contact_wrenches`,
`GroundContact`) is superseded. The new `step_generic` calls the shared solver:

```rust
pub(crate) fn step_generic<T: Scalar>(
    model: &Model,
    inertias: &[SpatialInertia<T>],
    contact: Option<&ContactSetup<T>>,   // was (&GroundContact, &[CollisionMesh])
    ext: Option<&[SpatialVec<T>]>,
    q: &[T], v: &[T], ctrl: &[T],
) -> (Vec<T>, Vec<T>, Vec<T>)
```

The adjoint driver in `adjoint.rs` keeps its structure. Its vertex channel
(`adjoint.rs:230-291`) already prices a per-body 6-vector wrench cotangent `χ_b` and
contracts local per-vertex Jacobians against it — **that pattern survives verbatim**,
with `vertex_wrench` replaced by the contact-solution wrench and the local Jacobian
now coming from the IFT block rather than from three dual evaluations of a penalty
law. This is the strongest argument that the design fits: the existing adjoint
architecture was built around exactly this contraction.

One genuine change: with a *coupled* solve, vertices no longer contribute
independently — contact `c` on body `b` depends on all other contacts through `A`. The
per-vertex local-Jacobian shortcut becomes a per-contact block that must go through
the `(∂F/∂z)⁻ᵀ` back-substitution. Cost per step goes from `O(3N)` trivial evaluations
to one back-substitution of size `3·n_contacts` — still cheap, but no longer
embarrassingly local. This must be called out in `adjoint.rs`'s module docs when it
lands.

---

## 4. Contact physics

### 4.1 Coulomb friction with a real cone

Per contact, the impulse `f = (f_n, f_u, f_w)` is constrained to the second-order cone

```
K_μ = { f : ‖(f_u, f_w)‖ ≤ μ f_n,  f_n ≥ 0 }
```

- **Stiction** is `f` in the *interior*: the solver finds the impulse that drives
  relative tangential velocity to zero, and that impulse is feasible. This is the
  behavior the current model cannot produce at all.
- **Sliding** is `f` on the *boundary*: `‖f_t‖ = μ f_n`, anti-parallel to slip. This
  falls out of the maximum-dissipation structure of the convex problem; we do not
  special-case it.
- **The transition is handled by the solver**, not by a velocity threshold.

Elliptic (true SOC) cone is the default. A pyramidal option is retained for
MJCF-compatibility and because pyramidal turns the SOCP into a QP, which is faster;
its documented cost is direction-dependent friction (up to `√2` anisotropy in 3D)
([MuJoCo docs on `impratio`/cone types][mjdoc]).

### 4.2 Friction anchors (position-level stiction)

A velocity-level cone alone gives *zero drift rate*, not *zero drift*: a block resting
on an incline below the friction angle has `v_t → 0` but accumulates a small position
error each step, so it creeps at `O(dt)` per step. The fix is a **friction anchor**: on
first stick, record the contact-point pair in each body's frame; while the contact
stays stuck, the tangential constraint targets *zero relative displacement from the
anchor*, not zero relative velocity. On slip, the anchor is dragged to the current
point.

Gradient consequence, stated plainly: **the anchor is persistent state**, so the step
map becomes `x_{t+1}, a_{t+1} = f(x_t, a_t, u_t)`. The adjoint must carry an anchor
cotangent. This is a real complication — it is the same complication that makes TGS
warm-starting gradient-hostile (§1.1b). We accept it here because (i) the anchor is a
small, explicit, differentiable quantity (a 3-vector per stuck contact) rather than an
opaque impulse cache, (ii) it only affects stuck contacts, whose Jacobian is otherwise
degenerate, and (iii) without it the inclined-plane benchmark in §6 cannot pass. The
anchor-drag event on slip is one more non-smooth point, covered by §2.3.

Anchors are optional (`ContactSolverConfig::anchors: bool`, default on for simulation,
**off** for the gradient preset) so that users who need a stateless step map can have
one at the cost of creep.

### 4.3 Restitution that works

`bounce` (renamed `restitution`, see §5) becomes a target normal velocity in `b`:

```
b_n = −min(0, v_n⁻) · e        // Newtonian restitution, applied to the approach speed
```

with a **restitution velocity threshold** `v_rest` below which `e` is smoothly ramped
to zero — without it, a resting body micro-bounces forever and the stack never settles.
The ramp is `smoothstep`, not a hard cutoff, so it is `C¹`:

```
e_eff = e · smoothstep(v_rest, 2·v_rest, |v_n⁻|)
```

This makes restitution differentiable in both `e` and `v_n⁻` everywhere, which the
naive `if |v_n| < threshold { e = 0 }` would not be.

Validation: drop height → bounce height ratio must match `e²` (§6.2).

### 4.4 Multi-point manifolds

Single-point-per-pair is why resting boxes are unstable today. We need:

1. **Real narrow-phase output.** Replace the center-difference normal
   (`solver.rs:102-126`) with the EPA normal as the *primary* path (it is already
   available as `phyz_collision::epa_penetration_rot`), and surface-projected contact
   points, not the midpoint of body centers.
2. **Manifold generation.** For each colliding pair, produce up to 4 contact points
   via face-clipping (Sutherland–Hodgman clipping of the incident face against the
   reference face's side planes) for box/box and polytope pairs; for
   sphere/capsule/curved pairs, 1–2 analytic points. Reduce to at most 4 by maximizing
   contact-patch area, the standard reduction.
3. **Manifold persistence across frames** keyed by feature IDs, so friction anchors
   have somewhere to live and so contact ordering (which affects solver conditioning)
   is stable.

Differentiability note, honestly: **manifold generation is combinatorial and is *not*
on the differentiable path.** Feature selection (which face is the reference face,
which clip vertices survive) is treated as a *discrete decision held fixed for the
step*. The gap function and Jacobian of each surviving contact point *are*
differentiable in `q` and in the geometry. This is the same structure MuJoCo and Dojo
use and it is a genuine approximation: the gradient does not see "the manifold would
have had a different point if the box rotated slightly more." Recent work on smoothly
differentiable manifold construction exists ([Chen et al. 2026][manifold]) and is
noted as future work.

### 4.5 Reference model and parameters

`solref`-style spring-damper for penetration recovery and `solimp`-style impedance for
the regularizer `R`, matching MuJoCo semantics so `phyz-mjcf` can import parameters
without translation ([MuJoCo computation docs][mjdoc]):

```rust
pub struct ContactMaterial {
    pub friction: f64,          // μ, tangential
    pub restitution: f64,       // e ∈ [0, 1]
    pub sol_ref: [f64; 2],      // (time_const, damp_ratio) — the a_ref spring-damper
    pub sol_imp: [f64; 5],      // (d0, d_width, width, midpoint, power) — impedance
    pub restitution_threshold: f64,
}
```

The old `stiffness`/`damping` are derivable from `sol_ref` and get deprecated
constructors that map onto it (§5).

---

## 5. Migration plan

### 5.1 What breaks

**`crates/phyz-contact` public API — breaking, all of it.**

| Item | Fate |
|---|---|
| `compute_contact_force` | **Removed.** Per-contact force is no longer a local function of one contact; contacts are coupled through `A`. No compatible signature exists. |
| `compute_contact_force_implicit` | **Removed.** Its one-step Newton with effective mass is a degenerate special case of the real solve. |
| `contact_forces`, `contact_forces_implicit` | **Replaced** by `solve_contacts(&ContactProblem, &ContactSolverConfig) -> ContactSolution` plus `ContactSolution::body_wrenches()`. A shim with the old signature is possible for one release and is *not* recommended — it would have to build and solve a problem per call, which is the wrong shape. |
| `find_contacts` | **Signature preserved, semantics fixed** — returns a manifold (multiple `Collision` per pair) with EPA normals and surface contact points. Callers that assumed one contact per pair break behaviorally, not at compile time. This is the most dangerous change and is why it gets its own staged step (§5.3, stage 3). |
| `find_ground_contacts` | **Preserved**, extended to emit multi-point manifolds for boxes (4 corners) rather than one point at the body center's `(x, y)`. |
| `ContactMaterial::{stiffness, damping}` | **Deprecated fields**, kept for one release with a documented `sol_ref` mapping. `ContactMaterial::new(k, c, μ, e)` keeps working. |
| `ContactMaterial::{bounce, soft_cfm, soft_erp}` | `bounce` → renamed `restitution` (with a deprecated alias) and **now actually implemented**. `soft_cfm`/`soft_erp` → **removed**, superseded by `sol_ref`/`sol_imp`; they were never read, so removal changes no behavior. |
| `phyz_diff::rollout::step::{GroundContact, CollisionMesh, vertex_wrench}` | `GroundContact` **removed** (superseded by a plane collider in the shared model). `CollisionMesh` **retained** — it is the vertex-gradient channel's input and vcad's integration point. `vertex_wrench` **removed** (was `pub(crate)`). |
| `phyz_diff::rollout::adjoint::{ContactSetup, AdjointRollout}` | `ContactSetup` changes shape (`ground: GroundContact` → `world: ContactWorld`). `AdjointGradients` is **unchanged** — same `d_inertia`, same `d_vertices`. Downstream vcad consumers see no change in the gradient contract. |

### 5.2 The vendored copy

`crates/phyz/src/contact/` is a **hand-vendored duplicate** of `crates/phyz-contact`,
differing only in import paths (verified: the only diff is `crate::collision` vs
`phyz_collision` etc.). `phyz-contact` is depended on by *no other crate in the
workspace* — only `crates/phyz/src/lib.rs:4,15` re-exports the vendored copy, and only
`crates/phyz/tests/{integration,contact_stability}.rs` exercise it.

This is good news: the blast radius is small and entirely inside `phyz` + `phyz-diff`.

**Resolution: delete the vendored copy; `phyz` depends on `phyz-contact` and
re-exports.** The duplication exists for no reason we can find (there is precedent for
de-duplication in this tree — see `9d6373c refactor(math): dedupe math module — phyz
re-exports phyz-math`), and maintaining two copies of a solver this intricate is not
viable. `crates/phyz/src/contact/mod.rs` becomes:

```rust
pub use phyz_contact::*;
```

matching the `phyz-math` precedent exactly.

### 5.3 Staged sequence — every stage leaves the tree green

Each stage is a mergeable PR; `cargo test --workspace` passes at every one.

**Stage 0 — dedupe (no behavior change).**
Delete `crates/phyz/src/contact/{mod,material,solver}.rs`; add `phyz-contact` to
`crates/phyz/Cargo.toml`; `crates/phyz/src/contact.rs` becomes a re-export. Existing
tests pass unmodified. This is the `phyz-math` refactor applied again, and it is a
prerequisite for everything else — otherwise every later stage is done twice.

**Stage 1 — manifold-quality narrow phase, penalty solver unchanged.**
Fix `find_contacts`: EPA normal as the primary path, real surface contact points,
multi-point manifolds, feature-ID persistence. Keep the existing penalty force law.
Tests: new manifold unit tests (a box on a plane yields 4 coplanar points with the
plane normal); `contact_stability.rs` should *improve* here, and any test that
implicitly depended on one-contact-per-pair is updated in this PR with a note.
This stage alone fixes the worst simulation bug and is independently valuable.

**Stage 2 — the convex solver, `f64` only, behind a feature/config flag.**
Land `ContactProblem`, the SOCP interior-point solve, the elliptic cone, `sol_ref` /
`sol_imp`, restitution, friction anchors. `ContactSolverConfig::penalty_legacy()`
selects the old path so nothing regresses while the new path is validated. Analytic
benchmarks from §6.1–6.3 land with it and must pass on the new path.

**Stage 3 — make the convex solver the default; delete the penalty path.**
Flip the default, remove `compute_contact_force{,_implicit}`,
`contact_forces{,_implicit}`, and the NaN-plumbing. Remove `soft_cfm`/`soft_erp`.
Bump the minor version; changelog entry enumerates §5.1.

**Stage 4 — generic-over-scalar.**
Make the residual, Jacobian blocks, cone projection generic over `T: Scalar`. `f64`
behavior must be bit-identical to stage 3 (assert this with a golden-trajectory test
before and after — the `f64` instantiation is the same arithmetic in the same order).

**Stage 5 — IFT gradients + `phyz-diff` integration.**
Add `∂F_κ/∂z` factor caching, the `∂F_κ/∂θ` blocks, and the back-substitution
contraction. Rewire `rollout/step.rs` to call the shared solver and `rollout/adjoint.rs`
to route the contact channel through the IFT block. `phyz-diff`'s existing
`rollout_adjoint.rs` FD tests must still pass (they will need retuned tolerances, since
the contact model underneath changed — this is expected and is documented in the PR,
not silently absorbed).

**Stage 6 — smoothing controls and bundled gradients.**
`ContactSolverConfig::{simulation, gradients}` presets; the bundling wrapper; the
gradient-quality documentation of §2.3(d).

Stages 0 and 1 are worth landing regardless of whether the rest of the plan survives
review.

---

## 6. Test and validation plan

Analytic benchmarks first — they are the only thing that distinguishes "the solver
converged" from "the solver converged to the right physics." Each has a closed-form
answer.

### 6.1 Block on an inclined plane at the friction angle

The sharpest single test of a Coulomb model, and the one the current code cannot pass
at any angle.

- Plane at angle `α`, block with `μ`. Theory: the block is static iff
  `tan α ≤ μ`; when sliding, `a = g(sin α − μ cos α)`.
- **Test A (stiction):** `μ = 0.5`, sweep `α` over `[0°, 40°]`. Assert
  displacement after 5 s is `< 1 mm` for all `α < atan(0.5) ≈ 26.57°`. The current
  model fails this at *every* angle.
- **Test B (the angle itself):** binary-search the transition angle; assert it is
  within `0.5°` of `atan(μ)`. This is the test that catches a cone that is subtly
  pyramidal or a `μ` that is scaled wrong.
- **Test C (sliding acceleration):** `α = 40°`, assert measured `a` matches
  `g(sin α − μ cos α)` to `1%`.
- **Test D (direction isotropy):** slide the block down the same incline oriented at
  `0°, 15°, 30°, 45°` in the tangent plane; assert accelerations agree to `0.1%`. This
  is the elliptic-vs-pyramidal test.

### 6.2 Restitution from drop height

- Drop a sphere from `h₀`, measure apex `h₁`. Theory: `h₁/h₀ = e²`.
- Assert for `e ∈ {0.0, 0.3, 0.5, 0.8, 0.95}` to `2%` (allowing for the soft-contact
  energy loss, which is a real and documented effect — the test tolerance encodes the
  approximation rather than hiding it).
- **Settling test:** with `e = 0.8`, assert the sphere is at rest (`|v| < 1e-3`) within
  10 s and stays there for another 5 s. This is what the restitution threshold ramp
  (§4.3) exists to guarantee, and it is where naive restitution implementations fail.

### 6.3 Stable box stacking

- 5 unit boxes stacked, `μ = 0.6`, released at rest in exact contact.
- Assert after 10 s: max lateral drift `< 1 mm`, max tilt `< 0.5°`, max penetration
  `< 2 mm` (the soft-contact sink — asserted as a *bound*, and separately asserted to
  match the analytic `Σmg/k` prediction to `20%`, so we are testing that we understand
  the softness, not just that it is small).
- Assert no NaN anywhere in the trajectory — as an assertion, replacing the defensive
  filters being deleted.
- Second variant with a 100:1 mass ratio (heavy box on light boxes), which is where
  soft contact is genuinely worse than TGS. **We assert a degraded but bounded result
  and document the number rather than tuning until it looks good.**

### 6.4 Energy behavior

- Frictionless, `e = 1`, sphere bouncing: assert total energy drift over 100 bounces
  is monotonically non-increasing and bounded (soft contact dissipates; it must not
  *gain* energy, which is the failure mode of a badly-signed damping term).

### 6.5 Finite-difference validation of contact gradients

Following the established pattern in `crates/phyz-regge` (e.g.
`geometry.rs:496 test_dihedral_angle_finite_diff_consistency`) and
`crates/phyz-quantum/src/jacobson.rs`: central differences against the analytic
gradient, with a Richardson-style step sweep to confirm the error is
truncation-dominated rather than a wrong formula.

For each scenario below: compute `dJ/dθ` analytically via
`adjoint_rollout_gradient`, and by central difference
`(J(θ+h) − J(θ−h)) / 2h` with `h` swept over `{1e-3, 1e-4, 1e-5, 1e-6}`.
Assert relative error `< 1e-4` at the best `h`, and assert the error *decreases* as
`h` shrinks from `1e-3` to `1e-5` (the signature of a correct gradient; a wrong
formula plateaus).

| Scenario | `θ` | Why it matters |
|---|---|---|
| Sphere resting on plane | body mass, plane height, `μ`, `sol_ref` | Baseline: a single always-active contact, fully smooth. Must be near machine-precision-clean. |
| Block sliding on plane, well into slip | `μ`, initial velocity | Tests the cone-boundary branch of the IFT. |
| Block sticking on a shallow incline | `μ`, `α` | Tests the cone-interior branch **and** the friction anchor's contribution to the adjoint. |
| Two-box stack | upper-box mass | Tests contact coupling through `A` — the term a per-contact-local gradient would miss entirely. This test fails for any design that treats contacts independently, which is why it is in the list. |
| Bouncing sphere, `e = 0.5`, gradient w.r.t. `e` | `e` | Restitution-as-`b`-term is differentiable; restitution-as-post-solve-reset is not. This test discriminates. |
| Collision-mesh vertex channel | a box corner vertex | Validates `d_vertices` survives the rewrite — the vcad integration contract. |

**Explicit negative tests** (asserting the *known* limits, so they are documented and
regressions are visible):

- **Contact-making discontinuity:** a sphere dropped so that it first touches the
  ground exactly at the last step of the rollout. Assert FD and analytic gradients
  **disagree** by a documented margin, and assert that the disagreement *shrinks* as
  `smoothing` increases. This is the test that turns §2.3's honesty into something
  mechanically checked.
- **Stick-slip transition:** an incline at exactly `atan(μ)`. Same structure.

These two tests are the ones a competitor's benchmark would use against us. Better to
own the numbers.

### 6.6 Cross-validation against MuJoCo

Where `phyz-mjcf` can load a model, run the same MJCF in phyz and (offline, not in CI)
in MuJoCo, and compare trajectories for the §6.1–6.3 scenarios. Not a CI gate — a
periodic manual check recorded in this doc's appendix. It is the only external
ground truth available for the soft-contact regime, since the analytic benchmarks
above only pin down the rigid limit.

---

## 7. Open questions and future work

### 7.1 Deliberately out of scope for v1
- Multi-DOF joints (spherical, free) in the diff path — `step.rs` already restricts to
  single-DOF and panics otherwise; contact does not change that, but it is the largest
  practical limitation and should be the next project after this one.
- Continuous collision detection / tunneling.
- Soft bodies, `phyz-md` / `phyz-lbm` coupling through contact.

### 7.2 Performance, unmeasured
The interior-point solve is more expensive per step than the current penalty
evaluation — probably `5–20×` for small contact counts. We have no measurements and
will not claim otherwise until stage 2 lands with benchmarks. Mitigations if it is a
problem: warm-start `z` from the previous step (which reintroduces the state-carrying
problem of §4.2, and must be off in the gradient preset), and a specialized
closed-form path for `n_contacts ≤ 2`. **`cargo bench` numbers before/after are a
required deliverable of stage 2**, not an afterthought.

### 7.3 Known future improvements
- **Time-of-impact correction** ([Hu et al. 2020][difftaichi]) for impact-dominated
  gradients (§2.3d).
- **Smoothly differentiable manifold construction** ([Chen et al. 2026][manifold]) to
  remove the discrete feature-selection approximation of §4.4.
- **A `Solver::Tgs` backend** for users who want Rapier-class robustness and do not
  need gradients. This is a real use case and the architecture (a `ContactProblem` +
  pluggable solve) admits it cleanly. It would be honest to offer it rather than to
  insist convex soft contact is best for everyone.

### 7.4 The tradeoff, summarized without spin

| | phyz (this design) | Rapier / Avian | MuJoCo / MJX | Dojo |
|---|---|---|---|---|
| Gradients | yes, IFT, exact for the relaxed model | none | yes, in JAX only | yes, IFT |
| Language | Rust, embeddable, WASM | Rust | C, but gradients need Python/JAX | Julia |
| Stacking robustness | good, worse than TGS at high mass ratio | best-in-class | good | good |
| Contact hardness | soft, tunable | hard-ish (TGS) | soft | tunable to hard |
| Gradient bias at contact events | present, documented, tunable via κ | n/a | present | present, tunable via κ |

The row phyz wins is *gradients + embeddable compiled language*. We should not claim
the stacking row, and this design does not try to.

---

## References

- [todorov2011]: E. Todorov, "A convex, smooth and invertible contact model for
  trajectory optimization," ICRA 2011. — the original convex relaxation.
- [todorov2014]: E. Todorov, "Convex and analytically-invertible dynamics with contacts
  and constraints: Theory and implementation in MuJoCo," ICRA 2014.
  <https://www.researchgate.net/publication/288485003>
- [mjdoc]: MuJoCo documentation, *Computation* (constraint model, `solref`, `solimp`,
  elliptic vs pyramidal cones, `impratio`).
  <https://mujoco.readthedocs.io/en/stable/computation/index.html>
- [dojo]: T. Howell, S. Le Cleac'h, J. Brüdigam, J. Z. Kolter, M. Schwager, Z. Manchester,
  "Dojo: A Differentiable Physics Engine for Robotics," 2022.
  <https://arxiv.org/abs/2203.00806> — NCP with second-order cones, primal-dual
  interior point, IFT gradients, central-path parameter as a smoothing knob.
- [suh2022]: H. J. T. Suh, M. Simchowitz, K. Zhang, R. Tedrake, "Do Differentiable
  Simulators Give Better Policy Gradients?", ICML 2022.
  <https://arxiv.org/abs/2202.00817> — bias/variance of first-order estimators,
  empirical bias, α-order estimator.
- [bundled]: H. J. T. Suh, T. Pang, R. Tedrake, "Bundled Gradients through Contact via
  Randomized Smoothing," 2021. <https://arxiv.org/pdf/2109.05143>
- [difftaichi]: Y. Hu et al., "DiffTaichi: Differentiable Programming for Physical
  Simulation," ICLR 2020. <https://arxiv.org/abs/1910.00935> — time-of-impact
  correction for contact gradients.
- Y. D. Zhong, J. Han, G. O. Brikis, "Differentiable Physics Simulations with Contacts:
  Do They Have Correct Gradients w.r.t. Position, Velocity and Control?", 2022.
  <https://arxiv.org/pdf/2207.05060>
- C. D. Freeman et al., "Brax — A Differentiable Physics Engine for Large Scale Rigid
  Body Simulation," 2021. <https://arxiv.org/pdf/2106.13281> — spring/positional/
  generalized backends; see also the community reports that the generalized backend's
  contact gradients are of limited use for training
  (<https://github.com/google/brax/discussions/529>).
- Q. Le Lidec et al., "Augmenting Differentiable Physics with Randomized Smoothing,"
  2022. <https://arxiv.org/pdf/2206.11884>
- [manifold]: "Novel Algorithms for Smoothly Differentiable and Efficiently
  Vectorizable Contact Manifold Construction," 2026.
  <https://arxiv.org/html/2604.17538>
- "A Review of Differentiable Simulators," 2024. <https://arxiv.org/pdf/2407.05560>


---

## 8. Implementation status

Added 2026-08-16. The header said "design (not implemented)" for months after most
of this shipped, which is worse than no status line at all: it sent at least one
reader off to build a solver that already existed. What follows is the state of
the tree, and it should be updated in the same PR as any change below.

Delivered across #28 (narrow phase, convex solver, IFT gradients), #37
(stabilization, pair materials, warm starting), #39/#41 (redundant-manifold
convergence), #42 (contact margin), #48 (trajectory adjoint), and this PR
(body-body adjoint).

### 8.1 Landed as designed

- §1.1(c) convex soft contact, chosen over LCP/PGS and TGS — `phyz-contact/src/convex.rs`.
- §2.1–2.2 gradients by the implicit function theorem on the converged solution,
  not by unrolling — `phyz-contact/src/gradient.rs`, `FixedPointSensitivity`.
- §2.4 restitution as a term in `b`, never a post-solve velocity reset.
- §4.1 a real second-order friction cone with genuine stiction. The isotropy
  test (§6.1 D) holds to `1e-9`, well inside the `0.1%` the plan asked for.
- §4.3 restitution with the `smoothstep` low-speed ramp.
- §4.4 multi-point manifolds, EPA normals, surface contact points, persistence.
- §4.5 MuJoCo `solref`/`solimp` semantics — `phyz-contact/src/material.rs`.
- §5.2 / stage 0 the vendored `crates/phyz/src/contact/` is gone; `phyz`
  re-exports `phyz-contact`.
- §6.1 the full block-on-incline battery A–D.

### 8.2 Landed differently, on purpose

- **§1.2 the solver is not a primal-dual interior-point SOCP.** It is an
  alternating PGS / active-set Newton. Consequence: there is no central-path
  parameter κ, and `ContactSolverConfig::regularization` plays the smoothing
  role instead. The `simulation()` / `gradients()` presets of §2.5 exist and
  mean what §2.5 says they mean; `cone: FrictionCone` and `anchors: bool` do not
  exist, the cone being elliptic always.
- **§3 generic-over-scalar did not happen, and the goal it served was met
  another way.** `phyz-contact` is entirely `f64`; there is no `ContactScalar`,
  no `tang` dependency. §0.1 wanted one thing from genericity — that the
  simulated and differentiated contact models cannot drift apart — and
  `phyz-diff/src/contact_adjoint.rs` secures it directly instead: its forward
  pass *is* `Simulator::step_with_contacts`, operation for operation, asserted
  bit-identical by `phyz/tests/diff_convex_contact.rs`. The derivative is then
  analytic (IFT) through the solve and central-difference per lane through the
  smooth blocks around it (ABA, FK, assembly, Φ).

  This is a real trade, not a free substitution. It costs exactness in the
  smooth blocks (`~1e-9` relative, against machine precision for a dual number)
  and it costs speed — the measured gradient is `33.7x` one forward rollout on a
  200-step box drop. Making the solver generic remains the right end state and
  is the largest single open item; it is not, however, load-bearing for
  correctness the way §3 implies, because the drift it was meant to prevent is
  already prevented.
- **§5.1 `GroundContact` / `vertex_wrench` were not removed.** The per-vertex
  penalty model survives in `phyz-diff/src/rollout/` as the `d_vertices`
  surface-gradient channel vcad integrates against, which the convex path does
  not reproduce. So two contact models do still coexist — but their roles are
  now disjoint and documented, rather than being two answers to the same
  question. State/control/inertia gradients all route through the convex path.
- **§5.1 the deprecated penalty API is still shipped** (`compute_contact_force`,
  `contact_forces`), marked `#[deprecated]` rather than deleted. Stage 3 is
  therefore only partly done.

### 8.3 Measured shortfalls against §6

Implementing §6.1–§6.5 as *trajectories* rather than as single solves turned up
three places where the shipped engine does not meet the spec this document
wrote. All three are invisible to `phyz-contact`'s `analytic_benchmarks.rs`,
which exercises hand-built single-contact `ContactProblem`s. Each is now pinned
by a regression guard whose doc comment states plainly that it guards a measured
number rather than checking physics.

| What | Spec | Measured | Issue |
|---|---|---|---|
| §6.1 C sliding acceleration, box on a 40° slope | within 1% | **16% excess** (`a = 2.0838` vs `1.7968`; effective `mu` `0.5618` vs `0.600`) | [#63] |
| §6.2 restitution, dropped sphere | `h1/h0 = e²` within 2% | **81% of nominal `e` from 20 cm, 92% from 80 cm** (8–19% energy shortfall); no measurable rebound at all from 5 cm at any `e` | [#64] |
| §6.3 stacking at high mass ratio | degraded but bounded | **no bound exists** — tilt after settling is 0.01° / 0.00° / 0.89° / **40.85°** / 0.00° / **180.36°** / **190.40°** at ratios 1 / 2 / 5 / 10 / 20 / 50 / 100 | [#65] |

The friction one is the most surprising, because three natural explanations are
ruled out by measurement: it is not the solver preset, not the impedance
regularizer (sweeping `solimp` over `0.9`…`0.9999` changes nothing), and not the
box rotating (final pitch `4.2e-4 rad`). It is something the multi-point path
does that the single-contact benchmark cannot see.

The stacking one has a consequence for what this document claims. §7.4's table
gives phyz *"stacking robustness: good, worse than TGS at high mass ratio"*.
That reads as graceful degradation. A 20:1 stack standing perfectly while a 10:1
stack falls flat is not degradation, it is an instability with a non-monotone
onset, and **that row should not be published in its current form.** The
equal-mass case genuinely is good — five boxes drift 5.8 µm and tilt 6.8e-5 rad
over 10 s — and that is what the row should say.

What *does* meet spec, measured the same way: §6.1 A/B/D (stiction, the
transition angle, isotropy to `1e-9`), §6.2's settling test (a bouncy sphere
comes to rest inside 10 s and moves 0.000 m over the next 5 s), §6.3's
equal-mass stack, and §6.4's energy bound (30 s of `e = 1` bouncing never
exceeds the starting energy).

### 8.4 Gradient validation, as it now stands

Rollout-level FD gates, worst relative error per scenario:

| Scenario | Worst lane |
|---|---|
| Block on an incline, sticking (20°, `mu = 0.6`) | `9.7e-9` |
| Block on an incline, sliding (40°, `mu = 0.6`) | `3.0e-7` |
| Box tipping on an edge (edge→face manifold change) | `1.5e-6` |
| Block carried by friction on a driven plank (body-body) | `9.8e-4` |
| Block sliding on a plank (body-body) | `4.3e-4` |
| `dJ/dmu`, sliding box | `8.2e-8` |
| `dJ/de`, bouncing sphere | `1.0e-7` |
| Flat-ground box: impact / settled / slide / driven slide | `1e-3` gate |

Note the pattern: the material channels and the single-body ground scenarios are
four to five orders tighter than the body-body ones. That is the FD lanes of
§8.2 showing up — `dJ/dmu` is analytic end to end, while a body-body lane
accumulates central-difference error through assembly on both bodies. It is the
clearest available argument for finishing §3.

Two limits are pinned as tests rather than described:

- The slip↔stick transition on a redundant eight-contact manifold stalls the
  active-set Newton at `~1e-7` and the adjoint returns `Unconverged` rather than
  differentiating a non-KKT point.
- On an *exactly* symmetric manifold, the symmetry-breaking lane reports `0`
  while the one-sided derivatives are `-5.457e-3` and `-1.391e-2` — so the
  returned value is not merely the wrong branch, it is outside the Clarke
  interval. Measure-zero, and 1 mrad off symmetry the lane agrees, but
  hand-built initial conditions are frequently exactly symmetric.

[#63]: https://github.com/ecto/phyz/issues/63
[#64]: https://github.com/ecto/phyz/issues/64
[#65]: https://github.com/ecto/phyz/issues/65

### 8.5 Open

Roughly in descending order of what a caller would actually notice.

- §3 the generic-over-scalar solver, per §8.2.
- The three §6 shortfalls of §8.3 — issues [#63], [#64], [#65]. #63 and #64 are
  correctness bugs in the shipped forward model, which makes them higher
  priority than anything else on this list.
- `dJ/dsol_ref` is still unplumbed. `dJ/dmu` and `dJ/de` now reach the rollout;
  `depth_sensitivity` exists at solver level and the stabilization parameters do
  not.
- §6.5 the contact-making discontinuity negative test exists only at solver
  level (`depth_gradient_has_a_documented_hinge_at_zero_depth`), not as a
  rollout. The body-body analogue now does exist —
  `phyz-diff/tests/body_body_adjoint.rs::exact_symmetry_gives_a_lane_outside_the_clarke_set`.
- §6.3's analytic-sink check (that the penetration matches `Σmg/k` to 20%) is
  still not asserted; only the bound is. The measured sink is 342 µm for five
  unit boxes.
- §6.6 no recorded MuJoCo trajectory comparison, though `mujoco_compat` and the
  creep-rate test exist to make one meaningful. Issues #63 and #65 are both
  cases where an external oracle would settle the question quickly, so this has
  become more valuable than it looked.
- Stage 6 randomized smoothing / bundled gradients: not implemented anywhere.
- §4.2 position-level friction anchors: not implemented. `convex.rs` argues the
  solref bias removes the creep they were meant to fix, and the eight-box stack
  holds to `1e-9` m over 3 s, so this may be a design item to retire rather than
  build.
