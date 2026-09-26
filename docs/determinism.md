# Determinism: what is guaranteed reproducible and what is not

This is the document to cite when you publish a number produced by phyz. It
states exactly what the engine promises about reproducing a rollout, what it
does not promise, and how to tell the two apart when a result does not
reproduce.

The short version:

| Question | Answer |
|---|---|
| Same rollout, same process, twice? | **Bit-identical.** Gated in CI. |
| Same rollout, two threads? | **Bit-identical.** Gated in CI. |
| Same rollout, debug vs release? | **Bit-identical.** Gated in CI. |
| Same rollout, x86-64 vs aarch64? | **Bit-identical.** Gated in CI on Linux x86-64, Linux aarch64 and macOS aarch64. |
| Same rollout, different phyz version? | **No.** A physics change moves the bits, on purpose. Pin the version. |
| Same rollout, one input moved by 1 ulp? | **No, and the divergence is large.** See [Chaos](#chaos-and-how-to-recognise-it). |
| Same rollout, GPU backend (`phyz-gpu`)? | **No.** Out of scope; see [Not covered](#not-covered). |

---

## The guarantee

> Given a `Model`, an initial `State`, a timestep, and a `Simulator` in a known
> initial state, the sequence of states produced by `step`,
> `step_with_contacts` and `step_with_contacts_heightfield` is a **pure
> function of those inputs** and is identical, bit for bit, on every supported
> target and at every optimization level, for a fixed version of phyz.

"Known initial state" means either a freshly constructed `Simulator`, one you
have called `reset_contact_cache()` on, or one built with
`with_warm_start(false)`. See [The warm-start cache](#the-warm-start-cache).

### How it is gated

`crates/phyz/tests/determinism.rs` holds golden 64-bit fingerprints of three
contact-rich rollouts — a box landing on a corner and tipping, a cylinder
rolling, and a free-based chain collapsing onto the ground. The fingerprint is
FNV-1a over the IEEE bit patterns of `q`, `v` and `time` after every step, so a
single ulp anywhere in 1500 steps changes it.

The same file also gates:

- `a_rollout_replays_bit_for_bit` — the same rollout twice in one process.
- `concurrent_rollouts_agree_with_the_serial_one` — eight threads stepping the
  same scene simultaneously must agree with the serial answer. This is the
  "parallel equals serial" gate; see [Parallelism](#parallelism) for why it is
  phrased that way.
- `independent_simulators_agree` — two separately constructed `Simulator`s.
- `a_reused_simulator_does_not_contaminate_the_next_rollout` — both documented
  escapes from the warm-start cache actually work.
- `a_zero_ulp_perturbation_never_separates` — two rollouts from identical
  inputs stay at *exactly* zero separation for the whole horizon, no tolerance.

The CI job `determinism` runs all of it across `{ubuntu x86-64, ubuntu
aarch64, macOS aarch64} x {debug, release}`.

---

## What had to change to make that true

Four things in the rollout path were order- or platform-dependent. All four are
fixed; they are listed because knowing *what* used to break tells you where to
look when something breaks again.

### 1. Broad-phase pair order was a function of position

`sweep_and_prune` visits endpoints in order of their `x` coordinate, so the
order it emitted pairs in depended on where the bodies happened to be. That
order propagates: `find_contacts` emits contacts in pair order, `assemble`
builds the Delassus rows in contact order, and projected Gauss-Seidel sweeps
those rows in order. **Gauss-Seidel is not order-invariant.** Its iterates
differ under a permutation, and because the solve terminates on a tolerance or
an iteration cap rather than at the exact minimizer, so does its answer.

The consequence was worse than a small numerical difference. Two bodies whose
`x`-extents cross swap the solver's sweep order *mid-rollout*, at a moment
determined by a continuous coordinate — so a perturbation one ulp wide can flip
that crossing one step early and change the trajectory discontinuously. That is
indistinguishable from a bug by inspection.

`sweep_and_prune` now sorts its output by `(i, j)`. The sort is on `usize`, so
it introduces no floating-point comparison of its own, and the solve order
becomes a function of the model's *identity* — which does not move under
perturbation at all.

### 2. Transcendental functions came from the platform's libm

`+`, `-`, `*`, `/` and `sqrt` are correctly rounded by IEEE-754 and agree
everywhere. `sin`, `cos`, `atan2`, `pow` and `tanh` are not covered by the
standard, and Apple's, glibc's and musl's implementations disagree. Measured on
one machine over 900k calls spanning `[-10, 10]`, musl's libm and Apple's
disagree in the last bit on **6.5%** of them.

That is not a small effect once amplified: a free joint calls the quaternion
exponential and logarithm twice per step, the impedance sigmoid calls `pow`
once per contact per step, and every hinge calls `sin_cos`. A single ulp
introduced at step 1 and grown at a Lyapunov rate of ~1 /s is macroscopic
within a minute.

The rollout path no longer calls the platform's libm. It calls
`phyz_math::fp`, which forwards to the pure-Rust [`libm`] crate — the same
source compiled on every target, and therefore the same bits. The call sites
converted were:

- `phyz_rigid::integrate_configuration` — via the new `phyz_math::quat_exp` and
  `quat_log`, which reproduce `tang::Quat::exp`/`log` exactly apart from the
  `sin_cos`/`atan2` they route through;
- `phyz_model::Joint::joint_transform_slice` (hinge `sin_cos`, ball and free
  `quat_exp`) and `Joint::passive_force` (friction `tanh`);
- `phyz_contact::material::SolImp::impedance` and `impedance_derivative` (four
  `pow` each);
- `phyz_contact::solver` (the cylinder rim sampler).

This is behind the `portable-fp` feature of `phyz-math`, on by default.
Disabling it restores the platform libm — faster, and no cross-platform
guarantee.

Rust never contracts to FMA or reassociates floating-point without an explicit
opt-in, and phyz sets no `RUSTFLAGS` and no `-ffast-math` equivalent, so with
libm handled there is nothing left that varies by target. That is the reasoning
behind the cross-ISA claim; the CI matrix is what checks it.

### 3. The warm-start cache was documented as inert, and is not

See [The warm-start cache](#the-warm-start-cache) below.

### 4. Nothing else — the audit's negative results

Worth recording, because they are the things you would otherwise re-check:

- **No `HashMap` iteration on the rollout path.** The only hash container is
  `phyz_contact::ContactCache`, and it is only ever *looked up* — its output is
  ordered by the contact list, never by the table. `phyz_world::trajectory`
  uses maps for output formatting, outside the stepping path.
- **No parallelism.** `rayon` appears nowhere in `phyz-rigid`,
  `phyz-collision`, `phyz-contact`, `phyz-model` or `phyz`. (It is an optional
  feature of `phyz-md` and `phyz-regge`, which are not rigid-body crates and
  are not covered here — their own docs say the feature trades bit-for-bit
  reproducibility for speed.)
- **Sorts already used `total_cmp`,** not `partial_cmp().unwrap()`, so a NaN
  cannot panic or produce an inconsistent ordering.
- **GJK and EPA are pure arithmetic.** No transcendentals outside their tests.

One pre-existing wart, left alone because it is a robustness question rather
than a determinism one: `manifold.rs` folds with `f64::min`, which returns the
non-NaN operand and so silently swallows a NaN rather than propagating it. The
fold order is fixed, so it is deterministic; it is noted here so the next
person does not have to rediscover it.

---

## The warm-start cache

`Simulator` carries a `ContactCache`: last step's contact impulses, keyed by
feature, used to seed the next solve. It used to be documented as unable to
change results, on the grounds that the contact problem is strongly convex, so
the seed can only change how many iterations the solve takes.

**That argument is sound about the minimizer and wrong about what the solver
returns.** `solve_contacts_warm` stops at `config.tolerance`, or at
`max_iterations` if it never gets there, and the PGS warm-up is also what hands
the active-set Newton stage its active set. Two different seeds stop at two
different points inside the same tolerance ball — and on a redundant manifold
that has not converged, at genuinely different active sets.

So the cache is hidden mutable state that moves the answer. Concretely: reusing
one `Simulator` across the candidates of a parameter search makes each
candidate's score depend on what was evaluated before it.

Three ways to be safe, in order of preference:

1. **One `Simulator` per rollout.** `Simulator::new()` starts with an empty
   cache. Cheapest and clearest.
2. **`sim.reset_contact_cache()` between rollouts.** Same guarantee, if you
   want to keep the allocation.
3. **`Simulator::new().with_warm_start(false)`.** Removes the cache from the
   picture entirely: `step_with_contacts` becomes a pure function of `(model,
   state)` with no dependence on the simulator's history at all. Costs
   iterations — a resting stack can take several times as many PGS sweeps from
   cold.

`contact_acceleration` already reads the cache without writing it, so asking
for sensor data cannot perturb the stepping trajectory.

---

## Chaos, and how to recognise it

Everything above is about the engine computing the same numbers from the same
inputs. It says nothing about what happens when the inputs differ in the last
bit — and there the answer is that a contact-rich rollout amplifies enormously.

`phyz-bench --suite divergence` measures this per scene. Measured at
`dt = 1 ms` over 5 s of simulated time, perturbing one coordinate by exactly
one representable step:

| scene | amplification | lyapunov | mantissa horizon |
|---|---|---|---|
| `pendulum` | 8.7x | −0.23 /s | — (contracting) |
| `double_pendulum` | 35x | +0.10 /s | 345 s |
| `ant` (free-flying, no contact) | perturbation absorbed | — | — |
| `box_stack_8` | **1.2e13x** | +1.16 /s | 31 s |

Read that bottom row carefully, because it is the whole point. **A stack of
eight boxes turns a `7e-18` difference into `8.5e-5` in five seconds.** If you
change one parameter by one ulp and a downstream score moves by hundreds of
points, that is not evidence of a bug — it is exactly what this table predicts.

Three readings the table supports:

- **`amplification`** is the number to compare an unexplained discrepancy
  against. Divide the discrepancy by `1e-16`. Comparable to the scene's
  amplification: chaos. Much larger, or present at step 0: bug.
- **Negative `lyapunov` means the scene contracts.** Dissipative contact
  forgets a perturbation once a body settles. This is the common case for
  anything that comes to rest, and it inverts the usual advice: an unexplained
  discrepancy in a *contracting* scene is more suspicious, not less.
- **`mantissa_horizon`** is `52 * ln(2) / lambda`: how long before two rollouts
  from bit-identical inputs would have lost every digit of agreement had `f64`
  rounding seeded them one ulp apart. Past that horizon, a trajectory-by-
  trajectory comparison of two runs is meaningless even when both are correct.

The same measurement is available as a library call —
`phyz::determinism::divergence` — so you can calibrate your own scene rather
than extrapolating from these.

### The trap that motivated all of this

A score that moves by hundreds of points under a 1-ulp perturbation is *not*
necessarily a chaotic state trajectory. It is more often a **discontinuous
objective**: a score that counts frames of flight, or thresholds a height, or
integrates up to the first contact event, changes by a whole step's worth the
moment the perturbation moves a discrete event across a step boundary. The
state can be `1e-15` apart and the objective `100` apart.

So when a score does not reproduce, measure the *state* divergence, not the
score's. If the states agree bitwise and the scores differ, the objective is
the discontinuous thing and no amount of engine determinism will fix it —
smooth the objective or report its distribution.

---

## Tolerance comparison, when bitwise is not what you want

Bitwise equality is the right test for "did the engine change". It is the wrong
test for "are these two runs the same experiment" across a version bump, a
model edit, or a deliberate physics change. For those:

```rust
use phyz::determinism::{state_distance, divergence, RolloutHasher, ulp_offset};

// How far apart, in stacked (q, v) units.
let d = state_distance(&a, &b);

// Is that difference explainable as chaos? Compare against what one ulp does
// to this scene over the same horizon.
let calib = divergence(model.nq, &initial, /* q_index */ 5, 1, steps, 10, |which, s| {
    sims[which].step_with_contacts(&model, s, 0.0, &material);
});
assert!(d < 100.0 * calib.final_distance(), "larger than chaos accounts for");
```

`Divergence` also carries the whole separation curve (`samples`), which is what
you want in a plot: a difference that appears at step 0 is a bug, one that
starts at `1e-16` and grows smoothly is chaos, and one that is zero for 400
steps and then jumps is a discrete event reached at a different step.

---

## Parallelism

The rigid rollout path is single-threaded. There is no parallel reduction whose
order could vary, which is why there is no "parallel vs serial" flag to set:
they are the same code path.

`concurrent_rollouts_agree_with_the_serial_one` is what keeps that true. It
runs the same rollout on eight threads and requires all eight to match the
serial answer, which fails the moment a step starts depending on process-global
mutable state, a shared scratch buffer, or a floating-point mode set per
thread. If phyz ever does gain a parallel solver, that test is where its
determinism obligation is already written down.

---

## Not covered

Be explicit about these when publishing:

- **The GPU backend (`phyz-gpu`).** Different arithmetic, different reduction
  order, and driver-dependent. Nothing here applies to it, and a GPU rollout
  should not be compared bitwise against a CPU one.
- **`phyz-md` and `phyz-regge` with the `parallel` feature.** Both trade
  bit-for-bit reproducibility for speed, by their own documentation.
- **Across phyz versions.** A physics fix moves the golden hashes, and should.
  The golden constants are updated in the same commit as the change that moved
  them, so `git log crates/phyz/tests/determinism.rs` is a list of every commit
  that altered simulated results. Pin the version in a publication; cite the
  commit if you can.
- **Across `tang` versions.** phyz's linear algebra lives there, and it is
  pinned by git rev in the workspace `Cargo.toml` for exactly this reason.
- **Model construction from files.** MJCF and URDF parsing is deterministic,
  but a model that comes out of a mesh pipeline or a fitting procedure may not
  be. Fingerprint your `Model`, not just your rollout, if this matters.

---

## If a golden hash fails

In order:

1. **Did you change physics?** If yes, check the physics gates
   (`analytic_benchmarks`, `contact_stability`, the energy suites) and update
   the constants **in the same commit**, so the history records which change
   moved the numbers.
2. **If you cannot name a change, you have found a bug**, and it is almost
   certainly non-determinism rather than physics. Look for: a new `HashMap`
   iterated rather than looked up; a sort by address, pointer or discovery
   order; a parallel reduction; a call to `f64::sin`/`cos`/`powf`/`atan2`
   instead of `phyz_math::fp`; a `partial_cmp().unwrap()` where `total_cmp`
   belongs.
3. **If it fails on only some cells of the CI matrix**, the difference is
   platform-dependent by construction. That narrows it to codegen or libm:
   check §2 above for a call site that was missed.
4. **Never update the constants to make a red test green** without doing step 1
   or 2. The value of a golden hash is entirely in the fact that nobody does
   that.

[`libm`]: https://crates.io/crates/libm
