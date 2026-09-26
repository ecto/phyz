# MuJoCo harnesses

Two separate scripts asking two different questions:

| Script | Question |
|---|---|
| `mujoco_bench.py` | How fast is phyz next to MuJoCo? |
| `mujoco_agreement.py` | Does phyz compute the **same trajectory** as MuJoCo? |

A speed comparison says nothing about correctness, so the second one is the
more important of the two. It is documented at the bottom of this file.

# MuJoCo / MJX comparison harness

Optional, separately invoked, and **not** wired into CI or `make bench`.
MuJoCo needs a Python environment; MJX additionally needs JAX and a GPU to be
worth running. Making either a hard dependency would make the phyz benchmark
suite unrunnable for most people.

## Running it

```bash
cd crates/phyz-bench/python
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python mujoco_bench.py --json ../../../mujoco-results.json
```

MJX batch sweep (uncomment a `jax` line in `requirements.txt` first):

```bash
python mujoco_bench.py --mjx --json ../../../mujoco-results.json
```

Output is the same JSON schema the Rust harness emits (`schema_version: 1`), so
the two files merge into one document.

## What is held equal

Same as the Rust side, and for the same reason — a comparison against
differently-configured physics is not a comparison:

| Setting | Value |
|---|---|
| Timestep | 1 ms (pendulums, ant), 2 ms (box stack) |
| Gravity | `(0, 0, -9.81)` m/s² |
| Solver iterations | 4 |
| Friction | 0.5 |
| Restitution | 0 |
| Link geometry | 1 m rods, 1 kg, thin-rod inertia, hinged about +Y |
| Boxes | 8 × 10 cm cubes, 1 kg each, 1 cm initial gaps |
| Measurement | 2 warmup reps, 7 timed reps of 20 000 steps, median |

The MJCF in `mujoco_bench.py` is written to match
`crates/phyz-bench/src/scenes.rs` term for term — same masses, same
`diaginertia`, same initial `qpos`.

## What cannot be held equal — read this before quoting a number

- **Contact models differ three ways.** phyz uses a compliant penalty force
  (stiffness/damping), Rapier uses an impulse solver, MuJoCo uses a soft
  constraint parameterised by `solref`/`solimp`. `solref`/`solimp` are left at
  MuJoCo's defaults rather than tuned to imitate phyz — tuning them to match
  would amount to choosing the answer in advance. Only the friction
  coefficient is genuinely like-for-like.
- **Solver iterations are not the same unit of work.** Setting `iterations = 4`
  everywhere equalises the number we asked for, not the arithmetic performed.
- **Python overhead is in the MuJoCo numbers.** Each step is one FFI crossing
  into MuJoCo's C core. That is negligible for the ant and the box stack, and
  is *not* negligible for the single pendulum, where the step itself is
  sub-microsecond. Treat the pendulum row as an upper bound on MuJoCo's cost.
- **MJX is f32 on GPU; phyz's CPU path is f64.** Different precision means a
  different computation, not just a faster one.
- **MJX JIT compilation is excluded from the timing.** Compiled once before the
  timed repetitions. That cost is real at small batch sizes and is not
  represented in the throughput figure.

## Recording a result

Every published number must carry its hardware, OS, and library versions. The
harness collects these automatically into the `meta` block. If you run on a
laptop, say so — thermal throttling on a sustained benchmark is real, and a
number from a warm laptop is not a number from a cold one.


---

# Trajectory agreement harness

```bash
pip install mujoco
make agreement            # from the repository root
```

or directly:

```bash
cargo build --release -p phyz-bench --bin phyz-traj
python3 crates/phyz-bench/python/mujoco_agreement.py \
  --json agreement-results.json --markdown agreement-results.md
```

`phyz-traj` rolls an MJCF model forward with phyz and writes the trajectory as
JSON; the Python side runs the same model through MuJoCo and compares.

## What is held equal, and what is switched off

An agreement test is only meaningful if both engines are solving the same
problem. These are switched off **on both sides**, because the two engines
model them differently and leaving them on would measure the difference
between two approximations of a *constraint* rather than two computations of
the *dynamics*:

| Feature | Why it is off |
|---|---|
| Joint damping | MuJoCo integrates it implicitly inside its Euler integrator; phyz applies it as an explicit passive force |
| Joint springs, dry friction | Same objection |
| Joint limits | phyz: soft penalty. MuJoCo: constraint |
| Contact, equality constraints | Three different algorithms; see the contact caveats above |

**Armature is deliberately kept.** It is a constant added to the mass-matrix
diagonal, implemented identically by both engines, so it is part of the
dynamics — and on a model with very light distal links it is what keeps the
mass matrix well conditioned. Zeroing it makes both engines integrate an
ill-conditioned system at 1 ms and measures the timestep instead of the
agreement. (Discovered the hard way: with armature stripped, the hand model
blows up in phyz within 10 ms.)

Both sides report what they stripped, and the harness refuses to compare two
models whose stripping lists disagree.

## Scope

Fixed-base **and** free-floating kinematic trees. The two engines parameterise
a free joint differently:

| | phyz `Model` | MuJoCo `qpos` |
|---|---|---|
| free joint | `[ωx ωy ωz, x y z]` (6) | `[x y z, qw qx qy qz]` (7) |
| ball joint | `[ωx ωy ωz]` (3) | `[qw qx qy qz]` (4) |

`ω` is the rotation vector whose exponential is the body's orientation in its
parent frame — the same rotation MuJoCo stores as a quaternion, so
`phyz-traj --layout mujoco` converts with `quat_exp` / `quat_log` and nothing
else. Velocity differs too: MuJoCo orders a free joint linear-first with the
linear part in the **world** frame, phyz angular-first in the **body** frame,
so `--v0 --layout mujoco` swaps the halves and rotates the linear one.

Contact is not covered. Comparing two contact models compares modelling
choices, not implementations.

## Reading the result

**`step 1` is the column that matters.** One step from an identical state has
no room for accumulation, so it isolates whether the two engines compute the
same accelerations. At f64 epsilon (~2×10⁻¹⁶) they do.

The later columns are accumulation. Two algebraically equivalent but
numerically distinct factorizations — ABA against MuJoCo's LDL — integrating
ten thousand steps from an epsilon-sized seed separate polynomially, reaching
order 1 rad on a freely tumbling humanoid by 10 s. That is not a correctness
result in either direction; it is how long a shared trajectory survives.

Do not reach for "chaos" without checking. On these models it is *not* the
explanation: a broadband 1-ulp perturbation of the same initial state stays at
8×10⁻¹⁵ over the same window while the two engines reach 10⁻⁴. When a model
*is* in a sensitive regime the two effects look alike, so compare against
phyz's own 1-ulp divergence figure (`phyz-bench --suite divergence`) before
attributing a separation to either.

## A trap worth knowing about

A floating articulated body released **from rest** under uniform gravity does
not move internally: gravity accelerates the whole system rigidly, so no joint
sees a differential torque. Compared from rest, the free-base models agree to
10⁻¹⁵ while every joint coordinate stays frozen and the comparison measures
nothing at all. The harness gives them an initial joint velocity for exactly
this reason — the body tumbles, the joints swing, and the Coriolis and
centrifugal terms that free fall never exercises have to agree too.
