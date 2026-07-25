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
