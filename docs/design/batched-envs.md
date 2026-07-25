# Batched RL environments

Status: **design + working Rust implementation**. The API in `crates/phyz-env`
is implemented and tested. Multi-DOF joints, joint limits, armature, real
sensors and the four benchmark models all landed. The GPU *backend* for the
`VecEnv` trait and the Python bindings remain specified but unbuilt.

---

## 1. Why this is the flagship surface

MJX and Isaac Gym took RL mindshare with one property: thousands of environments
on one GPU, with observations and actions never leaving device memory.
Everything else — model coverage, solver quality, documentation — followed that.

phyz has two things neither of them has:

- **One owner across sim and learning.** `tang` is in-house, so a rollout tensor
  can be handed to a policy without a serialization boundary, a DLPack dance, or
  a framework mismatch. Brax users must adopt JAX; MJX users must adopt Python
  *and* JAX; Rapier users have no ML story at all.
- **Portability.** `wgpu` runs on Metal, Vulkan, DX12 and WebGPU. Isaac needs
  CUDA; MJX needs a working JAX install. "Train on your MacBook, deploy the same
  code to a Linux box, demo it in a browser tab" is a claim only phyz can make.

The API below is designed so neither advantage is an afterthought: the tensor
bridge is in the core type, and nothing assumes a particular GPU vendor.

## 2. Core API

### Shape

`VecEnv` is `N` copies of one `Model` stepped in lockstep.

```rust
pub trait VecEnv {
    fn num_envs(&self) -> usize;
    fn observation_space(&self) -> &BoxSpace;
    fn action_space(&self) -> &BoxSpace;
    fn reset(&mut self, seed: Option<u64>) -> &StepBatch;
    fn step(&mut self, actions: &[f32]) -> &StepBatch;
    fn observations(&self) -> &[f32];
}
```

`StepBatch` is a struct-of-arrays, every field flat, row-major, environment-major:

| field | shape | meaning |
|---|---|---|
| `obs` | `N × obs_dim` | current observation |
| `rewards` | `N` | reward for this step |
| `terminated` | `N` | MDP failure (fell over, diverged) |
| `truncated` | `N` | time limit reached |
| `final_obs` / `final_valid` | `N × obs_dim`, `N` | terminal obs under `SameStep` autoreset |
| `episode_return` / `episode_length` / `episode_done` | `N` | Gymnasium's `info["episode"]` |

Struct-of-arrays, not array-of-structs, because every consumer — numpy, a `tang`
tensor, a GPU buffer — wants one contiguous block per field. A
`Vec<StepResult>` would force a gather every step.

### Gymnasium compatibility

The point of the shape above is that porting existing code is mechanical:

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(actions)
```

Deliberate matches to Gymnasium ≥ 1.0 `VectorEnv`:

- **`terminated` and `truncated` are separate.** Bootstrapping a value function
  correctly requires the distinction; collapsing them into `done` is the single
  most common source of silently wrong RL results.
- **`AutoresetMode::NextStep` is the default**, matching Gymnasium 1.0. On a
  terminal step you get the *terminal* observation. The following `step()`
  ignores its action, resets, and returns the reset observation with
  `reward = 0` and both flags `false`. `SameStep` (Gym ≤ 0.26, EnvPool) is
  available and reports the terminal observation in `final_obs`; `Disabled`
  hands reset control to the caller.
- **Out-of-range actions are clamped, not rejected.** A Gaussian policy emits
  them; crashing the simulator on step 3 of training is not acceptable.
- **`reset(seed)` seeds the whole batch deterministically** (§5).

Deliberate divergences:

- Only `Box` spaces. Discrete/Dict spaces have no place in a rigid-body vector
  env, and omitting them keeps the Python binding a field-for-field copy.
- Observations are `f32` even on the f64 CPU backend, because that is what a
  policy consumes and what the GPU backend can produce (§4).
- `step()` returns a borrow of internal buffers rather than fresh allocations.
  At 4096 envs × 376 floats × 1 kHz, allocating per step is the whole budget.

### Task definition

Rewards and terminations are **declarative**, not `Box<dyn Fn>`:

```rust
TaskSpec {
    reward: vec![
        RewardTerm::Velocity { body: 0, axis: Axis::X, weight: 1.0 },
        RewardTerm::Alive { weight: 1.0 },
        RewardTerm::CtrlCost { weight: 0.005 },
    ],
    termination: Termination {
        healthy_z: Some((0, 0.26, 1.2)),
        terminate_on_nonfinite: true,
        max_velocity: Some(1.0e3),
    },
    max_episode_steps: 1000,
    reset_qpos_noise: 0.1,
    reset_qvel_noise: 0.1,
}
```

A closure would pin reward evaluation to the CPU and force a full state readback
every step — exactly the cost the GPU backend exists to avoid. A declarative
term list can be compiled into the same compute dispatch as the physics. It is
also what makes the task serializable, hashable for run provenance, and
expressible from Python without a callback across FFI on every step.

Observations are declarative for the same reason (`ObsSpec` / `ObsTerm`).

**`max_velocity` is a divergence guard, not a task rule.** Numerical blow-up
passes through enormous-but-finite values long before it reaches NaN, and by
then the observation clip has been saturated for many steps, feeding the policy
garbage. Catching it early converts an unrecoverable batch into one terminated
episode. Isaac Gym and Brax both do this.

### Ready-made benchmarks

```rust
let mut env = phyz_env::make(Benchmark::Ant, "models", 4096)?;
```

`make` refuses a model whose MJCF used features the parser dropped, and refuses
a model that fails `Model::validate` (§8, B6).

### Backends

```
              ┌─────────────────┐
   MJCF ────► │     Model       │
              └────────┬────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
  BatchEnv (CPU, f64)          GpuBatchEnv (f32)     [designed, not built]
  reference semantics          phyz-gpu wgpu compute
        │                             │
        └────────► StepBatch ◄────────┘
                       │
                 tang tensors / numpy
```

The CPU backend is the *specification*. `crates/phyz-env/tests/vec_env.rs` is
written against the trait, so the GPU backend must pass the same file — that is
the acceptance criterion, not a hand-written oracle.

The GPU *kernel* underneath (`phyz-gpu`) is now capable of the benchmark models
(§8, B4); what is missing is the `VecEnv` wrapper around it — observations,
rewards and autoreset as compute passes.

## 3. Zero-copy interop with `tang`

### What is true today

| path | status |
|---|---|
| Rust → Rust, CPU | **zero-copy.** `BatchView` borrows the env's buffer. |
| Rust → `tang::Tensor`, owned | **zero-copy.** `Batch::into_tang()` moves the `Vec<f32>`. |
| Rust → `tang::Tensor`, borrowed | **copies.** `tang_tensor::Tensor` owns its `Vec<S>` and has no borrowed constructor. |
| Rust → numpy | zero-copy by design (borrowed `PyArray` over sim memory); not yet built. |
| GPU → GPU | **phyz half done, tang half pending.** See below. |

`crates/phyz-env/src/tensor.rs` says exactly this, per method. A "zero-copy"
claim that quietly memcpys 4096 × 376 floats per step is worse than no claim.

### The GPU→GPU gap, concretely

`phyz_gpu::GpuBatchSimulator::interop()` now exposes the device, queue and the
`q` / `v` / `ctrl` buffers, each `num_envs`-major and tightly packed so it maps
onto a `[num_envs, dim]` tensor with no reshape (`crates/phyz-gpu/src/interop.rs`).
That is phyz's entire half of the contract.

Three additive changes remain, all in `tang`:

1. **Align `wgpu` versions.** `phyz-gpu` pins `wgpu = "23.0"`; `tang` pins
   `wgpu = "24"`. Two different `wgpu::Device` types, so no buffer can cross.
   Best fixed by hoisting `wgpu` into one shared workspace dependency.
2. **`tang_gpu::GpuDevice::from_raw(Arc<Device>, Arc<Queue>)`.** `GpuDevice`
   owns its device by value and can only be built by `GpuDevice::new()`, which
   requests its own adapter. phyz already stores both as `Arc` and already
   accepts a caller-supplied pair via `GpuBatchSimulator::with_device_queue`.
3. **`tang_gpu::GpuBuffer::from_wgpu(wgpu::Buffer, len)`.** `GpuBuffer`'s fields
   are private with no constructor from an existing buffer.
   `tang_compute::ComputeTensor::from_buffer` is the precedent.

With those, the loop becomes: physics pass writes the observation buffer → that
same `wgpu::Buffer` is wrapped as a `GpuTensor` → the policy's matmuls read it →
the action tensor's buffer binds directly as the env's `ctrl_buffer`. One queue,
one submission per step, zero host round-trips. **No competitor can do this**:
Brax goes through JAX's XLA buffers, MJX through `jax.dlpack`, Isaac through a
CUDA-only path.

A shared device also fixes a correctness hazard, not just performance: today
`GpuBatchSimulator::new` and `GpuDevice::new` each request their own adapter, so
on a multi-GPU box the simulator and the policy can silently land on *different*
GPUs.

**Sequencing:** land the CPU path first (real zero-copy for `into_tang`, and it
unblocks end-to-end training today), then the three `tang` changes, then the
fused GPU loop. Do not market GPU→GPU until it runs.

## 4. Precision: f32 and f64

wgpu has no native f64 — WGSL does not have the type, and neither Metal nor
WebGPU expose one. This is not a phyz limitation to engineer around; it is the
platform. So take a position and document it.

### The position

| path | precision | why |
|---|---|---|
| CPU dynamics (`phyz-rigid` ABA/RNEA/CRBA, `BatchEnv`) | **f64 throughout** | unchanged; this is the scientific path |
| GPU batch dynamics (`phyz-gpu`) | **f32 throughout** | the only option on wgpu |
| Observations, actions, rewards, both backends | **f32** | policy inputs; f64 buys nothing |
| Differentiable/adjoint paths (`phyz-diff`) | **f64, CPU only** | gradient accuracy is the entire product |
| Inverse problems (`phyz-real2sim`) | **f64, CPU only** | see below |

The boundary sits at the *backend*, not scattered through the code.
`gpu_state.rs` downcasts on upload and upcasts on readback; that is the only
place the conversion happens.

### What users should expect

- **RL: f32 is fine, and everyone else already does it.** MJX, Brax and Isaac
  are all f32. Policy gradients are estimated from noisy Monte-Carlo returns;
  f32 dynamics error is far below the sampling noise. Measured GPU-vs-CPU
  divergence over one step is ~1e-4 for a free body and ~5e-4 for a floating
  base with a limb (`crates/phyz-gpu/tests/multidof_vs_cpu.rs`) — irrelevant
  against a policy's own variance.
- **Long unstable rollouts: f32 is where divergence starts.** A stiff contact in
  f32 can blow up an environment that would survive in f64. Hence
  `terminate_on_nonfinite` *and* `max_velocity` default on.
- **Scientific inverse problems: do not use the GPU backend.** Trajectory
  matching, parameter identification and adjoint gradients need f64. The honest
  statement is: *phyz's GPU path is an RL accelerator, not a double-precision
  solver.*

### Should we build emulated-f64 or a high-precision GPU fallback?

**No, and here is the reasoning to point at when someone asks.**

- *Emulated f64 (double-float / Dekker arithmetic).* Costs 10–20× per
  arithmetic op and roughly doubles register and bandwidth pressure. A
  double-float ABA kernel would land near CPU throughput while being far harder
  to verify. Very high engineering cost; the population wanting both "on a GPU"
  and "in double precision" is small.
- *Compensated summation in the hot accumulations only.* Cheap (≈2× on the
  accumulate, not the whole kernel), and it addresses the dominant error source:
  long recursive accumulation down a kinematic chain. Worth doing **if and when
  GPU precision is empirically shown to matter**. Park it.
- *CPU f64 fallback.* Already exists — it is `BatchEnv`, the reference backend.
  Slower but exact, selected by choosing a backend rather than a flag. This is
  the supported high-precision path and should be named as such.

Rejecting emulated f64 explicitly, with numbers, is better positioning than a
vague "high precision available". It also lets the docs say something MJX's
cannot: *the same API gives you a bit-exact f64 reference to validate against.*

## 5. Determinism

### What phyz can guarantee

**Same machine, same binary, CPU backend: bit-identical.** Enforced by
`same_seed_gives_bit_identical_rollouts`, and by
`benchmarks_are_bit_reproducible` across all four models (so contact, servos and
floating bases are covered, not just a pendulum).

**Same machine, same binary, CPU backend, *different `num_envs`*:
bit-identical per environment.** Environment `k` produces the same trajectory
whether it runs in a batch of 4 or 4096, because randomness is *counter-based*:
every draw is a pure function of `(base_seed, env_index, episode_index,
draw_index)` via SplitMix64 (`src/rng.rs`), with no shared mutable RNG state and
no dependence on iteration order or on how many environments reset in the same
step. Enforced by `env_k_is_independent_of_batch_size`.

This is stronger than what Rapier offers and it is what researchers actually
need: a reported result reproduces even at a different batch size.

**Across platforms, CPU backend, IEEE-754 f64: bit-identical, conditionally.**
Conditions, stated plainly:

- No fast-math / FMA-contraction differences. Rust does not enable fast-math and
  phyz must never opt in.
- No platform-libm transcendentals in the dynamics feedback path. Today
  `Joint::joint_transform_slice`, `Quat::exp` and `Quat::log` call `sin`/`cos`/
  `atan2`, which can differ by 1 ULP between glibc, macOS and musl. **So
  cross-platform bit-identity is not currently guaranteed.** Claiming it would
  require vendoring correctly-rounded implementations — bounded work, and the
  right follow-up if networked-game determinism becomes a target market.
- Reduction order must be fixed. It is: ABA is a serial tree recursion.

**GPU backend: not bit-reproducible across devices, and never will be.** Say so
up front. Different vendors' compilers make different FMA-contraction and
reassociation choices for the same WGSL. The guarantee phyz *can* make is: same
device + same driver + same shader build → bit-identical. That is enough to
reproduce a training run on the machine it ran on, and it is what MJX and Isaac
deliver in practice too.

### Summary table

| scope | CPU f64 | GPU f32 |
|---|---|---|
| same machine, same `num_envs` | ✅ bit-identical | ✅ same device+driver |
| same machine, different `num_envs` | ✅ bit-identical | ✅ (counter-based RNG) |
| across OS / CPU vendor | ⚠️ pending vendored libm | ❌ |
| across GPU vendor / driver | n/a | ❌ by construction |

Publishing this table is itself a competitive move. Rapier "partially" delivers
determinism and does not say where the line is; being precise about the line is
what a researcher choosing an engine wants.

## 6. Python bindings

`crates/phyz-py` already exists (pyo3 0.28 + numpy 0.28, `crate-type =
["cdylib"]`, its own `[workspace]`). The vector env is a new module in it, not a
new crate. **Design only — not implemented.**

### API surface

```python
import phyz, numpy as np

env = phyz.make_vec("models/ant.xml", num_envs=4096, backend="gpu")
obs, info = env.reset(seed=0)                # obs: (4096, obs_dim) float32
obs, rew, term, trunc, info = env.step(act)  # act: (4096, nu) float32
```

- Registered as a `gymnasium.vector.VectorEnv` subclass, so SB3, CleanRL and
  RSL-RL work unmodified. This is the whole point: reach MuJoCo users without
  first building a Rust-native RL ecosystem.
- `obs` is a numpy array **borrowing** simulator memory, marked read-only, and
  invalidated by the next `step()`. Documented explicitly, with a `.copy()`
  escape hatch — a stale-view bug in someone's replay buffer would be a very bad
  first impression.
- Actions accept any C-contiguous `float32` array; non-contiguous or `float64`
  input is converted once with a warning rather than silently per step.
- With the GPU backend active, `env.step_tensor(...)` takes and returns `tang`
  device tensors, skipping numpy entirely — the zero-copy path of §3.
- The GIL is released around `step()`.

### Packaging and release

- **maturin** with `abi3-py39`, so one wheel per platform covers every Python
  ≥ 3.9 instead of one per minor version.
- **Wheels:** `manylinux_2_28 x86_64`, `macos arm64`, `macos x86_64`,
  `windows x86_64`, plus a CPU-only sdist fallback. wgpu means no CUDA toolkit
  in the build matrix and no `+cu121` wheel variants — a genuinely smaller
  release surface than any competitor's.
- **GPU at runtime, not build time.** The wheel links `wgpu`, which picks
  Metal/Vulkan/DX12 at load. `backend="gpu"` raises a clear error listing the
  adapters it did find, rather than failing at import.
- **CI:** GitHub Actions matrix via `maturin-action`, publishing to PyPI on tag
  with trusted publishing (no long-lived token). Wheel smoke test = import,
  `make_vec` an ant, 100 steps, assert finite.
- **Versioning:** the Python package version tracks the Rust crate version.
  Pre-1.0, document that observation layouts may change between minors — a
  changed `ObsSpec` silently invalidates pretrained policies, so it is a
  release-note item, not a footnote.
- **Name:** publish as `phyz` on PyPI.

Bindings should not be written until the Rust `VecEnv` has a GPU backend behind
it. A Gymnasium-compatible wrapper over a single-threaded f64 CPU sim is slower
than MuJoCo itself and would establish exactly the wrong reputation.

## 7. Benchmark models

Credibility with this audience is a fixed list: **ant, half-cheetah, humanoid,
shadow hand.** All four now load, batch, reset, step and stay finite.

| model | file | nq = nv | actuators | status |
|---|---|---|---|---|
| ant | `models/ant.xml` | 14 | 8 motors | ✅ stable, lands and settles |
| half-cheetah | `models/half_cheetah.xml` | 12 | 6 motors | ⚠️ **contact unstable** (§8, B5) |
| humanoid | `models/humanoid.xml` | 23 | 17 motors | ✅ stable, falls and terminates |
| shadow hand | `models/shadow_hand.xml` | 24 | 20 position servos | ✅ stable, servos track |

**These are re-authored, not copies of the Gym/MuJoCo files.** Topology, DOF
count and action space match, so *throughput* comparisons are meaningful. Link
lengths, masses and gearing do not, so **episode returns are not comparable to
published Gym numbers** and must never be presented as if they were. Each file
says so in a header comment.

Two further honesty notes:

- `models/shadow_hand.xml` is a **capsule approximation** of the Shadow Hand,
  not the Shadow Hand. The real model is defined by mesh assets and
  tendon-coupled distal joints, neither of which phyz supports. The file matches
  the articulation (24 hinge DOF, 20 actuators, same finger topology) so the
  high-DOF regime is exercised. It must not be presented as the real model.
- The humanoid uses hinge triples where the reference uses 3-DOF ball joints, so
  that the GPU and CPU paths agree today. Once ball joints are exercised
  end-to-end they should collapse back to `<joint type="ball"/>`.

### Measured CPU floor

M4 Max, single thread, f64, ant, 1024 envs, `frame_skip=5`, **with ground
contact and joint limits active**: **2.7k env-steps/s / 13k physics-steps/s**.

An earlier figure of 23.5k env-steps/s was measured before the ant had any
geoms — i.e. with contact doing nothing. The two are not comparable, and the
smaller number is the honest one.

### MJCF parser coverage

Now handled: `<compiler>`, `<option>`, `<worldbody>`, `<body>` (incl.
`childclass`), `<joint>`/`<freejoint>` (hinge/slide/ball/free, `range`,
`limited`, `damping`, `armature`), `<inertial>` (`diaginertia`, `fullinertia`,
`quat`), `<geom>` (sphere/capsule/box/cylinder/plane, `fromto`, `quat`,
`density`, `mass`, `contype`/`conaffinity`), `<default>` class trees with
inheritance, `<actuator>` (`motor`, `position`, `velocity`, affine `general`),
`<sensor>` (recorded), and geom-derived mass/inertia.

Remaining gaps, all *reported* rather than silently dropped via
`MjcfLoader::unsupported()`:

| gap | blocks | scope |
|---|---|---|
| `<asset>` / `<mesh>` | the real shadow hand, realistic humanoids | large: STL/OBJ loading, convex decomposition, mesh–mesh collision |
| `<equality>` | closed kinematic loops | medium: needs a constraint solver |
| `<tendon>` | coupled finger joints | medium; `phyz-world/src/tendon.rs` exists but is unwired |
| `<sensor>` → observations | models whose Gym obs uses sensors | small, now that sensors are real (B1) |
| site/tendon actuators | some manipulation models | small |

## 8. Blockers and what happened to them

### B1 — Sensors returned placeholder zeros ✅ **fixed**

`phyz-world/src/sensor.rs` returned hard-coded zeros for `BodyAccel`,
`ForceTorque` and the accelerometer half of `Imu`, and `max_dist` for
`Rangefinder`. A policy fed these would train against dead channels with no
error ever surfacing.

Fixed by exposing what the dynamics recursions already computed and discarded:

- `phyz_rigid::aba_dynamics` returns per-body spatial velocities **and
  accelerations** alongside `qdd`. `BodyAccel`/`Imu` now report true proper
  acceleration (a resting body reads `+g`; a free-falling one reads zero, and
  both are tested).
- `phyz_rigid::rnea_with_wrenches` returns the per-body joint wrench.
  `ForceTorque` reports it.
- `Rangefinder` now returns `SensorError::NotImplemented` rather than a
  plausible number, because `phyz-collision` has GJK but no ray cast. **A sensor
  that cannot be computed must not exist.**
- `Sensor::read` returns `Result`, and out-of-range targets are errors.

`phyz-env` still does not route observations through `Sensor` — `ObsSpec`
offers only directly-computed quantities. Converging the two types so MJCF
`<sensor>` elements map to observation terms is the remaining follow-up.

### B2 — MJCF parser coverage ✅ **largely fixed**

`<default>` class trees with `childclass` inheritance, geom-derived inertia,
`fromto`, `fullinertia`, position/velocity/general actuators and `<freejoint>`
all landed; see §7. Remaining gaps are listed there and are now *reported*
rather than silently dropped.

One latent bug fixed along the way: MJCF `size[1]` for a capsule is the **half**
length, but `Geometry::Capsule::length` is the full length (contact code
computes `length * 0.5`). The parser passed it through unconverted, so every
capsule was half its intended size.

### B3 — Free-joint `q`/`v` layout mismatch ✅ **fixed**

`q` for a free joint is `[pos(3), exp-coords(3)]` while `v` is
`[angular(3), linear(3)]`. A flat `q += dt*v` adds angular velocity into
position — wrong for every floating-base model, which is all four benchmarks.

The joint-aware integrator now lives in `phyz_rigid::integrate` as the canonical
one, `phyz-env` re-exports it, and the GPU `INTEGRATE_SHADER` was rewritten to
match it exactly (one thread per (env, joint), quaternion compose-and-relog for
ball and free joints). `GpuSimulator`, which is pendulum-only, keeps the flat
version under the name `INTEGRATE_SIMPLE_SHADER`.

### B4 — GPU kernel supported only 1-DOF joints ✅ **fixed**

`pack_bodies` mapped ball and free joints to *fixed*, silently welding them.
Since all four benchmarks have a floating base, no benchmark model ran correctly
on the GPU.

The WGSL ABA now handles 1–6 DOF joints generically: `S` as a 6×n subspace
matrix, `U = I_A S`, `D = Sᵀ U` inverted by Gauss-Jordan with partial pivoting
(n ≤ 6), and `qdd = D⁻¹(u − Uᵀ a_c)`. `BODY_STRIDE` did **not** need to change —
`ndof` is derived from the joint type. `MAX_BODIES` went from 16 to 32 so the
humanoid fits; that raises private-storage pressure, which is the natural limit
of the one-thread-per-environment design and the thing to revisit first if the
kernel gets slow.

The contact shader's forward kinematics was fixed too: it treated ball and free
joints as identity, which pinned every body of a floating-base model at the
origin so no foot ever reached the ground.

Verified by `crates/phyz-gpu/tests/multidof_vs_cpu.rs`: free, ball and
floating-base-with-limb models match the f64 CPU reference to ~1e-4, a free body
actually falls, batches stay independent, and ant runs.

### B5 — Contact ⚠️ **improved, still the weakest link**

The prototype ground contact was rewritten and now:

- respects **geom offsets** (MJCF geoms sit where `fromto`/`pos` put them,
  usually well below the body origin — testing the body origin made the cheetah
  collide with its hips while its feet passed through the floor);
- decomposes shapes into **multiple contact points** (a capsule is exactly a
  segment ⊕ sphere, so a plane contact is exactly two sphere contacts at its
  endpoints; collapsing it to one put the contact at an end cap and gave a
  horizontal limb a half-metre lever arm);
- derives stiffness from **mass and a response time constant** (MuJoCo `solref`
  style) instead of a fixed constant that is stable for a torso and explosive
  for a toe;
- bounds the impulse by a **Baumgarte-style partial correction**, since a link
  in a chain presents the chain's articulated effective mass, not its own.

**Half-cheetah still diverges.** Its torso is a 1 m capsule; landing on one end
produces a contact torque the penalty model overshoots on. The divergence guard
catches it and resets that environment, so the batch stays usable and finite,
but the cheetah is **not** credibly simulated today and must not be presented as
working. `half_cheetah_contact_is_known_to_be_unstable` asserts this so it
cannot be quietly forgotten; when a real solver lands, that test should start
failing and be deleted.

The fix is a real constraint solver, not more penalty tuning. `phyz-contact`
(CPU) and `phyz_gpu::ContactPipeline` (GPU) both exist and disagree on
parameterisation; unifying them behind one contact-parameter struct is the next
step.

### B6 — New: joint limits, armature and rest-pose validation ✅ **fixed**

Three defects found while getting the benchmarks to run, each of which produced
a *plausible-looking* simulation rather than an obvious failure:

- **Joint limits were parsed and never enforced.** `Joint::limits` was stored
  and read by no dynamics code, so legs folded through themselves without
  resistance. Now implemented as a soft limit scaled by the joint's own
  articulated inertia and a response time constant, with the damping term folded
  into the effective inertia so it is unconditionally stable.
- **`armature` was parsed and dropped.** It is the standard cure for a light
  distal link driven by a stiff servo (`ω = sqrt(kp/I)` exceeds what an explicit
  integrator can follow when `I` is tiny). Adding it to the mass-matrix diagonal
  is what made the 24-DOF hand stable.
- **Explicit joint damping was only conditionally stable.** `dt < 2M/c` is
  violated by light links at RL timesteps. Damping is now implicit via
  mass-matrix augmentation (`D += dt·c`), matching MuJoCo's `implicitfast`.
  Mirrored exactly in the WGSL kernel, or the backends would disagree on any
  damped model.

Plus `Model::validate()`, which catches the authoring mistake that motivated it:
a joint whose `range` excludes zero. `default_state()` starts every joint at
`q = 0`, so such a model begins already violating its own limit, the limit
spring fires at full strength on step one, and the robot catapults — with
nothing pointing at the model file. `phyz_env::make` refuses such models.

### B7 — New: frame convention was ambiguous ✅ **fixed**

`forward_kinematics` is documented as returning "world_to_body" transforms,
which reads as though the world pose needs `.inverse()`. It does not: `.pos` is
already the world position and `.rot` is world→body. I got this wrong first,
which mirrored models through the origin. Now locked down by
`crates/phyz-rigid/tests/frame_conventions.rs`.

## 9. What exists now

| crate | change |
|---|---|
| `phyz-env` **(new)** | `VecEnv`, `BatchEnv`, `ObsSpec`, `TaskSpec`, counter-based RNG, contact, benchmark presets, tensor bridge |
| `phyz-mjcf` | `<default>` class tree, geom-derived inertia, `fromto`, actuator types, sensors, unsupported-feature reporting |
| `phyz-model` | affine `Actuator`, `GeomOffset`, `Joint::armature`, `Model::validate` |
| `phyz-rigid` | `aba_dynamics`, `rnea_with_wrenches`, canonical `semi_implicit_euler`, implicit damping, joint limits, armature |
| `phyz-gpu` | multi-DOF ABA, joint-aware integrate shader, ball/free FK in contact, `interop()` |
| `phyz-world` | real sensors, `SensorError`, `SensorContext` |
| `models/` | ant rebuilt; half-cheetah, humanoid, shadow-hand approximation added |

## 10. Recommended next steps

1. **B5 — a real contact solver.** Half-cheetah is the visible symptom; contact
   is the last thing standing between "loads and steps" and "trains".
2. **The GPU `VecEnv` backend.** The kernel handles the models now; what is
   missing is observations, rewards and autoreset as compute passes so a rollout
   never touches the host.
3. **The three `tang` changes** in §3, then the fused GPU rollout loop.
4. **Train something.** A learning curve on ant is worth more than any further
   feature; it is also the only real test that the observation, reward and
   autoreset semantics are right.
5. **Python bindings**, only once 2–4 land.
6. Vendored correctly-rounded transcendentals, if cross-platform determinism
   becomes a target.
