# CUDA execution path for `phyz-gpu`

Status: **implemented, gated behind `phyz-gpu/cuda`**; verified on the host
against the CPU reference and against the WGSL kernels, and on an RTX PRO
6000 (Blackwell) pod: ant, 4096 worlds × 500 steps, 5.06e7 env-steps/s,
404× a CPU thread, parity 2.7e-5 (2026-08-18). The unified contact model —
heightfield terrain, the body-attached finite face (`BodyPlane`), Coulomb
friction and box manifolds, the velocity-level impulse solve with sweeps,
passive joint springs and armature — is ported too, and held to the WGSL
kernel-for-kernel in `tests/cuda_vs_cpu.rs::suite_unified_contact`.

---

## 1. Why a second backend

`phyz-gpu` runs its batch simulation through wgpu compute shaders. That is the
right default: Metal on a laptop, Vulkan on a workstation, one WGSL source.

It is not enough for rented cloud GPUs. On a RunPod RTX PRO 6000 (Blackwell)
pod, verified 2026-08-18: CUDA works, Vulkan cannot. The container's user
namespace cannot open `/dev/dri/renderD*` (the host leaves them `0660
root:root`; only `/dev/nvidia*` are made `0666`), and the NVIDIA 570 ICD has no
non-DRM fallback — hide `/dev/dri` and the ICD fails without ever touching
`/dev/nvidia*`. This is nvidia-container-toolkit issue #1041, and RunPod does
not officially support Vulkan. Every containerized GPU rental is exposed to it;
`/dev/nvidia*` — i.e. CUDA — is the one device path providers reliably deliver.

wgpu has no CUDA backend and no plan for one, so the choice was between
"reach the driver some other way" and "run the same physics through CUDA".

Options weighed:

| | verdict |
| --- | --- |
| **wgpu GL backend via EGL** — NVIDIA's EGL uses `/dev/nvidia*`, not DRM | plausible zero-kernel-change escape, but needs the container to mount `libnvidia-egl*` (`NVIDIA_DRIVER_CAPABILITIES=graphics`, not what compute images ship), and GLES compute has its own limits. Untested; worth a 20-minute probe on the pod, noted here so it is not forgotten. |
| **naga WGSL → something CUDA consumes** | naga emits SPIR-V, MSL, HLSL, GLSL, WGSL. No PTX, no CUDA C. Dead end. |
| **rust-gpu / rust-cuda** | means rewriting the kernels in Rust anyway, with a young toolchain, for the same one-source benefit as below. Not lower friction than CUDA C. |
| **CUDA C via `cudarc`, hand port** ✅ | ~900 lines of C mirroring ~1400 lines of WGSL; NVRTC compiles it at runtime, so **no CUDA toolkit at build time**; `cudarc`'s `dynamic-loading` dlopens `libcuda`/`libnvrtc`, so **the crate builds on any machine** and a missing driver is an `Err`. This is what landed. |

## 2. Shape

```
phyz_gpu::layout        one definition of every buffer layout (BODY_STRIDE …)
                        ├── used by the wgpu path (GpuBatchSimulator, unchanged)
                        └── used by phyz_gpu::cuda::BatchSim<B>

phyz_gpu::cuda::BatchSim<B: KernelBackend>
    same surface as GpuBatchSimulator: new / enable_ground_contact[_per_body] /
    enable_ground_contact_with_plane / enable_contact_terrain /
    enable_contact_impulse / set_heightfield / contact_sweeps /
    enable_pd_control / set_position_targets / load_states / set_controls /
    step / readback_states / readback_contacts
    │
    ├── CudaBackend   (feature `cuda`)      cudarc + NVRTC, the real thing
    └── HostBackend   (feature `cuda-host`) the .cu compiled as host C++ by build.rs
```

`KernelBackend` is deliberately tiny — allocate, upload, download, sync, and
one launch per pass with the scalar arguments spelled out. It is plumbing, not
physics; the sequencing (PD → contact → ABA → integrate) and all packing live
once in `BatchSim`. Adding a backend is ~200 lines and touches no physics.

`CudaBatchSimulator = BatchSim<CudaBackend>`; `HostBatchSimulator =
BatchSim<HostBackend>`.

The wgpu path is untouched apart from calling `layout::*` for packing (a pure
move) and one sign fix in the ABA shader (§4).

## 3. Kernel-source strategy

**One CUDA C file, `crates/phyz-gpu/cuda/phyz_kernels.cu`, hand-ported
function-for-function from the WGSL** in `src/shaders.rs` and
`src/pd_pipeline.rs`, and written in the subset that both NVRTC and a plain
C++ compiler accept (no CUDA vector types, own `v3`/`sv6`/`m66` structs,
`__float_as_int` behind a shim). The same text is:

* compiled by NVRTC at `CudaBackend::new` and launched on the device;
* compiled by `cc` in `build.rs` (feature `cuda-host`) with the thread bodies
  wrapped in serial `phyz_host_*` loops, and run on the CPU.

The second use is why the port could be built and gated on a machine with no
NVIDIA GPU: `tests/cuda_vs_cpu.rs` runs one generic suite (every joint type
single-step, 200-step trajectory, PD servo, ground contact incl. readback,
ant.xml, and CUDA C vs WGSL kernel-for-kernel) on the host backend always,
and on the device when there is one.

There is **no automatic translation** between WGSL and CUDA C. The two kernel
sources are held together by discipline plus tests: the CPU (`phyz-rigid`,
f64) is the reference for both, and the kernel-vs-kernel test pins them to
each other at 2e-4 over 300 steps (both are f32 doing the same arithmetic;
anything past that is a port bug, not precision). Layout constants are
mirrored by name and comment (`MAX_BODIES`, `BODY_STRIDE`, `GEOM_STRIDE`,
`CS_STRIDE`, `PD_DOF_STRIDE`) and asserted on the host side by `layout.rs`.

When a WGSL kernel changes, the `.cu` must change with it, and the parity
suite is what says whether it did. That is the cost of two sources; it was
judged smaller than a translator or a Rust-side single source, and it keeps
the Vulkan/Metal path exactly as it was.

### Precision / determinism

NVRTC is invoked with `--use_fast_math=false`, `--prec-div=true`,
`--prec-sqrt=true`; FMA contraction is left on, as it is under Metal and
Vulkan. So CUDA-vs-WGSL agreement is at the f32 rounding level, not
bit-exact — the same relationship the wgpu path already has across its own
backends. Kernels are single-threaded per world with no atomics, so the CUDA
path is deterministic run-to-run on one device.

## 4. What the port found: a sign bug in the WGSL ABA

Porting `build_motion_transform` and running the port on the host against the
f64 CPU showed the two disagreeing on any model whose child joint has a
non-zero `parent_to_joint` translation — i.e. every real robot. The WGSL
stored `skew(p)` transposed (= `-skew(p)`) while indexing it row-major, so the
translation block of the 6×6 motion transform, and with it every articulated
inertia propagated up through a translated joint in ABA pass 2, had the wrong
sign. `apply_motion` / `inv_apply_force` were hand-written and correct, so the
error only entered through the `Xᵀ I X` products; single-step tests at 5e-3
tolerated the per-step error. On a two-rod pendulum with the elbow 1 m down,
the GPU left the CPU by 0.2 rad in 200 steps and swung the wrong way from
rest; the closed-form double pendulum agrees with the CPU to 1e-9 and now
with both kernel sets to 1e-3 (f32).

Fixed in the WGSL (one block, `shaders.rs`) and pinned by
`tests/joint_offset_vs_cpu.rs`. Present since the shader's first commit
(Feb 2026). Anything trained on the wgpu path before this fix was trained on
slightly different dynamics than the CPU referee — the mismatch is largest
for long, heavy limbs on offset joints and negligible for the free base.

## 5. Running it

```bash
# builds anywhere (no CUDA toolkit needed); runs where libcuda + libnvrtc exist
cargo build --release -p phyz-gpu --features cuda

# smoke: 4096 ant worlds x 500 steps on CUDA, CPU parity check, env-steps/s
cargo run --release -p phyz-gpu --features cuda --example cuda_batch -- 4096 500

# the same program through the host-compiled CUDA C (no GPU needed)
PHYZ_KERNELS=host cargo run --release -p phyz-gpu --features cuda,cuda-host --example cuda_batch -- 4096 500

# parity suite: host always; device when present
cargo test -p phyz-gpu --features cuda,cuda-host --test cuda_vs_cpu

# the bench sweep on CUDA (reports `skipped` with the reason when it cannot run)
cargo run --release -p phyz-bench --bin phyz-bench --features gpu-cuda -- --suite gpu-cuda
```

Runtime requirements on the pod: an NVIDIA driver whose CUDA API is ≥ 12.8
(`cudarc` is pinned to `cuda-12080`; a Blackwell pod's R570 driver is exactly
that) and `libnvrtc.so` on the loader path (it ships with the CUDA runtime in
every `nvidia/cuda:*-devel` and `*-runtime` image; if the image is `-base`,
`apt install cuda-nvrtc-12-8` or point `LD_LIBRARY_PATH` at it).

Measured so far, ant.xml (9 bodies, nv 14), 4096 worlds × 500 steps:

| path | env-steps/s | vs CPU (1 thread, f64) |
| --- | ---: | ---: |
| CPU `phyz-rigid` ABA, 1 thread, f64 (Apple M-series) | 2.7e5 | 1× |
| CUDA C kernels compiled as host C++, 1 thread, f32 | 8.7e5 | 3.2× |
| wgpu / Metal on the same laptop | 4.9e6 | 18× |
| **CUDA on an NVIDIA device** | **not yet measured** | — |

Parity on the host-kernel run: max |q_kernel − q_cpu| = 2.7e-5 after 500
steps across 8 worlds. The device number and the device parity line are the
first thing to fill in on the pod; the example prints both.

## 6. What remains

Ported: PD servo, ground-plane penalty contact (sphere/box/capsule/cylinder/
mesh-AABB, per-body gains, contact readback), general ABA (revolute,
prismatic, fixed, ball, free; implicit damping), joint-aware semi-implicit
Euler — everything `GpuBatchSimulator` on `main` runs.

Not ported, deliberately:

* `GpuSimulator` (single-revolute pendulum-only, `ABA_SIMPLE` /
  `INTEGRATE_SIMPLE`) — subsumed by the general kernels.
* `sparse.rs` / `sparse_shaders.rs` — the sparse LTL path is a separate
  experiment and not on the RL hot path.
* **The `claude/unify-contact` branch's contact work** (heightfield terrain,
  body-attached plane, impulse solve, armature on device, `BODY_STRIDE` 36,
  `enable_contact_terrain` / `BodyPlane`) — that is what `ipse-sim`'s
  `k1_rl_skate_gpu` currently calls, and it is not on `main`. When it merges,
  the CUDA port needs: the ABA changes (armature, joint spring terms), the
  new body-table layout, and the terrain/impulse contact kernel. The backend
  seam is unaffected — it is kernel text plus one new launch. Until then the
  CUDA path runs the `main` API, which `ipse-sim` would use through the same
  calls it makes today minus `enable_contact_terrain`.

`ipse-sim` wiring: `rl_gpu::GpuCollector` holds a `GpuBatchSimulator`; the
CUDA type has the same methods (with `&mut self` where wgpu takes `&self`),
so a `type Sim = …` chosen by a `gpu-cuda` feature is the whole change on that
side.

## 7. The device-resident control loop (2026-08-19)

`ipse-sim`'s PPO collector measured, at 4096 K1-on-skateboard worlds on the
RTX PRO 6000: sim launches 0.0 s (async), `readback_states` 2–4 s, FK/observe
1–2 s, actor forward 1 s, per-world bookkeeping 0.5 s — per iteration, on ~15
host cores, GPU utilisation 0–19 %. Every control step read all worlds back,
observed, forwarded and sampled on the host, and uploaded targets. The sim was
never the clock; the host round trip was.

Three passes and two buffers move that loop on-device, behind the same
`KernelBackend` seam (three new `launch_*`, one `copy`, one `download_range`):

| pass | threads | what |
| --- | --- | --- |
| `phyz_fk` | nworld | the contact pass's FK chain (factored into `fk_world`, verbatim), written out per body as `XF_STRIDE` = 18 floats: world→body rotation, origin, body-frame angular/linear velocity |
| `phyz_obs` | nworld | one observation row per world from a small op table (`ObsOp`: const, `q−ref`, `v`, body pitch/roll/yaw-error/height) into a device history `[step][world][n_in]` |
| `phyz_policy` | nworld | two-hidden-layer tanh MLP over that row, per-input Gaussian noise (recorded in place), AR(1) diagonal-Gaussian sample, log-prob, and `base + clamp(action)` written straight into the PD target row; actions + log-prob into `[step][world][n_out+1]` |

Plus a **state history**: `record_state(slot)` is a device-to-device copy of
`q`/`v` ordered after the launches (the host never waits), and
`readback_state_history(range)` downloads a range once. A collector runs
`run_policy(step)`, `step()` × control_every, `record_state(step)`, and reads
everything back at the end (or in chunks it processes on a worker thread while
the device continues) — the reward and anything the op table cannot express
run on the host from the history, once per rollout instead of once per control
step. `set_model` re-uploads the body table in place for a domain-randomised
rebuild of the same robot; `enable_pd_control` / `enable_policy` /
`enable_state_history` keep their buffers when the shape is unchanged, so a
simulator lives across iterations.

Randomness is one xorshift64 stream per world (`world_seed(seed, w)`,
SplitMix-mixed), Box–Muller in double — the same arithmetic as
`ipse_dojo::search::XorShift::normal`, mirrored on the host by
`policy_pipeline::KernelRng`. That is what makes the parity test exact rather
than statistical: `tests/cuda_vs_cpu.rs::suite_policy` replays every world's
stream on the CPU (`policy_reference`, `observe_reference`) and holds
observations, noise, actions, log-probs, PD targets, FK rows and history slots
to f32 precision, on the host backend always and on the device when there is
one. Same seed, same device, same actions.

What it does not do, deliberately: reward and termination. `ipse-sim`'s
`SkateReturn::tick` is stateful and large; the honest cut was to feed it from
one bulk readback rather than port it. The device keeps stepping every world to
the horizon (as the collector already did — lanes are not reclaimable), and the
host discards samples after the step the referee says the episode ended.

Measured, `gpu_sim_bench` in `ipse-sim` (K1 + board, 32 bodies): the sim floor
alone is ~0.49 ms per step at ≤ 4096 worlds — 1024 and 4096 cost the same, so
the kernels are latency-bound (one long thread per world, ~22 threads per SM
on this part), and 16384 worlds cost 2×. Per 2500-step iteration that is
~1.2 s before any host work; the control loop above removes the host from the
per-step path, it does not make the kernels themselves faster.

## The update pass

`cuda/phyz_train.cu` is the second kernel file, and its own thing: the
*update* half of a reinforcement-learning iteration, where `phyz_kernels.cu`
is the collection half. Once the control loop runs on device, the PPO update
is what is left on the CPU — an f64 minibatch loop over a three-layer tanh
actor and critic — and it grows with the sample count until it is the clock.

`cuda::TrainPipeline` runs that loop. Its seam is a second trait,
`TrainBackend`, not more methods on `KernelBackend`: the simulation passes are
f32 throughout, while the optimizer keeps an f64 master weight and f64 Adam
moments, and the minibatch index list is `u32`. The same two backends exist
for the same reason — `CudaTrainBackend` (NVRTC, and `on_context` so a run
that collects and updates on one GPU opens one context) and
`HostTrainBackend` (the same `.cu` text as host C++, the CI referee).

Scope is the `epochs × minibatches` loop and nothing else. GAE, advantage
normalization and the Huber delta's own statistics are one linear sweep of the
batch, sequential per episode, and nowhere near the clock; the host does them
and uploads finished `adv`/`ret` columns. The minibatch *order* is the host's
too — the caller shuffles with its own PRNG and hands the indices over — so a
same-seed comparison against a CPU loop is meaningful, which is exactly what
`tests/train_parity.rs` does against tang's own `Linear`/`ModuleAdam` in f64.

Precision: activations, gradients and the forward weights are f32; parameters
and both Adam moments are f64, mirrored to f32 on every step. That is not a
compromise for the device's benefit — tang's `ModuleAdam` keeps `m`/`v` in f64
regardless of the parameter scalar, so this *is* the CPU optimizer's state.
Only the gradient carries f32 error, and the master weight is never rounded
through f32 between steps.

Every accumulation over the minibatch — weight gradients, bias gradients, the
loss and KL scalars — is a sequential double-accumulated loop in one thread.
That trades parallelism for two things worth more: the device and the host
walk of the same source agree, and the answer does not depend on how blocks
were scheduled. It is also the pass's remaining headroom: a weight gradient is
one thread per weight, so the actor's first layer launches 3520 threads on a
part with far more, and the update is throughput-bound on those row loops
rather than on launch latency (total time is flat in minibatch size — 2048,
4096 and 16384 all land within 25% of each other). A fixed-tile two-stage
reduction would keep the determinism and raise the occupancy; it is not done.

Measured on an RTX PRO 6000 Blackwell, `examples/cuda_train` at the shape a
real locomotion iteration produces — 1M samples, obs 55, action 22, hidden 64,
minibatch 4096, one epoch, 245 minibatches:

| | wall | per minibatch |
|---|---|---|
| tang f64 CPU | 35.26 s | 143.9 ms |
| device | 3.57 s | 14.6 ms |

9.9×, with the batch upload at 0.03 s. Parity on the same hardware, against
tang in f64: the actor's weight delta after one update agrees to 3.2e-7
relative, the critic's to 3.1e-7; after three updates 2.0e-7 and 1.7e-7; the
policy and value losses to ~1e-7, and the KL brake stops on the same
minibatch. The host walk of the same source lands in the same place.
