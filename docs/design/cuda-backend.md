# CUDA execution path for `phyz-gpu`

Status: **implemented, gated behind `phyz-gpu/cuda`**; verified on the host
against the CPU reference and against the WGSL kernels; **not yet run on an
NVIDIA device** at the time of writing (this machine has none). The first
device run is a smoke example + a test binary away — see §5.

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
