//! CUDA execution path for batched simulation.
//!
//! The wgpu path in [`crate::GpuBatchSimulator`] needs a Vulkan (or Metal /
//! DX12) device. Rented cloud GPUs frequently cannot provide one — the
//! container sees `/dev/nvidia*` but not `/dev/dri/renderD*`, and the NVIDIA
//! Vulkan ICD has no non-DRM fallback — while CUDA works out of the box. This
//! module runs the same physics through CUDA instead.
//!
//! # Shape
//!
//! [`BatchSim`] is the simulator: it owns the flat buffers, packs the model
//! with [`crate::layout`], and sequences the four passes (PD → contact → ABA
//! → integrate) exactly as the wgpu simulator does. It is generic over a
//! [`KernelBackend`], which is only "allocate / upload / download / launch":
//!
//! * [`CudaBackend`] (feature `cuda`) — the real thing. Kernels are compiled
//!   from `cuda/phyz_kernels.cu` at construction time with NVRTC and launched
//!   through `cudarc`. `libcuda`/`libnvrtc` are loaded lazily at runtime, so
//!   the crate builds on any machine and [`CudaBatchSimulator::new`] returns
//!   `Err` where there is no driver.
//! * [`HostBackend`] (feature `cuda-host`) — the same `.cu` file compiled as
//!   plain C++ by `build.rs`, with the thread bodies walked serially. It
//!   exists to pin the CUDA C port against the CPU reference on machines
//!   without an NVIDIA GPU; it is a verification tool, not a fast path.
//!
//! # Kernel source
//!
//! `cuda/phyz_kernels.cu` is a hand port of the WGSL in [`crate::shaders`]
//! and [`crate::pd_pipeline`], written in the subset both NVRTC and a host
//! C++ compiler accept. There is no automatic translation between the two
//! kernel languages; the CPU-as-reference parity tests are what hold them
//! together. See `docs/design/cuda-backend.md`.

use crate::contact_pipeline::{
    BodyContactGains, BodyContactState, BodyPlane, ContactParams, GroundContactParams,
    pack_contact_geometry, validate_heightfield,
};
use crate::gpu_batch_simulator::default_contact_sweeps;
use crate::layout::{
    self, CONTACT_STATE_STRIDE, check_pd_dofs, pack_bodies, pack_pd_dofs, pack_rows,
    pack_states, unpack_contacts, unpack_states,
};
use crate::pd_pipeline::PdDof;
use crate::policy_pipeline::{
    OBS_OP_STRIDE, PolicySpec, XF_STRIDE, pack_com_table, pack_obs_ops, world_seed,
};
use phyz_model::{Heightfield, Model, State};

#[cfg(feature = "cuda-host")]
pub mod host;
#[cfg(feature = "cuda")]
pub mod nvrtc;
pub mod train;
#[cfg(feature = "cuda-host")]
pub mod train_host;
#[cfg(feature = "cuda")]
pub mod train_nvrtc;

#[cfg(feature = "cuda-host")]
pub use host::HostBackend;
#[cfg(feature = "cuda")]
pub use nvrtc::CudaBackend;
pub use train::{
    AdamCfg, KlMode, NetDims, PpoUpdateCfg, SampleBatch, TrainBackend, TrainPipeline, UpdateStats,
};
#[cfg(feature = "cuda-host")]
pub use train_host::HostTrainBackend;
#[cfg(feature = "cuda")]
pub use train_nvrtc::CudaTrainBackend;

/// The CUDA source every backend in this module executes.
pub const KERNEL_SOURCE: &str = include_str!("../../cuda/phyz_kernels.cu");

/// Threads per block for every launch (matches the WGSL workgroup size).
pub const BLOCK_SIZE: u32 = 64;

/// Scalar arguments of the PD servo pass.
#[derive(Debug, Clone, Copy)]
pub struct PdArgs {
    /// Worlds.
    pub nworld: u32,
    /// `q` width per world.
    pub nq: u32,
    /// `v` width per world.
    pub nv: u32,
    /// Servoed DOFs per world.
    pub n_dofs: u32,
    /// Non-zero when the pass should sum the policy's feedforward-torque
    /// row into the servo term before the effort clamp. 0 leaves the
    /// `tau_ff` buffer untouched and the arithmetic exactly as it was.
    pub has_tau: u32,
}

/// Scalar arguments of the contact pass. Everything else the pass needs
/// travels in the `cparams` buffer (see [`ContactParams`]), which is why
/// this is only the launch width.
#[derive(Debug, Clone, Copy)]
pub struct ContactArgs {
    /// Worlds.
    pub nworld: u32,
}

/// Scalar arguments of the ABA pass.
#[derive(Debug, Clone, Copy)]
pub struct AbaArgs {
    /// Worlds.
    pub nworld: u32,
    /// `v` (and `q`) width per world.
    pub nv: u32,
    /// Timestep.
    pub dt: f32,
    /// Bodies per world.
    pub nbodies: u32,
    /// Gravity.
    pub gx: f32,
    /// Gravity.
    pub gy: f32,
    /// Gravity.
    pub gz: f32,
}

/// Scalar arguments of the fused impulse step.
///
/// One launch covers `nsteps` whole steps of the impulse sequence — PD, the
/// leading ABA, `sweeps` x [contact, ABA], integrate — for one world per
/// thread. See `step_impulse_thread` in `cuda/phyz_kernels.cu`.
#[derive(Debug, Clone, Copy)]
pub struct StepImpulseArgs {
    /// Worlds.
    pub nworld: u32,
    /// `q` width per world.
    pub nq: u32,
    /// `v` width per world.
    pub nv: u32,
    /// Servoed DOFs, or 0 when there is no PD pass.
    pub n_dofs: u32,
    /// Whether to run the PD pass at all.
    pub has_pd: u32,
    /// As [`PdArgs::has_tau`], for the PD pass this fused step runs.
    pub has_tau: u32,
    /// Timestep.
    pub dt: f32,
    /// Bodies per world.
    pub nbodies: u32,
    /// Gravity.
    pub gx: f32,
    /// Gravity.
    pub gy: f32,
    /// Gravity.
    pub gz: f32,
    /// Contact sweeps per step.
    pub sweeps: u32,
    /// Steps fused into this launch.
    pub nsteps: u32,
}

/// Scalar arguments of the integration pass.
#[derive(Debug, Clone, Copy)]
pub struct IntegrateArgs {
    /// Worlds.
    pub nworld: u32,
    /// `v` (and `q`) width per world.
    pub nv: u32,
    /// Timestep.
    pub dt: f32,
    /// Bodies per world.
    pub nbodies: u32,
}

/// Scalar arguments of the FK readout pass.
#[derive(Debug, Clone, Copy)]
pub struct FkArgs {
    /// Worlds.
    pub nworld: u32,
    /// `v` (and `q`) width per world.
    pub nv: u32,
    /// Bodies per world.
    pub nbodies: u32,
}

/// Scalar arguments of the observation pass.
#[derive(Debug, Clone, Copy)]
pub struct ObsArgs {
    /// Worlds.
    pub nworld: u32,
    /// `q` width per world.
    pub nq: u32,
    /// `v` width per world.
    pub nv: u32,
    /// Bodies per world.
    pub nbodies: u32,
    /// Features per row.
    pub n_in: u32,
    /// Float offset of this step's rows in the observation history.
    pub obs_off: u32,
    /// Per-world constants per world (the `wconst` row width); 0 when the
    /// spec reads none.
    pub n_wc: u32,
}

/// Scalar arguments of the policy pass.
#[derive(Debug, Clone, Copy)]
pub struct PolicyArgs {
    /// Worlds.
    pub nworld: u32,
    /// MLP input width.
    pub n_in: u32,
    /// MLP hidden width.
    pub n_h: u32,
    /// MLP output width (actions).
    pub n_out: u32,
    /// Servoed DOFs per world (the PD target row width).
    pub n_dofs: u32,
    /// How many of the `n_out` outputs are PD-target offsets. The rest —
    /// `n_out - n_pos` of them — are feedforward torques. Equal to `n_out`
    /// for a spec with no torque channel, and then nothing in the kernel
    /// past the position branch executes.
    pub n_pos: u32,
    /// Applied-action clamp, used when `has_clamp_slots` is 0.
    pub act_clamp: f32,
    /// Non-zero when the kernel should read the per-action-slot clamp
    /// buffer instead of the `act_clamp` scalar.
    pub has_clamp_slots: u32,
    /// AR(1) noise coefficient.
    pub rho: f32,
    /// Float offset of this step's rows in the observation history.
    pub obs_off: u32,
    /// Float offset of this step's rows in the action/log-prob history.
    pub out_off: u32,
}

/// Where the kernels run. Buffers are flat `f32` arrays; launches are the
/// passes with their arguments spelled out, so a backend is only the
/// plumbing and never the physics.
pub trait KernelBackend {
    /// A device-side `f32` buffer.
    type Buffer;

    /// A captured, replayable sequence of launches. `()` on backends that
    /// do not capture — see [`KernelBackend::supports_graphs`].
    type Graph;

    /// Human-readable name of the device the kernels run on.
    fn device_name(&self) -> String;

    /// The body count this backend's kernels were compiled for.
    ///
    /// Not a property of the hardware — a property of the module that was
    /// built. The NVRTC backend specialises it per model; the host C++
    /// backend's translation unit is compiled once by `build.rs` and is
    /// stuck at [`layout::DEFAULT_MAX_BODIES`]. [`BatchSim::with_backend`]
    /// checks the model against this, and sizes the step caches from it.
    fn max_bodies(&self) -> usize {
        layout::DEFAULT_MAX_BODIES
    }

    // ── Graph capture ─────────────────────────────────────────────────────
    //
    // A physics step is ~35 launches in impulse mode (PD, then a leading ABA
    // and `contact_sweeps` × [contact, ABA], then integrate), each a few
    // microseconds of work. At the world counts RL wants the step time is
    // flat in `nworld` — it is the per-launch cost, not the arithmetic. The
    // sequence is also fixed: every argument that changes between steps
    // lives in a device buffer already, and the scalars (world count, DOF
    // widths, `dt`, gravity) do not change between steps at all. That is
    // exactly the shape CUDA Graphs exist for: record the launches once,
    // replay the whole span with one call.
    //
    // Backends that cannot capture leave these at their defaults and
    // [`BatchSim`] executes the same call sequence directly, so the physics
    // is identical either way and the parity harness compares the same math.

    /// Whether this backend can capture launches into a replayable graph.
    fn supports_graphs(&self) -> bool {
        false
    }

    /// Start capturing launches instead of executing them. Every launch
    /// until [`KernelBackend::capture_end`] is recorded.
    fn capture_begin(&self) -> Result<(), String> {
        Err("this backend does not capture launch graphs".into())
    }

    /// Stop capturing and instantiate what was recorded.
    fn capture_end(&self) -> Result<Self::Graph, String> {
        Err("this backend does not capture launch graphs".into())
    }

    /// Replay a captured sequence.
    fn graph_launch(&self, _graph: &Self::Graph) -> Result<(), String> {
        Err("this backend does not capture launch graphs".into())
    }

    /// Allocate a zero-filled buffer of `len` floats.
    fn alloc(&self, len: usize) -> Result<Self::Buffer, String>;
    /// Copy `data` into the front of `buf`.
    fn upload(&self, buf: &mut Self::Buffer, data: &[f32]) -> Result<(), String>;
    /// Copy the whole of `buf` back to the host.
    fn download(&self, buf: &Self::Buffer) -> Result<Vec<f32>, String>;
    /// The address a launch would pass for `buf` — the device pointer on
    /// CUDA, the host pointer on the C mirror. Only a graph-safety
    /// assertion needs this: a buffer written between control steps has to
    /// keep the address a captured graph baked in, and the address is the
    /// only direct evidence of that.
    fn buffer_addr(&self, buf: &Self::Buffer) -> u64;
    /// Block until every launch so far has completed.
    fn synchronize(&self) -> Result<(), String>;
    /// Copy `len` floats of `buf` starting at `start` back to the host.
    fn download_range(
        &self,
        buf: &Self::Buffer,
        start: usize,
        len: usize,
    ) -> Result<Vec<f32>, String>;
    /// Device-to-device: copy the whole of `src` into `dst` at `dst_offset`,
    /// ordered after every launch so far and never blocking the host.
    fn copy(
        &self,
        src: &Self::Buffer,
        dst: &mut Self::Buffer,
        dst_offset: usize,
    ) -> Result<(), String>;

    /// PD servo pass: `nworld * n_dofs` threads.
    #[allow(clippy::too_many_arguments)]
    fn launch_pd(
        &self,
        args: PdArgs,
        dofs: &Self::Buffer,
        q: &Self::Buffer,
        v: &Self::Buffer,
        targets: &Self::Buffer,
        tau_ff: &Self::Buffer,
        ctrl: &mut Self::Buffer,
    ) -> Result<(), String>;

    /// Contact pass: `nworld` threads. `cparams` is the packed
    /// [`ContactParams`]; `hf_heights` the heightfield nodes (a placeholder
    /// when there is no terrain); `qdd` the free acceleration the impulse
    /// solve reads.
    #[allow(clippy::too_many_arguments)]
    fn launch_contact(
        &self,
        args: ContactArgs,
        cparams: &Self::Buffer,
        bodies: &Self::Buffer,
        geometry: &Self::Buffer,
        q: &Self::Buffer,
        v: &Self::Buffer,
        ext_forces: &mut Self::Buffer,
        contact_state: &mut Self::Buffer,
        hf_heights: &Self::Buffer,
        qdd: &Self::Buffer,
    ) -> Result<(), String>;

    /// ABA pass: `nworld` threads.
    #[allow(clippy::too_many_arguments)]
    fn launch_aba(
        &self,
        args: AbaArgs,
        bodies: &Self::Buffer,
        q: &Self::Buffer,
        v: &Self::Buffer,
        ctrl: &Self::Buffer,
        qdd: &mut Self::Buffer,
        ext_forces: &Self::Buffer,
    ) -> Result<(), String>;

    /// Integration pass: `nworld * nbodies` threads.
    fn launch_integrate(
        &self,
        args: IntegrateArgs,
        q: &mut Self::Buffer,
        v: &mut Self::Buffer,
        qdd: &Self::Buffer,
        bodies: &Self::Buffer,
    ) -> Result<(), String>;

    /// Whether this backend has the fused impulse step
    /// ([`KernelBackend::launch_step_impulse`]).
    fn supports_fused_step(&self) -> bool {
        false
    }

    /// Whether this backend has the FISSIONED stage passes
    /// ([`KernelBackend::launch_aba_c`], [`KernelBackend::launch_contact_c`]).
    fn supports_fission(&self) -> bool {
        false
    }

    /// ABA with the per-step cache in a global SoA buffer: `nworld` threads,
    /// `mode` 0 to build the cache and 1 to reuse it. Same arithmetic in the
    /// same order as [`KernelBackend::launch_aba`] — a building call IS an
    /// `launch_aba`, plus the stores.
    #[allow(clippy::too_many_arguments)]
    fn launch_aba_c(
        &self,
        _args: AbaArgs,
        _bodies: &Self::Buffer,
        _q: &Self::Buffer,
        _v: &Self::Buffer,
        _ctrl: &Self::Buffer,
        _qdd: &mut Self::Buffer,
        _ext_forces: &Self::Buffer,
        _aba_cache: &mut Self::Buffer,
        _mode: u32,
    ) -> Result<(), String> {
        Err("this backend has no fissioned ABA pass".into())
    }

    /// The contact pass with the FK/manifold cache in a global SoA buffer:
    /// `nworld` threads, `fk_mode` 1 to build and 2 to reuse (0 is the
    /// cacheless [`KernelBackend::launch_contact`]).
    #[allow(clippy::too_many_arguments)]
    fn launch_contact_c(
        &self,
        _args: ContactArgs,
        _cparams: &Self::Buffer,
        _bodies: &Self::Buffer,
        _geometry: &Self::Buffer,
        _q: &Self::Buffer,
        _v: &Self::Buffer,
        _ext_forces: &mut Self::Buffer,
        _contact_state: &mut Self::Buffer,
        _hf_heights: &Self::Buffer,
        _qdd: &Self::Buffer,
        _fk_cache: &mut Self::Buffer,
        _fk_mode: u32,
    ) -> Result<(), String> {
        Err("this backend has no fissioned contact pass".into())
    }

    /// The whole impulse-mode step, `nworld` threads: PD, ABA, `sweeps` x
    /// [contact, ABA], integrate, `nsteps` times. Same arithmetic in the
    /// same order as the separate launches, with the ABA factorisation
    /// carried across the sweeps of a step.
    #[allow(clippy::too_many_arguments)]
    fn launch_step_impulse(
        &self,
        _args: StepImpulseArgs,
        _pd_dofs: &Self::Buffer,
        _targets: &Self::Buffer,
        _tau_ff: &Self::Buffer,
        _cparams: &Self::Buffer,
        _bodies: &Self::Buffer,
        _geometry: &Self::Buffer,
        _hf_heights: &Self::Buffer,
        _q: &mut Self::Buffer,
        _v: &mut Self::Buffer,
        _ctrl: &mut Self::Buffer,
        _qdd: &mut Self::Buffer,
        _ext_forces: &mut Self::Buffer,
        _contact_state: &mut Self::Buffer,
    ) -> Result<(), String> {
        Err("this backend has no fused impulse step".into())
    }

    /// FK readout pass: `nworld` threads, `XF_STRIDE` floats per body.
    fn launch_fk(
        &self,
        args: FkArgs,
        bodies: &Self::Buffer,
        q: &Self::Buffer,
        v: &Self::Buffer,
        xforms: &mut Self::Buffer,
    ) -> Result<(), String>;

    /// Observation pass: `nworld` threads, one op-table row each.
    #[allow(clippy::too_many_arguments)]
    fn launch_obs(
        &self,
        args: ObsArgs,
        ops: &Self::Buffer,
        aux: &Self::Buffer,
        com: &Self::Buffer,
        wconst: &Self::Buffer,
        q: &Self::Buffer,
        v: &Self::Buffer,
        xforms: &Self::Buffer,
        obs: &mut Self::Buffer,
    ) -> Result<(), String>;

    /// Policy pass: `nworld` threads — MLP, Gaussian sample, PD targets.
    #[allow(clippy::too_many_arguments)]
    fn launch_policy(
        &self,
        args: PolicyArgs,
        weights: &Self::Buffer,
        stdv: &Self::Buffer,
        in_noise: &Self::Buffer,
        obs: &mut Self::Buffer,
        rng: &mut Self::Buffer,
        z: &mut Self::Buffer,
        act_slots: &Self::Buffer,
        act_clamp_slots: &Self::Buffer,
        tau_slots: &Self::Buffer,
        tau_scale: &Self::Buffer,
        base_targets: &Self::Buffer,
        targets: &mut Self::Buffer,
        tau_ff: &mut Self::Buffer,
        out: &mut Self::Buffer,
    ) -> Result<(), String>;
}

struct ContactPass<B: KernelBackend> {
    geometry: B::Buffer,
    params: ContactParams,
    params_buf: B::Buffer,
    hf_buf: B::Buffer,
    /// Node capacity of `hf_buf`; `set_heightfield` may not outgrow it.
    hf_capacity: usize,
    collidable_bodies: usize,
}

struct PdPass<B: KernelBackend> {
    dofs: B::Buffer,
    targets: B::Buffer,
    /// The feedforward-torque row the policy pass writes and the PD pass
    /// sums inside the effort clamp: `nworld * n_dofs` newton-metres in
    /// registration order. Allocated with the pass and zeroed, so a spec
    /// with no torque channel hands the kernels a live pointer they never
    /// read (`has_tau` is 0) rather than a null one.
    tau_ff: B::Buffer,
    n_dofs: usize,
    /// Set by `enable_policy` when the spec declares a torque channel.
    /// Nothing else writes `tau_ff`, so it is the single switch.
    has_tau: bool,
}

struct PolicyPass<B: KernelBackend> {
    spec: PolicySpec,
    ops: B::Buffer,
    /// Variable-length payload for the reducing observation ops (see
    /// [`crate::policy_pipeline::pack_obs_ops`]).
    obs_aux: B::Buffer,
    /// `[mass, com.x, com.y, com.z]` per body — the mass distribution
    /// `ObsOp::ComOverSupport` reduces over, uploaded once with the spec.
    com: B::Buffer,
    /// `[nworld][spec.n_world_consts()]`, the per-world constant row
    /// `ObsOp::WorldSlot` / `ObsOp::QMinusWorld` read. Allocated once with
    /// the spec and never re-allocated while the spec stands, so its
    /// address survives a graph capture; `set_policy_world_consts` writes
    /// it between control steps.
    world_consts: B::Buffer,
    weights: B::Buffer,
    stdv: B::Buffer,
    in_noise: B::Buffer,
    rng: B::Buffer,
    z: B::Buffer,
    act_slots: B::Buffer,
    act_clamp_slots: B::Buffer,
    /// The torque channel's PD slot per torque output, and the newton-metres
    /// one unit of it commands. Length-1 placeholders when the spec declares
    /// no channel (`PolicyArgs::n_pos == n_out`, so neither is read).
    tau_slots: B::Buffer,
    tau_scale: B::Buffer,
    base_targets: B::Buffer,
    /// `[history_steps][nworld][n_in]`, written by the observation pass and
    /// noised in place by the policy pass.
    obs_hist: B::Buffer,
    /// `[history_steps][nworld][n_out + 1]`: actions then log-prob.
    out_hist: B::Buffer,
}

/// Everything about a captured step span that must still hold for the
/// recording to be a correct replay. The capture bakes in buffer addresses,
/// launch widths and the scalar arguments; anything that moves one of those
/// makes the graph stale, and the epoch covers the rest (a re-allocated
/// buffer, a changed `dt`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GraphKey {
    /// Physics steps recorded in this graph.
    steps: usize,
    /// Contact sweeps at capture time — [`BatchSim::contact_sweeps`] is
    /// public, so it is compared rather than hooked.
    sweeps: usize,
    /// Bumped by anything that re-allocates a buffer the step reads or
    /// changes a scalar the step bakes in.
    epoch: u64,
}

struct StepGraph<B: KernelBackend> {
    key: GraphKey,
    exec: B::Graph,
}

struct HistoryPass<B: KernelBackend> {
    q: B::Buffer,
    v: B::Buffer,
    slots: usize,
}

/// Batched simulator over a [`KernelBackend`].
///
/// Same surface as [`crate::GpuBatchSimulator`] — build one from a `Model`
/// and a world count, optionally enable ground contact and PD servos, then
/// `load_states` / `set_controls` / `step` / `readback_states`. Errors are
/// `String`s, as in the wgpu path.
pub struct BatchSim<B: KernelBackend> {
    backend: B,
    /// The physics model.
    pub model: Model,
    /// Number of parallel worlds.
    pub nworld: usize,
    bodies: B::Buffer,
    q: B::Buffer,
    v: B::Buffer,
    ctrl: B::Buffer,
    qdd: B::Buffer,
    ext_forces: B::Buffer,
    contact_state: B::Buffer,
    contact: Option<ContactPass<B>>,
    pd: Option<PdPass<B>>,
    /// FK readout buffer, `[nworld][nbodies][XF_STRIDE]`.
    kin: Option<B::Buffer>,
    policy: Option<PolicyPass<B>>,
    history: Option<HistoryPass<B>>,
    /// Contact sweeps per step in impulse mode; see
    /// [`crate::GpuBatchSimulator::contact_sweeps`].
    pub contact_sweeps: usize,
    /// The most recently captured step span, replayed while its key holds.
    graph: Option<StepGraph<B>>,
    /// See [`GraphKey::epoch`].
    graph_epoch: u64,
    /// Whether to capture at all. Defaults to on where the backend supports
    /// it; `PHYZ_CUDA_GRAPHS=0` and [`BatchSim::set_graphs_enabled`] turn it
    /// off, which is also how the determinism test gets its reference.
    graphs_enabled: bool,
    /// Whether an impulse step goes out as one fused launch where the
    /// backend has one. `PHYZ_FUSED_STEP=0` and
    /// [`BatchSim::set_fused_step_enabled`] turn it off — which is how the
    /// parity test gets its unfused reference.
    fused_enabled: bool,
    /// How an impulse step is issued; see [`StepMode`].
    step_mode: StepMode,
    /// Per-world ABA cache for the fissioned path, `[ABA_CACHE_FLOATS][nworld]`
    /// — world index fastest, so the stage kernels' loads coalesce. Allocated
    /// once, on the first fissioned step, and graph-stable thereafter.
    aba_cache: Option<B::Buffer>,
    /// Per-world FK/manifold cache for the fissioned path,
    /// `[FK_CACHE_FLOATS][nworld]`.
    fk_cache: Option<B::Buffer>,
}

/// How an impulse-mode step reaches the device.
///
/// All three run the same sequence — PD, a leading ABA, `sweeps` x
/// [contact, ABA], integrate — with the same arithmetic in the same order,
/// and are bit-identical to each other. They differ only in where the two
/// per-step caches live.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum StepMode {
    /// One launch per step (per control period under `step_many`), both
    /// caches on the thread's local stack. 255 registers and a 30 KB frame
    /// with spills — and, on the humanoid ruler, still the fastest of the
    /// three. The default. `PHYZ_STEP_MODE=fused`.
    Fused,
    /// The stage kernels, with both caches in global structure-of-arrays
    /// buffers: ~34 launches per step (free under CUDA Graphs) and a frame of
    /// 2.0–2.9 KB with no spills instead of 30 KB with them.
    ///
    /// **Measured slower**, and the reason is the useful part: at 4096 worlds
    /// a one-thread-per-world grid is 64 blocks over 128 SMs, so occupancy is
    /// bounded by the GRID, not by the register file — cutting the frame 11x
    /// buys no extra resident warps, while every sweep now pays a dependent
    /// global round-trip for the cache. See the roofline table in the PR.
    /// Kept as an opt-in (`PHYZ_STEP_MODE=fission`) because the SoA cache is
    /// what a warp-per-world rewrite needs to exist at all.
    Fission,
    /// The original per-pass launches with no cache at all: every sweep
    /// refactorises. The slowest, kept as the reference.
    /// `PHYZ_STEP_MODE=unfused`.
    Unfused,
}

fn default_step_mode() -> StepMode {
    match std::env::var("PHYZ_STEP_MODE").as_deref() {
        Ok("fused") => return StepMode::Fused,
        Ok("fission") => return StepMode::Fission,
        Ok("unfused") | Ok("legacy") => return StepMode::Unfused,
        _ => {}
    }
    // Back-compat with the flag #84 shipped: `PHYZ_FUSED_STEP=1` pins the
    // fused kernel, `=0` pins the original unfused sequence.
    match std::env::var("PHYZ_FUSED_STEP").as_deref() {
        Ok("0") | Ok("off") => StepMode::Unfused,
        _ => StepMode::Fused,
    }
}

/// `mode` for [`KernelBackend::launch_aba_c`]: build the cache. Mirrors
/// `PHYZ_ABA_BUILD` in the kernels.
pub const ABA_MODE_BUILD: u32 = 0;
/// `mode` for [`KernelBackend::launch_aba_c`]: reuse it. `PHYZ_ABA_REUSE`.
pub const ABA_MODE_REUSE: u32 = 1;
/// `fk_mode` for [`KernelBackend::launch_contact_c`]: build the FK and
/// manifold cache. Mirrors `PHYZ_FK_BUILD`.
pub const FK_MODE_BUILD: u32 = 1;
/// `fk_mode` for [`KernelBackend::launch_contact_c`]: reuse it.
/// `PHYZ_FK_REUSE`.
pub const FK_MODE_REUSE: u32 = 2;

/// The CUDA batch simulator: [`BatchSim`] on [`CudaBackend`].
#[cfg(feature = "cuda")]
pub type CudaBatchSimulator = BatchSim<CudaBackend>;

/// The host-check simulator: [`BatchSim`] on [`HostBackend`].
#[cfg(feature = "cuda-host")]
pub type HostBatchSimulator = BatchSim<HostBackend>;

#[cfg(feature = "cuda")]
impl CudaBatchSimulator {
    /// Create a CUDA batch simulator on device 0.
    ///
    /// Returns `Err` when there is no CUDA driver, no device, or NVRTC
    /// rejects the kernels — never panics, so callers can fall back to the
    /// CPU path the way they do when wgpu finds no adapter.
    pub fn new(model: Model, nworld: usize) -> Result<Self, String> {
        Self::on_device(model, nworld, 0)
    }

    /// Create a CUDA batch simulator on a specific device ordinal.
    /// The kernels are compiled for this model's own body count, so there is
    /// no cap to run into — a 34-body rig (K1 + a faithful board) compiles a
    /// 34-body module.
    pub fn on_device(model: Model, nworld: usize, ordinal: usize) -> Result<Self, String> {
        let backend = CudaBackend::with_max_bodies(ordinal, model.nbodies())?;
        Self::with_backend(backend, model, nworld)
    }
}

#[cfg(feature = "cuda-host")]
impl HostBatchSimulator {
    /// Create a simulator that runs the CUDA C kernels on the host CPU.
    pub fn new(model: Model, nworld: usize) -> Result<Self, String> {
        Self::with_backend(HostBackend, model, nworld)
    }
}

impl<B: KernelBackend> BatchSim<B> {
    /// Build the simulator on an existing backend.
    pub fn with_backend(backend: B, model: Model, nworld: usize) -> Result<Self, String> {
        let nb = model.nbodies();
        let mb = backend.max_bodies();
        if nb > mb {
            // Not a fixed ceiling any more: it means this backend's module
            // was built for a narrower model than the one handed over.
            // `CudaBatchSimulator::on_device` compiles for the model, so this
            // only fires on a hand-built backend or the `cuda-host` path.
            return Err(format!(
                "model has {nb} bodies but this backend's kernels were compiled for {mb} per \
                 world ({} bytes of per-thread private storage); rebuild the backend with \
                 CudaBackend::with_max_bodies({nb}) — {nb} bodies would need {} bytes",
                layout::private_bytes_per_world(mb),
                layout::private_bytes_per_world(nb),
            ));
        }
        if model.nq != model.nv {
            return Err(format!(
                "kernels assume nq == nv (exponential-coordinate ball/free joints); model has nq {} nv {}",
                model.nq, model.nv
            ));
        }
        let mut bodies = backend.alloc((nb * layout::BODY_STRIDE).max(1))?;
        backend.upload(&mut bodies, &pack_bodies(&model))?;
        let q = backend.alloc((nworld * model.nq).max(1))?;
        let v = backend.alloc((nworld * model.nv).max(1))?;
        let ctrl = backend.alloc((nworld * model.nv).max(1))?;
        let qdd = backend.alloc((nworld * model.nv).max(1))?;
        let ext_forces = backend.alloc((nworld * nb * 6).max(1))?;
        let contact_state = backend.alloc((nworld * nb * CONTACT_STATE_STRIDE).max(1))?;
        Ok(Self {
            backend,
            model,
            nworld,
            bodies,
            q,
            v,
            ctrl,
            qdd,
            ext_forces,
            contact_state,
            contact: None,
            pd: None,
            kin: None,
            policy: None,
            history: None,
            contact_sweeps: default_contact_sweeps(),
            graph: None,
            graph_epoch: 0,
            fused_enabled: !matches!(
                std::env::var("PHYZ_FUSED_STEP").as_deref(),
                Ok("0") | Ok("off")
            ),
            step_mode: default_step_mode(),
            aba_cache: None,
            fk_cache: None,
            graphs_enabled: !matches!(
                std::env::var("PHYZ_CUDA_GRAPHS").as_deref(),
                Ok("0") | Ok("off") | Ok("false")
            ),
        })
    }

    /// Turn launch-graph capture on or off. On by default where the backend
    /// supports it; with it off every step issues its launches one at a
    /// time. The arithmetic is the same either way — this is a performance
    /// switch and the reference path for the replay-determinism test.
    pub fn set_graphs_enabled(&mut self, on: bool) {
        self.graphs_enabled = on;
        if !on {
            self.graph = None;
        }
    }

    /// Turn the fused impulse step on or off. On by default where the
    /// backend has one; the unfused sequence is the same arithmetic.
    pub fn set_fused_step_enabled(&mut self, on: bool) {
        self.set_step_mode(if on {
            StepMode::Fused
        } else {
            StepMode::Unfused
        });
    }

    /// Choose how an impulse step is issued. The three modes are
    /// bit-identical; this is a performance switch and the A/B the roofline
    /// example drives. See [`StepMode`].
    pub fn set_step_mode(&mut self, mode: StepMode) {
        if self.step_mode != mode {
            self.step_mode = mode;
            self.fused_enabled = mode == StepMode::Fused;
            self.invalidate_graph();
        }
    }

    /// The current [`StepMode`].
    pub fn step_mode(&self) -> StepMode {
        self.step_mode
    }

    /// Whether an impulse step currently goes out as the fissioned stage
    /// kernels with their caches in global memory.
    pub fn fission_enabled(&self) -> bool {
        self.step_mode == StepMode::Fission
            && self.backend.supports_fission()
            && self.impulse_sweeps() > 0
    }

    /// Allocate the fissioned path's caches if they are not there yet.
    /// Called before a capture, never inside one — the allocation would be
    /// uncaptured work and the addresses have to be stable for the replay.
    fn ensure_step_caches(&mut self) -> Result<(), String> {
        if !self.fission_enabled() {
            return Ok(());
        }
        if self.aba_cache.is_none() {
            let mb = self.backend.max_bodies();
            self.aba_cache = Some(self.backend.alloc(self.nworld * layout::aba_cache_floats(mb))?);
            self.fk_cache = Some(self.backend.alloc(self.nworld * layout::fk_cache_floats(mb))?);
            self.invalidate_graph();
        }
        Ok(())
    }

    /// Whether an impulse step currently goes out as one fused launch.
    pub fn fused_step_enabled(&self) -> bool {
        self.step_mode == StepMode::Fused
            && self.backend.supports_fused_step()
            && self.impulse_sweeps() > 0
    }

    /// Whether steps are currently replayed from a captured graph.
    pub fn graphs_enabled(&self) -> bool {
        self.graphs_enabled && self.backend.supports_graphs()
    }

    /// Discard any captured span. Called by everything that re-allocates a
    /// buffer the step sequence reads or changes a scalar it bakes in.
    fn invalidate_graph(&mut self) {
        self.graph = None;
        self.graph_epoch += 1;
    }

    /// Replace the model in place — same body count and DOF widths, new
    /// masses, inertias, joint placements or limits. What a domain-
    /// randomised rebuild of the same robot needs, without re-allocating a
    /// simulator: re-run `enable_contact_*` / `enable_pd_control` after it
    /// if their geometry or gains changed too.
    pub fn set_model(&mut self, model: Model) -> Result<(), String> {
        if model.nbodies() != self.model.nbodies()
            || model.nq != self.model.nq
            || model.nv != self.model.nv
        {
            return Err(format!(
                "set_model: shape changed ({} bodies, nq {}, nv {} vs {} bodies, nq {}, nv {}); build a new simulator",
                model.nbodies(),
                model.nq,
                model.nv,
                self.model.nbodies(),
                self.model.nq,
                self.model.nv
            ));
        }
        self.backend
            .upload(&mut self.bodies, &pack_bodies(&model))?;
        self.model = model;
        // `dt` and gravity are baked into the ABA launch arguments.
        self.invalidate_graph();
        Ok(())
    }

    /// The backend the kernels run on.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Enable ground contact with one global stiffness/damping.
    ///
    /// Same contract as [`crate::GpuBatchSimulator::enable_ground_contact`]:
    /// returns the collidable-body count and errors when it is zero.
    pub fn enable_ground_contact(
        &mut self,
        ground_height: f64,
        stiffness: f64,
        damping: f64,
        friction: f64,
    ) -> Result<usize, String> {
        self.enable_contact(
            GroundContactParams {
                ground_height,
                stiffness,
                damping,
                friction,
                ..Default::default()
            },
            None,
            &[],
            None,
        )
    }

    /// Enable ground contact with per-body penalty gains.
    ///
    /// Same contract as
    /// [`crate::GpuBatchSimulator::enable_ground_contact_per_body`].
    pub fn enable_ground_contact_per_body(
        &mut self,
        ground_height: f64,
        friction: f64,
        gains: &[BodyContactGains],
    ) -> Result<usize, String> {
        self.enable_ground_contact_with_plane(ground_height, friction, gains, &[])
    }

    /// [`Self::enable_ground_contact_per_body`] plus an optional
    /// body-attached contact plane. Same contract as
    /// [`crate::GpuBatchSimulator::enable_ground_contact_with_plane`].
    pub fn enable_ground_contact_with_plane(
        &mut self,
        ground_height: f64,
        friction: f64,
        gains: &[BodyContactGains],
        planes: &[BodyPlane],
    ) -> Result<usize, String> {
        self.enable_contact_terrain(ground_height, friction, gains, planes, None)
    }

    /// [`Self::enable_ground_contact_with_plane`] over heightfield terrain.
    /// Same contract as [`crate::GpuBatchSimulator::enable_contact_terrain`]:
    /// `ground_height` is ignored while a heightfield is loaded, and the node
    /// buffer is sized to this first field.
    pub fn enable_contact_terrain(
        &mut self,
        ground_height: f64,
        friction: f64,
        gains: &[BodyContactGains],
        planes: &[BodyPlane],
        heightfield: Option<&Heightfield>,
    ) -> Result<usize, String> {
        self.enable_contact(
            GroundContactParams {
                ground_height,
                stiffness: 0.0,
                damping: 0.0,
                friction,
                ..Default::default()
            },
            Some(gains),
            planes,
            heightfield,
        )
    }

    /// Enable contact as the velocity-level convex impulse solve. Same
    /// contract as [`crate::GpuBatchSimulator::enable_contact_impulse`].
    pub fn enable_contact_impulse(
        &mut self,
        ground_height: f64,
        friction: f64,
        gains: &[BodyContactGains],
        planes: &[BodyPlane],
        heightfield: Option<&Heightfield>,
    ) -> Result<usize, String> {
        let dt = self.model.dt;
        self.enable_contact(
            GroundContactParams {
                ground_height,
                friction,
                impulse_solve: true,
                solref_erp: GroundContactParams::solref_erp_from(0.02, 1.0, dt),
                ..Default::default()
            },
            Some(gains),
            planes,
            heightfield,
        )
    }

    fn enable_contact(
        &mut self,
        contact: GroundContactParams,
        gains: Option<&[BodyContactGains]>,
        planes: &[BodyPlane],
        heightfield: Option<&Heightfield>,
    ) -> Result<usize, String> {
        let (geom_data, collidable) =
            pack_contact_geometry(&self.model, &contact, gains, planes, heightfield)?;
        let mut geometry = self.backend.alloc(geom_data.len().max(1))?;
        self.backend.upload(&mut geometry, &geom_data)?;

        let params = ContactParams::pack(&self.model, self.nworld, &contact, planes, heightfield);
        let mut params_buf = self.backend.alloc(params.as_f32s().len())?;
        self.backend.upload(&mut params_buf, params.as_f32s())?;

        // Heightfield nodes; a single placeholder when there is no terrain,
        // since hf_nx == 0 routes the kernel around it.
        let hf_capacity = heightfield.map_or(1, |h| h.heights.len().max(1));
        let mut hf_buf = self.backend.alloc(hf_capacity)?;
        if let Some(h) = heightfield {
            self.backend.upload(&mut hf_buf, &h.heights)?;
        }

        self.contact = Some(ContactPass {
            geometry,
            params,
            params_buf,
            hf_buf,
            hf_capacity,
            collidable_bodies: collidable,
        });
        // Fresh geometry / params / heightfield buffers, and possibly a
        // different pass count (penalty vs impulse).
        self.invalidate_graph();
        Ok(collidable)
    }

    /// Replace the contact terrain in place. Same contract as
    /// [`crate::GpuBatchSimulator::set_heightfield`].
    pub fn set_heightfield(&mut self, hf: &Heightfield) -> Result<(), String> {
        let c = self
            .contact
            .as_mut()
            .ok_or("contact not enabled — call enable_contact_terrain first")?;
        validate_heightfield(hf)?;
        if hf.heights.len() > c.hf_capacity {
            return Err(format!(
                "heightfield has {} nodes but the buffer was sized for {}; \
                 enable contact with the largest grid first",
                hf.heights.len(),
                c.hf_capacity
            ));
        }
        c.params.set_heightfield(hf);
        self.backend.upload(&mut c.params_buf, c.params.as_f32s())?;
        self.backend.upload(&mut c.hf_buf, &hf.heights)
    }

    /// Enable PD position servos on the given DOFs.
    ///
    /// Same contract as [`crate::GpuBatchSimulator::enable_pd_control`].
    pub fn enable_pd_control(&mut self, dofs: &[PdDof]) -> Result<(), String> {
        check_pd_dofs(dofs, self.model.nq, self.model.nv)?;
        let dof_data = pack_pd_dofs(dofs);
        let mut dof_buf = self.backend.alloc(dof_data.len())?;
        self.backend.upload(&mut dof_buf, &dof_data)?;
        // Re-enabling with the same servo count (a gain re-draw) keeps the
        // target buffer — the policy pass holds no reference to it, but the
        // last targets stay valid across the swap.
        let n = (self.nworld * dofs.len()).max(1);
        let (targets, tau_ff, has_tau) = match self.pd.take() {
            Some(old) if old.n_dofs == dofs.len() => (old.targets, old.tau_ff, old.has_tau),
            _ => {
                // Zeroed on allocation, not merely on the first policy step:
                // a caller may drive the servos directly with
                // `set_position_targets` while a torque-channel policy is
                // enabled, and that path must read a defined row.
                let mut t = self.backend.alloc(n)?;
                self.backend.upload(&mut t, &vec![0.0f32; n])?;
                (self.backend.alloc(n)?, t, false)
            }
        };
        self.pd = Some(PdPass {
            dofs: dof_buf,
            targets,
            tau_ff,
            n_dofs: dofs.len(),
            has_tau,
        });
        // A fresh DOF buffer, and the PD pass may not have been in the
        // sequence at all before.
        self.invalidate_graph();
        Ok(())
    }

    /// Upload per-environment position targets for the PD servos.
    pub fn set_position_targets(&mut self, targets: &[Vec<f64>]) -> Result<(), String> {
        let pd = self
            .pd
            .as_mut()
            .ok_or("PD control not enabled — call enable_pd_control first")?;
        let data = pack_rows(targets, self.nworld, pd.n_dofs);
        self.backend.upload(&mut pd.targets, &data)
    }

    /// Upload initial states.
    pub fn load_states(&mut self, states: &[State]) {
        assert_eq!(states.len(), self.nworld);
        let (q, v, ctrl) = pack_states(states, self.model.nq, self.model.nv);
        self.backend.upload(&mut self.q, &q).expect("upload q");
        self.backend.upload(&mut self.v, &v).expect("upload v");
        self.backend
            .upload(&mut self.ctrl, &ctrl)
            .expect("upload ctrl");
    }

    /// Upload control inputs for all environments.
    pub fn set_controls(&mut self, controls: &[Vec<f64>]) {
        let data = pack_rows(controls, self.nworld, self.model.nv);
        self.backend
            .upload(&mut self.ctrl, &data)
            .expect("upload ctrl");
    }

    /// Run one simulation step (PD → contact → ABA → integration).
    pub fn step(&mut self) {
        self.try_step().expect("CUDA step failed");
    }

    /// [`Self::step`], returning launch errors instead of panicking.
    ///
    /// In impulse mode this is ONE launch of the fused step where the
    /// backend has it (see [`Self::set_fused_step_enabled`]); otherwise, and
    /// where the backend captures graphs, it replays a one-step recording
    /// rather than issuing the ~35 launches by hand. [`Self::step_many`]
    /// covers a whole control period in one launch either way.
    pub fn try_step(&mut self) -> Result<(), String> {
        self.step_many(1)
    }

    /// Run `n` simulation steps as one captured span.
    ///
    /// The launch sequence of a step is fixed and every argument that varies
    /// between steps already lives in a device buffer, so `n` steps can be
    /// recorded once and replayed with a single call — which is the whole
    /// point, since the step is bound by per-launch cost and not by the
    /// arithmetic. The recording is cached and reused until the world count,
    /// sweep count, model scalars or any buffer it reads change.
    ///
    /// Semantically identical to calling [`Self::try_step`] `n` times, and
    /// bit-identical on replay: the graph replays the same kernels on the
    /// same addresses with the same arguments.
    pub fn step_many(&mut self, n: usize) -> Result<(), String> {
        if n == 0 {
            return Ok(());
        }
        self.ensure_step_caches()?;
        if !self.graphs_enabled() {
            if self.fuses(n) {
                return self.issue_fused(n);
            }
            for _ in 0..n {
                self.issue_step()?;
            }
            return Ok(());
        }
        let key = GraphKey {
            steps: n,
            sweeps: self.contact_sweeps,
            epoch: self.graph_epoch,
        };
        if self.graph.as_ref().map(|g| g.key) != Some(key) {
            // Drop the stale recording before capturing: on backends that
            // hold a device-side instantiation this frees it first.
            self.graph = None;
            self.backend.capture_begin()?;
            let issued = if self.fuses(n) {
                self.issue_fused(n)
            } else {
                (0..n).try_for_each(|_| self.issue_step())
            };
            // End the capture even when a launch failed, or the stream stays
            // in capture mode and every later call fails with it.
            let captured = self.backend.capture_end();
            issued?;
            self.graph = Some(StepGraph {
                key,
                exec: captured?,
            });
        }
        let g = self.graph.as_ref().expect("step graph captured just above");
        self.backend.graph_launch(&g.exec)
    }

    /// Sweeps per step in impulse mode, 0 in penalty mode.
    fn impulse_sweeps(&self) -> usize {
        match &self.contact {
            Some(c) if c.params.solve_mode == 1 => self.contact_sweeps,
            _ => 0,
        }
    }

    /// Whether `n` steps can go out as one fused launch.
    fn fuses(&self, n: usize) -> bool {
        self.fused_step_enabled() && n <= u32::MAX as usize
    }

    /// `n` steps of the impulse sequence in ONE launch — see
    /// `step_impulse_thread` in the kernels. Identical arithmetic to `n`
    /// calls of [`Self::issue_step`], with the ABA factorisation reused
    /// across the sweeps of each step.
    fn issue_fused(&mut self, n: usize) -> Result<(), String> {
        let m = &self.model;
        let (n_dofs, has_pd, has_tau) = match &self.pd {
            Some(pd) => (pd.n_dofs as u32, 1u32, u32::from(pd.has_tau)),
            None => (0, 0, 0),
        };
        let args = StepImpulseArgs {
            nworld: self.nworld as u32,
            nq: m.nq as u32,
            nv: m.nv as u32,
            n_dofs,
            has_pd,
            has_tau,
            dt: m.dt as f32,
            nbodies: m.nbodies() as u32,
            gx: m.gravity.x as f32,
            gy: m.gravity.y as f32,
            gz: m.gravity.z as f32,
            sweeps: self.impulse_sweeps() as u32,
            nsteps: n as u32,
        };
        let c = self.contact.as_ref().expect("fused step needs contact");
        // Unused when `has_pd` is 0; the kernel never reads them, and the
        // bodies table is the one buffer guaranteed to exist and to be
        // borrowed immutably here.
        let (pd_dofs, targets, tau_ff) = match &self.pd {
            Some(pd) => (&pd.dofs, &pd.targets, &pd.tau_ff),
            None => (&self.bodies, &self.bodies, &self.bodies),
        };
        self.backend.launch_step_impulse(
            args,
            pd_dofs,
            targets,
            tau_ff,
            &c.params_buf,
            &self.bodies,
            &c.geometry,
            &c.hf_buf,
            &mut self.q,
            &mut self.v,
            &mut self.ctrl,
            &mut self.qdd,
            &mut self.ext_forces,
            &mut self.contact_state,
        )
    }

    /// One step's launches, issued directly. This is the sequence
    /// [`Self::step_many`] records.
    fn issue_step(&mut self) -> Result<(), String> {
        let m = &self.model;
        let nworld = self.nworld as u32;
        let nv = m.nv as u32;
        let nbodies = m.nbodies() as u32;
        let dt = m.dt as f32;

        if let Some(pd) = &self.pd {
            self.backend.launch_pd(
                PdArgs {
                    nworld,
                    nq: m.nq as u32,
                    nv,
                    n_dofs: pd.n_dofs as u32,
                    has_tau: u32::from(pd.has_tau),
                },
                &pd.dofs,
                &self.q,
                &self.v,
                &pd.targets,
                &pd.tau_ff,
                &mut self.ctrl,
            )?;
        }

        // Contact + ABA. Penalty mode: one contact pass, one ABA. Impulse
        // mode interleaves [contact, ABA] `contact_sweeps` times after a
        // leading ABA — the same sequencing as `GpuBatchSimulator::step`,
        // for the same reason (each sweep reads a `qdd` carrying the previous
        // sweep's impulses; that is the matrix-free Delassus application).
        let aba_args = AbaArgs {
            nworld,
            nv,
            dt,
            nbodies,
            gx: m.gravity.x as f32,
            gy: m.gravity.y as f32,
            gz: m.gravity.z as f32,
        };
        let sweeps = match &self.contact {
            Some(c) if c.params.solve_mode == 1 => self.contact_sweeps,
            _ => 0,
        };
        if sweeps > 0 && self.fission_enabled() {
            // The fissioned sequence: the same launches, with the ABA
            // factorisation and the clipped face manifold carried between
            // them in global SoA buffers instead of a 19.3 KB local frame.
            self.launch_aba_c(aba_args, ABA_MODE_BUILD)?;
            for w in 0..sweeps {
                let fk_mode = if w == 0 { FK_MODE_BUILD } else { FK_MODE_REUSE };
                self.launch_contact_c(nworld, fk_mode)?;
                self.launch_aba_c(aba_args, ABA_MODE_REUSE)?;
            }
        } else if sweeps > 0 {
            self.launch_aba(aba_args)?;
            for _ in 0..sweeps {
                self.launch_contact(nworld)?;
                self.launch_aba(aba_args)?;
            }
        } else {
            if self.contact.is_some() {
                self.launch_contact(nworld)?;
            }
            self.launch_aba(aba_args)?;
        }

        self.backend.launch_integrate(
            IntegrateArgs {
                nworld,
                nv,
                dt,
                nbodies,
            },
            &mut self.q,
            &mut self.v,
            &self.qdd,
            &self.bodies,
        )
    }

    fn launch_contact(&mut self, nworld: u32) -> Result<(), String> {
        let Some(c) = &self.contact else {
            return Ok(());
        };
        self.backend.launch_contact(
            ContactArgs { nworld },
            &c.params_buf,
            &self.bodies,
            &c.geometry,
            &self.q,
            &self.v,
            &mut self.ext_forces,
            &mut self.contact_state,
            &c.hf_buf,
            &self.qdd,
        )
    }

    fn launch_contact_c(&mut self, nworld: u32, fk_mode: u32) -> Result<(), String> {
        let Some(c) = &self.contact else {
            return Ok(());
        };
        let cache = self
            .fk_cache
            .as_mut()
            .ok_or("fissioned contact pass without an FK cache")?;
        self.backend.launch_contact_c(
            ContactArgs { nworld },
            &c.params_buf,
            &self.bodies,
            &c.geometry,
            &self.q,
            &self.v,
            &mut self.ext_forces,
            &mut self.contact_state,
            &c.hf_buf,
            &self.qdd,
            cache,
            fk_mode,
        )
    }

    fn launch_aba_c(&mut self, args: AbaArgs, mode: u32) -> Result<(), String> {
        let cache = self
            .aba_cache
            .as_mut()
            .ok_or("fissioned ABA pass without a cache")?;
        self.backend.launch_aba_c(
            args,
            &self.bodies,
            &self.q,
            &self.v,
            &self.ctrl,
            &mut self.qdd,
            &self.ext_forces,
            cache,
            mode,
        )
    }

    fn launch_aba(&mut self, args: AbaArgs) -> Result<(), String> {
        self.backend.launch_aba(
            args,
            &self.bodies,
            &self.q,
            &self.v,
            &self.ctrl,
            &mut self.qdd,
            &self.ext_forces,
        )
    }

    /// Download states.
    pub fn readback_states(&self) -> Vec<State> {
        self.backend.synchronize().expect("synchronize");
        let q = self.backend.download(&self.q).expect("download q");
        let v = self.backend.download(&self.v).expect("download v");
        unpack_states(&self.model, self.nworld, &q, &v)
    }

    /// Download per-body ground-contact state (`result[env][body]`).
    ///
    /// Errors when ground contact is not enabled.
    pub fn readback_contacts(&self) -> Result<Vec<Vec<BodyContactState>>, String> {
        if self.contact.is_none() {
            return Err(
                "ground contact not enabled — call enable_ground_contact first".to_string(),
            );
        }
        self.backend.synchronize()?;
        let data = self.backend.download(&self.contact_state)?;
        Ok(unpack_contacts(&data, self.nworld, self.model.nbodies()))
    }

    /// Number of bodies the contact pass can collide, or 0 if disabled.
    pub fn collidable_bodies(&self) -> usize {
        self.contact.as_ref().map_or(0, |c| c.collidable_bodies)
    }

    // ── FK readout ────────────────────────────────────────────────────────

    /// Allocate the FK readout buffer (`[nworld][nbodies][XF_STRIDE]`).
    /// Idempotent.
    pub fn enable_kinematics(&mut self) -> Result<(), String> {
        if self.kin.is_none() {
            let nb = self.model.nbodies();
            self.kin = Some(self.backend.alloc((self.nworld * nb * XF_STRIDE).max(1))?);
        }
        Ok(())
    }

    /// Run the FK readout pass over the current `q`/`v`.
    pub fn compute_kinematics(&mut self) -> Result<(), String> {
        let kin = self
            .kin
            .as_mut()
            .ok_or("kinematics not enabled — call enable_kinematics first")?;
        self.backend.launch_fk(
            FkArgs {
                nworld: self.nworld as u32,
                nv: self.model.nv as u32,
                nbodies: self.model.nbodies() as u32,
            },
            &self.bodies,
            &self.q,
            &self.v,
            kin,
        )
    }

    /// Download the FK readout, raw: `[nworld][nbodies][XF_STRIDE]` — see
    /// [`crate::policy_pipeline::XF_STRIDE`] for the row layout.
    pub fn readback_kinematics(&self) -> Result<Vec<f32>, String> {
        let kin = self.kin.as_ref().ok_or("kinematics not enabled")?;
        self.backend.synchronize()?;
        self.backend.download(kin)
    }

    // ── Policy pass ───────────────────────────────────────────────────────

    /// Enable the observation + policy pass. Needs PD control enabled (the
    /// actions land in its target rows). Re-enabling with the same widths
    /// and history length keeps the history buffers.
    pub fn enable_policy(&mut self, spec: PolicySpec) -> Result<(), String> {
        let n_dofs = self
            .pd
            .as_ref()
            .ok_or("policy pass needs PD control — call enable_pd_control first")?
            .n_dofs;
        spec.validate(self.model.nq, self.model.nv, self.model.nbodies(), n_dofs)?;
        self.enable_kinematics()?;
        let nworld = self.nworld;
        let (n_in, n_out) = (spec.n_in(), spec.n_out());

        let (op_table, aux_table) = pack_obs_ops(&spec.obs);
        let mut ops = self.backend.alloc((n_in * OBS_OP_STRIDE).max(1))?;
        self.backend.upload(&mut ops, &op_table)?;
        let mut obs_aux = self.backend.alloc(aux_table.len().max(1))?;
        if !aux_table.is_empty() {
            self.backend.upload(&mut obs_aux, &aux_table)?;
        }
        let com_table = pack_com_table(&self.model);
        let mut com = self.backend.alloc(com_table.len().max(1))?;
        if !com_table.is_empty() {
            self.backend.upload(&mut com, &com_table)?;
        }
        let mut in_noise = self.backend.alloc(n_in.max(1))?;
        self.backend.upload(
            &mut in_noise,
            &spec
                .input_noise
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<_>>(),
        )?;
        let mut act_slots = self.backend.alloc(n_out.max(1))?;
        self.backend.upload(
            &mut act_slots,
            &spec.act_slots.iter().map(|&s| s as f32).collect::<Vec<_>>(),
        )?;
        let mut act_clamp_slots = self.backend.alloc(n_out.max(1))?;
        self.backend.upload(
            &mut act_clamp_slots,
            &spec
                .clamp_row()
                .iter()
                .map(|&c| c as f32)
                .collect::<Vec<_>>(),
        )?;

        // The torque channel's two rows. Allocated at length 1 and never
        // read when the spec declares no channel, so the pointers the
        // kernel is handed are always live.
        let n_tau = spec.n_tau();
        let mut tau_slots = self.backend.alloc(n_tau.max(1))?;
        let mut tau_scale = self.backend.alloc(n_tau.max(1))?;
        if let Some(t) = &spec.tau {
            self.backend.upload(
                &mut tau_slots,
                &t.slots.iter().map(|&s| s as f32).collect::<Vec<_>>(),
            )?;
            self.backend.upload(
                &mut tau_scale,
                &t.scale.iter().map(|&s| s as f32).collect::<Vec<_>>(),
            )?;
        }
        // The one switch the PD pass reads. Set here rather than in
        // `enable_pd_control` because it is a property of the POLICY, and a
        // spec swapped from a torque one back to a position one must turn
        // the summand off again.
        let pd = self.pd.as_mut().expect("PD pass checked above");
        if pd.has_tau != (n_tau > 0) {
            pd.has_tau = n_tau > 0;
            // `has_tau` is a scalar the step sequence BAKES IN — a captured
            // graph carries whatever it was at capture, so a span recorded
            // before the channel came on would replay without the summand
            // and a policy would train twelve outputs that reach no joint.
            // Exactly what `invalidate_graph` is for.
            self.invalidate_graph();
        }

        let n_wc = spec.n_world_consts();
        let reuse = self.policy.take().filter(|p| {
            p.spec.n_in() == n_in
                && p.spec.hidden == spec.hidden
                && p.spec.n_out() == n_out
                && p.spec.history_steps == spec.history_steps
                && p.spec.n_world_consts() == n_wc
        });
        let (weights, stdv, rng, z, base_targets, obs_hist, out_hist, world_consts) = match reuse {
            Some(p) => (
                p.weights,
                p.stdv,
                p.rng,
                p.z,
                p.base_targets,
                p.obs_hist,
                p.out_hist,
                p.world_consts,
            ),
            None => (
                self.backend.alloc(spec.n_weights())?,
                self.backend.alloc(n_out)?,
                self.backend.alloc(nworld * 2)?,
                self.backend.alloc(nworld * n_out)?,
                self.backend.alloc(nworld * n_dofs)?,
                self.backend.alloc(spec.history_steps * nworld * n_in)?,
                self.backend
                    .alloc(spec.history_steps * nworld * (n_out + 1))?,
                self.backend.alloc((nworld * n_wc).max(1))?,
            ),
        };
        self.policy = Some(PolicyPass {
            spec,
            ops,
            obs_aux,
            com,
            weights,
            stdv,
            in_noise,
            rng,
            z,
            act_slots,
            act_clamp_slots,
            tau_slots,
            tau_scale,
            base_targets,
            obs_hist,
            out_hist,
            world_consts,
        });
        Ok(())
    }

    fn policy_parts(&mut self) -> Result<(&B, &mut PolicyPass<B>), String> {
        let p = self
            .policy
            .as_mut()
            .ok_or_else(|| "policy pass not enabled — call enable_policy first".to_string())?;
        Ok((&self.backend, p))
    }

    /// Upload the MLP weights, flat, in the kernel's order (see
    /// [`crate::policy_pipeline`]).
    pub fn set_policy_weights(&mut self, weights: &[f64]) -> Result<(), String> {
        let (backend, p) = self.policy_parts()?;
        if weights.len() != p.spec.n_weights() {
            return Err(format!(
                "policy expects {} weights, got {}",
                p.spec.n_weights(),
                weights.len()
            ));
        }
        let data: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
        backend.upload(&mut p.weights, &data)
    }

    /// Upload the per-action Gaussian standard deviations.
    pub fn set_policy_std(&mut self, std: &[f64]) -> Result<(), String> {
        let (backend, p) = self.policy_parts()?;
        if std.len() != p.spec.n_out() {
            return Err(format!(
                "policy has {} actions, got {} stds",
                p.spec.n_out(),
                std.len()
            ));
        }
        let data: Vec<f32> = std.iter().map(|&s| s as f32).collect();
        backend.upload(&mut p.stdv, &data)
    }

    /// Upload the per-world base PD targets the actions are added to
    /// (`rows[world][slot]`, registration order — as `set_position_targets`).
    pub fn set_policy_base_targets(&mut self, rows: &[Vec<f64>]) -> Result<(), String> {
        let n_dofs = self.pd.as_ref().map_or(0, |pd| pd.n_dofs);
        let nworld = self.nworld;
        let (backend, p) = self.policy_parts()?;
        let data = pack_rows(rows, nworld, n_dofs);
        backend.upload(&mut p.base_targets, &data)
    }

    /// Write the per-world constant rows the observation ops
    /// [`crate::policy_pipeline::ObsOp::WorldSlot`] and
    /// [`crate::policy_pipeline::ObsOp::QMinusWorld`] read: one row of
    /// [`PolicySpec::n_world_consts`] values per world, in world order.
    ///
    /// This is the one per-world table that is *meant* to change between
    /// control steps — a time-varying command, a per-world reference
    /// posture — so the buffer is allocated with the spec and written in
    /// place. The address never moves while the spec stands, which is what
    /// makes it safe to write inside a captured step span: the graph baked
    /// in the pointer, not the contents.
    ///
    /// Rows shorter than the spec's width are zero-filled and longer ones
    /// are truncated, as in
    /// [`BatchSim::set_policy_base_targets`]; a spec that reads no
    /// per-world constant accepts any rows and stores nothing.
    pub fn set_policy_world_consts(&mut self, rows: &[Vec<f64>]) -> Result<(), String> {
        let nworld = self.nworld;
        let (backend, p) = self.policy_parts()?;
        let n_wc = p.spec.n_world_consts();
        if n_wc == 0 {
            return Ok(());
        }
        let data = pack_rows(rows, nworld, n_wc);
        backend.upload(&mut p.world_consts, &data)
    }

    /// The address of the per-world constants buffer — what a captured
    /// graph bakes into its `phyz_obs` node. It must not change across
    /// [`BatchSim::set_policy_world_consts`] calls, or a replay would read
    /// a freed allocation; the tests assert exactly that.
    pub fn policy_world_consts_addr(&self) -> Result<u64, String> {
        let p = self.policy.as_ref().ok_or("policy pass not enabled")?;
        Ok(self.backend.buffer_addr(&p.world_consts))
    }

    /// The per-world constant rows, downloaded (`[world][slot]`).
    pub fn readback_policy_world_consts(&self) -> Result<Vec<f32>, String> {
        let p = self.policy.as_ref().ok_or("policy pass not enabled")?;
        let n_wc = p.spec.n_world_consts();
        if n_wc == 0 {
            return Ok(Vec::new());
        }
        self.backend.download(&p.world_consts)
    }

    /// Seed every world's random stream (see
    /// [`crate::policy_pipeline::world_seed`]) and reset the AR(1) noise
    /// state. Same seed, same device, same actions.
    pub fn seed_policy(&mut self, seed: u64) -> Result<(), String> {
        let nworld = self.nworld;
        let (backend, p) = self.policy_parts()?;
        let mut words = Vec::with_capacity(nworld * 2);
        for w in 0..nworld {
            let s = world_seed(seed, w);
            words.push(f32::from_bits((s & 0xffff_ffff) as u32));
            words.push(f32::from_bits((s >> 32) as u32));
        }
        let n_out = p.spec.n_out();
        backend.upload(&mut p.rng, &words)?;
        backend.upload(&mut p.z, &vec![0.0f32; nworld * n_out])
    }

    /// One control step's worth of the loop, on device: FK readout,
    /// observation row `step`, policy forward + sample into action row
    /// `step` and the PD targets. `step` indexes the history and must be
    /// below `history_steps`.
    pub fn run_policy(&mut self, step: usize) -> Result<(), String> {
        self.compute_kinematics()?;
        let nworld = self.nworld;
        let m = &self.model;
        let (nq, nv, nb) = (m.nq, m.nv, m.nbodies());
        let pd = self.pd.as_mut().ok_or("policy pass needs PD control")?;
        let p = self
            .policy
            .as_mut()
            .ok_or("policy pass not enabled — call enable_policy first")?;
        let kin = self.kin.as_ref().ok_or("kinematics not enabled")?;
        if step >= p.spec.history_steps {
            return Err(format!(
                "policy step {step} exceeds history_steps {}",
                p.spec.history_steps
            ));
        }
        let (n_in, n_h, n_out) = (p.spec.n_in(), p.spec.hidden, p.spec.n_out());
        let obs_off = step * nworld * n_in;
        let out_off = step * nworld * (n_out + 1);
        self.backend.launch_obs(
            ObsArgs {
                nworld: nworld as u32,
                nq: nq as u32,
                nv: nv as u32,
                nbodies: nb as u32,
                n_in: n_in as u32,
                obs_off: obs_off as u32,
                n_wc: p.spec.n_world_consts() as u32,
            },
            &p.ops,
            &p.obs_aux,
            &p.com,
            &p.world_consts,
            &self.q,
            &self.v,
            kin,
            &mut p.obs_hist,
        )?;
        self.backend.launch_policy(
            PolicyArgs {
                nworld: nworld as u32,
                n_in: n_in as u32,
                n_h: n_h as u32,
                n_out: n_out as u32,
                n_dofs: pd.n_dofs as u32,
                n_pos: p.spec.n_pos() as u32,
                act_clamp: p.spec.act_clamp as f32,
                has_clamp_slots: u32::from(p.spec.act_clamp_slots.is_some()),
                rho: p.spec.noise_rho as f32,
                obs_off: obs_off as u32,
                out_off: out_off as u32,
            },
            &p.weights,
            &p.stdv,
            &p.in_noise,
            &mut p.obs_hist,
            &mut p.rng,
            &mut p.z,
            &p.act_slots,
            &p.act_clamp_slots,
            &p.tau_slots,
            &p.tau_scale,
            &p.base_targets,
            &mut pd.targets,
            &mut pd.tau_ff,
            &mut p.out_hist,
        )
    }

    /// Download the observation and action history for control steps
    /// `steps` (a range below `history_steps`): `(obs, out)` with `obs`
    /// `[step][world][n_in]` and `out` `[step][world][n_out + 1]` (actions
    /// then log-prob), flattened.
    pub fn readback_policy_history(
        &self,
        steps: std::ops::Range<usize>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let p = self.policy.as_ref().ok_or("policy pass not enabled")?;
        if steps.end > p.spec.history_steps || steps.start > steps.end {
            return Err(format!(
                "policy history range {steps:?} outside 0..{}",
                p.spec.history_steps
            ));
        }
        let (n_in, n_out) = (p.spec.n_in(), p.spec.n_out());
        let n = steps.len();
        self.backend.synchronize()?;
        let obs = self.backend.download_range(
            &p.obs_hist,
            steps.start * self.nworld * n_in,
            n * self.nworld * n_in,
        )?;
        let out = self.backend.download_range(
            &p.out_hist,
            steps.start * self.nworld * (n_out + 1),
            n * self.nworld * (n_out + 1),
        )?;
        Ok((obs, out))
    }

    /// The policy pass's PD target rows, downloaded (`[world][slot]`).
    pub fn readback_targets(&self) -> Result<Vec<f32>, String> {
        let pd = self.pd.as_ref().ok_or("PD control not enabled")?;
        self.backend.synchronize()?;
        self.backend.download(&pd.targets)
    }

    /// The feedforward-torque rows the policy pass wrote, downloaded
    /// (`[world][slot]`, newton-metres, PD registration order). All zero
    /// unless the enabled spec declared a
    /// [`crate::policy_pipeline::TauChannel`].
    pub fn readback_tau_ff(&self) -> Result<Vec<f32>, String> {
        let pd = self.pd.as_ref().ok_or("PD control not enabled")?;
        self.backend.synchronize()?;
        self.backend.download(&pd.tau_ff)
    }

    /// The generalized force the last step's PD pass wrote, downloaded
    /// (`[world][nv]`). This is what ABA integrated, torque channel and
    /// effort clamp included — the only readout that can witness the clamp.
    pub fn readback_ctrl(&self) -> Result<Vec<f32>, String> {
        self.backend.synchronize()?;
        self.backend.download(&self.ctrl)
    }

    // ── State history ─────────────────────────────────────────────────────

    /// Keep `slots` snapshots of `q`/`v` on device, so a rollout can be
    /// read back once instead of once per control step. Re-enabling with
    /// the same slot count keeps the buffers.
    pub fn enable_state_history(&mut self, slots: usize) -> Result<(), String> {
        if slots == 0 {
            return Err("state history needs at least one slot".into());
        }
        if self.history.as_ref().is_some_and(|h| h.slots == slots) {
            return Ok(());
        }
        let m = &self.model;
        self.history = Some(HistoryPass {
            q: self.backend.alloc((slots * self.nworld * m.nq).max(1))?,
            v: self.backend.alloc((slots * self.nworld * m.nv).max(1))?,
            slots,
        });
        Ok(())
    }

    /// Snapshot the current `q`/`v` into history `slot` — a device copy
    /// ordered after every launch so far; the host does not wait.
    pub fn record_state(&mut self, slot: usize) -> Result<(), String> {
        let m = &self.model;
        let h = self
            .history
            .as_mut()
            .ok_or("state history not enabled — call enable_state_history first")?;
        if slot >= h.slots {
            return Err(format!("history slot {slot} exceeds {} slots", h.slots));
        }
        self.backend
            .copy(&self.q, &mut h.q, slot * self.nworld * m.nq)?;
        self.backend
            .copy(&self.v, &mut h.v, slot * self.nworld * m.nv)
    }

    /// Download history slots `slots` raw: `(q, v)` with `q` `[slot][world][nq]`
    /// and `v` `[slot][world][nv]`, flattened. Rebuild `State`s with
    /// [`crate::layout::unpack_states`] per slot, or read the slices in
    /// place — at thousands of worlds times hundreds of steps the `State`
    /// objects are the expensive part.
    pub fn readback_state_history(
        &self,
        slots: std::ops::Range<usize>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let h = self.history.as_ref().ok_or("state history not enabled")?;
        if slots.end > h.slots || slots.start > slots.end {
            return Err(format!("history range {slots:?} outside 0..{}", h.slots));
        }
        let m = &self.model;
        let n = slots.len();
        self.backend.synchronize()?;
        let q = self.backend.download_range(
            &h.q,
            slots.start * self.nworld * m.nq,
            n * self.nworld * m.nq,
        )?;
        let v = self.backend.download_range(
            &h.v,
            slots.start * self.nworld * m.nv,
            n * self.nworld * m.nv,
        )?;
        Ok((q, v))
    }
}
