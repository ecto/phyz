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
use crate::gpu_batch_simulator::DEFAULT_CONTACT_SWEEPS;
use crate::layout::{
    self, CONTACT_STATE_STRIDE, MAX_BODIES, check_pd_dofs, pack_bodies, pack_pd_dofs, pack_rows,
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
    fn launch_pd(
        &self,
        args: PdArgs,
        dofs: &Self::Buffer,
        q: &Self::Buffer,
        v: &Self::Buffer,
        targets: &Self::Buffer,
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
        base_targets: &Self::Buffer,
        targets: &mut Self::Buffer,
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
    n_dofs: usize,
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
    weights: B::Buffer,
    stdv: B::Buffer,
    in_noise: B::Buffer,
    rng: B::Buffer,
    z: B::Buffer,
    act_slots: B::Buffer,
    act_clamp_slots: B::Buffer,
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
}

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
    pub fn on_device(model: Model, nworld: usize, ordinal: usize) -> Result<Self, String> {
        let backend = CudaBackend::new(ordinal)?;
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
        if nb > MAX_BODIES {
            return Err(format!(
                "model has {nb} bodies but the kernels hold at most {MAX_BODIES} per world"
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
            contact_sweeps: DEFAULT_CONTACT_SWEEPS,
            graph: None,
            graph_epoch: 0,
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
        let targets = match self.pd.take() {
            Some(old) if old.n_dofs == dofs.len() => old.targets,
            _ => self.backend.alloc((self.nworld * dofs.len()).max(1))?,
        };
        self.pd = Some(PdPass {
            dofs: dof_buf,
            targets,
            n_dofs: dofs.len(),
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
    /// Where the backend captures graphs this replays a one-step recording
    /// rather than issuing the ~35 launches by hand; [`Self::step_many`]
    /// records a longer span and is the cheaper way to run a control period.
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
        if !self.graphs_enabled() {
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
            let issued = (0..n).try_for_each(|_| self.issue_step());
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
                },
                &pd.dofs,
                &self.q,
                &self.v,
                &pd.targets,
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
        if sweeps > 0 {
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

        let reuse = self.policy.take().filter(|p| {
            p.spec.n_in() == n_in
                && p.spec.hidden == spec.hidden
                && p.spec.n_out() == n_out
                && p.spec.history_steps == spec.history_steps
        });
        let (weights, stdv, rng, z, base_targets, obs_hist, out_hist) = match reuse {
            Some(p) => (
                p.weights,
                p.stdv,
                p.rng,
                p.z,
                p.base_targets,
                p.obs_hist,
                p.out_hist,
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
            base_targets,
            obs_hist,
            out_hist,
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
            },
            &p.ops,
            &p.obs_aux,
            &p.com,
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
            &p.base_targets,
            &mut pd.targets,
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
