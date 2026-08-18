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

use crate::contact_pipeline::{BodyContactGains, BodyContactState, GroundContactParams};
use crate::layout::{
    self, CONTACT_STATE_STRIDE, MAX_BODIES, check_pd_dofs, no_collidable_geometry_error,
    pack_bodies, pack_geometries, pack_pd_dofs, pack_rows, pack_states, unpack_contacts,
    unpack_states,
};
use crate::pd_pipeline::PdDof;
use phyz_model::{Model, State};

#[cfg(feature = "cuda-host")]
pub mod host;
#[cfg(feature = "cuda")]
pub mod nvrtc;

#[cfg(feature = "cuda-host")]
pub use host::HostBackend;
#[cfg(feature = "cuda")]
pub use nvrtc::CudaBackend;

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

/// Scalar arguments of the ground-contact pass.
#[derive(Debug, Clone, Copy)]
pub struct ContactArgs {
    /// Worlds.
    pub nworld: u32,
    /// Bodies per world.
    pub nbodies: u32,
    /// `v` (and `q`) width per world.
    pub nv: u32,
    /// Ground plane height.
    pub ground_height: f32,
    /// Timestep.
    pub dt: f32,
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

/// Where the kernels run. Buffers are flat `f32` arrays; launches are the
/// four passes with their arguments spelled out, so a backend is only the
/// plumbing and never the physics.
pub trait KernelBackend {
    /// A device-side `f32` buffer.
    type Buffer;

    /// Human-readable name of the device the kernels run on.
    fn device_name(&self) -> String;

    /// Allocate a zero-filled buffer of `len` floats.
    fn alloc(&self, len: usize) -> Result<Self::Buffer, String>;
    /// Copy `data` into the front of `buf`.
    fn upload(&self, buf: &mut Self::Buffer, data: &[f32]) -> Result<(), String>;
    /// Copy the whole of `buf` back to the host.
    fn download(&self, buf: &Self::Buffer) -> Result<Vec<f32>, String>;
    /// Block until every launch so far has completed.
    fn synchronize(&self) -> Result<(), String>;

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

    /// Ground contact pass: `nworld` threads.
    #[allow(clippy::too_many_arguments)]
    fn launch_contact(
        &self,
        args: ContactArgs,
        bodies: &Self::Buffer,
        geometry: &Self::Buffer,
        q: &Self::Buffer,
        v: &Self::Buffer,
        ext_forces: &mut Self::Buffer,
        contact_state: &mut Self::Buffer,
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
}

struct ContactPass<B: KernelBackend> {
    geometry: B::Buffer,
    ground_height: f32,
    collidable_bodies: usize,
}

struct PdPass<B: KernelBackend> {
    dofs: B::Buffer,
    targets: B::Buffer,
    n_dofs: usize,
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
        })
    }

    /// The backend the kernels run on.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Enable ground contact with one global stiffness/damping.
    ///
    /// Same contract as [`crate::GpuBatchSimulator::enable_ground_contact`]:
    /// returns the collidable-body count, errors when it is zero, and
    /// `friction` is accepted and ignored (the pass is normal-force only).
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
        if gains.len() != self.model.nbodies() {
            return Err(format!(
                "body_gains has {} entries but the model has {} bodies",
                gains.len(),
                self.model.nbodies()
            ));
        }
        self.enable_contact(
            GroundContactParams {
                ground_height,
                stiffness: 0.0,
                damping: 0.0,
                friction,
                ..Default::default()
            },
            Some(gains),
        )
    }

    fn enable_contact(
        &mut self,
        params: GroundContactParams,
        gains: Option<&[BodyContactGains]>,
    ) -> Result<usize, String> {
        let (geom_data, collidable) = pack_geometries(&self.model, &params, gains);
        if collidable == 0 {
            return Err(no_collidable_geometry_error(&self.model));
        }
        let mut geometry = self.backend.alloc(geom_data.len().max(1))?;
        self.backend.upload(&mut geometry, &geom_data)?;
        self.contact = Some(ContactPass {
            geometry,
            ground_height: params.ground_height as f32,
            collidable_bodies: collidable,
        });
        Ok(collidable)
    }

    /// Enable PD position servos on the given DOFs.
    ///
    /// Same contract as [`crate::GpuBatchSimulator::enable_pd_control`].
    pub fn enable_pd_control(&mut self, dofs: &[PdDof]) -> Result<(), String> {
        check_pd_dofs(dofs, self.model.nq, self.model.nv)?;
        let dof_data = pack_pd_dofs(dofs);
        let mut dof_buf = self.backend.alloc(dof_data.len())?;
        self.backend.upload(&mut dof_buf, &dof_data)?;
        let targets = self.backend.alloc((self.nworld * dofs.len()).max(1))?;
        self.pd = Some(PdPass {
            dofs: dof_buf,
            targets,
            n_dofs: dofs.len(),
        });
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
    pub fn try_step(&mut self) -> Result<(), String> {
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

        if let Some(c) = &self.contact {
            self.backend.launch_contact(
                ContactArgs {
                    nworld,
                    nbodies,
                    nv,
                    ground_height: c.ground_height,
                    dt,
                },
                &self.bodies,
                &c.geometry,
                &self.q,
                &self.v,
                &mut self.ext_forces,
                &mut self.contact_state,
            )?;
        }

        self.backend.launch_aba(
            AbaArgs {
                nworld,
                nv,
                dt,
                nbodies,
                gx: m.gravity.x as f32,
                gy: m.gravity.y as f32,
                gz: m.gravity.z as f32,
            },
            &self.bodies,
            &self.q,
            &self.v,
            &self.ctrl,
            &mut self.qdd,
            &self.ext_forces,
        )?;

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
}
