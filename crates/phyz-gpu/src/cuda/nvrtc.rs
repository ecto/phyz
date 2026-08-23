//! CUDA backend: NVRTC-compiled kernels launched through `cudarc`.
//!
//! `cudarc` is built with `dynamic-loading`, so `libcuda.so` and
//! `libnvrtc.so` are `dlopen`ed the first time a context is created — the
//! crate links and runs on machines without CUDA, and [`CudaBackend::new`]
//! reports the missing driver as an `Err`.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::CompileOptions;

use super::{
    AbaArgs, BLOCK_SIZE, ContactArgs, FkArgs, IntegrateArgs, KERNEL_SOURCE, KernelBackend, ObsArgs,
    PdArgs, PolicyArgs,
};

/// A CUDA device, stream and the four compiled kernels.
pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    pd: CudaFunction,
    contact: CudaFunction,
    aba: CudaFunction,
    integrate: CudaFunction,
    step_impulse: CudaFunction,
    fk: CudaFunction,
    obs: CudaFunction,
    policy: CudaFunction,
    /// False when the context could not give us a capturable stream and we
    /// fell back to the legacy default stream, which cannot be captured.
    capturable: bool,
}

fn err<E: std::fmt::Display>(what: &str) -> impl FnOnce(E) -> String + '_ {
    move |e| format!("{what}: {e}")
}

impl CudaBackend {
    /// Open device `ordinal` and compile the kernels with NVRTC.
    pub fn new(ordinal: usize) -> Result<Self, String> {
        // cudarc panics if the shared libraries are absent; probe first so a
        // machine without a driver gets an Err like a machine without a wgpu
        // adapter does.
        // SAFETY: only dlopens candidate library names and drops the handles.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err("libcuda not found: no NVIDIA driver on this machine".into());
        }
        // SAFETY: as above.
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            return Err(
                "libnvrtc not found: install the CUDA runtime (nvrtc) or set LD_LIBRARY_PATH"
                    .into(),
            );
        }
        let ctx = CudaContext::new(ordinal).map_err(err("CUDA context"))?;
        // Every launch, copy and allocation goes through one stream, so the
        // ordering is the default stream's — but it must be an explicitly
        // created stream rather than the legacy null stream, because the
        // legacy stream cannot be put into capture mode. If the context
        // will not give us one, fall back and run without graphs.
        //
        // Creating a second stream also puts cudarc into multi-stream mode,
        // where every buffer carries read/write events and every launch waits
        // on them. With one stream those events order nothing, and inside a
        // capture they are fatal: waiting on an event recorded outside the
        // capture is a dependency on uncaptured work, which the driver
        // rejects with CUDA_ERROR_STREAM_CAPTURE_ISOLATION. Turn the tracking
        // off before allocating anything — the safety condition is that no
        // buffer is touched from a second stream, and this backend has none.
        let (stream, capturable) = match ctx.new_stream() {
            Ok(s) => {
                // SAFETY: every allocation, copy and launch in this backend
                // goes through `s` and nothing else; the events cudarc would
                // record exist only to order streams that are not there.
                unsafe { ctx.disable_event_tracking() };
                (s, true)
            }
            Err(_) => (ctx.default_stream(), false),
        };

        let opts = CompileOptions {
            // Match the WGSL backends: no fast-math, keep IEEE division and
            // sqrt so parity with the CPU is a precision question only.
            use_fast_math: Some(false),
            prec_div: Some(true),
            prec_sqrt: Some(true),
            name: Some("phyz_kernels.cu".into()),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SOURCE, opts)
            .map_err(err("NVRTC compile of phyz_kernels.cu"))?;
        let module = ctx.load_module(ptx).map_err(err("load PTX module"))?;
        let load = |name: &str| module.load_function(name).map_err(err("load kernel"));

        Ok(Self {
            pd: load("phyz_pd")?,
            contact: load("phyz_contact")?,
            aba: load("phyz_aba")?,
            integrate: load("phyz_integrate")?,
            step_impulse: load("phyz_step_impulse")?,
            fk: load("phyz_fk")?,
            obs: load("phyz_obs")?,
            policy: load("phyz_policy")?,
            ctx,
            stream,
            capturable,
        })
    }

    /// The underlying context, for callers that want to share the device.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// The stream every launch and copy goes through.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

fn cfg(threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (threads.div_ceil(BLOCK_SIZE).max(1), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl KernelBackend for CudaBackend {
    type Buffer = CudaSlice<f32>;
    type Graph = CudaGraph;

    fn supports_graphs(&self) -> bool {
        self.capturable
    }

    fn capture_begin(&self) -> Result<(), String> {
        if !self.capturable {
            return Err("no capturable stream on this context".into());
        }
        // THREAD_LOCAL: only this thread's work on this stream is captured,
        // so a capture here cannot swallow launches another thread makes on
        // an unrelated stream, and unsafe calls elsewhere are not our error.
        self.stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(err("cuStreamBeginCapture"))
    }

    fn capture_end(&self) -> Result<Self::Graph, String> {
        // The instantiate flag is a no-op for us and is only passed because
        // cudarc's safe wrapper takes the enum, which has no zero variant:
        // AUTO_FREE_ON_LAUNCH governs graph-owned async allocations, and a
        // capture of nothing but kernel launches has none.
        let graph = self
            .stream
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(err("cuStreamEndCapture"))?
            .ok_or("stream capture produced an empty graph")?;
        // Push the graph's resources to the device now, so the first replay
        // is not paying the upload the timing is about to attribute to it.
        graph.upload().map_err(err("cuGraphUpload"))?;
        Ok(graph)
    }

    fn graph_launch(&self, graph: &Self::Graph) -> Result<(), String> {
        graph.launch().map_err(err("cuGraphLaunch"))
    }

    fn device_name(&self) -> String {
        self.ctx
            .name()
            .map(|n| format!("CUDA: {n}"))
            .unwrap_or_else(|_| "CUDA".into())
    }

    fn alloc(&self, len: usize) -> Result<Self::Buffer, String> {
        self.stream
            .alloc_zeros::<f32>(len)
            .map_err(err("cuMemAlloc"))
    }

    fn upload(&self, buf: &mut Self::Buffer, data: &[f32]) -> Result<(), String> {
        if data.len() > buf.len() {
            return Err(format!(
                "upload of {} floats into a {}-float buffer",
                data.len(),
                buf.len()
            ));
        }
        let mut view = buf.slice_mut(0..data.len());
        self.stream
            .memcpy_htod(data, &mut view)
            .map_err(err("memcpy_htod"))
    }

    fn download(&self, buf: &Self::Buffer) -> Result<Vec<f32>, String> {
        let out = self.stream.clone_dtoh(buf).map_err(err("memcpy_dtoh"))?;
        self.stream.synchronize().map_err(err("synchronize"))?;
        Ok(out)
    }

    fn synchronize(&self) -> Result<(), String> {
        self.stream.synchronize().map_err(err("synchronize"))
    }

    fn download_range(
        &self,
        buf: &Self::Buffer,
        start: usize,
        len: usize,
    ) -> Result<Vec<f32>, String> {
        if start + len > buf.len() {
            return Err(format!(
                "download of {start}..{} from a {}-float buffer",
                start + len,
                buf.len()
            ));
        }
        let view = buf.slice(start..start + len);
        let out = self.stream.clone_dtoh(&view).map_err(err("memcpy_dtoh"))?;
        self.stream.synchronize().map_err(err("synchronize"))?;
        Ok(out)
    }

    fn copy(
        &self,
        src: &Self::Buffer,
        dst: &mut Self::Buffer,
        dst_offset: usize,
    ) -> Result<(), String> {
        if dst_offset + src.len() > dst.len() {
            return Err(format!(
                "copy of {} floats at {dst_offset} into a {}-float buffer",
                src.len(),
                dst.len()
            ));
        }
        let mut view = dst.slice_mut(dst_offset..dst_offset + src.len());
        self.stream
            .memcpy_dtod(src, &mut view)
            .map_err(err("memcpy_dtod"))
    }

    fn launch_pd(
        &self,
        a: PdArgs,
        dofs: &Self::Buffer,
        q: &Self::Buffer,
        v: &Self::Buffer,
        targets: &Self::Buffer,
        ctrl: &mut Self::Buffer,
    ) -> Result<(), String> {
        let n = a.nworld * a.n_dofs;
        // SAFETY: argument order and types match `phyz_pd` in phyz_kernels.cu;
        // every buffer is sized by BatchSim for the indices the kernel touches.
        let r = unsafe {
            self.stream
                .launch_builder(&self.pd)
                .arg(&a.nworld)
                .arg(&a.nq)
                .arg(&a.nv)
                .arg(&a.n_dofs)
                .arg(dofs)
                .arg(q)
                .arg(v)
                .arg(targets)
                .arg(ctrl)
                .launch(cfg(n))
        };
        r.map(|_| ()).map_err(err("launch phyz_pd"))
    }

    fn launch_contact(
        &self,
        a: ContactArgs,
        cparams: &Self::Buffer,
        bodies: &Self::Buffer,
        geometry: &Self::Buffer,
        q: &Self::Buffer,
        v: &Self::Buffer,
        ext_forces: &mut Self::Buffer,
        contact_state: &mut Self::Buffer,
        hf_heights: &Self::Buffer,
        qdd: &Self::Buffer,
    ) -> Result<(), String> {
        // SAFETY: as in launch_pd, against `phyz_contact`.
        let r = unsafe {
            self.stream
                .launch_builder(&self.contact)
                .arg(cparams)
                .arg(bodies)
                .arg(geometry)
                .arg(q)
                .arg(v)
                .arg(ext_forces)
                .arg(contact_state)
                .arg(hf_heights)
                .arg(qdd)
                .launch(cfg(a.nworld))
        };
        r.map(|_| ()).map_err(err("launch phyz_contact"))
    }

    fn launch_aba(
        &self,
        a: AbaArgs,
        bodies: &Self::Buffer,
        q: &Self::Buffer,
        v: &Self::Buffer,
        ctrl: &Self::Buffer,
        qdd: &mut Self::Buffer,
        ext_forces: &Self::Buffer,
    ) -> Result<(), String> {
        // SAFETY: as in launch_pd, against `phyz_aba`.
        let r = unsafe {
            self.stream
                .launch_builder(&self.aba)
                .arg(&a.nworld)
                .arg(&a.nv)
                .arg(&a.dt)
                .arg(&a.nbodies)
                .arg(&a.gx)
                .arg(&a.gy)
                .arg(&a.gz)
                .arg(bodies)
                .arg(q)
                .arg(v)
                .arg(ctrl)
                .arg(qdd)
                .arg(ext_forces)
                .launch(cfg(a.nworld))
        };
        r.map(|_| ()).map_err(err("launch phyz_aba"))
    }

    fn launch_integrate(
        &self,
        a: IntegrateArgs,
        q: &mut Self::Buffer,
        v: &mut Self::Buffer,
        qdd: &Self::Buffer,
        bodies: &Self::Buffer,
    ) -> Result<(), String> {
        let n = a.nworld * a.nbodies;
        // SAFETY: as in launch_pd, against `phyz_integrate`.
        let r = unsafe {
            self.stream
                .launch_builder(&self.integrate)
                .arg(&a.nworld)
                .arg(&a.nv)
                .arg(&a.dt)
                .arg(&a.nbodies)
                .arg(q)
                .arg(v)
                .arg(qdd)
                .arg(bodies)
                .launch(cfg(n))
        };
        r.map(|_| ()).map_err(err("launch phyz_integrate"))
    }

    fn supports_fused_step(&self) -> bool {
        true
    }

    fn launch_step_impulse(
        &self,
        a: super::StepImpulseArgs,
        pd_dofs: &Self::Buffer,
        targets: &Self::Buffer,
        cparams: &Self::Buffer,
        bodies: &Self::Buffer,
        geometry: &Self::Buffer,
        hf_heights: &Self::Buffer,
        q: &mut Self::Buffer,
        v: &mut Self::Buffer,
        ctrl: &mut Self::Buffer,
        qdd: &mut Self::Buffer,
        ext_forces: &mut Self::Buffer,
        contact_state: &mut Self::Buffer,
    ) -> Result<(), String> {
        // SAFETY: as in launch_pd, against `phyz_step_impulse`.
        let r = unsafe {
            self.stream
                .launch_builder(&self.step_impulse)
                .arg(&a.nworld)
                .arg(&a.nq)
                .arg(&a.nv)
                .arg(&a.n_dofs)
                .arg(&a.has_pd)
                .arg(&a.dt)
                .arg(&a.nbodies)
                .arg(&a.gx)
                .arg(&a.gy)
                .arg(&a.gz)
                .arg(&a.sweeps)
                .arg(&a.nsteps)
                .arg(pd_dofs)
                .arg(targets)
                .arg(cparams)
                .arg(bodies)
                .arg(geometry)
                .arg(hf_heights)
                .arg(q)
                .arg(v)
                .arg(ctrl)
                .arg(qdd)
                .arg(ext_forces)
                .arg(contact_state)
                .launch(cfg(a.nworld))
        };
        r.map(|_| ()).map_err(err("launch phyz_step_impulse"))
    }

    fn launch_fk(
        &self,
        a: FkArgs,
        bodies: &Self::Buffer,
        q: &Self::Buffer,
        v: &Self::Buffer,
        xforms: &mut Self::Buffer,
    ) -> Result<(), String> {
        // SAFETY: as in launch_pd, against `phyz_fk`.
        let r = unsafe {
            self.stream
                .launch_builder(&self.fk)
                .arg(&a.nworld)
                .arg(&a.nv)
                .arg(&a.nbodies)
                .arg(bodies)
                .arg(q)
                .arg(v)
                .arg(xforms)
                .launch(cfg(a.nworld))
        };
        r.map(|_| ()).map_err(err("launch phyz_fk"))
    }

    fn launch_obs(
        &self,
        a: ObsArgs,
        ops: &Self::Buffer,
        aux: &Self::Buffer,
        com: &Self::Buffer,
        q: &Self::Buffer,
        v: &Self::Buffer,
        xforms: &Self::Buffer,
        obs: &mut Self::Buffer,
    ) -> Result<(), String> {
        // SAFETY: as in launch_pd, against `phyz_obs`.
        let r = unsafe {
            self.stream
                .launch_builder(&self.obs)
                .arg(&a.nworld)
                .arg(&a.nq)
                .arg(&a.nv)
                .arg(&a.nbodies)
                .arg(&a.n_in)
                .arg(&a.obs_off)
                .arg(ops)
                .arg(aux)
                .arg(com)
                .arg(q)
                .arg(v)
                .arg(xforms)
                .arg(obs)
                .launch(cfg(a.nworld))
        };
        r.map(|_| ()).map_err(err("launch phyz_obs"))
    }

    fn launch_policy(
        &self,
        a: PolicyArgs,
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
    ) -> Result<(), String> {
        // SAFETY: as in launch_pd, against `phyz_policy`.
        let r = unsafe {
            self.stream
                .launch_builder(&self.policy)
                .arg(&a.nworld)
                .arg(&a.n_in)
                .arg(&a.n_h)
                .arg(&a.n_out)
                .arg(&a.n_dofs)
                .arg(&a.act_clamp)
                .arg(&a.has_clamp_slots)
                .arg(&a.rho)
                .arg(&a.obs_off)
                .arg(&a.out_off)
                .arg(weights)
                .arg(stdv)
                .arg(in_noise)
                .arg(obs)
                .arg(rng)
                .arg(z)
                .arg(act_slots)
                .arg(act_clamp_slots)
                .arg(base_targets)
                .arg(targets)
                .arg(out)
                .launch(cfg(a.nworld))
        };
        r.map(|_| ()).map_err(err("launch phyz_policy"))
    }
}
