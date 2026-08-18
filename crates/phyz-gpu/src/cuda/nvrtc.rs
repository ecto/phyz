//! CUDA backend: NVRTC-compiled kernels launched through `cudarc`.
//!
//! `cudarc` is built with `dynamic-loading`, so `libcuda.so` and
//! `libnvrtc.so` are `dlopen`ed the first time a context is created — the
//! crate links and runs on machines without CUDA, and [`CudaBackend::new`]
//! reports the missing driver as an `Err`.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::CompileOptions;

use super::{
    AbaArgs, BLOCK_SIZE, ContactArgs, IntegrateArgs, KERNEL_SOURCE, KernelBackend, PdArgs,
};

/// A CUDA device, stream and the four compiled kernels.
pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    pd: CudaFunction,
    contact: CudaFunction,
    aba: CudaFunction,
    integrate: CudaFunction,
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
        let stream = ctx.default_stream();

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
            ctx,
            stream,
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
}
