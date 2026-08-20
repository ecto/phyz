//! CUDA backend for the training kernels: NVRTC-compiled, launched through
//! `cudarc`.
//!
//! Constructed either standalone with [`CudaTrainBackend::new`], or on the
//! context an existing [`super::nvrtc::CudaBackend`] already opened, so a
//! run that collects and updates on the same GPU does not open two contexts
//! and does not copy through the host between the halves.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::CompileOptions;

use super::train::{AdamCfg, NetDims, TRAIN_BLOCK, TRAIN_KERNEL_SOURCE, TrainBackend};

/// A CUDA device, stream and the nine compiled training kernels.
pub struct CudaTrainBackend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    fwd: CudaFunction,
    ppo_grad: CudaFunction,
    value_grad: CudaFunction,
    bwd_hidden: CudaFunction,
    grad_w_idx: CudaFunction,
    grad_w: CudaFunction,
    grad_b: CudaFunction,
    adam: CudaFunction,
    reduce_stats: CudaFunction,
}

fn err<E: std::fmt::Display>(what: &str) -> impl FnOnce(E) -> String + '_ {
    move |e| format!("{what}: {e}")
}

fn cfg(threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (threads.div_ceil(TRAIN_BLOCK).max(1), 1, 1),
        block_dim: (TRAIN_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl CudaTrainBackend {
    /// Open device `ordinal` and compile the training kernels.
    pub fn new(ordinal: usize) -> Result<Self, String> {
        // SAFETY: only dlopens candidate library names and drops the handles.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err("libcuda not found: no NVIDIA driver on this machine".into());
        }
        // SAFETY: as above.
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            return Err("libnvrtc not found: install the CUDA runtime (nvrtc)".into());
        }
        let ctx = CudaContext::new(ordinal).map_err(err("CUDA context"))?;
        Self::on_context(ctx)
    }

    /// Compile onto an already-open context — the collector's, normally.
    pub fn on_context(ctx: Arc<CudaContext>) -> Result<Self, String> {
        let stream = ctx.default_stream();
        let opts = CompileOptions {
            // Same as the simulation kernels: no fast-math, IEEE division and
            // sqrt, so a parity gap is a precision question and nothing else.
            use_fast_math: Some(false),
            prec_div: Some(true),
            prec_sqrt: Some(true),
            name: Some("phyz_train.cu".into()),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(TRAIN_KERNEL_SOURCE, opts)
            .map_err(err("NVRTC compile of phyz_train.cu"))?;
        let module = ctx.load_module(ptx).map_err(err("load PTX module"))?;
        let load = |name: &str| module.load_function(name).map_err(err("load kernel"));
        Ok(Self {
            fwd: load("phyz_train_fwd")?,
            ppo_grad: load("phyz_train_ppo_grad")?,
            value_grad: load("phyz_train_value_grad")?,
            bwd_hidden: load("phyz_train_bwd_hidden")?,
            grad_w_idx: load("phyz_train_grad_w_idx")?,
            grad_w: load("phyz_train_grad_w")?,
            grad_b: load("phyz_train_grad_b")?,
            adam: load("phyz_train_adam")?,
            reduce_stats: load("phyz_train_reduce_stats")?,
            ctx,
            stream,
        })
    }

    /// The underlying context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

impl TrainBackend for CudaTrainBackend {
    type F32 = CudaSlice<f32>;
    type F64 = CudaSlice<f64>;
    type U32 = CudaSlice<u32>;

    fn device_name(&self) -> String {
        self.ctx
            .name()
            .map(|n| format!("CUDA: {n}"))
            .unwrap_or_else(|_| "CUDA".into())
    }

    fn alloc_f32(&self, len: usize) -> Result<CudaSlice<f32>, String> {
        self.stream
            .alloc_zeros::<f32>(len.max(1))
            .map_err(err("cuMemAlloc"))
    }
    fn alloc_f64(&self, len: usize) -> Result<CudaSlice<f64>, String> {
        self.stream
            .alloc_zeros::<f64>(len.max(1))
            .map_err(err("cuMemAlloc"))
    }
    fn alloc_u32(&self, len: usize) -> Result<CudaSlice<u32>, String> {
        self.stream
            .alloc_zeros::<u32>(len.max(1))
            .map_err(err("cuMemAlloc"))
    }

    fn upload_f32(&self, buf: &mut CudaSlice<f32>, data: &[f32]) -> Result<(), String> {
        if data.len() > buf.len() {
            return Err(format!("upload of {} into {}", data.len(), buf.len()));
        }
        let mut view = buf.slice_mut(0..data.len());
        self.stream
            .memcpy_htod(data, &mut view)
            .map_err(err("memcpy_htod"))
    }
    fn upload_f64(&self, buf: &mut CudaSlice<f64>, data: &[f64]) -> Result<(), String> {
        if data.len() > buf.len() {
            return Err(format!("upload of {} into {}", data.len(), buf.len()));
        }
        let mut view = buf.slice_mut(0..data.len());
        self.stream
            .memcpy_htod(data, &mut view)
            .map_err(err("memcpy_htod"))
    }
    fn upload_u32(&self, buf: &mut CudaSlice<u32>, data: &[u32]) -> Result<(), String> {
        if data.len() > buf.len() {
            return Err(format!("upload of {} into {}", data.len(), buf.len()));
        }
        let mut view = buf.slice_mut(0..data.len());
        self.stream
            .memcpy_htod(data, &mut view)
            .map_err(err("memcpy_htod"))
    }

    fn download_f32(
        &self,
        buf: &CudaSlice<f32>,
        start: usize,
        len: usize,
    ) -> Result<Vec<f32>, String> {
        if start + len > buf.len() {
            return Err(format!("download {len} from {start} of {}", buf.len()));
        }
        let view = buf.slice(start..start + len);
        let out = self.stream.clone_dtoh(&view).map_err(err("memcpy_dtoh"))?;
        self.stream.synchronize().map_err(err("synchronize"))?;
        Ok(out)
    }

    fn download_f64(
        &self,
        buf: &CudaSlice<f64>,
        start: usize,
        len: usize,
    ) -> Result<Vec<f64>, String> {
        if start + len > buf.len() {
            return Err(format!("download {len} from {start} of {}", buf.len()));
        }
        let view = buf.slice(start..start + len);
        let out = self.stream.clone_dtoh(&view).map_err(err("memcpy_dtoh"))?;
        self.stream.synchronize().map_err(err("synchronize"))?;
        Ok(out)
    }

    fn synchronize(&self) -> Result<(), String> {
        self.stream.synchronize().map_err(err("synchronize"))
    }

    fn launch_fwd(
        &self,
        b: u32,
        dims: NetDims,
        w: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        h1: &mut CudaSlice<f32>,
        h2: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        let (n_in, n_h, n_out) = (dims.n_in as u32, dims.n_h as u32, dims.n_out as u32);
        // SAFETY: the pipeline sized every buffer for this launch.
        unsafe {
            self.stream
                .launch_builder(&self.fwd)
                .arg(&b)
                .arg(&n_in)
                .arg(&n_h)
                .arg(&n_out)
                .arg(w)
                .arg(x)
                .arg(idx)
                .arg(h1)
                .arg(h2)
                .arg(out)
                .launch(cfg(b))
        }
        .map(|_| ())
        .map_err(err("launch phyz_train_fwd"))
    }

    fn launch_ppo_grad(
        &self,
        b: u32,
        n_out: u32,
        clip: f32,
        entropy_coef: f32,
        mean: &CudaSlice<f32>,
        logstd: &CudaSlice<f32>,
        logstd_off: u32,
        act: &CudaSlice<f32>,
        logp_old: &CudaSlice<f32>,
        adv: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        dmean: &mut CudaSlice<f32>,
        dlogstd: &mut CudaSlice<f32>,
        stats: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        let ls = logstd.slice(logstd_off as usize..logstd_off as usize + n_out as usize);
        // SAFETY: as above.
        unsafe {
            self.stream
                .launch_builder(&self.ppo_grad)
                .arg(&b)
                .arg(&n_out)
                .arg(&clip)
                .arg(&entropy_coef)
                .arg(mean)
                .arg(&ls)
                .arg(act)
                .arg(logp_old)
                .arg(adv)
                .arg(idx)
                .arg(dmean)
                .arg(dlogstd)
                .arg(stats)
                .launch(cfg(b))
        }
        .map(|_| ())
        .map_err(err("launch phyz_train_ppo_grad"))
    }

    fn launch_value_grad(
        &self,
        b: u32,
        vdelta: f32,
        values: &CudaSlice<f32>,
        ret: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        dv: &mut CudaSlice<f32>,
        stats: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        // SAFETY: as above.
        unsafe {
            self.stream
                .launch_builder(&self.value_grad)
                .arg(&b)
                .arg(&vdelta)
                .arg(values)
                .arg(ret)
                .arg(idx)
                .arg(dv)
                .arg(stats)
                .launch(cfg(b))
        }
        .map(|_| ())
        .map_err(err("launch phyz_train_value_grad"))
    }

    fn launch_bwd_hidden(
        &self,
        b: u32,
        n_out: u32,
        n_h: u32,
        dout: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        w_off: u32,
        hact: &CudaSlice<f32>,
        dh: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        let off = w_off as usize;
        let wv = w.slice(off..off + (n_out * n_h) as usize);
        // SAFETY: as above.
        unsafe {
            self.stream
                .launch_builder(&self.bwd_hidden)
                .arg(&b)
                .arg(&n_out)
                .arg(&n_h)
                .arg(dout)
                .arg(&wv)
                .arg(hact)
                .arg(dh)
                .launch(cfg(b * n_h))
        }
        .map(|_| ())
        .map_err(err("launch phyz_train_bwd_hidden"))
    }

    fn launch_grad_w_idx(
        &self,
        b: u32,
        n_out: u32,
        n_in: u32,
        dout: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        gw: &mut CudaSlice<f32>,
        gw_off: u32,
    ) -> Result<(), String> {
        // SAFETY: as above.
        unsafe {
            self.stream
                .launch_builder(&self.grad_w_idx)
                .arg(&b)
                .arg(&n_out)
                .arg(&n_in)
                .arg(dout)
                .arg(x)
                .arg(idx)
                .arg(gw)
                .arg(&gw_off)
                .launch(cfg(n_out * n_in))
        }
        .map(|_| ())
        .map_err(err("launch phyz_train_grad_w_idx"))
    }

    fn launch_grad_w(
        &self,
        b: u32,
        n_out: u32,
        n_in: u32,
        dout: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        gw: &mut CudaSlice<f32>,
        gw_off: u32,
    ) -> Result<(), String> {
        // SAFETY: as above.
        unsafe {
            self.stream
                .launch_builder(&self.grad_w)
                .arg(&b)
                .arg(&n_out)
                .arg(&n_in)
                .arg(dout)
                .arg(x)
                .arg(gw)
                .arg(&gw_off)
                .launch(cfg(n_out * n_in))
        }
        .map(|_| ())
        .map_err(err("launch phyz_train_grad_w"))
    }

    fn launch_grad_b(
        &self,
        b: u32,
        n_out: u32,
        dout: &CudaSlice<f32>,
        gb: &mut CudaSlice<f32>,
        gb_off: u32,
    ) -> Result<(), String> {
        // SAFETY: as above.
        unsafe {
            self.stream
                .launch_builder(&self.grad_b)
                .arg(&b)
                .arg(&n_out)
                .arg(dout)
                .arg(gb)
                .arg(&gb_off)
                .launch(cfg(n_out))
        }
        .map(|_| ())
        .map_err(err("launch phyz_train_grad_b"))
    }

    fn launch_adam(
        &self,
        n: u32,
        c: AdamCfg,
        bc1: f64,
        bc2: f64,
        g: &CudaSlice<f32>,
        m: &mut CudaSlice<f64>,
        v: &mut CudaSlice<f64>,
        p: &mut CudaSlice<f64>,
        w: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        let (lr, b1, b2, eps, wd) = (c.lr, c.beta1, c.beta2, c.epsilon, c.weight_decay);
        // SAFETY: as above.
        unsafe {
            self.stream
                .launch_builder(&self.adam)
                .arg(&n)
                .arg(&lr)
                .arg(&b1)
                .arg(&b2)
                .arg(&eps)
                .arg(&wd)
                .arg(&bc1)
                .arg(&bc2)
                .arg(g)
                .arg(m)
                .arg(v)
                .arg(p)
                .arg(w)
                .launch(cfg(n))
        }
        .map(|_| ())
        .map_err(err("launch phyz_train_adam"))
    }

    fn launch_reduce_stats(
        &self,
        b: u32,
        stats: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        // SAFETY: as above.
        unsafe {
            self.stream
                .launch_builder(&self.reduce_stats)
                .arg(&b)
                .arg(stats)
                .arg(out)
                .launch(cfg(4))
        }
        .map(|_| ())
        .map_err(err("launch phyz_train_reduce_stats"))
    }
}
