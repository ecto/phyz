//! The PPO update pass, on device.
//!
//! `phyz_kernels.cu` moved the *collection* half of a reinforcement-learning
//! iteration onto the GPU. That left the *update* half — the minibatch loop
//! of a clipped-surrogate PPO step — on the CPU, in f64, where it grows with
//! the sample count and eventually becomes the clock: at 4096 worlds and
//! long episodes it is 600k-1.2M samples an iteration, and 40-60% of the
//! wall time of an iteration that collects on device.
//!
//! [`TrainPipeline`] runs that loop where the samples already are.
//!
//! # What it is, exactly
//!
//! A clipped-surrogate PPO update over two three-layer tanh MLPs — a
//! Gaussian-policy actor with a global learned `log_std`, and a critic — with
//! Adam on both and a per-minibatch KL brake. It is a port of the CPU
//! reference term for term, including the awkward details: the gradient
//! flowing only through whichever surrogate branch is the active one, the
//! Huber-clipped *value gradient* against an unclipped reported value loss,
//! and Adam's step counter shared across every parameter of a network so the
//! bias correction is one number per step.
//!
//! # What it is not
//!
//! It does not do GAE, advantage normalization, or the Huber delta's own
//! statistics. Those are one linear sweep of the batch, they are sequential
//! per episode, and they are nowhere near the clock; the host does them and
//! uploads the finished `adv` and `ret` columns. The pipeline's scope is the
//! `epochs x minibatches` loop and nothing else.
//!
//! # Determinism
//!
//! The minibatch order is the *host's*: the caller shuffles indices with its
//! own PRNG and hands them over, exactly as it would to a CPU loop, so a
//! same-seed run is comparable across backends. Every reduction inside a
//! kernel is a sequential double-accumulated loop in one thread, so the
//! device and the host walk of the same kernel source agree bit for bit and
//! the answer does not depend on block scheduling.

/// Threads per block for every training launch.
pub const TRAIN_BLOCK: u32 = 128;

/// The widest hidden layer the kernels will hold in a thread's local array.
/// Mirrors `TRAIN_MAX_H` in `cuda/phyz_train.cu`.
pub const TRAIN_MAX_H: usize = 256;
/// The widest action dimension a per-row thread will hold. Mirrors
/// `TRAIN_MAX_OUT` in the kernel source.
pub const TRAIN_MAX_OUT: usize = 64;

/// The CUDA source every training backend executes.
pub const TRAIN_KERNEL_SOURCE: &str = include_str!("../../cuda/phyz_train.cu");

/// A three-layer tanh MLP's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetDims {
    /// Input width.
    pub n_in: usize,
    /// Hidden width, both layers.
    pub n_h: usize,
    /// Output width.
    pub n_out: usize,
}

impl NetDims {
    /// Parameters in `W1, b1, W2, b2, W3, b3` order — tang's `Mlp::to_flat`.
    pub fn param_count(&self) -> usize {
        self.n_h * self.n_in
            + self.n_h
            + self.n_h * self.n_h
            + self.n_h
            + self.n_out * self.n_h
            + self.n_out
    }

    fn off_b1(&self) -> usize {
        self.n_h * self.n_in
    }
    fn off_w2(&self) -> usize {
        self.off_b1() + self.n_h
    }
    fn off_b2(&self) -> usize {
        self.off_w2() + self.n_h * self.n_h
    }
    fn off_w3(&self) -> usize {
        self.off_b2() + self.n_h
    }
    fn off_b3(&self) -> usize {
        self.off_w3() + self.n_out * self.n_h
    }
    /// Offset of the actor's `log_std` block, which follows `b3` in the
    /// actor's flat parameter array.
    pub fn off_logstd(&self) -> usize {
        self.off_b3() + self.n_out
    }
}

/// Adam's hyper-parameters, matching tang's `ModuleAdam`.
#[derive(Debug, Clone, Copy)]
pub struct AdamCfg {
    /// Learning rate.
    pub lr: f64,
    /// First-moment decay.
    pub beta1: f64,
    /// Second-moment decay.
    pub beta2: f64,
    /// Denominator floor.
    pub epsilon: f64,
    /// Decoupled (AdamW) weight decay; `0.0` for plain Adam.
    pub weight_decay: f64,
}

impl Default for AdamCfg {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// The PPO objective's constants for one update.
#[derive(Debug, Clone, Copy)]
pub struct PpoUpdateCfg {
    /// Surrogate clip range.
    pub clip: f64,
    /// Entropy bonus coefficient, applied to `log_std` only.
    pub entropy_coef: f64,
    /// Huber delta on the *value gradient*, in return units. Pass
    /// `f64::INFINITY` for a plain squared-error gradient.
    pub vdelta: f64,
    /// Minibatch size.
    pub minibatch: usize,
    /// Epochs over the batch.
    pub epochs: usize,
    /// The KL brake's threshold. The loop stops after the first minibatch
    /// whose brake reading exceeds it.
    pub target_kl: f64,
    /// Divisor applied to the brake reading — `ACT_DIM` for a per-dimension
    /// KL, `1.0` for a whole-vector one.
    pub kl_scale: f64,
    /// Which brake reading to test.
    pub kl_mode: KlMode,
}

/// How the KL brake reads the divergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KlMode {
    /// Mean of `logp_old - logp` over every row seen so far this update.
    Cumulative,
    /// Schulman's k3 estimator over the current minibatch alone.
    Minibatch,
}

/// What one update did.
#[derive(Debug, Clone, Copy, Default)]
pub struct UpdateStats {
    /// Mean clipped-surrogate loss over the minibatches that ran.
    pub policy_loss: f64,
    /// Mean squared value error over the minibatches that ran.
    pub value_loss: f64,
    /// Mean brake reading over the minibatches that ran — the number that
    /// decides the early stop, so the log says what stopped the update.
    pub kl: f64,
    /// Minibatches that ran.
    pub grad_steps: usize,
    /// Whether the KL brake stopped the loop early.
    pub stopped_early: bool,
}

/// Allocate / upload / download / launch, for the training kernels.
///
/// Deliberately a second trait rather than more methods on
/// [`super::KernelBackend`]: the simulation passes are f32 throughout, while
/// the optimizer keeps an f64 master weight and f64 moments, and the
/// minibatch index list is `u32`. Three buffer types on the simulation trait
/// would be three buffer types every simulation backend has to carry.
pub trait TrainBackend {
    /// A device-side `f32` buffer.
    type F32;
    /// A device-side `f64` buffer.
    type F64;
    /// A device-side `u32` buffer.
    type U32;

    /// Human-readable name of the device the kernels run on.
    fn device_name(&self) -> String;

    /// Allocate a zero-filled `f32` buffer.
    fn alloc_f32(&self, len: usize) -> Result<Self::F32, String>;
    /// Allocate a zero-filled `f64` buffer.
    fn alloc_f64(&self, len: usize) -> Result<Self::F64, String>;
    /// Allocate a zero-filled `u32` buffer.
    fn alloc_u32(&self, len: usize) -> Result<Self::U32, String>;

    /// Copy `data` into the front of `buf`.
    fn upload_f32(&self, buf: &mut Self::F32, data: &[f32]) -> Result<(), String>;
    /// Copy `data` into the front of `buf`.
    fn upload_f64(&self, buf: &mut Self::F64, data: &[f64]) -> Result<(), String>;
    /// Copy `data` into the front of `buf`.
    fn upload_u32(&self, buf: &mut Self::U32, data: &[u32]) -> Result<(), String>;

    /// Copy `len` floats of `buf` from `start` back to the host.
    fn download_f32(&self, buf: &Self::F32, start: usize, len: usize) -> Result<Vec<f32>, String>;
    /// Copy `len` doubles of `buf` from `start` back to the host.
    fn download_f64(&self, buf: &Self::F64, start: usize, len: usize) -> Result<Vec<f64>, String>;

    /// Block until every launch so far has completed.
    fn synchronize(&self) -> Result<(), String>;

    /// Forward pass: `b` threads.
    #[allow(clippy::too_many_arguments)]
    fn launch_fwd(
        &self,
        b: u32,
        dims: NetDims,
        w: &Self::F32,
        x: &Self::F32,
        idx: &Self::U32,
        h1: &mut Self::F32,
        h2: &mut Self::F32,
        out: &mut Self::F32,
    ) -> Result<(), String>;

    /// PPO policy gradient: `b` threads.
    #[allow(clippy::too_many_arguments)]
    fn launch_ppo_grad(
        &self,
        b: u32,
        n_out: u32,
        clip: f32,
        entropy_coef: f32,
        mean: &Self::F32,
        logstd: &Self::F32,
        logstd_off: u32,
        act: &Self::F32,
        logp_old: &Self::F32,
        adv: &Self::F32,
        idx: &Self::U32,
        dmean: &mut Self::F32,
        dlogstd: &mut Self::F32,
        stats: &mut Self::F32,
    ) -> Result<(), String>;

    /// Value gradient: `b` threads.
    #[allow(clippy::too_many_arguments)]
    fn launch_value_grad(
        &self,
        b: u32,
        vdelta: f32,
        values: &Self::F32,
        ret: &Self::F32,
        idx: &Self::U32,
        dv: &mut Self::F32,
        stats: &mut Self::F32,
    ) -> Result<(), String>;

    /// Backprop through a weight matrix and a tanh: `b * n_h` threads.
    #[allow(clippy::too_many_arguments)]
    fn launch_bwd_hidden(
        &self,
        b: u32,
        n_out: u32,
        n_h: u32,
        dout: &Self::F32,
        w: &Self::F32,
        w_off: u32,
        hact: &Self::F32,
        dh: &mut Self::F32,
    ) -> Result<(), String>;

    /// Weight gradient against a row-indexed input: `n_out * n_in` threads.
    #[allow(clippy::too_many_arguments)]
    fn launch_grad_w_idx(
        &self,
        b: u32,
        n_out: u32,
        n_in: u32,
        dout: &Self::F32,
        x: &Self::F32,
        idx: &Self::U32,
        gw: &mut Self::F32,
        gw_off: u32,
    ) -> Result<(), String>;

    /// Weight gradient against a dense input: `n_out * n_in` threads.
    #[allow(clippy::too_many_arguments)]
    fn launch_grad_w(
        &self,
        b: u32,
        n_out: u32,
        n_in: u32,
        dout: &Self::F32,
        x: &Self::F32,
        gw: &mut Self::F32,
        gw_off: u32,
    ) -> Result<(), String>;

    /// Bias gradient: `n_out` threads.
    fn launch_grad_b(
        &self,
        b: u32,
        n_out: u32,
        dout: &Self::F32,
        gb: &mut Self::F32,
        gb_off: u32,
    ) -> Result<(), String>;

    /// Adam step: `n` threads.
    #[allow(clippy::too_many_arguments)]
    fn launch_adam(
        &self,
        n: u32,
        cfg: AdamCfg,
        bc1: f64,
        bc2: f64,
        g: &Self::F32,
        m: &mut Self::F64,
        v: &mut Self::F64,
        p: &mut Self::F64,
        w: &mut Self::F32,
    ) -> Result<(), String>;

    /// Statistic reduction: four threads.
    fn launch_reduce_stats(
        &self,
        b: u32,
        stats: &Self::F32,
        out: &mut Self::F32,
    ) -> Result<(), String>;
}

/// One network's device state: forward weights, Adam master state, gradient,
/// and the activation scratch its widest minibatch needs.
struct DeviceNet<B: TrainBackend> {
    dims: NetDims,
    /// Parameters including the actor's trailing `log_std`, if any.
    n_param: usize,
    w: B::F32,
    p: B::F64,
    m: B::F64,
    v: B::F64,
    g: B::F32,
    h1: B::F32,
    h2: B::F32,
    out: B::F32,
    dh: B::F32,
    dout: B::F32,
    t: usize,
}

impl<B: TrainBackend> DeviceNet<B> {
    fn new(be: &B, dims: NetDims, extra: usize, max_b: usize) -> Result<Self, String> {
        let n_param = dims.param_count() + extra;
        Ok(Self {
            dims,
            n_param,
            w: be.alloc_f32(n_param)?,
            p: be.alloc_f64(n_param)?,
            m: be.alloc_f64(n_param)?,
            v: be.alloc_f64(n_param)?,
            g: be.alloc_f32(n_param)?,
            h1: be.alloc_f32(max_b * dims.n_h)?,
            h2: be.alloc_f32(max_b * dims.n_h)?,
            out: be.alloc_f32(max_b * dims.n_out)?,
            // `dh` and `dout` trade places as the backward walk moves down
            // the layers, so both are sized for the widest thing either can
            // hold — a hidden layer or the output.
            dh: be.alloc_f32(max_b * dims.n_h.max(dims.n_out))?,
            dout: be.alloc_f32(max_b * dims.n_h.max(dims.n_out))?,
            t: 0,
        })
    }

    /// Set the weights from a host `f64` flat array (tang's order), seeding
    /// both the master copy and the `f32` mirror.
    fn set_params(&mut self, be: &B, flat: &[f64]) -> Result<(), String> {
        if flat.len() != self.n_param {
            return Err(format!(
                "parameter array is {} long, network wants {}",
                flat.len(),
                self.n_param
            ));
        }
        be.upload_f64(&mut self.p, flat)?;
        let mirror: Vec<f32> = flat.iter().map(|&x| x as f32).collect();
        be.upload_f32(&mut self.w, &mirror)
    }

    fn params(&self, be: &B) -> Result<Vec<f64>, String> {
        be.download_f64(&self.p, 0, self.n_param)
    }

    /// Backprop the whole net given `self.dout` already holding dL/d(output),
    /// then take one Adam step. `x`/`idx` are the row-indexed input batch.
    fn backward_and_step(
        &mut self,
        be: &B,
        b: u32,
        x: &B::F32,
        idx: &B::U32,
        adam: AdamCfg,
    ) -> Result<(), String> {
        let d = self.dims;
        let (n_in, n_h, n_out) = (d.n_in as u32, d.n_h as u32, d.n_out as u32);
        // Layer 3: input is h2.
        be.launch_grad_w(
            b,
            n_out,
            n_h,
            &self.dout,
            &self.h2,
            &mut self.g,
            d.off_w3() as u32,
        )?;
        be.launch_grad_b(b, n_out, &self.dout, &mut self.g, d.off_b3() as u32)?;
        be.launch_bwd_hidden(
            b,
            n_out,
            n_h,
            &self.dout,
            &self.w,
            d.off_w3() as u32,
            &self.h2,
            &mut self.dh,
        )?;
        // Layer 2: input is h1. `dh` now holds dL/d(pre-activation 2); it is
        // both the gradient source and, next round, the destination, so the
        // hidden gradient moves through `dout` in between.
        std::mem::swap(&mut self.dh, &mut self.dout);
        be.launch_grad_w(
            b,
            n_h,
            n_h,
            &self.dout,
            &self.h1,
            &mut self.g,
            d.off_w2() as u32,
        )?;
        be.launch_grad_b(b, n_h, &self.dout, &mut self.g, d.off_b2() as u32)?;
        be.launch_bwd_hidden(
            b,
            n_h,
            n_h,
            &self.dout,
            &self.w,
            d.off_w2() as u32,
            &self.h1,
            &mut self.dh,
        )?;
        // Layer 1: input is the batch itself.
        std::mem::swap(&mut self.dh, &mut self.dout);
        be.launch_grad_w_idx(b, n_h, n_in, &self.dout, x, idx, &mut self.g, 0)?;
        be.launch_grad_b(b, n_h, &self.dout, &mut self.g, d.off_b1() as u32)?;

        self.t += 1;
        let bc1 = 1.0 - adam.beta1.powi(self.t as i32);
        let bc2 = 1.0 - adam.beta2.powi(self.t as i32);
        be.launch_adam(
            self.n_param as u32,
            adam,
            bc1,
            bc2,
            &self.g,
            &mut self.m,
            &mut self.v,
            &mut self.p,
            &mut self.w,
        )
    }
}

/// The device-resident PPO update.
///
/// Construct once per training run, then per iteration: [`upload_batch`] the
/// finished sample columns, [`update`] with the shuffle the host drew, and
/// [`actor_params`] / [`critic_params`] to read the weights back out. The
/// Adam moments and step counter live on the device across iterations, so the
/// optimizer state is never round-tripped.
///
/// [`upload_batch`]: TrainPipeline::upload_batch
/// [`update`]: TrainPipeline::update
/// [`actor_params`]: TrainPipeline::actor_params
/// [`critic_params`]: TrainPipeline::critic_params
pub struct TrainPipeline<B: TrainBackend> {
    be: B,
    actor: DeviceNet<B>,
    critic: DeviceNet<B>,
    max_b: usize,
    capacity: usize,
    n_rows: usize,
    // Sample columns, whole-iteration resident.
    obs_a: B::F32,
    obs_c: B::F32,
    act: B::F32,
    logp_old: B::F32,
    adv: B::F32,
    ret: B::F32,
    // Per-minibatch scratch.
    idx: B::U32,
    dlogstd: B::F32,
    stats: B::F32,
    reduced: B::F32,
    adam_actor: AdamCfg,
    adam_critic: AdamCfg,
}

/// One iteration's samples, already GAE'd and advantage-normalized by the
/// host. Column-major: every field is `n_rows` rows of its own width.
pub struct SampleBatch<'a> {
    /// Actor inputs, `n_rows * actor.n_in`, with any blind mask already
    /// applied.
    pub obs_a: &'a [f32],
    /// Critic inputs, `n_rows * critic.n_in`.
    pub obs_c: &'a [f32],
    /// Actions taken, `n_rows * actor.n_out`.
    pub act: &'a [f32],
    /// Log-probability under the collecting policy, `n_rows`.
    pub logp_old: &'a [f32],
    /// Normalized advantages, `n_rows`.
    pub adv: &'a [f32],
    /// GAE returns, `n_rows`.
    pub ret: &'a [f32],
}

impl<B: TrainBackend> TrainPipeline<B> {
    /// Allocate for an actor of `actor` shape, a critic of `critic` shape, at
    /// most `capacity` samples an iteration and at most `max_b` per
    /// minibatch.
    pub fn new(
        be: B,
        actor: NetDims,
        critic: NetDims,
        capacity: usize,
        max_b: usize,
    ) -> Result<Self, String> {
        for (what, d) in [("actor", actor), ("critic", critic)] {
            if d.n_h > TRAIN_MAX_H {
                return Err(format!(
                    "{what} hidden width {} exceeds TRAIN_MAX_H {TRAIN_MAX_H}",
                    d.n_h
                ));
            }
            if d.n_out > TRAIN_MAX_OUT {
                return Err(format!(
                    "{what} output width {} exceeds TRAIN_MAX_OUT {TRAIN_MAX_OUT}",
                    d.n_out
                ));
            }
        }
        if critic.n_out != 1 {
            return Err(format!("critic must have one output, got {}", critic.n_out));
        }
        // The actor's flat array carries `log_std` after `b3`.
        let a = DeviceNet::new(&be, actor, actor.n_out, max_b)?;
        let c = DeviceNet::new(&be, critic, 0, max_b)?;
        Ok(Self {
            obs_a: be.alloc_f32(capacity * actor.n_in)?,
            obs_c: be.alloc_f32(capacity * critic.n_in)?,
            act: be.alloc_f32(capacity * actor.n_out)?,
            logp_old: be.alloc_f32(capacity)?,
            adv: be.alloc_f32(capacity)?,
            ret: be.alloc_f32(capacity)?,
            idx: be.alloc_u32(max_b)?,
            dlogstd: be.alloc_f32(max_b * actor.n_out)?,
            stats: be.alloc_f32(max_b * 4)?,
            reduced: be.alloc_f32(4)?,
            actor: a,
            critic: c,
            max_b,
            capacity,
            n_rows: 0,
            adam_actor: AdamCfg::default(),
            adam_critic: AdamCfg::default(),
            be,
        })
    }

    /// The device the kernels run on.
    pub fn device_name(&self) -> String {
        self.be.device_name()
    }

    /// Seed the actor's weights: tang's flat `Mlp` order with `log_std`
    /// appended.
    pub fn set_actor_params(&mut self, flat: &[f64]) -> Result<(), String> {
        self.actor.set_params(&self.be, flat)
    }

    /// Seed the critic's weights: tang's flat `Mlp` order.
    pub fn set_critic_params(&mut self, flat: &[f64]) -> Result<(), String> {
        self.critic.set_params(&self.be, flat)
    }

    /// Read the actor's weights back, `log_std` included.
    pub fn actor_params(&self) -> Result<Vec<f64>, String> {
        self.actor.params(&self.be)
    }

    /// Read the critic's weights back.
    pub fn critic_params(&self) -> Result<Vec<f64>, String> {
        self.critic.params(&self.be)
    }

    /// Upload one iteration's samples.
    pub fn upload_batch(&mut self, n_rows: usize, batch: &SampleBatch<'_>) -> Result<(), String> {
        if n_rows > self.capacity {
            return Err(format!(
                "{n_rows} samples into a pipeline built for {}",
                self.capacity
            ));
        }
        let (ai, ci, ao) = (
            self.actor.dims.n_in,
            self.critic.dims.n_in,
            self.actor.dims.n_out,
        );
        for (what, got, want) in [
            ("obs_a", batch.obs_a.len(), n_rows * ai),
            ("obs_c", batch.obs_c.len(), n_rows * ci),
            ("act", batch.act.len(), n_rows * ao),
            ("logp_old", batch.logp_old.len(), n_rows),
            ("adv", batch.adv.len(), n_rows),
            ("ret", batch.ret.len(), n_rows),
        ] {
            if got != want {
                return Err(format!("{what} is {got} long, expected {want}"));
            }
        }
        self.be.upload_f32(&mut self.obs_a, batch.obs_a)?;
        self.be.upload_f32(&mut self.obs_c, batch.obs_c)?;
        self.be.upload_f32(&mut self.act, batch.act)?;
        self.be.upload_f32(&mut self.logp_old, batch.logp_old)?;
        self.be.upload_f32(&mut self.adv, batch.adv)?;
        self.be.upload_f32(&mut self.ret, batch.ret)?;
        self.n_rows = n_rows;
        Ok(())
    }

    /// Run `cfg.epochs` passes over the uploaded batch.
    ///
    /// `order` is one shuffle per epoch — `order[e]` must be a permutation of
    /// `0..n_rows`. The caller draws them, because the caller owns the run's
    /// PRNG stream and a same-seed comparison against a CPU loop is only
    /// meaningful if both see the same minibatches.
    pub fn update(&mut self, order: &[Vec<u32>], cfg: PpoUpdateCfg) -> Result<UpdateStats, String> {
        if self.n_rows == 0 {
            return Ok(UpdateStats::default());
        }
        if cfg.minibatch > self.max_b {
            return Err(format!(
                "minibatch {} exceeds the pipeline's {}",
                cfg.minibatch, self.max_b
            ));
        }
        if order.len() < cfg.epochs {
            return Err(format!(
                "{} shuffles for {} epochs",
                order.len(),
                cfg.epochs
            ));
        }
        let ad = self.actor.dims;
        let logstd_off = ad.off_logstd() as u32;

        let mut policy_acc = 0.0;
        let mut value_acc = 0.0;
        let mut brake_acc = 0.0;
        let mut batches = 0usize;
        let mut kl_acc = 0.0;
        let mut kl_rows = 0usize;
        let mut stopped = false;

        'epochs: for ep in order.iter().take(cfg.epochs) {
            if ep.len() != self.n_rows {
                return Err(format!(
                    "shuffle is {} long, batch is {}",
                    ep.len(),
                    self.n_rows
                ));
            }
            for chunk in ep.chunks(cfg.minibatch) {
                let b = chunk.len() as u32;
                self.be.upload_u32(&mut self.idx, chunk)?;

                // ── actor ──
                self.be.launch_fwd(
                    b,
                    ad,
                    &self.actor.w,
                    &self.obs_a,
                    &self.idx,
                    &mut self.actor.h1,
                    &mut self.actor.h2,
                    &mut self.actor.out,
                )?;
                // `logstd` is a slice of the actor's own weight buffer; the
                // kernel takes it as a base pointer plus the offset the
                // backend applies.
                self.be.launch_ppo_grad(
                    b,
                    ad.n_out as u32,
                    cfg.clip as f32,
                    cfg.entropy_coef as f32,
                    &self.actor.out,
                    &self.actor.w,
                    logstd_off,
                    &self.act,
                    &self.logp_old,
                    &self.adv,
                    &self.idx,
                    &mut self.actor.dout,
                    &mut self.dlogstd,
                    &mut self.stats,
                )?;
                // `log_std`'s gradient is the column sum of the per-row
                // contributions; the bias-gradient kernel is exactly that.
                self.be.launch_grad_b(
                    b,
                    ad.n_out as u32,
                    &self.dlogstd,
                    &mut self.actor.g,
                    logstd_off,
                )?;
                // Split borrow: the batch buffers are fields of `self` and
                // the net is too, so hand the net its own borrow explicitly.
                let adam = self.adam_actor;
                let Self {
                    be,
                    actor,
                    obs_a,
                    idx,
                    ..
                } = self;
                actor.backward_and_step(be, b, obs_a, idx, adam)?;

                // ── critic ──
                let cd = self.critic.dims;
                self.be.launch_fwd(
                    b,
                    cd,
                    &self.critic.w,
                    &self.obs_c,
                    &self.idx,
                    &mut self.critic.h1,
                    &mut self.critic.h2,
                    &mut self.critic.out,
                )?;
                self.be.launch_value_grad(
                    b,
                    cfg.vdelta as f32,
                    &self.critic.out,
                    &self.ret,
                    &self.idx,
                    &mut self.critic.dout,
                    &mut self.stats,
                )?;
                let adam = self.adam_critic;
                let Self {
                    be,
                    critic,
                    obs_c,
                    idx,
                    ..
                } = self;
                critic.backward_and_step(be, b, obs_c, idx, adam)?;

                // ── the brake ──
                self.be
                    .launch_reduce_stats(b, &self.stats, &mut self.reduced)?;
                let s = self.be.download_f32(&self.reduced, 0, 4)?;
                let (ploss, kl_sum, k3_sum, vloss) =
                    (s[0] as f64, s[1] as f64, s[2] as f64, s[3] as f64);
                policy_acc += ploss / b as f64;
                value_acc += vloss / b as f64;
                kl_acc += kl_sum;
                kl_rows += b as usize;
                let brake = match cfg.kl_mode {
                    KlMode::Cumulative => (kl_acc / kl_rows.max(1) as f64).abs(),
                    KlMode::Minibatch => k3_sum / b as f64,
                } / cfg.kl_scale;
                brake_acc += brake;
                batches += 1;
                if brake > cfg.target_kl {
                    stopped = true;
                    break 'epochs;
                }
            }
        }

        Ok(UpdateStats {
            policy_loss: policy_acc / batches.max(1) as f64,
            value_loss: value_acc / batches.max(1) as f64,
            kl: brake_acc / batches.max(1) as f64,
            grad_steps: batches,
            stopped_early: stopped,
        })
    }

    /// Per-network Adam settings. Set before the first [`update`]; the
    /// moments and step counter persist across iterations.
    ///
    /// [`update`]: TrainPipeline::update
    pub fn set_adam(&mut self, actor: AdamCfg, critic: AdamCfg) {
        self.adam_actor = actor;
        self.adam_critic = critic;
    }
}
