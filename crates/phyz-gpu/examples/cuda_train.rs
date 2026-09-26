//! Time the PPO update: the f64 CPU optimizer against the device pass.
//!
//! The batch is the shape a real locomotion iteration produces — obs 55,
//! action 22, hidden 64, minibatch 4096, a million samples — because the
//! whole point of the pass is that the CPU update grows with the sample
//! count until it is the clock.
//!
//! ```text
//! cargo run --release -p phyz-gpu --features cuda,cuda-host --example cuda_train
//! ```
//!
//! Environment: `N` samples, `EPOCHS`, `MB` minibatch, `H` hidden,
//! `NO_CPU=1` to skip the reference (the CPU side is minutes at a million
//! samples, which is the finding, but not one worth re-measuring every run).

use std::time::Instant;

use phyz_gpu::cuda::{AdamCfg, KlMode, NetDims, PpoUpdateCfg, SampleBatch, TrainPipeline};
use tang_tensor::{Shape, Tensor};
use tang_train::{Linear, Module, ModuleAdam, Optimizer, Parameter, Tanh};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * self.uniform()).cos()
    }
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let n = env("N", 1_000_000);
    let epochs = env("EPOCHS", 1);
    let mb = env("MB", 4096);
    let h = env("H", 64);
    let (a_in, c_in, act) = (55usize, 67usize, 22usize);
    let skip_cpu = std::env::var("NO_CPU").is_ok();

    let actor_dims = NetDims {
        n_in: a_in,
        n_h: h,
        n_out: act,
    };
    let critic_dims = NetDims {
        n_in: c_in,
        n_h: h,
        n_out: 1,
    };
    println!(
        "batch {n} samples, obs {a_in}, act {act}, hidden {h}, minibatch {mb}, {epochs} epoch(s)"
    );
    println!(
        "actor {} params, critic {} params, {} minibatches/epoch",
        actor_dims.param_count() + act,
        critic_dims.param_count(),
        n.div_ceil(mb)
    );

    let mut r = Rng(0xC0FFEE);
    let t = Instant::now();
    let obs_a: Vec<f32> = (0..n * a_in).map(|_| (r.normal() * 0.5) as f32).collect();
    let obs_c: Vec<f32> = (0..n * c_in).map(|_| (r.normal() * 0.5) as f32).collect();
    let acts: Vec<f32> = (0..n * act).map(|_| (r.normal() * 0.3) as f32).collect();
    let logp: Vec<f32> = (0..n).map(|_| (-20.0 + r.normal() * 0.4) as f32).collect();
    let adv: Vec<f32> = (0..n).map(|_| r.normal() as f32).collect();
    let ret: Vec<f32> = (0..n).map(|_| (r.normal() * 2.0) as f32).collect();
    println!("batch generated in {:.2}s", t.elapsed().as_secs_f64());

    let order: Vec<Vec<u32>> = (0..epochs)
        .map(|_| {
            let mut o: Vec<u32> = (0..n as u32).collect();
            for i in (1..n).rev() {
                let j = (r.next() % (i as u64 + 1)) as usize;
                o.swap(i, j);
            }
            o
        })
        .collect();

    let cfg = PpoUpdateCfg {
        clip: 0.2,
        entropy_coef: 0.003,
        vdelta: 6.0,
        minibatch: mb,
        epochs,
        // No brake: a timing run must run every minibatch on both sides.
        target_kl: f64::INFINITY,
        kl_scale: act as f64,
        kl_mode: KlMode::Minibatch,
    };
    let adam = AdamCfg::default();

    // ── device ──
    let be = match backend() {
        Some(b) => b,
        None => return,
    };
    let mut pipe = TrainPipeline::new(be, actor_dims, critic_dims, n, mb).expect("pipeline");
    println!("device: {}", pipe.device_name());
    pipe.set_adam(adam, adam);

    let mut ra = Rng(7);
    let a_flat: Vec<f64> = (0..actor_dims.param_count() + act)
        .map(|_| ra.normal() * 0.1)
        .collect();
    let c_flat: Vec<f64> = (0..critic_dims.param_count())
        .map(|_| ra.normal() * 0.1)
        .collect();
    pipe.set_actor_params(&a_flat).expect("actor weights");
    pipe.set_critic_params(&c_flat).expect("critic weights");

    let t = Instant::now();
    pipe.upload_batch(
        n,
        &SampleBatch {
            obs_a: &obs_a,
            obs_c: &obs_c,
            act: &acts,
            logp_old: &logp,
            adv: &adv,
            ret: &ret,
        },
    )
    .expect("upload");
    let upload = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let stats = pipe.update(&order, cfg).expect("update");
    let device = t.elapsed().as_secs_f64();
    println!(
        "device update: {device:.3}s ({} minibatches, {:.2} ms each), upload {upload:.3}s",
        stats.grad_steps,
        1e3 * device / stats.grad_steps.max(1) as f64
    );
    println!(
        "  policy {:.6e}  value {:.6e}  kl {:.3e}",
        stats.policy_loss, stats.value_loss, stats.kl
    );

    if skip_cpu {
        return;
    }

    // ── the f64 CPU reference ──
    let t = Instant::now();
    let (cpu_steps, cpu_p, cpu_v) = cpu_update(
        a_in, c_in, h, act, mb, &obs_a, &obs_c, &acts, &logp, &adv, &ret, &order, &cfg, &a_flat,
        &c_flat,
    );
    let cpu = t.elapsed().as_secs_f64();
    println!(
        "tang f64 CPU update: {cpu:.3}s ({cpu_steps} minibatches, {:.2} ms each)",
        1e3 * cpu / cpu_steps.max(1) as f64
    );
    println!("  policy {cpu_p:.6e}  value {cpu_v:.6e}");
    println!("speedup: {:.1}x", cpu / device.max(1e-9));
}

#[cfg(feature = "cuda")]
fn backend() -> Option<phyz_gpu::cuda::CudaTrainBackend> {
    match phyz_gpu::cuda::CudaTrainBackend::new(0) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("no CUDA device: {e}");
            None
        }
    }
}

#[cfg(all(not(feature = "cuda"), feature = "cuda-host"))]
fn backend() -> Option<phyz_gpu::cuda::HostTrainBackend> {
    eprintln!("built without `cuda`: timing the host C++ walk, which is not a fast path");
    Some(phyz_gpu::cuda::HostTrainBackend)
}

struct Mlp {
    l1: Linear<f64>,
    a1: Tanh<f64>,
    l2: Linear<f64>,
    a2: Tanh<f64>,
    l3: Linear<f64>,
}

impl Mlp {
    fn from_flat(n_in: usize, n_h: usize, n_out: usize, flat: &[f64]) -> Self {
        let mut m = Self {
            l1: Linear::new(n_in, n_h, 1),
            a1: Tanh::new(),
            l2: Linear::new(n_h, n_h, 2),
            a2: Tanh::new(),
            l3: Linear::new(n_h, n_out, 3),
        };
        let mut at = 0;
        for t in [
            &mut m.l1.weight.data,
            &mut m.l1.bias.data,
            &mut m.l2.weight.data,
            &mut m.l2.bias.data,
            &mut m.l3.weight.data,
            &mut m.l3.bias.data,
        ] {
            let k = t.data().len();
            t.data_mut().copy_from_slice(&flat[at..at + k]);
            at += k;
        }
        m
    }
    fn forward(&mut self, x: &Tensor<f64>) -> Tensor<f64> {
        let x = self.l1.forward(x);
        let x = self.a1.forward(&x);
        let x = self.l2.forward(&x);
        let x = self.a2.forward(&x);
        self.l3.forward(&x)
    }
    fn backward(&mut self, g: &Tensor<f64>) {
        let g = self.l3.backward(g);
        let g = self.a2.backward(&g);
        let g = self.l2.backward(&g);
        let g = self.a1.backward(&g);
        let _ = self.l1.backward(&g);
    }
    fn params_mut(&mut self) -> Vec<&mut Parameter<f64>> {
        let mut p = self.l1.parameters_mut();
        p.extend(self.l2.parameters_mut());
        p.extend(self.l3.parameters_mut());
        p
    }
    fn zero_grad(&mut self) {
        for p in self.params_mut() {
            p.zero_grad();
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn cpu_update(
    a_in: usize,
    c_in: usize,
    h: usize,
    act: usize,
    mb: usize,
    obs_a: &[f32],
    obs_c: &[f32],
    acts: &[f32],
    logp: &[f32],
    adv: &[f32],
    ret: &[f32],
    order: &[Vec<u32>],
    cfg: &PpoUpdateCfg,
    a_flat: &[f64],
    c_flat: &[f64],
) -> (usize, f64, f64) {
    let mut actor = Mlp::from_flat(a_in, h, act, a_flat);
    let mut critic = Mlp::from_flat(c_in, h, 1, c_flat);
    let n_mlp = a_flat.len() - act;
    let mut log_std = Parameter::new({
        let mut t = Tensor::zeros(Shape::from_slice(&[act]));
        t.data_mut().copy_from_slice(&a_flat[n_mlp..]);
        t
    });
    let mut opt_a = ModuleAdam::new(3e-4);
    let mut opt_c = ModuleAdam::new(3e-4);
    let mut pl = 0.0;
    let mut vl = 0.0;
    let mut steps = 0usize;

    for ep in order.iter().take(cfg.epochs) {
        for chunk in ep.chunks(mb) {
            let nb = chunk.len();
            let mut xa = Tensor::zeros(Shape::from_slice(&[nb, a_in]));
            let mut xc = Tensor::zeros(Shape::from_slice(&[nb, c_in]));
            for (row, &i) in chunk.iter().enumerate() {
                let i = i as usize;
                for d in 0..a_in {
                    xa.data_mut()[row * a_in + d] = obs_a[i * a_in + d] as f64;
                }
                for d in 0..c_in {
                    xc.data_mut()[row * c_in + d] = obs_c[i * c_in + d] as f64;
                }
            }
            actor.zero_grad();
            if let Some(g) = log_std.grad.as_mut() {
                for v in g.data_mut() {
                    *v = 0.0;
                }
            }
            let mean = actor.forward(&xa);
            let std: Vec<f64> = log_std.data.data().iter().map(|l| l.exp()).collect();
            let mut dmean = Tensor::zeros(Shape::from_slice(&[nb, act]));
            let mut dlogstd = vec![0.0; act];
            let mut ploss = 0.0;
            for (row, &i) in chunk.iter().enumerate() {
                let i = i as usize;
                let mut lp = 0.0;
                for d in 0..act {
                    let m = mean.data()[row * act + d];
                    let z = (acts[i * act + d] as f64 - m) / std[d];
                    lp += -0.5 * z * z - std[d].ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
                }
                let ratio = (lp - logp[i] as f64).exp();
                let clipped = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip);
                let a = adv[i] as f64;
                ploss += -(ratio * a).min(clipped * a);
                if (ratio * a) <= (clipped * a) + 1e-12 {
                    let coef = -a * ratio / nb as f64;
                    for d in 0..act {
                        let m = mean.data()[row * act + d];
                        let z = (acts[i * act + d] as f64 - m) / std[d];
                        dmean.data_mut()[row * act + d] = coef * (z / std[d]);
                        dlogstd[d] += coef * (z * z - 1.0);
                    }
                }
            }
            for d in dlogstd.iter_mut() {
                *d -= cfg.entropy_coef;
            }
            actor.backward(&dmean);
            {
                let g = log_std
                    .grad
                    .get_or_insert_with(|| Tensor::zeros(Shape::from_slice(&[act])));
                for (gd, d) in g.data_mut().iter_mut().zip(&dlogstd) {
                    *gd += *d;
                }
            }
            let mut ap = actor.params_mut();
            ap.push(&mut log_std);
            opt_a.step(&mut ap);

            critic.zero_grad();
            let values = critic.forward(&xc);
            let mut dv = Tensor::zeros(Shape::from_slice(&[nb, 1]));
            let mut vloss = 0.0;
            for (row, &i) in chunk.iter().enumerate() {
                let err = values.data()[row] - ret[i as usize] as f64;
                vloss += err * err;
                dv.data_mut()[row] = 2.0 * err.clamp(-cfg.vdelta, cfg.vdelta) / nb as f64;
            }
            critic.backward(&dv);
            let mut cp = critic.params_mut();
            opt_c.step(&mut cp);

            pl += ploss / nb as f64;
            vl += vloss / nb as f64;
            steps += 1;
        }
    }
    (steps, pl / steps.max(1) as f64, vl / steps.max(1) as f64)
}
