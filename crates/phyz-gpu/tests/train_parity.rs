//! The PPO update pass against the f64 CPU optimizer it replaces.
//!
//! The referee is not a re-derivation of the update — it is tang itself:
//! `Linear`/`Tanh` modules, `ModuleAdam`, and the same clipped-surrogate
//! arithmetic the CPU trainer runs, in `f64`. The device pass (here compiled
//! as host C++ by the `cuda-host` feature, which is the *exact text* NVRTC
//! compiles on a GPU) sees the same weights, the same batch and the same
//! minibatch order, and the test compares what actually matters: the weight
//! delta after the update, and the training curve over three updates.
//!
//! Tolerances are set by f32: gradients, activations and the forward weights
//! are single precision on the device, the Adam state is double on both
//! sides. Measured, the host walk agrees with tang to 2.8e-7 relative on the
//! actor's weight delta after one update and 1.6e-7 after three; the
//! assertions sit at 2e-4 and 1e-3 to leave room for a GPU's own `tanhf` and
//! `expf`, which are not required to match a libm's last bit. Anything
//! looser failing means a real disagreement rather than rounding.
//!
//! The same tests run against a real GPU under `--features cuda`.

#![cfg(any(feature = "cuda-host", feature = "cuda"))]

use phyz_gpu::cuda::{
    AdamCfg, KlMode, NetDims, PpoUpdateCfg, SampleBatch, TrainBackend, TrainPipeline,
};
use tang_tensor::{Shape, Tensor};
use tang_train::{Linear, Module, ModuleAdam, Optimizer, Parameter, Tanh};

// ── the problem ────────────────────────────────────────────────────────────

const N: usize = 512;
const A_IN: usize = 12;
const C_IN: usize = 20;
const H: usize = 16;
const ACT: usize = 4;
const MB: usize = 64;

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
        // Box-Muller; the test only needs plausible spread, not a good
        // generator.
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

struct Batch {
    obs_a: Vec<f64>,
    obs_c: Vec<f64>,
    act: Vec<f64>,
    logp_old: Vec<f64>,
    adv: Vec<f64>,
    ret: Vec<f64>,
}

fn make_batch(seed: u64) -> Batch {
    let mut r = Rng(seed);
    Batch {
        obs_a: (0..N * A_IN).map(|_| r.normal() * 0.5).collect(),
        obs_c: (0..N * C_IN).map(|_| r.normal() * 0.5).collect(),
        act: (0..N * ACT).map(|_| r.normal() * 0.3).collect(),
        // A plausible spread of old log-probs: the update is only interesting
        // when the ratio is not identically one.
        logp_old: (0..N).map(|_| -4.0 + r.normal() * 0.4).collect(),
        adv: (0..N).map(|_| r.normal()).collect(),
        ret: (0..N).map(|_| r.normal() * 2.0).collect(),
    }
}

fn shuffles(seed: u64, epochs: usize) -> Vec<Vec<u32>> {
    let mut r = Rng(seed);
    (0..epochs)
        .map(|_| {
            let mut o: Vec<u32> = (0..N as u32).collect();
            for i in (1..N).rev() {
                let j = (r.next() % (i as u64 + 1)) as usize;
                o.swap(i, j);
            }
            o
        })
        .collect()
}

// ── the f64 CPU reference ──────────────────────────────────────────────────

/// tang's three-layer tanh MLP, in the parameter order the device pipeline
/// expects.
struct Mlp {
    l1: Linear<f64>,
    a1: Tanh<f64>,
    l2: Linear<f64>,
    a2: Tanh<f64>,
    l3: Linear<f64>,
}

impl Mlp {
    fn new(n_in: usize, n_h: usize, n_out: usize, seed: u64) -> Self {
        Self {
            l1: Linear::new(n_in, n_h, seed ^ 0x11),
            a1: Tanh::new(),
            l2: Linear::new(n_h, n_h, seed ^ 0x22),
            a2: Tanh::new(),
            l3: Linear::new(n_h, n_out, seed ^ 0x33),
        }
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
    fn to_flat(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for t in [
            &self.l1.weight.data,
            &self.l1.bias.data,
            &self.l2.weight.data,
            &self.l2.bias.data,
            &self.l3.weight.data,
            &self.l3.bias.data,
        ] {
            out.extend_from_slice(t.data());
        }
        out
    }
}

struct CpuStats {
    policy_loss: f64,
    value_loss: f64,
    kl: f64,
    grad_steps: usize,
}

/// The CPU update loop, term for term what `TrainPipeline::update` runs.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn cpu_update(
    actor: &mut Mlp,
    log_std: &mut Parameter<f64>,
    critic: &mut Mlp,
    opt_a: &mut ModuleAdam,
    opt_c: &mut ModuleAdam,
    b: &Batch,
    order: &[Vec<u32>],
    cfg: PpoUpdateCfg,
) -> CpuStats {
    let mut policy_acc = 0.0;
    let mut value_acc = 0.0;
    let mut brake_acc = 0.0;
    let mut batches = 0usize;
    let mut kl_acc = 0.0;
    let mut kl_rows = 0usize;

    'epochs: for ep in order.iter().take(cfg.epochs) {
        for chunk in ep.chunks(cfg.minibatch) {
            let nb = chunk.len();
            let mut xa = Tensor::zeros(Shape::from_slice(&[nb, A_IN]));
            let mut xc = Tensor::zeros(Shape::from_slice(&[nb, C_IN]));
            for (row, &i) in chunk.iter().enumerate() {
                let i = i as usize;
                xa.data_mut()[row * A_IN..(row + 1) * A_IN]
                    .copy_from_slice(&b.obs_a[i * A_IN..(i + 1) * A_IN]);
                xc.data_mut()[row * C_IN..(row + 1) * C_IN]
                    .copy_from_slice(&b.obs_c[i * C_IN..(i + 1) * C_IN]);
            }

            actor.zero_grad();
            if let Some(g) = log_std.grad.as_mut() {
                for v in g.data_mut() {
                    *v = 0.0;
                }
            }
            let mean = actor.forward(&xa);
            let std: Vec<f64> = log_std.data.data().iter().map(|l| l.exp()).collect();

            let mut dmean = Tensor::zeros(Shape::from_slice(&[nb, ACT]));
            let mut dlogstd = vec![0.0; ACT];
            let mut ploss = 0.0;
            let mut mb_k3 = 0.0;
            for (row, &i) in chunk.iter().enumerate() {
                let i = i as usize;
                let mut logp = 0.0;
                for d in 0..ACT {
                    let m = mean.data()[row * ACT + d];
                    let z = (b.act[i * ACT + d] - m) / std[d];
                    logp += -0.5 * z * z - std[d].ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
                }
                kl_acc += b.logp_old[i] - logp;
                kl_rows += 1;
                let log_ratio = logp - b.logp_old[i];
                let ratio = log_ratio.exp();
                mb_k3 += ratio - 1.0 - log_ratio;
                let clipped = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip);
                let adv = b.adv[i];
                ploss += -(ratio * adv).min(clipped * adv);
                let active = (ratio * adv) <= (clipped * adv) + 1e-12;
                if active {
                    let coef = -adv * ratio / nb as f64;
                    for d in 0..ACT {
                        let m = mean.data()[row * ACT + d];
                        let z = (b.act[i * ACT + d] - m) / std[d];
                        dmean.data_mut()[row * ACT + d] = coef * (z / std[d]);
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
                    .get_or_insert_with(|| Tensor::zeros(Shape::from_slice(&[ACT])));
                for (gd, d) in g.data_mut().iter_mut().zip(&dlogstd) {
                    *gd += *d;
                }
            }
            let mut ap = actor.params_mut();
            ap.push(log_std);
            opt_a.step(&mut ap);

            critic.zero_grad();
            let values = critic.forward(&xc);
            let mut dv = Tensor::zeros(Shape::from_slice(&[nb, 1]));
            let mut vloss = 0.0;
            for (row, &i) in chunk.iter().enumerate() {
                let err = values.data()[row] - b.ret[i as usize];
                vloss += err * err;
                let d = err.clamp(-cfg.vdelta, cfg.vdelta);
                dv.data_mut()[row] = 2.0 * d / nb as f64;
            }
            critic.backward(&dv);
            let mut cp = critic.params_mut();
            opt_c.step(&mut cp);

            policy_acc += ploss / nb as f64;
            value_acc += vloss / nb as f64;
            let brake = match cfg.kl_mode {
                KlMode::Cumulative => (kl_acc / kl_rows.max(1) as f64).abs(),
                KlMode::Minibatch => mb_k3 / nb as f64,
            } / cfg.kl_scale;
            brake_acc += brake;
            batches += 1;
            if brake > cfg.target_kl {
                break 'epochs;
            }
        }
    }

    CpuStats {
        policy_loss: policy_acc / batches.max(1) as f64,
        value_loss: value_acc / batches.max(1) as f64,
        kl: brake_acc / batches.max(1) as f64,
        grad_steps: batches,
    }
}

// ── the harness ────────────────────────────────────────────────────────────

fn cfg(target_kl: f64) -> PpoUpdateCfg {
    PpoUpdateCfg {
        clip: 0.2,
        entropy_coef: 0.003,
        vdelta: 6.0,
        minibatch: MB,
        epochs: 3,
        target_kl,
        kl_scale: ACT as f64,
        kl_mode: KlMode::Minibatch,
    }
}

const ADAM: AdamCfg = AdamCfg {
    lr: 3e-4,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    weight_decay: 0.0,
};

fn adam() -> ModuleAdam {
    ModuleAdam::new(ADAM.lr)
}

struct Sides {
    cpu_actor: Vec<f64>,
    cpu_critic: Vec<f64>,
    gpu_actor: Vec<f64>,
    gpu_critic: Vec<f64>,
    cpu: Vec<CpuStats>,
    gpu: Vec<phyz_gpu::cuda::UpdateStats>,
    actor0: Vec<f64>,
    critic0: Vec<f64>,
}

/// Run `iters` updates on both sides from identical initial weights.
fn both<B: TrainBackend>(be: B, iters: usize, target_kl: f64) -> Sides {
    let batch = make_batch(0xC0FFEE);
    let cfg = cfg(target_kl);

    let mut actor = Mlp::new(A_IN, H, ACT, 7);
    let mut critic = Mlp::new(C_IN, H, 1, 9);
    let mut log_std = Parameter::new({
        let mut t = Tensor::zeros(Shape::from_slice(&[ACT]));
        for v in t.data_mut() {
            *v = -0.7;
        }
        t
    });

    let mut actor_flat = actor.to_flat();
    actor_flat.extend_from_slice(log_std.data.data());
    let critic_flat = critic.to_flat();
    let actor0 = actor_flat.clone();
    let critic0 = critic_flat.clone();

    let mut pipe = TrainPipeline::new(
        be,
        NetDims {
            n_in: A_IN,
            n_h: H,
            n_out: ACT,
        },
        NetDims {
            n_in: C_IN,
            n_h: H,
            n_out: 1,
        },
        N,
        MB,
    )
    .expect("pipeline");
    pipe.set_adam(ADAM, ADAM);
    pipe.set_actor_params(&actor_flat).expect("actor weights");
    pipe.set_critic_params(&critic_flat)
        .expect("critic weights");

    let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let (oa, oc, ac) = (f32s(&batch.obs_a), f32s(&batch.obs_c), f32s(&batch.act));
    let (lp, ad, rt) = (f32s(&batch.logp_old), f32s(&batch.adv), f32s(&batch.ret));
    pipe.upload_batch(
        N,
        &SampleBatch {
            obs_a: &oa,
            obs_c: &oc,
            act: &ac,
            logp_old: &lp,
            adv: &ad,
            ret: &rt,
        },
    )
    .expect("upload");

    let mut opt_a = adam();
    let mut opt_c = adam();
    let mut cpu = Vec::new();
    let mut gpu = Vec::new();
    for it in 0..iters {
        let order = shuffles(0x5EED + it as u64, cfg.epochs);
        cpu.push(cpu_update(
            &mut actor,
            &mut log_std,
            &mut critic,
            &mut opt_a,
            &mut opt_c,
            &batch,
            &order,
            cfg,
        ));
        gpu.push(pipe.update(&order, cfg).expect("device update"));
    }

    let mut cpu_actor = actor.to_flat();
    cpu_actor.extend_from_slice(log_std.data.data());
    Sides {
        cpu_actor,
        cpu_critic: critic.to_flat(),
        gpu_actor: pipe.actor_params().expect("actor readback"),
        gpu_critic: pipe.critic_params().expect("critic readback"),
        cpu,
        gpu,
        actor0,
        critic0,
    }
}

/// Largest relative disagreement between two weight *deltas*, scaled by the
/// largest delta on the reference side — the update is what is being
/// compared, not the weights it started from.
fn delta_gap(before: &[f64], cpu: &[f64], gpu: &[f64]) -> f64 {
    let scale = before
        .iter()
        .zip(cpu)
        .map(|(b, c)| (c - b).abs())
        .fold(0.0f64, f64::max)
        .max(1e-30);
    before
        .iter()
        .zip(cpu)
        .zip(gpu)
        .map(|((b, c), g)| ((c - b) - (g - b)).abs())
        .fold(0.0f64, f64::max)
        / scale
}

fn one_update<B: TrainBackend>(be: B) {
    // A KL threshold high enough that neither side brakes: this test is about
    // the arithmetic, and the brake gets its own test below.
    let s = both(be, 1, 1e9);
    assert_eq!(
        s.gpu[0].grad_steps, s.cpu[0].grad_steps,
        "the two sides ran different numbers of minibatches"
    );
    let a = delta_gap(&s.actor0, &s.cpu_actor, &s.gpu_actor);
    let c = delta_gap(&s.critic0, &s.cpu_critic, &s.gpu_critic);
    // The gradient is f32; the update it produces agrees to a few parts in
    // 10^5 of the largest weight movement in the step.
    eprintln!("one update: actor delta gap {a:.3e}, critic delta gap {c:.3e}");
    assert!(a < 2e-4, "actor weight delta differs by {a:.3e} relative");
    assert!(c < 2e-4, "critic weight delta differs by {c:.3e} relative");

    let dp =
        (s.gpu[0].policy_loss - s.cpu[0].policy_loss).abs() / s.cpu[0].policy_loss.abs().max(1e-12);
    let dv =
        (s.gpu[0].value_loss - s.cpu[0].value_loss).abs() / s.cpu[0].value_loss.abs().max(1e-12);
    let dk = (s.gpu[0].kl - s.cpu[0].kl).abs() / s.cpu[0].kl.abs().max(1e-12);
    eprintln!("one update: policy {dp:.3e}, value {dv:.3e}, kl {dk:.3e}");
    assert!(dp < 1e-4, "policy loss differs by {dp:.3e} relative");
    assert!(dv < 1e-4, "value loss differs by {dv:.3e} relative");
    assert!(dk < 1e-3, "KL differs by {dk:.3e} relative");
}

fn three_updates<B: TrainBackend>(be: B) {
    // The identity check that matters for a training run: not one step, but
    // whether the *sequence* of steps stays together once Adam's moments and
    // the weights themselves start to differ by f32 rounding.
    let s = both(be, 3, 1e9);
    for (i, (c, g)) in s.cpu.iter().zip(&s.gpu).enumerate() {
        let dp = (g.policy_loss - c.policy_loss).abs() / c.policy_loss.abs().max(1e-12);
        let dv = (g.value_loss - c.value_loss).abs() / c.value_loss.abs().max(1e-12);
        eprintln!("iteration {i}: policy {dp:.3e}, value {dv:.3e}");
        assert!(dp < 1e-3, "iteration {i}: policy loss differs by {dp:.3e}");
        assert!(dv < 1e-3, "iteration {i}: value loss differs by {dv:.3e}");
    }
    let a = delta_gap(&s.actor0, &s.cpu_actor, &s.gpu_actor);
    let c = delta_gap(&s.critic0, &s.cpu_critic, &s.gpu_critic);
    eprintln!("three updates: actor delta gap {a:.3e}, critic delta gap {c:.3e}");
    assert!(a < 1e-3, "actor drifted {a:.3e} over three updates");
    assert!(c < 1e-3, "critic drifted {c:.3e} over three updates");
}

fn kl_brake<B: TrainBackend>(be: B) {
    // A threshold low enough that the k3 estimator trips partway through.
    let s = both(be, 1, 2e-4);
    assert!(
        s.gpu[0].stopped_early,
        "the brake never fired: pick a lower threshold for this test"
    );
    assert_eq!(
        s.gpu[0].grad_steps, s.cpu[0].grad_steps,
        "the brake fired on different minibatches: cpu {} vs device {}",
        s.cpu[0].grad_steps, s.gpu[0].grad_steps
    );
}

// ── the two backends ───────────────────────────────────────────────────────

#[cfg(feature = "cuda-host")]
mod host {
    use super::*;
    use phyz_gpu::cuda::HostTrainBackend;

    #[test]
    fn one_update_moves_the_weights_the_same_way_as_the_f64_cpu_optimizer() {
        one_update(HostTrainBackend);
    }

    #[test]
    fn three_updates_track_the_cpu_training_curve() {
        three_updates(HostTrainBackend);
    }

    #[test]
    fn the_kl_brake_stops_both_sides_on_the_same_minibatch() {
        kl_brake(HostTrainBackend);
    }
}

// On a machine with no NVIDIA driver these skip rather than fail: the CUDA
// feature builds everywhere, only running needs the hardware, and the
// `cuda-host` tests above are the always-on gate.
#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use phyz_gpu::cuda::CudaTrainBackend;

    fn backend() -> Option<CudaTrainBackend> {
        match CudaTrainBackend::new(0) {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("skipping: no CUDA device ({e})");
                None
            }
        }
    }

    #[test]
    fn one_update_on_device_matches_the_f64_cpu_optimizer() {
        if let Some(b) = backend() {
            one_update(b);
        }
    }

    #[test]
    fn three_updates_on_device_track_the_cpu_training_curve() {
        if let Some(b) = backend() {
            three_updates(b);
        }
    }

    #[test]
    fn the_kl_brake_on_device_stops_on_the_same_minibatch() {
        if let Some(b) = backend() {
            kl_brake(b);
        }
    }
}
