//! Train a K1 policy by backpropagating through the simulator.
//!
//! The K1 on a fixed base, joints driven by torque motors, must move from a
//! random pose to a random target pose and hold it under gravity. The policy
//! is a small MLP `(q - q*, v, q*) -> torques`. Each update rolls a batch of
//! episodes forward through the traced step ([`Step`]), then runs the
//! reverse sweep through the traced step adjoint ([`StepAdjoint`], tang-expr
//! reverse mode) and the MLP, giving the exact gradient of the episode loss
//! with respect to every weight. No sampling-based policy-gradient estimator.
//!
//! Compares the learned policy against the best PD control (no gravity
//! compensation) over a gain sweep.
//!
//! `cargo run --release -p phyz-diff --example train_reach [ITERS] [--check]`

use std::time::Instant;

use phyz_diff::forward::{Step, StepAdjoint};
use phyz_model::{Actuator, Model};
use tang_expr::compile::CompiledMany;

const HORIZON: usize = 150; // 0.3 s at dt 2 ms
const BATCH: usize = 64;
const HIDDEN: usize = 64;
const TORQUE: f64 = 20.0; // output scale (N m)

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let iters: usize = args.iter().find_map(|a| a.parse().ok()).unwrap_or(300);
    let check = args.iter().any(|a| a == "--check");

    let model = k1_fixed();
    let t = Instant::now();
    let step = Step::build(&model);
    let adj = StepAdjoint::build(&step);
    let task = Task {
        nq: model.nq,
        nv: model.nv,
        nu: step.nu,
        fwd: step.compile(),
        back: adj.compile(),
        lo: model
            .joints
            .iter()
            .filter(|j| j.ndof() == 1)
            .map(|j| j.limits.map_or(-1.0, |l| l[0]))
            .collect(),
        hi: model
            .joints
            .iter()
            .filter(|j| j.ndof() == 1)
            .map(|j| j.limits.map_or(1.0, |l| l[1]))
            .collect(),
    };
    eprintln!(
        "K1 fixed base: nq {} nv {} nu {}; step {} nodes, adjoint {} nodes, traced in {:.1}s",
        model.nq,
        model.nv,
        step.nu,
        step.live_nodes(),
        adj.live_nodes(),
        t.elapsed().as_secs_f64()
    );

    let mut rng = Lcg(7);
    let mut pol = Mlp::new(3 * task.nq, HIDDEN, task.nu, &mut rng);
    if check {
        gradient_check(&task, &mut pol, &mut rng);
        return;
    }
    let eval_set: Vec<Episode> = (0..256).map(|_| task.episode(&mut rng)).collect();
    // the best stable PD over a gain sweep (uniform gains; explicit kd, so
    // high gains diverge at dt 2 ms)
    let mut pd = f64::INFINITY;
    let mut gains = (0.0, 0.0);
    for kp in [5.0, 10.0, 20.0, 40.0, 80.0] {
        for kd in [0.05, 0.1, 0.2, 0.5, 1.0] {
            let e = task.mean_final_error(&eval_set, &|q, v, qs, u| pd_policy(kp, kd, q, v, qs, u));
            if e < pd {
                pd = e;
                gains = (kp, kd);
            }
        }
    }
    eprintln!(
        "PD baseline: best kp {} kd {}: mean final joint error {pd:.4} rad",
        gains.0, gains.1
    );

    let mut adam = Adam::new(pol.params.len(), 3e-3);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let t = Instant::now();
    for it in 0..iters {
        let eps: Vec<Episode> = (0..BATCH).map(|_| task.episode(&mut rng)).collect();
        let (loss, grad) = task.loss_and_grad(&pol, &eps, threads);
        adam.step(&mut pol.params, &grad);
        if it % 25 == 0 || it + 1 == iters {
            let err = task.mean_final_error(&eval_set, &|q, v, qs, u| pol.act(q, v, qs, u));
            eprintln!(
                "iter {it:4}  loss {loss:9.4}  final error {err:.4} rad (best PD {pd:.4})  {:.1}s",
                t.elapsed().as_secs_f64()
            );
        }
    }
    let steps = (iters * BATCH * HORIZON) as f64;
    println!(
        "trained {iters} iters x {BATCH} episodes x {HORIZON} steps: {:.0} k env-steps/s forward+backward on {threads} threads",
        steps / t.elapsed().as_secs_f64() / 1e3
    );
    let err = task.mean_final_error(&eval_set, &|q, v, qs, u| pol.act(q, v, qs, u));
    println!("final joint error: learned {err:.4} rad vs PD {pd:.4} rad");
}

/// The K1 on a fixed base at dt 2 ms with a torque motor per joint and light
/// joint damping.
fn k1_fixed() -> Model {
    let root = std::env::var_os("IPSE_K1_ASSETS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                .join("Developer/booster_assets/robots/K1")
        });
    let opts = phyz_urdf::UrdfOptions {
        base: phyz_urdf::BaseKind::Fixed,
        dt: Some(0.002),
        gravity: None,
    };
    let mut model = phyz_urdf::load_file(root.join("K1_22dof.urdf"), &opts)
        .expect("K1 URDF")
        .model;
    model.actuators.clear();
    for j in 0..model.joints.len() {
        if model.joints[j].ndof() == 1 {
            let name = model.joints[j].name.clone();
            model.joints[j].damping = 0.5;
            model.actuators.push(Actuator::motor(&name, &name, j, 1.0));
        }
    }
    model
}

/// `(q, v, q*, torques out)`.
type Policy = dyn Fn(&[f64], &[f64], &[f64], &mut [f64]);

struct Task {
    nq: usize,
    nv: usize,
    nu: usize,
    fwd: CompiledMany,
    back: CompiledMany,
    lo: Vec<f64>,
    hi: Vec<f64>,
}

#[derive(Clone)]
struct Episode {
    q0: Vec<f64>,
    target: Vec<f64>,
}

impl Task {
    /// Start and target poses drawn from the middle half of each joint's range.
    fn episode(&self, rng: &mut Lcg) -> Episode {
        let pose = |rng: &mut Lcg| -> Vec<f64> {
            (0..self.nq)
                .map(|i| {
                    let (lo, hi) = (self.lo[i], self.hi[i]);
                    let mid = 0.5 * (lo + hi);
                    mid + 0.25 * (hi - lo) * rng.next()
                })
                .collect()
        };
        Episode {
            q0: pose(rng),
            target: pose(rng),
        }
    }

    /// Per-step loss weight on joint error: ramps up so the policy may move
    /// first and must hold at the end.
    fn weight(t: usize) -> f64 {
        let s = (t + 1) as f64 / HORIZON as f64;
        s * s
    }

    /// Mean absolute joint error at the end of each episode under `policy`.
    fn mean_final_error(
        &self,
        eps: &[Episode],
        policy: &Policy,
    ) -> f64 {
        let (nq, nv, nu) = (self.nq, self.nv, self.nu);
        let mut total = 0.0;
        for ep in eps {
            let mut x = vec![0.0; nq + nv + nu];
            x[..nq].copy_from_slice(&ep.q0);
            let mut out = vec![0.0; nq + nv];
            for _ in 0..HORIZON {
                let (s, u) = x.split_at_mut(nq + nv);
                policy(&s[..nq], &s[nq..], &ep.target, u);
                (self.fwd)(&x, &mut out);
                x[..nq + nv].copy_from_slice(&out);
            }
            total += (0..nq).map(|i| (x[i] - ep.target[i]).abs()).sum::<f64>() / nq as f64;
        }
        total / eps.len() as f64
    }

    /// Batch loss and its exact gradient with respect to the policy weights.
    fn loss_and_grad(&self, pol: &Mlp, eps: &[Episode], threads: usize) -> (f64, Vec<f64>) {
        let chunk = eps.len().div_ceil(threads);
        let parts: Vec<(f64, Vec<f64>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = eps
                .chunks(chunk)
                .map(|c| {
                    sc.spawn(move || {
                        let mut grad = vec![0.0; pol.params.len()];
                        let mut loss = 0.0;
                        for ep in c {
                            loss += self.episode_grad(pol, ep, &mut grad);
                        }
                        (loss, grad)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let n = eps.len() as f64;
        let mut grad = vec![0.0; pol.params.len()];
        let mut loss = 0.0;
        for (l, g) in parts {
            loss += l;
            for (a, b) in grad.iter_mut().zip(g) {
                *a += b;
            }
        }
        grad.iter_mut().for_each(|g| *g /= n);
        (loss / n, grad)
    }

    /// One episode: forward rollout keeping every step's input, then the
    /// reverse sweep. Adds `∂loss/∂θ` into `grad`; returns the loss.
    fn episode_grad(&self, pol: &Mlp, ep: &Episode, grad: &mut [f64]) -> f64 {
        let (nq, nv, nu) = (self.nq, self.nv, self.nu);
        let ns = nq + nv;
        let mut xs: Vec<Vec<f64>> = Vec::with_capacity(HORIZON);
        let mut caches = Vec::with_capacity(HORIZON);
        let mut x = vec![0.0; ns + nu];
        x[..nq].copy_from_slice(&ep.q0);
        let mut out = vec![0.0; ns];
        let mut loss = 0.0;
        for t in 0..HORIZON {
            let obs = obs(&x[..nq], &x[nq..ns], &ep.target);
            let cache = pol.forward(&obs);
            x[ns..].copy_from_slice(&cache.out);
            (self.fwd)(&x, &mut out);
            xs.push(x.clone());
            caches.push((obs, cache));
            x[..ns].copy_from_slice(&out);
            loss += step_loss(
                &x[..nq],
                &x[nq..ns],
                &xs[t][ns..],
                &ep.target,
                Self::weight(t),
            );
        }
        // reverse sweep: λ is ∂loss/∂(state after step t)
        let mut lam = vec![0.0; ns];
        let mut xin = vec![0.0; 2 * ns + nu];
        let mut back = vec![0.0; ns + nu];
        for t in (0..HORIZON).rev() {
            // this step's loss terms on its output state
            let next: Vec<f64> = if t + 1 < HORIZON {
                xs[t + 1][..ns].to_vec()
            } else {
                x[..ns].to_vec()
            };
            let w = Self::weight(t);
            for i in 0..nq {
                lam[i] += 2.0 * w * (next[i] - ep.target[i]);
            }
            for i in 0..nv {
                lam[nq + i] += 2.0 * 1e-3 * next[nq + i];
            }
            xin[..ns + nu].copy_from_slice(&xs[t]);
            xin[ns + nu..].copy_from_slice(&lam);
            (self.back)(&xin, &mut back);
            // through the policy: ∂/∂u, plus the control penalty
            let mut gu: Vec<f64> = (0..nu)
                .map(|i| back[ns + i] + 2.0 * 1e-5 * xs[t][ns + i])
                .collect();
            let (obs, cache) = &caches[t];
            let gobs = pol.backward(obs, cache, &mut gu, grad);
            lam.copy_from_slice(&back[..ns]);
            // obs = (q - q*, v, q*): q and v feed the state adjoint
            for i in 0..nq {
                lam[i] += gobs[i];
            }
            for i in 0..nv {
                lam[nq + i] += gobs[nq + i];
            }
        }
        loss
    }
}

fn obs(q: &[f64], v: &[f64], target: &[f64]) -> Vec<f64> {
    q.iter()
        .zip(target)
        .map(|(a, b)| a - b)
        .chain(v.iter().copied())
        .chain(target.iter().copied())
        .collect()
}

fn step_loss(q: &[f64], v: &[f64], u: &[f64], target: &[f64], w: f64) -> f64 {
    let e: f64 = q.iter().zip(target).map(|(a, b)| (a - b) * (a - b)).sum();
    let vv: f64 = v.iter().map(|x| x * x).sum();
    let uu: f64 = u.iter().map(|x| x * x).sum();
    w * e + 1e-3 * vv + 1e-5 * uu
}

fn pd_policy(kp: f64, kd: f64, q: &[f64], v: &[f64], target: &[f64], u: &mut [f64]) {
    for i in 0..u.len() {
        u[i] = kp * (target[i] - q[i]) - kd * v[i];
    }
}

/// Two-hidden-layer tanh MLP with a linear head scaled by [`TORQUE`].
struct Mlp {
    sizes: [usize; 4],
    params: Vec<f64>,
}

struct Cache {
    h1: Vec<f64>,
    h2: Vec<f64>,
    out: Vec<f64>,
}

impl Mlp {
    fn new(nin: usize, nh: usize, nout: usize, rng: &mut Lcg) -> Self {
        let sizes = [nin, nh, nh, nout];
        let mut params = Vec::new();
        for l in 0..3 {
            let (a, b) = (sizes[l], sizes[l + 1]);
            let s = (1.0 / a as f64).sqrt() * if l == 2 { 0.1 } else { 1.0 };
            params.extend((0..a * b).map(|_| s * rng.next()));
            params.extend(std::iter::repeat_n(0.0, b));
        }
        Self { sizes, params }
    }

    /// `(weights offset, bias offset)` of layer `l` (weights `[out][in]`).
    fn offsets(&self, l: usize) -> (usize, usize) {
        let mut o = 0;
        for k in 0..l {
            o += self.sizes[k] * self.sizes[k + 1] + self.sizes[k + 1];
        }
        (o, o + self.sizes[l] * self.sizes[l + 1])
    }

    fn layer(&self, l: usize, x: &[f64]) -> Vec<f64> {
        let (w, b) = self.offsets(l);
        let (a, n) = (self.sizes[l], self.sizes[l + 1]);
        (0..n)
            .map(|j| {
                let row = &self.params[w + j * a..w + (j + 1) * a];
                self.params[b + j] + row.iter().zip(x).map(|(p, v)| p * v).sum::<f64>()
            })
            .collect()
    }

    fn forward(&self, x: &[f64]) -> Cache {
        let h1: Vec<f64> = self.layer(0, x).into_iter().map(f64::tanh).collect();
        let h2: Vec<f64> = self.layer(1, &h1).into_iter().map(f64::tanh).collect();
        let out = self.layer(2, &h2).into_iter().map(|v| TORQUE * v).collect();
        Cache { h1, h2, out }
    }

    fn act(&self, q: &[f64], v: &[f64], target: &[f64], u: &mut [f64]) {
        u.copy_from_slice(&self.forward(&obs(q, v, target)).out);
    }

    /// Backprop `gout = ∂L/∂out` (consumed): adds `∂L/∂θ` to `grad`,
    /// returns `∂L/∂x`.
    fn backward(&self, x: &[f64], c: &Cache, gout: &mut [f64], grad: &mut [f64]) -> Vec<f64> {
        let back = |l: usize, input: &[f64], gz: &[f64], grad: &mut [f64]| -> Vec<f64> {
            let (w, b) = self.offsets(l);
            let (a, n) = (self.sizes[l], self.sizes[l + 1]);
            let mut gx = vec![0.0; a];
            for j in 0..n {
                grad[b + j] += gz[j];
                for i in 0..a {
                    grad[w + j * a + i] += gz[j] * input[i];
                    gx[i] += gz[j] * self.params[w + j * a + i];
                }
            }
            gx
        };
        let gz3: Vec<f64> = gout.iter().map(|g| TORQUE * g).collect();
        let gh2 = back(2, &c.h2, &gz3, grad);
        let gz2: Vec<f64> = gh2
            .iter()
            .zip(&c.h2)
            .map(|(g, h)| g * (1.0 - h * h))
            .collect();
        let gh1 = back(1, &c.h1, &gz2, grad);
        let gz1: Vec<f64> = gh1
            .iter()
            .zip(&c.h1)
            .map(|(g, h)| g * (1.0 - h * h))
            .collect();
        back(0, x, &gz1, grad)
    }
}

struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
    lr: f64,
}

impl Adam {
    fn new(n: usize, lr: f64) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            lr,
        }
    }

    fn step(&mut self, p: &mut [f64], g: &[f64]) {
        self.t += 1;
        let (b1, b2) = (0.9f64, 0.999f64);
        // clip the global norm: long contact-free unrolls still spike
        let norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        let s = if norm > 10.0 { 10.0 / norm } else { 1.0 };
        for i in 0..p.len() {
            let gi = g[i] * s;
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi;
            let mh = self.m[i] / (1.0 - b1.powi(self.t));
            let vh = self.v[i] / (1.0 - b2.powi(self.t));
            p[i] -= self.lr * mh / (vh.sqrt() + 1e-8);
        }
    }
}

/// The backpropagated gradient against central differences of the episode
/// loss, on a few weights.
fn gradient_check(task: &Task, pol: &mut Mlp, rng: &mut Lcg) {
    let ep = task.episode(rng);
    let mut grad = vec![0.0; pol.params.len()];
    task.episode_grad(pol, &ep, &mut grad);
    let mut worst: f64 = 0.0;
    let n = pol.params.len();
    for k in [0, 17, n / 3, n / 2, n - task.nu - 5, n - 1] {
        let h = 1e-6;
        let p0 = pol.params[k];
        let mut scratch = vec![0.0; n];
        pol.params[k] = p0 + h;
        let lp = task.episode_grad(pol, &ep, &mut scratch);
        pol.params[k] = p0 - h;
        let lm = task.episode_grad(pol, &ep, &mut scratch);
        pol.params[k] = p0;
        let fd = (lp - lm) / (2.0 * h);
        let err = (grad[k] - fd).abs() / (1e-8 + fd.abs().max(grad[k].abs()));
        println!(
            "weight {k:6}: backprop {:+.6e}  central difference {fd:+.6e}  rel {err:.1e}",
            grad[k]
        );
        worst = worst.max(err);
    }
    println!("worst relative gradient error {worst:.1e}");
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}
