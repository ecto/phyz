//! Push recovery for a standing K1, trained by backpropagating through the
//! contact simulator.
//!
//! The floating-base K1 stands on eight sole points under position servos
//! (D as implicit joint damping, the `contact_bench` setup). Each episode
//! starts from a settled stance with a random shove to the trunk; the
//! policy, an MLP `(pose error, v) -> joint-target offsets`, adds to the
//! nominal servo targets. The loss keeps the trunk at its standing height and
//! still. Gradients come from the traced contact step's reverse-mode adjoint
//! ([`ContactStep::adjoint`]: through the fixed PGS sweeps and the warm
//! start), chained over the horizon. Baseline: zero offsets (plain PD).
//!
//! `cargo run --release -p phyz-diff --example train_stand [ITERS]`

#[path = "train_reach.rs"]
#[allow(dead_code)]
mod reach;

use std::time::Instant;

use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::contact_step::{ContactPoint, ContactStep};
use phyz_math::{SpatialTransformExt as _, Vec3};
use phyz_model::{Actuator, Model};
use reach::{Adam, Lcg, Mlp, TORQUE};
use tang_expr::compile::CompiledMany;

/// Steps per episode (`STAND_H`, default 300 = 0.3 s at dt 1 ms).
fn horizon() -> usize {
    std::env::var("STAND_H")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300)
}
const BATCH: usize = 32;
const HIDDEN: usize = 64;
const OFFSET: f64 = 0.3; // policy output scale (rad)
const SWEEPS: usize = 5;

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(100);
    let (model, points) = k1();
    let mat = ContactMaterial {
        friction: 0.8,
        margin: 0.005,
        ..ContactMaterial::default()
    };
    let config = ContactSolverConfig::default();
    let t = Instant::now();
    let step = ContactStep::build(&model, &points, &mat, &config, SWEEPS);
    let (ag, ao) = step.adjoint();
    let task = Stand::new(
        &model,
        step.compile(),
        ag.compile_many(&ao),
        step.nu,
        points.len(),
    );
    eprintln!(
        "K1 floating, {} points: step {} nodes, adjoint {} nodes, traced in {:.1}s; standing height {:.3} m",
        points.len(),
        step.live_nodes(),
        ag.live_set(&ao).len(),
        t.elapsed().as_secs_f64(),
        task.z0
    );

    let mut rng = Lcg(11);
    let mut pol = Mlp::new(task.n_obs(), HIDDEN, task.nu, &mut rng);
    // start from zero offsets: the PD baseline
    let (w, _) = pol.offsets(2);
    let tail = pol.params.len();
    pol.params[w..tail].iter_mut().for_each(|p| *p = 0.0);

    if std::env::args().any(|a| a == "--check") {
        // the chained gradient against central differences of the episode
        // loss, with random (nonzero) head weights
        for p in pol.params.iter_mut() {
            if *p == 0.0 {
                *p = 0.01 * rng.next();
            }
        }
        let x0 = task.push(&mut rng);
        let mut grad = vec![0.0; pol.params.len()];
        task.episode_grad(&pol, &x0, &mut grad);
        let n = pol.params.len();
        for k in [0, 101, n / 2, n - 200, n - 30, n - 1] {
            let mut scratch = vec![0.0; n];
            let p0 = pol.params[k];
            let mut fd = Vec::new();
            for h in [1e-4, 1e-5, 1e-6] {
                pol.params[k] = p0 + h;
                let lp = task.episode_grad(&pol, &x0, &mut scratch);
                pol.params[k] = p0 - h;
                let lm = task.episode_grad(&pol, &x0, &mut scratch);
                fd.push((lp - lm) / (2.0 * h));
            }
            pol.params[k] = p0;
            println!(
                "weight {k:5}: backprop {:+.5e}  FD h=1e-4 {:+.5e}  1e-5 {:+.5e}  1e-6 {:+.5e}",
                grad[k], fd[0], fd[1], fd[2]
            );
        }
        return;
    }
    let eval: Vec<Vec<f64>> = (0..64).map(|_| task.push(&mut rng)).collect();
    let zero = |_: &[f64], u: &mut [f64]| u.iter_mut().for_each(|x| *x = 0.0);
    let (base_loss, base_drop) = task.evaluate(&eval, &zero);
    eprintln!(
        "PD baseline: loss {base_loss:.4}, mean trunk drop {:.1} mm",
        base_drop * 1e3
    );

    let mut adam = Adam::new(pol.params.len(), 1e-3);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let t = Instant::now();
    for it in 0..iters {
        let pushes: Vec<Vec<f64>> = (0..BATCH).map(|_| task.push(&mut rng)).collect();
        let (loss, grad) = task.loss_and_grad(&pol, &pushes, threads);
        adam.step(&mut pol.params, &grad);
        if it % 10 == 0 || it + 1 == iters {
            let (l, drop) = task.evaluate(&eval, &|o, u| task.act(&pol, o, u));
            eprintln!(
                "iter {it:4}  batch loss {loss:.4}  eval loss {l:.4} (PD {base_loss:.4})  trunk drop {:.1} mm (PD {:.1})  {:.0}s",
                drop * 1e3,
                base_drop * 1e3,
                t.elapsed().as_secs_f64()
            );
        }
    }
    let (l, drop) = task.evaluate(&eval, &|o, u| task.act(&pol, o, u));
    println!(
        "push recovery after {iters} iters x {BATCH} episodes: loss {l:.4} vs PD {base_loss:.4}, trunk drop {:.1} mm vs {:.1} mm; {:.1} k env-steps/s forward+backward",
        drop * 1e3,
        base_drop * 1e3,
        (iters * BATCH * horizon()) as f64 / t.elapsed().as_secs_f64() / 1e3
    );
}

/// The K1 at dt = 1 ms with position servos and the sole corners (the
/// `contact_bench` model).
fn k1() -> (Model, Vec<ContactPoint>) {
    let root = std::env::var_os("IPSE_K1_ASSETS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                .join("Developer/booster_assets/robots/K1")
        });
    let opts = phyz_urdf::UrdfOptions {
        base: phyz_urdf::BaseKind::Fixed,
        dt: Some(0.001),
        gravity: None,
    };
    let mut model = phyz_urdf::load_file(root.join("K1_22dof_floating.urdf"), &opts)
        .expect("K1 URDF")
        .model;
    model.actuators.clear();
    for j in 0..model.joints.len() {
        if model.joints[j].ndof() == 1 {
            let name = model.joints[j].name.clone();
            let leg = ["Hip", "Knee", "Ankle"].iter().any(|k| name.contains(k));
            let (kp, kd) = if leg { (80.0, 2.0) } else { (20.0, 0.5) };
            model.joints[j].damping = kd;
            model
                .actuators
                .push(Actuator::position(&name, &name, j, 1.0, kp, 0.0));
        }
    }
    let mut points = Vec::new();
    for foot in ["left_foot_link", "right_foot_link"] {
        let body = model.bodies.iter().position(|b| b.name == foot).unwrap();
        for (x, y) in [
            (0.116, 0.035),
            (0.116, -0.035),
            (-0.064, 0.035),
            (-0.064, -0.035),
        ] {
            points.push(ContactPoint {
                body,
                local: Vec3::new(x, y, -0.038),
            });
        }
    }
    (model, points)
}

struct Stand {
    nq: usize,
    nv: usize,
    nu: usize,
    nf: usize,
    fwd: CompiledMany,
    back: CompiledMany,
    /// Settled standing input `[q, v, ctrl, warm]` (ctrl = nominal targets).
    rest: Vec<f64>,
    z0: f64,
}

impl Stand {
    fn new(model: &Model, fwd: CompiledMany, back: CompiledMany, nu: usize, np: usize) -> Self {
        let (nq, nv, nf) = (model.nq, model.nv, 3 * np);
        let mut s = model.default_state();
        let (xf, _) = phyz_rigid::forward_kinematics(model, &s);
        let low = model
            .bodies
            .iter()
            .enumerate()
            .filter(|(_, b)| b.name.contains("foot"))
            .map(|(i, _)| xf[i].body_to_world_point(Vec3::new(0.0, 0.0, -0.038)).z)
            .fold(f64::INFINITY, f64::min);
        s.q[5] += 0.002 - low;
        let mut x: Vec<f64> =
            s.q.as_slice()
                .iter()
                .chain(s.v.as_slice())
                .chain(s.ctrl.as_slice())
                .copied()
                .collect();
        x.extend(std::iter::repeat_n(0.0, nf));
        // settle 1 s under the nominal targets
        let mut out = vec![0.0; nq + nv + nf];
        for _ in 0..1000 {
            fwd(&x, &mut out);
            x[..nq + nv].copy_from_slice(&out[..nq + nv]);
            x[nq + nv + nu..].copy_from_slice(&out[nq + nv..]);
        }
        let z0 = x[5];
        Self {
            nq,
            nv,
            nu,
            nf,
            fwd,
            back,
            rest: x,
            z0,
        }
    }

    /// Observation: pose error (orientation, height, joints; not x, y) and
    /// velocity.
    fn n_obs(&self) -> usize {
        self.nq - 2 + self.nv
    }

    fn obs(&self, x: &[f64]) -> Vec<f64> {
        let (nq, nv) = (self.nq, self.nv);
        (0..nq)
            .filter(|&i| i != 3 && i != 4)
            .map(|i| x[i] - self.rest[i])
            .chain(x[nq..nq + nv].iter().copied())
            .collect()
    }

    /// Indices into `x` of each observation entry (for the adjoint).
    fn obs_index(&self) -> Vec<usize> {
        (0..self.nq)
            .filter(|&i| i != 3 && i != 4)
            .chain(self.nq..self.nq + self.nv)
            .collect()
    }

    /// A shove: the standing state with a random trunk velocity.
    fn push(&self, rng: &mut Lcg) -> Vec<f64> {
        let mut x = self.rest.clone();
        let nq = self.nq;
        // v[0..3] angular, v[3..6] linear (free joint)
        for i in 0..3 {
            x[nq + i] += 0.5 * rng.next();
        }
        x[nq + 3] += 0.4 * rng.next();
        x[nq + 4] += 0.4 * rng.next();
        x
    }

    fn act(&self, pol: &Mlp, obs: &[f64], u: &mut [f64]) {
        let c = pol.forward(obs);
        for (o, a) in u.iter_mut().zip(&c.out) {
            *o = a * (OFFSET / TORQUE);
        }
    }

    /// Per-step loss on the state after the step, and its gradient.
    fn step_loss(&self, x: &[f64], g: Option<&mut [f64]>) -> f64 {
        let (nq, nv) = (self.nq, self.nv);
        let dz = x[5] - self.z0;
        let tilt = x[0] * x[0] + x[1] * x[1];
        let vb: f64 = x[nq..nq + 6].iter().map(|v| v * v).sum();
        let vj: f64 = x[nq + 6..nq + nv].iter().map(|v| v * v).sum();
        let s = 1.0 / horizon() as f64;
        if let Some(g) = g {
            g[5] += s * 200.0 * dz;
            g[0] += s * 20.0 * x[0];
            g[1] += s * 20.0 * x[1];
            for i in 0..6 {
                g[nq + i] += s * 2.0 * x[nq + i];
            }
            for i in 6..nv {
                g[nq + i] += s * 0.02 * x[nq + i];
            }
        }
        s * (100.0 * dz * dz + 10.0 * tilt + vb + 0.01 * vj)
    }

    /// Mean loss and trunk drop at the end over `pushes` under `policy`
    /// (offsets).
    fn evaluate(&self, pushes: &[Vec<f64>], policy: &dyn Fn(&[f64], &mut [f64])) -> (f64, f64) {
        let (nq, nv, nu) = (self.nq, self.nv, self.nu);
        let (mut loss, mut drop) = (0.0, 0.0);
        let mut out = vec![0.0; nq + nv + self.nf];
        let mut off = vec![0.0; nu];
        for x0 in pushes {
            let mut x = x0.clone();
            for _ in 0..horizon() {
                policy(&self.obs(&x), &mut off);
                for i in 0..nu {
                    x[nq + nv + i] = self.rest[nq + nv + i] + off[i];
                }
                (self.fwd)(&x, &mut out);
                x[..nq + nv].copy_from_slice(&out[..nq + nv]);
                x[nq + nv + nu..].copy_from_slice(&out[nq + nv..]);
                loss += self.step_loss(&x, None);
            }
            drop += self.z0 - x[5];
        }
        let n = pushes.len() as f64;
        (loss / n, drop / n)
    }

    fn loss_and_grad(&self, pol: &Mlp, pushes: &[Vec<f64>], threads: usize) -> (f64, Vec<f64>) {
        let chunk = pushes.len().div_ceil(threads);
        let parts: Vec<(f64, Vec<f64>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = pushes
                .chunks(chunk)
                .map(|c| {
                    sc.spawn(move || {
                        let mut grad = vec![0.0; pol.params.len()];
                        let loss: f64 = c
                            .iter()
                            .map(|x0| self.episode_grad(pol, x0, &mut grad))
                            .sum();
                        (loss, grad)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let n = pushes.len() as f64;
        let mut grad = vec![0.0; pol.params.len()];
        let mut loss = 0.0;
        for (l, g) in parts {
            loss += l;
            grad.iter_mut().zip(g).for_each(|(a, b)| *a += b);
        }
        grad.iter_mut().for_each(|g| *g /= n);
        (loss / n, grad)
    }

    /// Forward rollout keeping each step's input, then the reverse sweep:
    /// `λ` is the adjoint of the state `(q, v, warm)` after each step.
    fn episode_grad(&self, pol: &Mlp, x0: &[f64], grad: &mut [f64]) -> f64 {
        let (nq, nv, nu, nf) = (self.nq, self.nv, self.nu, self.nf);
        let (ns, ni) = (nq + nv, nq + nv + nu + nf);
        let idx = self.obs_index();
        let mut xs = Vec::with_capacity(horizon() + 1);
        let mut caches = Vec::with_capacity(horizon());
        let mut x = x0.to_vec();
        let mut out = vec![0.0; ns + nf];
        let mut loss = 0.0;
        for _ in 0..horizon() {
            let obs = self.obs(&x);
            let c = pol.forward(&obs);
            for i in 0..nu {
                x[ns + i] = self.rest[ns + i] + c.out[i] * (OFFSET / TORQUE);
            }
            (self.fwd)(&x, &mut out);
            xs.push(x.clone());
            caches.push((obs, c));
            x[..ns].copy_from_slice(&out[..ns]);
            x[ns + nu..].copy_from_slice(&out[ns..]);
            loss += self.step_loss(&x, None);
        }
        xs.push(x);
        // λ over the step output (q', v', f)
        let mut lam = vec![0.0; ns + nf];
        let mut xin = vec![0.0; ni + ns + nf];
        let mut back = vec![0.0; ni];
        let mut gx = vec![0.0; ni];
        for t in (0..horizon()).rev() {
            gx.iter_mut().for_each(|g| *g = 0.0);
            self.step_loss(&xs[t + 1], Some(&mut gx));
            for i in 0..ns {
                lam[i] += gx[i];
            }
            xin[..ni].copy_from_slice(&xs[t]);
            xin[ni..].copy_from_slice(&lam);
            (self.back)(&xin, &mut back);
            // through the policy: ∂/∂ctrl is ∂/∂offset
            let mut gout: Vec<f64> = back[ns..ns + nu]
                .iter()
                .map(|g| g * (OFFSET / TORQUE))
                .collect();
            let (obs, c) = &caches[t];
            let gobs = pol.backward(obs, c, &mut gout, grad);
            // adjoint of the step input state (q, v, warm)
            lam[..ns].copy_from_slice(&back[..ns]);
            lam[ns..].copy_from_slice(&back[ns + nu..]);
            for (k, &i) in idx.iter().enumerate() {
                lam[i] += gobs[k];
            }
        }
        loss
    }
}
