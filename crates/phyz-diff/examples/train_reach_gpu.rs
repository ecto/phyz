//! [`train_reach`](./train_reach.rs) with every episode of a batch on the GPU
//! at once: the traced step and its adjoint as tang-expr WGSL kernels (one
//! environment per thread), the MLP's forward and backward as small WGSL
//! kernels, through tang-gpu (Metal on a Mac, Vulkan or DX12 elsewhere).
//! Only the weights (up) and their gradient (down) cross the bus per update;
//! Adam runs on the CPU. f32 throughout; checks its gradient against the f64
//! CPU path first.
//!
//! `cargo run --release -p phyz-diff --features gpu --example train_reach_gpu [ITERS] [ENVS]`

#[path = "train_reach.rs"]
#[allow(dead_code)]
mod cpu;

use std::time::Instant;

use cpu::{Adam, Episode, HIDDEN, HORIZON, Lcg, Mlp, TORQUE, Task, k1_fixed};
use phyz_diff::forward::{Step, StepAdjoint};
use tang_expr::codegen::Dialect;
use tang_gpu::{GpuBuffer, GpuDevice, KernelCache};

#[path = "common/gpu_mlp.rs"]
mod gpu_mlp;

use gpu_mlp::{WG, bwd_w_kernel, bwd_x_kernel, header, layer_offsets};

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let iters = args.first().copied().unwrap_or(300);
    let envs = args.get(1).copied().unwrap_or(4096);

    let model = k1_fixed();
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
    let device = GpuDevice::new_sync().expect("gpu");
    eprintln!("device: {}", device.name);
    let t = Instant::now();
    let mut gpu = Gpu::new(&device, &task, &step, &adj, envs);
    let mut rng = Lcg(7);
    let mut pol = Mlp::new(3 * task.nq, HIDDEN, task.nu, &mut rng);

    // the GPU gradient (f32) against the CPU one (f64) on the same batch
    let eps: Vec<Episode> = (0..envs).map(|_| task.episode(&mut rng)).collect();
    let g_gpu = gpu.grad(&device, &pol, &eps);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let check = &eps[..64.min(envs)];
    let g_gpu_small = {
        let mut small = Gpu::new(&device, &task, &step, &adj, check.len());
        small.grad(&device, &pol, check)
    };
    let (_, g_cpu) = task.loss_and_grad(&pol, check, threads);
    let dot: f64 = g_cpu.iter().zip(&g_gpu_small).map(|(a, b)| a * b).sum();
    let na = g_cpu.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = g_gpu_small.iter().map(|x| x * x).sum::<f64>().sqrt();
    let rel = g_cpu
        .iter()
        .zip(&g_gpu_small)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / na;
    eprintln!(
        "kernels built + first batch in {:.1}s; gradient f32 GPU vs f64 CPU: cosine {:.6}, relative error {rel:.1e} ({} params)",
        t.elapsed().as_secs_f64(),
        dot / (na * nb),
        g_gpu.len()
    );

    if std::env::var_os("TRAIN_PROFILE").is_some() {
        let t = Instant::now();
        for _ in 0..3 {
            gpu.grad(&device, &pol, &eps);
        }
        println!("grad: {:.0} ms", t.elapsed().as_secs_f64() * 1e3 / 3.0);
        return;
    }
    let eval_set: Vec<Episode> = (0..256).map(|_| task.episode(&mut rng)).collect();
    let mut adam = Adam::new(pol.params.len(), 3e-3);
    let t = Instant::now();
    for it in 0..iters {
        let eps: Vec<Episode> = (0..envs).map(|_| task.episode(&mut rng)).collect();
        let grad = gpu.grad(&device, &pol, &eps);
        adam.step(&mut pol.params, &grad);
        if it % 25 == 0 || it + 1 == iters {
            let err = task.mean_final_error(&eval_set, &|q, v, qs, u| pol.act(q, v, qs, u));
            eprintln!(
                "iter {it:4}  final error {err:.4} rad  {:.1}s",
                t.elapsed().as_secs_f64()
            );
        }
    }
    let secs = t.elapsed().as_secs_f64();
    let steps = (iters * envs * HORIZON) as f64;
    println!(
        "trained {iters} iters x {envs} episodes x {HORIZON} steps in {secs:.1}s: {:.2} M env-steps/s forward+backward",
        steps / secs / 1e6
    );
    let err = task.mean_final_error(&eval_set, &|q, v, qs, u| pol.act(q, v, qs, u));
    println!("final joint error: {err:.4} rad (CPU trainer 0.0106, best PD 0.0404)");
}

/// The batch on the device: one buffer per timestep for the step inputs and
/// the MLP's activations (kept for the reverse sweep), plus scratch.
struct Gpu {
    envs: usize,
    dims: Dims,
    k: Kernels,
    cache: KernelCache,
    xs: Vec<GpuBuffer>,  // [H+1] x [B, nx]: (q, v, u) at each step
    obs: Vec<GpuBuffer>, // [H] x [B, no]
    h1: Vec<GpuBuffer>,
    h2: Vec<GpuBuffer>,
    target: GpuBuffer,
    params: GpuBuffer,
    grad: GpuBuffer,
    out: GpuBuffer,  // [B, ns] step output
    lam: GpuBuffer,  // [B, ns]
    xin: GpuBuffer,  // [B, nx + ns] adjoint input
    back: GpuBuffer, // [B, nx] adjoint output
    gz3: GpuBuffer,  // [B, nu]
    gz2: GpuBuffer,  // [B, hidden]
    gz1: GpuBuffer,  // [B, hidden]
    gobs: GpuBuffer, // [B, no]
}

#[derive(Clone, Copy)]
struct Dims {
    nq: usize,
    nv: usize,
    nu: usize,
    ns: usize,
    nx: usize,
    no: usize,
    nh: usize,
}

struct Kernels {
    step: String,
    adjoint: String,
    obs: String,
    dense: [String; 3],
    copy_state: String,
    loss_grad: String,
    concat: String,
    gz3: String,
    bwd_w: [String; 3],
    bwd_x: [String; 3],
    lam: String,
}

impl Gpu {
    fn new(device: &GpuDevice, task: &Task, step: &Step, adj: &StepAdjoint, envs: usize) -> Self {
        let (nq, nv, nu) = (task.nq, task.nv, task.nu);
        let d = Dims {
            nq,
            nv,
            nu,
            ns: nq + nv,
            nx: nq + nv + nu,
            no: 3 * nq,
            nh: HIDDEN,
        };
        let rename = |src: String| src.replace(" k0(", " main(");
        let layer = [(d.no, d.nh), (d.nh, d.nh), (d.nh, d.nu)];
        let offs = layer_offsets(&layer);
        let k = Kernels {
            step: rename(step.kernel(Dialect::Wgsl).source),
            adjoint: rename(adj.kernel(Dialect::Wgsl).source),
            obs: obs_kernel(d),
            dense: std::array::from_fn(|l| dense_kernel(layer[l], offs[l], l == 2, d)),
            copy_state: copy_state_kernel(d),
            loss_grad: loss_grad_kernel(d),
            concat: concat_kernel(d),
            gz3: gz3_kernel(d),
            bwd_w: std::array::from_fn(|l| bwd_w_kernel(layer[l], offs[l], envs)),
            bwd_x: std::array::from_fn(|l| bwd_x_kernel(layer[l], offs[l], l > 0)),
            lam: lam_kernel(d),
        };
        let buf = |n: usize| GpuBuffer::from_slice(device, &vec![0.0f32; n.max(1)]);
        let np: usize = layer.iter().map(|(a, n)| a * n + n).sum();
        Self {
            envs,
            dims: d,
            k,
            cache: KernelCache::new(),
            xs: (0..=HORIZON).map(|_| buf(envs * d.nx)).collect(),
            obs: (0..HORIZON).map(|_| buf(envs * d.no)).collect(),
            h1: (0..HORIZON).map(|_| buf(envs * d.nh)).collect(),
            h2: (0..HORIZON).map(|_| buf(envs * d.nh)).collect(),
            target: buf(envs * nq),
            params: buf(np),
            grad: buf(np),
            out: buf(envs * d.ns),
            lam: buf(envs * d.ns),
            xin: buf(envs * (d.nx + d.ns)),
            back: buf(envs * d.nx),
            gz3: buf(envs * nu),
            gz2: buf(envs * d.nh),
            gz1: buf(envs * d.nh),
            gobs: buf(envs * d.no),
        }
    }

    /// Mean over `eps` of `∂loss/∂θ`.
    fn grad(&mut self, device: &GpuDevice, pol: &Mlp, eps: &[Episode]) -> Vec<f64> {
        assert_eq!(eps.len(), self.envs);
        let (d, b) = (self.dims, self.envs as u32);
        let mut x0 = vec![0.0f32; self.envs * d.nx];
        let mut tg = vec![0.0f32; self.envs * d.nq];
        for (e, ep) in eps.iter().enumerate() {
            for i in 0..d.nq {
                x0[e * d.nx + i] = ep.q0[i] as f32;
                tg[e * d.nq + i] = ep.target[i] as f32;
            }
        }
        self.xs[0].write(device, &x0);
        self.target.write(device, &tg);
        let p32: Vec<f32> = pol.params.iter().map(|&p| p as f32).collect();
        self.params.write(device, &p32);
        self.grad.write(device, &vec![0.0f32; p32.len()]);
        self.lam.write(device, &vec![0.0f32; self.envs * d.ns]);
        // TRAIN_SKIP (profiling, wrong results): 1 step, 2 adjoint, 4 weight
        // gradients, 8 MLP forward + input gradients
        let skip: u32 = std::env::var("TRAIN_SKIP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let c = &mut self.cache;
        let n = |k: usize| (k as u32) * b;
        c.begin_batch();
        for t in 0..HORIZON {
            let (xt, xn) = (&self.xs[t], &self.xs[t + 1]);
            c.dispatch_buffers(
                device,
                &self.k.obs,
                &[xt, &self.target, &self.obs[t]],
                &[2],
                &[n(d.no), 0, 0, 0],
                n(d.no),
                WG,
            );
            if skip & 8 == 0 {
                c.dispatch_buffers(
                    device,
                    &self.k.dense[0],
                    &[&self.params, &self.obs[t], &self.h1[t]],
                    &[2],
                    &[n(d.nh), 0, 0, 0],
                    n(d.nh),
                    WG,
                );
            }
            if skip & 8 == 0 {
                c.dispatch_buffers(
                    device,
                    &self.k.dense[1],
                    &[&self.params, &self.h1[t], &self.h2[t]],
                    &[2],
                    &[n(d.nh), 0, 0, 0],
                    n(d.nh),
                    WG,
                );
            }
            // the head writes u into this step's input
            if skip & 8 == 0 {
                c.dispatch_buffers(
                    device,
                    &self.k.dense[2],
                    &[&self.params, &self.h2[t], xt],
                    &[2],
                    &[n(d.nu), 0, 0, 0],
                    n(d.nu),
                    WG,
                );
            }
            if skip & 1 == 0 {
                c.dispatch_buffers(
                    device,
                    &self.k.step,
                    &[xt, &self.out],
                    &[1],
                    &[b, 0, 0, 0],
                    b,
                    WG,
                );
            }
            c.dispatch_buffers(
                device,
                &self.k.copy_state,
                &[&self.out, xn],
                &[1],
                &[n(d.ns), 0, 0, 0],
                n(d.ns),
                WG,
            );
        }
        for t in (0..HORIZON).rev() {
            let (xt, xn) = (&self.xs[t], &self.xs[t + 1]);
            let w = cpu::Task::weight(t) as f32;
            c.dispatch_buffers(
                device,
                &self.k.loss_grad,
                &[xn, &self.target, &self.lam],
                &[2],
                &[n(d.ns), w.to_bits(), 0, 0],
                n(d.ns),
                WG,
            );
            c.dispatch_buffers(
                device,
                &self.k.concat,
                &[xt, &self.lam, &self.xin],
                &[2],
                &[n(d.nx + d.ns), 0, 0, 0],
                n(d.nx + d.ns),
                WG,
            );
            if skip & 2 == 0 {
                c.dispatch_buffers(
                    device,
                    &self.k.adjoint,
                    &[&self.xin, &self.back],
                    &[1],
                    &[b, 0, 0, 0],
                    b,
                    WG,
                );
            }
            c.dispatch_buffers(
                device,
                &self.k.gz3,
                &[&self.back, xt, &self.gz3],
                &[2],
                &[n(d.nu), 0, 0, 0],
                n(d.nu),
                WG,
            );
            // layer 3 (head): weights, then into h2's pre-activation
            let lw = |l: usize| (layer_len(l, d)) as u32 * WG;
            if skip & 4 == 0 {
                c.dispatch_buffers(
                    device,
                    &self.k.bwd_w[2],
                    &[&self.gz3, &self.h2[t], &self.grad],
                    &[2],
                    &[lw(2), 0, 0, 0],
                    lw(2),
                    WG,
                );
            }
            if skip & 8 == 0 {
                c.dispatch_buffers(
                    device,
                    &self.k.bwd_x[2],
                    &[&self.params, &self.gz3, &self.h2[t], &self.gz2],
                    &[3],
                    &[n(d.nh), 0, 0, 0],
                    n(d.nh),
                    WG,
                );
            }
            if skip & 4 == 0 {
                c.dispatch_buffers(
                    device,
                    &self.k.bwd_w[1],
                    &[&self.gz2, &self.h1[t], &self.grad],
                    &[2],
                    &[lw(1), 0, 0, 0],
                    lw(1),
                    WG,
                );
            }
            if skip & 8 == 0 {
                c.dispatch_buffers(
                    device,
                    &self.k.bwd_x[1],
                    &[&self.params, &self.gz2, &self.h1[t], &self.gz1],
                    &[3],
                    &[n(d.nh), 0, 0, 0],
                    n(d.nh),
                    WG,
                );
            }
            if skip & 4 == 0 {
                c.dispatch_buffers(
                    device,
                    &self.k.bwd_w[0],
                    &[&self.gz1, &self.obs[t], &self.grad],
                    &[2],
                    &[lw(0), 0, 0, 0],
                    lw(0),
                    WG,
                );
            }
            if skip & 8 == 0 {
                c.dispatch_buffers(
                    device,
                    &self.k.bwd_x[0],
                    &[&self.params, &self.gz1, &self.obs[t], &self.gobs],
                    &[3],
                    &[n(d.no), 0, 0, 0],
                    n(d.no),
                    WG,
                );
            }
            c.dispatch_buffers(
                device,
                &self.k.lam,
                &[&self.back, &self.gobs, &self.lam],
                &[2],
                &[n(d.ns), 0, 0, 0],
                n(d.ns),
                WG,
            );
        }
        c.end_batch(device);
        c.flush(device);
        let g = self.grad.to_vec_sync(device);
        g.iter().map(|&x| x as f64 / self.envs as f64).collect()
    }
}

/// Weights + biases of layer `l`.
fn layer_len(l: usize, d: Dims) -> usize {
    let layer = [(d.no, d.nh), (d.nh, d.nh), (d.nh, d.nu)];
    layer[l].0 * layer[l].1 + layer[l].1
}

fn obs_kernel(d: Dims) -> String {
    let (nq, nv, nx, no) = (d.nq, d.nv, d.nx, d.no);
    header(&["x", "tg", "ob"], &[2])
        + &format!(
            "    let e = k / {no}u; let j = k % {no}u;\n\
             var v: f32;\n\
             if (j < {nq}u) {{ v = x[e * {nx}u + j] - tg[e * {nq}u + j]; }}\n\
             else if (j < {}u) {{ v = x[e * {nx}u + j]; }}\n\
             else {{ v = tg[e * {nq}u + j - {}u]; }}\n\
             ob[k] = v;\n}}\n",
            nq + nv,
            nq + nv
        )
}

/// The MLP's layers; the head (`last`) scales by TORQUE and writes into the
/// step input's control slots.
fn dense_kernel(l: (usize, usize), o: (usize, usize), last: bool, d: Dims) -> String {
    if last {
        gpu_mlp::dense_kernel(
            l,
            o,
            &format!("y[e * {}u + {}u + j]", d.nx, d.ns),
            &format!("{TORQUE:?} * z"),
        )
    } else {
        gpu_mlp::dense_kernel(l, o, "y[k]", "tanh(z)")
    }
}

fn copy_state_kernel(d: Dims) -> String {
    let (ns, nx) = (d.ns, d.nx);
    header(&["o", "xn"], &[1])
        + &format!("    let e = k / {ns}u; let i = k % {ns}u;\n    xn[e * {nx}u + i] = o[k];\n}}\n")
}

/// `λ += ∂(step loss)/∂s` at the state after step t (weight in `p.a`).
fn loss_grad_kernel(d: Dims) -> String {
    let (nq, ns, nx) = (d.nq, d.ns, d.nx);
    header(&["xn", "tg", "lam"], &[2])
        + &format!(
            "    let e = k / {ns}u; let i = k % {ns}u;\n\
             let s = xn[e * {nx}u + i];\n\
             if (i < {nq}u) {{ lam[k] += 2.0 * bitcast<f32>(p.a) * (s - tg[e * {nq}u + i]); }}\n\
             else {{ lam[k] += 2.0e-3 * s; }}\n}}\n"
        )
}

fn concat_kernel(d: Dims) -> String {
    let (ns, nx) = (d.ns, d.nx);
    let ni = nx + ns;
    header(&["x", "lam", "xin"], &[2])
        + &format!(
            "    let e = k / {ni}u; let j = k % {ni}u;\n\
             if (j < {nx}u) {{ xin[k] = x[e * {nx}u + j]; }} else {{ xin[k] = lam[e * {ns}u + j - {nx}u]; }}\n}}\n"
        )
}

/// The head's pre-activation gradient: `TORQUE (∂/∂u + 2e-5 u)`.
fn gz3_kernel(d: Dims) -> String {
    let (nu, ns, nx) = (d.nu, d.ns, d.nx);
    header(&["back", "x", "gz"], &[2])
        + &format!(
            "    let e = k / {nu}u; let j = k % {nu}u;\n\
             gz[k] = {TORQUE:?} * (back[e * {nx}u + {ns}u + j] + 2.0e-5 * x[e * {nx}u + {ns}u + j]);\n}}\n"
        )
}

/// `λ = ∂/∂s from the step adjoint + ∂/∂s through the observation` (the
/// observation's first `ns` entries are `q - q*` and `v`).
fn lam_kernel(d: Dims) -> String {
    let (ns, nx, no) = (d.ns, d.nx, d.no);
    header(&["back", "gobs", "lam"], &[2])
        + &format!(
            "    let e = k / {ns}u; let i = k % {ns}u;\n    lam[k] = back[e * {nx}u + i] + gobs[e * {no}u + i];\n}}\n"
        )
}
