//! [`train_stand`](./train_stand.rs) with the whole batch on the GPU: the
//! traced contact step and its reverse-mode adjoint as tang-expr WGSL
//! kernels (one environment per thread), the MLP as the shared WGSL kernels
//! (`common/gpu_mlp.rs`). Checks its f32 gradient against the f64 CPU one,
//! then trains push recovery.
//!
//! `cargo run --release -p phyz-diff --features gpu --example train_stand_gpu [ITERS] [ENVS]`
//! (`STAND_H` sets the horizon, default 150)

#[path = "common/gpu_mlp.rs"]
mod gpu_mlp;
#[path = "train_stand.rs"]
#[allow(dead_code)]
mod stand;

use std::time::Instant;

use gpu_mlp::{WG, bwd_w_kernel, bwd_x_kernel, dense_kernel, header, layer_offsets};
use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::contact_step::ContactStep;
use stand::reach::{Adam, Lcg, Mlp};
use stand::{HIDDEN, OFFSET, SWEEPS, Stand, k1};
use tang_expr::codegen::Dialect;
use tang_gpu::{GpuBuffer, GpuDevice, KernelCache};

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let iters = args.first().copied().unwrap_or(200);
    let envs = args.get(1).copied().unwrap_or(4096);
    let h = std::env::var("STAND_H")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(150);
    // the CPU task (gradient check, evaluation) reads STAND_H too
    if std::env::var_os("STAND_H").is_none() {
        unsafe { std::env::set_var("STAND_H", h.to_string()) };
    }

    let device = GpuDevice::new_sync().expect("gpu");
    // f64 for the contact physics where the device has it: in f32 the
    // rollout leaves the f64 trajectory within a few steps (STAND_F32=1
    // forces f32)
    let f64 = device.shader_f64 && std::env::var_os("STAND_F32").is_none();
    eprintln!(
        "device: {} (contact physics in {})",
        device.name,
        if f64 { "f64" } else { "f32" }
    );
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
    // STAND_SPLIT nodes per kernel (default 20000; 0 = one kernel each):
    // shader compilers choke on the 326k-node adjoint as one function
    let split: usize = std::env::var("STAND_SPLIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20000);
    let kern = |g: &tang_expr::ExprGraph, o: &[tang_expr::ExprId], ni: usize| -> Graph {
        if f64 {
            let s = g.to_wgsl_split_f64(o, ni, split.max(1));
            Graph::Split(s.parts, 2 * s.n_scratch)
        } else if split == 0 {
            Graph::Whole(
                g.to_kernel(o, ni, Dialect::Wgsl)
                    .source
                    .replace(" k0(", " main("),
            )
        } else {
            let s = g.to_wgsl_split(o, ni, split);
            Graph::Split(s.parts, s.n_scratch)
        }
    };
    let fwd_src = kern(&step.graph, &step.out, step.n_inputs());
    let adj_src = kern(&ag, &ao, step.n_inputs() + step.n_outputs());
    let task = Stand::new(
        &model,
        step.compile(),
        ag.compile_many(&ao),
        step.nu,
        points.len(),
    );
    eprintln!(
        "K1 contact step {} nodes, adjoint {} nodes; {} + {} kernels; traced in {:.1}s",
        step.live_nodes(),
        ag.live_set(&ao).len(),
        fwd_src.kernels(),
        adj_src.kernels(),
        t.elapsed().as_secs_f64()
    );

    // one cache for every batch size: the graph kernels compile once
    let mut cache = KernelCache::new();
    let mut rng = Lcg(11);
    let mut pol = Mlp::new(task.n_obs(), HIDDEN, task.nu, &mut rng);
    let (w, _) = pol.offsets(2);
    let np = pol.params.len();
    // a small random head for the gradient check, zero (= PD) for training
    for p in &mut pol.params[w..np] {
        *p = 0.01 * rng.next();
    }
    let t = Instant::now();
    let check: Vec<Vec<f64>> = (0..16).map(|_| task.push(&mut rng)).collect();
    let mut small = Gpu::new(&device, &task, &fwd_src, &adj_src, check.len(), h, f64);
    let g_gpu = small.grad(&device, &mut cache, &pol, &check);
    // the forward rollout too: the GPU state after each tenth of the
    // horizon against the f64 CPU rollout under the same policy
    {
        let (nq, nv, nu) = (task.nq, task.nv, task.nu);
        let ns = nq + nv;
        let mut worst = [0.0f64; 11];
        let mut out = vec![0.0; ns + task.nf];
        let mut off = vec![0.0; nu];
        for (e, x0) in check.iter().enumerate() {
            let mut x = x0.clone();
            for t in 0..=h {
                if t % (h / 10).max(1) == 0 && t / (h / 10).max(1) <= 10 {
                    let g = small.read_state(&device, t);
                    let gx = &g[e * small.d.nx..e * small.d.nx + ns];
                    let d = gx
                        .iter()
                        .zip(&x[..ns])
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max);
                    let k = t / (h / 10).max(1);
                    worst[k] = worst[k].max(d);
                }
                if t == h {
                    break;
                }
                task.act(&pol, &task.obs(&x), &mut off);
                for i in 0..nu {
                    x[ns + i] = task.rest[ns + i] + off[i];
                }
                (task.fwd)(&x, &mut out);
                x[..ns].copy_from_slice(&out[..ns]);
                x[ns + nu..].copy_from_slice(&out[ns..]);
            }
        }
        eprintln!(
            "forward GPU vs f64 CPU, worst state gap at each tenth of the horizon: {:?}",
            worst.iter().map(|w| format!("{w:.1e}")).collect::<Vec<_>>()
        );
    }
    eprintln!(
        "kernels compiled + first batch in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let (_, g_cpu) = task.loss_and_grad(&pol, &check, threads);
    let dot: f64 = g_cpu.iter().zip(&g_gpu).map(|(a, b)| a * b).sum();
    let (na, nb) = (norm(&g_cpu), norm(&g_gpu));
    let diff: Vec<f64> = g_cpu.iter().zip(&g_gpu).map(|(a, b)| a - b).collect();
    eprintln!(
        "gradient GPU vs f64 CPU ({h} steps): cosine {:.6}, relative error {:.1e}",
        dot / (na * nb),
        norm(&diff) / na
    );
    for p in &mut pol.params[w..np] {
        *p = 0.0;
    }

    let eval: Vec<Vec<f64>> = (0..64).map(|_| task.push(&mut rng)).collect();
    let zero = |_: &[f64], u: &mut [f64]| u.iter_mut().for_each(|x| *x = 0.0);
    let (base_loss, base_drop) = task.evaluate(&eval, &zero);
    eprintln!(
        "PD baseline: loss {base_loss:.4}, mean trunk drop {:.2} mm",
        base_drop * 1e3
    );

    let mut gpu = Gpu::new(&device, &task, &fwd_src, &adj_src, envs, h, f64);
    let mut adam = Adam::new(np, 1e-3);
    let t = Instant::now();
    let mut gpu_secs = 0.0;
    for it in 0..iters {
        let pushes: Vec<Vec<f64>> = (0..envs).map(|_| task.push(&mut rng)).collect();
        let tg = Instant::now();
        let grad = gpu.grad(&device, &mut cache, &pol, &pushes);
        gpu_secs += tg.elapsed().as_secs_f64();
        adam.step(&mut pol.params, &grad);
        if it % 20 == 0 || it + 1 == iters {
            let (l, drop) = task.evaluate(&eval, &|o, u| task.act(&pol, o, u));
            eprintln!(
                "iter {it:4}  eval loss {l:.4} (PD {base_loss:.4})  trunk drop {:.2} mm (PD {:.2})  {:.0}s",
                drop * 1e3,
                base_drop * 1e3,
                t.elapsed().as_secs_f64()
            );
        }
    }
    let (l, drop) = task.evaluate(&eval, &|o, u| task.act(&pol, o, u));
    println!(
        "push recovery, {iters} iters x {envs} episodes x {h} steps: loss {l:.4} vs PD {base_loss:.4} ({:+.1}%), trunk drop {:.2} vs {:.2} mm; {:.2} M env-steps/s forward+backward on the GPU",
        100.0 * (l / base_loss - 1.0),
        drop * 1e3,
        base_drop * 1e3,
        (iters * envs * h) as f64 / gpu_secs / 1e6
    );
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[derive(Clone, Copy)]
struct Dims {
    nq: usize,
    nu: usize,
    ns: usize,
    nx: usize,   // step input: q, v, ctrl, warm
    nout: usize, // step output: q', v', f
    no: usize,
    nh: usize,
    /// WGSL type of the state chain (`f64` or `f32`) and f32 slots per value.
    t: &'static str,
    w: usize,
}

/// A traced graph as one kernel, or split through a scratch buffer
/// (`ExprGraph::to_wgsl_split`: parts, scratch floats per environment).
enum Graph {
    Whole(String),
    Split(Vec<String>, usize),
}

impl Graph {
    fn kernels(&self) -> usize {
        match self {
            Graph::Whole(_) => 1,
            Graph::Split(p, _) => p.len(),
        }
    }

    fn scratch(&self, device: &GpuDevice, envs: usize) -> GpuBuffer {
        let n = match self {
            Graph::Whole(_) => 1,
            Graph::Split(_, s) => envs * s,
        };
        GpuBuffer::from_slice(device, &vec![0.0f32; n])
    }

    /// Launch over `b` environments, `input` to `output`.
    fn run(
        &self,
        c: &mut KernelCache,
        device: &GpuDevice,
        input: &GpuBuffer,
        scratch: &GpuBuffer,
        output: &GpuBuffer,
        b: u32,
    ) {
        match self {
            Graph::Whole(src) => {
                c.dispatch_buffers(device, src, &[input, output], &[1], &[b, 0, 0, 0], b, 256)
            }
            Graph::Split(parts, _) => {
                for src in parts {
                    c.dispatch_buffers(
                        device,
                        src,
                        &[input, scratch, output],
                        &[1, 2],
                        &[b, 0, 0, 0],
                        b,
                        256,
                    );
                }
            }
        }
    }
}

struct Gpu<'g> {
    fwd: &'g Graph,
    adj: &'g Graph,
    fwd_scratch: GpuBuffer,
    adj_scratch: GpuBuffer,
    envs: usize,
    h: usize,
    d: Dims,
    k: Kernels,
    xs: Vec<GpuBuffer>,
    obs: Vec<GpuBuffer>,
    h1: Vec<GpuBuffer>,
    h2: Vec<GpuBuffer>,
    params: GpuBuffer,
    grad: GpuBuffer,
    out: GpuBuffer,
    lam: GpuBuffer,
    xin: GpuBuffer,
    back: GpuBuffer,
    gz3: GpuBuffer,
    gz2: GpuBuffer,
    gz1: GpuBuffer,
    gobs: GpuBuffer,
}

struct Kernels {
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

impl<'g> Gpu<'g> {
    fn new(
        device: &GpuDevice,
        task: &Stand,
        fwd: &'g Graph,
        adj: &'g Graph,
        envs: usize,
        h: usize,
        f64: bool,
    ) -> Self {
        let (nq, nv, nu, nf) = (task.nq, task.nv, task.nu, task.nf);
        let d = Dims {
            nq,
            nu,
            ns: nq + nv,
            nx: nq + nv + nu + nf,
            nout: nq + nv + nf,
            no: task.n_obs(),
            nh: HIDDEN,
            t: if f64 { "f64" } else { "f32" },
            w: if f64 { 2 } else { 1 },
        };
        let layer = [(d.no, d.nh), (d.nh, d.nh), (d.nh, d.nu)];
        let offs = layer_offsets(&layer);
        let rest = wgsl_array("rest", &task.rest);
        let head_store = format!("y[e * {}u + {}u + j]", d.nx, d.ns);
        let head_act = format!(
            "{t}(rest[{}u + j]) + {t}({OFFSET:?}) * {t}(z)",
            d.ns,
            t = d.t
        );
        let k = Kernels {
            obs: obs_kernel(d, &rest),
            dense: [
                dense_kernel(layer[0], offs[0], "y[k]", "tanh(z)"),
                dense_kernel(layer[1], offs[1], "y[k]", "tanh(z)"),
                format!(
                    "{rest}{}",
                    retype(
                        &dense_kernel(layer[2], offs[2], &head_store, &head_act),
                        &["y"],
                        d.t
                    )
                ),
            ],
            copy_state: copy_state_kernel(d),
            loss_grad: loss_grad_kernel(d, task.z0, h),
            concat: concat_kernel(d),
            gz3: gz3_kernel(d),
            bwd_w: std::array::from_fn(|l| bwd_w_kernel(layer[l], offs[l], envs)),
            bwd_x: std::array::from_fn(|l| bwd_x_kernel(layer[l], offs[l], l > 0)),
            lam: lam_kernel(d),
        };
        let buf = |n: usize| GpuBuffer::from_slice(device, &vec![0.0f32; n.max(1)]);
        // the state chain, at d.w f32 slots per value
        let sbuf = |n: usize| buf(n * d.w);
        let np: usize = layer.iter().map(|(a, n)| a * n + n).sum();
        Self {
            fwd,
            adj,
            fwd_scratch: fwd.scratch(device, envs),
            adj_scratch: adj.scratch(device, envs),
            envs,
            h,
            d,
            k,
            xs: (0..=h).map(|_| sbuf(envs * d.nx)).collect(),
            obs: (0..h).map(|_| buf(envs * d.no)).collect(),
            h1: (0..h).map(|_| buf(envs * d.nh)).collect(),
            h2: (0..h).map(|_| buf(envs * d.nh)).collect(),
            params: buf(np),
            grad: buf(np),
            out: sbuf(envs * d.nout),
            lam: sbuf(envs * d.nout),
            xin: sbuf(envs * (d.nx + d.nout)),
            back: sbuf(envs * d.nx),
            gz3: buf(envs * nu),
            gz2: buf(envs * d.nh),
            gz1: buf(envs * d.nh),
            gobs: buf(envs * d.no),
        }
    }

    /// State values as the chain's buffer words.
    fn encode(&self, v: &[f64]) -> Vec<f32> {
        if self.d.w == 2 {
            bytemuck::cast_slice::<f64, f32>(v).to_vec()
        } else {
            v.iter().map(|&x| x as f32).collect()
        }
    }

    /// The step input at step `t`, every environment, as f64.
    fn read_state(&self, device: &GpuDevice, t: usize) -> Vec<f64> {
        let raw = self.xs[t].to_vec_sync(device);
        if self.d.w == 2 {
            bytemuck::cast_slice::<f32, f64>(&raw).to_vec()
        } else {
            raw.iter().map(|&x| x as f64).collect()
        }
    }

    /// Mean over `pushes` (start inputs `[q, v, ctrl, warm]`) of `∂loss/∂θ`.
    fn grad(
        &mut self,
        device: &GpuDevice,
        c: &mut KernelCache,
        pol: &Mlp,
        pushes: &[Vec<f64>],
    ) -> Vec<f64> {
        assert_eq!(pushes.len(), self.envs);
        let (d, b) = (self.d, self.envs as u32);
        let x0: Vec<f64> = pushes.iter().flat_map(|x| x.iter().copied()).collect();
        self.xs[0].write(device, &self.encode(&x0));
        let p32: Vec<f32> = pol.params.iter().map(|&p| p as f32).collect();
        self.params.write(device, &p32);
        self.grad.write(device, &vec![0.0f32; p32.len()]);
        self.lam
            .write(device, &vec![0.0f32; self.envs * d.nout * d.w]);
        let n = |k: usize| (k as u32) * b;
        let lw = |l: usize| {
            let layer = [(d.no, d.nh), (d.nh, d.nh), (d.nh, d.nu)];
            ((layer[l].0 * layer[l].1 + layer[l].1) as u32) * WG
        };
        c.begin_batch();
        for t in 0..self.h {
            let (xt, xn) = (&self.xs[t], &self.xs[t + 1]);
            c.dispatch_buffers(
                device,
                &self.k.obs,
                &[xt, &self.obs[t]],
                &[1],
                &[n(d.no), 0, 0, 0],
                n(d.no),
                WG,
            );
            c.dispatch_buffers(
                device,
                &self.k.dense[0],
                &[&self.params, &self.obs[t], &self.h1[t]],
                &[2],
                &[n(d.nh), 0, 0, 0],
                n(d.nh),
                WG,
            );
            c.dispatch_buffers(
                device,
                &self.k.dense[1],
                &[&self.params, &self.h1[t], &self.h2[t]],
                &[2],
                &[n(d.nh), 0, 0, 0],
                n(d.nh),
                WG,
            );
            c.dispatch_buffers(
                device,
                &self.k.dense[2],
                &[&self.params, &self.h2[t], xt],
                &[2],
                &[n(d.nu), 0, 0, 0],
                n(d.nu),
                WG,
            );
            self.fwd.run(c, device, xt, &self.fwd_scratch, &self.out, b);
            c.dispatch_buffers(
                device,
                &self.k.copy_state,
                &[&self.out, xn],
                &[1],
                &[n(d.nout), 0, 0, 0],
                n(d.nout),
                WG,
            );
        }
        for t in (0..self.h).rev() {
            let (xt, xn) = (&self.xs[t], &self.xs[t + 1]);
            c.dispatch_buffers(
                device,
                &self.k.loss_grad,
                &[xn, &self.lam],
                &[1],
                &[n(d.ns), 0, 0, 0],
                n(d.ns),
                WG,
            );
            c.dispatch_buffers(
                device,
                &self.k.concat,
                &[xt, &self.lam, &self.xin],
                &[2],
                &[n(d.nx + d.nout), 0, 0, 0],
                n(d.nx + d.nout),
                WG,
            );
            self.adj
                .run(c, device, &self.xin, &self.adj_scratch, &self.back, b);
            c.dispatch_buffers(
                device,
                &self.k.gz3,
                &[&self.back, &self.gz3],
                &[1],
                &[n(d.nu), 0, 0, 0],
                n(d.nu),
                WG,
            );
            c.dispatch_buffers(
                device,
                &self.k.bwd_w[2],
                &[&self.gz3, &self.h2[t], &self.grad],
                &[2],
                &[lw(2), 0, 0, 0],
                lw(2),
                WG,
            );
            c.dispatch_buffers(
                device,
                &self.k.bwd_x[2],
                &[&self.params, &self.gz3, &self.h2[t], &self.gz2],
                &[3],
                &[n(d.nh), 0, 0, 0],
                n(d.nh),
                WG,
            );
            c.dispatch_buffers(
                device,
                &self.k.bwd_w[1],
                &[&self.gz2, &self.h1[t], &self.grad],
                &[2],
                &[lw(1), 0, 0, 0],
                lw(1),
                WG,
            );
            c.dispatch_buffers(
                device,
                &self.k.bwd_x[1],
                &[&self.params, &self.gz2, &self.h1[t], &self.gz1],
                &[3],
                &[n(d.nh), 0, 0, 0],
                n(d.nh),
                WG,
            );
            c.dispatch_buffers(
                device,
                &self.k.bwd_w[0],
                &[&self.gz1, &self.obs[t], &self.grad],
                &[2],
                &[lw(0), 0, 0, 0],
                lw(0),
                WG,
            );
            c.dispatch_buffers(
                device,
                &self.k.bwd_x[0],
                &[&self.params, &self.gz1, &self.obs[t], &self.gobs],
                &[3],
                &[n(d.no), 0, 0, 0],
                n(d.no),
                WG,
            );
            c.dispatch_buffers(
                device,
                &self.k.lam,
                &[&self.back, &self.gobs, &self.lam],
                &[2],
                &[n(d.nout), 0, 0, 0],
                n(d.nout),
                WG,
            );
        }
        c.end_batch(device);
        c.flush(device);
        let g = self.grad.to_vec_sync(device);
        g.iter().map(|&x| x as f64 / self.envs as f64).collect()
    }
}

/// A WGSL module-scope constant array.
fn wgsl_array(name: &str, v: &[f64]) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{:?}", *x as f32)).collect();
    format!(
        "const {name} = array<f32, {}>({});\n",
        v.len(),
        items.join(", ")
    )
}

/// Observation `(pose error without x, y; v)`: entry `j` reads state index
/// `j` (`j < 3`), `j + 2` (pose) or the velocity.
fn obs_kernel(d: Dims, rest: &str) -> String {
    let (nq, nx, no, t) = (d.nq, d.nx, d.no, d.t);
    format!("{rest}{}", retype(&header(&["x", "ob"], &[1]), &["x"], t))
        + &format!(
            "    let e = k / {no}u; let j = k % {no}u;\n\
             var i = j;\n\
             if (j >= 3u) {{ i = j + 2u; }}\n\
             if (i < {nq}u) {{ ob[k] = f32(x[e * {nx}u + i] - {t}(rest[i])); }} else {{ ob[k] = f32(x[e * {nx}u + i]); }}\n}}\n"
        )
}

/// Step output `(q', v', f)` into the next input `(q, v, ·, warm)`.
fn copy_state_kernel(d: Dims) -> String {
    let (ns, nu, nx, nout) = (d.ns, d.nu, d.nx, d.nout);
    retype(&header(&["o", "xn"], &[1]), &["o", "xn"], d.t)
        + &format!(
            "    let e = k / {nout}u; let i = k % {nout}u;\n\
             if (i < {ns}u) {{ xn[e * {nx}u + i] = o[k]; }} else {{ xn[e * {nx}u + {nu}u + i] = o[k]; }}\n}}\n"
        )
}

/// `λ += ∂(step loss)/∂s` on the state after the step (`Stand::step_loss`).
fn loss_grad_kernel(d: Dims, z0: f64, h: usize) -> String {
    let (nq, ns, nx, nout) = (d.nq, d.ns, d.nx, d.nout);
    let s = 1.0 / h as f64;
    retype(&header(&["xn", "lam"], &[1]), &["xn", "lam"], d.t)
        + &format!(
            "    let e = k / {ns}u; let i = k % {ns}u;\n\
             let x = xn[e * {nx}u + i];\n\
             var g: {t} = 0.0;\n\
             if (i == 5u) {{ g = {:?} * (x - {:?}); }}\n\
             else if (i < 2u) {{ g = {:?} * x; }}\n\
             else if (i >= {nq}u && i < {}u) {{ g = {:?} * x; }}\n\
             else if (i >= {}u) {{ g = {:?} * x; }}\n\
             lam[e * {nout}u + i] += g;\n}}\n",
            200.0 * s,
            z0,
            20.0 * s,
            nq + 6,
            2.0 * s,
            nq + 6,
            0.02 * s,
            t = d.t
        )
}

fn concat_kernel(d: Dims) -> String {
    let (nx, nout) = (d.nx, d.nout);
    let ni = nx + nout;
    retype(
        &header(&["x", "lam", "xin"], &[2]),
        &["x", "lam", "xin"],
        d.t,
    ) + &format!(
        "    let e = k / {ni}u; let j = k % {ni}u;\n\
             if (j < {nx}u) {{ xin[k] = x[e * {nx}u + j]; }} else {{ xin[k] = lam[e * {nout}u + j - {nx}u]; }}\n}}\n"
    )
}

/// The head's pre-activation gradient: `OFFSET ∂/∂ctrl`.
fn gz3_kernel(d: Dims) -> String {
    let (nu, ns, nx) = (d.nu, d.ns, d.nx);
    retype(&header(&["back", "gz"], &[1]), &["back"], d.t)
        + &format!(
            "    let e = k / {nu}u; let j = k % {nu}u;\n\
             gz[k] = f32({OFFSET:?} * back[e * {nx}u + {ns}u + j]);\n}}\n"
        )
}

/// `λ` for the step input state `(q, v, warm)`: the adjoint's `∂/∂q, v` and
/// `∂/∂warm`, plus the observation's gradient (no x, y entries).
fn lam_kernel(d: Dims) -> String {
    let (ns, nu, nx, no, nout, t) = (d.ns, d.nu, d.nx, d.no, d.nout, d.t);
    retype(
        &header(&["back", "gobs", "lam"], &[2]),
        &["back", "lam"],
        d.t,
    ) + &format!(
        "    let e = k / {nout}u; let i = k % {nout}u;\n\
             if (i >= {ns}u) {{ lam[k] = back[e * {nx}u + {nu}u + i]; return; }}\n\
             var g = back[e * {nx}u + i];\n\
             if (i < 3u) {{ g += {t}(gobs[e * {no}u + i]); }} else if (i >= 5u) {{ g += {t}(gobs[e * {no}u + i - 2u]); }}\n\
             lam[k] = g;\n}}\n"
    )
}

/// Declare the storage buffers `names` of a kernel header as `array<t>`.
fn retype(src: &str, names: &[&str], t: &str) -> String {
    let mut s = src.to_string();
    for n in names {
        s = s.replace(
            &format!("> {n}: array<f32>;"),
            &format!("> {n}: array<{t}>;"),
        );
    }
    s
}
