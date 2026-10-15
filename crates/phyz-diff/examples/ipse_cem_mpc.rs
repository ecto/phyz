//! Exact gradients on top of ipse's CEM stepping policy: residual MPC.
//!
//! The CEM policy (`ipse_cem`: survives 18 N·s sagittal in the traced plant,
//! as in ipse) keeps running closed-loop; every `MPC_REPLAN` steps an L-BFGS
//! solve plans offsets added to its ten actions over `MPC_H` knots of
//! `MPC_KNOT` steps. Inside the planner's rollouts the policy also runs
//! closed-loop; the gradient holds its feedback frozen along each rollout
//! (the CEM observation is not traced), and the line search scores the true
//! closed-loop cost, so a step is only taken if it really helps. Scored on
//! the largest shove survived per direction, CEM alone against CEM + MPC.
//!
//! `cargo run --release -p phyz-diff --example ipse_cem_mpc RIG.txt CEM.policy [ITERS]`

#[path = "common/ipse_rig.rs"]
mod ipse_rig;
#[path = "mpc_stand.rs"]
#[allow(dead_code)]
mod mpc;

use phyz_contact::ContactSolverConfig;
use phyz_diff::contact_step::ContactStep;
use tang_expr::compile::CompiledMany;

const SETTLE: usize = 1500;
const SHOVE_AT: usize = 1700;
const AFTER: usize = 2000;

fn main() {
    for (k, v) in [
        ("MPC_H", "10"),
        ("MPC_KNOT", "20"),
        ("MPC_REPLAN", "20"),
        ("MPC_CFLAGS", "-O2 -march=native"),
        ("STAND_H", "600"),
    ] {
        if std::env::var_os(k).is_none() {
            unsafe { std::env::set_var(k, v) };
        }
    }
    let mut args = std::env::args().skip(1);
    let rig = ipse_rig::Rig::read(
        &args
            .next()
            .expect("usage: ipse_cem_mpc RIG.txt CEM.policy [ITERS]"),
    );
    let params = ipse_rig::read_cem(&args.next().expect("CEM.policy"));
    let iters: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(3);
    let model = rig.model();
    let pd = rig.pd();
    let step = ContactStep::build_pd(
        &model,
        &rig.points(),
        &rig.material(),
        &ContactSolverConfig::default(),
        5,
        &pd,
    );
    let (ag, ao) = step.adjoint();
    let fwd = mpc::native(&step.kernel_c_f64().source, step.n_outputs());
    let back = mpc::native(
        &ag.to_c_f64(&ao, step.n_inputs() + step.n_outputs()).source,
        ao.len(),
    );
    let dofs = ipse_rig::cem_dofs(&model);
    let slot: Vec<usize> = dofs
        .iter()
        .map(|d| pd.dofs.iter().position(|p| p == d).unwrap())
        .collect();
    let p = Plant {
        rig: &rig,
        model: &model,
        fwd,
        back,
        params,
        dofs,
        slot,
        nu: step.nu,
    };
    let (h, replan) = (mpc::env("MPC_H", 10), mpc::env("MPC_REPLAN", 20));
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let dirs = [
        ("forward", [1.0, 0.0, 0.0]),
        ("backward", [-1.0, 0.0, 0.0]),
        ("left", [0.0, 1.0, 0.0]),
        ("right", [0.0, -1.0, 0.0]),
    ];
    let levels: Vec<f64> = (2..=13).map(|k| 2.0 * k as f64).collect();
    let jobs: Vec<(usize, f64, bool)> = (0..dirs.len())
        .flat_map(|d| {
            levels
                .iter()
                .flat_map(move |&n| [(d, n, false), (d, n, true)])
        })
        .collect();
    let chunk = jobs.len().div_ceil(threads);
    let res: Vec<(usize, f64, bool, bool)> = std::thread::scope(|sc| {
        let hs: Vec<_> = jobs
            .chunks(chunk)
            .map(|c| {
                let (p, dirs) = (&p, &dirs);
                sc.spawn(move || {
                    c.iter()
                        .map(|&(d, n, m)| {
                            (
                                d,
                                n,
                                m,
                                p.episode(
                                    dirs[d].1.map(|u| u * n),
                                    m.then_some((h, replan, iters)),
                                ),
                            )
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let (mut tc, mut tm) = (0.0, 0.0);
    for (d, (name, _)) in dirs.iter().enumerate() {
        let best = |m: bool| {
            let mut last = 0.0;
            for &n in &levels {
                if !res.iter().any(|r| r.0 == d && r.1 == n && r.2 == m && r.3) {
                    break;
                }
                last = n;
            }
            last
        };
        let row: String = levels
            .iter()
            .map(|&n| {
                let ok = |m: bool| res.iter().any(|r| r.0 == d && r.1 == n && r.2 == m && r.3);
                format!(
                    " {n:.0}:{}{}",
                    if ok(false) { "C" } else { "." },
                    if ok(true) { "M" } else { "." }
                )
            })
            .collect();
        println!(
            "{name:>9}: CEM {:>4.0} N·s, CEM + MPC {:>4.0} N·s  |{row}",
            best(false),
            best(true)
        );
        tc += best(false) / 4.0;
        tm += best(true) / 4.0;
    }
    println!(
        "mean largest shove survived: CEM {tc:.1} N·s, CEM + residual MPC {tm:.1} N·s ({iters} iters, {h} knots x {}, replan {replan})",
        mpc::knot()
    );
}

/// The residual's bound per DOF, rad (`CEMMPC_BOUND`, default 0.1): a trust
/// region around the CEM policy's own action.
fn residual_bound() -> f64 {
    std::env::var("CEMMPC_BOUND")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.1)
}

struct Plant<'a> {
    rig: &'a ipse_rig::Rig,
    model: &'a phyz_model::Model,
    fwd: CompiledMany,
    back: CompiledMany,
    params: Vec<f64>,
    dofs: Vec<usize>,
    /// Where each CEM DOF sits among the PD targets.
    slot: Vec<usize>,
    nu: usize,
}

impl Plant<'_> {
    fn dims(&self) -> (usize, usize, usize, usize) {
        (
            self.model.nq,
            self.model.nv,
            self.nu - 6,
            3 * self.rig.points().len(),
        )
    }

    fn start(&self) -> Vec<f64> {
        let (_, _, _, nf) = self.dims();
        let pd = self.rig.pd();
        let mut x: Vec<f64> = self.rig.q0.iter().chain(&self.rig.v0).copied().collect();
        x.extend(pd.dofs.iter().map(|&d| self.rig.q_target[d]));
        x.extend([0.0; 6]);
        x.extend(std::iter::repeat_n(0.0, nf));
        x
    }

    /// Write the targets for this step: nominal + CEM's action (if `cem`) +
    /// the residual `res` (10, already bounded); return the CEM action.
    fn set_targets(&self, x: &mut [f64], cem: bool, res: &[f64]) -> Vec<f64> {
        let (nq, nv, _, _) = self.dims();
        let pd = self.rig.pd();
        for (s, &d) in pd.dofs.iter().enumerate() {
            x[nq + nv + s] = self.rig.q_target[d];
        }
        let a = if cem {
            let o = ipse_rig::observe_legs(
                self.model,
                &self.rig.q_target,
                &x[..nq],
                &x[nq..nq + nv],
                &self.dofs,
            );
            ipse_rig::cem_action(&self.params, &o)
        } else {
            vec![0.0; 10]
        };
        for i in 0..10 {
            x[nq + nv + self.slot[i]] += a[i] + res[i];
        }
        a
    }

    fn advance(&self, x: &mut [f64], out: &mut [f64]) {
        let (nq, nv, _, _) = self.dims();
        (self.fwd)(x, out);
        x[..nq + nv].copy_from_slice(&out[..nq + nv]);
        x[nq + nv + self.nu..].copy_from_slice(&out[nq + nv..]);
    }

    /// One shove: does it stay up (trunk above 0.35 m) for 2 s after?
    fn episode(&self, impulse: [f64; 3], mpc_cfg: Option<(usize, usize, usize)>) -> bool {
        let (nq, nv, nt, nf) = (self.dims().0, self.dims().1, self.dims().2, self.dims().3);
        let kn = mpc::knot();
        let (h, replan, iters) = mpc_cfg.unwrap_or((10, 20, 0));
        let shift = (replan / kn).min(h);
        let mut plan = vec![0.0; h * 10];
        let mut x = self.start();
        let mut out = vec![0.0; nq + nv + nf];
        let zero = [0.0; 10];
        for k in 0..SHOVE_AT + AFTER {
            let cem = k >= SETTLE;
            // MPC only once the robot has been shoved and the CEM is driving
            let mpc_on = mpc_cfg.is_some() && k >= SHOVE_AT;
            if mpc_on && (k - SHOVE_AT).is_multiple_of(replan) {
                if k > SHOVE_AT && shift < h {
                    plan.copy_within(shift * 10.., 0);
                    let tail: Vec<f64> = plan[(h - shift - 1) * 10..(h - shift) * 10].to_vec();
                    for j in h - shift..h {
                        plan[j * 10..(j + 1) * 10].copy_from_slice(&tail);
                    }
                }
                let x0 = x.clone();
                mpc::minimize(
                    &|z: &[f64], g: Option<&mut [f64]>| self.objective(&x0, z, h, g),
                    &mut plan,
                    iters,
                );
            }
            let res: Vec<f64> = if mpc_on {
                let kk = ((k - SHOVE_AT) % replan) / kn;
                (0..10)
                    .map(|i| residual_bound() * plan[kk * 10 + i].tanh())
                    .collect()
            } else {
                zero.to_vec()
            };
            self.set_targets(&mut x, cem, &res);
            let w = &mut x[nq + nv + nt..nq + nv + nt + 6];
            w.iter_mut().for_each(|v| *v = 0.0);
            if (SHOVE_AT..SHOVE_AT + 20).contains(&k) {
                for a in 0..3 {
                    w[3 + a] = impulse[a] / 0.02;
                }
            }
            self.advance(&mut x, &mut out);
            if !x.iter().all(|v| v.is_finite()) || x[5] < 0.35 {
                return false;
            }
        }
        true
    }

    /// Closed-loop cost of the residual plan from `x0` (CEM running, no
    /// shove), and its gradient with the CEM feedback frozen.
    fn objective(&self, x0: &[f64], plan: &[f64], h: usize, grad: Option<&mut [f64]>) -> f64 {
        let (nq, nv, nt, nf) = self.dims();
        let ns = nq + nv;
        let ni = ns + self.nu + nf;
        let kn = mpc::plan_knot();
        let steps = h * kn;
        let mut xs = vec![vec![0.0; ni]; steps + 1];
        let mut out = vec![0.0; ns + nf];
        let mut x = x0.to_vec();
        x[ns + nt..ns + nt + 6].iter_mut().for_each(|w| *w = 0.0);
        let mut cost = 0.0;
        let aligned = std::env::var_os("CEMMPC_OLD_LOSS").is_none();
        let loss = |x: &[f64], g: Option<&mut [f64]>| -> f64 {
            let z0 = self.rig.q0[5] - 0.0022;
            let (dz, tilt) = (x[5] - z0, x[0] * x[0] + x[1] * x[1]);
            if aligned {
                // ipse's CEM objective, in spirit: upright, level, not
                // rotating; height only when it is lost; the base may move
                // (a recovery step moves it)
                let low = dz.min(0.0);
                let w: f64 = x[nq..nq + 3].iter().map(|v| v * v).sum();
                if let Some(g) = g {
                    g[5] += 200.0 * low;
                    g[0] += 20.0 * x[0];
                    g[1] += 20.0 * x[1];
                    for i in 0..3 {
                        g[nq + i] += 0.2 * x[nq + i];
                    }
                }
                return 100.0 * low * low + 10.0 * tilt + 0.1 * w;
            }
            // trunk height, tilt and base speed, as the other benches score
            let vb: f64 = x[nq..nq + 6].iter().map(|v| v * v).sum();
            if let Some(g) = g {
                g[5] += 200.0 * dz;
                g[0] += 20.0 * x[0];
                g[1] += 20.0 * x[1];
                for i in 0..6 {
                    g[nq + i] += 2.0 * x[nq + i];
                }
            }
            100.0 * dz * dz + 10.0 * tilt + vb
        };
        for k in 0..steps {
            let res: Vec<f64> = (0..10)
                .map(|i| residual_bound() * plan[(k / kn) * 10 + i].tanh())
                .collect();
            self.set_targets(&mut x, true, &res);
            xs[k].copy_from_slice(&x);
            (self.fwd)(&x, &mut out);
            x[..ns].copy_from_slice(&out[..ns]);
            x[ns + self.nu..].copy_from_slice(&out[ns..]);
            cost += loss(&x, None);
            if !x.iter().all(|v| v.is_finite()) {
                return f64::INFINITY;
            }
        }
        xs[steps].copy_from_slice(&x);
        let Some(grad) = grad else { return cost };
        grad.iter_mut().for_each(|g| *g = 0.0);
        let mut xin = vec![0.0; ni + ns + nf];
        let mut back = vec![0.0; ni];
        let mut lam = vec![0.0; ns + nf];
        let mut gx = vec![0.0; ni];
        for k in (0..steps).rev() {
            gx.iter_mut().for_each(|g| *g = 0.0);
            loss(&xs[k + 1], Some(&mut gx));
            for i in 0..ns {
                lam[i] += gx[i];
            }
            xin[..ni].copy_from_slice(&xs[k]);
            xin[ni..].copy_from_slice(&lam);
            (self.back)(&xin, &mut back);
            for i in 0..10 {
                let j = (k / kn) * 10 + i;
                let th = plan[j].tanh();
                grad[j] += back[ns + self.slot[i]] * residual_bound() * (1.0 - th * th);
            }
            lam[..ns].copy_from_slice(&back[..ns]);
            lam[ns..].copy_from_slice(&back[ns + self.nu..]);
        }
        cost
    }
}
