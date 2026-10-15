//! Push recovery on ipse's K1 plant: PD against gradient MPC, in N·s.
//!
//! The plant is [`ContactStep::build_pd`] from an ipse rig dump
//! (`ipse_parity` checks it tracks ipse-sim to microns while the robot
//! stands). MPC plans joint-target offsets on the 22 PD DOFs (legs within
//! ±0.3 rad, upper body ±0.09, ipse's clamps) with L-BFGS through the step's
//! adjoint, `MPC_H` knots of `MPC_KNOT` steps, replanning every
//! `MPC_REPLAN`. The base wrench input is the shove, applied as ipse applies
//! it (20 ms, trunk frame) and never planned. For each direction the shove
//! is raised until the controller falls (trunk below 0.35 m within 2 s,
//! ipse's alive threshold); the score is the largest it survives.
//!
//! `cargo run --release -p phyz-diff --example ipse_mpc RIG.txt [ITERS]`

#[path = "common/ipse_rig.rs"]
mod ipse_rig;
#[path = "mpc_stand.rs"]
#[allow(dead_code)]
mod mpc;

use phyz_contact::ContactSolverConfig;
use phyz_diff::contact_step::ContactStep;
use tang_expr::compile::CompiledMany;

const STEPS: usize = 2000; // 2 s at 1 ms
const SHOVE_AT: usize = 200;

fn main() {
    for (k, v) in [
        ("MPC_H", "12"),
        ("MPC_KNOT", "4"),
        ("MPC_REPLAN", "8"),
        ("MPC_CFLAGS", "-O2 -march=native"),
    ] {
        if std::env::var_os(k).is_none() {
            unsafe { std::env::set_var(k, v) };
        }
    }
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: ipse_mpc RIG.txt [ITERS]");
    let iters: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(3);
    let rig = ipse_rig::Rig::read(&path);
    let model = rig.model();
    let config = ContactSolverConfig::default();
    let pd = rig.pd();
    let step = ContactStep::build_pd(&model, &rig.points(), &rig.material(), &config, 5, &pd);
    let (ag, ao) = step.adjoint();
    let fwd = mpc::native(&step.kernel_c_f64().source, step.n_outputs());
    let back = mpc::native(
        &ag.to_c_f64(&ao, step.n_inputs() + step.n_outputs()).source,
        ao.len(),
    );
    let leg: Vec<bool> = pd.dofs.iter().map(|&d| rig.kp[d] > 100.0).collect();
    let plant = Plant::new(&rig, fwd, back, &pd.dofs, leg);
    eprintln!(
        "ipse K1: {} targets, step {} + adjoint {} nodes; standing trunk {:.1} mm",
        plant.nt,
        step.live_nodes(),
        ag.live_set(&ao).len(),
        plant.z0 * 1e3
    );
    let (h, replan) = (mpc::env("MPC_H", 12), mpc::env("MPC_REPLAN", 8));
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let dirs: [(&str, [f64; 3]); 4] = [
        ("forward", [1.0, 0.0, 0.0]),
        ("backward", [-1.0, 0.0, 0.0]),
        ("left", [0.0, 1.0, 0.0]),
        ("right", [0.0, -1.0, 0.0]),
    ];
    let levels: Vec<f64> = (2..=12).map(|k| 2.0 * k as f64).collect(); // 4..24 N·s
    let jobs: Vec<(usize, f64, bool)> = (0..dirs.len())
        .flat_map(|d| {
            levels
                .iter()
                .flat_map(move |&n| [(d, n, false), (d, n, true)])
        })
        .collect();
    let chunk = jobs.len().div_ceil(threads);
    let results: Vec<(usize, f64, bool, bool, f64)> = std::thread::scope(|sc| {
        let hs: Vec<_> = jobs
            .chunks(chunk)
            .map(|c| {
                let (plant, dirs) = (&plant, &dirs);
                sc.spawn(move || {
                    c.iter()
                        .map(|&(d, n, use_mpc)| {
                            let imp = dirs[d].1.map(|u| u * n);
                            let (alive, drop) =
                                plant.episode(imp, use_mpc.then_some((h, replan, iters)));
                            (d, n, use_mpc, alive, drop)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let best = |d: usize, m: bool| {
        // largest level survived with every smaller one survived too
        let mut last = 0.0;
        for &n in &levels {
            let ok = results
                .iter()
                .any(|r| r.0 == d && r.1 == n && r.2 == m && r.3);
            if !ok {
                break;
            }
            last = n;
        }
        last
    };
    for (d, (name, _)) in dirs.iter().enumerate() {
        eprintln!(
            "{name:>9}: PD survives {:>4.0} N·s, MPC {:>4.0} N·s",
            best(d, false),
            best(d, true)
        );
        for &n in &levels {
            let get = |m: bool| {
                results
                    .iter()
                    .find(|r| r.0 == d && r.1 == n && r.2 == m)
                    .map(|r| (r.3, r.4))
                    .unwrap()
            };
            let (p, q) = (get(false), get(true));
            eprintln!(
                "    {n:>4.0} N·s  PD {} drop {:>6.0} mm   MPC {} drop {:>6.0} mm",
                if p.0 { "ok  " } else { "FALL" },
                p.1 * 1e3,
                if q.0 { "ok  " } else { "FALL" },
                q.1 * 1e3
            );
        }
    }
    let pd_total: f64 = (0..dirs.len()).map(|d| best(d, false)).sum::<f64>() / dirs.len() as f64;
    let mpc_total: f64 = (0..dirs.len()).map(|d| best(d, true)).sum::<f64>() / dirs.len() as f64;
    println!(
        "ipse plant, mean largest shove survived over 4 directions: PD {pd_total:.1} N·s, MPC {mpc_total:.1} N·s ({iters} iters, {h} knots x {}, replan {replan})",
        mpc::knot()
    );
}

/// ipse's plant in the shape the MPC solver wants: `Stand`-like, with the
/// controls split into planned targets and the external base wrench.
struct Plant {
    task: mpc::stand::Stand,
    nt: usize,
    z0: f64,
    targets: Vec<f64>,
    /// Per target: the offset bound (legs 0.3 rad, upper body 0.09).
    bounds: Vec<f64>,
}

impl Plant {
    fn new(
        rig: &ipse_rig::Rig,
        fwd: CompiledMany,
        back: CompiledMany,
        dofs: &[usize],
        leg: Vec<bool>,
    ) -> Self {
        let (nq, nv) = (rig.q0.len(), rig.v0.len());
        let nt = dofs.len();
        let nf = 3 * rig.points().len();
        let targets: Vec<f64> = dofs.iter().map(|&d| rig.q_target[d]).collect();
        // settle the drop transient under PD before anything is scored
        let mut x: Vec<f64> = rig.q0.iter().chain(&rig.v0).copied().collect();
        x.extend(&targets);
        x.extend([0.0; 6]);
        x.extend(std::iter::repeat_n(0.0, nf));
        let mut out = vec![0.0; nq + nv + nf];
        for _ in 0..1500 {
            fwd(&x, &mut out);
            x[..nq + nv].copy_from_slice(&out[..nq + nv]);
            x[nq + nv + nt + 6..].copy_from_slice(&out[nq + nv..]);
        }
        let z0 = x[5];
        let task = mpc::stand::Stand {
            nq,
            nv,
            nu: nt + 6,
            nf,
            fwd,
            back,
            rest: x,
            z0,
        };
        // IPSE_LEG_BOUND (default 0.3, ipse's clamp; crouch-era policies 0.8);
        // the upper body gets 0.3x, as in ipse
        let leg_bound: f64 = std::env::var("IPSE_LEG_BOUND")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.3);
        let bounds = leg
            .iter()
            .map(|&l| if l { leg_bound } else { 0.3 * leg_bound })
            .collect();
        Self {
            task,
            nt,
            z0,
            targets,
            bounds,
        }
    }

    /// One shove (impulse in N·s, trunk frame) under PD or MPC: whether the
    /// trunk stayed above 0.35 m, and its largest drop.
    fn episode(&self, impulse: [f64; 3], mpc_cfg: Option<(usize, usize, usize)>) -> (bool, f64) {
        let t = &self.task;
        let (nq, nv, nt, nf) = (t.nq, t.nv, self.nt, t.nf);
        let ns = nq + nv;
        let kn = mpc::knot();
        let (h, replan, iters) = mpc_cfg.unwrap_or((12, 8, 0));
        let shift = (replan / kn).min(h);
        let mut plan = vec![0.0; h * nt];
        let mut x = t.rest.clone();
        let mut out = vec![0.0; ns + nf];
        let mut lowest = x[5];
        for k in 0..STEPS {
            if mpc_cfg.is_some() && k % replan == 0 {
                if k > 0 && shift < h {
                    plan.copy_within(shift * nt.., 0);
                    let tail: Vec<f64> = plan[(h - shift - 1) * nt..(h - shift) * nt].to_vec();
                    for j in h - shift..h {
                        plan[j * nt..(j + 1) * nt].copy_from_slice(&tail);
                    }
                }
                let x0 = x.clone();
                mpc::minimize(
                    &|z: &[f64], g: Option<&mut [f64]>| self.objective(&x0, z, h, g),
                    &mut plan,
                    iters,
                );
            }
            let kk = (k % replan) / kn;
            for i in 0..nt {
                let off = if mpc_cfg.is_some() {
                    self.bounds[i] * plan[kk * nt + i].tanh()
                } else {
                    0.0
                };
                x[ns + i] = self.targets[i] + off;
            }
            let wrench = &mut x[ns + nt..ns + nt + 6];
            wrench.iter_mut().for_each(|w| *w = 0.0);
            if (SHOVE_AT..SHOVE_AT + 20).contains(&k) {
                for a in 0..3 {
                    wrench[3 + a] = impulse[a] / 0.02;
                }
            }
            (t.fwd)(&x, &mut out);
            x[..ns].copy_from_slice(&out[..ns]);
            x[ns + nt + 6..].copy_from_slice(&out[ns..]);
            if !x.iter().all(|v| v.is_finite()) {
                return (false, f64::INFINITY);
            }
            lowest = lowest.min(x[5]);
            if x[5] < 0.35 {
                return (false, self.z0 - lowest);
            }
        }
        (true, self.z0 - lowest)
    }

    /// The window's cost from `x0` under `plan` (h knots x nt raw offsets),
    /// no shove (the planner does not know one is coming), and its gradient.
    fn objective(&self, x0: &[f64], plan: &[f64], h: usize, grad: Option<&mut [f64]>) -> f64 {
        let t = &self.task;
        let (nq, nv, nt, nf) = (t.nq, t.nv, self.nt, t.nf);
        let (ns, nu) = (nq + nv, nt + 6);
        let ni = ns + nu + nf;
        let kn = mpc::plan_knot();
        let steps = h * kn;
        let weight = |k: usize| if k + 1 == steps { 5.0 } else { 1.0 };
        let mut xs = vec![vec![0.0; ni]; steps + 1];
        let mut out = vec![0.0; ns + nf];
        let mut x = x0.to_vec();
        x[ns + nt..ns + nt + 6].iter_mut().for_each(|w| *w = 0.0);
        let mut cost = 0.0;
        for k in 0..steps {
            for i in 0..nt {
                x[ns + i] = self.targets[i] + self.bounds[i] * plan[(k / kn) * nt + i].tanh();
            }
            (t.fwd)(&x, &mut out);
            xs[k].copy_from_slice(&x);
            x[..ns].copy_from_slice(&out[..ns]);
            x[ns + nu..].copy_from_slice(&out[ns..]);
            cost += weight(k) * t.step_loss(&x, None);
        }
        xs[steps].copy_from_slice(&x);
        let Some(grad) = grad else { return cost };
        grad.iter_mut().for_each(|g| *g = 0.0);
        let mut xin = vec![0.0; ni + ns + nf];
        let mut back = vec![0.0; ni];
        let mut gx = vec![0.0; ni];
        let mut lam = vec![0.0; ns + nf];
        for k in (0..steps).rev() {
            gx.iter_mut().for_each(|g| *g = 0.0);
            t.step_loss(&xs[k + 1], Some(&mut gx));
            for i in 0..ns {
                lam[i] += weight(k) * gx[i];
            }
            xin[..ni].copy_from_slice(&xs[k]);
            xin[ni..].copy_from_slice(&lam);
            (t.back)(&xin, &mut back);
            for i in 0..nt {
                let j = (k / kn) * nt + i;
                let th = plan[j].tanh();
                grad[j] += back[ns + i] * self.bounds[i] * (1.0 - th * th);
            }
            lam[..ns].copy_from_slice(&back[..ns]);
            lam[ns..].copy_from_slice(&back[ns + nu..]);
        }
        cost
    }
}
