//! Distil the push-recovery MPC into a policy (DAgger).
//!
//! The teacher is `mpc_stand`'s gradient MPC at the setting that holds every
//! recoverable shove in `mpc_robust` (L-BFGS, 12 knots of 4 steps, 3
//! iterations, a 3-sweep planner model, replanning every 8 steps): ~35 ms a
//! solve, too slow for 1 kHz, fine offline. The student is an MLP from the
//! observation (pose error, velocities) to the bounded joint-target offsets
//! the teacher would apply now, run every step: microseconds.
//!
//! Round 0 labels the teacher's own rollouts. Each later round rolls the
//! student (the teacher warm-started alongside it, labelling the states the
//! student visits every 8 steps), adds those labels to the set, and refits.
//! Scored on `mpc_robust`'s 20 shoves against PD and the teacher.
//!
//! `cargo run --release -p phyz-diff --example mpc_distill [ROUNDS]`

#[path = "mpc_stand.rs"]
#[allow(dead_code)]
mod mpc;

use std::time::Instant;

use mpc::stand::reach::{Adam, Lcg, Mlp, TORQUE};
use mpc::stand::{SWEEPS, Stand, k1};
use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::contact_step::ContactStep;

const STEPS: usize = 1500;
const EPISODES: usize = 32;
const HIDDEN: usize = 256;

/// What the student sees: the task's observation (pose error, velocities)
/// plus the contact solve's warm start (last step's impulses: which sole
/// points carry load), which the teacher's plan depends on too.
fn see(task: &Stand, x: &[f64]) -> Vec<f64> {
    let mut o = task.obs(x);
    let off = task.nq + task.nv + task.nu;
    o.extend(x[off..off + task.nf].iter().map(|f| f * 10.0));
    o
}

/// The teacher's knobs, unless the caller set them.
fn teacher_defaults() {
    for (k, v) in [
        ("STAND_H", "600"),
        ("MPC_H", "12"),
        ("MPC_KNOT", "4"),
        ("MPC_REPLAN", "8"),
        ("MPC_CFLAGS", "-O2 -march=native"),
    ] {
        if std::env::var_os(k).is_none() {
            unsafe { std::env::set_var(k, v) };
        }
    }
}

fn main() {
    teacher_defaults();
    let rounds: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(6);
    let (h, replan, iters) = (
        mpc::env("MPC_H", 12),
        mpc::env("MPC_REPLAN", 8),
        mpc::env("MPC_ITERS", 3),
    );
    let (model, points) = k1();
    let mat = ContactMaterial {
        friction: 0.8,
        margin: 0.005,
        ..ContactMaterial::default()
    };
    let config = ContactSolverConfig::default();
    let build = |sweeps: usize| {
        let st = ContactStep::build(&model, &points, &mat, &config, sweeps);
        let (ag, ao) = st.adjoint();
        let fwd = mpc::native(&st.kernel_c_f64().source, st.n_outputs());
        let back = mpc::native(
            &ag.to_c_f64(&ao, st.n_inputs() + st.n_outputs()).source,
            ao.len(),
        );
        Stand::new(&model, fwd, back, st.nu, points.len())
    };
    let plant = build(SWEEPS);
    let mut planner = build(mpc::env("MPC_MODEL_SWEEPS", 3));
    planner.rest = plant.rest.clone();
    planner.z0 = plant.z0;
    let ctx = Ctx {
        plant: &plant,
        planner: &planner,
        h,
        replan,
        iters,
    };

    // training shoves: random strength x0.8-1.4 and direction (roll, side)
    let mut rng = Lcg(31);
    let shove = |rng: &mut Lcg| {
        let s = 1.1 + 0.3 * rng.next();
        let mut x = plant.rest.clone();
        x[plant.nq] += 1.2 * s * rng.next().signum();
        x[plant.nq + 4] += 0.5 * s * rng.next().signum();
        x
    };
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut student = Mlp::new(plant.n_obs() + plant.nf, HIDDEN, plant.nu, &mut rng);
    // DISTILL_LOAD: refit on a saved set instead of collecting;
    // DISTILL_DATA: where to save the set (labels are the slow part)
    let mut data: Vec<(Vec<f64>, Vec<f64>)> = std::env::var("DISTILL_LOAD")
        .map(|p| load(&p))
        .unwrap_or_default();
    let collect = std::env::var_os("DISTILL_LOAD").is_none();
    let t = Instant::now();
    for round in 0..if collect { rounds } else { 0 } {
        let starts: Vec<Vec<f64>> = (0..EPISODES).map(|_| shove(&mut rng)).collect();
        // round 0 follows the teacher; later rounds the student
        let use_student = round > 0;
        let chunk = starts.len().div_ceil(threads);
        let labels: Vec<Vec<(Vec<f64>, Vec<f64>)>> = std::thread::scope(|sc| {
            let hs: Vec<_> = starts
                .chunks(chunk)
                .map(|c| {
                    let (ctx, student) = (&ctx, &student);
                    sc.spawn(move || {
                        c.iter()
                            .map(|x0| ctx.episode(x0, use_student.then_some(student), true).labels)
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        for l in labels {
            data.extend(l);
        }
        let loss = fit(&mut student, &data, 30, &mut rng);
        eprintln!(
            "round {round}: {} labels, fit loss {loss:.2e} (rad^2 per joint)  {:.0}s",
            data.len(),
            t.elapsed().as_secs_f64()
        );
    }
    if let Ok(p) = std::env::var("DISTILL_DATA") {
        save(&p, &data);
    }
    // a longer final fit on everything (DISTILL_EPOCHS, default 120)
    let loss = fit(
        &mut student,
        &data,
        mpc::env("DISTILL_EPOCHS", 120),
        &mut rng,
    );
    eprintln!("final fit on {} labels: loss {loss:.2e}", data.len());

    // the benchmark's 20 shoves
    let mut cases = Vec::new();
    for s in [1.0, 1.1, 1.2, 1.3, 1.4] {
        for (roll, side) in [(1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)] {
            let mut x = plant.rest.clone();
            x[plant.nq] += 1.2 * s * roll;
            x[plant.nq + 4] += 0.5 * s * side;
            cases.push(x);
        }
    }
    let chunk = cases.len().div_ceil(threads);
    let scored: Vec<(f64, f64, f64)> = std::thread::scope(|sc| {
        let hs: Vec<_> = cases
            .chunks(chunk)
            .map(|c| {
                let (ctx, student) = (&ctx, &student);
                sc.spawn(move || {
                    c.iter()
                        .map(|x0| {
                            let pd = ctx.pd_drop(x0);
                            let st = ctx.episode(x0, Some(student), false).drop;
                            let te = ctx.episode(x0, None, false).drop;
                            (pd, st, te)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let held = |d: f64| d < 0.03;
    let count = |k: usize| scored.iter().filter(|r| held([r.0, r.1, r.2][k])).count();
    let dirs = ["+roll +side", "-roll -side", "+roll -side", "-roll +side"];
    for (k, r) in scored.iter().enumerate() {
        eprintln!(
            "  x{:.1} {:12} drop PD {:5.0} mm  student {:5.0} mm  teacher {:5.0} mm",
            1.0 + 0.1 * (k / 4) as f64,
            dirs[k % 4],
            r.0 * 1e3,
            r.1 * 1e3,
            r.2 * 1e3
        );
    }
    // student cost per step
    let obs = see(&plant, &plant.rest);
    let n = 20000;
    let ts = Instant::now();
    for _ in 0..n {
        std::hint::black_box(student.forward(std::hint::black_box(&obs)));
    }
    let us = ts.elapsed().as_secs_f64() * 1e6 / n as f64;
    println!(
        "distilled over {rounds} rounds ({} labels): student holds {}/20, teacher {}/20, PD {}/20; student {us:.1} us per step vs teacher ~{:.0} ms per solve",
        data.len(),
        count(1),
        count(2),
        count(0),
        35.0
    );
}

struct Ctx<'a> {
    plant: &'a Stand,
    planner: &'a Stand,
    h: usize,
    replan: usize,
    iters: usize,
}

struct Episode {
    labels: Vec<(Vec<f64>, Vec<f64>)>,
    drop: f64,
}

impl Ctx<'_> {
    /// One shove, 1.5 s. Acting: the student every step if given, else the
    /// teacher. With `label`, the teacher solves every `replan` steps from
    /// the current state (warm-started) and its first knot is the label.
    fn episode(&self, x0: &[f64], student: Option<&Mlp>, label: bool) -> Episode {
        let task = self.plant;
        let (nq, nv, nu, nf) = (task.nq, task.nv, task.nu, task.nf);
        let ns = nq + nv;
        let (h, replan, kn) = (self.h, self.replan, mpc::knot());
        let shift = (replan / kn).min(h);
        let mut plan = vec![0.0; h * nu];
        let mut x = x0.to_vec();
        let mut out = vec![0.0; ns + nf];
        let mut labels = Vec::new();
        let mut lowest = x[5];
        let mut off = vec![0.0; nu];
        for t in 0..STEPS {
            let teacher_needed = student.is_none() || label;
            if teacher_needed && t % replan == 0 {
                if t > 0 && shift < h {
                    plan.copy_within(shift * nu.., 0);
                    let tail: Vec<f64> = plan[(h - shift - 1) * nu..(h - shift) * nu].to_vec();
                    for k in h - shift..h {
                        plan[k * nu..(k + 1) * nu].copy_from_slice(&tail);
                    }
                }
                mpc::solve(self.planner, &x, &mut plan, h, self.iters);
                if label {
                    let target: Vec<f64> =
                        plan[..nu].iter().map(|p| mpc::bound() * p.tanh()).collect();
                    labels.push((see(task, &x), target));
                }
            }
            match student {
                Some(s) => act(s, &see(task, &x), &mut off),
                None => {
                    let k = (t % replan) / kn;
                    for i in 0..nu {
                        off[i] = mpc::bound() * plan[k * nu + i].tanh();
                    }
                }
            }
            for i in 0..nu {
                x[ns + i] = task.rest[ns + i] + off[i];
            }
            (task.fwd)(&x, &mut out);
            x[..ns].copy_from_slice(&out[..ns]);
            x[ns + nu..].copy_from_slice(&out[ns..]);
            if !x.iter().all(|v| v.is_finite()) {
                return Episode {
                    labels,
                    drop: f64::INFINITY,
                };
            }
            lowest = lowest.min(x[5]);
        }
        Episode {
            labels,
            drop: task.z0 - lowest,
        }
    }

    fn pd_drop(&self, x0: &[f64]) -> f64 {
        let task = self.plant;
        let (ns, nu, nf) = (task.nq + task.nv, task.nu, task.nf);
        let mut x = x0.to_vec();
        let mut out = vec![0.0; ns + nf];
        let mut lowest = x[5];
        for _ in 0..STEPS {
            x[ns..ns + nu].copy_from_slice(&task.rest[ns..ns + nu]);
            (task.fwd)(&x, &mut out);
            x[..ns].copy_from_slice(&out[..ns]);
            x[ns + nu..].copy_from_slice(&out[ns..]);
            if !x.iter().all(|v| v.is_finite()) {
                return f64::INFINITY;
            }
            lowest = lowest.min(x[5]);
        }
        task.z0 - lowest
    }
}

/// The student's offsets: `bound tanh(out / TORQUE)`, the teacher's
/// parameterisation.
fn act(student: &Mlp, obs: &[f64], u: &mut [f64]) {
    let c = student.forward(obs);
    for (o, a) in u.iter_mut().zip(&c.out) {
        *o = mpc::bound() * (a / TORQUE).tanh();
    }
}

/// Regress the student onto `(obs, offsets)`: `epochs` passes of Adam over
/// shuffled mini-batches of 256. Returns the last epoch's mean squared error
/// per joint.
fn fit(student: &mut Mlp, data: &[(Vec<f64>, Vec<f64>)], epochs: usize, rng: &mut Lcg) -> f64 {
    let mut adam = Adam::new(student.params.len(), 1e-3);
    let mut order: Vec<usize> = (0..data.len()).collect();
    let mut last = 0.0;
    for _ in 0..epochs {
        for i in (1..order.len()).rev() {
            let j = ((rng.next() * 0.5 + 0.5) * (i + 1) as f64) as usize % (i + 1);
            order.swap(i, j);
        }
        let mut total = 0.0;
        for batch in order.chunks(256) {
            let mut g = vec![0.0; student.params.len()];
            for &k in batch {
                let (obs, target) = &data[k];
                let c = student.forward(obs);
                let mut gout: Vec<f64> = c
                    .out
                    .iter()
                    .zip(target)
                    .map(|(a, y)| {
                        let th = (a / TORQUE).tanh();
                        let e = mpc::bound() * th - y;
                        total += e * e;
                        2.0 * e * mpc::bound() * (1.0 - th * th) / TORQUE / batch.len() as f64
                    })
                    .collect();
                student.backward(obs, &c, &mut gout, &mut g);
            }
            adam.step(&mut student.params, &g);
        }
        last = total / (data.len() * data[0].1.len()) as f64;
    }
    last
}

/// The label set as text: one `obs | target` line per label.
fn save(path: &str, data: &[(Vec<f64>, Vec<f64>)]) {
    use std::fmt::Write as _;
    let mut s = String::new();
    for (o, y) in data {
        for v in o {
            write!(s, "{v:e} ").unwrap();
        }
        s += "|";
        for v in y {
            write!(s, " {v:e}").unwrap();
        }
        s += "\n";
    }
    std::fs::write(path, s).unwrap();
}

fn load(path: &str) -> Vec<(Vec<f64>, Vec<f64>)> {
    let parse = |t: &str| {
        t.split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect::<Vec<f64>>()
    };
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| {
            let (o, y) = l.split_once('|').unwrap();
            (parse(o), parse(y))
        })
        .collect()
}
