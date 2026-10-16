//! Sim-to-real gate for ipse's CEM shove policy: one command, PASS/FAIL.
//!
//! Runs the policy in the traced K1 plant under the robot's loop (the same
//! knobs as `ipse_cem`, but with hardware defaults: `IPSE_POLICY_EVERY=5`,
//! `IPSE_LATENCY=5` — a 200 Hz loop with one tick of latency — `IPSE_CLAMP=0.15`,
//! `IPSE_RAMP=500`) and the fitted series-compliance stand-in
//! (`IPSE_ACT_RES_HZ` / `IPSE_ACT_ZETA` / `IPSE_ACT_GAIN`, `ipse_rig::ActRes`;
//! fit them with ipse-deploy's `chirp_fit`). Checks:
//!
//! 1. **Quiet stance** at 1x and 1.5x policy weights: 2 s of standing under
//!    the policy; peak base rate < 0.3 rad/s and action sign flips < 2% of
//!    queries. 1.5x is the margin: a policy on the edge of chattering at 1x
//!    shook on the robot (shove-hw-01, 7.5 Hz).
//! 2. **Post-push settling**: sagittal shoves of 6, 10, 14 N·s forward and
//!    backward (ipse's 20 ms base wrench) after the 2 s stance. The robot must
//!    stay up and the trunk pitch rate's RMS over 1.0–2.0 s after the shove
//!    must be < 0.15 rad/s. Why 0.15: a recovery that has damped out sits at
//!    ≲0.05 rad/s by then in the plant (the calm policies' values), while the
//!    oscillation shove-hwr-01 showed on the robot after a big push — a
//!    sustained 5–6 Hz at ±1 rad/s — is RMS ~0.7; 0.15 is ~3x above settled
//!    and ~5x below that, and also below the ±0.3 rad/s quiet-stance bound's
//!    RMS (0.21) so a "settled" robot is at least as still as a quiet one.
//! 3. **Ladder** (informational): the largest forward, backward and lateral
//!    (worse of ±) impulse survived, 2..24 N·s in steps of 2, and the largest
//!    sagittal level that still settles.
//!
//! The calibration that reproduced the four hardware runs (shove-p96x48-s7
//! calm, shove-hw-p96x48-s7 shook, shove-hwm-p96x48-s7 calm, shove-hwr200-
//! p96x48-s11 calm): kp 90 / kd 6, 8 Hz, ζ 0.15, gain 1:
//!
//! ```sh
//! cd ~/Developer/phyz/.claude/worktrees/train-sim
//! cargo build --release -p phyz-diff --example ipse_gate
//! IPSE_KP_LEG=90 IPSE_KD_LEG=6 IPSE_ACT_RES_HZ=8 IPSE_ACT_ZETA=0.15 \
//!   target/release/examples/ipse_gate RIG.txt ~/Developer/ipse/models/shove-hw-p96x48-s7.policy
//! ```
//!
//! Prints one table and `GATE PASS` / `GATE FAIL (reasons)`; exits 1 on FAIL.
//! `IPSE_GATE_RMS` overrides the 0.15 rad/s settling bound; `IPSE_GATE_PD=1`
//! zeroes the policy (plain PD baseline, any .policy file as placeholder).
//!
//! `cargo run --release -p phyz-diff --example ipse_gate RIG.txt CEM.policy`

#[path = "common/ipse_rig.rs"]
mod ipse_rig;

use phyz_contact::ContactSolverConfig;
use phyz_diff::contact_step::ContactStep;
use tang_expr::compile::CompiledMany;

fn env(k: &str, d: f64) -> f64 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// PD settles alone for this many steps, then the policy fades in.
const SETTLE: usize = 1500;
/// Shove after 2 s under the policy.
const SHOVE_AT: usize = 3500;
/// Steps simulated after the shove.
const POST: usize = 2500;

#[derive(Clone, Copy)]
enum Job {
    Quiet { scale: f64 },
    Shove { imp: [f64; 3] },
}

#[derive(Clone, Copy, Default)]
struct Outcome {
    alive: bool,
    /// Pre-shove stance: peak |base rate| (rad/s), action sign flips (%).
    peak: f64,
    flips: f64,
    /// Trunk pitch rate RMS over 1.0–2.0 s after the shove (rad/s), and its
    /// zero-crossing frequency there (Hz).
    rms: f64,
    hz: f64,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let rig = ipse_rig::Rig::read(&args.next().expect("usage: ipse_gate RIG.txt CEM.policy"));
    let policy = args.next().expect("CEM.policy");
    let mut params = ipse_rig::read_cem(&policy);
    // IPSE_GATE_PD: zero the policy — the same rollouts under plain PD, the
    // baseline every shove level is measured against
    if std::env::var("IPSE_GATE_PD").is_ok() {
        params.iter_mut().for_each(|w| *w = 0.0);
    }
    let model = rig.model();
    let dofs = ipse_rig::cem_dofs(&model);
    let step = ContactStep::build_pd(
        &model,
        &rig.points(),
        &rig.material(),
        &ContactSolverConfig::default(),
        5,
        &rig.pd(),
    );
    let fwd = step.compile();
    let rms_max = env("IPSE_GATE_RMS", 0.15);

    let levels: Vec<f64> = (1..=12).map(|k| 2.0 * k as f64).collect();
    // forward, backward, left, right
    let dirs: [[f64; 3]; 4] = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ];
    let mut jobs = vec![Job::Quiet { scale: 1.0 }, Job::Quiet { scale: 1.5 }];
    for d in &dirs {
        for &n in &levels {
            jobs.push(Job::Shove {
                imp: d.map(|c| c * n),
            });
        }
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut res = vec![Outcome::default(); jobs.len()];
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                let (rig, model, fwd, params, dofs, jobs, next) =
                    (&rig, &model, &fwd, &params, &dofs, &jobs, &next);
                sc.spawn(move || {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&job) = jobs.get(i) else { break };
                        out.push((i, rollout(rig, model, fwd, step.nu, params, dofs, job)));
                    }
                    out
                })
            })
            .collect();
        for h in hs {
            for (i, o) in h.join().unwrap() {
                res[i] = o;
            }
        }
    });

    let mut fails: Vec<String> = Vec::new();
    println!("policy {policy}");
    println!(
        "plant: kp_leg {} kd_leg {} | act res {} Hz ζ {} gain {} | every {} latency {} clamp {} ramp {}",
        std::env::var("IPSE_KP_LEG").unwrap_or("rig".into()),
        std::env::var("IPSE_KD_LEG").unwrap_or("rig".into()),
        env("IPSE_ACT_RES_HZ", 0.0),
        env("IPSE_ACT_ZETA", 0.2),
        env("IPSE_ACT_GAIN", 1.0),
        env("IPSE_POLICY_EVERY", 5.0),
        env("IPSE_LATENCY", 5.0),
        env("IPSE_CLAMP", 0.15),
        env("IPSE_RAMP", 500.0),
    );
    println!("quiet stance (2 s):");
    for (i, s) in [1.0, 1.5].iter().enumerate() {
        let o = res[i];
        let ok = o.alive && o.peak < 0.3 && o.flips < 2.0;
        if !ok {
            fails.push(format!("quiet {s}x"));
        }
        println!(
            "  {s:.1}x  peak {:.2} rad/s  flips {:4.1}%  {}{}",
            o.peak,
            o.flips,
            if o.alive { "" } else { "FELL " },
            if ok { "ok" } else { "FAIL" }
        );
    }
    let idx =
        |d: usize, n: f64| 2 + d * levels.len() + levels.iter().position(|&l| l == n).unwrap();
    println!("post-push settling (gyro_y RMS 1.0–2.0 s after, bound {rms_max} rad/s):");
    for (d, name) in [(0, "forward"), (1, "backward")] {
        let cells: Vec<String> = [6.0, 10.0, 14.0]
            .iter()
            .map(|&n| {
                let o = res[idx(d, n)];
                let ok = o.alive && o.rms < rms_max;
                if !ok {
                    fails.push(format!("{name} {n}"));
                }
                if o.alive {
                    format!(
                        "{n:>2.0}: {:.3} @{:3.1}Hz {}",
                        o.rms,
                        o.hz,
                        if ok { "ok  " } else { "FAIL" }
                    )
                } else {
                    format!("{n:>2.0}: fell          FAIL")
                }
            })
            .collect();
        println!("  {name:>8}  {}", cells.join("  "));
    }
    println!("ladder (2..24 N·s; P survived, s settled under the bound):");
    let best = |d: usize, settle: bool| {
        let mut last = 0.0;
        for &n in &levels {
            let o = res[idx(d, n)];
            if !o.alive || (settle && o.rms >= rms_max) {
                break;
            }
            last = n;
        }
        last
    };
    for (d, name) in ["forward", "backward", "left", "right"].iter().enumerate() {
        let row: String = levels
            .iter()
            .map(|&n| {
                let o = res[idx(d, n)];
                match (o.alive, o.rms < rms_max) {
                    (false, _) => ".",
                    (true, false) => "P",
                    (true, true) => "s",
                }
            })
            .collect();
        let rms: String = levels
            .iter()
            .map(|&n| {
                let o = res[idx(d, n)];
                if o.alive {
                    format!(" {:.2}@{:.0}", o.rms, o.hz)
                } else {
                    "      -".into()
                }
            })
            .collect();
        println!(
            "  {name:>8}: survives {:>2.0} N·s, settles to {:>2.0}  |{row}|  rms{rms}",
            best(d, false),
            best(d, true)
        );
    }
    println!(
        "  largest survived: forward {:.0}, backward {:.0}, lateral {:.0} N·s",
        best(0, false),
        best(1, false),
        best(2, false).min(best(3, false))
    );
    if fails.is_empty() {
        println!("GATE PASS");
    } else {
        println!("GATE FAIL ({})", fails.join(", "));
        std::process::exit(1);
    }
}

/// PD stands 1.5 s, the policy fades in and stands 2 s more (the chatter
/// window), then — for a shove job — ipse's 20 ms base wrench and 2.5 s more.
/// Alive = finite and trunk above 0.35 m throughout.
fn rollout(
    rig: &ipse_rig::Rig,
    model: &phyz_model::Model,
    fwd: &CompiledMany,
    nu: usize,
    params: &[f64],
    dofs: &[usize],
    job: Job,
) -> Outcome {
    let (nq, nv) = (model.nq, model.nv);
    let pd = rig.pd();
    let nt = nu - 6;
    let nf = 3 * rig.points().len();
    let mut params = params.to_vec();
    let (imp, end) = match job {
        Job::Quiet { scale } => {
            params[..310].iter_mut().for_each(|w| *w *= scale);
            ([0.0; 3], SHOVE_AT)
        }
        Job::Shove { imp } => (imp, SHOVE_AT + POST),
    };
    let mut x: Vec<f64> = rig.q0.iter().chain(&rig.v0).copied().collect();
    x.extend(pd.dofs.iter().map(|&d| rig.q_target[d]));
    x.extend([0.0; 6]);
    x.extend(std::iter::repeat_n(0.0, nf));
    let mut out = vec![0.0; nq + nv + nf];
    let slot: Vec<usize> = dofs
        .iter()
        .map(|d| pd.dofs.iter().position(|p| p == d).unwrap())
        .collect();
    let every = env("IPSE_POLICY_EVERY", 5.0) as usize;
    // IPSE_PLANT_DELAY steps between a command and the joint acting on it,
    // on top of the policy's own latency (see ipse_chirp)
    let latency = (env("IPSE_LATENCY", 5.0) + env("IPSE_PLANT_DELAY", 0.0)) as usize;
    let clamp = env("IPSE_CLAMP", 0.15);
    let ramp = env("IPSE_RAMP", 500.0).max(1.0);
    let (obs_pitch, noise) = (env("IPSE_OBS_PITCH", 0.0), env("IPSE_GYRO_NOISE", 0.0));
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut uniform = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let mut res = ipse_rig::ActRes::from_env(rig.dt);
    let mut sea = ipse_rig::Sea::from_env(rig, &pd);
    let mut lugre = ipse_rig::LuGre::from_env(rig, &pd);
    let mut pending: std::collections::VecDeque<(usize, Vec<f64>)> = Default::default();
    let mut action = vec![0.0; dofs.len()];
    let mut prev: Vec<f64> = Vec::new();
    let (mut flips, mut queries) = (0usize, 0usize);
    let mut o = Outcome::default();
    let mut gy: Vec<f64> = Vec::new();
    // the post-shove settling window
    let (w0, w1) = (SHOVE_AT + 1000, SHOVE_AT + 2000);
    for k in 0..end {
        for (s, &d) in pd.dofs.iter().enumerate() {
            x[nq + nv + s] = rig.q_target[d];
        }
        if k >= SETTLE {
            if (k - SETTLE).is_multiple_of(every) {
                let mut ob =
                    ipse_rig::observe_legs(model, &rig.q_target, &x[..nq], &x[nq..nq + nv], dofs);
                // IPSE_OBS_PITCH rad on the trunk pitch the policy sees (an
                // IMU mount / stance offset the plant does not have) and
                // IPSE_GYRO_NOISE rad/s uniform on the base and joint rates,
                // as ipse_cem
                ob[0] += obs_pitch;
                if noise > 0.0 {
                    for i in (2..5).chain((12..ob.len()).step_by(2)) {
                        ob[i] += noise * uniform();
                    }
                }
                let a = ipse_rig::cem_action(&params, &ob);
                if k < SHOVE_AT {
                    if !prev.is_empty() {
                        flips += a
                            .iter()
                            .zip(&prev)
                            .filter(|(p, q)| p.signum() != q.signum())
                            .count();
                        queries += a.len();
                    }
                    prev = a.clone();
                }
                pending.push_back((
                    k + latency,
                    a.iter().map(|v| v.clamp(-clamp, clamp)).collect(),
                ));
            }
            while pending.front().is_some_and(|(t, _)| *t <= k) {
                action = pending.pop_front().unwrap().1;
            }
            let fade = ((k - SETTLE) as f64 / ramp).min(1.0);
            let eff: Vec<f64> = match res.as_mut() {
                Some(r) => r.step(&action).to_vec(),
                None => action.clone(),
            };
            for (i, a) in eff.iter().enumerate() {
                x[nq + nv + slot[i]] += fade * a;
            }
            if k < SHOVE_AT {
                o.peak = x[nq..nq + 3].iter().fold(o.peak, |m, v| m.max(v.abs()));
            }
        }
        let w = &mut x[nq + nv + nt..nq + nv + nt + 6];
        w.iter_mut().for_each(|v| *v = 0.0);
        if (SHOVE_AT..SHOVE_AT + 20).contains(&k) {
            for a in 0..3 {
                w[3 + a] = imp[a] / 0.02;
            }
        }
        if let Some(sea) = sea.as_mut() {
            sea.apply(&mut x, nq, nv);
        }
        if let Some(l) = lugre.as_mut() {
            l.apply(&mut x, nq, nv);
        }
        fwd(&x, &mut out);
        x[..nq + nv].copy_from_slice(&out[..nq + nv]);
        x[nq + nv + nu..].copy_from_slice(&out[nq + nv..]);
        if !x.iter().all(|v| v.is_finite()) || x[5] < 0.35 {
            return o;
        }
        if (w0..w1).contains(&k) {
            gy.push(x[nq + 1]);
        }
    }
    o.alive = true;
    o.flips = 100.0 * flips as f64 / queries.max(1) as f64;
    if !gy.is_empty() {
        let m = gy.iter().sum::<f64>() / gy.len() as f64;
        o.rms = (gy.iter().map(|v| v * v).sum::<f64>() / gy.len() as f64).sqrt();
        // zero crossings with hysteresis (a third of the window's std), so
        // stick-slip jitter at a few mrad/s does not count as oscillation
        let sd = (gy.iter().map(|v| (v - m).powi(2)).sum::<f64>() / gy.len() as f64).sqrt();
        let (mut side, mut zc) = (0.0f64, 0usize);
        for v in &gy {
            let d = v - m;
            if d.abs() > sd / 3.0 && d.signum() != side {
                zc += (side != 0.0) as usize;
                side = d.signum();
            }
        }
        o.hz = zc as f64 / 2.0 / (gy.len() as f64 * rig.dt);
    }
    o
}
