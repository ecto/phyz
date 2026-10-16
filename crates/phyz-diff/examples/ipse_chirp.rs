//! `hold --wiggle Left_Hip_Pitch --wiggle Right_Hip_Pitch --chirp F0,F1` in
//! the traced K1 plant: both hip pitch targets swept by a log chirp while the
//! robot stands, and the trunk pitch rate's gain (rad/s per rad of command)
//! per 1 s window — the same table `chirp-03.trace` gave on the robot
//! (standing at kp 90 / kd 6: a peak of ~6 near 8 Hz).
//!
//! Gains come from the rig unless `IPSE_KP_LEG` etc. override them (see
//! `ipse_rig::Rig::pd`).
//!
//! `IPSE_ACT_RES_HZ` / `IPSE_ACT_ZETA` pass the chirp offset through the same
//! second-order stage `ipse_cem` applies to policy offsets
//! (`ipse_rig::ActRes`), so the stand-in can be fitted against the robot's
//! chirp. `IPSE_CHIRP_TSV=PATH` writes every 5th step (the robot's 200 Hz
//! command rate) in `trace_rate`'s line format, which ipse-deploy's
//! `chirp_fit` example reads next to the robot's:
//! `t q0 <cmd> kp <kp> gyro_y <trunk pitch rate> qm <hip q> tau 0`.
//!
//! `IPSE_CHIRP_HANG=1` pins the base 0.3 m up (chirp-01, robot hanging): only
//! the hip channel means anything then.
//!
//! `cargo run --release -p phyz-diff --example ipse_chirp RIG.txt [F0 F1 AMP SECONDS]`

#[path = "common/ipse_rig.rs"]
mod ipse_rig;

use phyz_contact::ContactSolverConfig;
use phyz_diff::contact_step::ContactStep;

fn main() {
    let mut args = std::env::args().skip(1);
    let rig = ipse_rig::Rig::read(
        &args
            .next()
            .expect("usage: ipse_chirp RIG.txt [F0 F1 AMP SECONDS]"),
    );
    let mut num = |d: f64| args.next().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (f0, f1, amp, secs) = (num(4.0), num(10.0), num(0.01), num(28.0));
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
    let fwd = step.compile();
    let (nq, nv, nu) = (model.nq, model.nv, step.nu);
    let nt = nu - 6;
    let nf = 3 * rig.points().len();
    // the PD slots that drive a hip pitch joint
    let hips: Vec<usize> = pd
        .dofs
        .iter()
        .enumerate()
        .filter(|&(_, &d)| {
            model.joints.iter().enumerate().any(|(j, jt)| {
                jt.ndof() == 1 && model.v_offsets[j] == d && jt.name.contains("Hip_Pitch")
            })
        })
        .map(|(s, _)| s)
        .collect();
    assert_eq!(hips.len(), 2, "expected two hip pitch joints");
    let hd = pd.dofs[hips[0]];
    println!(
        "nq {nq} nv {nv}; hip pitch v index {hd}; q[{hd}] {:.4} q[{}] {:.4} target {:.4}",
        rig.q0[hd],
        hd + 1,
        rig.q0[hd + 1],
        rig.q_target[hd]
    );

    let dt = rig.dt;
    let mut x: Vec<f64> = rig.q0.iter().chain(&rig.v0).copied().collect();
    x.extend(pd.dofs.iter().map(|&d| rig.q_target[d]));
    x.extend([0.0; 6]);
    x.extend(std::iter::repeat_n(0.0, nf));
    let mut out = vec![0.0; nq + nv + nf];
    let settle = (2.0 / dt) as usize;
    let n = settle + (secs / dt) as usize;
    let r = f1 / f0;
    let (mut win, mut wt) = (Vec::new(), 0.0);
    // measured hip pitch q, for actual / commanded
    let mut hq: Vec<f64> = Vec::new();
    let mut res = ipse_rig::ActRes::from_env(dt);
    let mut sea = ipse_rig::Sea::from_env(&rig, &pd);
    let mut lugre = ipse_rig::LuGre::from_env(&rig, &pd);
    let hang = std::env::var("IPSE_CHIRP_HANG").is_ok();
    let plant_delay: usize = std::env::var("IPSE_PLANT_DELAY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut delay = std::collections::VecDeque::new();
    if hang {
        x[5] += 0.3;
    }
    let mut tsv = std::env::var("IPSE_CHIRP_TSV")
        .ok()
        .map(|p| std::io::BufWriter::new(std::fs::File::create(p).unwrap()));
    println!("chirp {f0}-{f1} Hz, ±{amp} rad on both hip pitches, {secs} s");
    for k in 0..n {
        let t = k.saturating_sub(settle) as f64 * dt;
        // log chirp, eased in over its first period
        let phase = std::f64::consts::TAU * f0 * secs / r.ln() * (r.powf(t / secs) - 1.0);
        let ease = (t * f0).min(1.0);
        let cmd = if k < settle {
            0.0
        } else {
            amp * ease * phase.sin()
        };
        // what the joint's PD sees: the commanded offset, or its filtered
        // version under the series-compliance stand-in
        // IPSE_PLANT_DELAY steps before the joint acts on a command: the
        // robot's hip lags its chirp by ~linear-in-f phase
        delay.push_back(cmd);
        let late = if delay.len() > plant_delay {
            delay.pop_front().unwrap()
        } else {
            0.0
        };
        let c = res.as_mut().map_or(late, |r| r.step(&[late])[0]);
        for (s, &d) in pd.dofs.iter().enumerate() {
            x[nq + nv + s] = rig.q_target[d];
        }
        for &s in &hips {
            x[nq + nv + s] += c;
        }
        x[nq + nv + nt..nq + nv + nt + 6]
            .iter_mut()
            .for_each(|v| *v = 0.0);
        if let Some(sea) = sea.as_mut() {
            sea.apply(&mut x, nq, nv);
        }
        if let Some(l) = lugre.as_mut() {
            l.apply(&mut x, nq, nv);
        }
        fwd(&x, &mut out);
        x[..nq + nv].copy_from_slice(&out[..nq + nv]);
        x[nq + nv + nu..].copy_from_slice(&out[nq + nv..]);
        // IPSE_CHIRP_HANG: the base pinned 0.3 m up, feet clear of the
        // floor — chirp-01 was run with the robot hanging
        if hang {
            x[..6].copy_from_slice(&rig.q0[..6]);
            x[5] += 0.3;
            x[nq..nq + 6].iter_mut().for_each(|v| *v = 0.0);
        }
        if !x.iter().all(|v| v.is_finite()) || x[5] < 0.35 {
            println!("fell at t {t:.2} s");
            return;
        }
        if k >= settle
            && (k - settle).is_multiple_of(5)
            && let Some(w) = tsv.as_mut()
        {
            use std::io::Write;
            writeln!(
                w,
                "{t:9.4}  q0 {:+.5}  kp {:.0}  gyro_y {:+.5}  qm {:+.5}  tau 0",
                rig.q_target[hd] + cmd,
                pd.kp[hips[0]],
                x[nq + 1],
                x[hd]
            )
            .unwrap();
        }
        if k >= settle {
            win.push((cmd, x[nq + 1]));
            hq.push(x[pd.dofs[hips[0]]]);
            if t - wt >= 1.0 {
                let rms = |f: &dyn Fn(&(f64, f64)) -> f64| {
                    let m = win.iter().map(f).sum::<f64>() / win.len() as f64;
                    (win.iter().map(|p| (f(p) - m).powi(2)).sum::<f64>() / win.len() as f64).sqrt()
                };
                let (aq, ag) = (rms(&|p| p.0), rms(&|p| p.1));
                let f = f0 * r.powf((wt + 0.5) / secs);
                let gain = ag / aq;
                let mh = hq.iter().sum::<f64>() / hq.len() as f64;
                let ah =
                    (hq.iter().map(|v| (v - mh).powi(2)).sum::<f64>() / hq.len() as f64).sqrt();
                hq.clear();
                println!(
                    "  t {wt:5.1}  f {f:5.1} Hz  hip actual/cmd {:.2}  gyro_y {:.3}  gain {gain:6.1}  {}",
                    ah / aq,
                    ag * 2f64.sqrt(),
                    "#".repeat((gain / 2.0) as usize)
                );
                win.clear();
                wt = t;
            }
        }
    }
}
