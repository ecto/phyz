//! `hold --wiggle Left_Hip_Pitch --wiggle Right_Hip_Pitch --chirp F0,F1` in
//! the traced K1 plant: both hip pitch targets swept by a log chirp while the
//! robot stands, and the trunk pitch rate's gain (rad/s per rad of command)
//! per 1 s window — the same table `chirp-03.trace` gave on the robot
//! (standing at kp 90 / kd 6: a peak of ~6 near 8 Hz).
//!
//! Gains come from the rig unless `IPSE_KP_LEG` etc. override them (see
//! `ipse_rig::Rig::pd`).
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
    println!("chirp {f0}-{f1} Hz, ±{amp} rad on both hip pitches, {secs} s");
    for k in 0..n {
        let t = k.saturating_sub(settle) as f64 * dt;
        // log chirp, eased in over its first period
        let phase = std::f64::consts::TAU * f0 * secs / r.ln() * (r.powf(t / secs) - 1.0);
        let ease = (t * f0).min(1.0);
        let c = if k < settle {
            0.0
        } else {
            amp * ease * phase.sin()
        };
        for (s, &d) in pd.dofs.iter().enumerate() {
            x[nq + nv + s] = rig.q_target[d];
        }
        for &s in &hips {
            x[nq + nv + s] += c;
        }
        x[nq + nv + nt..nq + nv + nt + 6]
            .iter_mut()
            .for_each(|v| *v = 0.0);
        fwd(&x, &mut out);
        x[..nq + nv].copy_from_slice(&out[..nq + nv]);
        x[nq + nv + nu..].copy_from_slice(&out[nq + nv..]);
        if !x.iter().all(|v| v.is_finite()) || x[5] < 0.35 {
            println!("fell at t {t:.2} s");
            return;
        }
        if k >= settle {
            win.push((c, x[nq + 1]));
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
