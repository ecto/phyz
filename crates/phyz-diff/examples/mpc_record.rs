//! Record PD and MPC push recovery for a video: the K1's links as boxes
//! fitted to its STL meshes, and every link's world pose per frame, as text
//! (`kosm`'s `k1_reel` renders it).
//!
//! Sweeps the shove's strength for the strongest one MPC holds (trunk within
//! 3 cm) while PD sags past 10 cm, then records both from that same start.
//! (Only the soles have contact points: a fall would pass through the floor,
//! so the sweep stays below that.)
//!
//! `cargo run --release -p phyz-diff --example mpc_record OUT.txt`

#[path = "mpc_stand.rs"]
#[allow(dead_code)]
mod mpc;

use std::fmt::Write as _;

use mpc::stand::{OFFSET, SWEEPS, Stand, k1};
use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::contact_step::ContactStep;
use phyz_math::Vec3;

const STEPS: usize = 1500; // 1.5 s at 1 ms
const EVERY: usize = 10; // a frame per 10 ms of simulated time

fn main() {
    let out_path = std::env::args().nth(1).expect("usage: mpc_record OUT.txt");
    if std::env::var_os("STAND_H").is_none() {
        unsafe { std::env::set_var("STAND_H", "600") };
    }
    let (model, points) = k1();
    let mat = ContactMaterial {
        friction: 0.8,
        margin: 0.005,
        ..ContactMaterial::default()
    };
    let config = ContactSolverConfig::default();
    let step = ContactStep::build(&model, &points, &mat, &config, SWEEPS);
    let (ag, ao) = step.adjoint();
    let fwd = mpc::native(&step.kernel_c_f64().source, step.n_outputs());
    let back = mpc::native(
        &ag.to_c_f64(&ao, step.n_inputs() + step.n_outputs()).source,
        ao.len(),
    );
    let task = Stand::new(&model, fwd, back, step.nu, points.len());

    // a lateral shove with some roll: stronger until PD loses it
    let shove = |s: f64| {
        let mut x = task.rest.clone();
        x[task.nq] += 1.2 * s; // roll rate
        x[task.nq + 4] += 0.5 * s; // sideways
        x
    };
    let mut chosen = None;
    for s in [1.0, 1.1, 1.2, 1.3, 1.4] {
        let (pd, _) = rollout(&task, &shove(s), false);
        let (mp, _) = rollout(&task, &shove(s), true);
        let low = |tr: &[Vec<f64>]| tr.iter().map(|x| x[5]).fold(f64::INFINITY, f64::min);
        let (dp, dm) = (task.z0 - low(&pd), task.z0 - low(&mp));
        eprintln!(
            "shove x{s}: lowest trunk PD -{:.0} mm, MPC -{:.0} mm",
            dp * 1e3,
            dm * 1e3
        );
        if dp > 0.1 && dp < 0.45 && dm < 0.03 {
            chosen = Some(s);
        }
    }
    let s = chosen.unwrap_or(1.0);
    eprintln!("recording shove x{s}");
    let (pd, _) = rollout(&task, &shove(s), false);
    let (mp, ms) = rollout(&task, &shove(s), true);
    eprintln!("MPC solve {ms:.1} ms mean");

    // links: an axis-aligned box in each link frame around its visual STL;
    // a plain text file (kosm's k1_push reads it without a JSON parser):
    //   meta z0 shove solve_ms every
    //   link body lox loy loz hix hiy hiz
    //   pose ctrl frame body px py pz r00 r01 r02 r10 r11 r12 r20 r21 r22
    // (r: the world→body rotation, row-major)
    let mut js = String::new();
    writeln!(js, "meta {} {s} {ms:.2} {EVERY}", task.z0).unwrap();
    for (body, lo, hi) in urdf_meshes() {
        writeln!(
            js,
            "link {body} {} {} {} {} {} {}",
            lo.x, lo.y, lo.z, hi.x, hi.y, hi.z
        )
        .unwrap();
    }
    for (name, tr) in [("pd", &pd), ("mpc", &mp)] {
        for (f, x) in tr.iter().step_by(EVERY).enumerate() {
            let mut st = model.default_state();
            st.q.as_mut_slice().copy_from_slice(&x[..task.nq]);
            let (xf, _) = phyz_rigid::forward_kinematics(&model, &st);
            for (b, t) in xf.iter().enumerate() {
                let r = &t.rot;
                writeln!(
                    js,
                    "pose {name} {f} {b} {:.5} {:.5} {:.5} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                    t.pos.x, t.pos.y, t.pos.z, r[(0, 0)], r[(0, 1)], r[(0, 2)], r[(1, 0)], r[(1, 1)], r[(1, 2)], r[(2, 0)], r[(2, 1)], r[(2, 2)]
                )
                .unwrap();
            }
        }
    }
    std::fs::write(&out_path, js).unwrap();
    eprintln!(
        "wrote {out_path}: {} frames per controller",
        pd.len().div_ceil(EVERY)
    );
}

/// A rollout from `x0` under PD (zero offsets) or MPC: every step's input
/// state, and the mean ms per MPC solve.
fn rollout(task: &Stand, x0: &[f64], use_mpc: bool) -> (Vec<Vec<f64>>, f64) {
    let (nq, nv, nu, nf) = (task.nq, task.nv, task.nu, task.nf);
    let ns = nq + nv;
    let (h, replan, iters) = (16, 10, 3);
    let mut plan = vec![0.0; h * nu];
    let mut x = x0.to_vec();
    let mut out = vec![0.0; ns + nf];
    let mut traj = Vec::with_capacity(STEPS);
    let (mut secs, mut solves) = (0.0, 0);
    for t in 0..STEPS {
        if use_mpc && t % replan == 0 {
            if t > 0 {
                plan.copy_within(replan * nu.., 0);
                let tail: Vec<f64> = plan[(h - replan - 1) * nu..(h - replan) * nu].to_vec();
                for k in h - replan..h {
                    plan[k * nu..(k + 1) * nu].copy_from_slice(&tail);
                }
            }
            let start = std::time::Instant::now();
            mpc::solve(task, &x, &mut plan, h, iters);
            secs += start.elapsed().as_secs_f64();
            solves += 1;
        }
        let k = t % replan;
        for i in 0..nu {
            let off = if use_mpc {
                OFFSET * plan[k * nu + i].tanh()
            } else {
                0.0
            };
            x[ns + i] = task.rest[ns + i] + off;
        }
        traj.push(x.clone());
        (task.fwd)(&x, &mut out);
        x[..ns].copy_from_slice(&out[..ns]);
        x[ns + nu..].copy_from_slice(&out[ns..]);
        if !x.iter().all(|v| v.is_finite()) {
            break;
        }
    }
    (traj, 1e3 * secs / solves.max(1) as f64)
}

/// `(body, lo, hi)`: each visual STL's bounding box in its link frame.
fn urdf_meshes() -> Vec<(usize, Vec3, Vec3)> {
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
    let urdf = phyz_urdf::load_file(root.join("K1_22dof_floating.urdf"), &opts).expect("K1 URDF");
    let mut out = Vec::new();
    for m in urdf.mesh_refs.iter().filter(|m| m.visual) {
        let file = m
            .filename
            .trim_start_matches("package://")
            .rsplit('/')
            .next()
            .unwrap();
        let Ok(bytes) = std::fs::read(root.join("meshes").join(file)) else {
            eprintln!("missing mesh {}", m.filename);
            continue;
        };
        let n = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
        let scale = m.scale.unwrap_or(Vec3::new(1.0, 1.0, 1.0));
        let (mut lo, mut hi) = (
            Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        );
        for t in 0..n.min((bytes.len() - 84) / 50) {
            for v in 0..3 {
                let at = 84 + t * 50 + 12 + v * 12;
                let f = |k: usize| {
                    f32::from_le_bytes(bytes[at + 4 * k..at + 4 * k + 4].try_into().unwrap()) as f64
                };
                let p_mesh = Vec3::new(f(0) * scale.x, f(1) * scale.y, f(2) * scale.z);
                // mesh frame → link frame (origin.rot maps link into mesh)
                let p = m.origin.pos + m.origin.rot.transpose() * p_mesh;
                lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
                hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
            }
        }
        out.push((m.body_idx, lo, hi));
    }
    out
}
