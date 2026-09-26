//! Dump a phyz trajectory for cross-engine agreement testing.
//!
//! ```text
//! phyz-traj --model models/simple_arm.xml --steps 5000 --dt 0.001 [--q0 0.3,0.1] [--out traj.json]
//! ```
//!
//! Rolls an MJCF model forward with phyz's Featherstone ABA and semi-implicit
//! Euler, and writes every step's joint configuration as JSON. The companion
//! [`mujoco_agreement.py`](../../python/mujoco_agreement.py) runs the same
//! model through MuJoCo and compares.
//!
//! # Why this neutralises half the model
//!
//! An agreement test is only meaningful if both engines are solving the same
//! problem. Several MJCF features are modelled differently by the two engines
//! — or in phyz's case approximately, or not at all — and leaving them on
//! would produce a divergence number that says nothing about the dynamics:
//!
//! * **Joint damping, springs, dry friction** — MuJoCo integrates damping
//!   implicitly inside its Euler integrator; phyz applies it as an explicit
//!   passive force. That difference alone would dominate the comparison, and
//!   it is a difference in *integration*, not in dynamics.
//! * **Joint limits** — phyz models them as a soft penalty, MuJoCo as a
//!   constraint. Two different physical models.
//! * **Contact** — the same objection, more so. This binary never enables it.
//!
//! All of the above are zeroed here and must be zeroed on the MuJoCo side too;
//! the Python harness does that and refuses to run if it cannot. What remains
//! is the thing worth comparing: **articulated-body forward dynamics on a
//! kinematic tree under gravity**, integrated the same way.
//!
//! **Armature is deliberately kept.** It is a constant added to the mass
//! matrix diagonal, implemented the same way by both engines, so it is part of
//! the dynamics rather than a modelling difference — and on a model with very
//! light distal links (a hand's fingertips) it is the term that keeps the mass
//! matrix well conditioned. Zeroing it does not make the comparison fairer; it
//! makes both engines integrate an ill-conditioned system and measures the
//! timestep instead.
//!
//! Anything this binary neutralises is reported in the `neutralised` field of
//! its output, so a reader can see what was switched off rather than having to
//! trust that the list matches this comment.

use phyz::{Model, State};
use phyz_math::{Vec3, quat_exp, quat_log};
use phyz_mjcf::MjcfLoader;
use phyz_model::JointType;
use phyz_rigid::{aba, forward_kinematics, integrate_configuration};

/// Number of MuJoCo `qpos` coordinates a joint occupies.
///
/// The two engines parameterise rotation differently, so these do not match
/// phyz's own counts: a free joint is 6 coordinates in phyz (exponential
/// coordinates plus position) and 7 in MuJoCo (position plus quaternion).
fn mujoco_joint_nq(t: JointType) -> usize {
    match t {
        JointType::Free => 7,
        JointType::Spherical | JointType::Ball => 4,
        JointType::Fixed => 0,
        _ => 1,
    }
}

/// Total MuJoCo `qpos` width of a model.
fn mujoco_nq(model: &Model) -> usize {
    model
        .joints
        .iter()
        .map(|j| mujoco_joint_nq(j.joint_type))
        .sum()
}

/// Rewrite a phyz `Model`-layout `q` into MuJoCo's `qpos` layout.
///
/// Per joint:
///
/// | joint | phyz | MuJoCo |
/// |---|---|---|
/// | free | `[ωx ωy ωz, x y z]` (6) | `[x y z, qw qx qy qz]` (7) |
/// | ball | `[ωx ωy ωz]` (3) | `[qw qx qy qz]` (4) |
/// | hinge/slide | `[q]` | `[q]` |
///
/// `ω` is the rotation vector (axis × angle) whose exponential is the body's
/// orientation in its parent frame — the same rotation MuJoCo stores as a
/// quaternion, so the two are related by `quat_exp` / `quat_log` and nothing
/// else. Position is in parent coordinates on both sides.
fn to_mujoco(model: &Model, q: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(mujoco_nq(model));
    for (j, joint) in model.joints.iter().enumerate() {
        let o = model.q_offsets[j];
        match joint.joint_type {
            JointType::Fixed => {}
            JointType::Free => {
                let quat = quat_exp(&Vec3::new(q[o], q[o + 1], q[o + 2]));
                out.extend_from_slice(&[q[o + 3], q[o + 4], q[o + 5]]);
                out.extend_from_slice(&[quat.w, quat.v.x, quat.v.y, quat.v.z]);
            }
            JointType::Spherical | JointType::Ball => {
                let quat = quat_exp(&Vec3::new(q[o], q[o + 1], q[o + 2]));
                out.extend_from_slice(&[quat.w, quat.v.x, quat.v.y, quat.v.z]);
            }
            _ => out.push(q[o]),
        }
    }
    out
}

/// Rewrite a MuJoCo `qvel` into phyz's velocity layout.
///
/// Scalar joints map one-to-one. A free joint does not: MuJoCo orders it
/// **linear then angular** with the linear part in the **world** frame, while
/// phyz orders it **angular then linear** with the linear part in the **body**
/// frame (`SpatialVec` is `[angular; linear]` throughout). So the two halves
/// swap places *and* the linear half changes frame, which needs the body's
/// orientation — hence `q`, in phyz layout, as an argument.
///
/// Getting either half of that wrong is invisible when the body starts at rest
/// and unmistakable the moment it does not, which is exactly why the harness
/// tests a moving free base rather than a dropped one.
fn v_from_mujoco(model: &Model, q: &[f64], qvel: &[f64]) -> Vec<f64> {
    let mut v = vec![0.0; model.nv];
    let mut m = 0;
    for (j, joint) in model.joints.iter().enumerate() {
        let vo = model.v_offsets[j];
        let qo = model.q_offsets[j];
        match joint.joint_type {
            JointType::Fixed => {}
            JointType::Free => {
                let orientation = quat_exp(&Vec3::new(q[qo], q[qo + 1], q[qo + 2]));
                let world_lin = Vec3::new(qvel[m], qvel[m + 1], qvel[m + 2]);
                let body_lin = orientation.conjugate().rotate(world_lin);
                // Angular is body-frame on both sides; only the order differs.
                v[vo] = qvel[m + 3];
                v[vo + 1] = qvel[m + 4];
                v[vo + 2] = qvel[m + 5];
                v[vo + 3] = body_lin.x;
                v[vo + 4] = body_lin.y;
                v[vo + 5] = body_lin.z;
                m += 6;
            }
            JointType::Spherical | JointType::Ball => {
                v[vo] = qvel[m];
                v[vo + 1] = qvel[m + 1];
                v[vo + 2] = qvel[m + 2];
                m += 3;
            }
            _ => {
                v[vo] = qvel[m];
                m += 1;
            }
        }
    }
    v
}

/// The inverse of [`to_mujoco`].
fn from_mujoco(model: &Model, qpos: &[f64]) -> Vec<f64> {
    let mut q = vec![0.0; model.nq];
    let mut m = 0;
    for (j, joint) in model.joints.iter().enumerate() {
        let o = model.q_offsets[j];
        match joint.joint_type {
            JointType::Fixed => {}
            JointType::Free => {
                let quat = phyz_math::Quat {
                    w: qpos[m + 3],
                    v: Vec3::new(qpos[m + 4], qpos[m + 5], qpos[m + 6]),
                }
                .normalize();
                let w = quat_log(&quat);
                q[o] = w.x;
                q[o + 1] = w.y;
                q[o + 2] = w.z;
                q[o + 3] = qpos[m];
                q[o + 4] = qpos[m + 1];
                q[o + 5] = qpos[m + 2];
                m += 7;
            }
            JointType::Spherical | JointType::Ball => {
                let quat = phyz_math::Quat {
                    w: qpos[m],
                    v: Vec3::new(qpos[m + 1], qpos[m + 2], qpos[m + 3]),
                }
                .normalize();
                let w = quat_log(&quat);
                q[o] = w.x;
                q[o + 1] = w.y;
                q[o + 2] = w.z;
                m += 4;
            }
            _ => {
                q[o] = qpos[m];
                m += 1;
            }
        }
    }
    q
}

fn usage() -> ! {
    eprintln!(
        "usage: phyz-traj --model PATH [--steps N] [--dt SECONDS] [--q0 a,b,c] [--v0 a,b,c] [--out PATH]\n\
         \n\
         Writes {{model, dt, steps, nq, nv, joint_names, neutralised, q: [[...]]}} as JSON.\n\
         With no --out, writes to stdout.\n\
         \n\
         --layout mujoco  read --q0 and write q in MuJoCo qpos layout (position\n\
         then quaternion for free and ball joints) instead of phyz's own\n\
         (exponential coordinates then position)."
    );
    std::process::exit(2)
}

/// Strip every model feature the two engines do not share. Returns the list of
/// what was actually changed, so the output can carry it.
fn neutralise(model: &mut Model) -> Vec<String> {
    let mut changed = Vec::new();
    let note = |s: &str, list: &mut Vec<String>| {
        if !list.iter().any(|x| x == s) {
            list.push(s.to_string());
        }
    };

    for joint in &mut model.joints {
        if joint.damping != 0.0 {
            joint.damping = 0.0;
            note("joint damping", &mut changed);
        }
        if joint.stiffness != 0.0 || joint.spring_ref != 0.0 {
            joint.stiffness = 0.0;
            joint.spring_ref = 0.0;
            note("joint springs", &mut changed);
        }
        if joint.friction_loss != 0.0 {
            joint.friction_loss = 0.0;
            note("dry friction", &mut changed);
        }
        if joint.limits.is_some() {
            joint.limits = None;
            note("joint limits", &mut changed);
        }
    }
    changed
}

fn main() {
    let mut model_path: Option<String> = None;
    let mut steps = 2000usize;
    let mut dt: Option<f64> = None;
    let mut q0: Option<Vec<f64>> = None;
    let mut v0: Option<Vec<f64>> = None;
    let mut out: Option<String> = None;
    // Emit (and read `--q0` in) MuJoCo's qpos layout instead of phyz's. The
    // two differ wherever a joint carries a rotation: see `to_mujoco`.
    let mut mujoco_layout = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_path = Some(args.next().unwrap_or_else(|| usage())),
            "--steps" => {
                steps = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage())
            }
            "--dt" => {
                dt = Some(
                    args.next()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or_else(|| usage()),
                )
            }
            "--q0" => {
                let s = args.next().unwrap_or_else(|| usage());
                q0 = Some(
                    s.split(',')
                        .map(|x| x.trim().parse::<f64>().unwrap_or_else(|_| usage()))
                        .collect(),
                )
            }
            "--v0" => {
                let s = args.next().unwrap_or_else(|| usage());
                v0 = Some(
                    s.split(',')
                        .map(|x| x.trim().parse::<f64>().unwrap_or_else(|_| usage()))
                        .collect(),
                )
            }
            "--out" => out = Some(args.next().unwrap_or_else(|| usage())),
            "--layout" => match args.next().as_deref() {
                Some("mujoco") => mujoco_layout = true,
                Some("phyz") => mujoco_layout = false,
                _ => usage(),
            },
            "-h" | "--help" => usage(),
            other => {
                eprintln!("unknown argument: {other}");
                usage()
            }
        }
    }

    let Some(model_path) = model_path else {
        usage()
    };

    let loader = match MjcfLoader::from_file(&model_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("failed to load {model_path}: {e}");
            std::process::exit(1)
        }
    };
    let mut model = loader.build_model();
    if let Some(dt) = dt {
        model.dt = dt;
    }
    let neutralised = neutralise(&mut model);

    let mut state: State = model.default_state();
    if let Some(q0) = &q0 {
        let expected = if mujoco_layout {
            mujoco_nq(&model)
        } else {
            model.nq
        };
        if q0.len() != expected {
            eprintln!(
                "--q0 has {} entries but the model needs {expected} in the {} layout",
                q0.len(),
                if mujoco_layout { "mujoco" } else { "phyz" }
            );
            std::process::exit(1)
        }
        let q0 = if mujoco_layout {
            from_mujoco(&model, q0)
        } else {
            q0.clone()
        };
        for (i, &x) in q0.iter().enumerate() {
            state.q[i] = x;
        }
    }
    if let Some(v0) = &v0 {
        if v0.len() != model.nv {
            eprintln!(
                "--v0 has {} entries but the model has nv = {}",
                v0.len(),
                model.nv
            );
            std::process::exit(1)
        }
        let v0 = if mujoco_layout {
            v_from_mujoco(&model, state.q.as_slice(), v0)
        } else {
            v0.clone()
        };
        for (i, &x) in v0.iter().enumerate() {
            state.v[i] = x;
        }
    }

    let (xf, _) = forward_kinematics(&model, &state);
    state.body_xform = xf;

    // Semi-implicit Euler over ABA — no contact, no actuation, gravity only.
    // Recorded from step 0 so the first row is the initial condition, which is
    // what makes an off-by-one in the comparison visible instead of silent.
    let record = |q: &[f64]| -> Vec<f64> {
        if mujoco_layout {
            to_mujoco(&model, q)
        } else {
            q.to_vec()
        }
    };
    let mut trajectory: Vec<Vec<f64>> = Vec::with_capacity(steps + 1);
    trajectory.push(record(state.q.as_slice()));
    for _ in 0..steps {
        let qdd = aba(&model, &state);
        state.v += &(&qdd * model.dt);
        let v = state.v.clone();
        integrate_configuration(&model, state.q.as_mut_slice(), v.as_slice(), model.dt);
        trajectory.push(record(state.q.as_slice()));
    }

    let joint_names: Vec<String> = model.joints.iter().map(|j| j.name.clone()).collect();
    let joint_types: Vec<String> = model
        .joints
        .iter()
        .map(|j| format!("{:?}", j.joint_type))
        .collect();
    // Zero-DOF joints occupy no configuration coordinate. phyz materialises one
    // for every body that MJCF attaches rigidly to its parent; MuJoCo has no
    // entry for those at all, so a consumer aligning the two coordinate vectors
    // has to drop them, and needs this to know which.
    let joint_ndof: Vec<usize> = model.joints.iter().map(|j| j.ndof()).collect();

    let json = serde_json::json!({
        "engine": "phyz",
        "model": model_path,
        "dt": model.dt,
        "steps": steps,
        "nq": if mujoco_layout { mujoco_nq(&model) } else { model.nq },
        "phyz_nq": model.nq,
        "layout": if mujoco_layout { "mujoco" } else { "phyz" },
        "nv": model.nv,
        "joint_names": joint_names,
        "joint_types": joint_types,
        "joint_ndof": joint_ndof,
        "q_offsets": model.q_offsets,
        "neutralised": neutralised,
        "q": trajectory,
    });
    let text = serde_json::to_string(&json).expect("serialise trajectory");

    match out {
        Some(path) => {
            if let Err(e) = std::fs::write(&path, &text) {
                eprintln!("failed to write {path}: {e}");
                std::process::exit(1)
            }
            eprintln!("wrote {path}");
        }
        None => println!("{text}"),
    }
}
