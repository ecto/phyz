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
use phyz_mjcf::MjcfLoader;
use phyz_rigid::{aba, forward_kinematics, integrate_configuration};

fn usage() -> ! {
    eprintln!(
        "usage: phyz-traj --model PATH [--steps N] [--dt SECONDS] [--q0 a,b,c] [--out PATH]\n\
         \n\
         Writes {{model, dt, steps, nq, nv, joint_names, neutralised, q: [[...]]}} as JSON.\n\
         With no --out, writes to stdout."
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
    let mut out: Option<String> = None;

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
            "--out" => out = Some(args.next().unwrap_or_else(|| usage())),
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
        if q0.len() != model.nq {
            eprintln!(
                "--q0 has {} entries but the model has nq = {}",
                q0.len(),
                model.nq
            );
            std::process::exit(1)
        }
        for (i, &x) in q0.iter().enumerate() {
            state.q[i] = x;
        }
    }
    let (xf, _) = forward_kinematics(&model, &state);
    state.body_xform = xf;

    // Semi-implicit Euler over ABA — no contact, no actuation, gravity only.
    // Recorded from step 0 so the first row is the initial condition, which is
    // what makes an off-by-one in the comparison visible instead of silent.
    let mut trajectory: Vec<Vec<f64>> = Vec::with_capacity(steps + 1);
    trajectory.push(state.q.as_slice().to_vec());
    for _ in 0..steps {
        let qdd = aba(&model, &state);
        state.v += &(&qdd * model.dt);
        let v = state.v.clone();
        integrate_configuration(&model, state.q.as_mut_slice(), v.as_slice(), model.dt);
        trajectory.push(state.q.as_slice().to_vec());
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
        "nq": model.nq,
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
