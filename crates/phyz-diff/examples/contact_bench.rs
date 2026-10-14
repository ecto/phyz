//! Emit the K1 contact step as C and time phyz's CPU contact step on the
//! same states.
//!
//! ```text
//! cargo run -p phyz-diff --release --example contact_bench -- OUT_DIR [SWEEPS]
//! ```
//!
//! Also writes `contact.metal` for `bench-c/metal_bench.m`, and `split.c` /
//! `split.metal` (kinematics, rolled solve, finish) for `bench-c/split_bench.c`
//! and `bench-c/metal_split.m`.
//!
//! States come from a 1 s CPU standing rollout (1 cm drop, servos at the
//! default pose, dt = 1 ms), so every one has realistic foot contact. Writes
//! `contact_f32.c`, `contact_f64.c`, `contact_inputs.bin` (state + the CPU's
//! own warm start) and `contact_reference.bin` (CPU `q', v', f`); check only
//! the first `nq + nv` outputs, since impulses are unique only up to
//! `null(Jᵀ)`.

use std::io::Write;
use std::time::Instant;

use phyz_collision::Collision;
use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::contact_split::ContactKernels;
use phyz_diff::contact_step::{ContactPoint, ContactStep};
use phyz_math::{DVec, SpatialTransformExt as _, Vec3};
use phyz_model::{Actuator, Model, State};
use tang_expr::codegen::Dialect;

fn main() {
    let mut args = std::env::args().skip(1);
    let out = std::path::PathBuf::from(args.next().expect("usage: contact_bench OUT_DIR [SWEEPS]"));
    let sweeps: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(5);
    std::fs::create_dir_all(&out).unwrap();

    let (model, points) = k1();
    let mat = ContactMaterial {
        friction: 0.8,
        margin: 0.005,
        ..ContactMaterial::default()
    };
    let config = ContactSolverConfig::default();

    // Standing rollout on the CPU; keep each step's input state and warm start.
    let mut s = model.default_state();
    let (xf, _) = phyz_rigid::forward_kinematics(&model, &s);
    let low = points
        .iter()
        .map(|p| xf[p.body].body_to_world_point(p.local).z)
        .fold(f64::INFINITY, f64::min);
    s.q[5] += 0.01 - low;
    let mut f = vec![0.0; 3 * points.len()];
    let mut states = Vec::new();
    let mut warms = Vec::new();
    let mut reference = Vec::new();
    for _ in 0..1000 {
        states.push(s.clone());
        warms.push(f.clone());
        let (q1, v1, f1) = cpu_step(&model, &points, &mat, &config, &s, &f);
        reference.extend_from_slice(&q1);
        reference.extend_from_slice(&v1);
        reference.extend_from_slice(&f1);
        s.q.as_mut_slice().copy_from_slice(&q1);
        s.v.as_mut_slice().copy_from_slice(&v1);
        f = f1;
    }

    let mut ns: Vec<f64> = Vec::new();
    for _ in 0..5 {
        for (s, w) in states.iter().zip(&warms) {
            let t = Instant::now();
            std::hint::black_box(cpu_step(&model, &points, &mat, &config, s, w));
            ns.push(t.elapsed().as_nanos() as f64);
        }
    }
    ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| ns[((ns.len() - 1) as f64 * p) as usize] / 1e3;

    let t = Instant::now();
    let step = ContactStep::build(&model, &points, &mat, &config, sweeps);
    let trace_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut inputs = Vec::new();
    for (s, w) in states.iter().zip(&warms) {
        inputs.extend(step.inputs(s, w));
    }
    std::fs::write(out.join("contact_f32.c"), step.kernel(Dialect::C).source).unwrap();
    std::fs::write(out.join("contact_f64.c"), step.kernel_c_f64().source).unwrap();
    std::fs::write(out.join("contact.metal"), step.kernel(Dialect::Msl).source).unwrap();

    // The split pipeline: traced kinematics, rolled solve, traced finish.
    let split = ContactKernels::build(&model, &points, &mat, &config);
    let (kin, fin) = split.traced_sources(Dialect::C);
    let src = format!("{kin}\n{}\n{fin}", split.solve_source(Dialect::C));
    std::fs::write(out.join("split.c"), src).unwrap();
    // The GPU pipeline is structure-of-arrays throughout.
    let (kin, fin) = split.traced_sources_soa();
    let src = format!("{kin}\n{}\n{fin}", split.solve_source(Dialect::Msl));
    std::fs::write(out.join("split.metal"), src).unwrap();
    let lay = split.layout();
    let (nk, nf) = split.live_nodes();
    write_f64(&out.join("contact_inputs.bin"), &inputs);
    write_f64(&out.join("contact_reference.bin"), &reference);

    let meta = format!(
        "split: nx {} nrec {} nscr {} nfin {} dim {} nout {} kinematics_nodes {nk} finish_nodes {nf}\n\
         contact: sweeps {sweeps} points {} n_inputs {} n_out {} n_check {} live_nodes {} trace_ms {trace_ms:.0}\n\
         phyz CPU contact step us: median {:.2} p99 {:.2}\n",
        split.kinematics_inputs(),
        lay.len,
        lay.scratch,
        split.finish_inputs(),
        3 * points.len(),
        model.nq + model.nv,
        points.len(),
        step.n_inputs(),
        step.n_outputs(),
        model.nq + model.nv,
        step.live_nodes(),
        pct(0.5),
        pct(0.99),
    );
    std::fs::write(out.join("contact_meta.txt"), &meta).unwrap();
    print!("{meta}");
}

/// The K1 at dt = 1 ms with position servos (D as implicit joint damping) and
/// the vendor MJCF foot pads as eight sole corners. Mirrors
/// `tests/contact_parity.rs`.
fn k1() -> (Model, Vec<ContactPoint>) {
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
    let mut model = phyz_urdf::load_file(root.join("K1_22dof_floating.urdf"), &opts)
        .expect("K1 URDF")
        .model;
    model.actuators.clear();
    for j in 0..model.joints.len() {
        if model.joints[j].ndof() == 1 {
            let name = model.joints[j].name.clone();
            let leg = ["Hip", "Knee", "Ankle"].iter().any(|k| name.contains(k));
            let (kp, kd) = if leg { (80.0, 2.0) } else { (20.0, 0.5) };
            model.joints[j].damping = kd;
            model
                .actuators
                .push(Actuator::position(&name, &name, j, 1.0, kp, 0.0));
        }
    }
    let mut points = Vec::new();
    for foot in ["left_foot_link", "right_foot_link"] {
        let body = model.bodies.iter().position(|b| b.name == foot).unwrap();
        for (x, y) in [
            (0.116, 0.035),
            (0.116, -0.035),
            (-0.064, 0.035),
            (-0.064, -0.035),
        ] {
            points.push(ContactPoint {
                body,
                local: Vec3::new(x, y, -0.038),
            });
        }
    }
    (model, points)
}

/// FK, ABA, contact assembly, warm-started solve, integration: ipse's
/// booster substep without its plant extras. Returns `(q', v', f)`.
fn cpu_step(
    model: &Model,
    points: &[ContactPoint],
    mat: &ContactMaterial,
    config: &ContactSolverConfig,
    s: &State,
    warm: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let dt = model.dt;
    let mut st = s.clone();
    let (xf, _) = phyz_rigid::forward_kinematics(model, &st);
    st.body_xform = xf.clone();
    let mut qdd = phyz_rigid::aba(model, &st);
    let v_before = st.v.clone();
    phyz_rigid::strip_free_joint_coriolis(model, v_before.as_slice(), qdd.as_mut_slice());
    let free_qd: DVec = &st.v + &(&qdd * dt);
    let mut contacts = Vec::new();
    let mut which = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let w = xf[p.body].body_to_world_point(p.local);
        let depth = -w.z;
        if depth > -mat.margin {
            contacts.push(Collision {
                body_i: p.body,
                body_j: Collision::WORLD,
                contact_point: w + Vec3::new(0.0, 0.0, depth * 0.5),
                contact_normal: Vec3::new(0.0, 0.0, 1.0),
                penetration_depth: depth,
            });
            which.push(i);
        }
    }
    let mut f = vec![0.0; 3 * points.len()];
    if contacts.is_empty() {
        st.v = free_qd;
    } else {
        let mats = model.contact_materials(mat);
        let asm = phyz_contact::assemble(model, &st, &contacts, &mats, &free_qd, dt, config);
        let seed: Vec<Vec3> = which
            .iter()
            .map(|&i| Vec3::new(warm[3 * i], warm[3 * i + 1], warm[3 * i + 2]))
            .collect();
        let sol = phyz_contact::solve_contacts_warm(&asm.problem, config, &seed);
        for (k, &i) in which.iter().enumerate() {
            let imp = sol.impulses[k];
            f[3 * i..3 * i + 3].copy_from_slice(&[imp.x, imp.y, imp.z]);
        }
        st.v = &free_qd + &asm.velocity_delta(&sol.impulses);
    }
    phyz_rigid::rotate_free_joint_velocities(model, v_before.as_slice(), st.v.as_mut_slice(), dt);
    let v = st.v.clone();
    phyz_rigid::integrate_configuration(model, st.q.as_mut_slice(), v.as_slice(), dt);
    (st.q.as_slice().to_vec(), st.v.as_slice().to_vec(), f)
}

fn write_f64(path: &std::path::Path, xs: &[f64]) {
    let mut f = std::fs::File::create(path).unwrap();
    for x in xs {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
}
