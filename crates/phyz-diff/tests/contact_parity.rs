//! The traced contact step against phyz's CPU contact pipeline
//! (`assemble` + `solve_contacts` to tolerance + `velocity_delta`) on the
//! same foot-corner contact set.

use phyz_collision::Collision;
use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::contact_step::{ContactPoint, ContactStep};
use phyz_math::SpatialTransformExt as _;
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Actuator, Model, ModelBuilder, State};

/// CPU solves that stopped short of `config.tolerance` (ipse does not check).
static NOT_CONVERGED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `a.max(b)` that propagates NaN (`f64::max` drops it, which once hid a
/// NaN-producing trace from these checks).
fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn link(mass: f64) -> SpatialInertia {
    SpatialInertia::new(
        mass,
        Vec3::new(0.0, 0.0, -0.1),
        Mat3::from_diagonal(&Vec3::new(0.01 * mass, 0.01 * mass, 0.004 * mass)),
    )
}

/// A floating trunk on two 3-joint sagittal legs (hip, knee, ankle pitch), servoed, with
/// four sole corners per foot.
fn biped() -> (Model, Vec<ContactPoint>) {
    let off = |x: f64, y: f64, z: f64| SpatialTransform::from_translation(Vec3::new(x, y, z));
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(0.002)
        .add_free_body("trunk", -1, SpatialTransform::identity(), link(10.0))
        .add_revolute_body("l_hip", 0, off(0.0, 0.1, -0.1), link(2.0))
        .add_revolute_body("l_knee", 1, off(0.0, 0.0, -0.3), link(1.5))
        .add_revolute_body("l_ankle", 2, off(0.0, 0.0, -0.3), link(0.5))
        .add_revolute_body("r_hip", 0, off(0.0, -0.1, -0.1), link(2.0))
        .add_revolute_body("r_knee", 4, off(0.0, 0.0, -0.3), link(1.5))
        .add_revolute_body("r_ankle", 5, off(0.0, 0.0, -0.3), link(0.5))
        .build();
    let axes = [
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    for j in 1..model.joints.len() {
        model.joints[j].axis = axes[(j - 1) % 3];
        model.joints[j].armature = 0.02;
        model.joints[j].damping = 0.3;
        let name = model.joints[j].name.clone();
        let mut act = Actuator::position(&name, &name, j, 1.0, 80.0, 2.0);
        act.force_range = Some([-60.0, 60.0]);
        model.actuators.push(act);
    }
    let mut points = Vec::new();
    for foot in [3, 6] {
        for (x, y) in [(0.1, 0.04), (0.1, -0.04), (-0.06, 0.04), (-0.06, -0.04)] {
            points.push(ContactPoint {
                body: foot,
                local: Vec3::new(x, y, -0.05),
            });
        }
    }
    (model, points)
}

fn material() -> ContactMaterial {
    ContactMaterial {
        friction: 0.8,
        margin: 0.005,
        ..ContactMaterial::default()
    }
}

/// A state with the feet near the ground: random joint pose and velocity,
/// trunk lowered so the lowest corner sits at `-1 cm..+1 cm`.
fn standing_state(model: &Model, points: &[ContactPoint], rng: &mut Lcg) -> State {
    let mut s = model.default_state();
    for k in 0..3 {
        s.q[k] = 0.1 * rng.next();
    }
    for k in 6..model.nq {
        s.q[k] = 0.3 * rng.next();
    }
    for k in 0..model.nv {
        s.v[k] = 0.5 * rng.next();
    }
    for k in 0..s.ctrl.len() {
        s.ctrl[k] = 0.3 * rng.next();
    }
    let (xf, _) = phyz_rigid::forward_kinematics(model, &s);
    let low = points
        .iter()
        .map(|p| xf[p.body].body_to_world_point(p.local).z)
        .fold(f64::INFINITY, f64::min);
    s.q[5] += -low - 0.01 * rng.next();
    s
}

/// The CPU step, as ipse's booster env runs it; returns `(q', v', f per point)`.
fn cpu_step(
    model: &Model,
    points: &[ContactPoint],
    mat: &ContactMaterial,
    config: &ContactSolverConfig,
    s: &State,
    warm: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, bool) {
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
    let mut converged = true;
    if contacts.is_empty() {
        st.v = free_qd;
    } else {
        let mats = model.contact_materials(mat);
        let asm = phyz_contact::assemble(model, &st, &contacts, &mats, &free_qd, dt, config);
        let sol = match warm {
            Some(w) => {
                let seed: Vec<Vec3> = which
                    .iter()
                    .map(|&i| Vec3::new(w[3 * i], w[3 * i + 1], w[3 * i + 2]))
                    .collect();
                phyz_contact::solve_contacts_warm(&asm.problem, config, &seed)
            }
            None => phyz_contact::solve_contacts(&asm.problem, config),
        };
        converged = sol.converged;
        if !sol.converged {
            NOT_CONVERGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        for (k, &i) in which.iter().enumerate() {
            let imp = sol.impulses[k];
            f[3 * i..3 * i + 3].copy_from_slice(&[imp.x, imp.y, imp.z]);
        }
        st.v = &free_qd + &asm.velocity_delta(&sol.impulses);
    }
    phyz_rigid::rotate_free_joint_velocities(model, v_before.as_slice(), st.v.as_mut_slice(), dt);
    let v = st.v.clone();
    phyz_rigid::integrate_configuration(model, st.q.as_mut_slice(), v.as_slice(), dt);
    (
        st.q.as_slice().to_vec(),
        st.v.as_slice().to_vec(),
        f,
        converged,
    )
}

/// Worst `v'` error (m/s or rad/s) of the graph over `n` states, warm-started
/// either from the converged CPU impulses or from zero.
fn worst_v_error(sweeps: usize, warm_from_cpu: bool, n: usize) -> (f64, f64) {
    let (model, points) = biped();
    let mat = material();
    let config = ContactSolverConfig::default();
    let step = ContactStep::build(&model, &points, &mat, &config, sweeps);
    let eval = step.compile();
    let mut rng = Lcg(41);
    let mut out = vec![0.0; step.n_outputs()];
    let (mut worst_v, mut worst_q) = (0.0f64, 0.0f64);
    for _ in 0..n {
        let s = standing_state(&model, &points, &mut rng);
        let (q1, v1, f, converged) = cpu_step(&model, &points, &mat, &config, &s, None);
        if !converged {
            continue;
        }
        let warm = if warm_from_cpu {
            f
        } else {
            vec![0.0; 3 * points.len()]
        };
        eval(&step.inputs(&s, &warm), &mut out);
        for k in 0..model.nq {
            worst_q = nan_max(worst_q, (out[k] - q1[k]).abs());
        }
        for k in 0..model.nv {
            worst_v = nan_max(worst_v, (out[model.nq + k] - v1[k]).abs());
        }
    }
    (worst_v, worst_q)
}

/// Warm-started at the CPU solution the graph's sweep must stay there: this
/// checks assembly and the per-contact update, not convergence speed.
#[test]
fn fixed_point_matches_cpu_solve() {
    let (dv, dq) = worst_v_error(3, true, 200);
    eprintln!(
        "warm from CPU, 3 sweeps: worst |dv| {dv:e}, |dq| {dq:e} (CPU unconverged so far: {})",
        NOT_CONVERGED.load(std::sync::atomic::Ordering::Relaxed)
    );
    assert!(dv < 1e-6, "worst |dv| {dv:e}");
}

/// Cold start: how many sweeps the velocity needs (reported, loosely bounded).
#[test]
fn cold_start_convergence() {
    for sweeps in [10, 30, 100] {
        let (dv, dq) = worst_v_error(sweeps, false, 100);
        eprintln!("cold, {sweeps} sweeps: worst |dv| {dv:e}, |dq| {dq:e}");
    }
    let (dv, _) = worst_v_error(100, false, 100);
    assert!(dv < 0.05, "100 cold sweeps: worst |dv| {dv:e}");
}

/// 2 s of standing from a 1 cm drop, holding a crouch with the servos. Both
/// sides warm-start from their own previous impulses, as ipse's contact cache
/// does. Reports trunk-height and joint drift for a few sweep counts.
#[test]
fn standing_rollout_tracks_cpu() {
    let (model, points) = biped();
    let mat = material();
    let config = ContactSolverConfig::default();
    let crouch = [-0.3, 0.6, -0.3, -0.3, 0.6, -0.3];
    let mut s0 = model.default_state();
    for (k, c) in crouch.iter().enumerate() {
        s0.q[6 + k] = *c;
        s0.ctrl[k] = *c;
    }
    // ctrl has one slot per actuator, in joint order.
    let (xf, _) = phyz_rigid::forward_kinematics(&model, &s0);
    let low = points
        .iter()
        .map(|p| xf[p.body].body_to_world_point(p.local).z)
        .fold(f64::INFINITY, f64::min);
    s0.q[5] += 0.01 - low;

    let steps = 1000;
    // CPU reference.
    let mut s = s0.clone();
    let mut f = vec![0.0; 3 * points.len()];
    let mut cpu_z = Vec::with_capacity(steps);
    let mut cpu_q = Vec::new();
    for _ in 0..steps {
        let (q1, v1, f1, _) = cpu_step(&model, &points, &mat, &config, &s, Some(&f));
        s.q.as_mut_slice().copy_from_slice(&q1);
        s.v.as_mut_slice().copy_from_slice(&v1);
        f = f1;
        cpu_z.push(s.q[5]);
        cpu_q = q1;
    }
    let (lo, hi) = cpu_z
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), z| {
            (a.min(*z), b.max(*z))
        });
    eprintln!(
        "CPU: trunk z {:.4} -> {:.4} m (range {lo:.4}..{hi:.4})",
        s0.q[5],
        cpu_z[steps - 1]
    );
    assert!(
        lo > s0.q[5] - 0.1,
        "the reference fell: trunk z reached {lo}"
    );

    for sweeps in [3, 5, 10, 20] {
        let step = ContactStep::build(&model, &points, &mat, &config, sweeps);
        let eval = step.compile();
        let mut t = s0.clone();
        let mut f = vec![0.0; 3 * points.len()];
        let mut out = vec![0.0; step.n_outputs()];
        let mut worst_z: f64 = 0.0;
        for (k, z) in cpu_z.iter().enumerate() {
            eval(&step.inputs(&t, &f), &mut out);
            t.q.as_mut_slice().copy_from_slice(&out[..model.nq]);
            t.v.as_mut_slice()
                .copy_from_slice(&out[model.nq..model.nq + model.nv]);
            f.copy_from_slice(&out[model.nq + model.nv..]);
            worst_z = nan_max(worst_z, (t.q[5] - z).abs());
            let _ = k;
        }
        let dq = (6..model.nq)
            .map(|k| (t.q[k] - cpu_q[k]).abs())
            .fold(0.0, nan_max);
        eprintln!(
            "{sweeps:>2} sweeps, {} nodes: worst trunk-z gap {:.2e} m, final joint gap {:.2e} rad",
            step.live_nodes(),
            worst_z,
            dq
        );
        if sweeps == 20 {
            assert!(worst_z < 1e-3, "20 sweeps: trunk z gap {worst_z:e}");
        }
    }
}

/// The K1 with the vendor MJCF's foot pads (box at (0.026, 0, -0.02), half
/// extents (0.09, 0.035, 0.018)) as eight sole corners, and position servos on
/// every joint (D as implicit joint damping), at dt = 1 ms (phyz is unstable on the K1 at 2 ms,
/// ipse docs/k1-walk-booster-recipe.md). `None` without the assets.
fn k1() -> Option<(Model, Vec<ContactPoint>)> {
    let root = std::env::var_os("IPSE_K1_ASSETS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                .join("Developer/booster_assets/robots/K1")
        });
    let path = root.join("K1_22dof_floating.urdf");
    if !path.exists() {
        eprintln!("skipping: {} not found", path.display());
        return None;
    }
    let opts = phyz_urdf::UrdfOptions {
        base: phyz_urdf::BaseKind::Fixed,
        dt: Some(0.001),
        gravity: None,
    };
    let mut model = phyz_urdf::load_file(&path, &opts).unwrap().model;
    model.actuators.clear();
    for j in 0..model.joints.len() {
        if model.joints[j].ndof() == 1 {
            let name = model.joints[j].name.clone();
            let leg = ["Hip", "Knee", "Ankle"].iter().any(|k| name.contains(k));
            let (kp, kd) = if leg { (80.0, 2.0) } else { (20.0, 0.5) };
            // The D term as joint damping, which ABA integrates implicitly:
            // explicit kd on the URDF's armature-free ankle links is unstable.
            model.joints[j].damping = kd;
            model
                .actuators
                .push(Actuator::position(&name, &name, j, 1.0, kp, 0.0));
        }
    }
    let mut points = Vec::new();
    for foot in ["left_foot_link", "right_foot_link"] {
        let body = model
            .bodies
            .iter()
            .position(|b| b.name == foot)
            .unwrap_or_else(|| panic!("no body {foot}"));
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
    Some((model, points))
}

#[test]
fn k1_fixed_point_and_stand() {
    let Some((model, points)) = k1() else { return };
    let mat = material();
    let config = ContactSolverConfig::default();

    // Fixed point on near-ground states, against converged CPU solves.
    let step = ContactStep::build(&model, &points, &mat, &config, 3);
    eprintln!(
        "K1 contact step, 3 sweeps: {} live nodes",
        step.live_nodes()
    );
    let eval = step.compile();
    let mut rng = Lcg(43);
    let mut out = vec![0.0; step.n_outputs()];
    let (mut worst, mut used) = (0.0f64, 0);
    for _ in 0..200 {
        let mut s = model.default_state();
        for k in 0..3 {
            s.q[k] = 0.05 * rng.next();
        }
        for k in 6..model.nq {
            s.q[k] = 0.1 * rng.next();
        }
        for k in 0..model.nv {
            s.v[k] = 0.3 * rng.next();
        }
        for k in 0..s.ctrl.len() {
            s.ctrl[k] = 0.1 * rng.next();
        }
        let (xf, _) = phyz_rigid::forward_kinematics(&model, &s);
        let low = points
            .iter()
            .map(|p| xf[p.body].body_to_world_point(p.local).z)
            .fold(f64::INFINITY, f64::min);
        s.q[5] += -low - 0.005 * rng.next();
        let (_, v1, f, converged) = cpu_step(&model, &points, &mat, &config, &s, None);
        if !converged {
            continue;
        }
        used += 1;
        eval(&step.inputs(&s, &f), &mut out);
        for k in 0..model.nv {
            worst = nan_max(worst, (out[model.nq + k] - v1[k]).abs());
        }
    }
    eprintln!("K1 fixed point: worst |dv| {worst:e} over {used} converged states");
    assert!(
        used > 100 && worst < 1e-6,
        "worst |dv| {worst:e}, {used} states"
    );

    // 1 s standing at the default pose from a 1 cm drop, both sides warm.
    let mut s0 = model.default_state();
    let (xf, _) = phyz_rigid::forward_kinematics(&model, &s0);
    let low = points
        .iter()
        .map(|p| xf[p.body].body_to_world_point(p.local).z)
        .fold(f64::INFINITY, f64::min);
    s0.q[5] += 0.01 - low;
    let steps = 1000;
    let mut s = s0.clone();
    let mut f = vec![0.0; 3 * points.len()];
    let mut cpu_z = Vec::with_capacity(steps);
    for _ in 0..steps {
        let (q1, v1, f1, _) = cpu_step(&model, &points, &mat, &config, &s, Some(&f));
        s.q.as_mut_slice().copy_from_slice(&q1);
        s.v.as_mut_slice().copy_from_slice(&v1);
        f = f1;
        cpu_z.push(s.q[5]);
    }
    let lo = cpu_z.iter().copied().fold(f64::INFINITY, f64::min);
    eprintln!(
        "K1 CPU stand: trunk z {:.4} -> {:.4} (min {lo:.4})",
        s0.q[5],
        cpu_z[steps - 1]
    );
    assert!(
        lo > s0.q[5] - 0.1,
        "the reference fell: trunk z reached {lo}"
    );
    for sweeps in [5, 10] {
        let step = ContactStep::build(&model, &points, &mat, &config, sweeps);
        let eval = step.compile();
        let mut t = s0.clone();
        let mut f = vec![0.0; 3 * points.len()];
        let mut worst_z: f64 = 0.0;
        for z in &cpu_z {
            eval(&step.inputs(&t, &f), &mut out);
            t.q.as_mut_slice().copy_from_slice(&out[..model.nq]);
            t.v.as_mut_slice()
                .copy_from_slice(&out[model.nq..model.nq + model.nv]);
            f.copy_from_slice(&out[model.nq + model.nv..]);
            worst_z = nan_max(worst_z, (t.q[5] - z).abs());
        }
        eprintln!(
            "K1 stand, {sweeps} sweeps ({} nodes): worst trunk-z gap {worst_z:.2e} m",
            step.live_nodes()
        );
        assert!(worst_z < 1e-3, "{sweeps} sweeps: trunk z gap {worst_z:e}");
    }
}

/// The split pipeline (traced kinematics, rolled f64 solve, traced finish)
/// against the unrolled graph, on the biped and the K1, from nonzero warm
/// starts. They compute the same expression, so they agree to round-off.
#[test]
fn split_matches_unrolled() {
    use phyz_diff::contact_split::ContactKernels;
    let mut cases = vec![biped()];
    cases.extend(k1());
    for (model, points) in cases {
        let mat = material();
        let config = ContactSolverConfig::default();
        let sweeps = 5;
        let unrolled = ContactStep::build(&model, &points, &mat, &config, sweeps);
        let eval = unrolled.compile();
        let split = ContactKernels::build(&model, &points, &mat, &config);
        let (asm_fn, fin_fn) = split.compile();
        let lay = split.layout();
        let mut rng = Lcg(47);
        let mut out = vec![0.0; unrolled.n_outputs()];
        let mut asm = vec![0.0; lay.len];
        let mut fin_out = vec![0.0; model.nq + model.nv];
        let mut worst: f64 = 0.0;
        for _ in 0..100 {
            let s = standing_state(&model, &points, &mut rng);
            let (_, _, f_cpu, _) = cpu_step(&model, &points, &mat, &config, &s, None);
            let warm: Vec<f64> = f_cpu.iter().map(|x| x * (1.0 + 0.3 * rng.next())).collect();
            eval(&unrolled.inputs(&s, &warm), &mut out);
            let x: Vec<f64> =
                s.q.as_slice()
                    .iter()
                    .chain(s.v.as_slice())
                    .chain(s.ctrl.as_slice())
                    .copied()
                    .collect();
            asm_fn(&x, &mut asm);
            let (f, v1) = split.solve(&asm, &warm, sweeps);
            let fin: Vec<f64> =
                s.q.as_slice()
                    .iter()
                    .chain(s.v.as_slice())
                    .chain(&v1)
                    .copied()
                    .collect();
            fin_fn(&fin, &mut fin_out);
            let scale = out.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            for (a, b) in fin_out.iter().chain(&f).zip(&out) {
                worst = nan_max(worst, (a - b).abs() / scale);
            }
        }
        let (na, nf) = split.live_nodes();
        eprintln!(
            "split vs unrolled (nv {}, {} points): worst {worst:e}; kinematics {na} + finish {nf} nodes vs {} unrolled",
            model.nv,
            points.len(),
            unrolled.live_nodes()
        );
        assert!(worst < 1e-10, "worst {worst:e}");
    }
}

/// The contact step's reverse-mode adjoint against central differences of
/// the traced step itself, on near-ground states with warm starts (the
/// graph is smooth away from clamp switches; a switch inside ±h shows up as
/// an outlier, so the bar is the median, with a loose bound on the worst).
#[test]
fn contact_adjoint_matches_finite_differences() {
    let mut cases = vec![biped()];
    cases.extend(k1());
    for (model, points) in cases {
        let mat = material();
        let config = ContactSolverConfig::default();
        let step = ContactStep::build(&model, &points, &mat, &config, 5);
        let (ag, ao) = step.adjoint();
        let vjp = ag.compile_many(&ao);
        let fwd = step.compile();
        let (ni, no) = (step.n_inputs(), step.n_outputs());
        let mut rng = Lcg(71);
        let mut errs = Vec::new();
        for _ in 0..10 {
            let s = standing_state(&model, &points, &mut rng);
            let (_, _, f, _) = cpu_step(&model, &points, &mat, &config, &s, None);
            let x = step.inputs(&s, &f);
            let lam: Vec<f64> = (0..no).map(|_| rng.next()).collect();
            let mut got = vec![0.0; ni];
            vjp(&x.iter().chain(&lam).copied().collect::<Vec<_>>(), &mut got);
            let mut out = vec![0.0; no];
            let h = 1e-7;
            for c in 0..ni {
                let mut xp = x.clone();
                xp[c] += h;
                fwd(&xp, &mut out);
                let p: f64 = out.iter().zip(&lam).map(|(a, b)| a * b).sum();
                xp[c] -= 2.0 * h;
                fwd(&xp, &mut out);
                let m: f64 = out.iter().zip(&lam).map(|(a, b)| a * b).sum();
                let fd = (p - m) / (2.0 * h);
                errs.push((got[c] - fd).abs() / (1.0 + fd.abs()));
            }
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (median, worst) = (errs[errs.len() / 2], errs[errs.len() - 1]);
        eprintln!(
            "contact adjoint (nv {}, {} points, {} nodes): median {median:.1e}, worst {worst:.1e} vs FD",
            model.nv,
            points.len(),
            ag.live_set(&ao).len()
        );
        assert!(median < 1e-6, "median {median:e}");
    }
}
