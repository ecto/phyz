//! The traced ABA graph must match `phyz_rigid::aba` on random states.

use phyz_diff::forward::{ForwardDynamics, Step};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Actuator, Model, ModelBuilder, State};

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
        Vec3::new(0.02, -0.1, 0.01),
        Mat3::from_diagonal(&Vec3::new(0.01 * mass, 0.004 * mass, 0.012 * mass)),
    )
}

/// A floating trunk with two 3-link legs and an arm, random axes, armature
/// and damping: the shape of a humanoid without its size.
fn floating_tree() -> Model {
    let off = |x: f64, y: f64, z: f64| SpatialTransform::from_translation(Vec3::new(x, y, z));
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(0.002)
        .add_free_body("trunk", -1, SpatialTransform::identity(), link(8.0))
        .add_revolute_body("l_hip", 0, off(0.0, 0.1, -0.1), link(1.5))
        .add_revolute_body("l_knee", 1, off(0.0, 0.0, -0.3), link(1.0))
        .add_revolute_body("l_ankle", 2, off(0.0, 0.0, -0.3), link(0.4))
        .add_revolute_body("r_hip", 0, off(0.0, -0.1, -0.1), link(1.5))
        .add_revolute_body("r_knee", 4, off(0.0, 0.0, -0.3), link(1.0))
        .add_revolute_body("r_ankle", 5, off(0.0, 0.0, -0.3), link(0.4))
        .add_revolute_body("arm", 0, off(0.0, 0.2, 0.3), link(0.8))
        .build();
    let mut rng = Lcg(7);
    for j in model.joints.iter_mut().skip(1) {
        let a = Vec3::new(rng.next(), rng.next(), rng.next());
        j.axis = a * (1.0 / a.norm());
        j.armature = 0.01 + 0.02 * rng.next().abs();
        j.damping = 0.5 * rng.next().abs();
    }
    model
}

/// Worst relative error of the graph against `phyz_rigid::aba` over `n` states.
fn worst_error(model: &Model, n: usize, seed: u64) -> f64 {
    let fd = ForwardDynamics::build(model);
    let eval = fd.compile();
    let mut rng = Lcg(seed);
    let mut out = vec![0.0; model.nv];
    let mut worst: f64 = 0.0;
    for _ in 0..n {
        let mut state = model.default_state();
        for k in 0..model.nq {
            state.q[k] = rng.next();
        }
        for k in 0..model.nv {
            state.v[k] = 2.0 * rng.next();
        }
        for k in 0..state.ctrl.len() {
            state.ctrl[k] = rng.next();
        }
        let qfrc = phyz_rigid::actuation::generalized_forces(model, &state);
        let want = phyz_rigid::aba(model, &state);
        eval(&fd.inputs(&state, qfrc.as_slice()), &mut out);
        let scale = want.as_slice().iter().fold(1.0f64, |m, x| m.max(x.abs()));
        for k in 0..model.nv {
            worst = worst.max((out[k] - want[k]).abs() / scale);
        }
    }
    worst
}

#[test]
fn floating_tree_matches_phyz_aba() {
    let model = floating_tree();
    let err = worst_error(&model, 1000, 11);
    assert!(err < 1e-10, "worst relative error {err:e}");
}

/// The real K1, when its assets are installed (`IPSE_K1_ASSETS` or
/// `~/Developer/booster_assets/robots/K1`).
fn k1() -> Option<Model> {
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
        dt: Some(0.002),
        gravity: None,
    };
    Some(phyz_urdf::load_file(&path, &opts).unwrap().model)
}

#[test]
fn k1_matches_phyz_aba() {
    let Some(model) = k1() else { return };
    let fd = ForwardDynamics::build(&model);
    eprintln!(
        "K1: nq {} nv {} bodies {} live nodes {}",
        model.nq,
        model.nv,
        model.nbodies(),
        fd.live_nodes()
    );
    let err = worst_error(&model, 1000, 13);
    eprintln!("K1 worst relative error {err:e}");
    assert!(err < 1e-9, "worst relative error {err:e}");
}

/// `floating_tree` plus position servos with clamps, soft limits, springs and
/// dry friction, so every branch of the force law is live.
fn actuated_tree() -> Model {
    let mut model = floating_tree();
    let mut rng = Lcg(23);
    for j in 1..model.joints.len() {
        let joint = &mut model.joints[j];
        joint.limits = Some([-0.4, 0.5]);
        joint.limit_stiffness = 200.0;
        joint.limit_damping = 2.0;
        joint.stiffness = 3.0 * rng.next().abs();
        joint.spring_ref = 0.1 * rng.next();
        joint.friction_loss = 0.2 * rng.next().abs();
        let name = joint.name.clone();
        let mut act = Actuator::position(&name, &name, j, 1.0, 40.0, 1.5);
        act.ctrl_range = Some([-0.6, 0.6]);
        act.force_range = Some([-15.0, 15.0]);
        model.actuators.push(act);
    }
    model
}

fn random_state(model: &Model, rng: &mut Lcg) -> State {
    let mut s = model.default_state();
    for k in 0..model.nq {
        s.q[k] = rng.next();
    }
    for k in 0..model.nv {
        s.v[k] = 2.0 * rng.next();
    }
    for k in 0..s.ctrl.len() {
        s.ctrl[k] = rng.next();
    }
    s
}

fn phyz_step(model: &Model, s: &mut State) {
    let qdd = phyz_rigid::aba(model, s);
    phyz_rigid::semi_implicit_euler(model, s, qdd.as_slice(), model.dt);
}

/// Worst error of one traced step against phyz over `n` random states, scaled
/// per state by `max(1, |x|_inf)` over `[q', v']`.
fn worst_step_error(model: &Model, n: usize, seed: u64) -> f64 {
    let step = Step::build(model);
    let eval = step.compile();
    let mut rng = Lcg(seed);
    let mut out = vec![0.0; model.nq + model.nv];
    let mut worst: f64 = 0.0;
    for _ in 0..n {
        let mut s = random_state(model, &mut rng);
        eval(&step.inputs(&s), &mut out);
        phyz_step(model, &mut s);
        let want: Vec<f64> =
            s.q.as_slice()
                .iter()
                .chain(s.v.as_slice())
                .copied()
                .collect();
        let scale = want.iter().fold(1.0f64, |m, x| m.max(x.abs()));
        for (a, b) in out.iter().zip(&want) {
            worst = worst.max((a - b).abs() / scale);
        }
    }
    worst
}

#[test]
fn actuated_tree_step_matches_phyz() {
    let model = actuated_tree();
    let err = worst_step_error(&model, 1000, 29);
    assert!(err < 1e-12, "worst step error {err:e}");
}

/// 1000 steps (2 s) from rest under a fixed servo command: the traced and
/// reference trajectories stay together.
#[test]
fn actuated_tree_rollout_matches_phyz() {
    let model = actuated_tree();
    let step = Step::build(&model);
    let eval = step.compile();
    let mut rng = Lcg(31);
    let mut s = model.default_state();
    for k in 0..s.ctrl.len() {
        s.ctrl[k] = 0.5 * rng.next();
    }
    let mut t = s.clone();
    let mut out = vec![0.0; model.nq + model.nv];
    let mut worst: f64 = 0.0;
    for _ in 0..1000 {
        eval(&step.inputs(&t), &mut out);
        t.q.as_mut_slice().copy_from_slice(&out[..model.nq]);
        t.v.as_mut_slice().copy_from_slice(&out[model.nq..]);
        phyz_step(&model, &mut s);
        for k in 0..model.nq {
            worst = worst.max((t.q[k] - s.q[k]).abs());
        }
    }
    assert!(worst < 1e-9, "rollout drift {worst:e}");
}

#[test]
fn k1_step_matches_phyz() {
    let Some(model) = k1() else { return };
    let step = Step::build(&model);
    eprintln!("K1 step: nu {} live nodes {}", step.nu, step.live_nodes());
    let err = worst_step_error(&model, 1000, 37);
    eprintln!("K1 step worst error {err:e}");
    assert!(err < 1e-12, "worst step error {err:e}");
}
