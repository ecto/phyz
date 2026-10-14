//! The traced ABA graph must match `phyz_rigid::aba` on random states.

use phyz_diff::forward::ForwardDynamics;
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Model, ModelBuilder};

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
#[test]
fn k1_matches_phyz_aba() {
    let root = std::env::var_os("IPSE_K1_ASSETS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                .join("Developer/booster_assets/robots/K1")
        });
    let path = root.join("K1_22dof_floating.urdf");
    if !path.exists() {
        eprintln!("skipping: {} not found", path.display());
        return;
    }
    let opts = phyz_urdf::UrdfOptions {
        base: phyz_urdf::BaseKind::Fixed,
        dt: Some(0.002),
        gravity: None,
    };
    let model = phyz_urdf::load_file(&path, &opts).unwrap().model;
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
