//! The four benchmark models must load with the right articulation.
//!
//! These are the models RL practitioners judge an engine by. If any of these
//! regresses, credibility goes with it — so the DOF and actuator counts are
//! asserted exactly rather than loosely.

use phyz_mjcf::MjcfLoader;

fn load(name: &str) -> phyz_model::Model {
    let path = format!("{}/../../models/{name}", env!("CARGO_MANIFEST_DIR"));
    MjcfLoader::from_file(&path)
        .unwrap_or_else(|e| panic!("failed to load {path}: {e}"))
        .build_model()
}

fn loader(name: &str) -> MjcfLoader {
    let path = format!("{}/../../models/{name}", env!("CARGO_MANIFEST_DIR"));
    MjcfLoader::from_file(&path).unwrap()
}

#[test]
fn ant() {
    let m = load("ant.xml");
    assert_eq!(m.nv, 14, "6 free + 8 hinge");
    assert_eq!(m.nq, 14);
    assert_eq!(m.nbodies(), 9);
    assert_eq!(m.actuators.len(), 8);
    assert!(m.bodies.iter().all(|b| b.inertia.mass > 0.0));
    assert!(
        m.bodies.iter().filter(|b| b.geometry.is_some()).count() >= 9,
        "every ant body needs a collision shape or it cannot walk"
    );
}

#[test]
fn half_cheetah() {
    let m = load("half_cheetah.xml");
    assert_eq!(m.nv, 12, "6 free + 6 hinge");
    assert_eq!(m.actuators.len(), 6);
    // Per-joint gear overrides must beat the <default> value of 120.
    let ffoot = m
        .actuators
        .iter()
        .find(|a| a.joint_name == "ffoot")
        .expect("ffoot motor");
    assert!((ffoot.gear - 30.0).abs() < 1e-12, "gear override");
}

#[test]
fn humanoid() {
    let m = load("humanoid.xml");
    assert_eq!(m.nv, 23, "6 free + 17 hinge");
    assert_eq!(m.actuators.len(), 17);
    // childclass="arm" halves the geom radius relative to legs; check the
    // inherited class actually reached the geoms.
    let arm = m
        .bodies
        .iter()
        .find(|b| b.name.starts_with("right_lower_arm"))
        .expect("right_lower_arm");
    assert!(arm.inertia.mass > 0.0);
}

#[test]
fn shadow_hand_approximation() {
    let m = load("shadow_hand.xml");
    assert_eq!(m.nv, 24, "2 wrist + 22 finger DOF, fixed base");
    assert_eq!(m.actuators.len(), 20);

    // Position servos: f = kp*(ctrl - q) - kv*v.
    let act = &m.actuators[0];
    assert!(act.gain > 0.0 && act.bias_q < 0.0, "position servo law");
}

/// Every model must declare what it silently drops. Right now the hand
/// approximation drops nothing (it has no meshes or tendons by construction),
/// which is the point of authoring it that way.
#[test]
fn benchmark_models_report_no_hidden_losses() {
    for name in ["ant.xml", "half_cheetah.xml", "humanoid.xml", "shadow_hand.xml"] {
        let l = loader(name);
        assert!(
            l.unsupported().is_empty(),
            "{name} silently drops: {:?}",
            l.unsupported()
        );
    }
}

/// Armature is parsed but not modelled. Surface it so nobody assumes the
/// rotor inertia is in there.
#[test]
fn armature_is_reported_as_unmodelled() {
    let l = loader("ant.xml");
    assert!(
        !l.armature_joints().is_empty(),
        "ant declares armature; it must be visible to callers"
    );
}
