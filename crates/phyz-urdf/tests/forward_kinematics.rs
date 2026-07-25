//! Forward-kinematics validation for the URDF importer.
//!
//! The importer's job is to translate URDF's pose/axis conventions into phyz's
//! Plücker-transform conventions. A structural test (right number of bodies,
//! right parents) would pass even with a transposed rotation or an axis in the
//! wrong frame, so the checks here compare phyz's Featherstone forward
//! kinematics against two independent references:
//!
//! 1. A plain 4×4 homogeneous-transform walk of the URDF, written here from the
//!    `urdf_rs` structs and sharing no code with the importer.
//! 2. Hard-coded link poses for the Franka Panda that are documented in
//!    Franka's own specification.

use phyz_math::{Mat3, Vec3};
use phyz_model::JointType;
use phyz_rigid::forward_kinematics;
use phyz_urdf::{UrdfModel, UrdfOptions};
use std::collections::HashMap;

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

fn load(name: &str) -> UrdfModel {
    phyz_urdf::load_file(format!("{DATA}/{name}"), &UrdfOptions::default())
        .unwrap_or_else(|e| panic!("failed to import {name}: {e}"))
}

// ── Independent reference FK ────────────────────────────────────────────────

/// A rigid pose: `p` is the frame origin and `r` maps local → parent.
#[derive(Clone, Copy)]
struct Pose {
    r: Mat3,
    p: Vec3,
}

impl Pose {
    fn identity() -> Self {
        Self {
            r: Mat3::identity(),
            p: Vec3::zeros(),
        }
    }

    /// `self` then `rhs`, i.e. the usual 4×4 product.
    fn then(&self, rhs: &Pose) -> Pose {
        Pose {
            r: self.r.mul_mat(&rhs.r),
            p: self.p + self.r.mul_vec(rhs.p),
        }
    }
}

fn rpy(v: [f64; 3]) -> Mat3 {
    Mat3::rotation_z(v[2])
        .mul_mat(&Mat3::rotation_y(v[1]))
        .mul_mat(&Mat3::rotation_x(v[0]))
}

/// Walk the URDF with ordinary homogeneous transforms: link poses in the root
/// frame, keyed by link name.
fn reference_fk(robot: &urdf_rs::Robot, q: &HashMap<&str, f64>) -> HashMap<String, Pose> {
    let mut poses: HashMap<String, Pose> = HashMap::new();

    let is_child: HashMap<&str, &urdf_rs::Joint> = robot
        .joints
        .iter()
        .map(|j| (j.child.link.as_str(), j))
        .collect();
    let root = robot
        .links
        .iter()
        .find(|l| !is_child.contains_key(l.name.as_str()))
        .expect("single root");
    poses.insert(root.name.clone(), Pose::identity());

    // Repeatedly place any joint whose parent is already placed.
    let mut remaining: Vec<&urdf_rs::Joint> = robot.joints.iter().collect();
    while !remaining.is_empty() {
        let before = remaining.len();
        remaining.retain(|j| {
            let Some(parent) = poses.get(&j.parent.link).copied() else {
                return true;
            };
            let origin = Pose {
                r: rpy(j.origin.rpy.0),
                p: Vec3::new(j.origin.xyz[0], j.origin.xyz[1], j.origin.xyz[2]),
            };
            let axis = Vec3::new(j.axis.xyz[0], j.axis.xyz[1], j.axis.xyz[2]);
            let qi = q.get(j.name.as_str()).copied().unwrap_or(0.0);

            let motion = match &j.joint_type {
                urdf_rs::JointType::Revolute | urdf_rs::JointType::Continuous => Pose {
                    r: Mat3::rotation_axis(axis.normalize(), qi),
                    p: Vec3::zeros(),
                },
                urdf_rs::JointType::Prismatic => Pose {
                    r: Mat3::identity(),
                    p: axis.normalize() * qi,
                },
                urdf_rs::JointType::Fixed => Pose::identity(),
                other => panic!("reference FK does not handle {other:?}"),
            };

            poses.insert(j.child.link.clone(), parent.then(&origin).then(&motion));
            false
        });
        assert!(
            remaining.len() < before,
            "URDF joint graph is not a tree rooted at {}",
            root.name
        );
    }
    poses
}

/// Set `state.q` for a named joint.
fn set_q(robot: &UrdfModel, state: &mut phyz_model::State, joint_name: &str, value: f64) {
    let ji = robot
        .joint_index(joint_name)
        .unwrap_or_else(|| panic!("no joint `{joint_name}`"));
    state.q[robot.model.q_offsets[ji]] = value;
}

fn assert_close(a: Vec3, b: Vec3, tol: f64, what: &str) {
    let d = (a - b).norm();
    assert!(d < tol, "{what}: expected {b:?}, got {a:?} (|Δ| = {d:e})");
}

/// Compare phyz FK against the reference walk for a given configuration.
fn check_against_reference(file: &str, q: &[(&str, f64)]) {
    let robot = load(file);
    let raw = urdf_rs::read_file(format!("{DATA}/{file}")).unwrap();

    let mut state = robot.model.default_state();
    let mut qmap = HashMap::new();
    for (name, value) in q {
        set_q(&robot, &mut state, name, *value);
        qmap.insert(*name, *value);
    }

    let (xforms, _) = forward_kinematics(&robot.model, &state);
    let reference = reference_fk(&raw, &qmap);

    for (i, body) in robot.model.bodies.iter().enumerate() {
        let Some(expected) = reference.get(&body.name) else {
            continue; // helper bodies inserted for planar joints
        };
        // `forward_kinematics` returns world→body Plücker transforms: `pos` is
        // the body origin in world coordinates and `rot` maps world→body, so
        // the body's orientation in world is its transpose.
        assert_close(
            xforms[i].pos,
            expected.p,
            1e-9,
            &format!("{file}: position of `{}`", body.name),
        );
        let r_world = xforms[i].rot.transpose();
        let err = (r_world - expected.r).norm_sq().sqrt();
        assert!(
            err < 1e-9,
            "{file}: orientation of `{}` differs by {err:e}",
            body.name
        );
    }
}

// ── Panda ───────────────────────────────────────────────────────────────────

#[test]
fn panda_imports_expected_structure() {
    let robot = load("panda.urdf");
    let m = &robot.model;

    assert_eq!(robot.robot_name, "panda");
    // 12 links in the chain plus panda_link0; every link becomes one body
    // because the file has no planar joints.
    assert_eq!(m.nbodies(), 13, "one body per URDF link");

    // 7 revolute arm joints + 2 prismatic finger joints = 9 DOF; the base weld
    // and the two fixed joints contribute nothing.
    assert_eq!(m.nv, 9);
    assert_eq!(m.nq, 9);

    for j in 1..=7 {
        let ji = m.joint_index(&format!("panda_joint{j}")).unwrap();
        assert_eq!(m.joints[ji].joint_type, JointType::Revolute);
        // Every Panda joint rotates about its own local Z.
        assert_close(
            m.joints[ji].axis,
            Vec3::new(0.0, 0.0, 1.0),
            1e-12,
            "panda joint axis",
        );
    }
    let f1 = m.joint_index("panda_finger_joint1").unwrap();
    assert_eq!(m.joints[f1].joint_type, JointType::Prismatic);
    assert_close(
        m.joints[f1].axis,
        Vec3::new(0.0, 1.0, 0.0),
        1e-12,
        "finger axis",
    );

    // Fixed joints, including the implicit weld of the root to the world.
    let base = m.joint_index("panda_link0_base").unwrap();
    assert_eq!(m.joints[base].joint_type, JointType::Fixed);
    assert_eq!(m.bodies[m.body_index("panda_link0").unwrap()].parent, -1);
    assert_eq!(
        m.joints[m.joint_index("panda_joint8").unwrap()].joint_type,
        JointType::Fixed
    );

    // Parents precede children, which the Featherstone passes require.
    for (i, b) in m.bodies.iter().enumerate() {
        assert!(b.parent < i as i32, "body {i} has parent {}", b.parent);
    }
}

#[test]
fn panda_limits_and_dynamics() {
    let robot = load("panda.urdf");
    let m = &robot.model;

    let j1 = &m.joints[m.joint_index("panda_joint1").unwrap()];
    let [lo, hi] = j1.limits.expect("revolute joints carry limits");
    assert!((lo - -2.9671).abs() < 1e-9, "lower {lo}");
    assert!((hi - 2.9671).abs() < 1e-9, "upper {hi}");
    assert_eq!(j1.effort_limit, Some(87.0));
    assert!((j1.velocity_limit.unwrap() - 2.175).abs() < 1e-9);

    // Wrist joints are the weaker ones.
    let j5 = &m.joints[m.joint_index("panda_joint5").unwrap()];
    assert_eq!(j5.effort_limit, Some(12.0));

    let f = &m.joints[m.joint_index("panda_finger_joint1").unwrap()];
    assert_eq!(f.limits, Some([0.0, 0.04]));

    // Fixed joints have no limits.
    assert_eq!(
        m.joints[m.joint_index("panda_joint8").unwrap()].limits,
        None
    );
}

#[test]
fn panda_inertials() {
    let robot = load("panda.urdf");
    let m = &robot.model;

    let l0 = &m.bodies[m.body_index("panda_link0").unwrap()].inertia;
    assert!((l0.mass - 2.9).abs() < 1e-12);
    assert_close(l0.com, Vec3::new(0.0, 0.0, 0.05), 1e-12, "link0 com");
    assert!((l0.inertia[(0, 0)] - 0.1).abs() < 1e-12);
    assert!((l0.inertia[(0, 1)]).abs() < 1e-12, "off-diagonals are zero");

    // Total mass of the imported robot should equal the sum in the file.
    let total: f64 = m.bodies.iter().map(|b| b.inertia.mass).sum();
    let raw = urdf_rs::read_file(format!("{DATA}/panda.urdf")).unwrap();
    let expected: f64 = raw.links.iter().map(|l| l.inertial.mass.value).sum();
    assert!((total - expected).abs() < 1e-12, "{total} vs {expected}");
}

#[test]
fn panda_flange_pose_at_zero_matches_spec() {
    // Franka publishes the Panda's flange (`panda_link8`) as sitting at
    // (0.088, 0, 0.926) in the base frame with all joints at zero, rotated 180°
    // about X relative to the base. This is the single best check that the
    // rpy → rotation-matrix → Plücker-transpose chain is right: any sign or
    // transpose error moves this point.
    let robot = load("panda.urdf");
    let state = robot.model.default_state();
    let (xforms, _) = forward_kinematics(&robot.model, &state);

    let link8 = robot.body_index("panda_link8").unwrap();
    assert_close(
        xforms[link8].pos,
        Vec3::new(0.088, 0.0, 0.926),
        1e-9,
        "panda_link8 at q = 0",
    );

    let r_world = xforms[link8].rot.transpose();
    let expected = Mat3::rotation_x(std::f64::consts::PI);
    let err = (r_world - expected).norm_sq().sqrt();
    assert!(err < 1e-9, "panda_link8 orientation off by {err:e}");

    // Intermediate landmarks along the arm, from the joint origins in the URDF.
    let l1 = robot.body_index("panda_link1").unwrap();
    assert_close(
        xforms[l1].pos,
        Vec3::new(0.0, 0.0, 0.333),
        1e-12,
        "panda_link1 at q = 0",
    );
    let l5 = robot.body_index("panda_link5").unwrap();
    assert_close(
        xforms[l5].pos,
        Vec3::new(0.0, 0.0, 1.033),
        1e-9,
        "panda_link5 at q = 0",
    );
}

#[test]
fn panda_fk_matches_reference_in_many_configurations() {
    // Zero, one joint at a time, and a general pose with every DOF nonzero.
    check_against_reference("panda.urdf", &[]);

    for (i, angle) in [0.3, -0.7, 1.1, -1.9, 0.55, 2.2, -0.9].iter().enumerate() {
        check_against_reference("panda.urdf", &[(&format!("panda_joint{}", i + 1), *angle)]);
    }

    check_against_reference(
        "panda.urdf",
        &[
            ("panda_joint1", 0.4),
            ("panda_joint2", -0.6),
            ("panda_joint3", 1.2),
            ("panda_joint4", -2.0),
            ("panda_joint5", 0.9),
            ("panda_joint6", 1.8),
            ("panda_joint7", -1.1),
            ("panda_finger_joint1", 0.03),
            ("panda_finger_joint2", 0.02),
        ],
    );
}

#[test]
fn panda_prismatic_fingers_move_along_their_axis() {
    // The fingers translate along local Y of `panda_hand`, and `panda_hand` is
    // yawed -45° from `panda_link8`, so opening the gripper must move the
    // finger perpendicular to the hand's approach direction, not along Z.
    let robot = load("panda.urdf");
    let finger = robot.body_index("panda_leftfinger").unwrap();

    let closed = robot.model.default_state();
    let (x0, _) = forward_kinematics(&robot.model, &closed);

    let mut open = robot.model.default_state();
    set_q(&robot, &mut open, "panda_finger_joint1", 0.04);
    let (x1, _) = forward_kinematics(&robot.model, &open);

    let delta = x1[finger].pos - x0[finger].pos;
    assert!(
        (delta.norm() - 0.04).abs() < 1e-9,
        "finger should travel exactly 0.04 m, got {}",
        delta.norm()
    );
    // Orientation must be unchanged by a prismatic joint.
    let err = (x1[finger].rot - x0[finger].rot).norm_sq().sqrt();
    assert!(err < 1e-12, "prismatic joint rotated the finger by {err:e}");
}

// ── KUKA iiwa ───────────────────────────────────────────────────────────────

#[test]
fn iiwa_imports_and_matches_reference() {
    let robot = load("kuka_iiwa.urdf");
    assert_eq!(robot.model.nv, 7, "iiwa is a 7-DOF arm");

    check_against_reference("kuka_iiwa.urdf", &[]);
    check_against_reference(
        "kuka_iiwa.urdf",
        &[
            ("lbr_iiwa_joint_1", 0.5),
            ("lbr_iiwa_joint_2", -0.8),
            ("lbr_iiwa_joint_3", 1.3),
            ("lbr_iiwa_joint_4", -1.5),
            ("lbr_iiwa_joint_5", 0.7),
            ("lbr_iiwa_joint_6", 1.1),
            ("lbr_iiwa_joint_7", -0.4),
        ],
    );
}

#[test]
fn iiwa_alternating_axes_are_preserved() {
    // The iiwa alternates Z / Y joint axes; if the importer rotated axes into
    // the wrong frame this pattern would be destroyed.
    let robot = load("kuka_iiwa.urdf");
    let raw = urdf_rs::read_file(format!("{DATA}/kuka_iiwa.urdf")).unwrap();

    for j in &raw.joints {
        if j.joint_type != urdf_rs::JointType::Revolute {
            continue;
        }
        let ji = robot.joint_index(&j.name).unwrap();
        let expected = Vec3::new(j.axis.xyz[0], j.axis.xyz[1], j.axis.xyz[2]).normalize();
        assert_close(
            robot.model.joints[ji].axis,
            expected,
            1e-12,
            &format!("axis of `{}`", j.name),
        );
    }
}

// ── Floating base ───────────────────────────────────────────────────────────

#[test]
fn floating_base_adds_six_dofs() {
    let fixed = load("panda.urdf");
    let floating = phyz_urdf::load_file(
        format!("{DATA}/panda.urdf"),
        &UrdfOptions {
            base: phyz_urdf::BaseKind::Floating,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(floating.model.nv, fixed.model.nv + 6);
    let base = floating.model.joint_index("panda_link0_base").unwrap();
    assert_eq!(floating.model.joints[base].joint_type, JointType::Free);

    // With the free joint at zero the robot must be exactly where the fixed
    // base put it.
    let state = floating.model.default_state();
    let (xf, _) = forward_kinematics(&floating.model, &state);
    let link8 = floating.body_index("panda_link8").unwrap();
    assert_close(
        xf[link8].pos,
        Vec3::new(0.088, 0.0, 0.926),
        1e-9,
        "floating base at identity",
    );
}
