//! Importer coverage for joint types, geometry, and error handling, using
//! small hand-written URDFs where the expected answer is obvious.

use phyz_math::{Mat3, Vec3};
use phyz_model::{Geometry, JointType};
use phyz_rigid::forward_kinematics;
use phyz_urdf::{UrdfError, UrdfOptions, load_str};

fn opts() -> UrdfOptions {
    UrdfOptions::default()
}

fn assert_close(a: Vec3, b: Vec3, tol: f64, what: &str) {
    let d = (a - b).norm();
    assert!(d < tol, "{what}: expected {b:?}, got {a:?} (|Δ| = {d:e})");
}

/// A two-link arm: shoulder 1 m up rotating about Y, elbow 1 m further along X.
const TWO_LINK: &str = r#"
<robot name="two_link">
  <link name="base">
    <inertial><mass value="10"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>
  <link name="upper">
    <inertial>
      <origin xyz="0.5 0 0"/>
      <mass value="2"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.2"/>
    </inertial>
  </link>
  <link name="fore">
    <inertial>
      <origin xyz="0.5 0 0"/>
      <mass value="1"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="upper"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="3"/>
    <dynamics damping="0.7" friction="0.25"/>
  </joint>
  <joint name="elbow" type="continuous">
    <parent link="upper"/><child link="fore"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
"#;

#[test]
fn two_link_arm_kinematics() {
    let r = load_str(TWO_LINK, &opts()).unwrap();
    assert_eq!(r.model.nbodies(), 3);
    assert_eq!(r.model.nv, 2);

    let fore = r.body_index("fore").unwrap();

    // Straight out: fore link origin at (1, 0, 1).
    let state = r.model.default_state();
    let (x, _) = forward_kinematics(&r.model, &state);
    assert_close(x[fore].pos, Vec3::new(1.0, 0.0, 1.0), 1e-12, "zero pose");

    // Shoulder +90° about Y swings +X down to -Z: (0, 0, 0).
    let mut state = r.model.default_state();
    let sh = r.joint_index("shoulder").unwrap();
    state.q[r.model.q_offsets[sh]] = std::f64::consts::FRAC_PI_2;
    let (x, _) = forward_kinematics(&r.model, &state);
    assert_close(x[fore].pos, Vec3::new(0.0, 0.0, 0.0), 1e-12, "shoulder +90");

    // Shoulder -90° lifts it straight up to (0, 0, 2).
    state.q[r.model.q_offsets[sh]] = -std::f64::consts::FRAC_PI_2;
    let (x, _) = forward_kinematics(&r.model, &state);
    assert_close(x[fore].pos, Vec3::new(0.0, 0.0, 2.0), 1e-12, "shoulder -90");
}

#[test]
fn dynamics_and_limits_are_imported() {
    let r = load_str(TWO_LINK, &opts()).unwrap();
    let sh = &r.model.joints[r.joint_index("shoulder").unwrap()];
    assert!((sh.damping - 0.7).abs() < 1e-12);
    assert!((sh.friction_loss - 0.25).abs() < 1e-12);
    assert_eq!(sh.limits, Some([-1.57, 1.57]));
    assert_eq!(sh.effort_limit, Some(50.0));
    assert_eq!(sh.velocity_limit, Some(3.0));

    // `continuous` is unlimited by definition, and carries no <dynamics> here.
    let el = &r.model.joints[r.joint_index("elbow").unwrap()];
    assert_eq!(el.joint_type, JointType::Revolute);
    assert_eq!(el.limits, None, "continuous joints must not gain limits");
    assert_eq!(el.damping, 0.0);
    assert_eq!(el.effort_limit, None, "absent effort is not a zero cap");
}

#[test]
fn geometry_primitives_convert() {
    let xml = r#"
    <robot name="shapes">
      <link name="base">
        <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
        <visual>
          <origin xyz="0 0 0.1"/>
          <geometry><box size="2 4 6"/></geometry>
        </visual>
        <collision>
          <geometry><sphere radius="0.3"/></geometry>
        </collision>
        <collision>
          <origin xyz="1 0 0" rpy="0 1.5707963267948966 0"/>
          <geometry><cylinder radius="0.2" length="0.9"/></geometry>
        </collision>
      </link>
    </robot>"#;

    let r = load_str(xml, &opts()).unwrap();
    let b = &r.model.bodies[0];

    assert_eq!(b.visuals.len(), 1);
    match b.visuals[0].geometry {
        // URDF gives full extents, phyz stores halves.
        Geometry::Box { half_extents } => {
            assert_close(half_extents, Vec3::new(1.0, 2.0, 3.0), 1e-12, "box")
        }
        ref g => panic!("expected box, got {g:?}"),
    }
    assert_close(
        b.visuals[0].origin.pos,
        Vec3::new(0.0, 0.0, 0.1),
        1e-12,
        "visual origin",
    );

    assert_eq!(b.collisions.len(), 2);
    match b.collisions[0].geometry {
        Geometry::Sphere { radius } => assert!((radius - 0.3).abs() < 1e-12),
        ref g => panic!("expected sphere, got {g:?}"),
    }
    match b.collisions[1].geometry {
        Geometry::Cylinder { radius, height } => {
            assert!((radius - 0.2).abs() < 1e-12);
            assert!((height - 0.9).abs() < 1e-12, "URDF length maps to height");
        }
        ref g => panic!("expected cylinder, got {g:?}"),
    }
    // The rotated cylinder's origin round-trips: rot is the world→shape
    // coordinate transform, so its transpose is the shape's orientation.
    let expected = Mat3::rotation_y(std::f64::consts::FRAC_PI_2);
    let err = (b.collisions[1].origin.rot.transpose() - expected)
        .norm_sq()
        .sqrt();
    assert!(err < 1e-12, "cylinder orientation off by {err:e}");

    // Only the centred collision shape is promoted to the contact pipeline's
    // single-geometry slot.
    match b.geometry {
        Some(Geometry::Sphere { radius }) => assert!((radius - 0.3).abs() < 1e-12),
        ref g => panic!("expected the centred sphere to be primary, got {g:?}"),
    }
}

#[test]
fn mesh_references_are_reported_not_faked() {
    let xml = r#"
    <robot name="meshy">
      <link name="base">
        <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
        <collision>
          <origin xyz="0 0 0.5"/>
          <geometry><mesh filename="package://robot/meshes/base.stl" scale="0.5 0.5 2"/></geometry>
        </collision>
      </link>
    </robot>"#;

    let r = load_str(xml, &opts()).unwrap();
    // No invented bounding box.
    assert!(r.model.bodies[0].collisions.is_empty());
    assert!(r.model.bodies[0].geometry.is_none());

    assert_eq!(r.mesh_refs.len(), 1);
    let m = &r.mesh_refs[0];
    assert_eq!(m.filename, "package://robot/meshes/base.stl");
    assert!(!m.visual);
    assert_close(m.scale.unwrap(), Vec3::new(0.5, 0.5, 2.0), 1e-12, "scale");
    assert_close(m.origin.pos, Vec3::new(0.0, 0.0, 0.5), 1e-12, "mesh origin");
    assert!(
        r.warnings.iter().any(|w| w.contains("base.stl")),
        "a skipped mesh must warn, not vanish: {:?}",
        r.warnings
    );
}

#[test]
fn rotated_inertial_frame_rotates_the_tensor() {
    // A tensor that is diag(1, 2, 3) in a frame yawed 90° about Z becomes
    // diag(2, 1, 3) in the link frame.
    let xml = r#"
    <robot name="inertia">
      <link name="base">
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 1.5707963267948966"/>
          <mass value="4"/>
          <inertia ixx="1" ixy="0" ixz="0" iyy="2" iyz="0" izz="3"/>
        </inertial>
      </link>
    </robot>"#;

    let r = load_str(xml, &opts()).unwrap();
    let i = &r.model.bodies[0].inertia;
    assert!((i.mass - 4.0).abs() < 1e-12);
    assert!(
        (i.inertia[(0, 0)] - 2.0).abs() < 1e-12,
        "{}",
        i.inertia[(0, 0)]
    );
    assert!(
        (i.inertia[(1, 1)] - 1.0).abs() < 1e-12,
        "{}",
        i.inertia[(1, 1)]
    );
    assert!(
        (i.inertia[(2, 2)] - 3.0).abs() < 1e-12,
        "{}",
        i.inertia[(2, 2)]
    );
    assert!(i.inertia[(0, 1)].abs() < 1e-12, "should stay diagonal");
}

#[test]
fn floating_joint_becomes_a_free_joint() {
    let xml = r#"
    <robot name="drone">
      <link name="world_link">
        <inertial><mass value="0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>
      </link>
      <link name="body">
        <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
      </link>
      <joint name="float" type="floating">
        <parent link="world_link"/><child link="body"/>
        <origin xyz="0 0 2"/>
      </joint>
    </robot>"#;

    let r = load_str(xml, &opts()).unwrap();
    let j = &r.model.joints[r.joint_index("float").unwrap()];
    assert_eq!(j.joint_type, JointType::Free);
    assert_eq!(j.ndof(), 6);
    assert_eq!(r.model.nv, 6);

    // Translating the free joint moves the body, on top of its 2 m origin.
    let mut state = r.model.default_state();
    let off = r.model.q_offsets[r.joint_index("float").unwrap()];
    state.q[off] = 1.0; // +X
    let (x, _) = forward_kinematics(&r.model, &state);
    let body = r.body_index("body").unwrap();
    assert_close(x[body].pos, Vec3::new(1.0, 0.0, 2.0), 1e-12, "free joint");
}

#[test]
fn planar_joint_expands_to_three_dofs() {
    // Planar about +Z: two translations in the XY plane plus a yaw.
    let xml = r#"
    <robot name="slider">
      <link name="ground">
        <inertial><mass value="0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>
      </link>
      <link name="puck">
        <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
      </link>
      <joint name="slide" type="planar">
        <parent link="ground"/><child link="puck"/>
        <origin xyz="0 0 0.05"/>
        <axis xyz="0 0 1"/>
      </joint>
    </robot>"#;

    let r = load_str(xml, &opts()).unwrap();
    assert_eq!(r.model.nv, 3, "planar joints contribute exactly 3 DOF");
    // Two massless helper bodies were inserted, and the user is told so.
    assert_eq!(r.model.nbodies(), 4);
    assert!(
        r.warnings.iter().any(|w| w.contains("planar")),
        "{:?}",
        r.warnings
    );

    let puck = r.body_index("puck").unwrap();
    let rot = &r.model.joints[r.joint_index("slide").unwrap()];
    assert_eq!(rot.joint_type, JointType::Revolute);
    assert_close(rot.axis, Vec3::new(0.0, 0.0, 1.0), 1e-12, "planar normal");

    // Drive both prismatic DOFs and confirm the puck stays in its plane and
    // moves the expected total distance.
    let mut state = r.model.default_state();
    state.q[0] = 0.3;
    state.q[1] = -0.4;
    let (x, _) = forward_kinematics(&r.model, &state);
    assert!(
        (x[puck].pos.z - 0.05).abs() < 1e-12,
        "planar motion must not leave the plane: z = {}",
        x[puck].pos.z
    );
    let in_plane = (x[puck].pos.x.powi(2) + x[puck].pos.y.powi(2)).sqrt();
    assert!((in_plane - 0.5).abs() < 1e-12, "travelled {in_plane}");
}

#[test]
fn multiple_roots_are_rejected() {
    let xml = r#"
    <robot name="two_trees">
      <link name="a"/>
      <link name="b"/>
    </robot>"#;
    match load_str(xml, &opts()) {
        Err(UrdfError::MultipleRoots(n, names)) => {
            assert_eq!(n, 2);
            assert!(names.contains('a') && names.contains('b'));
        }
        other => panic!("expected MultipleRoots, got {other:?}"),
    }
}

#[test]
fn unknown_link_is_rejected() {
    let xml = r#"
    <robot name="broken">
      <link name="a"/>
      <link name="b"/>
      <joint name="j" type="fixed">
        <parent link="a"/><child link="b"/>
      </joint>
      <joint name="k" type="fixed">
        <parent link="a"/><child link="ghost"/>
      </joint>
    </robot>"#;
    match load_str(xml, &opts()) {
        Err(UrdfError::UnknownLink { joint, link }) => {
            assert_eq!(joint, "k");
            assert_eq!(link, "ghost");
        }
        other => panic!("expected UnknownLink, got {other:?}"),
    }
}

#[test]
fn a_link_with_two_parents_is_rejected() {
    let xml = r#"
    <robot name="loop">
      <link name="a"/>
      <link name="b"/>
      <link name="c"/>
      <joint name="j1" type="fixed"><parent link="a"/><child link="c"/></joint>
      <joint name="j2" type="fixed"><parent link="b"/><child link="c"/></joint>
    </robot>"#;
    assert!(matches!(
        load_str(xml, &opts()),
        Err(UrdfError::DuplicateChild { .. })
    ));
}

#[test]
fn degenerate_axis_is_rejected() {
    let xml = r#"
    <robot name="bad_axis">
      <link name="a"/>
      <link name="b"/>
      <joint name="spin" type="revolute">
        <parent link="a"/><child link="b"/>
        <axis xyz="0 0 0"/>
        <limit lower="-1" upper="1" effort="1" velocity="1"/>
      </joint>
    </robot>"#;
    match load_str(xml, &opts()) {
        Err(UrdfError::DegenerateAxis { joint }) => assert_eq!(joint, "spin"),
        other => panic!("expected DegenerateAxis, got {other:?}"),
    }
}

#[test]
fn joint_axes_are_normalized() {
    let xml = r#"
    <robot name="unnormalized">
      <link name="a"/>
      <link name="b"/>
      <joint name="spin" type="continuous">
        <parent link="a"/><child link="b"/>
        <axis xyz="0 0 5"/>
      </joint>
    </robot>"#;
    let r = load_str(xml, &opts()).unwrap();
    let j = &r.model.joints[r.joint_index("spin").unwrap()];
    assert_close(j.axis, Vec3::new(0.0, 0.0, 1.0), 1e-12, "normalized axis");
}

#[test]
fn empty_robot_is_rejected() {
    assert!(matches!(
        load_str(r#"<robot name="void"></robot>"#, &opts()),
        Err(UrdfError::NoLinks { .. })
    ));
}

#[test]
fn malformed_xml_is_rejected() {
    assert!(matches!(
        load_str("<robot name=\"oops\"><link", &opts()),
        Err(UrdfError::Parse(_))
    ));
}

#[test]
fn mimic_joints_warn() {
    let xml = r#"
    <robot name="gripper">
      <link name="palm"/>
      <link name="f1"/>
      <link name="f2"/>
      <joint name="drive" type="prismatic">
        <parent link="palm"/><child link="f1"/>
        <axis xyz="0 1 0"/>
        <limit lower="0" upper="0.04" effort="10" velocity="1"/>
      </joint>
      <joint name="follow" type="prismatic">
        <parent link="palm"/><child link="f2"/>
        <axis xyz="0 -1 0"/>
        <limit lower="0" upper="0.04" effort="10" velocity="1"/>
        <mimic joint="drive" multiplier="1"/>
      </joint>
    </robot>"#;

    let r = load_str(xml, &opts()).unwrap();
    // Imported as an independent DOF rather than dropped.
    assert_eq!(r.model.nv, 2);
    assert!(
        r.warnings.iter().any(|w| w.contains("mimic")),
        "{:?}",
        r.warnings
    );
}

#[test]
fn options_override_world_settings() {
    let r = load_str(
        TWO_LINK,
        &UrdfOptions {
            dt: Some(0.004),
            gravity: Some(Vec3::new(0.0, 0.0, -1.62)),
            ..Default::default()
        },
    )
    .unwrap();
    assert!((r.model.dt - 0.004).abs() < 1e-12);
    assert!((r.model.gravity.z + 1.62).abs() < 1e-12);
}

#[test]
fn a_missing_limit_element_does_not_weld_the_joint_shut() {
    // `urdf-rs` defaults an absent <limit> to lower = upper = 0. Passing that
    // through as a real limit would lock the joint the moment the solver
    // applies its limit force, so it must import as unlimited instead.
    let xml = r#"
    <robot name="no_limit">
      <link name="a"/>
      <link name="b"/>
      <joint name="spin" type="revolute">
        <parent link="a"/><child link="b"/>
        <axis xyz="0 0 1"/>
      </joint>
    </robot>"#;

    let r = load_str(xml, &opts()).unwrap();
    let j = &r.model.joints[r.joint_index("spin").unwrap()];
    assert_eq!(j.limits, None);
    // And the solver agrees there is nothing to push against.
    assert_eq!(j.limit_force(1.0, 0.0), 0.0);
}
