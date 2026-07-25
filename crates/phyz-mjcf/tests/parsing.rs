//! Integration tests for the MJCF parser.

use phyz_mjcf::{MjcfError, MjcfLoader};
use phyz_model::{ActuatorType, Geometry, JointType};

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn load(name: &str) -> MjcfLoader {
    MjcfLoader::from_file(fixture(name)).unwrap_or_else(|e| panic!("failed to load {name}: {e}"))
}

// ---------------------------------------------------------------------------
// Baseline behaviour (previously the parser's inline unit tests)
// ---------------------------------------------------------------------------

#[test]
fn simple_model_builds() {
    let mjcf = r#"
    <mujoco>
        <option gravity="0 0 -9.81" timestep="0.001"/>
        <worldbody>
            <body name="link1" pos="0 0 0">
                <inertial pos="0 0 0" mass="1.0" diaginertia="0.1 0.1 0.1"/>
                <joint name="joint1" type="hinge" axis="0 0 1"/>
            </body>
        </worldbody>
    </mujoco>"#;
    let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
    assert_eq!(model.nbodies(), 1);
    assert_eq!(model.nv, 1);
    assert!((model.dt - 0.001).abs() < 1e-12);
}

#[test]
fn nested_bodies_chain() {
    let mjcf = r#"
    <mujoco>
        <worldbody>
            <body name="link1">
                <inertial mass="1.0" diaginertia="0.1 0.1 0.1"/>
                <joint type="hinge" axis="0 0 1"/>
                <body name="link2" pos="1 0 0">
                    <inertial mass="0.5" diaginertia="0.05 0.05 0.05"/>
                    <joint type="slide" axis="1 0 0"/>
                </body>
            </body>
        </worldbody>
    </mujoco>"#;
    let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
    assert_eq!(model.nbodies(), 2);
    assert_eq!(model.nv, 2);
}

#[test]
fn ant_model_still_parses() {
    let model = MjcfLoader::from_file("../../models/ant.xml")
        .unwrap()
        .build_model();
    assert_eq!(model.nbodies(), 9);
    assert_eq!(model.nv, 14); // free joint (6) + 8 hinges
    assert!((model.gravity.z + 9.81).abs() < 1e-10);
}

#[test]
fn simple_arm_model_still_parses() {
    let model = MjcfLoader::from_file("../../models/simple_arm.xml")
        .unwrap()
        .build_model();
    assert!(model.nbodies() > 0);
}

#[test]
fn degree_ranges_convert_to_radians() {
    let mjcf = r#"
    <mujoco>
        <compiler angle="degree"/>
        <worldbody>
            <body name="link1">
                <inertial mass="1.0" diaginertia="0.1 0.1 0.1"/>
                <joint name="j1" type="hinge" axis="0 0 1" range="-90 90"/>
            </body>
        </worldbody>
    </mujoco>"#;
    let loader = MjcfLoader::from_xml_str(mjcf).unwrap();
    assert!(loader.angle_in_degrees());
    let range = loader.build_model().joints[0].limits.unwrap();
    assert!((range[0] + std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    assert!((range[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
}

#[test]
fn slide_joint_range_stays_a_length_under_degree_mode() {
    // A slide joint's range is metres even when angle="degree"; converting it
    // would shrink the travel by a factor of ~57.
    let mjcf = r#"
    <mujoco>
        <compiler angle="degree"/>
        <worldbody>
            <body name="slider">
                <inertial mass="1.0" diaginertia="0.1 0.1 0.1"/>
                <joint name="s1" type="slide" axis="1 0 0" range="-0.5 0.5"/>
            </body>
        </worldbody>
    </mujoco>"#;
    let range = MjcfLoader::from_xml_str(mjcf).unwrap().build_model().joints[0]
        .limits
        .unwrap();
    assert!((range[0] + 0.5).abs() < 1e-12, "{range:?}");
}

// ---------------------------------------------------------------------------
// 1. <default> classes
// ---------------------------------------------------------------------------

#[test]
fn root_class_defaults_apply_to_unclassed_elements() {
    let model = load("defaults.xml").build_model();
    let j = model
        .joints
        .iter()
        .zip(model.bodies.iter())
        .find(|(_, b)| b.name == "plain")
        .map(|(j, _)| j)
        .expect("plain body");
    assert!((j.damping - 1.0).abs() < 1e-12, "root default damping");
    let range = j.limits.expect("root default range");
    assert!((range[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-9);

    // <geom type="capsule" size="0.02 0.1"/> from the root class.
    let geom = model
        .bodies
        .iter()
        .find(|b| b.name == "plain")
        .unwrap()
        .geometry
        .as_ref()
        .expect("plain geom from defaults");
    match geom {
        Geometry::Capsule { radius, length } => {
            assert!((radius - 0.02).abs() < 1e-12);
            // MJCF stores half-length, phyz stores full length.
            assert!((length - 0.2).abs() < 1e-12, "length {length}");
        }
        other => panic!("expected capsule from defaults, got {other:?}"),
    }
}

#[test]
fn explicit_class_overrides_root() {
    let model = load("defaults.xml").build_model();
    let (j, b) = model
        .joints
        .iter()
        .zip(model.bodies.iter())
        .find(|(_, b)| b.name == "explicit")
        .unwrap();
    assert!((j.damping - 7.5).abs() < 1e-12, "class 'heavy' damping");
    match b.geometry.as_ref().unwrap() {
        Geometry::Sphere { radius } => assert!((radius - 0.01).abs() < 1e-12),
        other => panic!("class 'light' should give a sphere, got {other:?}"),
    }
}

#[test]
fn childclass_propagates_to_descendants() {
    let model = load("defaults.xml").build_model();
    for name in ["parent", "child"] {
        let (j, b) = model
            .joints
            .iter()
            .zip(model.bodies.iter())
            .find(|(_, b)| b.name == name)
            .unwrap();
        assert!(
            (j.damping - 7.5).abs() < 1e-12,
            "{name} should inherit childclass 'heavy' damping, got {}",
            j.damping
        );
        match b.geometry.as_ref().unwrap() {
            // 'heavy' overrides size but inherits type=capsule from the root class.
            Geometry::Capsule { radius, length } => {
                assert!((radius - 0.2).abs() < 1e-12, "{name} radius");
                assert!((length - 1.0).abs() < 1e-12, "{name} length");
            }
            other => panic!("{name}: expected capsule, got {other:?}"),
        }
    }
}

#[test]
fn nested_class_inherits_through_the_chain() {
    let model = load("defaults.xml").build_model();
    let (j, _) = model
        .joints
        .iter()
        .zip(model.bodies.iter())
        .find(|(_, b)| b.name == "child")
        .unwrap();
    // 'heavy_limited' overrides range but inherits damping from 'heavy'.
    assert!((j.damping - 7.5).abs() < 1e-12);
    let range = j.limits.unwrap();
    assert!(
        (range[1] - 30f64.to_radians()).abs() < 1e-9,
        "range {range:?}"
    );
}

#[test]
fn actuator_defaults_apply() {
    let model = load("defaults.xml").build_model();
    let by_name = |n: &str| {
        model
            .actuators
            .iter()
            .find(|a| a.name == n)
            .unwrap()
            .clone()
    };
    assert!((by_name("m_default").gear - 50.0).abs() < 1e-12);
    assert_eq!(by_name("m_default").ctrl_range, Some([-1.0, 1.0]));
    assert!((by_name("m_override").gear - 200.0).abs() < 1e-12);
}

#[test]
fn unknown_class_is_rejected() {
    let mjcf = r#"
    <mujoco>
        <worldbody>
            <body name="b">
                <inertial mass="1" diaginertia="0.1 0.1 0.1"/>
                <joint name="j" class="typo"/>
            </body>
        </worldbody>
    </mujoco>"#;
    let err = MjcfLoader::from_xml_str(mjcf).unwrap_err();
    assert!(matches!(err, MjcfError::UnknownClass { .. }), "{err}");
    assert!(err.to_string().contains("typo"), "{err}");
}

// ---------------------------------------------------------------------------
// 2. Non-quaternion orientations
// ---------------------------------------------------------------------------

#[test]
fn all_orientation_forms_agree_where_they_should() {
    let model = load("orientations.xml").build_model();
    let rot_of = |name: &str| {
        let (_, i) = model
            .bodies
            .iter()
            .enumerate()
            .map(|(i, b)| (b, i))
            .find(|(b, _)| b.name == name)
            .unwrap();
        model.joints[i].parent_to_joint.rot
    };

    // quat, euler(0 0 90) and axisangle(z, 90) all describe the same rotation.
    let q = rot_of("by_quat");
    for other in ["by_euler", "by_axisangle"] {
        let r = rot_of(other);
        assert!(
            (q - r).norm_sq().sqrt() < 1e-9,
            "{other} disagrees with quat"
        );
    }

    // zaxis="1 0 0" maps +Z onto +X.
    let z = rot_of("by_zaxis") * phyz_math::Vec3::new(0.0, 0.0, 1.0);
    assert!(
        (z - phyz_math::Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-9,
        "{z:?}"
    );

    // xyaxes="0 1 0 -1 0 0" is a +90 deg turn about Z, same as by_quat.
    assert!((rot_of("by_xyaxes") - q).norm_sq().sqrt() < 1e-9);
}

#[test]
fn fromto_sets_capsule_length_and_frame() {
    let model = load("orientations.xml").build_model();
    let body = model
        .bodies
        .iter()
        .find(|b| b.name == "capsule_fromto")
        .unwrap();
    let geom = body.collisions.first().expect("fromto capsule");
    match &geom.geometry {
        Geometry::Capsule { radius, length } => {
            assert!((radius - 0.03).abs() < 1e-12);
            // fromto spans 0.4 m, so half-length 0.2 and full length 0.4.
            assert!((length - 0.4).abs() < 1e-12, "length {length}");
        }
        other => panic!("expected capsule, got {other:?}"),
    }
    // fromto="0 0 0  0 0 0.4" puts the capsule centre at z = 0.2, and the geom
    // carries that offset rather than being flattened onto the body frame.
    assert!(
        (geom.origin.pos - phyz_math::Vec3::new(0.0, 0.0, 0.2)).norm() < 1e-12,
        "origin {:?}",
        geom.origin.pos
    );
    assert!(!geom.is_centered());
    // An offset shape is not mirrored into the single-shape `geometry` field.
    assert!(body.geometry.is_none());
}

#[test]
fn every_geom_on_a_body_is_kept_with_its_pose() {
    // phyz_model::Body holds a list of placed shapes, so neither extra geoms
    // nor their body-relative poses are dropped.
    let mjcf = r#"
    <mujoco><worldbody><body name="multi" pos="0 0 1">
        <inertial mass="1" diaginertia="0.1 0.1 0.1"/>
        <joint name="j" type="hinge" axis="0 0 1"/>
        <geom name="centred" type="sphere" size="0.1"/>
        <geom name="offset" type="box" size="0.1 0.1 0.1" pos="0.5 0 0"/>
        <geom name="turned" type="capsule" size="0.02 0.1" euler="0 1.5707963267948966 0"/>
    </body></worldbody></mujoco>"#;
    let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
    let body = &model.bodies[0];

    assert_eq!(body.collisions.len(), 3);
    let named = |n: &str| {
        body.collisions
            .iter()
            .find(|g| g.name.as_deref() == Some(n))
            .unwrap()
    };
    assert!(named("centred").is_centered());
    assert!((named("offset").origin.pos.x - 0.5).abs() < 1e-12);

    // origin.rot is the body -> shape transform, so its transpose is the
    // shape's orientation in the body frame: +90 deg about Y takes z onto x.
    let shape_to_body = named("turned").origin.rot.transpose();
    let z_in_body = shape_to_body * phyz_math::Vec3::new(0.0, 0.0, 1.0);
    assert!(
        (z_in_body - phyz_math::Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-9,
        "{z_in_body:?}"
    );

    // The single-shape field mirrors the first centred geom.
    assert!(matches!(body.geometry, Some(Geometry::Sphere { .. })));
}

#[test]
fn fromto_with_coincident_endpoints_errors() {
    let mjcf = r#"
    <mujoco><worldbody><body name="b">
        <inertial mass="1" diaginertia="0.1 0.1 0.1"/>
        <joint name="j" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="1 1 1 1 1 1" size="0.1"/>
    </body></worldbody></mujoco>"#;
    let err = MjcfLoader::from_xml_str(mjcf).unwrap_err().to_string();
    assert!(err.contains("fromto"), "{err}");
}

// ---------------------------------------------------------------------------
// 3. <freejoint>
// ---------------------------------------------------------------------------

#[test]
fn freejoint_is_a_free_joint() {
    let model = load("freejoint.xml").build_model();
    assert_eq!(model.nbodies(), 2);
    assert_eq!(model.nv, 12); // two 6-DOF free joints
    for j in &model.joints {
        assert_eq!(j.joint_type, JointType::Free);
    }
}

#[test]
fn freejoint_and_joint_type_free_agree() {
    let with_freejoint = r#"
    <mujoco><worldbody><body name="b" pos="0 0 1">
        <freejoint/>
        <inertial mass="1" diaginertia="0.1 0.1 0.1"/>
    </body></worldbody></mujoco>"#;
    let with_joint = r#"
    <mujoco><worldbody><body name="b" pos="0 0 1">
        <joint type="free"/>
        <inertial mass="1" diaginertia="0.1 0.1 0.1"/>
    </body></worldbody></mujoco>"#;
    let a = MjcfLoader::from_xml_str(with_freejoint)
        .unwrap()
        .build_model();
    let b = MjcfLoader::from_xml_str(with_joint).unwrap().build_model();
    assert_eq!(a.nv, b.nv);
    assert_eq!(a.nq, b.nq);
    assert_eq!(a.joints[0].joint_type, b.joints[0].joint_type);
}

// ---------------------------------------------------------------------------
// 4. Non-motor actuators
// ---------------------------------------------------------------------------

#[test]
fn all_actuator_types_parse() {
    let model = load("actuators.xml").build_model();
    assert_eq!(model.actuators.len(), 4);
    let get = |n: &str| model.actuators.iter().find(|a| a.name == n).unwrap();

    let m = get("a_motor");
    assert_eq!(m.actuator_type, ActuatorType::Motor);
    assert!((m.gain - 1.0).abs() < 1e-12);
    assert_eq!(m.bias, [0.0; 3]);
    assert!((m.gear - 100.0).abs() < 1e-12);

    let p = get("a_position");
    assert_eq!(p.actuator_type, ActuatorType::Position);
    assert!((p.gain - 80.0).abs() < 1e-12);
    assert_eq!(p.bias, [0.0, -80.0, -3.0]);
    assert_eq!(p.ctrl_range, Some([-1.5, 1.5]));

    let v = get("a_velocity");
    assert_eq!(v.actuator_type, ActuatorType::Velocity);
    assert!((v.gain - 12.0).abs() < 1e-12);
    assert_eq!(v.bias, [0.0, 0.0, -12.0]);

    let g = get("a_general");
    assert_eq!(g.actuator_type, ActuatorType::General);
    assert!((g.gain - 5.0).abs() < 1e-12);
    assert_eq!(g.bias, [1.0, -2.0, -3.0]);
    assert_eq!(g.force_range, Some([-20.0, 20.0]));
}

#[test]
fn actuator_force_follows_the_affine_model() {
    let model = load("actuators.xml").build_model();
    let get = |n: &str| model.actuators.iter().find(|a| a.name == n).unwrap();

    // A position servo drives towards the setpoint and damps velocity.
    let p = get("a_position");
    // gear defaults to 1: 80*(0.5) + (-80)*0.2 + (-3)*1.0
    let expected = 80.0 * 0.5 - 80.0 * 0.2 - 3.0 * 1.0;
    assert!((p.force(0.5, 0.2, 1.0) - expected).abs() < 1e-9);
    // At the setpoint with zero velocity, no force.
    assert!(p.force(0.2, 0.2, 0.0).abs() < 1e-12);

    // A plain motor is just gear * ctrl, clamped to ctrlrange.
    let m = get("a_motor");
    assert!((m.force(0.5, 3.0, 7.0) - 50.0).abs() < 1e-9);
    assert!(
        (m.force(9.0, 0.0, 0.0) - 100.0).abs() < 1e-9,
        "ctrl clamped"
    );

    // forcerange clamps the output.
    let g = get("a_general");
    assert!((g.force(1000.0, 0.0, 0.0) - 20.0).abs() < 1e-9);
}

#[test]
fn actuator_without_a_transmission_errors() {
    let mjcf = r#"
    <mujoco>
        <worldbody><body name="b">
            <inertial mass="1" diaginertia="0.1 0.1 0.1"/>
            <joint name="j" type="hinge" axis="0 0 1"/>
        </body></worldbody>
        <actuator><motor name="m" gear="1"/></actuator>
    </mujoco>"#;
    let err = MjcfLoader::from_xml_str(mjcf).unwrap_err();
    assert!(matches!(err, MjcfError::MissingAttribute { .. }), "{err}");
    assert!(err.to_string().contains("joint"), "{err}");
}

#[test]
fn tendon_transmission_is_reported_not_silently_dropped() {
    let mjcf = r#"
    <mujoco>
        <worldbody><body name="b">
            <inertial mass="1" diaginertia="0.1 0.1 0.1"/>
            <joint name="j" type="hinge" axis="0 0 1"/>
        </body></worldbody>
        <actuator><motor name="m" tendon="t" gear="1"/></actuator>
    </mujoco>"#;
    let loader = MjcfLoader::from_xml_str(mjcf).unwrap();
    assert!(loader.build_model().actuators.is_empty());
    assert!(
        loader
            .unsupported()
            .iter()
            .any(|u| u.detail.contains("tendon")),
        "{:?}",
        loader.unsupported()
    );
}

// ---------------------------------------------------------------------------
// 5. Assets
// ---------------------------------------------------------------------------

#[test]
fn assets_are_parsed_and_meshes_loaded() {
    let loader = load("assets.xml");
    assert_eq!(loader.meshes().len(), 2);
    assert_eq!(loader.textures().len(), 1);
    assert_eq!(loader.materials().len(), 1);
    assert_eq!(loader.hfields().len(), 1);

    let tetra = loader.meshes().iter().find(|m| m.name == "tetra").unwrap();
    let data = tetra.data.as_ref().expect("tetra.obj should load");
    assert_eq!(data.vertices.len(), 4);
    assert_eq!(data.faces.len(), 4);

    // A referenced but absent mesh is recorded, not fatal.
    let missing = loader
        .meshes()
        .iter()
        .find(|m| m.name == "missing")
        .unwrap();
    assert!(missing.data.is_none());
    assert!(missing.load_error.is_some());
}

#[test]
fn mesh_geom_becomes_mesh_geometry() {
    let model = load("assets.xml").build_model();
    let body = model.bodies.iter().find(|b| b.name == "mesh_body").unwrap();
    match body.geometry.as_ref().expect("mesh geometry") {
        Geometry::Mesh { vertices, faces } => {
            assert_eq!(vertices.len(), 4);
            assert_eq!(faces.len(), 4);
        }
        other => panic!("expected mesh, got {other:?}"),
    }
}

#[test]
fn heightfield_gap_is_reported() {
    let loader = load("assets.xml");
    let note = loader
        .unsupported()
        .iter()
        .find(|u| u.element == "hfield")
        .expect("hfield should be reported as unsupported");
    assert!(note.detail.contains("heightfield"), "{}", note.detail);
}

// ---------------------------------------------------------------------------
// 6. <include>
// ---------------------------------------------------------------------------

#[test]
fn include_splices_bodies_and_defaults() {
    let model = load("include_main.xml").build_model();
    assert_eq!(model.nbodies(), 2);
    assert!(model.bodies.iter().any(|b| b.name == "included_child"));
    // The default class came from a different file entirely.
    for j in &model.joints {
        assert!((j.damping - 3.25).abs() < 1e-12, "damping {}", j.damping);
    }
    let child = model
        .bodies
        .iter()
        .find(|b| b.name == "included_child")
        .unwrap();
    assert!(matches!(
        child.geometry.as_ref().unwrap(),
        Geometry::Box { .. }
    ));
}

#[test]
fn cyclic_include_is_bounded_and_reported() {
    let err = MjcfLoader::from_file(fixture("include_cycle_a.xml")).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("include"), "{msg}");
    assert!(msg.contains("cyclic") || msg.contains("nesting"), "{msg}");
}

#[test]
fn missing_include_file_names_the_file() {
    let mjcf = r#"<mujoco><include file="does_not_exist.xml"/></mujoco>"#;
    let err = MjcfLoader::from_xml_str(mjcf).unwrap_err().to_string();
    assert!(err.contains("does_not_exist.xml"), "{err}");
}

// ---------------------------------------------------------------------------
// Error handling: malformed input must not panic
// ---------------------------------------------------------------------------

/// Each of these used to hit an `unwrap`/`unwrap_or` and either panic or silently
/// substitute a wrong value.
#[test]
fn malformed_attributes_produce_errors_not_panics() {
    let cases: &[(&str, &str)] = &[
        (
            "non-numeric gravity",
            r#"<mujoco><option gravity="0 0 down"/></mujoco>"#,
        ),
        (
            "non-numeric timestep",
            r#"<mujoco><option timestep="fast"/></mujoco>"#,
        ),
        (
            "short body pos",
            r#"<mujoco><worldbody><body name="b" pos="1 2"/></worldbody></mujoco>"#,
        ),
        (
            "bad body quat",
            r#"<mujoco><worldbody><body name="b" quat="1 0 0"/></worldbody></mujoco>"#,
        ),
        (
            "bad joint axis",
            r#"<mujoco><worldbody><body name="b"><joint axis="x y z"/></body></worldbody></mujoco>"#,
        ),
        (
            "zero joint axis",
            r#"<mujoco><worldbody><body name="b"><joint axis="0 0 0"/></body></worldbody></mujoco>"#,
        ),
        (
            "inverted joint range",
            r#"<mujoco><worldbody><body name="b"><joint range="1 -1"/></body></worldbody></mujoco>"#,
        ),
        (
            "bad mass",
            r#"<mujoco><worldbody><body name="b"><inertial mass="heavy"/></body></worldbody></mujoco>"#,
        ),
        (
            "negative mass",
            r#"<mujoco><worldbody><body name="b"><inertial mass="-1"/></body></worldbody></mujoco>"#,
        ),
        (
            "bad geom size",
            r#"<mujoco><worldbody><body name="b"><geom type="sphere" size="big"/></body></worldbody></mujoco>"#,
        ),
        (
            "bad ctrlrange",
            r#"<mujoco><worldbody><body name="b"><joint name="j"/></body></worldbody>
               <actuator><motor joint="j" ctrlrange="a b"/></actuator></mujoco>"#,
        ),
        (
            "bad compiler angle",
            r#"<mujoco><compiler angle="gradians"/></mujoco>"#,
        ),
        (
            "bad eulerseq",
            r#"<mujoco><compiler eulerseq="xyzw"/></mujoco>"#,
        ),
        (
            "bad limited flag",
            r#"<mujoco><worldbody><body name="b"><joint range="-1 1" limited="maybe"/></body></worldbody></mujoco>"#,
        ),
        (
            "unclosed tag",
            r#"<mujoco><worldbody><body name="b"></worldbody></mujoco>"#,
        ),
    ];

    for (label, xml) in cases {
        let result = MjcfLoader::from_xml_str(xml);
        assert!(result.is_err(), "{label}: expected an error, got Ok");
    }
}

#[test]
fn error_messages_name_the_element_and_attribute() {
    let err = MjcfLoader::from_xml_str(
        r#"<mujoco><worldbody><body name="arm" pos="1 2"/></worldbody></mujoco>"#,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("body"), "{err}");
    assert!(err.contains("pos"), "{err}");
    assert!(err.contains("expected 3 numbers"), "{err}");
}

#[test]
fn global_coordinates_are_refused_rather_than_misread() {
    let err = MjcfLoader::from_xml_str(r#"<mujoco><compiler coordinate="global"/></mujoco>"#)
        .unwrap_err();
    assert!(matches!(err, MjcfError::Unsupported(_)), "{err}");
}

// ---------------------------------------------------------------------------
// Compound joints
// ---------------------------------------------------------------------------

#[test]
fn multiple_joints_on_one_body_compose_serially() {
    // A humanoid-style 3-DOF joint written as three hinges on one body.
    let mjcf = r#"
    <mujoco>
        <worldbody>
            <body name="shoulder" pos="0 0 1">
                <inertial mass="2.0" diaginertia="0.1 0.1 0.1"/>
                <joint name="sx" type="hinge" axis="1 0 0"/>
                <joint name="sy" type="hinge" axis="0 1 0"/>
                <joint name="sz" type="hinge" axis="0 0 1"/>
                <body name="forearm" pos="0 0 -0.3">
                    <inertial mass="1.0" diaginertia="0.05 0.05 0.05"/>
                    <joint name="elbow" type="hinge" axis="0 1 0"/>
                </body>
            </body>
        </worldbody>
    </mujoco>"#;
    let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
    assert_eq!(model.nv, 4, "3 shoulder DOFs + 1 elbow");

    // The body's mass must be counted once, not once per joint.
    let total: f64 = model.bodies.iter().map(|b| b.inertia.mass).sum();
    assert!((total - 3.0).abs() < 1e-6, "total mass {total}");

    // The forearm must hang off the last shoulder link, not off the world.
    let forearm = model
        .bodies
        .iter()
        .position(|b| b.name == "forearm")
        .unwrap();
    assert!(model.bodies[forearm].parent >= 0);
}

#[test]
fn compound_joint_model_has_nonsingular_dynamics() {
    let mjcf = r#"
    <mujoco>
        <worldbody>
            <body name="shoulder" pos="0 0 1">
                <inertial mass="2.0" diaginertia="0.1 0.1 0.1"/>
                <joint name="sx" type="hinge" axis="1 0 0"/>
                <joint name="sy" type="hinge" axis="0 1 0"/>
                <body name="forearm" pos="0 0 -0.3">
                    <inertial mass="1.0" diaginertia="0.05 0.05 0.05"/>
                    <joint name="elbow" type="hinge" axis="0 1 0"/>
                </body>
            </body>
        </worldbody>
    </mujoco>"#;
    // Massless intermediate links must not make the articulated-body inertia
    // singular: gravity-driven accelerations should stay finite.
    let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
    let state = model.default_state();
    let qdd = phyz_rigid::aba(&model, &state);
    assert!(
        (0..model.nv).all(|i| qdd[i].is_finite()),
        "ABA produced non-finite accelerations: {qdd:?}"
    );
}

// ---------------------------------------------------------------------------
// Combined shape of a real robot model
// ---------------------------------------------------------------------------

/// A fixture in the shape of a Menagerie quadruped, combining every feature that
/// this parser previously ignored: `<freejoint>`, nested default classes,
/// `childclass`, `fromto` capsules, and `<position>` actuators. Before, this
/// model produced 2 DOFs, no actuators, and default-sized geoms.
#[test]
fn quadruped_shaped_model_is_fully_resolved() {
    let loader = load("quadruped_shape.xml");
    let model = loader.build_model();

    // freejoint contributes 6 DOFs, plus the two leg hinges.
    assert_eq!(model.nv, 8);
    assert_eq!(model.joints[0].joint_type, JointType::Free);

    let joint_of = |name: &str| {
        let i = model.bodies.iter().position(|b| b.name == name).unwrap();
        &model.joints[i]
    };

    // damping comes from class 'robot' via childclass, range from the leaf class.
    let abduct = joint_of("fl_hip");
    assert!((abduct.damping - 0.5).abs() < 1e-12, "{}", abduct.damping);
    assert_eq!(abduct.limits, Some([-0.8, 0.8]));
    assert!((abduct.axis.x - 1.0).abs() < 1e-12, "axis from 'abduction'");

    let knee = joint_of("fl_knee_link");
    assert!((knee.damping - 0.5).abs() < 1e-12);
    assert_eq!(knee.limits, Some([-2.7, -0.9]));
    assert!((knee.axis.y - 1.0).abs() < 1e-12, "axis from 'knee'");

    // The shank capsule gets its radius from the class and its length from fromto.
    let shank = &model
        .bodies
        .iter()
        .find(|b| b.name == "fl_knee_link")
        .unwrap()
        .collisions
        .first()
        .expect("shank geometry")
        .geometry;
    match shank {
        Geometry::Capsule { radius, length } => {
            assert!((radius - 0.02).abs() < 1e-12, "radius from class 'robot'");
            assert!((length - 0.2).abs() < 1e-12, "length from fromto");
        }
        other => panic!("expected capsule, got {other:?}"),
    }

    // Position actuators, with class defaults and a per-element override.
    assert_eq!(model.actuators.len(), 2);
    let act = |n: &str| model.actuators.iter().find(|a| a.name == n).unwrap();
    assert_eq!(act("fl_abduct_act").actuator_type, ActuatorType::Position);
    assert!((act("fl_abduct_act").gain - 60.0).abs() < 1e-12);
    assert_eq!(act("fl_abduct_act").bias, [0.0, -60.0, -2.0]);
    assert_eq!(act("fl_abduct_act").ctrl_range, Some([-1.5, 1.5]));
    assert!(
        (act("fl_knee_act").gain - 120.0).abs() < 1e-12,
        "kp override"
    );
    assert_eq!(act("fl_knee_act").bias, [0.0, -120.0, -2.0]);
}
