//! Integration tests for the features this crate gained on top of the
//! quaternion-only, include-less, mesh-less baseline: the full set of MJCF
//! orientation spellings, `<include>` expansion, and `<asset><mesh>` loading.
//!
//! Defaults, actuators and inertia-from-geom are covered by the unit tests in
//! `src/parser.rs` and by `tests/dynamics.rs`.

use phyz_math::Vec3;
use phyz_mjcf::{MjcfError, MjcfLoader};
use phyz_model::Geometry;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn load(name: &str) -> MjcfLoader {
    match MjcfLoader::from_file(fixture(name)) {
        Ok(l) => l,
        Err(e) => panic!("failed to load {name}: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Orientations
// ---------------------------------------------------------------------------

#[test]
fn every_orientation_spelling_is_honoured() {
    let model = load("orientations.xml").build_model();
    let rot_of = |name: &str| {
        let i = model.bodies.iter().position(|b| b.name == name).unwrap();
        model.joints[i].parent_to_joint.rot
    };

    // quat, euler(0 0 90), axisangle(z, 90) and xyaxes all describe the same
    // rotation.
    let q = rot_of("by_quat");
    for other in ["by_euler", "by_axisangle", "by_xyaxes"] {
        let r = rot_of(other);
        assert!(
            (q - r).norm_sq().sqrt() < 1e-9,
            "{other} disagrees with the equivalent quat"
        );
    }

    // zaxis="1 0 0" maps +Z onto +X.
    let z = rot_of("by_zaxis") * Vec3::new(0.0, 0.0, 1.0);
    assert!((z - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-9, "{z:?}");
}

/// Regression guard: before these were parsed, every non-`quat` orientation
/// silently resolved to identity, producing a plausible-looking wrong model.
#[test]
fn euler_is_not_silently_identity() {
    let model = load("orientations.xml").build_model();
    let i = model
        .bodies
        .iter()
        .position(|b| b.name == "by_euler")
        .unwrap();
    let rot = model.joints[i].parent_to_joint.rot;
    assert!(
        (rot - phyz_math::Mat3::identity()).norm_sq().sqrt() > 1e-6,
        "euler orientation resolved to identity"
    );
}

#[test]
fn fromto_sets_capsule_length_and_frame() {
    let model = load("orientations.xml").build_model();
    let body = model
        .bodies
        .iter()
        .find(|b| b.name == "capsule_fromto")
        .unwrap();
    match body.geometry.as_ref().expect("fromto capsule") {
        Geometry::Capsule { radius, length } => {
            assert!((radius - 0.03).abs() < 1e-12);
            // fromto spans 0.4 m, so half-length 0.2 and full length 0.4.
            assert!((length - 0.4).abs() < 1e-12, "length {length}");
        }
        other => panic!("expected capsule, got {other:?}"),
    }
}

#[test]
fn degenerate_orientations_are_errors_not_identity() {
    let cases = [
        r#"<mujoco><worldbody><body name="b" quat="0 0 0 0"/></worldbody></mujoco>"#,
        r#"<mujoco><worldbody><body name="b" zaxis="0 0 0"/></worldbody></mujoco>"#,
        r#"<mujoco><worldbody><body name="b" axisangle="0 0 0 1"/></worldbody></mujoco>"#,
        r#"<mujoco><worldbody><body name="b" euler="1 2"/></worldbody></mujoco>"#,
        r#"<mujoco><compiler eulerseq="xyzw"/></mujoco>"#,
    ];
    for xml in cases {
        assert!(
            MjcfLoader::from_xml_str(xml).is_err(),
            "expected an error for {xml}"
        );
    }
}

#[test]
fn orientation_errors_name_the_element_and_attribute() {
    let err = MjcfLoader::from_xml_str(
        r#"<mujoco><worldbody><body name="arm" euler="1 2"/></worldbody></mujoco>"#,
    )
    .err()
    .expect("should fail")
    .to_string();
    assert!(err.contains("body"), "{err}");
    assert!(err.contains("euler"), "{err}");
    assert!(err.contains("expected 3 numbers"), "{err}");
}

// ---------------------------------------------------------------------------
// <include>
// ---------------------------------------------------------------------------

#[test]
fn include_splices_bodies_and_defaults() {
    let model = load("include_main.xml").build_model();
    assert_eq!(model.nbodies(), 2);
    assert!(model.bodies.iter().any(|b| b.name == "included_child"));

    // The default class came from a third file entirely.
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
    let err = MjcfLoader::from_file(fixture("include_cycle_a.xml"))
        .err()
        .expect("cycle should fail")
        .to_string();
    assert!(err.contains("include"), "{err}");
    assert!(err.contains("cyclic") || err.contains("nesting"), "{err}");
}

#[test]
fn missing_include_file_names_the_file() {
    let err = MjcfLoader::from_xml_str(r#"<mujoco><include file="does_not_exist.xml"/></mujoco>"#)
        .err()
        .expect("should fail")
        .to_string();
    assert!(err.contains("does_not_exist.xml"), "{err}");
}

#[test]
fn include_without_a_file_attribute_errors() {
    let err = MjcfLoader::from_xml_str(r#"<mujoco><include/></mujoco>"#)
        .err()
        .expect("should fail");
    assert!(matches!(err, MjcfError::MissingAttribute { .. }), "{err}");
}

// ---------------------------------------------------------------------------
// <asset><mesh>
// ---------------------------------------------------------------------------

#[test]
fn mesh_assets_load_and_reach_the_model() {
    let loader = load("assets.xml");
    let tetra = loader
        .meshes()
        .iter()
        .find(|m| m.name == "tetra")
        .expect("tetra asset");
    let data = tetra.data.as_ref().expect("tetra.obj should load");
    assert_eq!(data.vertices.len(), 4);
    assert_eq!(data.faces.len(), 4);

    let model = loader.build_model();
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
fn unloadable_mesh_is_reported_not_fatal() {
    let loader = load("assets.xml");
    let missing = loader
        .meshes()
        .iter()
        .find(|m| m.name == "missing")
        .expect("missing asset should still be recorded");
    assert!(missing.data.is_none());
    assert!(missing.load_error.is_some());
    assert!(
        loader.unsupported().iter().any(|u| u.tag == "mesh"),
        "an unloadable mesh should be reported: {:?}",
        loader.unsupported()
    );
}

#[test]
fn heightfield_gap_is_reported() {
    let loader = load("assets.xml");
    let note = loader
        .unsupported()
        .iter()
        .find(|u| u.tag == "hfield")
        .expect("hfield should be reported as unsupported");
    assert!(note.detail.contains("heightfield"), "{}", note.detail);
}

// ---------------------------------------------------------------------------
// Repo models still load
// ---------------------------------------------------------------------------

#[test]
fn repo_models_parse() {
    for name in ["ant.xml", "simple_arm.xml"] {
        let path = format!("{}/../../models/{name}", env!("CARGO_MANIFEST_DIR"));
        let model = MjcfLoader::from_file(&path)
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .build_model();
        assert!(model.nbodies() > 0, "{name} produced an empty model");
    }
}
