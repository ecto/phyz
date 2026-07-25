//! Importers for converting common formats to .phyz specification.

use crate::domain::{BodySpec, DomainType, JointSpec, JointTypeSpec};
use crate::error::{Result, TauFormatError};
use crate::schema::{PhyzSpec, WorldConfig};
use phyz_mjcf::MjcfLoader;
use phyz_model::{JointType, Model};
use std::collections::HashMap;

/// Convert MJCF file to PhyzSpec.
pub fn from_mjcf(path: &str) -> Result<PhyzSpec> {
    let loader = MjcfLoader::from_file(path)?;
    let model = loader.build_model();

    model_to_phyz_spec(&model, path)
}

/// Convert a URDF file to PhyzSpec.
///
/// Handles plain URDF; `.xacro` files must be expanded first (see the
/// [`phyz_urdf`] crate docs). Use [`from_urdf_model`] instead if you need the
/// import warnings or mesh references, which `PhyzSpec` cannot carry.
pub fn from_urdf(path: &str) -> Result<PhyzSpec> {
    let robot = phyz_urdf::load_file(path, &Default::default())?;
    model_to_phyz_spec(&robot.model, path)
}

/// Import a URDF file, returning the full [`phyz_urdf::UrdfModel`] alongside
/// its `PhyzSpec` so callers can inspect warnings and unresolved meshes.
pub fn from_urdf_model(path: &str) -> Result<(phyz_urdf::UrdfModel, PhyzSpec)> {
    let robot = phyz_urdf::load_file(path, &Default::default())?;
    let spec = model_to_phyz_spec(&robot.model, path)?;
    Ok((robot, spec))
}

/// USD import — **not implemented**.
///
/// Always returns [`TauFormatError::UnsupportedImportFormat`]. USD scene
/// description needs the OpenUSD runtime (or a substantial subset of it) to
/// resolve layers, references, and variants; there is no partial version of
/// this worth shipping.
pub fn from_usd(_path: &str) -> Result<PhyzSpec> {
    Err(TauFormatError::UnsupportedImportFormat { format: "USD" })
}

/// SDF import — **not implemented**.
///
/// Always returns [`TauFormatError::UnsupportedImportFormat`]. SDF (Gazebo)
/// overlaps heavily with URDF but adds worlds, models, nested includes, and
/// plugin configuration; it is a plausible future addition, not a stub.
pub fn from_sdf(_path: &str) -> Result<PhyzSpec> {
    Err(TauFormatError::UnsupportedImportFormat { format: "SDF" })
}

/// Convert a tau Model to PhyzSpec.
fn model_to_phyz_spec(model: &Model, source_name: &str) -> Result<PhyzSpec> {
    // Preserve source names where the model has them (URDF always does, MJCF
    // sometimes does not), falling back to positional names.
    let body_name = |i: usize| -> String {
        let n = &model.bodies[i].name;
        if n.is_empty() {
            format!("body_{}", i)
        } else {
            n.clone()
        }
    };

    // Extract bodies
    let mut bodies = Vec::new();
    for (i, body) in model.bodies.iter().enumerate() {
        let name = body_name(i);
        bodies.push(BodySpec {
            name: name.clone(),
            mass: body.inertia.mass,
            inertia: [
                body.inertia.inertia[(0, 0)],
                body.inertia.inertia[(1, 1)],
                body.inertia.inertia[(2, 2)],
                body.inertia.inertia[(0, 1)],
                body.inertia.inertia[(0, 2)],
                body.inertia.inertia[(1, 2)],
            ],
            center_of_mass: [body.inertia.com.x, body.inertia.com.y, body.inertia.com.z],
        });
    }

    // Extract joints
    let mut joints = Vec::new();
    for (i, joint) in model.joints.iter().enumerate() {
        let body = &model.bodies[i];
        let parent_name = if body.parent < 0 {
            "world".to_string()
        } else {
            body_name(body.parent as usize)
        };
        let child_name = body_name(i);

        let joint_type = match joint.joint_type {
            JointType::Free => JointTypeSpec::Free,
            JointType::Revolute | JointType::Hinge => JointTypeSpec::Revolute,
            JointType::Prismatic | JointType::Slide => JointTypeSpec::Prismatic,
            JointType::Spherical | JointType::Ball => JointTypeSpec::Spherical,
            JointType::Fixed => JointTypeSpec::Fixed,
        };

        let pos = joint.parent_to_joint.pos;
        let axis = joint.axis;

        // Convert rotation matrix to quaternion (simplified)
        let quat = mat3_to_quat(&joint.parent_to_joint.rot);

        joints.push(JointSpec {
            joint_type,
            parent: parent_name,
            child: child_name,
            axis: [axis.x, axis.y, axis.z],
            position: [pos.x, pos.y, pos.z],
            orientation: quat,
            limits: joint.limits,
            damping: joint.damping,
        });
    }

    // Build domain config
    let mut rigid_config = HashMap::new();
    rigid_config.insert("bodies".to_string(), serde_json::to_value(&bodies)?);
    rigid_config.insert("joints".to_string(), serde_json::to_value(&joints)?);

    let mut domains = HashMap::new();
    domains.insert(
        "rigid_body".to_string(),
        crate::domain::Domain {
            domain_type: DomainType::RigidBodyDynamics,
            config: rigid_config,
        },
    );

    Ok(PhyzSpec {
        version: "1.0".to_string(),
        name: source_name.to_string(),
        description: format!("Imported from {}", source_name),
        world: WorldConfig {
            gravity: [model.gravity.x, model.gravity.y, model.gravity.z],
            dt: model.dt,
            default_contact_material: Default::default(),
        },
        domains,
        couplings: vec![],
        parameters: HashMap::new(),
        importers: vec![],
    })
}

/// Convert rotation matrix to quaternion [w, x, y, z].
fn mat3_to_quat(mat: &phyz_math::Mat3) -> [f64; 4] {
    // Shepperd's method
    let trace = mat[(0, 0)] + mat[(1, 1)] + mat[(2, 2)];

    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        let w = 0.25 * s;
        let x = (mat[(2, 1)] - mat[(1, 2)]) / s;
        let y = (mat[(0, 2)] - mat[(2, 0)]) / s;
        let z = (mat[(1, 0)] - mat[(0, 1)]) / s;
        [w, x, y, z]
    } else if mat[(0, 0)] > mat[(1, 1)] && mat[(0, 0)] > mat[(2, 2)] {
        let s = (1.0 + mat[(0, 0)] - mat[(1, 1)] - mat[(2, 2)]).sqrt() * 2.0;
        let w = (mat[(2, 1)] - mat[(1, 2)]) / s;
        let x = 0.25 * s;
        let y = (mat[(0, 1)] + mat[(1, 0)]) / s;
        let z = (mat[(0, 2)] + mat[(2, 0)]) / s;
        [w, x, y, z]
    } else if mat[(1, 1)] > mat[(2, 2)] {
        let s = (1.0 + mat[(1, 1)] - mat[(0, 0)] - mat[(2, 2)]).sqrt() * 2.0;
        let w = (mat[(0, 2)] - mat[(2, 0)]) / s;
        let x = (mat[(0, 1)] + mat[(1, 0)]) / s;
        let y = 0.25 * s;
        let z = (mat[(1, 2)] + mat[(2, 1)]) / s;
        [w, x, y, z]
    } else {
        let s = (1.0 + mat[(2, 2)] - mat[(0, 0)] - mat[(1, 1)]).sqrt() * 2.0;
        let w = (mat[(1, 0)] - mat[(0, 1)]) / s;
        let x = (mat[(0, 2)] + mat[(2, 0)]) / s;
        let y = (mat[(1, 2)] + mat[(2, 1)]) / s;
        let z = 0.25 * s;
        [w, x, y, z]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PANDA: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../phyz-urdf/tests/data/panda.urdf"
    );

    #[test]
    fn from_urdf_imports_a_real_robot() {
        let spec = from_urdf(PANDA).expect("panda.urdf should import");

        let rigid = spec
            .domains
            .get("rigid_body")
            .expect("URDF import produces a rigid-body domain");
        let bodies = rigid.config["bodies"].as_array().unwrap();
        let joints = rigid.config["joints"].as_array().unwrap();
        assert_eq!(bodies.len(), 13);
        assert_eq!(joints.len(), 13);

        // URDF link names survive into the spec instead of becoming `body_N`.
        assert_eq!(bodies[0]["name"], "panda_link0");
        assert_eq!(joints[0]["parent"], "world");
        assert!(
            joints
                .iter()
                .any(|j| j["child"] == "panda_link8" && j["type"] == "fixed")
        );

        // Masses came across, not zeros.
        let total: f64 = bodies.iter().map(|b| b["mass"].as_f64().unwrap()).sum();
        assert!(total > 1.0, "total mass was {total}");
    }

    #[test]
    fn from_urdf_model_exposes_warnings_and_meshes() {
        let (robot, _spec) = from_urdf_model(PANDA).unwrap();
        assert_eq!(robot.robot_name, "panda");
        // The Panda description is mesh-based, so every mesh must be reported.
        assert!(!robot.mesh_refs.is_empty());
        assert!(robot.warnings.iter().any(|w| w.contains("mesh")));
    }

    #[test]
    fn from_urdf_reports_missing_files() {
        assert!(matches!(
            from_urdf("does_not_exist.urdf"),
            Err(TauFormatError::UrdfError(_))
        ));
    }

    // USD and SDF remain documented gaps: they must fail loudly and be
    // identifiable as "unsupported", not as a malformed file.
    #[test]
    fn from_usd_is_an_explicit_unsupported_format() {
        assert!(matches!(
            from_usd("test.usd"),
            Err(TauFormatError::UnsupportedImportFormat { format: "USD" })
        ));
    }

    #[test]
    fn from_sdf_is_an_explicit_unsupported_format() {
        assert!(matches!(
            from_sdf("test.sdf"),
            Err(TauFormatError::UnsupportedImportFormat { format: "SDF" })
        ));
    }
}
