//! Importers for converting common formats to .tau specification.

use crate::domain::{BodySpec, DomainType, JointSpec, JointTypeSpec};
use crate::error::{Result, TauFormatError};
use crate::schema::{TauSpec, WorldConfig};
use std::collections::HashMap;
use tau_mjcf::MjcfLoader;
use tau_model::{JointType, Model};

/// Convert MJCF file to TauSpec.
pub fn from_mjcf(path: &str) -> Result<TauSpec> {
    let loader = MjcfLoader::from_file(path)?;
    let model = loader.build_model();

    model_to_tau_spec(&model, path)
}

/// Convert URDF file to TauSpec.
///
/// Note: URDF parsing not yet implemented. Returns placeholder.
pub fn from_urdf(path: &str) -> Result<TauSpec> {
    // Placeholder: URDF parsing would go here
    Err(TauFormatError::InvalidFormat(format!(
        "URDF import not yet implemented: {}",
        path
    )))
}

/// Convert USD file to TauSpec.
///
/// Note: USD parsing not yet implemented. Returns placeholder.
pub fn from_usd(path: &str) -> Result<TauSpec> {
    Err(TauFormatError::InvalidFormat(format!(
        "USD import not yet implemented: {}",
        path
    )))
}

/// Convert SDF file to TauSpec.
///
/// Note: SDF parsing not yet implemented. Returns placeholder.
pub fn from_sdf(path: &str) -> Result<TauSpec> {
    Err(TauFormatError::InvalidFormat(format!(
        "SDF import not yet implemented: {}",
        path
    )))
}

/// Convert a tau Model to TauSpec.
fn model_to_tau_spec(model: &Model, source_name: &str) -> Result<TauSpec> {
    // Extract bodies
    let mut bodies = Vec::new();
    for (i, body) in model.bodies.iter().enumerate() {
        let name = format!("body_{}", i);
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
            format!("body_{}", body.parent)
        };
        let child_name = format!("body_{}", i);

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

    Ok(TauSpec {
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
fn mat3_to_quat(mat: &tau_math::Mat3) -> [f64; 4] {
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

    #[test]
    fn test_from_urdf_not_implemented() {
        let result = from_urdf("test.urdf");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_usd_not_implemented() {
        let result = from_usd("test.usd");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_sdf_not_implemented() {
        let result = from_sdf("test.sdf");
        assert!(result.is_err());
    }
}
