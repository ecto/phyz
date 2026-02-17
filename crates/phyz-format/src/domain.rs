//! Domain specifications for different physics solvers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of physics domain.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum DomainType {
    /// Rigid body dynamics.
    RigidBodyDynamics,
    /// Convex collision detection.
    ConvexCollision,
    /// Material Point Method particles.
    Mpm,
    /// Electromagnetic FDTD.
    Electromagnetic,
    /// Molecular dynamics.
    MolecularDynamics,
    /// Quantum field theory (lattice gauge).
    QuantumFieldTheory,
    /// Gravitational N-body + GR.
    Gravity,
    /// Lattice Boltzmann Method.
    LatticeBoltzmann,
}

/// Generic domain specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Domain {
    /// Type of domain.
    #[serde(rename = "type")]
    pub domain_type: DomainType,

    /// Domain-specific configuration.
    #[serde(flatten)]
    pub config: HashMap<String, serde_json::Value>,
}

/// Rigid body domain specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigidBodyDomain {
    /// Body definitions.
    pub bodies: Vec<BodySpec>,
    /// Joint definitions.
    pub joints: Vec<JointSpec>,
}

/// Body specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodySpec {
    /// Body name.
    pub name: String,
    /// Mass (kg).
    pub mass: f64,
    /// Inertia matrix [Ixx, Iyy, Izz, Ixy, Ixz, Iyz].
    pub inertia: [f64; 6],
    /// Center of mass offset [x, y, z].
    #[serde(default)]
    pub center_of_mass: [f64; 3],
}

/// Joint specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointSpec {
    /// Joint type.
    #[serde(rename = "type")]
    pub joint_type: JointTypeSpec,
    /// Parent body name.
    pub parent: String,
    /// Child body name.
    pub child: String,
    /// Joint axis (for revolute/prismatic).
    #[serde(default)]
    pub axis: [f64; 3],
    /// Position of joint in parent frame.
    #[serde(default)]
    pub position: [f64; 3],
    /// Orientation of joint in parent frame (quaternion [w, x, y, z]).
    #[serde(default = "default_quaternion")]
    pub orientation: [f64; 4],
    /// Joint limits [lower, upper].
    #[serde(default)]
    pub limits: Option<[f64; 2]>,
    /// Joint damping.
    #[serde(default)]
    pub damping: f64,
}

fn default_quaternion() -> [f64; 4] {
    [1.0, 0.0, 0.0, 0.0]
}

/// Joint type specification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JointTypeSpec {
    /// Free 6-DOF joint.
    Free,
    /// Revolute (hinge) joint.
    Revolute,
    /// Prismatic (slider) joint.
    Prismatic,
    /// Spherical (ball) joint.
    Spherical,
    /// Fixed joint.
    Fixed,
}

/// Collision domain specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionDomain {
    /// Collision geometries.
    pub geometries: Vec<GeometrySpec>,
}

/// Geometry specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometrySpec {
    /// Body this geometry is attached to.
    pub body: String,
    /// Shape type.
    pub shape: ShapeType,
    /// Shape-specific parameters.
    #[serde(flatten)]
    pub params: HashMap<String, serde_json::Value>,
}

/// Collision shape types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ShapeType {
    /// Box shape.
    Box,
    /// Sphere shape.
    Sphere,
    /// Capsule shape.
    Capsule,
    /// Cylinder shape.
    Cylinder,
    /// Convex mesh.
    Mesh,
}

/// Particle domain specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleDomain {
    /// Material type.
    pub material: String,
    /// Number of particles.
    pub particles: usize,
    /// Grid resolution [nx, ny, nz].
    pub grid_resolution: [usize; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_type_serialization() {
        let domain_type = DomainType::RigidBodyDynamics;
        let json = serde_json::to_string(&domain_type).unwrap();
        assert_eq!(json, "\"rigid-body-dynamics\"");

        let parsed: DomainType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, domain_type);
    }

    #[test]
    fn test_body_spec_serialization() {
        let body = BodySpec {
            name: "torso".to_string(),
            mass: 1.0,
            inertia: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            center_of_mass: [0.0, 0.0, 0.0],
        };

        let json = serde_json::to_string(&body).unwrap();
        let parsed: BodySpec = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, "torso");
        assert_eq!(parsed.mass, 1.0);
    }

    #[test]
    fn test_joint_spec_serialization() {
        let joint = JointSpec {
            joint_type: JointTypeSpec::Revolute,
            parent: "world".to_string(),
            child: "link1".to_string(),
            axis: [0.0, 0.0, 1.0],
            position: [0.0, 0.0, 0.0],
            orientation: [1.0, 0.0, 0.0, 0.0],
            limits: Some([-3.14, 3.14]),
            damping: 0.1,
        };

        let json = serde_json::to_string(&joint).unwrap();
        let parsed: JointSpec = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.joint_type, JointTypeSpec::Revolute);
        assert_eq!(parsed.parent, "world");
    }
}
