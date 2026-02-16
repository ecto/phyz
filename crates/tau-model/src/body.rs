//! Rigid body definition.

use tau_math::SpatialInertia;

/// A rigid body in the kinematic tree.
#[derive(Debug, Clone)]
pub struct Body {
    /// Name of the body (optional, for debugging).
    pub name: String,
    /// Spatial inertia in body-local frame.
    pub inertia: SpatialInertia,
    /// Index of the parent body (-1 for world/root).
    pub parent: i32,
    /// Index of the joint connecting this body to its parent.
    pub joint_idx: usize,
    /// Collision geometry (if any).
    pub geometry: Option<Geometry>,
}

/// Collision geometry types (re-exported from tau-collision for convenience).
#[derive(Debug, Clone)]
pub enum Geometry {
    Sphere {
        radius: f64,
    },
    Capsule {
        radius: f64,
        length: f64,
    },
    Box {
        half_extents: tau_math::Vec3,
    },
    Cylinder {
        radius: f64,
        height: f64,
    },
    Mesh {
        vertices: Vec<tau_math::Vec3>,
        faces: Vec<[usize; 3]>,
    },
    Plane {
        normal: tau_math::Vec3,
    },
}
