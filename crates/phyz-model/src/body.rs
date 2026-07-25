//! Rigid body definition.

use phyz_math::SpatialInertia;

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

/// Collision geometry types (mirrors `phyz-collision`'s shapes for convenience).
#[derive(Debug, Clone)]
pub enum Geometry {
    /// A ball centred on the body frame origin.
    Sphere {
        /// Ball radius, in metres.
        radius: f64,
    },
    /// A capsule (swept sphere) aligned with the body frame Z axis.
    Capsule {
        /// Radius of the swept sphere, in metres.
        radius: f64,
        /// Distance between the two cap centres, in metres.
        length: f64,
    },
    /// An axis-aligned box centred on the body frame origin.
    Box {
        /// Half-extent along each body frame axis, in metres.
        half_extents: phyz_math::Vec3,
    },
    /// A cylinder aligned with the body frame Z axis.
    Cylinder {
        /// Cylinder radius, in metres.
        radius: f64,
        /// Cylinder height along Z, in metres.
        height: f64,
    },
    /// An arbitrary triangle mesh in body frame coordinates.
    Mesh {
        /// Vertex positions in body frame.
        vertices: Vec<phyz_math::Vec3>,
        /// Triangles, as triples of indices into `vertices`.
        faces: Vec<[usize; 3]>,
    },
    /// A half-space through the body frame origin.
    Plane {
        /// Outward unit normal of the plane, in body frame.
        normal: phyz_math::Vec3,
    },
}
