//! Rigid body definition.

use crate::math::{SpatialInertia, SpatialTransform};

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
    /// Primary collision geometry, centred on the body frame (if any).
    ///
    /// This is the shape used by the contact pipeline. For sources that
    /// describe several offset shapes per body (URDF, MJCF) the full set
    /// lives in [`Body::collisions`]; `geometry` mirrors the first one whose
    /// origin is the identity, so existing single-shape consumers keep working.
    pub geometry: Option<Geometry>,
    /// All collision shapes attached to this body, each with its own offset.
    pub collisions: Vec<GeomInstance>,
    /// Visual-only shapes attached to this body (not used for contact).
    pub visuals: Vec<GeomInstance>,
}

impl Body {
    /// Create a body with no attached geometry.
    pub fn new(name: &str, inertia: SpatialInertia, parent: i32, joint_idx: usize) -> Self {
        Self {
            name: name.to_string(),
            inertia,
            parent,
            joint_idx,
            geometry: None,
            collisions: Vec::new(),
            visuals: Vec::new(),
        }
    }
}

/// A geometry placed at an offset within a body frame.
#[derive(Debug, Clone)]
pub struct GeomInstance {
    /// Optional name from the source file.
    pub name: Option<String>,
    /// Placement of the shape within the body frame.
    ///
    /// Follows the same convention as [`crate::Joint::parent_to_joint`]:
    /// `origin.pos` is the shape origin expressed in body coordinates and
    /// `origin.rot` is the coordinate transform *body → shape*, so the shape's
    /// orientation in the body frame is `origin.rot.transpose()`.
    pub origin: SpatialTransform,
    /// The shape itself.
    pub geometry: Geometry,
}

impl GeomInstance {
    /// Create a geom instance at the given offset.
    pub fn new(geometry: Geometry, origin: SpatialTransform) -> Self {
        Self {
            name: None,
            origin,
            geometry,
        }
    }

    /// Create a geom instance centred on the body frame.
    pub fn centered(geometry: Geometry) -> Self {
        Self::new(geometry, SpatialTransform::identity())
    }

    /// True if this shape sits exactly on the body frame.
    pub fn is_centered(&self) -> bool {
        const EPS: f64 = 1e-12;
        self.origin.pos.norm_sq() < EPS
            && (self.origin.rot - crate::math::Mat3::identity()).norm_sq() < EPS
    }
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
        half_extents: crate::math::Vec3,
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
        vertices: Vec<crate::math::Vec3>,
        /// Triangles, as triples of indices into `vertices`.
        faces: Vec<[usize; 3]>,
    },
    /// A half-space through the body frame origin.
    Plane {
        /// Outward unit normal of the plane, in body frame.
        normal: crate::math::Vec3,
    },
}
