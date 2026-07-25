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
    /// Placement of [`Self::geometry`] within the body frame.
    ///
    /// MJCF geoms are almost never at the body origin — a capsule written with
    /// `fromto` sits at the midpoint of its two endpoints, typically well below
    /// the joint. Ignoring this offset makes limbs collide from the wrong
    /// place, so contact code must use it.
    pub geom_offset: GeomOffset,
}

/// Position and orientation of a body's collision geometry within the body
/// frame. `rot` maps geometry-local vectors into the body frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeomOffset {
    /// Geometry origin in body coordinates.
    pub pos: phyz_math::Vec3,
    /// Geometry→body rotation.
    pub rot: phyz_math::Mat3,
}

impl Default for GeomOffset {
    fn default() -> Self {
        Self {
            pos: phyz_math::Vec3::zeros(),
            rot: phyz_math::Mat3::identity(),
        }
    }
}

/// Collision geometry types (re-exported from phyz-collision for convenience).
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
        half_extents: phyz_math::Vec3,
    },
    Cylinder {
        radius: f64,
        height: f64,
    },
    Mesh {
        vertices: Vec<phyz_math::Vec3>,
        faces: Vec<[usize; 3]>,
    },
    Plane {
        normal: phyz_math::Vec3,
    },
}
