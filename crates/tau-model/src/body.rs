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
}
