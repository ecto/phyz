//! phyz — Multi-physics differentiable simulation engine.

pub mod collision;
pub mod contact;
pub mod diff;
pub use phyz_math as math;
pub mod model;
pub mod rigid;

// Re-export core types at crate root for convenience
pub use collision::{
    AABB, Collision, epa_penetration, epa_penetration_rot, gjk_distance, gjk_distance_rot,
    sweep_and_prune,
};
pub use contact::{ContactMaterial, contact_forces, find_contacts, find_ground_contacts};
pub use math::{DMat, DVec, GRAVITY, Mat3, Mat4, Vec3, skew};
pub use math::{Quat, SpatialInertia, SpatialMat, SpatialTransform, SpatialVec};
pub use model::{
    Actuator, Body, GeomInstance, Geometry, Joint, JointType, Model, ModelBuilder, State,
};
pub use rigid::{aba, aba_with_external_forces, crba, forward_kinematics, rnea};
