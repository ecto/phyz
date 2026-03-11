//! phyz — Multi-physics differentiable simulation engine.

pub mod math;
pub mod model;
pub mod rigid;
pub mod collision;
pub mod contact;

// Re-export core types at crate root for convenience
pub use math::{Vec3, Mat3, Mat4, DVec, DMat, skew, GRAVITY};
pub use math::{Quat, SpatialInertia, SpatialMat, SpatialTransform, SpatialVec};
pub use model::{Body, Geometry, Joint, JointType, Actuator, Model, ModelBuilder, State};
pub use rigid::{aba, aba_with_external_forces, crba, forward_kinematics, rnea};
pub use collision::{Collision, AABB, sweep_and_prune, gjk_distance, gjk_distance_rot, epa_penetration, epa_penetration_rot};
pub use contact::{ContactMaterial, contact_forces, find_contacts, find_ground_contacts};
