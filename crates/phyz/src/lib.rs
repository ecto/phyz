//! phyz — Multi-physics differentiable simulation engine.

pub mod collision;
pub mod contact;
pub mod diff;
pub use ::phyz_math;
pub use ::phyz_math as math;
pub mod model;
pub mod rigid;
pub mod sim;

// Legacy sub-crate-style aliases: the modules below were once standalone
// crates re-exported here. Kept so `phyz::phyz_rigid::…` paths stay valid.
pub use collision as phyz_collision;
pub use contact as phyz_contact;
pub use diff as phyz_diff;
pub use model as phyz_model;
pub use rigid as phyz_rigid;

// Re-export core types at crate root for convenience
pub use collision::{
    AABB, Collision, epa_penetration, epa_penetration_rot, gjk_distance, gjk_distance_rot,
    sweep_and_prune,
};
pub use contact::{
    ContactMaterial, compute_contact_force, compute_contact_force_implicit, contact_forces,
    contact_forces_implicit, find_contacts, find_ground_contacts,
};
pub use diff::StepJacobians;
pub use math::{DMat, DVec, GRAVITY, Mat3, Mat4, Vec3, skew};
pub use math::{Quat, SpatialInertia, SpatialMat, SpatialTransform, SpatialVec};
pub use model::{Actuator, Body, Geometry, Joint, JointType, Model, ModelBuilder, State};
pub use rigid::{
    aba, aba_with_external_forces, crba, forward_kinematics, kinetic_energy, potential_energy,
    rnea, total_energy,
};
pub use sim::{Rk4Solver, SemiImplicitEulerSolver, Simulator, Solver};
