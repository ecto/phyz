//! phyz — Multi-physics differentiable simulation engine.
//!
//! This crate is a **facade**: every module below is a re-export of the
//! corresponding split crate. It exists so downstream users can depend on one
//! `phyz` rather than assembling `phyz-math` + `phyz-model` + `phyz-rigid` +
//! … by hand.
//!
//! Historically `phyz` carried hand-vendored copies of each subcrate that
//! differed only in import paths (and, over time, in stale comments). Those
//! copies are gone — there is exactly one implementation of each algorithm in
//! the tree, following the precedent set when `phyz::math` became a re-export
//! of `phyz-math`.

// Crate-name re-exports (`phyz::phyz_rigid::…`), for callers that want to be
// explicit about which subcrate a symbol comes from.
pub use phyz_collision;
pub use phyz_contact;
pub use phyz_diff;
pub use phyz_math;
pub use phyz_model;
pub use phyz_rigid;

// Short module aliases (`phyz::rigid::…`) — the historical spelling.
pub use phyz_collision as collision;
pub use phyz_contact as contact;
pub use phyz_diff as diff;
pub use phyz_math as math;
pub use phyz_model as model;
pub use phyz_rigid as rigid;

// Re-export core types at crate root for convenience.
pub use collision::{
    AABB, Collision, epa_penetration, epa_penetration_rot, gjk_distance, gjk_distance_rot,
    sweep_and_prune,
};
pub use contact::{
    ContactMaterial, compute_contact_force, compute_contact_force_implicit, contact_forces,
    contact_forces_implicit, find_contacts, find_ground_contacts,
};
pub use math::{DMat, DVec, GRAVITY, Mat3, Mat4, Vec3, skew};
pub use math::{Quat, SpatialInertia, SpatialMat, SpatialTransform, SpatialVec};
pub use model::{Actuator, Body, Geometry, Joint, JointType, Model, ModelBuilder, State};
pub use rigid::{
    Simulator, aba, aba_with_external_forces, crba, forward_kinematics, rnea, total_energy,
};
