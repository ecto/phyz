//! URDF import for the phyz physics engine.
//!
//! Reads plain (non-xacro) URDF and produces a [`phyz_model::Model`] with the
//! kinematic tree, inertials, joint limits and dynamics, and visual/collision
//! primitives.
//!
//! ```no_run
//! let robot = phyz_urdf::load_file("panda.urdf", &Default::default()).unwrap();
//! println!("{} bodies, {} DOF", robot.model.nbodies(), robot.model.nv);
//! ```
//!
//! # Parser choice
//!
//! Parsing is delegated to [`urdf-rs`](https://crates.io/crates/urdf-rs), the
//! parser used by the OpenRR robotics stack. It already handles the parts of
//! URDF that are tedious to get right by hand — whitespace-separated numeric
//! attributes, the many optional elements and their defaults, `<mimic>`,
//! `<safety_controller>`, and `package://` path expansion — and it is
//! maintained alongside real robot descriptions. Hand-rolling the XML with the
//! `quick-xml` already in this workspace would avoid a few duplicate transitive
//! dependencies (it pulls its own `quick-xml` 0.36 and `thiserror` 1.0), but it
//! would mean re-implementing and re-testing that schema handling for no
//! correctness gain. All the *physics* conventions — the ones importers
//! actually get wrong — are handled explicitly in [`convert`], not by the
//! parser.
//!
//! # Not supported
//!
//! - **xacro**: `.xacro` files need macro expansion before they are URDF at
//!   all. They are normally preprocessed (`xacro model.xacro > model.urdf`),
//!   and [`urdf_rs::utils::convert_xacro_to_urdf`] can shell out to the ROS
//!   tool. Native expansion is future work.
//! - **Meshes**: URDF only names a mesh file; phyz geometry needs vertices.
//!   References are reported via [`UrdfModel::mesh_refs`] rather than being
//!   silently replaced by a made-up primitive.
//! - **`<mimic>` joints**: imported as independent DOFs, with a warning.
//! - **Transmissions, gazebo tags, sensors**: ignored.
//! - **Contact materials**: nothing is read, because URDF has nothing to
//!   read. Every imported body comes back with `Body::material == None` and
//!   therefore takes the scene material.
//!
//!   This is a gap in the format, not in the importer. URDF's `<material>`
//!   element is *appearance only* — a colour and a texture on `<visual>` —
//!   and the spec has no standard element anywhere for friction,
//!   restitution, or contact stiffness. What exists in practice is
//!   simulator-specific extension markup outside the URDF schema: Gazebo
//!   writes `<gazebo><mu1>/<mu2>` (SDF's `<surface><friction>`), and some
//!   toolchains attach Drake or MuJoCo blocks. Reading one of those would
//!   mean picking a single downstream simulator's convention and calling it
//!   "URDF", and it would not round-trip either — Gazebo's `mu1`/`mu2` are an
//!   anisotropic pair with per-direction axes that phyz's isotropic friction
//!   cone has no slot for.
//!
//!   So the supported path for a URDF robot is to set materials in code after
//!   import, by name:
//!
//!   ```no_run
//!   # use phyz_model::ContactMaterial;
//!   # let mut robot = phyz_urdf::load_file("k1.urdf", &Default::default()).unwrap();
//!   let sole = ContactMaterial { friction: 1.5, ..Default::default() };
//!   robot.model.set_body_material("left_foot", sole.clone());
//!   robot.model.set_body_material("right_foot", sole);
//!   ```
//!
//!   MJCF *does* express this, and `phyz-mjcf` reads it off `<geom>`. If a
//!   robot needs contact materials to come from its description file, MJCF is
//!   the format that carries them.

#![warn(missing_docs)]

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

pub mod convert;
pub mod error;

pub use convert::{
    BaseKind, MeshRef, UrdfModel, UrdfOptions, actuated_dofs, robot_to_model, rpy_to_matrix,
};
pub use error::{Result, UrdfError};

use std::path::Path;

/// Parse a URDF file and convert it to a phyz model.
pub fn load_file<P: AsRef<Path>>(path: P, options: &UrdfOptions) -> Result<UrdfModel> {
    let robot = urdf_rs::read_file(path.as_ref()).map_err(|e| UrdfError::Parse(e.to_string()))?;
    robot_to_model(&robot, options)
}

/// Parse a URDF from a string and convert it to a phyz model.
pub fn load_str(xml: &str, options: &UrdfOptions) -> Result<UrdfModel> {
    let robot = urdf_rs::read_from_string(xml).map_err(|e| UrdfError::Parse(e.to_string()))?;
    robot_to_model(&robot, options)
}
