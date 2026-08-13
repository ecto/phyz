//! Headless RGBD camera sensing for phyz.
//!
//! A robot-mounted pinhole camera that produces an RGB image and a **linear
//! depth image in metres** every step, cheap enough to run inside a training
//! loop. It is a wgpu *rasterizer*, not a ray tracer: at the 64–128 px
//! resolutions a world model actually consumes, rasterizing a few thousand
//! triangles costs microseconds and extends naturally to batched rendering,
//! whereas tracing rays does neither.
//!
//! # Conventions
//!
//! Getting these wrong is the usual way an RGBD sensor silently lies, so they
//! are stated once, loudly, and never varied:
//!
//! - **Optical frame is OpenCV.** The camera looks down its own **+Z**, **+X**
//!   points right across the image, and **+Y** points **down** the image. It is
//!   *not* the ROS/REP-103 body convention (+X forward, +Z up). A robot whose
//!   head link uses +X-forward expresses that as a rotation in the camera's
//!   *extrinsics*, not by reinterpreting the optical frame.
//! - **Pixels** are row-major from the top-left. `u` increases right, `v`
//!   increases down, `v = 0` is the top row, and pixel centres are at
//!   half-integers. Projection is `u = fx·X/Z + cx`, `v = fy·Y/Z + cy`.
//! - **Depth is linear metres of optical-axis `Z`** — the camera-space `Z` of
//!   the surface, *not* Euclidean ray length and *not* a normalised or
//!   reciprocal depth-buffer value. A wall 2 m ahead reads exactly `2.0` in
//!   every pixel that sees it, corners included. Pixels with no return read
//!   `0.0`.
//! - **Extrinsics** use the same Plücker convention as everything else in phyz:
//!   `origin.pos` is the optical-frame origin in body coordinates and
//!   `origin.rot` maps body coordinates into the optical frame, so the camera's
//!   orientation in the body frame is `origin.rot.transpose()`.
//!
//! # Where the poses come from
//!
//! Body poses are read from [`phyz_world::SensorContext::xforms`], the same
//! single per-step forward-kinematics pass every other sensor reads. The camera
//! never runs its own kinematics, so an image can never be a step out of sync
//! with the proprioception recorded beside it.
//!
//! # Example
//!
//! ```no_run
//! use phyz_camera::{CameraPose, RgbdCamera, RenderScene, SceneOptions};
//! use phyz_math::Vec3;
//! use phyz_world::{CameraIntrinsics, Scene, SensorContext};
//! # fn demo(model: &phyz_model::Model, state: &phyz_model::State) -> Result<(), Box<dyn std::error::Error>> {
//! let intrinsics = CameraIntrinsics::from_vfov(128, 128, 1.0, 0.05, 10.0);
//! let mut camera = RgbdCamera::new(intrinsics)?;
//!
//! let scene = Scene::empty().with_ground(0.0);
//! let ctx = SensorContext::free_flight(model, state, &scene);
//! let render_scene = RenderScene::from_context(&ctx, &SceneOptions::new());
//!
//! let pose = CameraPose::look_at(Vec3::new(0.0, -2.0, 1.0), Vec3::zeros(), Vec3::new(0.0, 0.0, 1.0));
//! let frame = camera.render(&render_scene, &pose)?;
//! println!("centre depth: {:?} m", frame.depth_at_principal_point());
//! # Ok(())
//! # }
//! ```
//!
//! # Not yet supported
//!
//! - Mesh formats other than STL (DAE, OBJ, glTF) — see
//!   [`CameraError::UnsupportedMeshFormat`].
//! - Textures, shadows and any form of global illumination. Shading is flat
//!   Lambert from one fixed directional light; geometric correctness is the
//!   goal, photorealism is not.
//! - Batched multi-environment rendering and GPU-resident output. The buffer
//!   enums in [`frame`] exist so those can be added without an API break.

#![warn(missing_docs)]

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

pub mod error;
pub mod frame;
pub mod mesh;
pub mod pose;
pub mod renderer;
pub mod scene;

pub use error::{CameraError, Result};
pub use frame::{CameraFrame, ColorBuffer, DepthBuffer};
pub use mesh::{Tessellation, TriMesh, Vertex, load_mesh, load_stl, tessellate};
pub use pose::{CameraPose, projection_matrix};
pub use renderer::{Lighting, RgbdCamera, body_pose, default_device, sensor_pose};
pub use scene::{GeometrySource, Instance, RenderScene, SceneOptions};

// Re-exported so callers configuring a camera need only one crate in scope.
pub use phyz_world::CameraIntrinsics;
