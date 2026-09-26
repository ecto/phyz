//! Errors produced while building or running a camera.

use thiserror::Error;

/// Errors that can occur while setting up or rendering an RGBD camera.
#[derive(Debug, Error)]
pub enum CameraError {
    /// No wgpu adapter could be found. Expected on headless CI without a
    /// software rasterizer; callers that want to degrade gracefully should
    /// match on this variant rather than unwrapping.
    #[error("no wgpu adapter available")]
    NoAdapter,

    /// The adapter refused to hand out a device.
    #[error("failed to create wgpu device: {0}")]
    DeviceCreation(String),

    /// The intrinsics do not describe a usable image.
    #[error(
        "invalid camera intrinsics: width={width} height={height} fx={fx} fy={fy} \
         near={near} far={far} (need positive sizes and focal lengths, 0 < near < far)"
    )]
    InvalidIntrinsics {
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
        /// Focal length along `x`.
        fx: f64,
        /// Focal length along `y`.
        fy: f64,
        /// Near clip distance.
        near: f64,
        /// Far clip distance.
        far: f64,
    },

    /// A sensor that is not [`phyz_world::Sensor::Camera`] was handed to a
    /// camera entry point.
    #[error("sensor {sensor_id} is not a Sensor::Camera")]
    NotACamera {
        /// Index of the offending sensor.
        sensor_id: usize,
    },

    /// The camera is mounted on a body that does not exist in the model, or one
    /// whose transform is missing from the kinematics pass.
    #[error("camera references body {body_idx}, but only {nbodies} bodies have transforms")]
    UnknownBody {
        /// The body index the camera asked for.
        body_idx: usize,
        /// How many body transforms were actually available.
        nbodies: usize,
    },

    /// A mesh file could not be read.
    #[error("failed to read mesh `{path}`: {source}")]
    MeshIo {
        /// The path that failed.
        path: String,
        /// Underlying IO error.
        source: std::io::Error,
    },

    /// A mesh file was read but could not be understood.
    #[error("failed to parse mesh `{path}`: {reason}")]
    MeshParse {
        /// The path that failed.
        path: String,
        /// What went wrong.
        reason: String,
    },

    /// A mesh format phyz-camera does not load yet (DAE, OBJ, glTF, …).
    #[error("unsupported mesh format for `{path}`: only STL is supported so far")]
    UnsupportedMeshFormat {
        /// The path that could not be dispatched.
        path: String,
    },

    /// Reading pixels back from the GPU failed.
    #[error("failed to map readback buffer: {0}")]
    Readback(String),
}

/// Result alias for camera operations.
pub type Result<T> = std::result::Result<T, CameraError>;
