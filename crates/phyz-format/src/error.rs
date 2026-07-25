//! Error types for phyz-format.

use thiserror::Error;

#[derive(Debug, Error)]
/// Anything that can go wrong reading or writing a `.phyz` scene.
pub enum TauFormatError {
    #[error("JSON parse error: {0}")]
    /// The document is not well-formed JSON.
    JsonError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    /// The file could not be read or written.
    IoError(#[from] std::io::Error),

    #[error("Invalid format: {0}")]
    /// Valid JSON, but not a valid `.phyz` document.
    InvalidFormat(String),

    #[error("Missing required field: {0}")]
    /// A required field is absent.
    MissingField(String),

    #[error("Unsupported domain type: {0}")]
    /// The document names a domain type this build does not support.
    UnsupportedDomain(String),

    #[error("Unsupported coupling type: {0}")]
    /// The document names a coupling type this build does not support.
    UnsupportedCoupling(String),

    #[error("MJCF import error: {0}")]
    /// An imported MJCF model failed to load.
    MjcfError(#[from] phyz_mjcf::MjcfError),

    #[error("Invalid parameter: {0}")]
    /// A field is present but its value is out of range or malformed.
    InvalidParameter(String),
}

/// `Result` specialised to [`TauFormatError`].
pub type Result<T> = std::result::Result<T, TauFormatError>;
