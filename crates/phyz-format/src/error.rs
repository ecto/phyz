//! Error types for phyz-format.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TauFormatError {
    #[error("JSON parse error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Unsupported domain type: {0}")]
    UnsupportedDomain(String),

    #[error("Unsupported coupling type: {0}")]
    UnsupportedCoupling(String),

    #[error("MJCF import error: {0}")]
    MjcfError(#[from] phyz_mjcf::MjcfError),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type Result<T> = std::result::Result<T, TauFormatError>;
