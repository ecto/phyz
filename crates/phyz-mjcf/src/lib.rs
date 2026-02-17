//! MuJoCo MJCF XML parser for phyz physics engine.
//!
//! Supports loading models from MJCF XML format and converting them to phyz Model.

mod defaults;
mod parser;

pub use defaults::ElementDefaults;
pub use parser::MjcfLoader;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MjcfError {
    #[error("XML parse error: {0}")]
    XmlError(#[from] quick_xml::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid MJCF: {0}")]
    InvalidMjcf(String),

    #[error("Unsupported feature: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, MjcfError>;
