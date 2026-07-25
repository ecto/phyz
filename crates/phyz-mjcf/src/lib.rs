//! MuJoCo MJCF XML parser for phyz physics engine.
//!
//! Supports loading models from MJCF XML format and converting them to phyz Model.

#![warn(missing_docs)]

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

mod defaults;
mod parser;

pub use defaults::ElementDefaults;
pub use parser::MjcfLoader;

use thiserror::Error;

#[derive(Debug, Error)]
/// Anything that can go wrong loading an MJCF model.
pub enum MjcfError {
    #[error("XML parse error: {0}")]
    /// The document is not well-formed XML.
    XmlError(#[from] quick_xml::Error),

    #[error("IO error: {0}")]
    /// The model file could not be read.
    IoError(#[from] std::io::Error),

    #[error("Invalid MJCF: {0}")]
    /// The document is valid XML but not a valid MJCF model.
    InvalidMjcf(String),

    #[error("Unsupported feature: {0}")]
    /// A valid MJCF construct this parser does not implement.
    Unsupported(String),
}

/// `Result` specialised to [`MjcfError`].
pub type Result<T> = std::result::Result<T, MjcfError>;
