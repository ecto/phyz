//! MuJoCo MJCF XML parser for phyz physics engine.
//!
//! Supports loading models from MJCF XML format and converting them to phyz Model.

#![warn(missing_docs)]

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

mod assets;
mod attrs;
pub mod defaults;
mod include;
pub mod inertia;
mod orientation;
mod parser;

pub use assets::{MeshAsset, MeshData};
pub use attrs::Attrs;
pub use defaults::DefaultsManager;
pub use parser::{MjcfLoader, SensorElement, UnsupportedFeature};

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

    #[error("<{element}> attribute '{attribute}' has invalid value {value:?}: {reason}")]
    /// An attribute was present but could not be interpreted. Carries the
    /// element and attribute names so the offending line is findable.
    InvalidAttribute {
        /// The element carrying the attribute.
        element: String,
        /// The attribute name.
        attribute: String,
        /// The value as written in the document.
        value: String,
        /// Why it could not be interpreted.
        reason: String,
    },

    #[error("<{element}> is missing required attribute '{attribute}'")]
    /// A required attribute was absent.
    MissingAttribute {
        /// The element missing the attribute.
        element: String,
        /// The attribute that was required.
        attribute: String,
    },

    #[error("Unsupported feature: {0}")]
    /// A valid MJCF construct this parser does not implement.
    Unsupported(String),
}

impl MjcfError {
    /// Build an [`MjcfError::InvalidAttribute`] naming the offending element,
    /// attribute and value.
    pub(crate) fn invalid_attr(
        element: &str,
        attribute: &str,
        value: &str,
        reason: impl Into<String>,
    ) -> Self {
        Self::InvalidAttribute {
            element: element.to_string(),
            attribute: attribute.to_string(),
            value: value.to_string(),
            reason: reason.into(),
        }
    }
}

/// `Result` specialised to [`MjcfError`].
pub type Result<T> = std::result::Result<T, MjcfError>;
