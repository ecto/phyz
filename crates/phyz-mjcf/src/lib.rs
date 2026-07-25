//! MuJoCo MJCF XML parser for phyz physics engine.
//!
//! Supports loading models from MJCF XML format and converting them to phyz Model.
//!
//! # Supported subset
//!
//! - `<compiler>`: `angle`, `coordinate`, `eulerseq`, `meshdir`, `assetdir`
//! - `<option>`: `gravity`, `timestep`
//! - `<default>`: class definitions with nesting/inheritance, plus `class` and `childclass`
//!   references on elements
//! - `<worldbody>`/`<body>`/`<joint>`/`<freejoint>`/`<inertial>`/`<geom>`/`<site>`
//! - Orientations: `quat`, `euler`, `axisangle`, `xyaxes`, `zaxis`, and `fromto` on geoms
//! - `<actuator>`: `motor`, `position`, `velocity`, `general`
//! - `<asset>`: `mesh`, `texture`, `material`, `hfield` (records; STL/OBJ meshes are loaded)
//! - `<include>` file inclusion
//!
//! Parsed-but-not-simulated (recorded on [`MjcfLoader`] for inspection, see
//! [`MjcfLoader::unsupported`]): `<equality>`, `<tendon>`, `<sensor>`, `<contact>`,
//! `<keyframe>`.

#![warn(missing_docs)]

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

mod assets;
mod attrs;
mod defaults;
mod orientation;
mod parser;

pub use assets::{HFieldAsset, MaterialAsset, MeshAsset, MeshData, TextureAsset};
pub use defaults::{ClassDefaults, DefaultsManager, MAIN_CLASS};
pub use parser::{MjcfLoader, SiteElement, UnsupportedFeature};

use thiserror::Error;

/// Anything that can go wrong loading an MJCF model.
#[derive(Debug, Error)]
pub enum MjcfError {
    /// The document is not well-formed XML.
    #[error("XML parse error: {0}")]
    XmlError(#[from] quick_xml::Error),

    /// An element's attribute list could not be read.
    #[error("XML attribute error in <{element}>: {source}")]
    AttrError {
        /// The element whose attributes failed to parse.
        element: String,
        /// The underlying quick-xml error.
        #[source]
        source: quick_xml::events::attributes::AttrError,
    },

    /// The model file could not be read.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// The document is valid XML but not a valid MJCF model.
    #[error("Invalid MJCF: {0}")]
    InvalidMjcf(String),

    /// An attribute was present but could not be interpreted.
    #[error("<{element}> attribute '{attribute}' has invalid value {value:?}: {reason}")]
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

    /// A required attribute was absent.
    #[error("<{element}> is missing required attribute '{attribute}'")]
    MissingAttribute {
        /// The element missing the attribute.
        element: String,
        /// The attribute that was required.
        attribute: String,
    },

    /// An element named a `<default>` class that was never declared.
    #[error("<{element}> references undefined default class '{class}'")]
    UnknownClass {
        /// The element naming the class.
        element: String,
        /// The undefined class name.
        class: String,
    },

    /// A valid MJCF construct this parser does not implement.
    #[error("Unsupported feature: {0}")]
    Unsupported(String),
}

impl MjcfError {
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
