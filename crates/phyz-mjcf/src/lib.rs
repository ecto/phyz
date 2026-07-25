//! MuJoCo MJCF XML parser for phyz physics engine.
//!
//! Supports loading models from MJCF XML format and converting them to phyz Model.
//!
//! # Supported subset
//!
//! - `<compiler>`: `angle`, `coordinate`, `eulerseq`, `meshdir`, `assetdir`, `inertiafromgeom`
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

mod assets;
mod attrs;
mod defaults;
mod orientation;
mod parser;

pub use assets::{HFieldAsset, MaterialAsset, MeshAsset, TextureAsset};
pub use defaults::{ClassDefaults, DefaultsManager};
pub use parser::{MjcfLoader, SiteElement};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MjcfError {
    #[error("XML parse error: {0}")]
    XmlError(#[from] quick_xml::Error),

    #[error("XML attribute error in <{element}>: {source}")]
    AttrError {
        element: String,
        #[source]
        source: quick_xml::events::attributes::AttrError,
    },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid MJCF: {0}")]
    InvalidMjcf(String),

    #[error("<{element}> attribute '{attribute}' has invalid value {value:?}: {reason}")]
    InvalidAttribute {
        element: String,
        attribute: String,
        value: String,
        reason: String,
    },

    #[error("<{element}> is missing required attribute '{attribute}'")]
    MissingAttribute { element: String, attribute: String },

    #[error("<{element}> references undefined default class '{class}'")]
    UnknownClass { element: String, class: String },

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

pub type Result<T> = std::result::Result<T, MjcfError>;
