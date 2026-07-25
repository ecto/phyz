//! Attribute reading with default-class fallback.

use crate::defaults::AttrMap;
use crate::{MjcfError, Result};
use phyz_math::Vec3;
use quick_xml::events::BytesStart;

/// An element's own attributes layered over its resolved class defaults.
///
/// Lookup order is element → class chain, which is MuJoCo's rule. Centralising
/// it here is what keeps `<default>` support from having to be threaded through
/// every `parse_*` function by hand.
#[derive(Debug, Clone, Default)]
pub struct Attrs {
    own: AttrMap,
    defaults: AttrMap,
}

impl Attrs {
    /// Read an element's attributes and layer them over `defaults`.
    pub fn read(e: &BytesStart, defaults: AttrMap) -> Result<Self> {
        let mut own = AttrMap::new();
        for attr in e.attributes() {
            let attr = attr.map_err(|e| MjcfError::InvalidMjcf(e.to_string()))?;
            own.insert(
                String::from_utf8_lossy(attr.key.as_ref()).to_string(),
                String::from_utf8_lossy(&attr.value).to_string(),
            );
        }
        Ok(Self { own, defaults })
    }

    /// The raw attributes written on the element itself, ignoring defaults.
    /// Used when recording `<default>` blocks, which must not inherit yet.
    pub fn into_own(self) -> AttrMap {
        self.own
    }

    /// Raw string lookup.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.own
            .get(key)
            .or_else(|| self.defaults.get(key))
            .map(String::as_str)
    }

    /// String with fallback.
    pub fn str_or(&self, key: &str, fallback: &str) -> String {
        self.get(key).unwrap_or(fallback).to_string()
    }

    /// Scalar float.
    pub fn f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.trim().parse().ok())
    }

    /// Scalar float with fallback.
    pub fn f64_or(&self, key: &str, fallback: f64) -> f64 {
        self.f64(key).unwrap_or(fallback)
    }

    /// Whitespace-separated float list.
    pub fn floats(&self, key: &str) -> Option<Vec<f64>> {
        self.get(key).map(|v| {
            v.split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect()
        })
    }

    /// Exactly-3 float list as a vector.
    pub fn vec3(&self, key: &str) -> Option<Vec3> {
        match self.floats(key) {
            Some(v) if v.len() == 3 => Some(Vec3::new(v[0], v[1], v[2])),
            _ => None,
        }
    }

    /// Exactly-3 float list with fallback.
    pub fn vec3_or(&self, key: &str, fallback: Vec3) -> Vec3 {
        self.vec3(key).unwrap_or(fallback)
    }

    /// Exactly-2 float list, e.g. `range` / `ctrlrange`.
    pub fn range(&self, key: &str) -> Option<[f64; 2]> {
        match self.floats(key) {
            Some(v) if v.len() == 2 => Some([v[0], v[1]]),
            _ => None,
        }
    }

    /// MJCF boolean (`"true"` / `"false"` / `"auto"`).
    pub fn bool(&self, key: &str) -> Option<bool> {
        match self.get(key) {
            Some("true") | Some("1") => Some(true),
            Some("false") | Some("0") => Some(false),
            _ => None,
        }
    }
}
