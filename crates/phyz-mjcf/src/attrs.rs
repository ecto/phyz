//! Attribute bag with fallible, well-described parsing.
//!
//! Every accessor names the element and attribute it failed on, so malformed MJCF
//! produces a [`MjcfError`] instead of a panic or a silently-substituted zero.

use crate::{MjcfError, Result};
use phyz_math::Vec3;
use quick_xml::events::BytesStart;
use std::collections::HashMap;

/// The attributes of a single XML element, after default-class merging.
#[derive(Debug, Clone, Default)]
pub struct Attrs {
    element: String,
    map: HashMap<String, String>,
}

impl Attrs {
    /// Collect the attributes of an XML start/empty event.
    pub fn from_event(element: &str, e: &BytesStart) -> Result<Self> {
        let mut map = HashMap::new();
        for attr in e.attributes() {
            let attr = attr.map_err(|source| MjcfError::AttrError {
                element: element.to_string(),
                source,
            })?;
            let key = String::from_utf8_lossy(attr.key.as_ref()).to_string();
            let value = attr
                .decode_and_unescape_value(quick_xml::Decoder {})
                .map_err(MjcfError::XmlError)?
                .to_string();
            map.insert(key, value);
        }
        Ok(Self {
            element: element.to_string(),
            map,
        })
    }

    /// Build directly from a map (test construction helper).
    #[cfg(test)]
    pub fn from_map(element: &str, map: HashMap<String, String>) -> Self {
        Self {
            element: element.to_string(),
            map,
        }
    }

    pub fn element(&self) -> &str {
        &self.element
    }

    /// Fill in any attribute not explicitly set on this element from `defaults`.
    pub fn merge_defaults(&mut self, defaults: &HashMap<String, String>) {
        for (k, v) in defaults {
            self.map.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }

    pub fn raw(&self) -> &HashMap<String, String> {
        &self.map
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.map.get(key).map(|s| s.as_str())
    }

    pub fn has(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    pub fn string(&self, key: &str) -> Option<String> {
        self.map.get(key).cloned()
    }

    pub fn required(&self, key: &str) -> Result<&str> {
        self.get(key).ok_or_else(|| MjcfError::MissingAttribute {
            element: self.element.clone(),
            attribute: key.to_string(),
        })
    }

    /// Parse a scalar float, if present.
    pub fn f64(&self, key: &str) -> Result<Option<f64>> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v.trim().parse::<f64>().map(Some).map_err(|e| {
                MjcfError::invalid_attr(&self.element, key, v, format!("expected a number ({e})"))
            }),
        }
    }

    pub fn f64_or(&self, key: &str, default: f64) -> Result<f64> {
        Ok(self.f64(key)?.unwrap_or(default))
    }

    /// Parse a whitespace-separated list of floats, if present.
    pub fn floats(&self, key: &str) -> Result<Option<Vec<f64>>> {
        let Some(v) = self.get(key) else {
            return Ok(None);
        };
        let mut out = Vec::new();
        for tok in v.split_whitespace() {
            let n = tok.parse::<f64>().map_err(|e| {
                MjcfError::invalid_attr(
                    &self.element,
                    key,
                    v,
                    format!("'{tok}' is not a number ({e})"),
                )
            })?;
            out.push(n);
        }
        Ok(Some(out))
    }

    /// Parse exactly `N` floats, if present.
    pub fn fixed<const N: usize>(&self, key: &str) -> Result<Option<[f64; N]>> {
        let Some(vals) = self.floats(key)? else {
            return Ok(None);
        };
        if vals.len() != N {
            let raw = self.get(key).unwrap_or_default();
            return Err(MjcfError::invalid_attr(
                &self.element,
                key,
                raw,
                format!("expected {N} numbers, found {}", vals.len()),
            ));
        }
        let mut out = [0.0; N];
        out.copy_from_slice(&vals);
        Ok(Some(out))
    }

    pub fn vec3(&self, key: &str) -> Result<Option<Vec3>> {
        Ok(self.fixed::<3>(key)?.map(|v| Vec3::new(v[0], v[1], v[2])))
    }

    pub fn vec3_or(&self, key: &str, default: Vec3) -> Result<Vec3> {
        Ok(self.vec3(key)?.unwrap_or(default))
    }

    /// Parse an MJCF boolean-ish attribute (`true`/`false`/`auto`).
    /// `auto` yields `None`, matching MuJoCo's tri-state semantics.
    pub fn tri_bool(&self, key: &str) -> Result<Option<Option<bool>>> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => match v.trim() {
                "true" | "1" => Ok(Some(Some(true))),
                "false" | "0" => Ok(Some(Some(false))),
                "auto" => Ok(Some(None)),
                other => Err(MjcfError::invalid_attr(
                    &self.element,
                    key,
                    other,
                    "expected 'true', 'false', or 'auto'",
                )),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs(pairs: &[(&str, &str)]) -> Attrs {
        Attrs::from_map(
            "geom",
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        )
    }

    #[test]
    fn bad_number_reports_element_and_attribute() {
        let a = attrs(&[("size", "0.1 bogus")]);
        let err = a.floats("size").unwrap_err().to_string();
        assert!(err.contains("geom"), "{err}");
        assert!(err.contains("size"), "{err}");
        assert!(err.contains("bogus"), "{err}");
    }

    #[test]
    fn wrong_arity_is_an_error() {
        let a = attrs(&[("pos", "1 2")]);
        let err = a.fixed::<3>("pos").unwrap_err().to_string();
        assert!(err.contains("expected 3 numbers, found 2"), "{err}");
    }

    #[test]
    fn defaults_do_not_override_explicit_values() {
        let mut a = attrs(&[("size", "0.1")]);
        a.merge_defaults(&HashMap::from([
            ("size".to_string(), "0.9".to_string()),
            ("type".to_string(), "capsule".to_string()),
        ]));
        assert_eq!(a.get("size"), Some("0.1"));
        assert_eq!(a.get("type"), Some("capsule"));
    }
}
