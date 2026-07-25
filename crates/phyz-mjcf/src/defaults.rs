//! MJCF `<default>` class system.
//!
//! MuJoCo defaults are per-element-type attribute bags organised into a tree of
//! named classes. A nested `<default class="child">` inherits every attribute of
//! its enclosing class and may override any of them. Elements select a class via
//! their own `class` attribute, or inherit one from the nearest enclosing body's
//! `childclass`; otherwise the root class (`main`) applies.
//!
//! Defaults are stored as raw attribute strings rather than typed structs, so a
//! defaulted attribute goes through exactly the same parsing path as an explicit
//! one.

use crate::{MjcfError, Result};
use std::collections::HashMap;

/// The root default class name, used when an element names no class.
pub const MAIN_CLASS: &str = "main";

/// Attribute defaults for every element type within one class.
#[derive(Debug, Clone, Default)]
pub struct ClassDefaults {
    /// Enclosing class this one refines, if any.
    parent: Option<String>,
    /// element tag -> attribute -> value
    attrs: HashMap<String, HashMap<String, String>>,
}

impl ClassDefaults {
    /// Attributes this class sets directly (no inheritance) for `tag`.
    pub fn own_attrs(&self, tag: &str) -> Option<&HashMap<String, String>> {
        self.attrs.get(tag)
    }

    pub fn parent(&self) -> Option<&str> {
        self.parent.as_deref()
    }
}

/// Holds every `<default>` class in a model and resolves inheritance.
#[derive(Debug)]
pub struct DefaultsManager {
    classes: HashMap<String, ClassDefaults>,
    /// Memoised resolution of (class, tag).
    cache: std::cell::RefCell<HashMap<(String, String), HashMap<String, String>>>,
}

impl Default for DefaultsManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultsManager {
    pub fn new() -> Self {
        let mut classes = HashMap::new();
        classes.insert(MAIN_CLASS.to_string(), ClassDefaults::default());
        Self {
            classes,
            cache: std::cell::RefCell::new(HashMap::new()),
        }
    }

    /// Declare a class nested inside `parent` (the root class has `parent = None`).
    pub fn declare_class(&mut self, name: &str, parent: Option<&str>) {
        let entry = self.classes.entry(name.to_string()).or_default();
        if entry.parent.is_none() && parent.map(|p| p != name).unwrap_or(false) {
            entry.parent = parent.map(str::to_string);
        }
        self.cache.borrow_mut().clear();
    }

    /// Record the attributes an element declares inside a `<default>` block.
    pub fn set_element_defaults(
        &mut self,
        class: &str,
        tag: &str,
        attrs: &HashMap<String, String>,
    ) {
        let entry = self.classes.entry(class.to_string()).or_default();
        let slot = entry.attrs.entry(tag.to_string()).or_default();
        for (k, v) in attrs {
            if k == "class" {
                continue;
            }
            slot.insert(k.clone(), v.clone());
        }
        self.cache.borrow_mut().clear();
    }

    pub fn has_class(&self, name: &str) -> bool {
        self.classes.contains_key(name)
    }

    pub fn class(&self, name: &str) -> Option<&ClassDefaults> {
        self.classes.get(name)
    }

    pub fn class_names(&self) -> impl Iterator<Item = &str> {
        self.classes.keys().map(|s| s.as_str())
    }

    /// Fully resolved defaults for `tag` in `class`: the class chain walked from the
    /// root down, with nearer classes overriding further ones.
    ///
    /// Returns an error if `class` was never declared, since silently ignoring a
    /// typo'd class name is exactly the kind of quiet wrongness this module exists
    /// to prevent.
    pub fn resolve(&self, class: &str, tag: &str) -> Result<HashMap<String, String>> {
        if !self.classes.contains_key(class) {
            return Err(MjcfError::UnknownClass {
                element: tag.to_string(),
                class: class.to_string(),
            });
        }
        let key = (class.to_string(), tag.to_string());
        if let Some(hit) = self.cache.borrow().get(&key) {
            return Ok(hit.clone());
        }

        // Walk up to the root, then apply from root downward.
        let mut chain = Vec::new();
        let mut cursor = Some(class.to_string());
        while let Some(name) = cursor {
            let Some(c) = self.classes.get(&name) else {
                break;
            };
            cursor = c.parent.clone();
            chain.push(c);
            if chain.len() > 64 {
                break; // cycle guard; declare_class already refuses self-parents
            }
        }

        let mut merged: HashMap<String, String> = HashMap::new();
        for class_defaults in chain.iter().rev() {
            if let Some(own) = class_defaults.attrs.get(tag) {
                for (k, v) in own {
                    merged.insert(k.clone(), v.clone());
                }
            }
        }

        self.cache.borrow_mut().insert(key, merged.clone());
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn child_class_inherits_and_overrides() {
        let mut d = DefaultsManager::new();
        d.set_element_defaults(
            MAIN_CLASS,
            "geom",
            &map(&[("density", "500"), ("type", "capsule")]),
        );
        d.declare_class("leg", Some(MAIN_CLASS));
        d.set_element_defaults("leg", "geom", &map(&[("type", "sphere")]));

        let resolved = d.resolve("leg", "geom").unwrap();
        assert_eq!(resolved.get("type").map(String::as_str), Some("sphere"));
        assert_eq!(resolved.get("density").map(String::as_str), Some("500"));
    }

    #[test]
    fn grandchild_walks_the_whole_chain() {
        let mut d = DefaultsManager::new();
        d.set_element_defaults(MAIN_CLASS, "joint", &map(&[("damping", "1")]));
        d.declare_class("a", Some(MAIN_CLASS));
        d.set_element_defaults("a", "joint", &map(&[("armature", "0.1")]));
        d.declare_class("b", Some("a"));
        d.set_element_defaults("b", "joint", &map(&[("damping", "5")]));

        let r = d.resolve("b", "joint").unwrap();
        assert_eq!(r.get("damping").map(String::as_str), Some("5"));
        assert_eq!(r.get("armature").map(String::as_str), Some("0.1"));
    }

    #[test]
    fn unknown_class_is_an_error() {
        let d = DefaultsManager::new();
        assert!(d.resolve("nope", "geom").is_err());
    }
}
