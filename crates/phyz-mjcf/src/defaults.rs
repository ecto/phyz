//! MuJoCo `<default>` class tree.
//!
//! MJCF defaults are *per element tag, per class, with class inheritance*:
//!
//! ```xml
//! <default>
//!   <geom density="1000"/>
//!   <default class="leg">
//!     <geom density="500" type="capsule"/>
//!     <joint damping="1" limited="true"/>
//!   </default>
//! </default>
//! ```
//!
//! A `<geom class="leg">` inherits `density=500`, `type=capsule` from `leg` and
//! anything else from the unnamed root class. Attributes on the element itself
//! win over everything.
//!
//! Rather than typing each defaulted attribute, this stores raw attribute
//! strings and resolves them at lookup. New attributes then need no change
//! here — only in the parser that reads them.

use std::collections::HashMap;

/// Attribute name → raw string value.
pub type AttrMap = HashMap<String, String>;

/// One `<default class="...">` node.
#[derive(Debug, Clone, Default)]
pub struct DefaultClass {
    /// Enclosing class, or `None` for the root.
    pub parent: Option<String>,
    /// Element tag (`geom`, `joint`, `motor`, …) → defaulted attributes.
    pub by_tag: HashMap<String, AttrMap>,
}

/// The whole `<default>` tree.
///
/// The root class is stored under the empty string, matching MuJoCo's unnamed
/// top-level `<default>`.
#[derive(Debug, Clone)]
pub struct DefaultsManager {
    classes: HashMap<String, DefaultClass>,
}

/// The name of the root (unnamed) default class.
pub const ROOT_CLASS: &str = "";

impl Default for DefaultsManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultsManager {
    /// A tree containing only the empty root class.
    pub fn new() -> Self {
        let mut classes = HashMap::new();
        classes.insert(ROOT_CLASS.to_string(), DefaultClass::default());
        Self { classes }
    }

    /// Declare a class nested inside `parent`.
    pub fn declare(&mut self, name: &str, parent: Option<&str>) {
        self.classes
            .entry(name.to_string())
            .or_default()
            .parent
            .clone_from(&parent.map(str::to_string));
    }

    /// Record defaulted attributes for `tag` within `class`.
    pub fn set(&mut self, class: &str, tag: &str, attrs: AttrMap) {
        let entry = self.classes.entry(class.to_string()).or_default();
        let slot = entry.by_tag.entry(tag.to_string()).or_default();
        slot.extend(attrs);
    }

    /// Whether a class has been declared.
    pub fn has(&self, class: &str) -> bool {
        self.classes.contains_key(class)
    }

    /// Resolve the defaults for `tag` in `class`, walking up the parent chain.
    /// Nearer classes override further ones.
    pub fn resolve(&self, tag: &str, class: Option<&str>) -> AttrMap {
        // Collect the chain leaf → root, then apply root-first so the leaf wins.
        let mut chain = Vec::new();
        let mut cursor = class.map(str::to_string).or_else(|| Some(ROOT_CLASS.into()));
        let mut guard = 0;
        while let Some(name) = cursor {
            let Some(node) = self.classes.get(&name) else {
                break;
            };
            chain.push(node);
            cursor = node.parent.clone();
            guard += 1;
            if guard > 64 {
                break; // malformed cyclic default tree; fail soft
            }
        }

        let mut out = AttrMap::new();
        for node in chain.iter().rev() {
            if let Some(attrs) = node.by_tag.get(tag) {
                for (k, v) in attrs {
                    out.insert(k.clone(), v.clone());
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs(pairs: &[(&str, &str)]) -> AttrMap {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn child_class_overrides_root() {
        let mut d = DefaultsManager::new();
        d.set(ROOT_CLASS, "geom", attrs(&[("density", "1000"), ("type", "sphere")]));
        d.declare("leg", Some(ROOT_CLASS));
        d.set("leg", "geom", attrs(&[("density", "500")]));

        let r = d.resolve("geom", Some("leg"));
        assert_eq!(r["density"], "500");
        assert_eq!(r["type"], "sphere", "unshadowed root attrs must survive");
    }

    #[test]
    fn unknown_class_falls_back_to_nothing() {
        let d = DefaultsManager::new();
        assert!(d.resolve("geom", Some("nope")).is_empty());
    }

    #[test]
    fn nested_classes_chain() {
        let mut d = DefaultsManager::new();
        d.set(ROOT_CLASS, "joint", attrs(&[("damping", "0.1")]));
        d.declare("a", Some(ROOT_CLASS));
        d.set("a", "joint", attrs(&[("armature", "1")]));
        d.declare("b", Some("a"));
        d.set("b", "joint", attrs(&[("damping", "5")]));

        let r = d.resolve("joint", Some("b"));
        assert_eq!(r["damping"], "5");
        assert_eq!(r["armature"], "1");
    }
}
