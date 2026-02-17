//! MuJoCo defaults system for element attributes.

use std::collections::HashMap;

/// Default values for MJCF elements.
#[derive(Debug, Clone, Default)]
pub struct ElementDefaults {
    pub joint: JointDefaults,
    pub geom: GeomDefaults,
}

#[derive(Debug, Clone)]
pub struct JointDefaults {
    pub damping: f64,
    pub limited: bool,
    pub range: Option<[f64; 2]>,
}

impl Default for JointDefaults {
    fn default() -> Self {
        Self {
            damping: 0.0,
            limited: false,
            range: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeomDefaults {
    pub density: f64,
    pub friction: [f64; 3],
}

impl Default for GeomDefaults {
    fn default() -> Self {
        Self {
            density: 1000.0,
            friction: [1.0, 0.005, 0.0001],
        }
    }
}

/// Manages default values for different element classes.
#[derive(Debug, Default)]
pub struct DefaultsManager {
    defaults: HashMap<String, ElementDefaults>,
}

impl DefaultsManager {
    pub fn new() -> Self {
        let mut manager = Self {
            defaults: HashMap::new(),
        };
        // Add the "main" default class
        manager
            .defaults
            .insert("main".to_string(), ElementDefaults::default());
        manager
    }

    #[allow(dead_code)]
    pub fn add_class(&mut self, name: String, defaults: ElementDefaults) {
        self.defaults.insert(name, defaults);
    }

    #[allow(dead_code)]
    pub fn get(&self, class: &str) -> ElementDefaults {
        self.defaults.get(class).cloned().unwrap_or_default()
    }

    #[allow(dead_code)]
    pub fn get_or_main(&self, class: Option<&str>) -> ElementDefaults {
        class
            .and_then(|c| self.defaults.get(c).cloned())
            .unwrap_or_else(|| self.get("main"))
    }
}
