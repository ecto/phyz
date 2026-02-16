//! Main .tau format schema and loader.

use crate::coupling::Coupling;
use crate::domain::Domain;
use crate::error::{Result, TauFormatError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Top-level .tau format specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TauSpec {
    /// Format version.
    pub version: String,
    /// Model name.
    pub name: String,
    /// Optional description.
    #[serde(default)]
    pub description: String,
    /// World configuration.
    pub world: WorldConfig,
    /// Physics domains.
    pub domains: HashMap<String, Domain>,
    /// Domain couplings.
    #[serde(default)]
    pub couplings: Vec<Coupling>,
    /// Parameter specifications.
    #[serde(default)]
    pub parameters: HashMap<String, ParameterSpec>,
    /// Import sources.
    #[serde(default)]
    pub importers: Vec<ImporterSpec>,
}

/// World-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldConfig {
    /// Gravity vector [x, y, z].
    pub gravity: [f64; 3],
    /// Default time step.
    pub dt: f64,
    /// Default contact material properties.
    #[serde(default)]
    pub default_contact_material: ContactMaterial,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            gravity: [0.0, 0.0, -9.81],
            dt: 0.001,
            default_contact_material: ContactMaterial::default(),
        }
    }
}

/// Contact material properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactMaterial {
    /// Contact stiffness.
    #[serde(default = "default_stiffness")]
    pub stiffness: f64,
    /// Contact damping.
    #[serde(default = "default_damping")]
    pub damping: f64,
    /// Friction coefficient.
    #[serde(default = "default_friction")]
    pub friction: f64,
}

fn default_stiffness() -> f64 {
    10000.0
}

fn default_damping() -> f64 {
    100.0
}

fn default_friction() -> f64 {
    0.5
}

impl Default for ContactMaterial {
    fn default() -> Self {
        Self {
            stiffness: default_stiffness(),
            damping: default_damping(),
            friction: default_friction(),
        }
    }
}

/// Parameter specification with optional uncertainty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSpec {
    /// Parameter type.
    #[serde(rename = "type")]
    pub param_type: ParameterType,
    /// Base value(s).
    pub value: serde_json::Value,
    /// Uncertainty (for probabilistic parameters).
    #[serde(default)]
    pub uncertainty: Option<f64>,
}

/// Parameter type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ParameterType {
    /// Scalar parameter.
    Scalar,
    /// Vector parameter.
    Vector,
    /// Distribution parameter.
    Distribution,
}

/// Import source specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImporterSpec {
    /// Source format.
    pub format: ImportFormat,
    /// Source file path.
    pub source: String,
}

/// Supported import formats.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ImportFormat {
    /// MuJoCo MJCF XML.
    Mjcf,
    /// URDF XML.
    Urdf,
    /// USD (Universal Scene Description).
    Usd,
    /// SDF (Simulation Description Format).
    Sdf,
}

/// Load a .tau model from file.
pub fn load_tau_model(path: &str) -> Result<TauSpec> {
    let json = std::fs::read_to_string(path)?;
    let spec: TauSpec = serde_json::from_str(&json)?;

    // Validate version
    if spec.version.is_empty() {
        return Err(TauFormatError::MissingField("version".to_string()));
    }

    Ok(spec)
}

/// Save a .tau model to file.
pub fn save_tau_model(path: &str, spec: &TauSpec) -> Result<()> {
    let json = serde_json::to_string_pretty(spec)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Export a TauSpec to JSON string.
pub fn export_tau(spec: &TauSpec) -> Result<String> {
    Ok(serde_json::to_string_pretty(spec)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::DomainType;

    #[test]
    fn test_world_config_serialization() {
        let config = WorldConfig {
            gravity: [0.0, 0.0, -9.81],
            dt: 0.001,
            default_contact_material: ContactMaterial::default(),
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: WorldConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.gravity, [0.0, 0.0, -9.81]);
        assert_eq!(parsed.dt, 0.001);
    }

    #[test]
    fn test_tau_spec_serialization() {
        let mut domains = HashMap::new();
        let mut rigid_config = HashMap::new();
        rigid_config.insert(
            "bodies".to_string(),
            serde_json::json!([{
                "name": "link",
                "mass": 1.0,
                "inertia": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                "center_of_mass": [0.0, 0.0, 0.0]
            }]),
        );
        rigid_config.insert("joints".to_string(), serde_json::json!([]));

        domains.insert(
            "rigid_body".to_string(),
            Domain {
                domain_type: DomainType::RigidBodyDynamics,
                config: rigid_config,
            },
        );

        let spec = TauSpec {
            version: "1.0".to_string(),
            name: "test-model".to_string(),
            description: "Test model".to_string(),
            world: WorldConfig::default(),
            domains,
            couplings: vec![],
            parameters: HashMap::new(),
            importers: vec![],
        };

        let json = serde_json::to_string_pretty(&spec).unwrap();
        let parsed: TauSpec = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, "test-model");
        assert_eq!(parsed.version, "1.0");
    }

    #[test]
    fn test_parameter_spec_serialization() {
        let param = ParameterSpec {
            param_type: ParameterType::Scalar,
            value: serde_json::json!(1.0),
            uncertainty: Some(0.1),
        };

        let json = serde_json::to_string(&param).unwrap();
        let parsed: ParameterSpec = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.param_type, ParameterType::Scalar);
        assert_eq!(parsed.uncertainty, Some(0.1));
    }

    #[test]
    fn test_importer_spec_serialization() {
        let importer = ImporterSpec {
            format: ImportFormat::Mjcf,
            source: "model.xml".to_string(),
        };

        let json = serde_json::to_string(&importer).unwrap();
        let parsed: ImporterSpec = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.format, ImportFormat::Mjcf);
        assert_eq!(parsed.source, "model.xml");
    }
}
