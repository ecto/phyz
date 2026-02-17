//! Coupling definitions between physics domains.

use serde::{Deserialize, Serialize};

/// Coupling between two physics domains.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coupling {
    /// Coupling name.
    pub name: String,
    /// Source domain name.
    pub source: String,
    /// Target domain name.
    pub target: String,
    /// Type of coupling.
    #[serde(rename = "type")]
    pub coupling_type: CouplingType,
    /// Force transfer mechanism.
    pub force_transfer: ForceTransfer,
    /// Coupling strength/stiffness.
    #[serde(default = "default_coupling_strength")]
    pub strength: f64,
}

fn default_coupling_strength() -> f64 {
    1.0
}

/// Type of coupling between domains.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum CouplingType {
    /// One-way coupling (source affects target).
    OneWay,
    /// Two-way coupling (mutual interaction).
    TwoWay,
    /// Overlap region coupling.
    OverlapRegion,
    /// Interface boundary coupling.
    Interface,
}

/// Mechanism for transferring forces between domains.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ForceTransfer {
    /// Direct force transfer.
    Direct,
    /// Lorentz force (EM coupling).
    Lorentz,
    /// Contact forces.
    Contact,
    /// Penalty-based coupling.
    Penalty,
    /// Lagrange multiplier constraints.
    Lagrange,
}

/// Region definition for overlap coupling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapRegion {
    /// Region type.
    #[serde(rename = "type")]
    pub region_type: RegionType,
    /// Region parameters (e.g., min/max for AABB).
    #[serde(flatten)]
    pub params: std::collections::HashMap<String, serde_json::Value>,
}

/// Type of overlap region.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RegionType {
    /// Axis-aligned bounding box.
    Aabb,
    /// Sphere.
    Sphere,
    /// Cylinder.
    Cylinder,
    /// Custom mesh.
    Mesh,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupling_serialization() {
        let coupling = Coupling {
            name: "rigid-em".to_string(),
            source: "rigid_body".to_string(),
            target: "em".to_string(),
            coupling_type: CouplingType::TwoWay,
            force_transfer: ForceTransfer::Lorentz,
            strength: 1.0,
        };

        let json = serde_json::to_string(&coupling).unwrap();
        let parsed: Coupling = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, "rigid-em");
        assert_eq!(parsed.coupling_type, CouplingType::TwoWay);
        assert_eq!(parsed.force_transfer, ForceTransfer::Lorentz);
    }

    #[test]
    fn test_coupling_type_serialization() {
        let ct = CouplingType::OverlapRegion;
        let json = serde_json::to_string(&ct).unwrap();
        assert_eq!(json, "\"overlap-region\"");
    }

    #[test]
    fn test_force_transfer_serialization() {
        let ft = ForceTransfer::Penalty;
        let json = serde_json::to_string(&ft).unwrap();
        assert_eq!(json, "\"penalty\"");
    }
}
