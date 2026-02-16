//! Contact material properties.

/// Material properties for contact interactions.
#[derive(Debug, Clone)]
pub struct ContactMaterial {
    /// Contact stiffness (N/m).
    pub stiffness: f64,
    /// Contact damping (N·s/m).
    pub damping: f64,
    /// Coefficient of friction (dimensionless).
    pub friction: f64,
    /// Coefficient of restitution (0 = inelastic, 1 = elastic).
    pub bounce: f64,
    /// Constraint force mixing (for numerical stability).
    pub soft_cfm: f64,
    /// Error reduction parameter (for constraint drift correction).
    pub soft_erp: f64,
}

impl Default for ContactMaterial {
    fn default() -> Self {
        Self {
            stiffness: 10000.0,
            damping: 100.0,
            friction: 0.5,
            bounce: 0.0,
            soft_cfm: 0.0001,
            soft_erp: 0.2,
        }
    }
}

impl ContactMaterial {
    /// Create a new contact material with custom parameters.
    pub fn new(stiffness: f64, damping: f64, friction: f64, bounce: f64) -> Self {
        Self {
            stiffness,
            damping,
            friction,
            bounce,
            soft_cfm: 0.0001,
            soft_erp: 0.2,
        }
    }

    /// Create a bouncy material (high restitution).
    pub fn bouncy() -> Self {
        Self {
            stiffness: 10000.0,
            damping: 50.0,
            friction: 0.3,
            bounce: 0.8,
            soft_cfm: 0.0001,
            soft_erp: 0.2,
        }
    }

    /// Create a soft material (low stiffness).
    pub fn soft() -> Self {
        Self {
            stiffness: 1000.0,
            damping: 200.0,
            friction: 0.7,
            bounce: 0.1,
            soft_cfm: 0.001,
            soft_erp: 0.2,
        }
    }

    /// Create a rigid material (high stiffness).
    pub fn rigid() -> Self {
        Self {
            stiffness: 50000.0,
            damping: 100.0,
            friction: 0.5,
            bounce: 0.0,
            soft_cfm: 0.00001,
            soft_erp: 0.2,
        }
    }
}
