//! Lorentz force and electromagnetic coupling to rigid bodies.

use tau_math::Vec3;

/// Compute Lorentz force on a charged particle in electromagnetic fields.
///
/// F = q (E + v × B)
///
/// # Arguments
/// * `charge` - Electric charge (C)
/// * `position` - Position of the charge (for field lookup, not used directly)
/// * `velocity` - Velocity of the charge (m/s)
/// * `e_field` - Electric field at the position (V/m)
/// * `b_field` - Magnetic field at the position (T)
///
/// # Returns
/// Force vector (N)
pub fn lorentz_force(
    charge: f64,
    _position: Vec3,
    velocity: Vec3,
    e_field: &Vec3,
    b_field: &Vec3,
) -> Vec3 {
    charge * (e_field + velocity.cross(b_field))
}

/// Compute magnetic torque on a magnetic dipole.
///
/// τ = μ × B
///
/// # Arguments
/// * `dipole_moment` - Magnetic dipole moment (A·m²)
/// * `b_field` - Magnetic field (T)
///
/// # Returns
/// Torque vector (N·m)
pub fn magnetic_torque(dipole_moment: Vec3, b_field: &Vec3) -> Vec3 {
    dipole_moment.cross(b_field)
}

/// Compute electric dipole force in non-uniform field.
///
/// F = (p · ∇) E
///
/// For a dipole p in a field with gradient ∇E, approximate as:
/// F ≈ p_x * dE/dx + p_y * dE/dy + p_z * dE/dz
///
/// # Arguments
/// * `dipole_moment` - Electric dipole moment (C·m)
/// * `field_gradient` - Gradient of electric field (V/m²), stored as 3x3 matrix
///
/// # Returns
/// Force vector (N)
pub fn electric_dipole_force(dipole_moment: Vec3, field_gradient: &[Vec3; 3]) -> Vec3 {
    // F_i = p_j * dE_i/dx_j (Einstein summation)
    Vec3::new(
        dipole_moment.dot(&field_gradient[0]),
        dipole_moment.dot(&field_gradient[1]),
        dipole_moment.dot(&field_gradient[2]),
    )
}

/// Compute radiation pressure force on a conducting surface.
///
/// Radiation pressure: P = ε₀ E² / 2
/// Force per unit area: dF/dA = P * n̂
///
/// # Arguments
/// * `e_field` - Electric field at surface (V/m)
/// * `area` - Surface area (m²)
/// * `normal` - Surface normal (unit vector)
///
/// # Returns
/// Force vector (N)
pub fn radiation_pressure_force(e_field: &Vec3, area: f64, normal: Vec3) -> Vec3 {
    const EPSILON_0: f64 = 8.854187817e-12; // F/m
    let e_squared = e_field.norm_squared();
    let pressure = EPSILON_0 * e_squared / 2.0;
    pressure * area * normal
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lorentz_force_electric_only() {
        let charge = 1e-6; // 1 μC
        let pos = Vec3::zeros();
        let vel = Vec3::zeros();
        let e_field = Vec3::new(1000.0, 0.0, 0.0); // 1 kV/m
        let b_field = Vec3::zeros();

        let force = lorentz_force(charge, pos, vel, &e_field, &b_field);

        // F = q E = 1e-6 * 1000 = 1e-3 N
        assert_relative_eq!(force.x, 1e-3, epsilon = 1e-10);
        assert_relative_eq!(force.y, 0.0);
        assert_relative_eq!(force.z, 0.0);
    }

    #[test]
    fn test_lorentz_force_magnetic_only() {
        let charge = 1e-6; // 1 μC
        let pos = Vec3::zeros();
        let vel = Vec3::new(100.0, 0.0, 0.0); // 100 m/s in x
        let e_field = Vec3::zeros();
        let b_field = Vec3::new(0.0, 0.0, 1.0); // 1 T in z

        let force = lorentz_force(charge, pos, vel, &e_field, &b_field);

        // F = q (v × B) = 1e-6 * (100, 0, 0) × (0, 0, 1)
        //   = 1e-6 * (0, -100, 0) = (0, -1e-4, 0)
        assert_relative_eq!(force.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(force.y, -1e-4, epsilon = 1e-10);
        assert_relative_eq!(force.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lorentz_force_combined() {
        let charge = 1e-6;
        let pos = Vec3::zeros();
        let vel = Vec3::new(100.0, 0.0, 0.0);
        let e_field = Vec3::new(1000.0, 0.0, 0.0);
        let b_field = Vec3::new(0.0, 0.0, 1.0);

        let force = lorentz_force(charge, pos, vel, &e_field, &b_field);

        // F = q (E + v × B) = 1e-6 * (1000, 0, 0) + 1e-6 * (0, -100, 0)
        //   = (1e-3, -1e-4, 0)
        assert_relative_eq!(force.x, 1e-3, epsilon = 1e-10);
        assert_relative_eq!(force.y, -1e-4, epsilon = 1e-10);
        assert_relative_eq!(force.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_magnetic_torque() {
        // Dipole aligned with x-axis in z-directed field
        let dipole = Vec3::new(1.0, 0.0, 0.0); // 1 A·m²
        let b_field = Vec3::new(0.0, 0.0, 1.0); // 1 T

        let torque = magnetic_torque(dipole, &b_field);

        // τ = μ × B = (1, 0, 0) × (0, 0, 1) = (0, -1, 0)
        assert_relative_eq!(torque.x, 0.0);
        assert_relative_eq!(torque.y, -1.0);
        assert_relative_eq!(torque.z, 0.0);
    }

    #[test]
    fn test_magnetic_torque_aligned() {
        // Dipole aligned with field → no torque
        let dipole = Vec3::new(0.0, 0.0, 1.0);
        let b_field = Vec3::new(0.0, 0.0, 2.0);

        let torque = magnetic_torque(dipole, &b_field);

        assert_relative_eq!(torque.norm(), 0.0);
    }

    #[test]
    fn test_electric_dipole_force() {
        // Dipole in x-direction, field gradient in x
        let dipole = Vec3::new(1e-9, 0.0, 0.0); // 1 nC·m
        let field_gradient = [
            Vec3::new(1e6, 0.0, 0.0), // dE/dx
            Vec3::zeros(),            // dE/dy
            Vec3::zeros(),            // dE/dz
        ];

        let force = electric_dipole_force(dipole, &field_gradient);

        // F_x = p_x * dE_x/dx = 1e-9 * 1e6 = 1e-3 N
        assert_relative_eq!(force.x, 1e-3, epsilon = 1e-10);
        assert_relative_eq!(force.y, 0.0);
        assert_relative_eq!(force.z, 0.0);
    }

    #[test]
    fn test_radiation_pressure() {
        let e_field = Vec3::new(1e3, 0.0, 0.0); // 1 kV/m
        let area = 1e-4; // 1 cm²
        let normal = Vec3::new(1.0, 0.0, 0.0);

        let force = radiation_pressure_force(&e_field, area, normal);

        // P = ε₀ E² / 2 = 8.854e-12 * 1e6 / 2 = 4.427e-6 Pa
        // F = P * A = 4.427e-6 * 1e-4 = 4.427e-10 N
        let expected = 8.854187817e-12 * 1e6 / 2.0 * 1e-4;
        assert_relative_eq!(force.x, expected, epsilon = 1e-15);
        assert_relative_eq!(force.y, 0.0);
        assert_relative_eq!(force.z, 0.0);
    }
}
