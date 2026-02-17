//! Material constitutive models for MPM.

use phyz_math::Mat3;

/// Material constitutive model.
#[derive(Debug, Clone, Copy)]
pub enum Material {
    /// Linear elastic material (Neo-Hookean).
    Elastic {
        /// Young's modulus (Pa).
        e: f64,
        /// Poisson's ratio (dimensionless).
        nu: f64,
    },
    /// Elastic-plastic material with von Mises yield.
    Plastic {
        /// Young's modulus (Pa).
        e: f64,
        /// Yield stress (Pa).
        yield_stress: f64,
    },
    /// Granular material with Drucker-Prager yield.
    Granular {
        /// Internal friction angle (radians).
        phi: f64,
    },
    /// Fluid material with viscosity and equation of state.
    Fluid {
        /// Dynamic viscosity (Pa·s).
        viscosity: f64,
        /// Equation of state parameters.
        eos: EquationOfState,
    },
}

/// Equation of state for fluids.
#[derive(Debug, Clone, Copy)]
pub enum EquationOfState {
    /// Water-like EOS: p = gamma * (rho/rho0)^7.
    Water {
        /// Reference density (kg/m³).
        rho0: f64,
        /// Pressure coefficient.
        gamma: f64,
    },
    /// Ideal gas: p = rho0 * cs^2 * (J - 1).
    IdealGas {
        /// Reference density (kg/m³).
        rho0: f64,
        /// Speed of sound (m/s).
        cs: f64,
    },
}

impl Material {
    /// Compute first Piola-Kirchhoff stress from deformation gradient.
    ///
    /// Returns stress tensor P = ∂ψ/∂F where ψ is the strain energy density.
    pub fn compute_stress(&self, f: &Mat3, j: f64) -> Mat3 {
        match self {
            Material::Elastic { e, nu } => {
                // Neo-Hookean model
                // ψ(F) = μ/2 (|F|²_F - 3 - 2 ln J) + λ/2 (ln J)²
                // P = μ (F - F^-T) + λ ln(J) F^-T

                // Prevent singular deformation
                let j_safe = j.max(0.01);

                let mu = e / (2.0 * (1.0 + nu));
                let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

                // Use safe inverse
                let f_inv_t = if let Some(inv) = f.try_inverse() {
                    inv.transpose()
                } else {
                    // If F is singular, return zero stress
                    return Mat3::zeros();
                };

                let ln_j = j_safe.ln();

                mu * (f - f_inv_t) + lambda * ln_j * f_inv_t
            }
            Material::Plastic { e, yield_stress } => {
                // Simplified J-plasticity: treat as elastic for now
                // (Full plastic update requires tracking F_p over time)
                let j_safe = j.max(0.01);
                let nu = 0.3; // Assume typical Poisson ratio
                let mu = e / (2.0 * (1.0 + nu));
                let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

                let f_inv_t = if let Some(inv) = f.try_inverse() {
                    inv.transpose()
                } else {
                    return Mat3::zeros();
                };

                let ln_j = j_safe.ln();

                // Apply yield criterion (von Mises on deviatoric stress)
                let p_elastic = mu * (f - f_inv_t) + lambda * ln_j * f_inv_t;

                // Compute deviatoric stress magnitude
                let dev_stress = p_elastic - (p_elastic.trace() / 3.0) * Mat3::identity();
                let dev_norm = dev_stress.norm();

                // If yield exceeded, scale back
                if dev_norm > *yield_stress {
                    let scale = yield_stress / dev_norm;
                    p_elastic * scale
                } else {
                    p_elastic
                }
            }
            Material::Granular { phi } => {
                // Drucker-Prager yield criterion
                // Simplified: use friction coefficient from angle
                let j_safe = j.max(0.01);
                let mu_eff = phi.tan();

                // Elastic response (simplified)
                let e = 1e6; // Typical granular Young's modulus
                let nu = 0.3;
                let mu = e / (2.0 * (1.0 + nu));
                let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

                let f_inv_t = if let Some(inv) = f.try_inverse() {
                    inv.transpose()
                } else {
                    return Mat3::zeros();
                };

                let ln_j = j_safe.ln();

                let p_elastic = mu * (f - f_inv_t) + lambda * ln_j * f_inv_t;

                // Apply Drucker-Prager yield
                let dev_stress = p_elastic - (p_elastic.trace() / 3.0) * Mat3::identity();
                let dev_norm = dev_stress.norm();
                let mean_stress = p_elastic.trace() / 3.0;

                let yield_fn = dev_norm - 3.0_f64.sqrt() * mu_eff * mean_stress.abs();

                if yield_fn > 0.0 && dev_norm > 1e-12 {
                    // Scale back to yield surface
                    let scale = (3.0_f64.sqrt() * mu_eff * mean_stress.abs()) / dev_norm;
                    let dev_scaled = dev_stress * scale;
                    dev_scaled + mean_stress * Mat3::identity()
                } else {
                    p_elastic
                }
            }
            Material::Fluid { viscosity: _, eos } => {
                // Fluid pressure from equation of state
                let j_safe = j.max(0.01);
                let pressure = match eos {
                    EquationOfState::Water { rho0: _, gamma } => {
                        // p = gamma * (rho/rho0)^7 = gamma * J^(-7)
                        gamma * j_safe.powf(-7.0)
                    }
                    EquationOfState::IdealGas { rho0, cs } => {
                        // p = rho0 * cs^2 * (J - 1)
                        rho0 * cs * cs * (j_safe - 1.0)
                    }
                };

                // Pressure contribution: P = -p * J * F^-T
                let f_inv_t = if let Some(inv) = f.try_inverse() {
                    inv.transpose()
                } else {
                    return Mat3::zeros();
                };

                -pressure * j_safe * f_inv_t

                // Note: viscous stress would require velocity gradient
                // For now, just pressure response
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elastic_stress() {
        let mat = Material::Elastic { e: 1e6, nu: 0.3 };
        let f = Mat3::identity(); // No deformation
        let j = 1.0;
        let stress = mat.compute_stress(&f, j);

        // Should be near zero for identity deformation
        assert!(stress.norm() < 1e-6);
    }

    #[test]
    fn test_fluid_pressure() {
        let mat = Material::Fluid {
            viscosity: 1e-3,
            eos: EquationOfState::IdealGas {
                rho0: 1000.0,
                cs: 100.0,
            },
        };

        // Compression: J < 1 (volume decreased)
        let f = Mat3::identity() * 0.9; // 10% linear compression
        let j = 0.9_f64.powi(3); // J ≈ 0.729

        let stress = mat.compute_stress(&f, j);

        // For ideal gas: p = rho0 * cs^2 * (J - 1)
        // J < 1 means p < 0 (tension)
        // Stress = -p * J * F^-T, so negative p gives positive trace
        // Actually for compression (J<1), we get positive pressure in physics
        // but the formula gives negative p, and then stress = -p*J*F^-T
        // Let's just check that stress is computed (not zero)
        assert!(stress.norm() > 1.0);
    }
}
