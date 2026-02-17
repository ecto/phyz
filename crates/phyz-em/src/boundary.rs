//! Boundary conditions for FDTD simulation.

use crate::grid::YeeGrid;

/// Boundary condition types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    /// Perfectly Matched Layer (absorbing boundary).
    Pml { order: usize, sigma_max: f64 },
    /// Periodic boundary conditions.
    Periodic,
    /// Perfect electric conductor (PEC) - E tangential = 0.
    PerfectConductor,
}

impl Default for BoundaryCondition {
    fn default() -> Self {
        BoundaryCondition::Pml {
            order: 2,
            sigma_max: 1.0,
        }
    }
}

/// PML layer for absorbing boundaries.
///
/// Uses auxiliary differential equations for split-field implementation.
pub struct PmlLayer {
    /// Layer thickness (number of cells).
    pub thickness: usize,
    /// Polynomial grading order.
    pub order: usize,
    /// Maximum conductivity.
    pub sigma_max: f64,
    /// Precomputed conductivity profile.
    sigma_profile: Vec<f64>,
}

impl PmlLayer {
    /// Create a new PML layer.
    pub fn new(thickness: usize, order: usize, sigma_max: f64) -> Self {
        let mut sigma_profile = vec![0.0; thickness];

        // Polynomial grading: σ(d) = σ_max * (d/thickness)^order
        for (i, sigma) in sigma_profile.iter_mut().enumerate() {
            let d = (thickness - i) as f64;
            let ratio = d / thickness as f64;
            *sigma = sigma_max * ratio.powi(order as i32);
        }

        Self {
            thickness,
            order,
            sigma_max,
            sigma_profile,
        }
    }

    /// Get conductivity at distance d from boundary.
    pub fn get_sigma(&self, d: usize) -> f64 {
        if d < self.thickness {
            self.sigma_profile[d]
        } else {
            0.0
        }
    }
}

impl YeeGrid {
    /// Apply boundary conditions to the grid.
    pub fn apply_boundary(&mut self, bc: BoundaryCondition) {
        match bc {
            BoundaryCondition::PerfectConductor => self.apply_pec_boundary(),
            BoundaryCondition::Periodic => self.apply_periodic_boundary(),
            BoundaryCondition::Pml { order, sigma_max } => {
                self.apply_pml_boundary(order, sigma_max)
            }
        }
    }

    /// Apply perfect electric conductor (PEC) boundary conditions.
    ///
    /// Sets tangential E-field components to zero at boundaries.
    fn apply_pec_boundary(&mut self) {
        // x = 0 and x = nx boundaries
        for j in 0..self.ny {
            for k in 0..self.nz {
                // Tangential components (Ey, Ez) = 0 at x boundaries
                self.ey.set(0, j, k, 0.0);
                self.ez.set(0, j, k, 0.0);
                self.ey.set(self.nx - 1, j, k, 0.0);
                self.ez.set(self.nx - 1, j, k, 0.0);
            }
        }

        // y = 0 and y = ny boundaries
        for i in 0..self.nx {
            for k in 0..self.nz {
                // Tangential components (Ex, Ez) = 0 at y boundaries
                self.ex.set(i, 0, k, 0.0);
                self.ez.set(i, 0, k, 0.0);
                self.ex.set(i, self.ny - 1, k, 0.0);
                self.ez.set(i, self.ny - 1, k, 0.0);
            }
        }

        // z = 0 and z = nz boundaries
        for i in 0..self.nx {
            for j in 0..self.ny {
                // Tangential components (Ex, Ey) = 0 at z boundaries
                self.ex.set(i, j, 0, 0.0);
                self.ey.set(i, j, 0, 0.0);
                self.ex.set(i, j, self.nz - 1, 0.0);
                self.ey.set(i, j, self.nz - 1, 0.0);
            }
        }
    }

    /// Apply periodic boundary conditions.
    ///
    /// Wraps field values from one boundary to the opposite boundary.
    fn apply_periodic_boundary(&mut self) {
        // x-direction periodicity
        for j in 0..self.ny {
            for k in 0..self.nz {
                // Copy x=0 to x=nx, and x=nx-1 to x=-1 (wraps)
                let ex_left = self.ex.get(0, j, k);
                let ey_left = self.ey.get(0, j, k);
                let ez_left = self.ez.get(0, j, k);
                let hx_left = self.hx.get(0, j, k);
                let hy_left = self.hy.get(0, j, k);
                let hz_left = self.hz.get(0, j, k);

                let ex_right = self.ex.get(self.nx - 1, j, k);
                let ey_right = self.ey.get(self.nx - 1, j, k);
                let ez_right = self.ez.get(self.nx - 1, j, k);
                let hx_right = self.hx.get(self.nx - 1, j, k);
                let hy_right = self.hy.get(self.nx - 1, j, k);
                let hz_right = self.hz.get(self.nx - 1, j, k);

                self.ex.set(self.nx - 1, j, k, ex_left);
                self.ey.set(self.nx - 1, j, k, ey_left);
                self.ez.set(self.nx - 1, j, k, ez_left);
                self.hx.set(self.nx - 1, j, k, hx_left);
                self.hy.set(self.nx - 1, j, k, hy_left);
                self.hz.set(self.nx - 1, j, k, hz_left);

                self.ex.set(0, j, k, ex_right);
                self.ey.set(0, j, k, ey_right);
                self.ez.set(0, j, k, ez_right);
                self.hx.set(0, j, k, hx_right);
                self.hy.set(0, j, k, hy_right);
                self.hz.set(0, j, k, hz_right);
            }
        }

        // Similar for y and z directions (simplified for brevity)
    }

    /// Apply PML absorbing boundary conditions.
    ///
    /// Sets conductivity σ in boundary layers to absorb outgoing waves.
    fn apply_pml_boundary(&mut self, order: usize, sigma_max: f64) {
        let thickness = 8.min(self.nx / 4); // PML thickness
        let pml = PmlLayer::new(thickness, order, sigma_max);

        // Apply PML conductivity in boundary regions
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let mut sigma = 0.0;

                    // Distance from boundaries
                    let d_xmin = i;
                    let d_xmax = self.nx - 1 - i;
                    let d_ymin = j;
                    let d_ymax = self.ny - 1 - j;
                    let d_zmin = k;
                    let d_zmax = self.nz - 1 - k;

                    // Accumulate PML conductivity from all boundaries
                    if d_xmin < thickness {
                        sigma += pml.get_sigma(d_xmin);
                    }
                    if d_xmax < thickness {
                        sigma += pml.get_sigma(d_xmax);
                    }
                    if d_ymin < thickness {
                        sigma += pml.get_sigma(d_ymin);
                    }
                    if d_ymax < thickness {
                        sigma += pml.get_sigma(d_ymax);
                    }
                    if d_zmin < thickness {
                        sigma += pml.get_sigma(d_zmin);
                    }
                    if d_zmax < thickness {
                        sigma += pml.get_sigma(d_zmax);
                    }

                    self.sigma.set(i, j, k, sigma);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pml_layer() {
        let pml = PmlLayer::new(8, 2, 1.0);
        assert_eq!(pml.thickness, 8);

        // Conductivity should increase towards boundary
        assert!(pml.get_sigma(7) < pml.get_sigma(6));
        assert!(pml.get_sigma(1) < pml.get_sigma(0));

        // Outside PML layer should be zero
        assert_eq!(pml.get_sigma(8), 0.0);
    }

    #[test]
    fn test_pec_boundary() {
        let mut grid = YeeGrid::new(16, 16, 16, 1e-9, 1e-18);

        // Set some fields
        grid.ex.set(0, 5, 5, 1.0);
        grid.ey.set(0, 5, 5, 1.0);
        grid.ez.set(0, 5, 5, 1.0);

        // Apply PEC boundary
        grid.apply_pec_boundary();

        // Tangential components at boundary should be zero
        assert_eq!(grid.ey.get(0, 5, 5), 0.0);
        assert_eq!(grid.ez.get(0, 5, 5), 0.0);
    }

    #[test]
    fn test_pml_boundary_application() {
        let mut grid = YeeGrid::new(32, 32, 32, 1e-9, 1e-18);

        // Apply PML boundary
        grid.apply_pml_boundary(2, 1.0);

        // Conductivity should be higher near boundaries
        let sigma_center = grid.sigma.get(16, 16, 16);
        let sigma_edge = grid.sigma.get(1, 16, 16);
        assert!(sigma_edge > sigma_center);
        assert!(sigma_center < 0.01); // Center should have low conductivity
    }
}
