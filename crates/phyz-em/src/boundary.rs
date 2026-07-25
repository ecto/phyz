//! Boundary conditions for FDTD simulation.

use crate::cpml::{Cpml, CpmlConfig};
use crate::grid::YeeGrid;

/// Boundary condition types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    /// Convolutional PML — a true impedance-matched absorbing layer.
    ///
    /// This is the boundary you want for open-domain problems. See
    /// [`crate::cpml`] for the formulation; reflection at normal incidence is
    /// typically below −60 dB for a 10-cell layer.
    Cpml(CpmlConfig),

    /// Graded-conductivity absorbing layer.
    ///
    /// **This is not a PML.** It adds electric loss σ in the boundary region
    /// without the matching magnetic loss σ\* = σμ/ε, so the layer has a
    /// different wave impedance than the interior and reflects strongly
    /// (typically −10 to −20 dB). It is kept because it is cheap — no
    /// auxiliary variables, no extra memory — and adequate when you only need
    /// to keep a cavity from ringing. Use [`BoundaryCondition::Cpml`] when
    /// reflections matter.
    LossyAbsorber {
        /// Layer thickness in cells.
        thickness: usize,
        /// Polynomial grading order.
        order: usize,
        /// Peak conductivity at the wall (S/m).
        sigma_max: f64,
    },

    /// Periodic boundary conditions on all three axes.
    Periodic,

    /// Perfect electric conductor (PEC) — tangential E = 0 on every wall.
    PerfectConductor,
}

impl Default for BoundaryCondition {
    fn default() -> Self {
        BoundaryCondition::Cpml(CpmlConfig::default())
    }
}

/// Conductivity profile for a [`BoundaryCondition::LossyAbsorber`].
///
/// Kept separate from the CPML so the cheap absorber stays dependency-free.
pub struct AbsorberLayer {
    /// Layer thickness (number of cells).
    pub thickness: usize,
    /// Polynomial grading order.
    pub order: usize,
    /// Maximum conductivity.
    pub sigma_max: f64,
    /// Precomputed conductivity profile, indexed by distance from the wall.
    sigma_profile: Vec<f64>,
}

impl AbsorberLayer {
    /// Create a new absorbing layer.
    pub fn new(thickness: usize, order: usize, sigma_max: f64) -> Self {
        let mut sigma_profile = vec![0.0; thickness];

        // Polynomial grading: σ(d) = σ_max * ((thickness − d)/thickness)^order,
        // so σ peaks at the wall (d = 0) and vanishes at the inner surface.
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

    /// Get conductivity at distance d (in cells) from the wall.
    pub fn get_sigma(&self, d: usize) -> f64 {
        if d < self.thickness {
            self.sigma_profile[d]
        } else {
            0.0
        }
    }
}

impl YeeGrid {
    /// Configure the boundary treatment for this grid.
    ///
    /// Unlike the previous design there is nothing to "apply" each step: PEC
    /// walls, periodic wrapping and the CPML convolution are all handled inside
    /// the field update loops.
    pub fn set_boundary(&mut self, bc: BoundaryCondition) {
        match bc {
            BoundaryCondition::Cpml(cfg) => {
                // A PML axis cannot also be periodic.
                for a in 0..3 {
                    if cfg.axes[a] {
                        self.periodic[a] = false;
                    }
                }
                self.cpml = Some(Cpml::new(
                    self.nx, self.ny, self.nz, self.dx, self.dy, self.dz, self.dt, cfg,
                ));
            }
            BoundaryCondition::LossyAbsorber {
                thickness,
                order,
                sigma_max,
            } => {
                self.cpml = None;
                self.apply_lossy_absorber(thickness, order, sigma_max);
            }
            BoundaryCondition::Periodic => {
                self.cpml = None;
                self.periodic = [true; 3];
            }
            BoundaryCondition::PerfectConductor => {
                self.cpml = None;
                self.periodic = [false; 3];
            }
        }
    }

    /// Set periodicity per axis. Periodic axes wrap in the update loops;
    /// non-periodic axes are terminated by a PEC wall (plus a CPML if one is
    /// configured for that axis).
    pub fn set_periodic(&mut self, axes: [bool; 3]) {
        self.periodic = axes;
    }

    /// Fill the boundary region with a graded conductivity.
    ///
    /// See [`BoundaryCondition::LossyAbsorber`] — this is a lossy layer, not a
    /// matched one.
    pub fn apply_lossy_absorber(&mut self, thickness: usize, order: usize, sigma_max: f64) {
        // Clamp per axis: a thin or periodic direction should not switch the
        // layer off along the other two.
        let n = [self.nx, self.ny, self.nz];
        let thick: Vec<usize> = (0..3)
            .map(|a| if self.periodic[a] { 0 } else { thickness.min(n[a] / 2) })
            .collect();
        if thick.iter().all(|&t| t == 0) {
            return;
        }
        let layers: Vec<Option<AbsorberLayer>> = thick
            .iter()
            .map(|&t| {
                if t == 0 {
                    None
                } else {
                    Some(AbsorberLayer::new(t, order, sigma_max))
                }
            })
            .collect();

        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let idx = [i, j, k];
                    // Take the strongest of the six wall contributions rather
                    // than the sum, so corners are not over-damped.
                    let mut sigma = 0.0_f64;
                    for a in 0..3 {
                        if let Some(layer) = &layers[a] {
                            sigma = sigma.max(layer.get_sigma(idx[a]));
                            sigma = sigma.max(layer.get_sigma(n[a] - 1 - idx[a]));
                        }
                    }
                    self.sigma.set(i, j, k, sigma);
                }
            }
        }
    }

    /// Zero the tangential E-field on every outer wall.
    ///
    /// The update loops already enforce this for non-periodic axes; this is
    /// exposed for tests and for callers driving the grid manually.
    pub fn apply_pec_boundary(&mut self) {
        for k in 0..self.nz {
            for j in 0..self.ny {
                self.ey.set(0, j, k, 0.0);
                self.ez.set(0, j, k, 0.0);
                self.ey.set(self.nx - 1, j, k, 0.0);
                self.ez.set(self.nx - 1, j, k, 0.0);
            }
        }
        for k in 0..self.nz {
            for i in 0..self.nx {
                self.ex.set(i, 0, k, 0.0);
                self.ez.set(i, 0, k, 0.0);
                self.ex.set(i, self.ny - 1, k, 0.0);
                self.ez.set(i, self.ny - 1, k, 0.0);
            }
        }
        for j in 0..self.ny {
            for i in 0..self.nx {
                self.ex.set(i, j, 0, 0.0);
                self.ey.set(i, j, 0, 0.0);
                self.ex.set(i, j, self.nz - 1, 0.0);
                self.ey.set(i, j, self.nz - 1, 0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_absorber_layer_grading() {
        let layer = AbsorberLayer::new(8, 2, 1.0);
        assert_eq!(layer.thickness, 8);
        assert_eq!(layer.order, 2);
        assert_eq!(layer.sigma_max, 1.0);

        // Conductivity increases toward the wall (d = 0).
        assert!(layer.get_sigma(7) < layer.get_sigma(6));
        assert!(layer.get_sigma(1) < layer.get_sigma(0));

        // Outside the layer it is zero.
        assert_eq!(layer.get_sigma(8), 0.0);
    }

    #[test]
    fn test_pec_boundary() {
        let mut grid = YeeGrid::new(16, 16, 16, 1e-9, 1e-18);
        grid.ex.set(0, 5, 5, 1.0);
        grid.ey.set(0, 5, 5, 1.0);
        grid.ez.set(0, 5, 5, 1.0);

        grid.apply_pec_boundary();

        assert_eq!(grid.ey.get(0, 5, 5), 0.0);
        assert_eq!(grid.ez.get(0, 5, 5), 0.0);
    }

    #[test]
    fn test_lossy_absorber_profile() {
        let mut grid = YeeGrid::new(32, 32, 32, 1e-9, 1e-18);
        grid.set_boundary(BoundaryCondition::LossyAbsorber {
            thickness: 8,
            order: 2,
            sigma_max: 1.0,
        });

        let sigma_center = grid.sigma.get(16, 16, 16);
        let sigma_edge = grid.sigma.get(1, 16, 16);
        assert!(sigma_edge > sigma_center);
        assert_eq!(sigma_center, 0.0);
        assert!(grid.cpml.is_none());
    }

    #[test]
    fn test_cpml_selection_disables_periodicity_on_pml_axes() {
        let dx = 1e-9;
        let dt = dx / (3e8 * 3_f64.sqrt() * 1.1);
        let mut grid = YeeGrid::new(48, 48, 48, dx, dt);
        grid.set_periodic([true, true, true]);
        grid.set_boundary(BoundaryCondition::Cpml(
            CpmlConfig::with_thickness(10).on_axes([false, false, true]),
        ));
        assert_eq!(grid.periodic, [true, true, false]);
        assert!(grid.cpml.as_ref().unwrap().is_active());
    }
}
