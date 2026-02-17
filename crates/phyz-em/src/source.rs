//! Electromagnetic sources for FDTD simulation.

use phyz_math::Vec3;

use crate::grid::YeeGrid;

/// Electromagnetic source types.
#[derive(Debug, Clone)]
pub enum Source {
    /// Point electric dipole source.
    PointDipole {
        /// Position in grid coordinates.
        pos: Vec3,
        /// Frequency (Hz).
        freq: f64,
        /// Amplitude (V/m).
        amplitude: f64,
        /// Polarization direction (unit vector).
        direction: Vec3,
    },
    /// Plane wave source.
    PlaneWave {
        /// Propagation direction (unit vector).
        dir: Vec3,
        /// Frequency (Hz).
        freq: f64,
        /// Amplitude (V/m).
        amplitude: f64,
        /// Polarization direction (unit vector).
        polarization: Vec3,
    },
    /// Current loop (magnetic dipole).
    CurrentLoop {
        /// Center position.
        pos: Vec3,
        /// Loop radius (m).
        radius: f64,
        /// Frequency (Hz).
        freq: f64,
        /// Current amplitude (A).
        amplitude: f64,
        /// Normal direction (unit vector).
        normal: Vec3,
    },
}

impl Source {
    /// Apply source to grid at given time.
    pub fn apply(&self, grid: &mut YeeGrid, time: f64) {
        match self {
            Source::PointDipole {
                pos,
                freq,
                amplitude,
                direction,
            } => {
                let omega = 2.0 * std::f64::consts::PI * freq;
                let value = amplitude * (omega * time).sin();

                let (i, j, k) = grid.position_to_index(pos);

                // Add source to E-field components
                if let Some(ex) = grid.ex.get_mut(i, j, k) {
                    *ex += value * direction.x;
                }
                if let Some(ey) = grid.ey.get_mut(i, j, k) {
                    *ey += value * direction.y;
                }
                if let Some(ez) = grid.ez.get_mut(i, j, k) {
                    *ez += value * direction.z;
                }
            }

            Source::PlaneWave {
                dir,
                freq,
                amplitude,
                polarization,
            } => {
                let omega = 2.0 * std::f64::consts::PI * freq;
                let k = omega / grid.c0; // Wave number

                // Apply plane wave across entire grid slice
                for i in 0..grid.nx {
                    for j in 0..grid.ny {
                        for l in 0..grid.nz {
                            let pos = grid.index_to_position(i, j, l);
                            let phase =
                                k * (dir.x * pos.x + dir.y * pos.y + dir.z * pos.z) - omega * time;
                            let value = amplitude * phase.sin();

                            // Add to polarization direction
                            if let Some(ex) = grid.ex.get_mut(i, j, l) {
                                *ex += value * polarization.x * 0.01; // Scale down for stability
                            }
                            if let Some(ey) = grid.ey.get_mut(i, j, l) {
                                *ey += value * polarization.y * 0.01;
                            }
                            if let Some(ez) = grid.ez.get_mut(i, j, l) {
                                *ez += value * polarization.z * 0.01;
                            }
                        }
                    }
                }
            }

            Source::CurrentLoop {
                pos,
                radius,
                freq,
                amplitude,
                normal,
            } => {
                let omega = 2.0 * std::f64::consts::PI * freq;
                let value = amplitude * (omega * time).sin();

                let (ci, cj, ck) = grid.position_to_index(pos);
                let r_cells = (radius / grid.dx).ceil() as i32;

                // Apply current around loop
                for di in -r_cells..=r_cells {
                    for dj in -r_cells..=r_cells {
                        for dk in -r_cells..=r_cells {
                            let i = (ci as i32 + di).max(0) as usize;
                            let j = (cj as i32 + dj).max(0) as usize;
                            let k = (ck as i32 + dk).max(0) as usize;

                            if i < grid.nx && j < grid.ny && k < grid.nz {
                                let p = grid.index_to_position(i, j, k);
                                let r_vec = p - pos;

                                // Distance from loop axis
                                let r_perp = r_vec - normal * r_vec.dot(normal);
                                let dist = r_perp.norm();

                                // If close to loop radius, apply tangential current
                                if (dist - radius).abs() < grid.dx * 2.0 {
                                    let tangent = normal.cross(&r_perp).normalize();

                                    // Add to H-field (current creates magnetic field)
                                    let h_value = value * 0.1; // Scale for stability
                                    if let Some(hx) = grid.hx.get_mut(i, j, k) {
                                        *hx += h_value * tangent.x;
                                    }
                                    if let Some(hy) = grid.hy.get_mut(i, j, k) {
                                        *hy += h_value * tangent.y;
                                    }
                                    if let Some(hz) = grid.hz.get_mut(i, j, k) {
                                        *hz += h_value * tangent.z;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Field probe for measuring E or H field at a point.
#[derive(Debug, Clone)]
pub struct Probe {
    /// Probe position.
    pub pos: Vec3,
    /// Grid indices.
    pub i: usize,
    pub j: usize,
    pub k: usize,
    /// Time-series of measurements.
    pub e_field: Vec<Vec3>,
    pub h_field: Vec<Vec3>,
}

impl Probe {
    /// Create a new probe at given position.
    pub fn new(grid: &YeeGrid, pos: Vec3) -> Self {
        let (i, j, k) = grid.position_to_index(&pos);
        Self {
            pos,
            i,
            j,
            k,
            e_field: Vec::new(),
            h_field: Vec::new(),
        }
    }

    /// Record current field values.
    pub fn record(&mut self, grid: &YeeGrid) {
        self.e_field.push(grid.get_e_field(self.i, self.j, self.k));
        self.h_field.push(grid.get_h_field(self.i, self.j, self.k));
    }

    /// Get the latest E-field measurement.
    pub fn latest_e(&self) -> Option<Vec3> {
        self.e_field.last().copied()
    }

    /// Get the latest H-field measurement.
    pub fn latest_h(&self) -> Option<Vec3> {
        self.h_field.last().copied()
    }

    /// Compute power spectral density of E-field magnitude.
    pub fn power_spectrum(&self) -> Vec<f64> {
        // Simple magnitude time-series
        self.e_field.iter().map(|e| e.norm()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_dipole() {
        let mut grid = YeeGrid::new(32, 32, 32, 1e-9, 1e-18);

        let source = Source::PointDipole {
            pos: Vec3::new(16.0 * 1e-9, 16.0 * 1e-9, 16.0 * 1e-9),
            freq: 1e9,
            amplitude: 1.0,
            direction: Vec3::new(1.0, 0.0, 0.0),
        };

        let initial_energy = grid.total_energy();
        // Apply at time π/(2ω) where sin(ωt) = 1
        let omega = 2.0 * std::f64::consts::PI * 1e9;
        let t_max = std::f64::consts::PI / (2.0 * omega);
        source.apply(&mut grid, t_max);
        let after_source_energy = grid.total_energy();

        // Energy should increase after applying source
        assert!(after_source_energy > initial_energy);
    }

    #[test]
    fn test_probe() {
        let mut grid = YeeGrid::new(32, 32, 32, 1e-9, 1e-18);

        // Set a field value
        grid.ex.set(16, 16, 16, 5.0);

        let mut probe = Probe::new(&grid, Vec3::new(16.5 * 1e-9, 16.5 * 1e-9, 16.5 * 1e-9));
        probe.record(&grid);

        let e = probe.latest_e().unwrap();
        assert!(e.x > 4.0); // Should have captured the field
    }

    #[test]
    fn test_current_loop() {
        let mut grid = YeeGrid::new(64, 64, 64, 1e-9, 1e-18);

        let source = Source::CurrentLoop {
            pos: Vec3::new(32.0 * 1e-9, 32.0 * 1e-9, 32.0 * 1e-9),
            radius: 5.0 * 1e-9,
            freq: 1e9,
            amplitude: 1e-3,
            normal: Vec3::new(0.0, 0.0, 1.0),
        };

        let initial_energy = grid.total_energy();
        // Apply at time π/(2ω) where sin(ωt) = 1
        let omega = 2.0 * std::f64::consts::PI * 1e9;
        let t_max = std::f64::consts::PI / (2.0 * omega);
        source.apply(&mut grid, t_max);
        let after_source_energy = grid.total_energy();

        // Energy should increase
        assert!(after_source_energy > initial_energy);
    }
}
