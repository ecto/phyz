//! Yee grid structure for FDTD simulation.

use tau_math::Vec3;

/// 3D array storage for grid fields.
#[derive(Debug, Clone)]
pub struct Array3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    data: Vec<f64>,
}

impl Array3D {
    /// Create a new 3D array filled with zeros.
    pub fn zeros(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            nx,
            ny,
            nz,
            data: vec![0.0; nx * ny * nz],
        }
    }

    /// Create a new 3D array filled with a constant value.
    pub fn filled(nx: usize, ny: usize, nz: usize, value: f64) -> Self {
        Self {
            nx,
            ny,
            nz,
            data: vec![value; nx * ny * nz],
        }
    }

    /// Get value at (i, j, k) with bounds checking.
    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        if i < self.nx && j < self.ny && k < self.nz {
            self.data[i + j * self.nx + k * self.nx * self.ny]
        } else {
            0.0
        }
    }

    /// Set value at (i, j, k) with bounds checking.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, k: usize, value: f64) {
        if i < self.nx && j < self.ny && k < self.nz {
            self.data[i + j * self.nx + k * self.nx * self.ny] = value;
        }
    }

    /// Get mutable reference at (i, j, k) with bounds checking.
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize, k: usize) -> Option<&mut f64> {
        if i < self.nx && j < self.ny && k < self.nz {
            Some(&mut self.data[i + j * self.nx + k * self.nx * self.ny])
        } else {
            None
        }
    }

    /// Add to value at (i, j, k).
    #[inline]
    pub fn add(&mut self, i: usize, j: usize, k: usize, value: f64) {
        if i < self.nx && j < self.ny && k < self.nz {
            self.data[i + j * self.nx + k * self.nx * self.ny] += value;
        }
    }

    /// Clear all values to zero.
    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    /// Sum all values in the array.
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Compute squared norm of all values.
    pub fn norm_squared(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum()
    }
}

/// Yee grid for FDTD simulation.
///
/// E-fields are located at cell edges, H-fields at cell faces.
/// Grid spacing is uniform with cell size dx × dy × dz.
pub struct YeeGrid {
    /// Number of cells in x, y, z directions.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,

    /// Grid spacing (m).
    pub dx: f64,

    /// Timestep (s).
    pub dt: f64,

    /// Relative permittivity (ε_r) at each cell.
    pub eps_r: Array3D,

    /// Relative permeability (μ_r) at each cell.
    pub mu_r: Array3D,

    /// Conductivity (σ) at each cell (S/m).
    pub sigma: Array3D,

    /// Electric field components (V/m).
    pub ex: Array3D,
    pub ey: Array3D,
    pub ez: Array3D,

    /// Magnetic field components (A/m).
    pub hx: Array3D,
    pub hy: Array3D,
    pub hz: Array3D,

    /// Physical constants.
    pub eps0: f64, // Vacuum permittivity (F/m)
    pub mu0: f64, // Vacuum permeability (H/m)
    pub c0: f64,  // Speed of light (m/s)
}

impl YeeGrid {
    /// Create a new Yee grid with default vacuum properties.
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dt: f64) -> Self {
        let eps0 = 8.854187817e-12; // F/m
        let mu0 = 4.0 * std::f64::consts::PI * 1e-7; // H/m
        let c0 = 1.0 / (eps0 * mu0).sqrt(); // ~3e8 m/s

        Self {
            nx,
            ny,
            nz,
            dx,
            dt,
            eps_r: Array3D::filled(nx, ny, nz, 1.0),
            mu_r: Array3D::filled(nx, ny, nz, 1.0),
            sigma: Array3D::zeros(nx, ny, nz),
            ex: Array3D::zeros(nx, ny, nz),
            ey: Array3D::zeros(nx, ny, nz),
            ez: Array3D::zeros(nx, ny, nz),
            hx: Array3D::zeros(nx, ny, nz),
            hy: Array3D::zeros(nx, ny, nz),
            hz: Array3D::zeros(nx, ny, nz),
            eps0,
            mu0,
            c0,
        }
    }

    /// Get E-field vector at cell (i, j, k).
    pub fn get_e_field(&self, i: usize, j: usize, k: usize) -> Vec3 {
        Vec3::new(
            self.ex.get(i, j, k),
            self.ey.get(i, j, k),
            self.ez.get(i, j, k),
        )
    }

    /// Get H-field vector at cell (i, j, k).
    pub fn get_h_field(&self, i: usize, j: usize, k: usize) -> Vec3 {
        Vec3::new(
            self.hx.get(i, j, k),
            self.hy.get(i, j, k),
            self.hz.get(i, j, k),
        )
    }

    /// Set material properties for a region.
    pub fn set_material(
        &mut self,
        i_range: (usize, usize),
        j_range: (usize, usize),
        k_range: (usize, usize),
        eps_r: f64,
        mu_r: f64,
        sigma: f64,
    ) {
        for i in i_range.0..i_range.1.min(self.nx) {
            for j in j_range.0..j_range.1.min(self.ny) {
                for k in k_range.0..k_range.1.min(self.nz) {
                    self.eps_r.set(i, j, k, eps_r);
                    self.mu_r.set(i, j, k, mu_r);
                    self.sigma.set(i, j, k, sigma);
                }
            }
        }
    }

    /// Convert world position to grid index.
    pub fn position_to_index(&self, pos: &Vec3) -> (usize, usize, usize) {
        let i = (pos.x / self.dx).floor().max(0.0) as usize;
        let j = (pos.y / self.dx).floor().max(0.0) as usize;
        let k = (pos.z / self.dx).floor().max(0.0) as usize;
        (i.min(self.nx - 1), j.min(self.ny - 1), k.min(self.nz - 1))
    }

    /// Convert grid index to world position (cell center).
    pub fn index_to_position(&self, i: usize, j: usize, k: usize) -> Vec3 {
        Vec3::new(
            (i as f64 + 0.5) * self.dx,
            (j as f64 + 0.5) * self.dx,
            (k as f64 + 0.5) * self.dx,
        )
    }

    /// Compute total electromagnetic energy in the grid.
    ///
    /// Energy density: u = (1/2)(ε|E|² + μ|H|²)
    pub fn total_energy(&self) -> f64 {
        let mut energy = 0.0;
        let dv = self.dx * self.dx * self.dx;

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let ex = self.ex.get(i, j, k);
                    let ey = self.ey.get(i, j, k);
                    let ez = self.ez.get(i, j, k);
                    let hx = self.hx.get(i, j, k);
                    let hy = self.hy.get(i, j, k);
                    let hz = self.hz.get(i, j, k);

                    let eps = self.eps0 * self.eps_r.get(i, j, k);
                    let mu = self.mu0 * self.mu_r.get(i, j, k);

                    let e_energy = 0.5 * eps * (ex * ex + ey * ey + ez * ez);
                    let h_energy = 0.5 * mu * (hx * hx + hy * hy + hz * hz);

                    energy += (e_energy + h_energy) * dv;
                }
            }
        }

        energy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array3d_basic() {
        let mut arr = Array3D::zeros(4, 4, 4);
        arr.set(1, 2, 3, 5.0);
        assert_eq!(arr.get(1, 2, 3), 5.0);
        assert_eq!(arr.get(0, 0, 0), 0.0);

        arr.add(1, 2, 3, 3.0);
        assert_eq!(arr.get(1, 2, 3), 8.0);
    }

    #[test]
    fn test_yee_grid_creation() {
        let grid = YeeGrid::new(10, 10, 10, 0.1e-6, 1e-15);
        assert_eq!(grid.nx, 10);
        assert_eq!(grid.ny, 10);
        assert_eq!(grid.nz, 10);
        assert!((grid.c0 - 3e8).abs() < 1e6); // Speed of light ~3e8 m/s
    }

    #[test]
    fn test_position_conversion() {
        let grid = YeeGrid::new(10, 10, 10, 0.1, 1e-12);
        let idx = grid.position_to_index(&Vec3::new(0.15, 0.25, 0.35));
        assert_eq!(idx, (1, 2, 3));

        let pos = grid.index_to_position(1, 2, 3);
        assert!((pos.x - 0.15).abs() < 1e-10);
        assert!((pos.y - 0.25).abs() < 1e-10);
        assert!((pos.z - 0.35).abs() < 1e-10);
    }
}
