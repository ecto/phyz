//! Yee grid structure for FDTD simulation.

use phyz_math::Vec3;

use crate::cpml::Cpml;
use crate::dispersion::{DispersiveMaterial, DispersiveState};

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

    #[inline]
    fn offset(&self, i: usize, j: usize, k: usize) -> usize {
        i + j * self.nx + k * self.nx * self.ny
    }

    #[inline]
    fn in_bounds(&self, i: usize, j: usize, k: usize) -> bool {
        i < self.nx && j < self.ny && k < self.nz
    }

    /// Get value at (i, j, k).
    ///
    /// Out-of-bounds access is a programming error: it panics in debug builds
    /// (`debug_assertions`) and returns 0.0 in release builds. Use
    /// [`Array3D::try_get`] when an index may legitimately be outside the grid.
    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        debug_assert!(
            self.in_bounds(i, j, k),
            "Array3D index ({i}, {j}, {k}) out of bounds for {}x{}x{} array",
            self.nx,
            self.ny,
            self.nz
        );
        if self.in_bounds(i, j, k) {
            self.data[self.offset(i, j, k)]
        } else {
            0.0
        }
    }

    /// Get value at (i, j, k), returning `None` when out of bounds.
    #[inline]
    pub fn try_get(&self, i: usize, j: usize, k: usize) -> Option<f64> {
        if self.in_bounds(i, j, k) {
            Some(self.data[self.offset(i, j, k)])
        } else {
            None
        }
    }

    /// Get value at (i, j, k), returning 0.0 when out of bounds.
    ///
    /// Explicit opt-in to the "outside the grid reads as vacuum" convention.
    #[inline]
    pub fn get_or_zero(&self, i: usize, j: usize, k: usize) -> f64 {
        self.try_get(i, j, k).unwrap_or(0.0)
    }

    /// Set value at (i, j, k). Panics in debug builds if out of bounds.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, k: usize, value: f64) {
        debug_assert!(
            self.in_bounds(i, j, k),
            "Array3D index ({i}, {j}, {k}) out of bounds for {}x{}x{} array",
            self.nx,
            self.ny,
            self.nz
        );
        if self.in_bounds(i, j, k) {
            let o = self.offset(i, j, k);
            self.data[o] = value;
        }
    }

    /// Get mutable reference at (i, j, k), or `None` if out of bounds.
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize, k: usize) -> Option<&mut f64> {
        if self.in_bounds(i, j, k) {
            let o = self.offset(i, j, k);
            Some(&mut self.data[o])
        } else {
            None
        }
    }

    /// Add to value at (i, j, k). Panics in debug builds if out of bounds.
    #[inline]
    pub fn add(&mut self, i: usize, j: usize, k: usize, value: f64) {
        debug_assert!(
            self.in_bounds(i, j, k),
            "Array3D index ({i}, {j}, {k}) out of bounds for {}x{}x{} array",
            self.nx,
            self.ny,
            self.nz
        );
        if self.in_bounds(i, j, k) {
            let o = self.offset(i, j, k);
            self.data[o] += value;
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

    /// Largest absolute value in the array.
    pub fn abs_max(&self) -> f64 {
        self.data.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
    }
}

/// Yee grid for FDTD simulation.
///
/// E-fields are located at cell edges, H-fields at cell faces:
/// `Ex` at (i+½, j, k), `Ey` at (i, j+½, k), `Ez` at (i, j, k+½),
/// `Hx` at (i, j+½, k+½), `Hy` at (i+½, j, k+½), `Hz` at (i+½, j+½, k).
///
/// Cells are rectangular with independent spacings `dx`, `dy`, `dz`.
pub struct YeeGrid {
    /// Number of cells in x, y, z directions.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,

    /// Grid spacing along x (m).
    pub dx: f64,
    /// Grid spacing along y (m).
    pub dy: f64,
    /// Grid spacing along z (m).
    pub dz: f64,

    /// Timestep (s).
    pub dt: f64,

    /// Relative permittivity (ε_r) at each cell (non-dispersive cells).
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

    /// Dispersive material library. Index 0 is reserved for "non-dispersive",
    /// meaning the cell uses `eps_r` / `sigma` directly.
    pub materials: Vec<DispersiveMaterial>,

    /// Per-cell dispersive material id (0 = non-dispersive).
    pub mat_id: Vec<u32>,

    /// ADE auxiliary state for dispersive materials, allocated on demand.
    pub dispersion: Option<DispersiveState>,

    /// CPML absorbing boundary state, if enabled.
    pub cpml: Option<Cpml>,

    /// Per-axis periodicity (x, y, z).
    pub periodic: [bool; 3],

    /// Physical constants.
    pub eps0: f64, // Vacuum permittivity (F/m)
    pub mu0: f64, // Vacuum permeability (H/m)
    pub c0: f64,  // Speed of light (m/s)
}

/// Vacuum permittivity (F/m).
pub const EPS0: f64 = 8.854187817e-12;
/// Vacuum permeability (H/m).
pub const MU0: f64 = 4.0 * std::f64::consts::PI * 1e-7;

impl YeeGrid {
    /// Create a new Yee grid with cubic cells and vacuum properties.
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dt: f64) -> Self {
        Self::new_rect(nx, ny, nz, dx, dx, dx, dt)
    }

    /// Create a new Yee grid with independent cell dimensions.
    #[allow(clippy::too_many_arguments)]
    pub fn new_rect(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64, dt: f64) -> Self {
        let eps0 = EPS0;
        let mu0 = MU0;
        let c0 = 1.0 / (eps0 * mu0).sqrt(); // ~3e8 m/s

        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
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
            materials: vec![DispersiveMaterial::vacuum()],
            mat_id: vec![0; nx * ny * nz],
            dispersion: None,
            cpml: None,
            periodic: [false; 3],
            eps0,
            mu0,
            c0,
        }
    }

    /// Number of cells.
    #[inline]
    pub fn len(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Linear index of cell (i, j, k).
    #[inline]
    pub fn cell_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + j * self.nx + k * self.nx * self.ny
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
        let j = (pos.y / self.dy).floor().max(0.0) as usize;
        let k = (pos.z / self.dz).floor().max(0.0) as usize;
        (i.min(self.nx - 1), j.min(self.ny - 1), k.min(self.nz - 1))
    }

    /// Convert grid index to world position (cell center).
    pub fn index_to_position(&self, i: usize, j: usize, k: usize) -> Vec3 {
        Vec3::new(
            (i as f64 + 0.5) * self.dx,
            (j as f64 + 0.5) * self.dy,
            (k as f64 + 0.5) * self.dz,
        )
    }

    /// Compute total electromagnetic energy in the grid.
    ///
    /// Energy density: u = (1/2)(ε|E|² + μ|H|²)
    pub fn total_energy(&self) -> f64 {
        let mut energy = 0.0;
        let dv = self.dx * self.dy * self.dz;

        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let ex = self.ex.get(i, j, k);
                    let ey = self.ey.get(i, j, k);
                    let ez = self.ez.get(i, j, k);
                    let hx = self.hx.get(i, j, k);
                    let hy = self.hy.get(i, j, k);
                    let hz = self.hz.get(i, j, k);

                    let eps = self.eps0 * self.eps_permittivity_r(i, j, k);
                    let mu = self.mu0 * self.mu_r.get(i, j, k);

                    let e_energy = 0.5 * eps * (ex * ex + ey * ey + ez * ez);
                    let h_energy = 0.5 * mu * (hx * hx + hy * hy + hz * hz);

                    energy += (e_energy + h_energy) * dv;
                }
            }
        }

        energy
    }

    /// Instantaneous relative permittivity used for energy accounting.
    ///
    /// For dispersive cells this is `eps_inf`; the pole contributions live in
    /// the ADE polarization state rather than in `eps_r`.
    #[inline]
    pub fn eps_permittivity_r(&self, i: usize, j: usize, k: usize) -> f64 {
        self.eps_r.get(i, j, k)
    }

    // ---- Dispersive material library -------------------------------------

    /// Register a material and return its id, for use with the
    /// `assign_material_*` methods.
    ///
    /// Call [`YeeGrid::build_dispersion`] once all materials are registered
    /// and placed.
    pub fn add_material(&mut self, material: DispersiveMaterial) -> u32 {
        self.materials.push(material);
        (self.materials.len() - 1) as u32
    }

    /// Look up a registered material.
    pub fn material(&self, id: u32) -> &DispersiveMaterial {
        &self.materials[id as usize]
    }

    /// Assign material `id` to one cell, mirroring its instantaneous
    /// properties into the `eps_r` / `mu_r` / `sigma` arrays.
    #[inline]
    pub fn assign_material_cell(&mut self, id: u32, i: usize, j: usize, k: usize) {
        let m = &self.materials[id as usize];
        let (eps_inf, mu_r, sigma) = (m.eps_inf, m.mu_r, m.sigma);
        let c = self.cell_index(i, j, k);
        self.mat_id[c] = id;
        self.eps_r.set(i, j, k, eps_inf);
        self.mu_r.set(i, j, k, mu_r);
        self.sigma.set(i, j, k, sigma);
    }

    /// Assign material `id` to a box of cells (half-open index ranges).
    pub fn assign_material_box(
        &mut self,
        id: u32,
        i_range: (usize, usize),
        j_range: (usize, usize),
        k_range: (usize, usize),
    ) {
        for k in k_range.0..k_range.1.min(self.nz) {
            for j in j_range.0..j_range.1.min(self.ny) {
                for i in i_range.0..i_range.1.min(self.nx) {
                    self.assign_material_cell(id, i, j, k);
                }
            }
        }
    }

    /// Assign material `id` to the half-space `k >= k0` (a slab boundary
    /// normal to z, the geometry used for Fresnel validation).
    pub fn assign_material_halfspace_z(&mut self, id: u32, k0: usize) {
        self.assign_material_box(id, (0, self.nx), (0, self.ny), (k0, self.nz));
    }

    /// Assign material `id` to every cell whose centre lies inside a sphere.
    pub fn assign_material_sphere(&mut self, id: u32, center: Vec3, radius: f64) {
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let p = self.index_to_position(i, j, k);
                    if (p - center).norm() <= radius {
                        self.assign_material_cell(id, i, j, k);
                    }
                }
            }
        }
    }

    /// Allocate the ADE auxiliary state for the registered materials.
    ///
    /// A no-op if no registered material has poles. Must be called after all
    /// [`YeeGrid::add_material`] calls and before stepping.
    pub fn build_dispersion(&mut self) {
        self.dispersion = DispersiveState::new(self.nx, self.ny, self.nz, self.dt, &self.materials);
    }

    /// Zero every field and all auxiliary (CPML, ADE) state.
    pub fn reset_fields(&mut self) {
        for a in [
            &mut self.ex,
            &mut self.ey,
            &mut self.ez,
            &mut self.hx,
            &mut self.hy,
            &mut self.hz,
        ] {
            a.clear();
        }
        if let Some(c) = self.cpml.as_mut() {
            c.reset();
        }
        if let Some(d) = self.dispersion.as_mut() {
            d.reset();
        }
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
    #[should_panic(expected = "out of bounds")]
    #[cfg(debug_assertions)]
    fn test_array3d_oob_panics_in_debug() {
        let arr = Array3D::zeros(4, 4, 4);
        let _ = arr.get(4, 0, 0);
    }

    #[test]
    fn test_array3d_explicit_oob_helpers() {
        let arr = Array3D::zeros(4, 4, 4);
        assert_eq!(arr.try_get(4, 0, 0), None);
        assert_eq!(arr.get_or_zero(4, 0, 0), 0.0);
        assert_eq!(arr.try_get(3, 3, 3), Some(0.0));
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

    #[test]
    fn test_rectangular_cells() {
        let grid = YeeGrid::new_rect(10, 10, 10, 0.1, 0.2, 0.4, 1e-12);
        let pos = grid.index_to_position(1, 1, 1);
        assert!((pos.x - 0.15).abs() < 1e-12);
        assert!((pos.y - 0.30).abs() < 1e-12);
        assert!((pos.z - 0.60).abs() < 1e-12);
        assert_eq!(grid.position_to_index(&pos), (1, 1, 1));
    }
}
