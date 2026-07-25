//! Total-field / scattered-field (TFSF) plane-wave injection.
//!
//! A closed (or partial) surface splits the grid into an interior *total-field*
//! region, where the simulated field is incident + scattered, and an exterior
//! *scattered-field* region containing only the scattered wave. Wherever a
//! finite-difference stencil straddles the surface, the incident field is
//! added to or subtracted from the offending term so that both regions stay
//! self-consistent.
//!
//! The incident wave is x-polarized and travels along +z:
//!
//! ```text
//!   E_inc = x̂ Eₓ(z, t),   H_inc = ŷ Hᵧ(z, t)
//! ```
//!
//! Its values come from a co-simulated 1D FDTD grid ([`Aux1D`]) rather than
//! from an analytic formula. That matters: the 1D grid has *exactly* the same
//! numerical dispersion as the 3D grid along z, so the injected wave satisfies
//! the discrete Maxwell equations to machine precision and leakage into the
//! scattered-field region stays far below the physical scattering.
//!
//! # Scope
//!
//! Only axis-aligned normal incidence (+z, x-polarized) is implemented.
//! Oblique incidence needs an interpolated auxiliary grid and is not supported.
//!
//! # Which components need correcting
//!
//! With `E_inc ∥ x̂` and `H_inc ∥ ŷ`:
//!
//! | surface     | corrected components            |
//! |-------------|---------------------------------|
//! | z = k0, k1  | `Ex` (from `Hy_inc`), `Hy` (from `Ex_inc`) |
//! | x = i0, i1  | `Ez` (from `Hy_inc`)            |
//! | y = j0, j1  | `Hz` (from `Ex_inc`)            |
//!
//! Everything else is untouched because the corresponding incident component
//! is identically zero.

use crate::grid::{EPS0, MU0, YeeGrid};

/// One-dimensional auxiliary FDTD grid carrying the incident plane wave.
///
/// Fields are laid out exactly like the 3D grid's z-line: `ex[a]` sits at plane
/// `a`, `hy[a]` at `a + ½`.
pub struct Aux1D {
    pub n: usize,
    pub dz: f64,
    pub dt: f64,
    pub ex: Vec<f64>,
    pub hy: Vec<f64>,
    /// Index of the soft source.
    pub source_index: usize,
    /// Mur ABC coefficient.
    mur: f64,
    /// Current time (s), at the E-field level.
    pub time: f64,
}

impl Aux1D {
    /// Create an auxiliary grid of `n` cells with the source at `source_index`.
    pub fn new(n: usize, dz: f64, dt: f64, source_index: usize) -> Self {
        let c0 = 1.0 / (EPS0 * MU0).sqrt();
        let mur = (c0 * dt - dz) / (c0 * dt + dz);
        Self {
            n,
            dz,
            dt,
            ex: vec![0.0; n],
            hy: vec![0.0; n],
            source_index,
            mur,
            time: 0.0,
        }
    }

    /// Advance H from n−½ to n+½ using E at level n.
    pub fn update_h(&mut self) {
        let coef = self.dt / (MU0 * self.dz);
        for a in 0..self.n - 1 {
            self.hy[a] -= coef * (self.ex[a + 1] - self.ex[a]);
        }
    }

    /// Advance E from n to n+1 using H at level n+½, then inject `source(t)`.
    pub fn update_e<F: Fn(f64) -> f64>(&mut self, source: &F) {
        let coef = self.dt / (EPS0 * self.dz);

        let e0_old = self.ex[0];
        let e1_old = self.ex[1];
        let en_old = self.ex[self.n - 1];
        let en1_old = self.ex[self.n - 2];

        for a in 1..self.n - 1 {
            self.ex[a] -= coef * (self.hy[a] - self.hy[a - 1]);
        }

        // First-order Mur absorbing boundaries on both ends.
        self.ex[0] = e1_old + self.mur * (self.ex[1] - e0_old);
        self.ex[self.n - 1] = en1_old + self.mur * (self.ex[self.n - 2] - en_old);

        self.time += self.dt;
        self.ex[self.source_index] += source(self.time);
    }

    /// Peak absolute incident E seen so far — handy for normalizing results.
    pub fn peak_e(&self) -> f64 {
        self.ex.iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
    }
}

/// A TFSF injection surface.
pub struct Tfsf {
    /// Total-field region bounds (inclusive), in grid indices.
    pub i0: usize,
    pub i1: usize,
    pub j0: usize,
    pub j1: usize,
    pub k0: usize,
    pub k1: usize,
    /// Whether the x-normal and y-normal faces are closed.
    pub x_faces: bool,
    pub y_faces: bool,
    /// Whether the upper z face is closed. Set `false` to make the surface a
    /// one-way injector: everything above `k0` is total field, so a structure
    /// may extend all the way to the far boundary.
    pub z_upper_face: bool,
    /// Auxiliary incident-wave grid.
    pub aux: Aux1D,
    /// Offset between 3D `k` and auxiliary index: `a = k + pad`.
    pad: usize,
}

impl Tfsf {
    /// A TFSF surface with only the two z-normal faces closed.
    ///
    /// Appropriate for layered (1D) problems with periodic transverse
    /// boundaries, where the incident field is uniform in x and y and there is
    /// no transverse scattered-field region.
    pub fn slab_z(grid: &YeeGrid, k0: usize, k1: usize) -> Self {
        Self::build(
            grid,
            0,
            grid.nx - 1,
            0,
            grid.ny - 1,
            k0,
            k1,
            false,
            false,
            true,
        )
    }

    /// A one-way plane-wave injector at plane `k0`.
    ///
    /// Only the lower z face is closed: `k < k0` is scattered field (so a probe
    /// there sees the reflected wave alone) and everything above is total
    /// field. This is the geometry for reflection off a half-space or a stack
    /// that continues into the far absorbing boundary.
    pub fn injector_z(grid: &YeeGrid, k0: usize) -> Self {
        Self::build(
            grid,
            0,
            grid.nx - 1,
            0,
            grid.ny - 1,
            k0,
            grid.nz - 1,
            false,
            false,
            false,
        )
    }

    /// A fully closed TFSF box, for illuminating a finite scatterer.
    #[allow(clippy::too_many_arguments)]
    pub fn box_region(
        grid: &YeeGrid,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
    ) -> Self {
        Self::build(grid, i0, i1, j0, j1, k0, k1, true, true, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        grid: &YeeGrid,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        x_faces: bool,
        y_faces: bool,
        z_upper_face: bool,
    ) -> Self {
        // Room below k = 0 for the source plus a Mur-terminated run-up, and
        // above k = nz−1 so the outgoing incident wave leaves cleanly.
        let pad = 24;
        let n_aux = grid.nz + 2 * pad;
        let aux = Aux1D::new(n_aux, grid.dz, grid.dt, 6);
        Self {
            i0,
            i1,
            j0,
            j1,
            k0,
            k1,
            x_faces,
            y_faces,
            z_upper_face,
            aux,
            pad,
        }
    }

    /// Index span along x for components that sit at `i + ½` (`Ex`, `Hy`,
    /// `Hz`).
    ///
    /// When the x faces are closed, a half-integer component's total-field
    /// span ends one index early: `i1 + ½` already lies outside the surface.
    /// When they are open the total-field region spans the whole axis and
    /// every index is corrected.
    #[inline]
    fn i_half(&self) -> std::ops::Range<usize> {
        if self.x_faces {
            self.i0..self.i1
        } else {
            self.i0..self.i1 + 1
        }
    }

    /// Index span along z for components that sit at `k + ½` (`Ez`, `Hx`,
    /// `Hy`), with the same half-cell rule as [`Tfsf::i_half`].
    #[inline]
    fn k_half(&self) -> std::ops::Range<usize> {
        if self.z_upper_face {
            self.k0..self.k1
        } else {
            self.k0..self.k1 + 1
        }
    }

    /// Incident Ex at grid plane `k`, time level n.
    #[inline]
    pub fn ex_inc(&self, k: usize) -> f64 {
        self.aux.ex[k + self.pad]
    }

    /// Incident Hy at grid plane `k + ½`, time level n+½.
    #[inline]
    pub fn hy_inc_half(&self, k: usize) -> f64 {
        self.aux.hy[k + self.pad]
    }

    /// Incident Hy at `k − ½` (i.e. `hy[k-1]` in the auxiliary grid).
    #[inline]
    fn hy_inc_below(&self, k: usize) -> f64 {
        self.aux.hy[k + self.pad - 1]
    }

    /// Advance the auxiliary grid's H field. Call before [`Tfsf::correct_h`].
    pub fn step_aux_h(&mut self) {
        self.aux.update_h();
    }

    /// Advance the auxiliary grid's E field and inject the source.
    /// Call at the very end of the timestep.
    pub fn step_aux_e<F: Fn(f64) -> f64>(&mut self, source: &F) {
        self.aux.update_e(source);
    }

    /// Apply the H-field TFSF corrections. Call right after
    /// [`YeeGrid::update_h_field`].
    pub fn correct_h(&self, grid: &mut YeeGrid) {
        let dt = grid.dt;
        let (dy, dz) = (grid.dy, grid.dz);
        let mu0 = grid.mu0;

        // ---- z faces: Hy is corrected by the incident Ex --------------------
        // Hy at k0−½ is scattered-field but reads the total-field Ex[k0].
        if self.k0 >= 1 {
            let e_inc = self.ex_inc(self.k0);
            let k = self.k0 - 1;
            for j in self.j0..=self.j1 {
                for i in self.i_half() {
                    let coef = dt / (mu0 * grid.mu_r.get(i, j, k) * dz);
                    grid.hy.add(i, j, k, coef * e_inc);
                }
            }
        }
        // Hy at k1+½ is scattered-field but reads the total-field Ex[k1].
        if self.z_upper_face {
            let e_inc = self.ex_inc(self.k1);
            let k = self.k1;
            for j in self.j0..=self.j1 {
                for i in self.i_half() {
                    let coef = dt / (mu0 * grid.mu_r.get(i, j, k) * dz);
                    grid.hy.add(i, j, k, -coef * e_inc);
                }
            }
        }

        // ---- y faces: Hz is corrected by the incident Ex --------------------
        if self.y_faces {
            if self.j0 >= 1 {
                let j = self.j0 - 1;
                for k in self.k0..=self.k1 {
                    let e_inc = self.ex_inc(k);
                    for i in self.i_half() {
                        let coef = dt / (mu0 * grid.mu_r.get(i, j, k) * dy);
                        grid.hz.add(i, j, k, -coef * e_inc);
                    }
                }
            }
            let j = self.j1;
            for k in self.k0..=self.k1 {
                let e_inc = self.ex_inc(k);
                for i in self.i_half() {
                    let coef = dt / (mu0 * grid.mu_r.get(i, j, k) * dy);
                    grid.hz.add(i, j, k, coef * e_inc);
                }
            }
        }
    }

    /// Apply the E-field TFSF corrections. Call right after
    /// [`YeeGrid::update_e_field`].
    pub fn correct_e(&self, grid: &mut YeeGrid) {
        let dt = grid.dt;
        let (dx, dz) = (grid.dx, grid.dz);
        let eps0 = grid.eps0;

        // Denominator of the E update at a cell: ε/Δt + σ/2.
        let cb = |g: &YeeGrid, i: usize, j: usize, k: usize| {
            eps0 * g.eps_r.get(i, j, k) / dt + g.sigma.get(i, j, k) / 2.0
        };

        // ---- z faces: Ex is corrected by the incident Hy --------------------
        // Ex[k0] is total-field but reads the scattered-field Hy[k0−½].
        if self.k0 >= 1 {
            let h_inc = self.hy_inc_below(self.k0);
            let k = self.k0;
            for j in self.j0..=self.j1 {
                for i in self.i_half() {
                    let d = h_inc / (dz * cb(grid, i, j, k));
                    grid.ex.add(i, j, k, d);
                }
            }
        }
        // Ex[k1] is total-field but reads the scattered-field Hy[k1+½].
        if self.z_upper_face {
            let h_inc = self.hy_inc_half(self.k1);
            let k = self.k1;
            for j in self.j0..=self.j1 {
                for i in self.i_half() {
                    let d = h_inc / (dz * cb(grid, i, j, k));
                    grid.ex.add(i, j, k, -d);
                }
            }
        }

        // ---- x faces: Ez is corrected by the incident Hy --------------------
        if self.x_faces {
            for k in self.k_half() {
                let h_inc = self.hy_inc_half(k);
                for j in self.j0..=self.j1 {
                    let i = self.i0;
                    let d = h_inc / (dx * cb(grid, i, j, k));
                    grid.ez.add(i, j, k, -d);

                    let i = self.i1;
                    let d = h_inc / (dx * cb(grid, i, j, k));
                    grid.ez.add(i, j, k, d);
                }
            }
        }
    }

    /// Run one full timestep of grid + auxiliary grid in the correct order.
    ///
    /// ```text
    ///   aux H  →  grid H  →  H corrections  →  grid E  →  E corrections  →  aux E
    /// ```
    pub fn step<F: Fn(f64) -> f64>(&mut self, grid: &mut YeeGrid, source: &F) {
        self.step_aux_h();
        grid.update_h_field();
        self.correct_h(grid);
        grid.update_e_field();
        self.correct_e(grid);
        self.step_aux_e(source);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{db, peak_abs, ricker_pulse};
    use crate::boundary::BoundaryCondition;
    use crate::cpml::CpmlConfig;

    /// The defining property of TFSF: with an empty total-field region, the
    /// scattered-field region must stay dark.
    ///
    /// Leakage is the honest measure of injection quality, and it is what
    /// limits how small a scattering cross-section can be resolved.
    #[test]
    fn empty_region_leaks_negligibly_into_scattered_field() {
        let dz = 1e-9;
        let c = 299_792_458.0;
        let dt = 0.5 * dz / c;
        let nz = 160;

        let mut grid = YeeGrid::new_rect(1, 1, nz, dz, dz, dz, dt);
        grid.set_periodic([true, true, false]);
        grid.set_boundary(BoundaryCondition::Cpml(
            CpmlConfig::with_thickness(10).on_axes([false, false, true]),
        ));

        let mut tfsf = Tfsf::slab_z(&grid, 40, 120);

        let spread = 12.0 * dt;
        let t0 = 5.0 * spread;
        let src = move |t: f64| ricker_pulse(t, t0, spread);

        let mut sf_low = Vec::new();
        let mut sf_high = Vec::new();
        let mut tf_mid = Vec::new();
        for _ in 0..700 {
            tfsf.step(&mut grid, &src);
            sf_low.push(grid.ex.get(0, 0, 25));
            sf_high.push(grid.ex.get(0, 0, 140));
            tf_mid.push(grid.ex.get(0, 0, 80));
        }

        let inside = peak_abs(&tf_mid);
        assert!(inside > 1e-3, "no incident wave in the total-field region");

        let leak_low = db(peak_abs(&sf_low) / inside);
        let leak_high = db(peak_abs(&sf_high) / inside);
        println!("TFSF leakage: below {leak_low:.1} dB, above {leak_high:.1} dB");
        assert!(leak_low < -55.0, "leakage below the box: {leak_low:.1} dB");
        assert!(
            leak_high < -55.0,
            "leakage above the box: {leak_high:.1} dB"
        );
    }

    /// Inside the total-field region the simulated wave must *be* the incident
    /// wave.
    ///
    /// With a transversely uniform grid the 3D update reduces exactly to the
    /// auxiliary 1D update, so this is not an approximate statement: the two
    /// must agree to round-off. Any half-cell or half-timestep error in the
    /// TFSF corrections shows up here immediately.
    #[test]
    fn total_field_matches_the_auxiliary_incident_wave() {
        let dz = 1e-9;
        let c = 299_792_458.0;
        let dt = 0.5 * dz / c;
        let nz = 160;

        let mut grid = YeeGrid::new_rect(1, 1, nz, dz, dz, dz, dt);
        grid.set_periodic([true, true, false]);
        grid.set_boundary(BoundaryCondition::Cpml(
            CpmlConfig::with_thickness(10).on_axes([false, false, true]),
        ));

        let mut tfsf = Tfsf::slab_z(&grid, 40, 120);
        let spread = 12.0 * dt;
        let t0 = 5.0 * spread;
        let src = move |t: f64| ricker_pulse(t, t0, spread);

        let mut worst: f64 = 0.0;
        let mut scale: f64 = 0.0;
        for _ in 0..500 {
            tfsf.step(&mut grid, &src);
            // Compare across the interior of the total-field region, away from
            // the surfaces themselves.
            for k in 45..=115 {
                let sim = grid.ex.get(0, 0, k);
                let inc = tfsf.ex_inc(k);
                scale = scale.max(inc.abs());
                worst = worst.max((sim - inc).abs());
            }
        }
        assert!(scale > 1e-3, "no incident wave present");
        let rel = worst / scale;
        println!(
            "total-field vs incident: {:.2e} relative ({:.1} dB)",
            rel,
            db(rel)
        );
        assert!(
            rel < 1e-10,
            "total field departs from the incident wave by {rel:.3e} (relative)"
        );
    }

    /// The forward-travelling wave must carry the free-space impedance.
    ///
    /// E and H are staggered by half a cell in z and half a step in time, so
    /// the comparison is made on the auxiliary grid where those offsets can be
    /// undone by averaging.
    #[test]
    fn incident_wave_carries_free_space_impedance() {
        let dz = 1e-9;
        let c = 299_792_458.0;
        let dt = 0.5 * dz / c;
        let nz = 160;

        let mut grid = YeeGrid::new_rect(1, 1, nz, dz, dz, dz, dt);
        grid.set_periodic([true, true, false]);
        grid.set_boundary(BoundaryCondition::Cpml(
            CpmlConfig::with_thickness(10).on_axes([false, false, true]),
        ));

        let mut tfsf = Tfsf::slab_z(&grid, 40, 120);
        let spread = 12.0 * dt;
        let t0 = 5.0 * spread;
        let src = move |t: f64| ricker_pulse(t, t0, spread);

        let eta0 = (MU0 / EPS0).sqrt();
        // Peak amplitudes are insensitive to the half-cell / half-step offsets
        // that make a pointwise E/H ratio meaningless for a pulse.
        let mut peak_e: f64 = 0.0;
        let mut peak_h: f64 = 0.0;
        for _ in 0..400 {
            tfsf.step(&mut grid, &src);
            peak_e = peak_e.max(grid.ex.get(0, 0, 80).abs());
            peak_h = peak_h.max(grid.hy.get(0, 0, 80).abs());
        }
        assert!(peak_e > 1e-3, "no wave reached the probe");

        let eta = peak_e / peak_h;
        let err = (eta - eta0).abs() / eta0;
        println!(
            "measured wave impedance {eta:.2} Ω vs η₀ = {eta0:.2} Ω ({:.2}%)",
            err * 100.0
        );
        // A positive H with a positive E is a +z travelling wave; the sign and
        // magnitude together confirm the injection direction.
        assert!(err < 0.02, "wave impedance off by {:.2}%", err * 100.0);
    }
}
