//! Convolutional Perfectly Matched Layer (CPML) absorbing boundary.
//!
//! Implements the complex-frequency-shifted (CFS) stretched-coordinate PML of
//! Roden & Gedney, *Convolutional PML (CPML): an efficient FDTD implementation
//! of the CFS-PML for arbitrary media*, Microwave Opt. Tech. Lett. 27 (2000).
//!
//! Unlike a plain graded-conductivity absorber (which is *not* matched and
//! reflects strongly), the CPML stretches the coordinate in the complex plane,
//!
//! ```text
//!   s_w(ω) = κ_w + σ_w / (α_w + i ω ε₀)
//! ```
//!
//! and folds the resulting convolution into a recursive auxiliary variable ψ.
//! For each field component and each derivative direction, the spatial
//! derivative used in the curl is replaced by
//!
//! ```text
//!   ∂/∂w  →  (1/κ_w) ∂/∂w + ψ_w
//!   ψ_w^{n+1} = b_w ψ_w^n + c_w (∂/∂w at step n+1)
//! ```
//!
//! with
//!
//! ```text
//!   b_w = exp(-((σ_w/κ_w) + α_w) Δt / ε₀)
//!   c_w = σ_w (b_w − 1) / (κ_w (σ_w + κ_w α_w))
//! ```
//!
//! The κ term improves absorption at grazing incidence and the α term (the
//! "complex frequency shift") makes the layer absorb evanescent waves and
//! suppresses the late-time drift that plagues Berenger's split-field PML.
//!
//! Both the magnetic and electric updates use the same stretching factors, so
//! the impedance-matching condition σ*/μ = σ/ε is satisfied *by construction* —
//! no separate magnetic loss term needs to be tuned.

use crate::grid::{Array3D, EPS0, MU0};

/// Configuration for a CPML boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpmlConfig {
    /// Layer thickness in cells.
    pub thickness: usize,
    /// Polynomial grading order for σ and κ (m). 3–4 is standard.
    pub order: f64,
    /// Maximum coordinate stretch κ at the outer wall (≥ 1).
    pub kappa_max: f64,
    /// Complex-frequency shift α at the *inner* PML surface, expressed as a
    /// dimensionless multiple of `ε₀ c / d_pml` (the inverse layer transit
    /// time).
    ///
    /// This must scale with the grid: a fixed value in S/m that is sensible at
    /// metre scale is utterly negligible on a nanometre grid, where σ_max is
    /// of order 10⁷ S/m. Expressing α relative to the layer transit time keeps
    /// the complex-frequency shift doing its job — absorbing evanescent and
    /// very-low-frequency content — at any scale.
    pub alpha_factor: f64,
    /// Grading order for α (usually 1).
    pub alpha_order: f64,
    /// Target theoretical reflection at normal incidence, used to pick σ_max.
    pub target_reflection: f64,
    /// Scale factor on the analytically optimal σ_max (1.0 = as computed).
    pub sigma_scale: f64,
    /// Which axes get a PML. Axes marked `false` are left untouched
    /// (e.g. periodic or PEC-terminated directions).
    pub axes: [bool; 3],
}

impl Default for CpmlConfig {
    fn default() -> Self {
        Self {
            thickness: 10,
            order: 3.0,
            kappa_max: 5.0,
            alpha_factor: 0.2,
            alpha_order: 1.0,
            target_reflection: 1e-8,
            sigma_scale: 1.0,
            axes: [true; 3],
        }
    }
}

impl CpmlConfig {
    /// A CPML of the given thickness on all three axes, otherwise default.
    pub fn with_thickness(thickness: usize) -> Self {
        Self {
            thickness,
            ..Self::default()
        }
    }

    /// Restrict the PML to the given axes.
    pub fn on_axes(mut self, axes: [bool; 3]) -> Self {
        self.axes = axes;
        self
    }
}

/// Per-axis CPML coefficient set (one entry per grid position along the axis).
#[derive(Debug, Clone)]
struct AxisCoeffs {
    /// Inverse coordinate stretch 1/κ at E-field (integer) positions.
    kinv_e: Vec<f64>,
    b_e: Vec<f64>,
    c_e: Vec<f64>,
    /// Same at H-field (half-integer) positions.
    kinv_h: Vec<f64>,
    b_h: Vec<f64>,
    c_h: Vec<f64>,
    /// True when this axis has any absorbing cells at all.
    active: bool,
}

impl AxisCoeffs {
    fn inactive(n: usize) -> Self {
        Self {
            kinv_e: vec![1.0; n],
            b_e: vec![0.0; n],
            c_e: vec![0.0; n],
            kinv_h: vec![1.0; n],
            b_h: vec![0.0; n],
            c_h: vec![0.0; n],
            active: false,
        }
    }

    fn build(n: usize, d: f64, dt: f64, cfg: &CpmlConfig) -> Self {
        let thk = cfg.thickness as f64;
        if cfg.thickness == 0 || n < 2 * cfg.thickness + 2 {
            return Self::inactive(n);
        }

        let eta0 = (MU0 / EPS0).sqrt();
        // σ_max from the analytic normal-incidence reflection of a graded layer:
        //   R(0) = exp(-2 η₀ ∫σ dw)  with σ(u) = σ_max u^m over depth d_pml
        let d_pml = thk * d;
        let sigma_max = -(cfg.order + 1.0) * cfg.target_reflection.ln()
            / (2.0 * eta0 * d_pml)
            * cfg.sigma_scale;
        let c0 = 1.0 / (EPS0 * MU0).sqrt();
        let alpha_max = cfg.alpha_factor * EPS0 * c0 / d_pml;

        // Normalized depth into the PML for a point at `dist` cells from the
        // domain wall (0 at the wall, growing inward). Returns 0 outside.
        let depth = |dist: f64| -> f64 {
            let u = (thk - dist) / thk;
            u.clamp(0.0, 1.0)
        };

        let mut me = Self::inactive(n);
        me.active = true;

        let fill = |kinv: &mut Vec<f64>, b: &mut Vec<f64>, c: &mut Vec<f64>, half: f64| {
            for idx in 0..n {
                let pos = idx as f64 + half;
                // Distance (in cells) from the lower wall (position 0) and the
                // upper wall (position n-1).
                let u_lo = depth(pos);
                let u_hi = depth((n - 1) as f64 - pos);
                let u = u_lo.max(u_hi);

                if u <= 0.0 {
                    kinv[idx] = 1.0;
                    b[idx] = 1.0;
                    c[idx] = 0.0;
                    continue;
                }

                let sigma = sigma_max * u.powf(cfg.order);
                let kappa = 1.0 + (cfg.kappa_max - 1.0) * u.powf(cfg.order);
                let alpha = alpha_max * (1.0 - u).powf(cfg.alpha_order);

                let bb = (-((sigma / kappa) + alpha) * dt / EPS0).exp();
                let denom = kappa * (sigma + kappa * alpha);
                let cc = if denom.abs() > 0.0 {
                    sigma * (bb - 1.0) / denom
                } else {
                    0.0
                };

                kinv[idx] = 1.0 / kappa;
                b[idx] = bb;
                c[idx] = cc;
            }
        };

        let mut kinv_e = vec![1.0; n];
        let mut b_e = vec![1.0; n];
        let mut c_e = vec![0.0; n];
        fill(&mut kinv_e, &mut b_e, &mut c_e, 0.0);

        let mut kinv_h = vec![1.0; n];
        let mut b_h = vec![1.0; n];
        let mut c_h = vec![0.0; n];
        fill(&mut kinv_h, &mut b_h, &mut c_h, 0.5);

        me.kinv_e = kinv_e;
        me.b_e = b_e;
        me.c_e = c_e;
        me.kinv_h = kinv_h;
        me.b_h = b_h;
        me.c_h = c_h;
        me
    }
}

/// CPML state: per-axis coefficients plus the twelve ψ convolution arrays.
pub struct Cpml {
    pub config: CpmlConfig,
    axis: [AxisCoeffs; 3],

    // ψ for the electric-field updates, named ψ_{field}{derivative direction}.
    psi_exy: Array3D,
    psi_exz: Array3D,
    psi_eyz: Array3D,
    psi_eyx: Array3D,
    psi_ezx: Array3D,
    psi_ezy: Array3D,

    // ψ for the magnetic-field updates.
    psi_hxy: Array3D,
    psi_hxz: Array3D,
    psi_hyz: Array3D,
    psi_hyx: Array3D,
    psi_hzx: Array3D,
    psi_hzy: Array3D,
}

impl Cpml {
    /// Build CPML state for a grid of the given size and spacing.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        config: CpmlConfig,
    ) -> Self {
        let mk = |n, d, on| {
            if on {
                AxisCoeffs::build(n, d, dt, &config)
            } else {
                AxisCoeffs::inactive(n)
            }
        };
        let axis = [
            mk(nx, dx, config.axes[0]),
            mk(ny, dy, config.axes[1]),
            mk(nz, dz, config.axes[2]),
        ];

        let z = || Array3D::zeros(nx, ny, nz);
        Self {
            config,
            axis,
            psi_exy: z(),
            psi_exz: z(),
            psi_eyz: z(),
            psi_eyx: z(),
            psi_ezx: z(),
            psi_ezy: z(),
            psi_hxy: z(),
            psi_hxz: z(),
            psi_hyz: z(),
            psi_hyx: z(),
            psi_hzx: z(),
            psi_hzy: z(),
        }
    }

    /// True if at least one axis actually absorbs.
    pub fn is_active(&self) -> bool {
        self.axis.iter().any(|a| a.active)
    }

    /// Zero all convolution memory.
    pub fn reset(&mut self) {
        for a in [
            &mut self.psi_exy,
            &mut self.psi_exz,
            &mut self.psi_eyz,
            &mut self.psi_eyx,
            &mut self.psi_ezx,
            &mut self.psi_ezy,
            &mut self.psi_hxy,
            &mut self.psi_hxz,
            &mut self.psi_hyz,
            &mut self.psi_hyx,
            &mut self.psi_hzx,
            &mut self.psi_hzy,
        ] {
            a.clear();
        }
    }

    /// Inverse stretch 1/κ at an E-position on `axis`.
    #[inline]
    pub fn kinv_e(&self, axis: usize, idx: usize) -> f64 {
        self.axis[axis].kinv_e[idx]
    }

    /// Inverse stretch 1/κ at an H-position on `axis`.
    #[inline]
    pub fn kinv_h(&self, axis: usize, idx: usize) -> f64 {
        self.axis[axis].kinv_h[idx]
    }
}

/// Which ψ array a CPML update refers to.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Psi {
    Exy,
    Exz,
    Eyz,
    Eyx,
    Ezx,
    Ezy,
    Hxy,
    Hxz,
    Hyz,
    Hyx,
    Hzx,
    Hzy,
}

impl Cpml {
    #[inline]
    pub(crate) fn psi_mut(&mut self, which: Psi) -> &mut Array3D {
        match which {
            Psi::Exy => &mut self.psi_exy,
            Psi::Exz => &mut self.psi_exz,
            Psi::Eyz => &mut self.psi_eyz,
            Psi::Eyx => &mut self.psi_eyx,
            Psi::Ezx => &mut self.psi_ezx,
            Psi::Ezy => &mut self.psi_ezy,
            Psi::Hxy => &mut self.psi_hxy,
            Psi::Hxz => &mut self.psi_hxz,
            Psi::Hyz => &mut self.psi_hyz,
            Psi::Hyx => &mut self.psi_hyx,
            Psi::Hzx => &mut self.psi_hzx,
            Psi::Hzy => &mut self.psi_hzy,
        }
    }

    /// Advance one ψ variable and return its new value.
    ///
    /// `deriv` is the *already scaled* spatial derivative (∂F/∂w, in SI units).
    /// `e_side` selects the E-grid (integer) or H-grid (half-integer)
    /// coefficient set along `axis`.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn advance(
        &mut self,
        which: Psi,
        axis: usize,
        idx: usize,
        e_side: bool,
        i: usize,
        j: usize,
        k: usize,
        deriv: f64,
    ) -> f64 {
        let (b, c) = {
            let a = &self.axis[axis];
            if e_side {
                (a.b_e[idx], a.c_e[idx])
            } else {
                (a.b_h[idx], a.c_h[idx])
            }
        };
        if c == 0.0 {
            // Outside the layer ψ stays zero for all time; skip the store.
            return 0.0;
        }
        let arr = self.psi_mut(which);
        let v = b * arr.get(i, j, k) + c * deriv;
        arr.set(i, j, k, v);
        v
    }

    /// True when `axis` has an active PML (lets the update loops skip the work).
    #[inline]
    pub(crate) fn axis_active(&self, axis: usize) -> bool {
        self.axis[axis].active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficients_are_passive_and_graded() {
        let dx = 1e-9;
        let dt = dx / (3e8 * 3_f64.sqrt() * 1.1);
        let cfg = CpmlConfig::with_thickness(10);
        let c = Cpml::new(64, 64, 64, dx, dx, dx, dt, cfg);

        assert!(c.is_active());

        // κ is 1 in the interior and grows toward the wall.
        assert!((c.kinv_e(0, 32) - 1.0).abs() < 1e-12);
        assert!(c.kinv_e(0, 0) < c.kinv_e(0, 5));
        assert!(c.kinv_e(0, 5) < c.kinv_e(0, 32));

        // b must be in (0, 1]: it is a decaying memory factor, never amplifying.
        for idx in 0..64 {
            let b = c.axis[0].b_e[idx];
            assert!(b > 0.0 && b <= 1.0 + 1e-15, "b_e[{idx}] = {b}");
            let bh = c.axis[0].b_h[idx];
            assert!(bh > 0.0 && bh <= 1.0 + 1e-15, "b_h[{idx}] = {bh}");
        }

        // Symmetry: the layer at each end should mirror.
        for d in 0..10 {
            let lo = c.kinv_e(0, d);
            let hi = c.kinv_e(0, 63 - d);
            assert!((lo - hi).abs() < 1e-12, "asymmetric at depth {d}");
        }
    }

    #[test]
    fn thin_grid_disables_layer() {
        let dx = 1e-9;
        let dt = dx / (3e8 * 3_f64.sqrt() * 1.1);
        // 8 cells cannot host two 10-cell layers.
        let c = Cpml::new(8, 8, 8, dx, dx, dx, dt, CpmlConfig::with_thickness(10));
        assert!(!c.is_active());
    }

    #[test]
    fn axes_can_be_selected() {
        let dx = 1e-9;
        let dt = dx / (3e8 * 3_f64.sqrt() * 1.1);
        let cfg = CpmlConfig::with_thickness(10).on_axes([false, false, true]);
        let c = Cpml::new(64, 64, 64, dx, dx, dx, dt, cfg);
        assert!(!c.axis_active(0));
        assert!(!c.axis_active(1));
        assert!(c.axis_active(2));
    }
}
