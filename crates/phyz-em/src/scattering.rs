//! Scattering cross-sections from a closed flux surface.
//!
//! Total scattered power is the outward time-averaged Poynting flux over any
//! closed surface drawn in the scattered-field region of a TFSF simulation:
//!
//! ```text
//!   P_sca(ω) = ½ ∮ Re[ E(ω) × H*(ω) ] · n̂ dA
//! ```
//!
//! and the cross-section is that power divided by the incident intensity,
//! `σ_sca = P_sca / I_inc` with `I_inc = |E_inc|² / (2 η₀)`.
//!
//! The phasors are accumulated as running discrete Fourier transforms while the
//! simulation steps, so a single broadband run yields the cross-section at many
//! frequencies without storing the field history.
//!
//! # Accuracy
//!
//! Two effects dominate the error and neither is a bug:
//!
//! * **Yee staggering.** `E` and `H` are offset by half a cell and half a
//!   timestep. Both are corrected here — spatially by averaging the H
//!   components onto the E positions, temporally by rotating the H phasor by
//!   `e^{iωΔt/2}` — but only to leading order.
//! * **Staircasing.** A curved scatterer resolved on a cubic grid has a
//!   serrated surface. For a sphere this is the larger error by far and it
//!   converges only slowly, roughly first order in the cell size. Expect
//!   agreement at the ten-percent level for spheres of a few cells' radius,
//!   improving as the radius grows.

use crate::dispersion::C64;
use crate::grid::YeeGrid;

/// Accumulates field phasors on a closed box and reports scattered power.
///
/// The box must lie entirely inside the scattered-field region of a
/// [`crate::tfsf::Tfsf`] surface and outside the scatterer.
pub struct CrossSectionMonitor {
    /// Box bounds (inclusive) in grid indices.
    pub i0: usize,
    pub i1: usize,
    pub j0: usize,
    pub j1: usize,
    pub k0: usize,
    pub k1: usize,
    /// Angular frequencies being accumulated.
    pub omegas: Vec<f64>,
    /// Running DFT of the six field components at every sampled point,
    /// indexed `[frequency][point][component]`.
    acc: Vec<Vec<[C64; 6]>>,
    /// The sample points and their outward face normals.
    points: Vec<(usize, usize, usize, usize, f64)>, // i, j, k, axis, sign
    /// Steps accumulated so far.
    steps: usize,
    dt: f64,
    spacing: (f64, f64, f64),
}

impl CrossSectionMonitor {
    /// Build a monitor over the closed box `[i0, i1] × [j0, j1] × [k0, k1]`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grid: &YeeGrid,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        omegas: Vec<f64>,
    ) -> Self {
        let mut points = Vec::new();
        // ±x faces
        for k in k0..=k1 {
            for j in j0..=j1 {
                points.push((i0, j, k, 0, -1.0));
                points.push((i1, j, k, 0, 1.0));
            }
        }
        // ±y faces
        for k in k0..=k1 {
            for i in i0..=i1 {
                points.push((i, j0, k, 1, -1.0));
                points.push((i, j1, k, 1, 1.0));
            }
        }
        // ±z faces
        for j in j0..=j1 {
            for i in i0..=i1 {
                points.push((i, j, k0, 2, -1.0));
                points.push((i, j, k1, 2, 1.0));
            }
        }

        let n_pts = points.len();
        let acc = omegas
            .iter()
            .map(|_| vec![[C64::new(0.0, 0.0); 6]; n_pts])
            .collect();

        Self {
            i0,
            i1,
            j0,
            j1,
            k0,
            k1,
            omegas,
            acc,
            points,
            steps: 0,
            dt: grid.dt,
            spacing: (grid.dx, grid.dy, grid.dz),
        }
    }

    /// Sample the fields for the current step. Call once per timestep, after
    /// the field updates.
    ///
    /// `time` is the simulation time at the E-field level.
    pub fn record(&mut self, grid: &YeeGrid, time: f64) {
        for (fi, &omega) in self.omegas.iter().enumerate() {
            let (c, s) = ((omega * time).cos(), (omega * time).sin());
            for (pi, &(i, j, k, _, _)) in self.points.iter().enumerate() {
                // Average H onto the E positions to undo the half-cell offset.
                let vals = [
                    grid.ex.get(i, j, k),
                    grid.ey.get(i, j, k),
                    grid.ez.get(i, j, k),
                    avg_h(&grid.hx, i, j, k, 1, 2),
                    avg_h(&grid.hy, i, j, k, 2, 0),
                    avg_h(&grid.hz, i, j, k, 0, 1),
                ];
                let slot = &mut self.acc[fi][pi];
                for (dst, &v) in slot.iter_mut().zip(vals.iter()) {
                    *dst = *dst + C64::new(v * c, -v * s);
                }
            }
        }
        self.steps += 1;
    }

    /// Total scattered power (W) at each sampled frequency.
    pub fn scattered_power(&self) -> Vec<f64> {
        let norm = 2.0 / self.steps as f64;
        let (dx, dy, dz) = (self.dx(), self.dy(), self.dz());
        let area = [dy * dz, dx * dz, dx * dy];

        self.omegas
            .iter()
            .enumerate()
            .map(|(fi, &omega)| {
                // H is sampled half a timestep behind E; rotate its phasor to
                // the E time level.
                let half = omega * self.dt / 2.0;
                let rot = C64::new(half.cos(), half.sin());

                let mut total = 0.0;
                for (pi, &(_, _, _, axis, sign)) in self.points.iter().enumerate() {
                    let a = &self.acc[fi][pi];
                    let e = [a[0].scale(norm), a[1].scale(norm), a[2].scale(norm)];
                    let h = [
                        a[3].scale(norm) * rot,
                        a[4].scale(norm) * rot,
                        a[5].scale(norm) * rot,
                    ];
                    // ½ Re[E × H*] · n̂, keeping only the normal component.
                    let (p, q) = ((axis + 1) % 3, (axis + 2) % 3);
                    #[allow(clippy::needless_range_loop)]
                    let s_n = 0.5 * ((e[p] * h[q].conj()).re - (e[q] * h[p].conj()).re);
                    total += sign * s_n * area[axis];
                }
                total
            })
            .collect()
    }

    /// Scattering cross-section (m²) given the incident E-field phasor
    /// magnitude at each frequency, for a plane wave in vacuum.
    ///
    /// `σ_sca = P_sca / I_inc`, `I_inc = |E_inc|² / (2 η₀)`.
    pub fn cross_section(&self, incident_e: &[f64], eta0: f64) -> Vec<f64> {
        self.scattered_power()
            .iter()
            .zip(incident_e.iter())
            .map(|(p, e)| {
                if *e == 0.0 {
                    f64::NAN
                } else {
                    p / (e * e / (2.0 * eta0))
                }
            })
            .collect()
    }

    fn dx(&self) -> f64 {
        self.spacing.0
    }
    fn dy(&self) -> f64 {
        self.spacing.1
    }
    fn dz(&self) -> f64 {
        self.spacing.2
    }
}

/// Average an H component onto the co-located E position by undoing its two
/// half-cell offsets (`axis_a` and `axis_b` are the two staggered directions).
fn avg_h(
    arr: &crate::grid::Array3D,
    i: usize,
    j: usize,
    k: usize,
    axis_a: usize,
    axis_b: usize,
) -> f64 {
    let idx = [i, j, k];
    let back = |a: usize, mut v: [usize; 3]| -> Option<[usize; 3]> {
        if v[a] == 0 {
            None
        } else {
            v[a] -= 1;
            Some(v)
        }
    };
    let mut sum = arr.get(idx[0], idx[1], idx[2]);
    let mut n = 1.0;
    if let Some(v) = back(axis_a, idx) {
        sum += arr.get(v[0], v[1], v[2]);
        n += 1.0;
        if let Some(w) = back(axis_b, v) {
            sum += arr.get(w[0], w[1], w[2]);
            n += 1.0;
        }
    }
    if let Some(v) = back(axis_b, idx) {
        sum += arr.get(v[0], v[1], v[2]);
        n += 1.0;
    }
    sum / n
}

/// Rayleigh (small-sphere) scattering cross-section.
///
/// Valid for `k a ≪ 1`:
///
/// ```text
///   σ_sca = (8/3) π k⁴ a⁶ |(m² − 1)/(m² + 2)|²
/// ```
///
/// where `m` is the refractive index of the sphere relative to the surrounding
/// medium. This is the leading term of the Mie series; use it only where the
/// size parameter is small.
pub fn rayleigh_cross_section(k: f64, radius: f64, m: f64) -> f64 {
    let alpha = (m * m - 1.0) / (m * m + 2.0);
    (8.0 / 3.0) * std::f64::consts::PI * k.powi(4) * radius.powi(6) * alpha * alpha
}
