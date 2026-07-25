//! FDTD solver with sources and observables.

use phyz_math::Vec3;

use crate::boundary::BoundaryCondition;
use crate::grid::YeeGrid;
use crate::source::{Probe, Source};

/// Power flowing through the z-plane at index `k` (W), `S = E × H`.
pub fn poynting_flux_z(grid: &YeeGrid, k: usize) -> f64 {
    let da = grid.dx * grid.dy;
    let mut flux = 0.0;
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let e = grid.get_e_field(i, j, k);
            let h = grid.get_h_field(i, j, k);
            flux += e.cross(h).z * da;
        }
    }
    flux
}

/// FDTD solver for Maxwell's equations.
pub struct FdtdSolver {
    /// Yee grid.
    pub grid: YeeGrid,

    /// Electromagnetic sources.
    pub sources: Vec<Source>,

    /// Field probes.
    pub probes: Vec<Probe>,

    /// Boundary condition.
    pub boundary: BoundaryCondition,

    /// Current simulation time (s).
    pub time: f64,

    /// Step count.
    pub step: usize,
}

impl FdtdSolver {
    /// Create a new FDTD solver.
    pub fn new(grid: YeeGrid) -> Self {
        let boundary = BoundaryCondition::default();

        Self {
            grid,
            sources: Vec::new(),
            probes: Vec::new(),
            boundary,
            time: 0.0,
            step: 0,
        }
    }

    /// Add a source to the simulation.
    pub fn add_source(&mut self, source: Source) {
        self.sources.push(source);
    }

    /// Add a field probe at given position.
    pub fn add_probe(&mut self, pos: Vec3) -> usize {
        let probe = Probe::new(&self.grid, pos);
        self.probes.push(probe);
        self.probes.len() - 1
    }

    /// Set boundary condition.
    pub fn set_boundary(&mut self, boundary: BoundaryCondition) {
        self.boundary = boundary;
        self.grid.set_boundary(boundary);
    }

    /// Advance simulation by one timestep.
    ///
    /// Boundary treatment (PEC walls, periodic wrapping, CPML convolution) is
    /// built into the field updates, so there is no separate boundary pass.
    pub fn step(&mut self) {
        self.grid.update_h_field();
        self.grid.update_e_field();

        for source in &self.sources {
            source.apply(&mut self.grid, self.time);
        }

        for probe in &mut self.probes {
            probe.record(&self.grid);
        }

        self.time += self.grid.dt;
        self.step += 1;
    }

    /// Run simulation for n steps.
    pub fn run(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            self.step();
        }
    }

    /// Get total electromagnetic energy.
    pub fn total_energy(&self) -> f64 {
        self.grid.total_energy()
    }

    /// Compute Poynting flux through a surface.
    ///
    /// Returns power flux (W) through z-plane at given k index.
    pub fn poynting_flux(&self, k: usize) -> f64 {
        poynting_flux_z(&self.grid, k)
    }

    /// Get probe by index.
    pub fn get_probe(&self, index: usize) -> Option<&Probe> {
        self.probes.get(index)
    }

    /// Check if simulation is stable (CFL condition).
    pub fn is_stable(&self) -> bool {
        self.grid.is_stable()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::db;
    use crate::cpml::CpmlConfig;
    use crate::source::Source;

    #[test]
    fn test_solver_creation() {
        let grid = YeeGrid::new(32, 32, 32, 1e-9, 1e-18);
        let solver = FdtdSolver::new(grid);

        assert_eq!(solver.time, 0.0);
        assert_eq!(solver.step, 0);
        assert!(solver.is_stable());
    }

    #[test]
    fn test_solver_step() {
        let grid = YeeGrid::new(32, 32, 32, 1e-9, 1e-18);
        let mut solver = FdtdSolver::new(grid);

        // Add a point dipole source
        solver.add_source(Source::PointDipole {
            pos: Vec3::new(16.0 * 1e-9, 16.0 * 1e-9, 16.0 * 1e-9),
            freq: 1e9,
            amplitude: 1.0,
            direction: Vec3::new(1.0, 0.0, 0.0),
        });

        // Add a probe near the source
        solver.add_probe(Vec3::new(16.0 * 1e-9, 16.0 * 1e-9, 20.0 * 1e-9));

        // Run for several steps
        solver.run(100);

        assert_eq!(solver.step, 100);
        assert!(solver.time > 0.0);

        // Probe should have recorded data
        let probe = solver.get_probe(0).unwrap();
        assert_eq!(probe.e_field.len(), 100);
    }

    /// A pulsed dipole in a CPML-terminated box must radiate its energy *away*:
    /// once the source is switched off, the residual energy has to fall by
    /// orders of magnitude, not merely "stay bounded".
    ///
    /// This replaces the old `final_energy < initial * 100` check, which passed
    /// for a boundary that reflected almost everything.
    #[test]
    fn cpml_box_drains_radiated_energy() {
        let n = 36;
        let dx = 1e-9;
        let c = 299_792_458.0;
        let dt = dx / (c * 3_f64.sqrt()) * 0.99;

        let grid = YeeGrid::new(n, n, n, dx, dt);
        let mut solver = FdtdSolver::new(grid);
        solver.set_boundary(BoundaryCondition::Cpml(CpmlConfig::with_thickness(10)));

        // Short Gaussian pulse from a central dipole, then silence.
        let spread = 10.0 * dt;
        let t0 = 4.0 * spread;
        let n_drive = 100;
        for step in 0..n_drive {
            solver.step();
            let t = step as f64 * dt;
            let v = crate::analysis::ricker_pulse(t, t0, spread);
            solver.grid.ez.add(n / 2, n / 2, n / 2, v);
        }

        let peak = solver.total_energy();
        assert!(peak > 0.0, "source deposited no energy");

        // Several transit times across a 36-cell box, ample for everything to
        // reach the layer and be absorbed.
        solver.run(300);
        let residual = solver.total_energy();

        // Field amplitude ratio, which is what "reflection in dB" means.
        let attenuation = db((residual / peak).sqrt());
        println!("residual field after CPML drain: {attenuation:.1} dB");
        // A dipole this close to the layer illuminates it at every angle up to
        // grazing, and the corners see three layers at once, so this is a
        // harsher test than the normal-incidence reflection measured in
        // tests/validation.rs. −40 dB in amplitude still means the box retains
        // under 1e-4 of the radiated energy — a lossy absorber leaves ~1e-1.
        assert!(
            attenuation < -40.0,
            "CPML failed to drain the box: residual/peak = {:.3e} ({attenuation:.1} dB)",
            residual / peak
        );
    }

    /// The cheap absorber is legitimately worse than the CPML. Assert the
    /// ordering so a regression that silently swaps them is caught.
    #[test]
    fn cpml_outperforms_the_cheap_absorber() {
        let n = 36;
        let dx = 1e-9;
        let c = 299_792_458.0;
        let dt = dx / (c * 3_f64.sqrt()) * 0.99;
        let spread = 10.0 * dt;
        let t0 = 4.0 * spread;

        let run = |bc: BoundaryCondition| {
            let grid = YeeGrid::new(n, n, n, dx, dt);
            let mut solver = FdtdSolver::new(grid);
            solver.set_boundary(bc);
            let mut peak: f64 = 0.0;
            for step in 0..100 {
                solver.step();
                let t = step as f64 * dt;
                solver
                    .grid
                    .ez
                    .add(n / 2, n / 2, n / 2, crate::analysis::ricker_pulse(t, t0, spread));
                peak = peak.max(solver.total_energy());
            }
            solver.run(300);
            solver.total_energy() / peak
        };

        let cpml = run(BoundaryCondition::Cpml(CpmlConfig::with_thickness(10)));
        let lossy = run(BoundaryCondition::LossyAbsorber {
            thickness: 10,
            order: 2,
            sigma_max: 1.0,
        });

        println!("residual energy fraction — CPML {cpml:.3e}, lossy absorber {lossy:.3e}");
        assert!(
            cpml < lossy * 1e-2,
            "CPML ({cpml:.3e}) should beat the cheap absorber ({lossy:.3e}) by orders of magnitude"
        );
    }

    /// A PEC box conserves energy: with no loss anywhere and no source, the
    /// Yee scheme is symplectic and total energy must be constant to within a
    /// small discretization ripple.
    #[test]
    fn pec_box_conserves_energy() {
        let n = 24;
        let dx = 1e-9;
        let c = 299_792_458.0;
        let dt = dx / (c * 3_f64.sqrt()) * 0.99;

        let grid = YeeGrid::new(n, n, n, dx, dt);
        let mut solver = FdtdSolver::new(grid);
        solver.set_boundary(BoundaryCondition::PerfectConductor);

        // Seed a smooth blob so the initial state is well resolved.
        for k in 1..n - 1 {
            for j in 1..n - 1 {
                for i in 1..n - 1 {
                    let r2 = ((i as f64 - 12.0).powi(2)
                        + (j as f64 - 12.0).powi(2)
                        + (k as f64 - 12.0).powi(2))
                        / 16.0;
                    solver.grid.ez.set(i, j, k, (-r2).exp());
                }
            }
        }

        let initial = solver.total_energy();
        assert!(initial > 0.0);

        // E and H live half a timestep apart, so the naive same-index energy
        // sum carries an O(ωΔt) ripple that is *not* a conservation error.
        // Averaging over a window removes it, leaving only genuine drift —
        // which for this lossless, source-free, PEC-terminated box must be
        // zero to discretization order.
        let mut history = Vec::with_capacity(400);
        for _ in 0..400 {
            solver.step();
            history.push(solver.total_energy());
        }

        let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
        let early = mean(&history[..100]);
        let late = mean(&history[300..]);
        let drift = (late - early).abs() / early;
        println!(
            "PEC energy drift over 300 steps: {:.3}% (ripple {:.1}%)",
            drift * 100.0,
            (history.iter().cloned().fold(f64::MIN, f64::max)
                - history.iter().cloned().fold(f64::MAX, f64::min))
                / early
                * 100.0
        );
        assert!(
            drift < 0.005,
            "energy drifted by {:.4} — a lossless PEC box must conserve",
            drift
        );
    }

    /// Poynting flux must be positive downstream of a source and negative
    /// upstream — the sign carries the direction of energy flow.
    #[test]
    fn poynting_flux_sign_follows_propagation_direction() {
        let dz = 1e-9;
        let c = 299_792_458.0;
        let dt = 0.5 * dz / c;
        let nz = 200;

        let mut grid = YeeGrid::new_rect(1, 1, nz, dz, dz, dz, dt);
        grid.set_periodic([true, true, false]);
        grid.set_boundary(BoundaryCondition::Cpml(
            CpmlConfig::with_thickness(10).on_axes([false, false, true]),
        ));

        // TFSF gives a strictly forward-travelling wave, so the sign test is
        // unambiguous.
        let mut tfsf = crate::tfsf::Tfsf::slab_z(&grid, 40, 160);
        let spread = 14.0 * dt;
        let t0 = 5.0 * spread;
        let src = move |t: f64| crate::analysis::gaussian_pulse(t, t0, spread);

        let mut flux_sum = 0.0;
        for _ in 0..600 {
            tfsf.step(&mut grid, &src);
            flux_sum += poynting_flux_z(&grid, 100);
        }

        assert!(
            flux_sum > 0.0,
            "net Poynting flux should be along +z, got {flux_sum:.3e}"
        );
    }
}
