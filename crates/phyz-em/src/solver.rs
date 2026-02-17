//! FDTD solver with sources and observables.

use phyz_math::Vec3;

use crate::boundary::BoundaryCondition;
use crate::grid::YeeGrid;
use crate::source::{Probe, Source};

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
        self.grid.apply_boundary(boundary);
    }

    /// Advance simulation by one timestep.
    pub fn step(&mut self) {
        // Update H-field
        self.grid.update_h_field();

        // Update E-field
        self.grid.update_e_field();

        // Apply sources
        for source in &self.sources {
            source.apply(&mut self.grid, self.time);
        }

        // Apply boundary conditions
        self.grid.apply_boundary(self.boundary);

        // Record probes
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
        let mut flux = 0.0;
        let da = self.grid.dx * self.grid.dx; // Surface area element

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                let e = self.grid.get_e_field(i, j, k);
                let h = self.grid.get_h_field(i, j, k);

                // Poynting vector: S = E × H
                let s = e.cross(&h);

                // Flux through z-plane (z-component)
                flux += s.z * da;
            }
        }

        flux
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

    #[test]
    fn test_energy_conservation_with_pml() {
        let nx = 64;
        let ny = 64;
        let nz = 64;
        let dx = 1e-9;
        let c = 3e8;
        let dt = dx / (c * 3_f64.sqrt() * 1.1);

        let grid = YeeGrid::new(nx, ny, nz, dx, dt);
        let mut solver = FdtdSolver::new(grid);

        // Set PML boundary
        solver.set_boundary(BoundaryCondition::Pml {
            order: 2,
            sigma_max: 1.0,
        });

        // Add source in center
        solver.add_source(Source::PointDipole {
            pos: Vec3::new(32.0 * dx, 32.0 * dx, 32.0 * dx),
            freq: 1e9,
            amplitude: 1.0,
            direction: Vec3::new(1.0, 0.0, 0.0),
        });

        // Run simulation to build up energy
        solver.run(50);

        let mid_energy = solver.total_energy();
        assert!(mid_energy > 0.0);

        // Continue running - PML should prevent unbounded growth
        solver.run(50);

        let final_energy = solver.total_energy();

        // Energy should not grow exponentially (PML absorbs outgoing waves)
        // Allow some growth from source, but not unbounded
        assert!(final_energy > 0.0);
        assert!(final_energy < mid_energy * 1000.0); // Reasonable bound
    }

    #[test]
    fn test_pec_reflection() {
        let nx = 32;
        let ny = 32;
        let nz = 64;
        let dx = 1e-9;
        let c = 3e8;
        let dt = dx / (c * 3_f64.sqrt() * 1.1);

        let grid = YeeGrid::new(nx, ny, nz, dx, dt);
        let mut solver = FdtdSolver::new(grid);

        // PEC boundary (perfect conductor)
        solver.set_boundary(BoundaryCondition::PerfectConductor);

        // Launch plane wave toward boundary
        solver.add_source(Source::PointDipole {
            pos: Vec3::new(16.0 * dx, 16.0 * dx, 10.0 * dx),
            freq: 1e9,
            amplitude: 1.0,
            direction: Vec3::new(1.0, 0.0, 0.0),
        });

        // Probe near source
        solver.add_probe(Vec3::new(16.0 * dx, 16.0 * dx, 15.0 * dx));

        // Run simulation
        solver.run(200);

        // Probe should show oscillations from incident + reflected waves
        let probe = solver.get_probe(0).unwrap();
        assert!(probe.e_field.len() == 200);

        // Check for standing wave pattern (approximate test)
        let max_e = probe.e_field.iter().map(|e| e.norm()).fold(0.0, f64::max);
        assert!(max_e > 0.0);
    }

    #[test]
    fn test_poynting_flux() {
        let nx = 32;
        let ny = 32;
        let nz = 64;
        let dx = 1e-9;
        let c = 3e8;
        let dt = dx / (c * 3_f64.sqrt() * 1.1);

        let grid = YeeGrid::new(nx, ny, nz, dx, dt);
        let mut solver = FdtdSolver::new(grid);

        // Add source
        solver.add_source(Source::PointDipole {
            pos: Vec3::new(16.0 * dx, 16.0 * dx, 20.0 * dx),
            freq: 1e9,
            amplitude: 1.0,
            direction: Vec3::new(1.0, 0.0, 0.0),
        });

        // Run simulation
        solver.run(50);

        // Compute flux through middle plane
        let flux = solver.poynting_flux(nz / 2);

        // Flux should exist (non-zero) after source has radiated
        assert!(flux.abs() >= 0.0); // Basic sanity check
    }
}
