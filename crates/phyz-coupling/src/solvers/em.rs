//! [`Solver`] adapter for the FDTD electromagnetic solver in `phyz-em`.

use phyz_em::{FdtdSolver, YeeGrid};
use phyz_math::Vec3;

use crate::coupling::SolverType;
use crate::solver::{ExternalInput, FieldSample, Solver};

/// An electromagnetic domain (Yee-grid FDTD) exposed to the coupling layer.
///
/// The grid steps at its CFL-limited [`YeeGrid::dt`]; [`Solver::advance`]
/// subcycles it to cover the requested interval, carrying the sub-`dt` remainder
/// forward so the domain clock does not drift against the coupling clock.
pub struct EmSolver {
    /// The underlying FDTD solver.
    pub fdtd: FdtdSolver,
    /// Queued current sources for the next advance: (position, `q·v` in A·m).
    pending_currents: Vec<(Vec3, Vec3)>,
    /// Momentum booked into this domain by the coupling handshake (kg·m/s).
    booked_momentum: Vec3,
    /// Unconsumed time carried between advances (s).
    time_debt: f64,
}

impl EmSolver {
    /// Wrap an existing FDTD solver.
    pub fn new(fdtd: FdtdSolver) -> Self {
        Self {
            fdtd,
            pending_currents: Vec::new(),
            booked_momentum: Vec3::zeros(),
            time_debt: 0.0,
        }
    }

    /// Build a domain holding a uniform, static magnetic flux density.
    ///
    /// A spatially uniform `H` with `E = 0` is an exact static solution of the
    /// discrete Yee update — every finite-difference curl vanishes identically —
    /// so the FDTD solver really does step, and really does reproduce this field
    /// step after step. That makes it the right configuration for an
    /// analytically checkable coupled test.
    pub fn uniform_b_field(grid_n: usize, dx: f64, b_field: Vec3) -> Self {
        let c = 299_792_458.0_f64;
        let dt = dx / (c * 3.0_f64.sqrt() * 1.1);
        let mut grid = YeeGrid::new(grid_n, grid_n, grid_n, dx, dt);

        let h = b_field / grid.mu0;
        for i in 0..grid_n {
            for j in 0..grid_n {
                for k in 0..grid_n {
                    grid.hx.set(i, j, k, h.x);
                    grid.hy.set(i, j, k, h.y);
                    grid.hz.set(i, j, k, h.z);
                }
            }
        }

        let mut fdtd = FdtdSolver::new(grid);
        // PML would ramp conductivity at the edges; with E ≡ 0 it is a no-op,
        // but Periodic keeps the uniform field exactly uniform by construction.
        fdtd.set_boundary(phyz_em::BoundaryCondition::Periodic);
        Self::new(fdtd)
    }

    /// Deposit queued currents into the E-field: `ΔE = -(J/ε) Δt`.
    ///
    /// This is nearest-cell (NGP) deposition. It is the physically correct
    /// back-reaction channel — a moving charge is a current, and a current
    /// sources the field — but for a *point* charge on a coarse grid the
    /// resulting self-field is a well-known artifact. See
    /// [`crate::ReactionMode`] for when this path is used.
    fn deposit_currents(&mut self, dt: f64) {
        let grid = &mut self.fdtd.grid;
        let dv = grid.dx * grid.dx * grid.dx;
        for (position, moment) in self.pending_currents.drain(..) {
            let (i, j, k) = grid.position_to_index(&position);
            let eps = grid.eps0 * grid.eps_r.get(i, j, k);
            let coef = -dt / (eps * dv);
            grid.ex.add(i, j, k, coef * moment.x);
            grid.ey.add(i, j, k, coef * moment.y);
            grid.ez.add(i, j, k, coef * moment.z);
        }
    }
}

impl Solver for EmSolver {
    fn solver_type(&self) -> SolverType {
        SolverType::Electromagnetic
    }

    fn natural_dt(&self) -> f64 {
        self.fdtd.grid.dt
    }

    fn time(&self) -> f64 {
        self.fdtd.time
    }

    fn apply_external(&mut self, input: ExternalInput) {
        match input {
            ExternalInput::Current { position, moment } => {
                self.pending_currents.push((position, moment));
            }
            ExternalInput::Reaction { impulse } => {
                self.booked_momentum += impulse;
            }
            // A field domain has no discrete sites to push on.
            ExternalInput::Force { .. } => {}
        }
    }

    fn advance(&mut self, dt: f64) {
        let dt_em = self.fdtd.grid.dt;
        let budget = dt + self.time_debt;
        let n = (budget / dt_em).floor().max(0.0) as usize;
        self.time_debt = budget - n as f64 * dt_em;

        if !self.pending_currents.is_empty() {
            self.deposit_currents(dt);
        }
        self.fdtd.run(n);
    }

    fn energy(&self) -> f64 {
        self.fdtd.total_energy()
    }

    fn field_at(&self, position: &Vec3) -> FieldSample {
        let grid = &self.fdtd.grid;
        let (i, j, k) = grid.position_to_index(position);
        let mu = grid.mu0 * grid.mu_r.get(i, j, k);
        FieldSample {
            e_field: grid.get_e_field(i, j, k),
            b_field: grid.get_h_field(i, j, k) * mu,
        }
    }

    /// Momentum booked into the field domain by the handshake.
    ///
    /// This is **not** the Maxwell-stress momentum integral `ε₀∫E×B dV` of the
    /// grid — computing that consistently with a nearest-cell point-charge
    /// deposition is a genuinely open piece of work here. It is the ledger
    /// counterpart of the impulses the coupling layer delivered to the other
    /// domain, which is what makes the exchange exactly antisymmetric.
    fn momentum(&self) -> Vec3 {
        self.booked_momentum
    }
}
