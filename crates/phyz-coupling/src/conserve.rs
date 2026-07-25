//! Conservation accounting across a coupling handshake.
//!
//! Naive force exchange between domains leaks energy and momentum in two
//! distinct ways, and this module separates them:
//!
//! * **Asymmetric exchange** — domain A is pushed but B never feels the
//!   reaction. This is a *structural* leak and is eliminated by construction:
//!   [`FluxLedger::record_exchange`] books `+F·dt` to one side and `−F·dt` to
//!   the other in a single call, so [`FluxLedger::net_impulse`] is zero by
//!   design. A nonzero value means some code path bypassed the ledger.
//!
//! * **Unabsorbed exchange** — the handshake said it delivered an impulse, but
//!   the receiving solver's own state does not reflect it (wrong frame, dropped
//!   site id, integrator that silently clamps). This is the leak that actually
//!   bites, and [`FluxLedger::absorption_error`] is what detects it, by
//!   comparing each domain's own momentum change against what it was handed.
//!
//! Energy is tracked the same way: the work `F·v dt` done on one domain is
//! booked as `−F·v dt` on the other, and [`FluxLedger::energy_drift`] compares
//! the summed domain energies against their initial total.

use phyz_math::Vec3;

/// Running record of what crossed the handshake between two domains.
#[derive(Clone, Debug)]
pub struct FluxLedger {
    /// Cumulative impulse booked into domain A (N·s).
    pub impulse_a: Vec3,
    /// Cumulative impulse booked into domain B (N·s).
    pub impulse_b: Vec3,
    /// Cumulative work booked into domain A (J).
    pub work_a: f64,
    /// Cumulative work booked into domain B (J).
    pub work_b: f64,
    /// Initial energy of domain A (J).
    pub energy_a0: f64,
    /// Initial energy of domain B (J).
    pub energy_b0: f64,
    /// Initial momentum of domain A (kg·m/s).
    pub momentum_a0: Vec3,
    /// Initial momentum of domain B (kg·m/s).
    pub momentum_b0: Vec3,
    /// Number of recorded exchanges.
    pub exchanges: usize,
}

impl FluxLedger {
    /// Open a ledger against the two domains' initial energy and momentum.
    pub fn new(energy_a0: f64, energy_b0: f64, momentum_a0: Vec3, momentum_b0: Vec3) -> Self {
        Self {
            impulse_a: Vec3::zeros(),
            impulse_b: Vec3::zeros(),
            work_a: 0.0,
            work_b: 0.0,
            energy_a0,
            energy_b0,
            momentum_a0,
            momentum_b0,
            exchanges: 0,
        }
    }

    /// Record one antisymmetric exchange: force `force_on_a` applied to domain
    /// A, whose site moves at `velocity_a`, over `dt`.
    ///
    /// Domain B is booked the equal-and-opposite impulse and the negated work.
    pub fn record_exchange(&mut self, force_on_a: Vec3, velocity_a: Vec3, dt: f64) {
        let impulse = force_on_a * dt;
        let work = force_on_a.dot(velocity_a) * dt;
        self.impulse_a += impulse;
        self.impulse_b -= impulse;
        self.work_a += work;
        self.work_b -= work;
        self.exchanges += 1;
    }

    /// Net impulse across the handshake. Zero by construction; a nonzero value
    /// means an exchange was applied to a solver without being recorded here.
    pub fn net_impulse(&self) -> Vec3 {
        self.impulse_a + self.impulse_b
    }

    /// Net work across the handshake. Zero by construction, same caveat.
    pub fn net_work(&self) -> f64 {
        self.work_a + self.work_b
    }

    /// How far each domain's actual momentum change differs from the impulse
    /// the handshake booked into it.
    ///
    /// Returns `(error_a, error_b)`. Both should be at integrator-truncation
    /// level; a growing error means a domain is not absorbing what it was
    /// handed.
    pub fn absorption_error(&self, momentum_a: Vec3, momentum_b: Vec3) -> (Vec3, Vec3) {
        (
            (momentum_a - self.momentum_a0) - self.impulse_a,
            (momentum_b - self.momentum_b0) - self.impulse_b,
        )
    }

    /// Total energy drift of the coupled system: `(E_a + E_b) − (E_a0 + E_b0)`.
    ///
    /// For a conservative handshake this is bounded by the integrators' own
    /// truncation error, not by the coupling.
    pub fn energy_drift(&self, energy_a: f64, energy_b: f64) -> f64 {
        (energy_a + energy_b) - (self.energy_a0 + self.energy_b0)
    }

    /// Relative energy drift against the initial total, guarded against a
    /// zero baseline.
    pub fn relative_energy_drift(&self, energy_a: f64, energy_b: f64) -> f64 {
        let total0 = self.energy_a0 + self.energy_b0;
        let drift = self.energy_drift(energy_a, energy_b);
        if total0.abs() > 1e-30 {
            drift.abs() / total0.abs()
        } else {
            drift.abs()
        }
    }

    /// Whether the handshake is closing to the given tolerances.
    pub fn is_conserving(
        &self,
        momentum_a: Vec3,
        momentum_b: Vec3,
        impulse_tol: f64,
        energy_a: f64,
        energy_b: f64,
        energy_tol: f64,
    ) -> bool {
        let (err_a, err_b) = self.absorption_error(momentum_a, momentum_b);
        self.net_impulse().norm() <= impulse_tol
            && err_a.norm() <= impulse_tol
            && err_b.norm() <= impulse_tol
            && self.relative_energy_drift(energy_a, energy_b) <= energy_tol
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn exchange_is_antisymmetric_by_construction() {
        let mut ledger = FluxLedger::new(1.0, 2.0, Vec3::zeros(), Vec3::zeros());
        ledger.record_exchange(Vec3::new(3.0, -1.0, 0.5), Vec3::new(2.0, 0.0, 0.0), 0.1);
        ledger.record_exchange(Vec3::new(-1.0, 4.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0.1);

        assert_relative_eq!(ledger.net_impulse().norm(), 0.0, epsilon = 1e-15);
        assert_relative_eq!(ledger.net_work(), 0.0, epsilon = 1e-15);
        assert_eq!(ledger.exchanges, 2);
    }

    #[test]
    fn magnetic_exchange_does_no_work() {
        let mut ledger = FluxLedger::new(0.0, 0.0, Vec3::zeros(), Vec3::zeros());
        let v = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(0.0, 0.0, 1.0);
        // A magnetic force is always perpendicular to v.
        ledger.record_exchange(v.cross(b), v, 0.01);
        assert_relative_eq!(ledger.work_a, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn absorption_error_flags_a_domain_that_ignored_its_impulse() {
        let mut ledger = FluxLedger::new(0.0, 0.0, Vec3::zeros(), Vec3::zeros());
        ledger.record_exchange(Vec3::new(1.0, 0.0, 0.0), Vec3::zeros(), 1.0);

        // Domain A absorbed it, domain B did not.
        let (err_a, err_b) =
            ledger.absorption_error(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0));
        assert_relative_eq!(err_a.norm(), 0.0, epsilon = 1e-15);
        assert_relative_eq!(err_b.norm(), 1.0, epsilon = 1e-15);
    }
}
