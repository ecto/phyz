//! End-to-end driver for a two-domain coupled simulation.
//!
//! [`CoupledSystem`] takes two [`Solver`]s — one exposing discrete
//! [`CouplingSite`]s, one exposing a field — computes the Lorentz handshake in
//! the overlap region, applies it antisymmetrically, and subcycles each domain
//! at its own [`Solver::natural_dt`].

use phyz_math::Vec3;

use crate::boundary::BoundingBox;
use crate::conserve::FluxLedger;
use crate::lorentz::lorentz_force;
use crate::solver::{CouplingSite, ExternalInput, Solver};
use crate::subcycling::SubcyclingSchedule;

/// How the field domain receives the reaction to a force applied to matter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReactionMode {
    /// Book the equal-and-opposite impulse into the field domain's momentum
    /// account without depositing a source term.
    ///
    /// Momentum closes exactly and no spurious self-force is created. This is
    /// the honest default for a point charge on a coarse grid.
    #[default]
    LedgerOnly,
    /// Additionally deposit the moving charge's current `q·v` into the field,
    /// so the field physically responds.
    ///
    /// This is the real back-reaction channel, but nearest-cell deposition of a
    /// point charge produces a large self-field that the same charge then feels
    /// on the next step. Use it when the charge is spread over many cells or
    /// when you want the radiated field, not when you want a clean orbit.
    CurrentDeposition,
}

/// Two solvers coupled through a Lorentz handshake in an overlap region.
///
/// `M` is the matter domain (supplies [`Solver::sites`]); `F` is the field
/// domain (supplies [`Solver::field_at`]).
pub struct CoupledSystem<M: Solver, F: Solver> {
    /// The matter domain.
    pub matter: M,
    /// The field domain.
    pub field: F,
    /// Handshake region — sites outside it are not coupled.
    pub region: BoundingBox,
    /// How the field domain receives the reaction.
    pub reaction: ReactionMode,
    /// Conservation accounting across the handshake.
    pub ledger: FluxLedger,
}

impl<M: Solver, F: Solver> CoupledSystem<M, F> {
    /// Couple two solvers over the given overlap region.
    pub fn new(matter: M, field: F, region: BoundingBox) -> Self {
        let ledger = FluxLedger::new(
            matter.energy(),
            field.energy(),
            matter.momentum(),
            field.momentum(),
        );
        Self {
            matter,
            field,
            region,
            reaction: ReactionMode::default(),
            ledger,
        }
    }

    /// Set the reaction mode.
    pub fn with_reaction(mut self, reaction: ReactionMode) -> Self {
        self.reaction = reaction;
        self
    }

    /// Subcycling schedule implied by the two domains' natural timesteps.
    ///
    /// Level 0 is the faster domain, level 1 the slower; the ratios are how
    /// many base steps each covers.
    pub fn schedule(&self) -> SubcyclingSchedule {
        let dt_m = self.matter.natural_dt();
        let dt_f = self.field.natural_dt();
        let base = dt_m.min(dt_f).max(f64::MIN_POSITIVE);
        SubcyclingSchedule::new(
            base,
            vec![
                (dt_f / base).round().max(1.0) as usize,
                (dt_m / base).round().max(1.0) as usize,
            ],
        )
    }

    /// Advance the coupled system by `dt`.
    ///
    /// Each domain internally subcycles to its own natural timestep, so `dt` is
    /// the *coupling* interval — how often the handshake is re-evaluated.
    pub fn step(&mut self, dt: f64) {
        for site in self.coupled_sites() {
            let sample = self.field.field_at(&site.position);
            let force = lorentz_force(
                site.charge,
                site.position,
                site.velocity,
                &sample.e_field,
                &sample.b_field,
            );

            self.matter.apply_external(ExternalInput::Force {
                site: site.id,
                force,
                torque: Vec3::zeros(),
            });

            if self.reaction == ReactionMode::CurrentDeposition {
                self.field.apply_external(ExternalInput::Current {
                    position: site.position,
                    moment: site.charge * site.velocity,
                });
            }
            self.field.apply_external(ExternalInput::Reaction {
                impulse: -force * dt,
            });

            self.ledger.record_exchange(force, site.velocity, dt);
        }

        self.matter.advance(dt);
        self.field.advance(dt);
    }

    /// Run `n` coupling steps of size `dt`.
    pub fn run(&mut self, n: usize, dt: f64) {
        for _ in 0..n {
            self.step(dt);
        }
    }

    /// Sites currently inside the handshake region.
    pub fn coupled_sites(&self) -> Vec<CouplingSite> {
        self.matter
            .sites()
            .into_iter()
            .filter(|s| self.region.contains(&s.position))
            .collect()
    }

    /// Total energy of both domains (J).
    pub fn total_energy(&self) -> f64 {
        self.matter.energy() + self.field.energy()
    }

    /// Relative drift of the coupled system's total energy since the ledger
    /// was opened.
    pub fn relative_energy_drift(&self) -> f64 {
        self.ledger
            .relative_energy_drift(self.matter.energy(), self.field.energy())
    }

    /// Per-domain difference between actual momentum change and booked impulse.
    pub fn absorption_error(&self) -> (Vec3, Vec3) {
        self.ledger
            .absorption_error(self.matter.momentum(), self.field.momentum())
    }
}
