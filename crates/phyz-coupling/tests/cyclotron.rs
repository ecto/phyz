//! End-to-end coupled simulation: a charged rigid body in an FDTD magnetic
//! field, checked against the closed-form cyclotron solution.
//!
//! Both domains are real solvers — `phyz-rigid`'s ABA and `phyz-em`'s Yee-grid
//! FDTD. Nothing here is mocked: the field the body feels each step is read out
//! of the grid the FDTD solver just stepped.
//!
//! Analytic reference for `F = q v × B` with uniform `B = B ẑ` and `v ⊥ B`:
//!
//! * angular frequency  ω = qB/m
//! * period             T = 2πm/(qB)
//! * orbit radius       r = mv/(qB) = v/ω
//! * speed is constant  (magnetic force does no work)

use approx::assert_relative_eq;
use phyz_coupling::{BoundingBox, CoupledSystem, EmSolver, ReactionMode, RigidSolver, Solver};
use phyz_guardian::{ConservationMonitor, ConservationState};
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::ModelBuilder;

// --- Scenario -------------------------------------------------------------
// Chosen so the cyclotron period is comparable to the grid's CFL timestep;
// otherwise the FDTD domain would need ~10^14 substeps per orbit.

const CHARGE: f64 = 6.283_185_307_179_586e-3; // C
const MASS: f64 = 1e-9; // kg
const B_MAG: f64 = 1.0; // T  →  ω = qB/m = 2π·10^6 rad/s, T = 1 µs
const SPEED: f64 = 1e5; // m/s →  r = v/ω ≈ 15.9 mm

const GRID_N: usize = 8;
const DX: f64 = 0.1; // m  → 0.8 m box
const CENTER: f64 = 0.4; // m  → box center

/// Analytic cyclotron angular frequency.
fn omega_c() -> f64 {
    CHARGE * B_MAG / MASS
}

/// Analytic cyclotron radius.
fn radius_c() -> f64 {
    SPEED / omega_c()
}

/// Build the coupled rigid + EM system with the body at the box center moving
/// in +x, and `B` along +z.
fn build(dt_rigid: f64) -> CoupledSystem<RigidSolver, EmSolver> {
    let model = ModelBuilder::new()
        .gravity(Vec3::zeros())
        .dt(dt_rigid)
        .add_free_body(
            "charged_bob",
            -1,
            SpatialTransform::identity(),
            // Point mass at the body origin; a diagonal inertia keeps the
            // rotational block well-conditioned (no torque is ever applied).
            SpatialInertia::new(
                MASS,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(1e-12, 1e-12, 1e-12)),
            ),
        )
        .build();

    let mut state = model.default_state();
    // Free joint: q = [wx, wy, wz, x, y, z], v = [wx, wy, wz, vx, vy, vz].
    state.q[3] = CENTER;
    state.q[4] = CENTER;
    state.q[5] = CENTER;
    state.v[3] = SPEED;

    let mut matter = RigidSolver::new(model, state);
    matter.couple_body(0, CHARGE);

    let field = EmSolver::uniform_b_field(GRID_N, DX, Vec3::new(0.0, 0.0, B_MAG));

    let box_max = GRID_N as f64 * DX;
    let region = BoundingBox::new(Vec3::zeros(), Vec3::new(box_max, box_max, box_max));

    CoupledSystem::new(matter, field, region)
}

/// The FDTD domain must actually reproduce the uniform field it was seeded
/// with — otherwise the analytic comparison below is meaningless.
#[test]
fn fdtd_domain_holds_the_uniform_field_while_stepping() {
    let mut em = EmSolver::uniform_b_field(GRID_N, DX, Vec3::new(0.0, 0.0, B_MAG));
    let probe = Vec3::new(CENTER, CENTER, CENTER);

    let before = em.field_at(&probe);
    assert_relative_eq!(before.b_field.z, B_MAG, epsilon = 1e-12);

    let steps_before = em.fdtd.step;
    em.advance(1000.0 * em.natural_dt());
    assert!(em.fdtd.step > steps_before, "FDTD did not actually step");

    let after = em.field_at(&probe);
    assert_relative_eq!(after.b_field.z, B_MAG, epsilon = 1e-12);
    assert_relative_eq!(after.e_field.norm(), 0.0, epsilon = 1e-12);
}

/// Each domain reports its own natural timestep, and the driver turns those
/// into a subcycling schedule rather than forcing a single global dt.
#[test]
fn subcycling_schedule_reflects_both_natural_timesteps() {
    let dt_rigid = 2.5e-10;
    let sys = build(dt_rigid);

    let dt_em = sys.field.natural_dt();
    assert!(
        dt_em < dt_rigid,
        "expected the EM domain to be the fast one ({dt_em:e} vs {dt_rigid:e})"
    );

    let schedule = sys.schedule();
    assert_relative_eq!(schedule.dt_base, dt_em, epsilon = 1e-30);
    // Level 0 = field (base), level 1 = matter, at round(dt_rigid/dt_em) base steps.
    assert_eq!(schedule.ratios[0], 1);
    assert_eq!(schedule.ratios[1], (dt_rigid / dt_em).round() as usize);
}

/// The headline test: radius, period, and speed against the closed form.
#[test]
fn cyclotron_orbit_matches_closed_form() {
    let period = 2.0 * std::f64::consts::PI / omega_c();
    let steps_per_orbit = 20_000;
    let dt = period / steps_per_orbit as f64;

    let mut sys = build(dt);

    // For B = +B ẑ, q > 0, v = +v x̂: F = qv×B points along −ŷ, so the guiding
    // center sits one radius in −ŷ from the start and the orbit is clockwise.
    let start = Vec3::new(CENTER, CENTER, CENTER);
    let guiding_center = start - Vec3::new(0.0, radius_c(), 0.0);

    let mut max_radius_err: f64 = 0.0;
    let mut max_speed_err: f64 = 0.0;
    let mut max_z_drift: f64 = 0.0;

    for _ in 0..steps_per_orbit {
        sys.step(dt);

        let site = sys.matter.sites()[0];
        let r = (site.position - guiding_center).norm();
        max_radius_err = max_radius_err.max((r - radius_c()).abs() / radius_c());
        max_speed_err = max_speed_err.max((site.velocity.norm() - SPEED).abs() / SPEED);
        max_z_drift = max_z_drift.max((site.position.z - CENTER).abs());
    }

    // Semi-implicit Euler applied to a rotation gains |v| by O((ωdt)²) per
    // step, i.e. ~2π²/N in energy over one orbit — 1e-3 at N = 20 000. That
    // bound is the integrator's, not the coupling's.
    assert!(
        max_radius_err < 1.5e-3,
        "cyclotron radius error {max_radius_err:.2e} (analytic r = {:.6} m)",
        radius_c()
    );
    assert!(
        max_speed_err < 1e-3,
        "speed drift {max_speed_err:.2e}; a magnetic force must do no work"
    );
    assert!(
        max_z_drift < 1e-12,
        "motion left the plane perpendicular to B: {max_z_drift:.2e} m"
    );

    // After exactly one period the body should be back where it started.
    let site = sys.matter.sites()[0];
    let closure = (site.position - start).norm() / radius_c();
    assert!(
        closure < 2e-3,
        "orbit did not close after one analytic period T = {period:.3e} s: \
         off by {closure:.2e} radii"
    );

    // And the returning velocity should again be +x at the original speed.
    assert_relative_eq!(site.velocity.x / SPEED, 1.0, epsilon = 2e-3);
    assert_relative_eq!(site.velocity.y / SPEED, 0.0, epsilon = 2e-3);
}

/// Energy is asserted across the *coupled* step, not inside either solver:
/// the ledger's total is matter energy + field energy.
#[test]
fn coupled_step_conserves_energy_and_momentum() {
    let period = 2.0 * std::f64::consts::PI / omega_c();
    let steps = 20_000;
    let dt = period / steps as f64;

    let mut sys = build(dt);
    let e0 = sys.total_energy();
    let ke0 = sys.matter.energy();
    let p0 = sys.matter.momentum();

    sys.run(steps, dt);

    // 1. Total energy across both domains. Normalised against the *matter*
    //    energy, not the total: a uniform 1 T field over a 0.5 m³ box holds
    //    ~10^5 J against the body's 5 mJ, so a relative-to-total check would
    //    pass no matter how badly the handshake leaked.
    let drift = sys
        .ledger
        .energy_drift(sys.matter.energy(), sys.field.energy());
    assert!(
        drift.abs() / ke0 < 2.5e-3,
        "coupled energy drift {drift:.3e} J against a {ke0:.3e} J budget          (E0 = {e0:.6e} J)"
    );

    // 2. The matter domain's own energy: a purely magnetic handshake transfers
    //    no work, so kinetic energy must be flat to integrator error. Energy
    //    goes as v², so the bound is twice the speed drift: ~2·(2π²/N).
    let ke_drift = (sys.matter.energy() - ke0).abs() / ke0;
    assert!(ke_drift < 2.5e-3, "kinetic energy drift {ke_drift:.2e}");

    // 3. The handshake booked no net work — this holds by construction and
    //    catches any path that pushes a solver without recording it.
    let scale = ke0.max(1e-30);
    assert!(
        sys.ledger.net_work().abs() / scale < 1e-12,
        "handshake leaked work: {:.3e} J",
        sys.ledger.net_work()
    );
    assert!(
        sys.ledger.net_impulse().norm() < 1e-18,
        "handshake leaked momentum: {:.3e} N·s",
        sys.ledger.net_impulse().norm()
    );

    // 4. Each domain actually absorbed the impulse it was handed. This is the
    //    check that catches a frame error or a dropped site id, and it is the
    //    sharp one: the rigid domain's momentum change tracks the booked
    //    impulse to ~1e-11 relative.
    let (err_matter, err_field) = sys.absorption_error();
    let booked = sys.ledger.impulse_a.norm();
    assert!(
        booked > 1e-9,
        "no meaningful impulse crossed the handshake ({booked:.3e} N·s);          |p0| = {:.3e}",
        p0.norm()
    );
    assert!(
        err_matter.norm() / booked < 1e-9,
        "rigid domain did not absorb its booked impulse: {:.3e} N·s of {booked:.3e}",
        err_matter.norm()
    );
    assert!(
        err_field.norm() == 0.0,
        "field domain did not absorb its booked impulse: {:.3e} N·s",
        err_field.norm()
    );
}

/// `phyz-guardian`'s conservation monitors run against the rigid domain of a
/// coupled system, so a coupling bug shows up as a conservation violation
/// through the existing machinery rather than a bespoke check.
#[test]
fn guardian_monitors_the_coupled_rigid_domain() {
    let period = 2.0 * std::f64::consts::PI / omega_c();
    let steps = 20_000;
    let dt = period / steps as f64;

    let mut sys = build(dt);
    let baseline = ConservationState::new(&sys.matter.model, &sys.matter.state);

    sys.run(steps, dt);

    let monitor = ConservationMonitor::check(&baseline, &sys.matter.model, &sys.matter.state);

    // Energy: gravity is off and the magnetic handshake does no work, so the
    // guardian should see a conserved rigid domain.
    assert!(
        monitor.energy_error < 2.5e-3,
        "guardian energy error {:.2e}",
        monitor.energy_error
    );

    // Momentum is *not* conserved in the rigid domain alone — the field is
    // continuously turning the body. What must hold is that the change equals
    // the impulse the ledger booked, which is exactly the absorption check.
    let booked = sys.ledger.impulse_a;
    assert!(
        booked.norm() > 0.0,
        "no impulse crossed the handshake — the domains are not coupled"
    );
    let (err_matter, _) = sys.absorption_error();
    assert!(
        err_matter.norm() / booked.norm() < 1e-9,
        "momentum change ({:.3e}) disagrees with booked impulse ({:.3e})",
        monitor.momentum_error.norm(),
        booked.norm()
    );
}

/// The physical back-reaction channel is wired and does move the field, but is
/// off by default. This documents the difference rather than asserting the
/// deposited self-field is accurate — it is not, for a point charge on a
/// 0.1 m grid.
#[test]
fn current_deposition_actually_perturbs_the_field() {
    let dt = 1e-10;

    let mut ledger_only = build(dt);
    let field_energy_0 = ledger_only.field.energy();
    ledger_only.step(dt);
    assert_relative_eq!(
        ledger_only.field.energy(),
        field_energy_0,
        max_relative = 1e-12
    );
    let e_norm_ledger = ledger_only
        .field
        .field_at(&Vec3::new(CENTER, CENTER, CENTER))
        .e_field
        .norm();
    assert_relative_eq!(e_norm_ledger, 0.0, epsilon = 1e-12);

    let mut depositing = build(dt).with_reaction(ReactionMode::CurrentDeposition);
    depositing.step(dt);
    let e_norm_deposit = depositing
        .field
        .field_at(&Vec3::new(CENTER, CENTER, CENTER))
        .e_field
        .norm();
    assert!(
        e_norm_deposit > 0.0,
        "current deposition did not source the field"
    );

    // Momentum bookkeeping closes identically in both modes.
    assert!(depositing.ledger.net_impulse().norm() < 1e-18);
}
