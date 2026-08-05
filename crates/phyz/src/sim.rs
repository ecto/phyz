//! Integrated time stepping: pluggable solvers and the [`Simulator`] driver.
//!
//! This is the one piece of the umbrella crate that isn't a re-export: it
//! composes forward dynamics ([`phyz_rigid`]), contact resolution
//! ([`phyz_contact`]) and integration into a single `step()`. The ordering
//! (FK → contacts → ABA with external forces → integrate → FK) matches
//! `phyz_world::World::step`.
//!
//! Requires the `contact` and `diff` features (both on by default).

use phyz_contact::{
    ContactCache, ContactMaterial, ContactSolverConfig, find_contacts, find_ground_contacts_model,
    solve_contacts_warm,
};
use phyz_diff::{StepJacobians, finite_diff_jacobians, semi_implicit_step_jacobians};
use phyz_math::DVec;
use phyz_model::{Model, State};
use phyz_rigid::{aba, forward_kinematics, integrate_configuration};
use std::cell::RefCell;

/// Pluggable solver trait.
///
/// Implementations define how to advance the simulation by one timestep.
pub trait Solver {
    /// Advance state by dt. Reads from `state` and writes the result back.
    fn step(&self, model: &Model, state: &mut State);

    /// Advance state by dt and return step Jacobians.
    fn step_with_jacobians(&self, model: &Model, state: &mut State) -> StepJacobians;
}

/// Semi-implicit Euler integrator using Featherstone ABA.
pub struct SemiImplicitEulerSolver;

impl Solver for SemiImplicitEulerSolver {
    fn step(&self, model: &Model, state: &mut State) {
        let dt = model.dt;
        let qdd = aba(model, state);

        // Semi-implicit Euler: update velocity first, then position
        state.v += &(&qdd * dt);
        let v_clone = state.v.clone();
        // NOT `q += v*dt`: free and ball joints parameterise rotation with
        // exponential coordinates, and a free joint's linear velocity is
        // body-frame. See `phyz_rigid::integrate_configuration`.
        integrate_configuration(model, state.q.as_mut_slice(), v_clone.as_slice(), dt);
        state.time += dt;

        // Update body transforms via FK
        let (xforms, _) = forward_kinematics(model, state);
        state.body_xform = xforms;
    }

    fn step_with_jacobians(&self, model: &Model, state: &mut State) -> StepJacobians {
        let jac = semi_implicit_step_jacobians(model, state);
        self.step(model, state);
        jac
    }
}

/// 4th-order Runge-Kutta integrator.
///
/// Much better energy conservation than semi-implicit Euler for systems
/// with configuration-dependent mass matrices (e.g., double pendulum).
pub struct Rk4Solver;

impl Rk4Solver {
    /// Evaluate derivatives: given (q, v, ctrl) → (dq/dt, dv/dt) = (v, ABA(q,v,ctrl)).
    fn derivatives(model: &Model, state: &State) -> (DVec, DVec) {
        let qdd = aba(model, state);
        (state.v.clone(), qdd)
    }
}

impl Solver for Rk4Solver {
    fn step(&self, model: &Model, state: &mut State) {
        let dt = model.dt;

        // k1
        let (dq1, dv1) = Self::derivatives(model, state);

        // k2
        let mut s2 = state.clone();
        integrate_configuration(model, s2.q.as_mut_slice(), dq1.as_slice(), dt / 2.0);
        s2.v += &(&dv1 * (dt / 2.0));
        let (dq2, dv2) = Self::derivatives(model, &s2);

        // k3
        let mut s3 = state.clone();
        integrate_configuration(model, s3.q.as_mut_slice(), dq2.as_slice(), dt / 2.0);
        s3.v += &(&dv2 * (dt / 2.0));
        let (dq3, dv3) = Self::derivatives(model, &s3);

        // k4
        let mut s4 = state.clone();
        integrate_configuration(model, s4.q.as_mut_slice(), dq3.as_slice(), dt);
        s4.v += &(&dv3 * dt);
        let (dq4, dv4) = Self::derivatives(model, &s4);

        // Combine
        let dq_sum = &(&(&dq1 + &(&dq2 * 2.0)) + &(&dq3 * 2.0)) + &dq4;
        // The RK4 weighted average is applied as a single configuration step so
        // rotational sub-blocks stay on the manifold. This makes the *stage*
        // combination first-order-ish for rotations rather than a true RK4 on
        // the Lie group (a Munthe-Kaas scheme would be); it is still strictly
        // better than adding angular rates into position slots.
        let dq_avg = &dq_sum * (1.0 / 6.0);
        integrate_configuration(model, state.q.as_mut_slice(), dq_avg.as_slice(), dt);
        let dv_sum = &(&(&dv1 + &(&dv2 * 2.0)) + &(&dv3 * 2.0)) + &dv4;
        state.v += &(&dv_sum * (dt / 6.0));
        state.time += dt;

        // Update body transforms via FK
        let (xforms, _) = forward_kinematics(model, state);
        state.body_xform = xforms;
    }

    fn step_with_jacobians(&self, model: &Model, state: &mut State) -> StepJacobians {
        // Use finite-diff Jacobians for RK4 (analytical would need chain rule through 4 stages)
        let jac = finite_diff_jacobians(model, state, 1e-7);
        self.step(model, state);
        jac
    }
}

/// Main simulation driver.
pub struct Simulator {
    solver: Box<dyn Solver>,
    /// Previous step's contact impulses, keyed by feature, for warm starting
    /// [`Simulator::step_with_contacts`].
    ///
    /// Behind a `RefCell` because stepping takes `&self` — the cache is a
    /// solver-internal accelerator, not simulation state, and making the whole
    /// API `&mut self` for it would ripple through every caller. It cannot
    /// change results: the contact problem is strongly convex, so the seed
    /// only moves the iteration count.
    contact_cache: RefCell<ContactCache>,
}

impl Simulator {
    /// Create a simulator with the default semi-implicit Euler solver.
    pub fn new() -> Self {
        Self {
            solver: Box::new(SemiImplicitEulerSolver),
            contact_cache: RefCell::new(ContactCache::default()),
        }
    }

    /// Create a simulator with the RK4 solver.
    pub fn rk4() -> Self {
        Self {
            solver: Box::new(Rk4Solver),
            contact_cache: RefCell::new(ContactCache::default()),
        }
    }

    /// Create a simulator with a custom solver.
    pub fn with_solver(solver: Box<dyn Solver>) -> Self {
        Self {
            solver,
            contact_cache: RefCell::new(ContactCache::default()),
        }
    }

    /// Forget the warm-start contact cache.
    ///
    /// Call after teleporting or resetting a state: last step's impulses are
    /// then a guess about a world that no longer exists. Purely a performance
    /// concern — the solve converges to the same answer either way.
    pub fn reset_contact_cache(&self) {
        self.contact_cache.borrow_mut().clear();
    }

    /// Advance simulation by one timestep.
    pub fn step(&self, model: &Model, state: &mut State) {
        self.solver.step(model, state);
    }

    /// Advance simulation by one timestep and return Jacobians.
    pub fn step_with_jacobians(&self, model: &Model, state: &mut State) -> StepJacobians {
        self.solver.step_with_jacobians(model, state)
    }

    /// Run simulation for `n` steps.
    pub fn simulate(&self, model: &Model, state: &mut State, n: usize) {
        for _ in 0..n {
            self.step(model, state);
        }
    }

    /// Advance simulation with contact detection and resolution.
    ///
    /// 1. Runs FK to get body transforms and velocities
    /// 2. Detects ground contacts (and body-body contacts if geometries provided)
    /// 3. Computes contact forces
    /// 4. Runs ABA with contact forces as external forces
    /// 5. Integrates and updates FK
    ///
    /// Returns the **realized** generalized acceleration `(v' − v) / dt` for
    /// the *pre-step* state — free dynamics plus the contact impulses the
    /// solver just found. That is what inertial sensors need: pass it to
    /// `phyz_world::SensorContext::with_acceleration` together with a
    /// snapshot of the pre-step state. To read sensors at the current state
    /// without integrating, use [`Simulator::contact_acceleration`] instead.
    pub fn step_with_contacts(
        &self,
        model: &Model,
        state: &mut State,
        ground_height: f64,
        material: &ContactMaterial,
    ) -> DVec {
        let dt = model.dt;
        let v_before = state.v.clone();

        // Run FK to get current transforms and velocities
        let (xforms, _velocities) = forward_kinematics(model, state);
        state.body_xform = xforms;

        // Find ground contacts against the full collision set — every shape
        // in `Body::collisions`, offsets included, not just the centred
        // `Body::geometry`. The margin is what keeps a lightly-loaded support
        // point from leaving the contact set while it is still carrying
        // force; see `find_ground_contacts`.
        let mut contacts = find_ground_contacts_model(model, state, ground_height, material.margin);

        // Find body-body contacts
        let body_contacts = find_contacts(model, state);
        contacts.extend(body_contacts);

        // Free velocity: where the system lands after one step with every
        // force except contact. The contact solve then finds the impulses
        // that correct it.
        let qdd = aba(model, state);
        let free_qd = &state.v + &(&qdd * dt);

        if contacts.is_empty() {
            state.v = free_qd;
        } else {
            // Convex contact solve. Unlike the penalty law this replaces,
            // every contact is solved *together* through the Delassus
            // operator, so pressing on one corner of a box correctly unloads
            // another — and friction is a real Coulomb cone with stiction
            // rather than a viscous damper that vanished at low sliding speed.
            let materials = vec![material.clone(); model.bodies.len().max(1)];
            let config = ContactSolverConfig::simulation();
            let asm =
                phyz_contact::assemble(model, state, &contacts, &materials, &free_qd, dt, &config);
            // Seed from the previous step's impulses. A stance foot is solving
            // nearly the same problem every step, and from a cold start PGS
            // spends its whole iteration budget rediscovering `m g dt`.
            let mut cache = self.contact_cache.borrow_mut();
            let seed = cache.warm_start(state, &contacts);
            let solution = solve_contacts_warm(&asm.problem, &config, &seed);
            cache.store(state, &contacts, &solution.impulses);
            // v' = v_free + M^-1 J^T f.
            state.v = &free_qd + &asm.velocity_delta(&solution.impulses);
        }

        // The acceleration the step actually realized, contacts included.
        let realized_qdd = &(&state.v - &v_before) * (1.0 / dt);

        let v_clone = state.v.clone();
        integrate_configuration(model, state.q.as_mut_slice(), v_clone.as_slice(), dt);

        state.time += dt;

        // Update body transforms
        let (xforms, _) = forward_kinematics(model, state);
        state.body_xform = xforms;

        realized_qdd
    }

    /// The **realized** generalized acceleration at `state`: free dynamics plus
    /// whatever the contact solver produces for the contacts active right now.
    ///
    /// This is the acceleration inertial sensors are supposed to see, and the
    /// one to hand to `phyz_world::SensorContext::with_acceleration`. Unlike
    /// [`Simulator::step_with_contacts`] it does not advance or mutate the
    /// state, so it can be called before or after a step to read sensors at
    /// that exact configuration.
    ///
    /// It runs the same detection and solve a step does — same contact set,
    /// same warm start — so it costs roughly one extra step's contact work.
    /// When a step is happening anyway, prefer the value
    /// [`Simulator::step_with_contacts`] returns.
    ///
    /// The contact solve is impulse-based over `model.dt`, so this is the
    /// average acceleration across a step rather than an instantaneous one;
    /// for a resting body the two agree to solver tolerance.
    pub fn contact_acceleration(
        &self,
        model: &Model,
        state: &State,
        ground_height: f64,
        material: &ContactMaterial,
    ) -> DVec {
        let dt = model.dt;
        let mut probe = state.clone();
        let (xforms, _) = forward_kinematics(model, &probe);
        probe.body_xform = xforms;

        let mut contacts =
            find_ground_contacts_model(model, &probe, ground_height, material.margin);
        contacts.extend(find_contacts(model, &probe));

        let qdd = aba(model, &probe);
        if contacts.is_empty() {
            return qdd;
        }

        let free_qd = &probe.v + &(&qdd * dt);
        let materials = vec![material.clone(); model.bodies.len().max(1)];
        let config = ContactSolverConfig::simulation();
        let asm =
            phyz_contact::assemble(model, &probe, &contacts, &materials, &free_qd, dt, &config);
        // Seed from the cache but never `store` back into it, so asking for
        // sensor data cannot perturb the stepping trajectory. The solve is
        // strongly convex, so the seed only moves the iteration count anyway.
        let seed = self
            .contact_cache
            .borrow_mut()
            .warm_start(&probe, &contacts);
        let solution = solve_contacts_warm(&asm.problem, &config, &seed);
        let v_next = &free_qd + &asm.velocity_delta(&solution.impulses);
        &(&v_next - &probe.v) * (1.0 / dt)
    }
}

impl Default for Simulator {
    fn default() -> Self {
        Self::new()
    }
}
