//! Convex soft-contact solve.
//!
//! Per timestep the contact impulses solve the strictly convex problem
//!
//! ```text
//! minimize_f   1/2 f^T (A + R) f + f^T b     subject to   f_c in K_mu(c)
//! ```
//!
//! where `A = J M^-1 J^T` is the Delassus operator (inverse inertia seen in
//! contact space), `R > 0` is the regularizer that MuJoCo calls constraint
//! impedance, `b` is the free (unconstrained) contact-space velocity plus the
//! reference-acceleration bias, and `K_mu` is the Coulomb cone of
//! [`crate::cone`].
//!
//! # Why this form
//!
//! Dropping strict complementarity in favour of a regularized convex program
//! is what makes contact differentiable, and is why MuJoCo contact is soft
//! (Todorov 2011, 2014). With `R > 0` the objective is *strongly* convex, so
//! the solution is unique and single-valued in the parameters — there is
//! exactly one thing to differentiate, and the implicit function theorem
//! applies to it. An LCP's complementarity condition, by contrast, makes the
//! solution piecewise-smooth over a combinatorial active set.
//!
//! # Method
//!
//! An **active-set Newton** solve, globalized by projected Gauss-Seidel.
//!
//! The PGS half is a sweep with a per-contact **staged Coulomb update** —
//! normal impulse first, then the tangential impulse clamped into the friction
//! disc that normal admits. It is cheap, monotone, and good at discovering
//! *which* contacts carry load and which way they slide. It is also, on its
//! own, hopeless on a redundant contact set, and for a structural reason:
//! its rate is set by the conditioning of `A + R`, and where `A` is singular
//! — eight coplanar contacts under one foot supply 24 impulse components to
//! constrain 6 body DoF — all that is left in the null space is `R`, which on
//! the tangential rows is deliberately tiny (see [`regularization_diag`]).
//! A null-space impulse decays over `A_tt / R_t ~ 1e5` sweeps. Acceleration
//! schemes move that constant; they do not move the exponent.
//!
//! The Newton half fixes the active set read off the current iterate, which
//! makes the KKT conditions *linear*, and solves that `3n x 3n` system
//! directly. Redundancy stops mattering: the null-space directions are
//! resolved exactly rather than asymptotically, and because `b = J qd_free`
//! lies in the range of `J` — orthogonal to `null(J^T)` — the minimizer has no
//! null-space component to remove in the first place. Measured on a symmetric
//! coplanar ring, this took thirty-two contacts from "never converges inside
//! 1000 sweeps" to 18 iterations, the same as four.
//!
//! The two alternate, with each Newton proposal accepted only if a real PGS
//! sweep from it reduces the residual. Convergence is still reported against
//! a residual tolerance rather than an iteration count, and the residual still
//! means exactly what it did — "a full staged sweep no longer moves anything"
//! — because a sweep is what measures it. Running to tolerance is what makes
//! the converged point a valid IFT anchor; differentiating a truncated iterate
//! would differentiate the algorithm, not the physics.
//!
//! The linear system Newton solves is `gradient::kkt_matrix` (crate-private),
//! and the regime is read with the same classifier the gradient uses. The
//! gradient's [`crate::gradient::FixedPointSensitivity`] starts from that same
//! matrix and adds the sliding slip-rotation rows (which the solver does not
//! need — its sweeps re-derive the slip direction, while a derivative must
//! linearize it). Sharing the base object is deliberate: this crate has twice
//! shipped a solver and a gradient that disagreed about the system being
//! solved, and both times the symptom was a confident wrong number rather
//! than a failure.
//!
//! The design doc specifies a primal-dual interior-point SOCP for the final
//! form, whose central-path parameter doubles as the gradient-smoothing knob.
//! This module solves the *same convex problem* by a different inner method;
//! `regularization` plays the smoothing role for now. Swapping the inner solver
//! does not change the problem being solved, the KKT conditions, or the IFT
//! derivation built on them.

use crate::material::ContactMaterial;
use phyz_math::Vec3;

/// How much of the Delassus operator the sweep is allowed to see.
///
/// This is the one knob that separates the CPU and GPU instantiations of the
/// contact model, and it exists so that the difference between them is a
/// *named, measurable approximation* rather than two divergent codebases.
///
/// The problem being solved — the objective, the friction cone, the solref
/// bias, the impedance regularizer, the staged Coulomb update — is identical
/// under both settings. Only the operator changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContactCoupling {
    /// The full `3n x 3n` Delassus operator: every contact sees the velocity
    /// that every other contact's impulse induces on it, through the
    /// articulated chain. This is the physically correct operator, and what
    /// the CPU simulation path uses.
    Full,
    /// Only each contact's own `3x3` diagonal block. Contacts are blind to
    /// one another within a sweep.
    ///
    /// # Why this mode exists
    ///
    /// The off-diagonal blocks `A_ck` are what make the assembled problem
    /// *coupled*, and assembling them requires `M^-1 J^T` for every contact
    /// row — an articulated-body solve per row, per step. On the GPU that is
    /// the difference between a pass that reads each body's own spatial
    /// inertia and a pass that runs `3n` backsubstitutions through the whole
    /// kinematic tree, per environment, per step. The block-diagonal operator
    /// is available for free: it is the isolated-body effective mass the
    /// contact shader already computes.
    ///
    /// # What it costs
    ///
    /// The approximation is *not* uniform. It is exact when contacts share
    /// neither a body nor a chain — two boxes far apart. It is worst for a
    /// redundant manifold on one body (a foot's corner contacts all pushing
    /// the same rigid plate) and for two contacts on opposite ends of a light
    /// chain (two feet coupled through a floating base), where the true `A`
    /// is strongly off-diagonal and the sweep, blind to it, over-assigns load
    /// to every contact at once.
    ///
    /// Running the CPU solver in this mode is how that cost gets measured
    /// without a GPU in the loop: `Full` versus `BlockDiagonal` in `f64` is
    /// the *approximation*, and `BlockDiagonal` versus the GPU is the
    /// *implementation gap*. Confounding those two is what made the earlier
    /// GPU-vs-CPU numbers uninterpretable.
    BlockDiagonal,
    /// Every off-diagonal block between contacts that **share a body**, and
    /// nothing across bodies.
    ///
    /// This is the GPU's operator. It is exact for a contact manifold on one
    /// rigid body — a foot's four corners, a box's face — because those blocks
    /// are `J_c M_i^-1 J_k^T` with `M_i` the single body's own spatial
    /// inertia, which needs no articulated solve and no factorization: it is
    /// the same isolated-body quantity the contact shader already forms for
    /// the diagonal, evaluated at a second contact's lever arm.
    ///
    /// What it still drops is coupling *through the chain* — two feet talking
    /// to each other via a floating base. That term genuinely does require
    /// `M^-1 J^T` and is what the GPU does not pay for.
    ///
    /// Measured (`examples/contact_parity`), the split is lopsided in a way
    /// that decided this design: on a single box landing on its face, going
    /// from `Full` to `BlockDiagonal` moved the trajectory by up to 78 mm,
    /// while `PerBody` is exact there by construction. Nearly all of the
    /// block-diagonal error on a contact manifold is *within-body* coupling,
    /// and within-body coupling is the part that is cheap.
    PerBody,
}

/// Tuning for the contact solve.
#[derive(Debug, Clone, Copy)]
pub struct ContactSolverConfig {
    /// Which Delassus operator the sweep iterates on. See [`ContactCoupling`].
    pub coupling: ContactCoupling,
    /// Allow the active-set Newton stage.
    ///
    /// Off in the GPU-equivalent preset: a dense `3n x 3n` factorization is
    /// not something a contact shader can do, so leaving it on would put the
    /// CPU reference out of the GPU's reach by construction.
    pub newton: bool,
    /// Diagonal regularization `R` added to the Delassus operator.
    ///
    /// Larger values soften contact: more penetration, but a better
    /// conditioned solve and smoother sensitivities. Smaller values approach
    /// rigid contact and sharpen both the physics and the gradients.
    pub regularization: f64,
    /// Residual tolerance for the fixed point (units of velocity).
    pub tolerance: f64,
    /// Iteration cap, counted in units of solver work: one PGS sweep or one
    /// dense KKT solve each. Reaching it means the solve did *not* converge;
    /// the result is still feasible but is not a valid point to differentiate.
    ///
    /// The solve may also stop short of the cap when the residual has stopped
    /// falling fast enough to reach [`Self::tolerance`] within it. That is
    /// reported identically — `converged: false` — because it means the same
    /// thing: this is not a KKT point.
    pub max_iterations: usize,
    /// Successive over-relaxation factor in `(0, 2)`.
    pub relaxation: f64,
    /// Below this approach speed, restitution ramps smoothly to zero.
    ///
    /// Without it a resting body micro-bounces forever and a stack never
    /// settles. The ramp is `smoothstep`, not a hard cutoff, so restitution
    /// stays differentiable in both `e` and the approach speed.
    pub restitution_threshold: f64,
    /// Reproduce MuJoCo's regularization on the **tangential** rows as well as
    /// the normal one. Off by default, and it should stay off outside tests.
    ///
    /// [`regularization_diag`] documents the divergence in full: MuJoCo applies
    /// the impedance-derived `R = (1-d)/d * A_nn` to all three rows of a
    /// contact frame, which on a friction row means a fraction `1-d` of the
    /// slip velocity survives every step. This crate applies it to the normal
    /// row only, so a block below the friction angle holds *exactly*.
    ///
    /// Turning this on re-introduces MuJoCo's steady-state creep —
    /// `g sin(alpha) dt (1-d)/d`, roughly a millimetre per second on a
    /// 25-degree slope at `dt = 2 ms` — for a block Coulomb's law says is
    /// stuck. That is the whole reason it is not the default.
    ///
    /// It exists so MuJoCo trajectories remain usable as a cross-validation
    /// oracle: a divergence in the tangential regularizer makes a step-by-step
    /// comparison against MuJoCo meaningless, and without a way to match its
    /// behaviour there is no way to tell "phyz is right and MuJoCo creeps"
    /// apart from a real bug. `mujoco_creep_matches_the_analytic_rate` in
    /// `tests/stabilization.rs` measures both modes side by side.
    pub mujoco_compat: bool,
}

impl Default for ContactSolverConfig {
    fn default() -> Self {
        Self::simulation()
    }
}

impl ContactSolverConfig {
    /// Fidelity-biased preset: crisp contact, minimal penetration.
    pub fn simulation() -> Self {
        Self {
            coupling: ContactCoupling::Full,
            newton: true,
            regularization: 1e-6,
            tolerance: 1e-10,
            // Generous, and it should stay generous, because it is now almost
            // never approached. With the active-set Newton in the loop a
            // coplanar ring converges in 18 iterations whether it has four
            // contacts or thirty-two, and the worst measured case — a tipped
            // foot sliding into a 2.5cm penetration at 6 m/s with dt = 5ms —
            // gives up at 100 on the stagnation check rather than by running
            // out of budget. The cap only costs anything when the solve is not
            // going to converge, and that case now exits on its own.
            //
            // It used to be the binding constraint: pure Gauss-Seidel wanted
            // ~180 sweeps for eight coplanar contacts, ~530 for sixteen, and
            // more than 1000 for thirty-two, so raising the cap was the only
            // thing keeping the gradient path from refusing points that were
            // in fact fine.
            max_iterations: 1000,
            relaxation: 1.0,
            restitution_threshold: 0.05,
            mujoco_compat: false,
        }
    }

    /// Gradient-biased preset: softer contact, better conditioned, smoother
    /// sensitivities. Documented in `docs/design/differentiable-contact.md`
    /// §2.5 — these differ on purpose, and a caller optimizing through the
    /// simulator should validate the result under [`Self::simulation`].
    pub fn gradients() -> Self {
        Self {
            coupling: ContactCoupling::Full,
            newton: true,
            regularization: 1e-3,
            tolerance: 1e-12,
            max_iterations: 2000,
            relaxation: 1.0,
            restitution_threshold: 0.05,
            mujoco_compat: false,
        }
    }

    /// [`Self::simulation`] with [`Self::mujoco_compat`] set — a test-only
    /// oracle preset for cross-validating trajectories against MuJoCo.
    ///
    /// Sited alongside [`Self::gradients`] because it is the same kind of
    /// thing: a named point in config space that trades one property away on
    /// purpose. `gradients` trades fidelity for smoothness; this one trades
    /// exact stiction for bit-comparability with a reference implementation.
    /// Unlike `gradients`, it is not a preset anyone should ship with — see
    /// the field docs for the creep it buys back.
    pub fn mujoco_compat() -> Self {
        Self {
            mujoco_compat: true,
            ..Self::simulation()
        }
    }

    /// The exact problem, operator and schedule the GPU contact pass runs.
    ///
    /// This is the CPU's *model* of the GPU: same convex problem, same staged
    /// Coulomb update, same solref bias and impedance regularizer, but the
    /// block-diagonal Delassus operator, no Newton stage, and a fixed sweep
    /// budget instead of a residual tolerance. Nothing here is a different
    /// contact model — it is the same model under a documented restriction.
    ///
    /// It is the referee for GPU parity. A gap between this and the GPU is an
    /// implementation bug; a gap between this and [`Self::simulation`] is the
    /// price of the restriction. The two must be measured separately or
    /// neither number means anything.
    ///
    /// `tolerance` is zero so the sweep budget always runs to completion: the
    /// shader has no early exit (a workgroup cannot cheaply agree that every
    /// contact has converged), so an early-exiting reference would drift from
    /// it on exactly the easy steps where the GPU keeps sweeping. `converged`
    /// is therefore `false` here by construction, which is honest — a
    /// fixed-budget iterate is not a KKT point and must not anchor an IFT
    /// gradient.
    pub fn gpu_equivalent() -> Self {
        Self {
            coupling: ContactCoupling::BlockDiagonal,
            newton: false,
            tolerance: 0.0,
            max_iterations: GPU_SWEEPS,
            ..Self::simulation()
        }
    }
}

/// Sweeps the GPU contact pass performs per step, and therefore the budget
/// [`ContactSolverConfig::gpu_equivalent`] matches.
///
/// The shader runs a fixed count with no early exit, so this number is part of
/// the model rather than a tuning detail: change it in one place and the other
/// stops being a reference for it.
pub const GPU_SWEEPS: usize = 16;

/// One contact's entry in the convex problem.
#[derive(Debug, Clone, Copy)]
pub struct ContactRow {
    /// Coefficient of friction.
    pub mu: f64,
    /// Coefficient of restitution, already effective (threshold applied).
    pub restitution: f64,
    /// Penetration depth (positive = overlapping).
    pub depth: f64,
    /// Target *separating* normal velocity, in m/s, from the solref reference
    /// response (see [`crate::material::SolRef::error_reduction`]).
    ///
    /// This is the position-stabilization term. The normal solve drives the
    /// post-step normal velocity to `bias` rather than to zero, so a contact
    /// that is `d` deep is pushed apart at `d * erp / dt` and the penetration
    /// is actually repaid. With `bias = 0` — the default, and what every
    /// hand-built problem in the tests uses — the solve is exactly the
    /// non-penetration constraint it always was, and a stack creeps down by
    /// one integration error per step forever.
    ///
    /// It is kept out of `free_velocity` on purpose: `b = J qd_free` stays a
    /// pure function of the state, and the bias enters as its own parameter,
    /// so `df/db` from [`crate::gradient`] needs no reinterpretation and
    /// `df/dbias = -df/db` falls out for free.
    pub bias: f64,
    /// Constraint impedance `d` in `(0, 1]` from solimp at this contact's
    /// depth. The diagonal regularizer for the contact is
    /// `R = (1-d)/d * diag(A)`, floored by
    /// [`ContactSolverConfig::regularization`]. `1` means "as rigid as the
    /// config floor allows".
    pub impedance: f64,
    /// `d(impedance)/d(depth)` at this contact's depth, from
    /// [`ContactMaterial::dimpedance_ddepth`].
    ///
    /// Carried on the row rather than recomputed because the material is only
    /// in scope at assembly time, and because [`crate::gradient`] must
    /// differentiate *the impedance this row was actually built with*. It is
    /// the derivative of the same `impedance` field directly above it; the two
    /// are set together by [`Self::from_material`] and should never be set
    /// apart.
    ///
    /// Zero on a hand-built row, which is the right default: a row whose
    /// impedance was pinned by hand does not vary with depth.
    pub dimpedance_ddepth: f64,
}

impl ContactRow {
    /// Build a row from a (already pair-combined) material.
    ///
    /// `depth` is the penetration in metres (negative or zero means the
    /// surfaces are not overlapping, and no stabilization is applied),
    /// `dt` the step the impulses will be applied over, and `restitution` the
    /// effective restitution after the low-speed ramp.
    ///
    /// The bias is `d * erp * depth / dt`: `erp` is the fraction of the
    /// violation the reference spring removes in one step, dividing by `dt`
    /// turns that displacement into the separating velocity the impulse solve
    /// speaks in, and scaling by the impedance `d` is what makes a
    /// just-touching contact push back gently instead of snapping. Since
    /// `erp <= 1` and `d < 1`, the bias can never exceed `depth/dt`, i.e. the
    /// solve can never push the bodies further apart than they overlap — the
    /// failure mode that makes naive Baumgarte stabilization explode.
    pub fn from_material(
        material: &ContactMaterial,
        depth: f64,
        dt: f64,
        restitution: f64,
    ) -> Self {
        let violation = depth.max(0.0);
        // `impedance_at` is `solimp.impedance` exactly on the penetrating side
        // and tapers to zero across `material.margin` below it, so a contact
        // detected within the margin sheds its force continuously instead of
        // being cut off while still carrying load. See
        // [`ContactMaterial::impedance_at`].
        let d = material.impedance_at(depth);
        let bias = if dt > 0.0 {
            d * material.solref.error_reduction(dt) * violation / dt
        } else {
            0.0
        };
        Self {
            mu: material.friction,
            restitution,
            depth,
            bias,
            impedance: d,
            dimpedance_ddepth: material.dimpedance_ddepth(depth),
        }
    }
}

impl Default for ContactRow {
    fn default() -> Self {
        Self {
            mu: 0.0,
            restitution: 0.0,
            depth: 0.0,
            bias: 0.0,
            // Fully rigid: the regularizer falls back to the config floor.
            impedance: 1.0,
            dimpedance_ddepth: 0.0,
        }
    }
}

/// The assembled convex contact problem.
///
/// `delassus` is row-major `3n x 3n`, ordered `[n, u, w]` per contact.
#[derive(Debug, Clone)]
pub struct ContactProblem {
    /// Number of contacts.
    pub n: usize,
    /// `A = J M^-1 J^T`, row-major `3n x 3n`.
    pub delassus: Vec<f64>,
    /// Free contact-space velocity `b`, length `3n`.
    pub free_velocity: Vec<f64>,
    /// Per-contact parameters.
    pub rows: Vec<ContactRow>,
    /// The body pair each contact acts between, `(body_i, body_j)`, with
    /// `usize::MAX` for the static world — the same sentinel
    /// [`phyz_collision::Collision`] uses.
    ///
    /// Only [`ContactCoupling::PerBody`] reads this; it is what decides
    /// whether two contacts share a body and therefore whether their
    /// off-diagonal block is one the GPU can afford.
    ///
    /// A map whose length is not `n` means "body structure unknown", and
    /// `PerBody` then degrades to [`ContactCoupling::Full`] rather than
    /// dropping blocks it cannot justify dropping. Failing toward *more*
    /// coupling is the safe direction: it costs time, where the other
    /// direction silently changes the physics. Hand-built problems in the
    /// tests take this path.
    pub bodies: Vec<(usize, usize)>,
}

/// The outcome of a contact solve.
#[derive(Debug, Clone)]
pub struct ContactSolution {
    /// Impulse per contact in its own frame, `[normal, u, w]`.
    pub impulses: Vec<Vec3>,
    /// Iterations actually used.
    pub iterations: usize,
    /// Final residual (max over contacts of the fixed-point movement).
    pub residual: f64,
    /// Whether the residual reached `config.tolerance`.
    ///
    /// A `false` here means the impulses are feasible but the point is not a
    /// converged KKT point, so it must not be used as an IFT anchor.
    pub converged: bool,
}

impl ContactProblem {
    /// Effective restitution after the smooth low-speed ramp.
    ///
    /// `smoothstep` between `v_rest` and `2*v_rest` keeps this `C^1` in the
    /// approach speed; a hard `if |v_n| < eps { 0 }` would not be, and a
    /// gradient through it would be wrong exactly at rest.
    pub fn effective_restitution(e: f64, approach_speed: f64, v_rest: f64) -> f64 {
        if v_rest <= 0.0 {
            return e;
        }
        let s = approach_speed.abs();
        if s <= v_rest {
            return 0.0;
        }
        if s >= 2.0 * v_rest {
            return e;
        }
        let t = (s - v_rest) / v_rest;
        e * t * t * (3.0 - 2.0 * t)
    }
}

/// Diagonal regularizer `R` for contact `c`, one entry per row of its frame.
///
/// On the **normal** row, `R = (1-d)/d * A_nn` — MuJoCo's impedance form.
/// Scaling by the Delassus diagonal is what makes it dimensionally consistent
/// and mass-independent, so the same `solimp` behaves the same on a gram and
/// on a tonne. A flat scalar (what `ContactSolverConfig::regularization` used
/// to be on its own) does not have that property: it softens light bodies into
/// mush and leaves heavy ones rigid.
///
/// On the **tangential** rows it is only the config floor, and that departure
/// from MuJoCo is deliberate. Impedance `d` means "cancel a fraction `d` of
/// the violation", and on a friction row the violation is the slip velocity,
/// so `d = 0.9` leaves a tenth of the slip in place *every step*. That is a
/// steady-state creep of `g sin(alpha) dt (1-d)/d` — about a millimetre per
/// second on a 25-degree slope at 2 ms, forever, for a block that Coulomb's
/// law says is stuck. Stiction is not a soft constraint that a reference
/// response has to reach gradually; it either holds exactly or the contact is
/// sliding, and the staged solve already decides which. MuJoCo pays this creep
/// (it is a known characteristic of its solver); this crate's analytic
/// benchmarks assert exact stiction, so it does not.
///
/// The config value is a floor on every row, which is what the
/// [`ContactSolverConfig::gradients`] preset uses to buy extra smoothing.
///
/// [`ContactSolverConfig::mujoco_compat`] opts back into MuJoCo's form on the
/// tangential rows, creep and all, for trajectory cross-validation. It is off
/// by default and the paragraph above is why.
pub fn regularization_diag(
    problem: &ContactProblem,
    c: usize,
    config: &ContactSolverConfig,
) -> [f64; 3] {
    let dim = 3 * problem.n;
    let base = 3 * c;
    let d = problem.rows[c].impedance.clamp(1e-6, 1.0);
    let scale = (1.0 - d) / d;
    let a_nn = problem.delassus[base * dim + base];
    let normal = (scale * a_nn).max(config.regularization);
    let tangent = if config.mujoco_compat {
        normal
    } else {
        config.regularization
    };
    [normal, tangent, tangent]
}

/// `d(R)/d(depth)` for contact `c`, one entry per row of its frame.
///
/// The exact derivative of [`regularization_diag`], and it must stay exact
/// against *that function* rather than against the formula in its docs — every
/// clamp and floor in the regularizer is a place where this channel is shut
/// off, and a derivative that ignored them would be reporting motion in a
/// quantity that is pinned:
///
/// - The impedance is clamped to `[1e-6, 1]` before use. Outside that range `R`
///   does not respond to `d` at all, so the derivative is zero.
/// - The normal entry is floored at `config.regularization`. Where the floor
///   binds, `R` is constant and the derivative is zero.
/// - The tangential entries are the config floor unless
///   [`ContactSolverConfig::mujoco_compat`], so they are zero unless that flag
///   puts the impedance-derived value on them too.
///
/// Inside the live region `R = (1-d)/d * A_nn`, so `dR/dd = -A_nn/d^2` and the
/// chain rule closes with `dd/ddepth` from the row.
///
/// This exists because [`crate::gradient::depth_sensitivity`] needs it: with a
/// contact margin, a separated-but-detected contact has zero stabilization
/// bias, which makes the regularizer the *only* route from depth to force.
pub fn regularization_depth_derivative(
    problem: &ContactProblem,
    c: usize,
    config: &ContactSolverConfig,
) -> [f64; 3] {
    let dim = 3 * problem.n;
    let base = 3 * c;
    let row = &problem.rows[c];
    let d_raw = row.impedance;
    let d = d_raw.clamp(1e-6, 1.0);
    // Where the clamp binds, `R` is frozen with respect to the impedance.
    let dd = if d_raw > 1e-6 && d_raw < 1.0 {
        row.dimpedance_ddepth
    } else {
        0.0
    };
    let a_nn = problem.delassus[base * dim + base];
    let normal = if (1.0 - d) / d * a_nn > config.regularization {
        -a_nn * dd / (d * d)
    } else {
        // The floor binds: `R` is the constant `config.regularization`.
        0.0
    };
    let tangent = if config.mujoco_compat { normal } else { 0.0 };
    [normal, tangent, tangent]
}

/// Solve the convex contact problem from a cold start (zero impulses).
///
/// [`solve_contacts_warm`] is the same solve seeded with a previous step's
/// impulses; prefer it inside a stepper.
// The loop indices here drive stride arithmetic into flat, row-major arrays
// (base = 3*c, k*dim + ...). Iterator form would hide the linear algebra, so
// the explicit ranges stay.
#[allow(clippy::needless_range_loop)]
pub fn solve_contacts(problem: &ContactProblem, config: &ContactSolverConfig) -> ContactSolution {
    solve_contacts_warm(problem, config, &[])
}

/// Solve the convex contact problem, seeded with `initial`.
///
/// Warm starting is worth a module-level comment because it is not merely an
/// optimization here. The stance contacts of a standing or walking body are
/// solving *almost the same problem* every step, and the seed carries both the
/// impulses and — more valuably — the active set, which is the input Newton
/// cannot derive for itself. Seeded with last step's impulses a resting
/// manifold terminates in two sweeps without ever building a KKT matrix.
/// Since the solver only reports `converged` when it reaches
/// `config.tolerance` — and [`crate::gradient`] refuses to differentiate
/// anything else — cheaper convergence is directly more usable gradients.
///
/// The seed cannot change the answer: the problem is strongly convex, so the
/// minimizer is unique and every start converges to it. `initial` may be
/// shorter than `problem.n` (or empty, for a cold start); missing entries
/// start at zero. Entries outside the friction cone are harmless — the first
/// sweep projects them back.
// The loop indices here drive stride arithmetic into flat, row-major arrays
// (base = 3*c, k*dim + ...). Iterator form would hide the linear algebra, so
// the explicit ranges stay.
#[allow(clippy::needless_range_loop)]
pub fn solve_contacts_warm(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
    initial: &[Vec3],
) -> ContactSolution {
    solve_contacts_warm_diff(problem, config, initial, None, None).0
}

/// [`solve_contacts_warm`], optionally carrying a parameter differential
/// through the very same iteration schedule.
///
/// # The solver-level adjoint, in one function
///
/// [`crate::gradient::FixedPointSensitivity`] differentiates the *fixed point*
/// via the implicit function theorem. That is exact only if the solve reached
/// one. This crate's own solver frequently does not — on a 15-contact skate
/// stance it stops at a residual of `1e-7` after hundreds of sweeps, and with
/// the stagnation exit on it gives up at `1.7e-4` after 136. The IFT then
/// differentiates a point the forward pass never computed, and the two answers
/// diverge in proportion to the leftover residual.
///
/// This differentiates *the algorithm*: the finite composition of sweeps and
/// Newton steps that actually ran. It is exact for any iteration count,
/// converged or not, because the iterate is a perfectly well-defined function
/// of the parameters — just not a fixed point.
///
/// # Why re-execution instead of a tape
///
/// The obvious construction records each iteration's branch data and replays it
/// transposed. Measured, this stance takes 112–4000 sweeps per step, which puts
/// a per-sweep-per-contact tape at hundreds of KB. But the solve is
/// *deterministic*: re-running it reproduces every branch bit-for-bit, so the
/// differential can simply ride alongside the primal in a second execution and
/// nothing needs storing at all. The "tape" costs `O(n)` — the seed — and the
/// derivative is guaranteed to follow the branch the primal took because it is
/// computed in the same expression that took it.
///
/// # What the caller must supply, and the one thing it cannot
///
/// `diff` carries `d(A + R)` and `d(b - e_n bias)`, and its `df` field is the
/// differential of `initial`. That last one is the honest caveat: under a warm
/// start the seed is the previous step's impulses, so `d(initial)` is not zero
/// and a caller that passes zero is differentiating at a frozen seed. At a
/// converged fixed point that is exactly right (the answer is seed-independent);
/// at a truncated one it is an approximation of the same order as the truncation
/// being removed. `PHYZ_CONTACT_COLD_START=1` makes the seed identically zero
/// and the result unconditionally exact, which is how this is validated.
pub(crate) fn solve_contacts_warm_diff(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
    initial: &[Vec3],
    diff: Option<SweepDiff>,
    mut tape: Option<&mut Vec<TapeStep>>,
) -> (ContactSolution, Option<Vec<Vec3>>) {
    let n = problem.n;
    let mut f = vec![Vec3::zeros(); n];
    for (slot, seed) in f.iter_mut().zip(initial) {
        *slot = *seed;
    }
    let mut diff = diff.map(|mut d| {
        d.df.resize(n, Vec3::zeros());
        d
    });
    if n == 0 {
        return (
            ContactSolution {
                impulses: f,
                iterations: 0,
                residual: 0.0,
                converged: true,
            },
            diff.map(|d| d.df),
        );
    }

    let dim = 3 * n;
    debug_assert_eq!(problem.delassus.len(), dim * dim);
    debug_assert_eq!(problem.free_velocity.len(), dim);

    // Per-contact 3x3 diagonal block, regularized and inverted once.
    let blocks = regularized_blocks(problem, config);

    let mut iterations = 0;

    // Preconditioned seed acceleration (`PHYZ_CONTACT_PRECOND=1`): drive the
    // seed to the staged fixed point with the complete-linearization Newton
    // before anything is recorded. Primal-only and deterministic, so both
    // adjoint modes replay it as part of the seed. See [`accelerate`].
    // Guarded off for fixed-budget presets: with `tolerance: 0` (the GPU
    // reference) there is no fixed point to certify and no early exit, so the
    // accelerator would be pure overhead.
    if precond_enabled() && config.tolerance > 0.0 {
        accelerate(problem, config, &blocks, &mut f);
    }

    // ---------------------------------------------------------------- stage 1
    //
    // Projected Gauss-Seidel warm-up. Two things come out of it that the
    // Newton stage below cannot get for itself: a feasible iterate, and an
    // *active set* — which contacts carry load, which slide, and in which
    // direction. Newton takes the active set as given and solves the resulting
    // linear system exactly; PGS is what discovers it, and it is good at that
    // even where it is hopeless at the last digits.
    //
    // Phase 1a holds the tangential impulses fixed and equilibrates the
    // normals alone; phase 1b is the full staged sweep. This is a
    // solver-internal schedule, not a change of problem — the same KKT system
    // is solved either way.
    //
    // It exists because of what redundancy does to the *transient*. On a
    // rank-deficient manifold — eight coplanar contacts under one foot, say —
    // the first sweeps hand almost the whole load to whichever contact is
    // swept first, which tips the body, which is real tangential velocity, so
    // friction engages hard. By the time the normals have levelled out that
    // slip is gone, but the friction impulses it created live in the null
    // space of `J^T`: they cancel in the net wrench, so `A` exerts no
    // restoring force on them at all. Equilibrating the normals first means
    // the slip that seeds them never happens, which leaves the active set
    // handed to Newton clean. Tangential impulses are held, not zeroed, so a
    // warm start keeps its friction seed.
    let normal_warmup = WARMUP_SWEEPS.min(config.max_iterations);
    for _ in 0..normal_warmup {
        iterations += 1;
        if let Some(t) = tape.as_deref_mut() {
            t.push(TapeStep::Sweep {
                entry: f.clone(),
                normals_only: true,
            });
        }
        if sweep(problem, config, &blocks, &mut f, true, diff.as_mut()) < config.tolerance {
            break;
        }
    }

    let mut residual = f64::INFINITY;
    while iterations < config.max_iterations.min(2 * WARMUP_SWEEPS) {
        iterations += 1;
        if let Some(t) = tape.as_deref_mut() {
            t.push(TapeStep::Sweep {
                entry: f.clone(),
                normals_only: false,
            });
        }
        residual = sweep(problem, config, &blocks, &mut f, false, diff.as_mut());
        if residual < config.tolerance {
            return (
                ContactSolution {
                    impulses: f,
                    iterations,
                    residual,
                    converged: true,
                },
                diff.map(|d| d.df),
            );
        }
    }

    // ---------------------------------------------------------------- stage 2
    //
    // Active-set Newton. This is the part that makes a redundant contact set
    // tractable, and the reason is structural rather than a matter of tuning.
    //
    // PGS iterates on `A + R`, so its rate is set by that operator's
    // conditioning. On a redundant manifold `A` is singular — thirty-two
    // coplanar contacts supply 96 impulse components to constrain 6 body DoF —
    // and in the null space the only thing left is `R`, which on the
    // tangential rows is deliberately tiny (`regularization_diag` explains at
    // length why it has to stay tiny for stiction to be exact). A null-space
    // impulse therefore decays with a time constant of `A_tt / R_t ~ 1e5`
    // sweeps. That is not a mode you accelerate your way out of: over-
    // relaxation, Chebyshev and momentum all move the constant, not the
    // exponent.
    //
    // A direct solve does not care. Given the active set, the KKT system is
    // *linear*, `3n x 3n`, and dense-factorizable in `O(n^3)` — 96 unknowns
    // for the thirty-two-contact case, which is nothing next to a thousand
    // sweeps of an `O(n^2)` kernel. The null-space directions are handled
    // exactly instead of asymptotically, and since `b = J qd_free` lies in the
    // range of `J`, which is orthogonal to `null(J^T)`, the minimizer's
    // null-space component is zero: the direct solve simply returns it.
    //
    // The matrix is [`crate::gradient::kkt_matrix`] — the same function
    // [`crate::gradient::impulse_sensitivity`] differentiates through. Solving
    // and differentiating the same object is the invariant that keeps the two
    // from drifting apart.
    //
    // Newton alternates with blocks of PGS rather than running to exhaustion.
    // The two failure modes are complementary: PGS is slow when the *linear
    // algebra* is ill-conditioned (redundancy), Newton is unreliable when the
    // *active set* is wrong (a sliding contact whose slip direction is still
    // rotating during an impact, where the semi-smooth system's linearization
    // point moves under it). A block of sweeps between attempts re-establishes
    // the slip directions cheaply and hands Newton a better linearization
    // point, and a damped step keeps a bad proposal from undoing progress.
    let mut newton_solves = 0;
    let mut newton_none = 0usize;
    let mut newton_rejected = 0usize;
    let mut newton_accepted = 0usize;
    let mut stalls = 0;
    while iterations < config.max_iterations {
        let entry_residual = residual;
        let proposal = if config.newton && newton_solves < NEWTON_ATTEMPTS {
            let p = newton_step_diff(problem, config, &f, diff.as_ref());
            if p.is_none() {
                newton_none += 1;
            }
            p
        } else {
            None
        };
        if let Some((candidate, d_candidate)) = proposal {
            newton_solves += 1;
            iterations += 1;
            // A Newton iterate is only a *proposal*: the active set it was
            // built from may be wrong, and nothing in the linear solve
            // enforces the cone. Verifying it with a real PGS sweep costs one
            // iteration and settles both questions at once — the sweep
            // re-projects into the cone, and the movement it reports is the
            // same residual the pure-PGS solver reported, so `converged` keeps
            // its exact former meaning: "a full staged sweep no longer moves
            // anything".
            //
            // Backtracking on that residual is the globalization. The friction
            // cone is convex, so every point on the segment from `f` to the
            // proposal is feasible and the halved steps need no re-projection;
            // and because the accepted iterate is the *swept* one, a rejected
            // Newton step costs sweeps, never progress.
            let mut alpha = 1.0;
            for _ in 0..LINE_SEARCH_STEPS {
                let mut trial: Vec<Vec3> = f
                    .iter()
                    .zip(&candidate)
                    .map(|(cur, cand)| *cur + (*cand - *cur) * alpha)
                    .collect();
                // The differential of the same interpolation. `alpha` is a
                // recorded discrete choice and enters as a constant: it is a
                // step length selected by a comparison, piecewise constant in
                // the parameters, and differentiating the branch the forward
                // took is the whole contract here.
                let mut d_trial: Option<Vec<Vec3>> = diff.as_ref().map(|d| {
                    d.df.iter()
                        .zip(&d_candidate)
                        .map(|(cur, cand)| *cur + (*cand - *cur) * alpha)
                        .collect()
                });
                iterations += 1;
                let trial_residual = match (diff.as_mut(), d_trial.take()) {
                    (Some(d), Some(dt)) => {
                        // Sweep the trial with its own differential, then keep
                        // both or discard both — a rejected proposal must leave
                        // the derivative exactly as untouched as it leaves `f`.
                        let saved = std::mem::replace(&mut d.df, dt);
                        let res = sweep(problem, config, &blocks, &mut trial, false, Some(d));
                        d_trial = Some(std::mem::replace(&mut d.df, saved));
                        res
                    }
                    _ => sweep(problem, config, &blocks, &mut trial, false, None),
                };
                if trial_residual < residual {
                    // Only an *accepted* proposal is on the tape. A rejected
                    // one leaves `f` and `df` byte-for-byte where they were, so
                    // it contributes nothing to the tangent map and the reverse
                    // pass must not walk it. The interpolated primal is
                    // recomputed rather than saved: it is the same expression
                    // evaluated on the same `f`, `candidate` and `alpha`, so it
                    // reproduces bit-for-bit, and saving a clone per *trial*
                    // would pay for the rejections too.
                    if let Some(t) = tape.as_deref_mut() {
                        t.push(TapeStep::Newton {
                            f: f.clone(),
                            alpha,
                        });
                        t.push(TapeStep::Sweep {
                            entry: f
                                .iter()
                                .zip(&candidate)
                                .map(|(cur, cand)| *cur + (*cand - *cur) * alpha)
                                .collect(),
                            normals_only: false,
                        });
                    }
                    residual = trial_residual;
                    f = trial;
                    if let (Some(d), Some(dt)) = (diff.as_mut(), d_trial) {
                        d.df = dt;
                    }
                    newton_accepted += 1;
                    break;
                }
                alpha *= 0.5;
            }
            if newton_accepted + newton_rejected < newton_solves {
                newton_rejected += 1;
            }
            if residual < config.tolerance {
                break;
            }
        }

        // PGS block: monotone coordinate descent on a strongly convex
        // objective, so this always makes progress even where Newton will not.
        // It is also what refines the active set for the next attempt.
        for _ in 0..PGS_BLOCK {
            if iterations >= config.max_iterations {
                break;
            }
            iterations += 1;
            if let Some(t) = tape.as_deref_mut() {
                t.push(TapeStep::Sweep {
                    entry: f.clone(),
                    normals_only: false,
                });
            }
            residual = sweep(problem, config, &blocks, &mut f, false, diff.as_mut());
            if residual < config.tolerance {
                break;
            }
        }
        if residual < config.tolerance {
            break;
        }

        // Stagnation exit. A block that removed less than `STALL_RATIO` of the
        // residual is on a linear rate that cannot reach the tolerance inside
        // the cap — `0.99` per block over the whole remaining budget is barely
        // a factor of two — so the sweeps that follow are pure cost. This
        // matters for throughput rather than correctness: a batched trainer's
        // step time is set by its *worst* contact solve, and burning the full
        // cap on a solve that was never going to converge is the difference
        // between a bounded step and a stall.
        //
        // The result is reported honestly as `converged: false`, exactly as an
        // exhausted cap would be, and [`crate::gradient`] refuses it on the
        // same grounds. Nothing downstream can mistake an early exit for
        // success.
        if census_enabled() {
            eprintln!(
                "  block: iters={} residual={:.3e} E={:.12e} newton_ok={} rej={}",
                iterations,
                residual,
                qp_objective(problem, config, &f),
                newton_accepted,
                newton_rejected,
            );
        }
        if residual > entry_residual * STALL_RATIO {
            stalls += 1;
            if stalls >= STALL_BLOCKS && !no_stall_exit() {
                break;
            }
        } else {
            stalls = 0;
        }
    }

    if let Ok(path) = std::env::var("PHYZ_PROBLEM_DUMP")
        && !path.is_empty()
        && residual >= config.tolerance
        && n >= dump_min()
        && !std::path::Path::new(&path).exists()
    {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(s, "n {n}");
        let _ = writeln!(
            s,
            "delassus {}",
            problem
                .delassus
                .iter()
                .map(|v| format!("{v:.17e}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        let _ = writeln!(
            s,
            "free_velocity {}",
            problem
                .free_velocity
                .iter()
                .map(|v| format!("{v:.17e}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        for row in &problem.rows {
            let _ = writeln!(
                s,
                "row {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
                row.mu, row.restitution, row.depth, row.bias, row.impedance, row.dimpedance_ddepth
            );
        }
        for b in &problem.bodies {
            let _ = writeln!(s, "body {} {}", b.0, b.1);
        }
        let _ = writeln!(
            s,
            "seed {}",
            initial
                .iter()
                .map(|v| format!("{:.17e} {:.17e} {:.17e}", v.x, v.y, v.z))
                .collect::<Vec<_>>()
                .join(" ")
        );
        let _ = std::fs::write(&path, s);
    }
    if census_enabled() {
        eprintln!(
            "PHYZ_CENSUS n={} iters={} residual={:.3e} converged={} \
             newton_ok={} newton_rejected={} newton_none={} tol={:.1e}",
            n,
            iterations,
            residual,
            residual < config.tolerance,
            newton_accepted,
            newton_rejected,
            newton_none,
            config.tolerance,
        );
    }
    (
        ContactSolution {
            impulses: f,
            iterations,
            residual,
            converged: residual < config.tolerance,
        },
        diff.map(|d| d.df),
    )
}

/// The regularized QP objective `E(f) = 1/2 f^T (A + R) f + f^T (b - bias e_n)`.
///
/// This is the strongly convex function the whole solve minimizes over the
/// friction cone (the module doc states it; the sweep's per-contact targets
/// are its exact coordinate minimizers). It exists for the preconditioned
/// line search: the sweep's `max_move` return is a fixed-point residual, not
/// a merit function — it is not monotone along a Newton direction, and
/// accepting on it rejects genuinely descending proposals wholesale
/// (measured on the K1 skate stance: 23-24 rejections per solve, zero
/// acceptances). The objective is what a descent step actually descends.
///
/// Honors the same coupling mask as [`sweep`], so the two always describe the
/// same problem.
#[allow(clippy::needless_range_loop)]
fn qp_objective(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
    f: &[Vec3],
) -> f64 {
    let n = problem.n;
    let dim = 3 * n;
    let at = |r: usize, c: usize| problem.delassus[r * dim + c];
    let mut e = 0.0;
    for c in 0..n {
        let base = 3 * c;
        let reg = regularization_diag(problem, c, config);
        let fc = f[c];
        // 1/2 f_c^T (A f)_c with the coupling mask, plus the diagonal block.
        for row in 0..3 {
            let mut acc = 0.0;
            for k in 0..n {
                if k != c {
                    match config.coupling {
                        ContactCoupling::BlockDiagonal => continue,
                        ContactCoupling::PerBody
                            if problem.bodies.len() == n
                                && !shares_body(problem.bodies[c], problem.bodies[k]) =>
                        {
                            continue;
                        }
                        _ => {}
                    }
                }
                let kb = 3 * k;
                let fk = f[k];
                acc += at(base + row, kb) * fk.x
                    + at(base + row, kb + 1) * fk.y
                    + at(base + row, kb + 2) * fk.z;
            }
            let fc_row = match row {
                0 => fc.x,
                1 => fc.y,
                _ => fc.z,
            };
            e += 0.5 * fc_row * (acc + reg[row] * fc_row);
            e += fc_row * problem.free_velocity[base + row];
        }
        e -= fc.x * problem.rows[c].bias;
    }
    e
}


/// The staged fixed-point residual at an iterate, row for row the equations
/// [`crate::gradient::complete_kkt`] linearizes:
///
/// - separating: `F = f_c` (all three rows);
/// - sticking (and sliding normal rows): the stationarity residual
///   `[(A + R) f + b]_row - bias` on the normal row, `[(A + R) f + b]_row`
///   tangentially;
/// - sliding tangential rows: `F_t = f_t - mu f_n t_hat`, with `t_hat` the
///   direction of the unconstrained tangential minimizer at the iterate
///   (from the same assembly); where that direction is undefined the pin is
///   `f_t` itself.
#[allow(clippy::needless_range_loop)]
fn staged_residual(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
    regimes: &[crate::gradient::ContactRegime],
    that: &[Option<[f64; 2]>],
    f: &[Vec3],
) -> Vec<f64> {
    use crate::gradient::ContactRegime;
    let n = problem.n;
    let dim = 3 * n;
    let at = |r: usize, c: usize| problem.delassus[r * dim + c];
    let mut res = vec![0.0; dim];
    for c in 0..n {
        let base = 3 * c;
        match regimes[c] {
            ContactRegime::Separating => {
                res[base] = f[c].x;
                res[base + 1] = f[c].y;
                res[base + 2] = f[c].z;
            }
            ContactRegime::Sticking => {
                let reg = regularization_diag(problem, c, config);
                for r in 0..3 {
                    let row = base + r;
                    let mut acc = problem.free_velocity[row];
                    for k in 0..n {
                        let kb = 3 * k;
                        let fk = f[k];
                        acc += at(row, kb) * fk.x + at(row, kb + 1) * fk.y + at(row, kb + 2) * fk.z;
                    }
                    let own = match r {
                        0 => f[c].x,
                        1 => f[c].y,
                        _ => f[c].z,
                    };
                    acc += reg[r] * own;
                    res[row] = acc;
                }
                res[base] -= problem.rows[c].bias;
            }
            ContactRegime::Sliding => {
                let reg = regularization_diag(problem, c, config);
                let row = base;
                let mut acc = problem.free_velocity[row];
                for k in 0..n {
                    let kb = 3 * k;
                    let fk = f[k];
                    acc += at(row, kb) * fk.x + at(row, kb + 1) * fk.y + at(row, kb + 2) * fk.z;
                }
                acc += reg[0] * f[c].x;
                res[base] = acc - problem.rows[c].bias;
                match that[c] {
                    Some(that) => {
                        let lim = problem.rows[c].mu * f[c].x;
                        res[base + 1] = f[c].y - lim * that[0];
                        res[base + 2] = f[c].z - lim * that[1];
                    }
                    None => {
                        res[base + 1] = f[c].y;
                        res[base + 2] = f[c].z;
                    }
                }
            }
        }
    }
    res
}


/// The slip direction of the unconstrained tangential minimizer at `f`, per
/// sliding contact — the `t_hat` the staged residual pins against. `None`
/// where the direction (or the tangential block) is degenerate.
///
/// The formula is the one [`crate::gradient::complete_kkt`] uses; this
/// exists so a line search can re-evaluate the *residual* at a trial point
/// without paying for (or freezing) the full linearization: the Newton
/// direction descends the rotating-`t_hat` residual, so the merit must
/// rotate too — measured against a frozen `t_hat`, every near-solution step
/// is rejected.
#[allow(clippy::needless_range_loop)]
fn tangential_dirs(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
    regimes: &[crate::gradient::ContactRegime],
    f: &[Vec3],
) -> Vec<Option<[f64; 2]>> {
    use crate::gradient::ContactRegime;
    let n = problem.n;
    let dim = 3 * n;
    let a = &problem.delassus;
    let mut out: Vec<Option<[f64; 2]>> = vec![None; n];
    for c in 0..n {
        if regimes[c] != ContactRegime::Sliding {
            continue;
        }
        let base = 3 * c;
        let reg = regularization_diag(problem, c, config);
        let m = [
            [a[(base + 1) * dim + base + 1] + reg[1], a[(base + 1) * dim + base + 2]],
            [a[(base + 2) * dim + base + 1], a[(base + 2) * dim + base + 2] + reg[2]],
        ];
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        if det.abs() < 1e-30 {
            continue;
        }
        let minv = [
            [m[1][1] / det, -m[0][1] / det],
            [-m[1][0] / det, m[0][0] / det],
        ];
        // r_t = [(A + R) f + b]_t - M_t f_t, then t* = -M_t^-1 r_t.
        let mut r = [0.0f64; 2];
        for (i, ri) in r.iter_mut().enumerate() {
            let row = base + 1 + i;
            let mut acc = problem.free_velocity[row];
            for k in 0..n {
                let kb = 3 * k;
                let fk = f[k];
                acc += a[row * dim + kb] * fk.x
                    + a[row * dim + kb + 1] * fk.y
                    + a[row * dim + kb + 2] * fk.z;
            }
            let own = if i == 0 { f[c].y } else { f[c].z };
            acc += reg[1 + i] * own;
            acc -= m[i][0] * f[c].y + m[i][1] * f[c].z;
            *ri = acc;
        }
        let t_star = [
            -(minv[0][0] * r[0] + minv[0][1] * r[1]),
            -(minv[1][0] * r[0] + minv[1][1] * r[1]),
        ];
        let t_norm = (t_star[0] * t_star[0] + t_star[1] * t_star[1]).sqrt();
        if t_norm <= 1e-14 {
            continue;
        }
        out[c] = Some([t_star[0] / t_norm, t_star[1] / t_norm]);
    }
    out
}

/// Probe sweeps the accelerator may spend watching for the wedged tail.
///
/// The wedged regime is not visible up front: the skate stance contracts at
/// a healthy 0.74 per sweep for its first dozens of sweeps and only then
/// levels off toward 1, so a short probe reads it as healthy and walks away.
/// The probe sweeps are ordinary solver progress on the actual iterate —
/// a healthy problem simply converges inside them — so the budget is cheap
/// to hold open.
const ACCEL_PROBE_SWEEPS: usize = 64;

/// Iteration units the preconditioned accelerator may spend before handing
/// the iterate to the recorded solve. Each unit is one semismooth Newton
/// solve (`O((3n)^3)`) plus one verification sweep.
const ACCEL_ROUNDS: usize = 40;

/// Drive the iterate to the staged fixed point with semismooth Newton on the
/// *complete* linearization — the preconditioned solve behind
/// `PHYZ_CONTACT_PRECOND=1`.
///
/// # Why this exists, and why it is primal-only
///
/// The census on the K1 skate stance (15-18 contacts) measured the shipped
/// solver converging on ~0/64 to 2/2000 steps: PGS moves the iterate by
/// `~1e-5` per sweep at a linear rate wedged against 1 (the redundant
/// manifold's null space is restrained only by the tiny tangential
/// regularizer), and the in-solve Newton stage — whose sliding pins hold
/// `t_hat` fixed at the direction the iterate happened to have — cannot
/// rotate a slip direction, so on a stance with rotating slip the line
/// search rejects all its proposals and the stagnation exit fires at
/// `1e-4`..`1e-7`.
///
/// This accelerator Newton-iterates on the full staged conditions with the
/// slip-direction rotation channel included — [`crate::gradient::complete_kkt`],
/// the same linearization [`crate::gradient::FixedPointSensitivity`]
/// differentiates at the solution. Near the fixed point that is a genuine
/// Newton method: the same stance lands at machine precision in a handful of
/// solves.
///
/// # What it does and does not achieve, measured
///
/// It converges problems the shipped solver stalls on — an 8-contact K1
/// stance snapshot goes from 272 iterations at `4e-8` (stall exit, refused by
/// the gradient) to 2 iterations at `2e-13`, and a settled 4-contact stack
/// from 192 at `2e-9` to 2 at machine precision.
///
/// It does **not** solve the case it was built for. On the live 16-contact
/// K1 skate stance, a 2 s settle converges 14/2000 steps against the shipped
/// path's 18/2000, at `11.2 ms/step` against `2.2` — five times the cost for
/// no gain. Two things go wrong there and both are about the *active set*
/// rather than the linear algebra: the refinement below cycles (contacts
/// trading regimes round after round while `||F||` sits at `1e-2`), and when
/// the LM does converge it can converge to a stationary point of the wrong
/// system, which the certification below then rejects — so the work is spent
/// and discarded. Pure PGS on that stance reaches only `3e-8` in 200 000
/// sweeps against a `1e-9` tolerance, so the shipped solver is not merely
/// stopping early there either.
///
/// That is why this is gated off and why the gate should stay off: it is a
/// working mechanism on determinate and mildly redundant contact sets, and an
/// unfinished one on the deeply redundant stance. The next thing to try is a
/// non-monotone or anti-cycling active-set rule (a Fletcher-style filter, or
/// pivoting one contact per round rather than all violators at once).
///
/// It runs *before* anything is recorded, differentiated, or taped, and only
/// mutates the effective warm-start seed: the solve that follows starts at
/// (or near) the fixed point and terminates through the exact code path that
/// always ran, so `converged`, `iterations` and `residual` keep their
/// meanings, and both adjoint modes replay it deterministically as part of
/// the seed. The seed's differential is the caller's, unchanged — at a
/// converged fixed point the answer is seed-independent, which is the same
/// (documented) contract warm starting already relies on.
// Stride arithmetic into flat, row-major arrays (base = 3*c), as
// throughout this module.
#[allow(clippy::needless_range_loop)]
fn accelerate(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
    blocks: &[[[f64; 3]; 3]],
    f: &mut [Vec3],
) {
    let n = problem.n;
    let dim = 3 * n;

    // Health probe: sweep until converged, budget, or a wedged tail rate.
    // The first sweeps of any solve make transient progress, so a short probe
    // cannot tell a healthy solve (contraction well under 1, finishes inside
    // the recorded budget) from a wedged one (rate against 1 on a redundant
    // manifold); the tail rate can. Newton only engages on two consecutive
    // near-1 contractions — on healthy problems the probe either finishes the
    // solve outright or hands the recorded solve a better seed, and the
    // frozen-classification Newton (which can pick the wrong attractor when a
    // contact sits within rounding of its cone boundary) never runs.
    // Anything past the probe must certify a fixed point or leave no trace:
    // even the probe's own sweeps can walk a healthy-looking iterate into the
    // boundary-hugging region where the recorded solve's Newton stage is then
    // trapped (measured on a cold incline start: ungated converges at step 0,
    // a probe-advanced-then-reverted-to-probe-endpoint gated solve stalled at
    // `1e-7`). On failure the seed is restored byte-for-byte, so a gated
    // solve the accelerator cannot finish is exactly the shipped solve.
    let checkpoint = f.to_vec();
    let mut prev = f64::INFINITY;
    let mut engage = false;
    for i in 0..ACCEL_PROBE_SWEEPS {
        let mv = sweep(problem, config, blocks, f, false, None);
        if mv < config.tolerance {
            return;
        }
        let ratio = mv / prev;
        prev = mv;
        // Engage on the *projection*, not on a rate threshold: at a
        // contraction of `r` per sweep, reaching the tolerance takes
        // `ln(tol/mv) / ln(r)` more sweeps, and the question is only whether
        // that fits in what the recorded solve has left. A stance contracting
        // at a respectable 0.9 still needs thousands of sweeps from `1e-4`,
        // which is why a rate threshold misses it.
        if i >= 3 && ratio < 1.0 {
            let need = (config.tolerance / mv).ln() / ratio.ln();
            if !need.is_finite() || need > config.max_iterations as f64 {
                engage = true;
                break;
            }
        } else if i >= 3 {
            engage = true;
            break;
        }
    }
    if !engage {
        // The probe's sweeps are ordinary solver progress on the real
        // iterate, so they are kept: the recorded solve simply starts closer.
        return;
    }

    // The Newton phase is pure Levenberg-Marquardt on the staged residual:
    // assemble the complete linearization at `f`, refine the active set from
    // the undamped solution's cone violations, then take a damped step
    // accepted on the *rotated-t_hat* residual. A rejection raises the
    // damping and retries the same linearization — it does NOT fall back to
    // sweeps: on stances with many friction-saturated contacts a sweep
    // rotates the slip directions enough to raise `||F||` twenty-fold and
    // erase the phase's progress (measured on a live 16-contact stance).
    // The sweep's only role here is the final certification below.
    let mut best: Option<(f64, Vec<Vec3>)> = None;
    let mut best_fnorm = f64::INFINITY;
    let mut flat_rounds = 0usize;
    let mut lambda = 1e-4;
    'rounds: for _ in 0..ACCEL_ROUNDS {
        // Active-set refinement (see above): the classification band cannot
        // see a contact the solution wants outside its cone.
        let mut regimes = crate::gradient::classify_impulses(problem, f, 1e-7);
        let mut lin;
        let mut res;
        let mut refine = 0;
        loop {
            lin = crate::gradient::complete_kkt(problem, config, f, &regimes);
            res = staged_residual(problem, config, &lin.regimes, &lin.that, f);
            let mut k = lin.k.clone();
            let mut d: Vec<f64> = res.iter().map(|v| -v).collect();
            if crate::gradient::solve_dense(&mut k, &mut d, dim, 1).is_none()
                || d.iter().any(|v| !v.is_finite())
            {
                break;
            }
            let mut changed = false;
            if refine < 8 {
                for c in 0..n {
                    let base = 3 * c;
                    let t_n = f[c].x + d[base];
                    let (t_u, t_w) = (f[c].y + d[base + 1], f[c].z + d[base + 2]);
                    match regimes[c] {
                        crate::gradient::ContactRegime::Sticking => {
                            if t_n <= 0.0 {
                                regimes[c] = crate::gradient::ContactRegime::Separating;
                                changed = true;
                            } else if (t_u * t_u + t_w * t_w).sqrt() > problem.rows[c].mu * t_n {
                                regimes[c] = crate::gradient::ContactRegime::Sliding;
                                changed = true;
                            }
                        }
                        crate::gradient::ContactRegime::Sliding => {
                            if t_n <= 0.0 {
                                regimes[c] = crate::gradient::ContactRegime::Separating;
                                changed = true;
                            }
                        }
                        crate::gradient::ContactRegime::Separating => {
                            // Re-engagement. Without this the refinement is a
                            // one-way street into Separating and the LM
                            // happily converges to a stationary point of the
                            // wrong system — measured: `||F||` at 7.6e-17 on
                            // an active set the certification sweep rejects,
                            // because a contact that was carrying load had
                            // been released and never taken back. A released
                            // contact whose *own* stationarity residual is
                            // negative is being driven into penetration, so
                            // it belongs back in the active set.
                            let a_nn = problem.delassus[base * dim + base]
                                + regularization_diag(problem, c, config)[0];
                            let mut acc = problem.free_velocity[base] - problem.rows[c].bias;
                            for kk in 0..n {
                                let kb = 3 * kk;
                                acc += problem.delassus[base * dim + kb] * f[kk].x
                                    + problem.delassus[base * dim + kb + 1] * f[kk].y
                                    + problem.delassus[base * dim + kb + 2] * f[kk].z;
                            }
                            if a_nn > 0.0 && acc < 0.0 {
                                regimes[c] = crate::gradient::ContactRegime::Sticking;
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
            refine += 1;
        }
        let fnorm = res.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        if fnorm < best_fnorm {
            if fnorm < 0.7 * best_fnorm {
                flat_rounds = 0;
            }
            best_fnorm = fnorm;
            best = Some((fnorm, f.to_vec()));
        } else {
            flat_rounds += 1;
            if flat_rounds >= 6 {
                break;
            }
        }

        // K^T K and -K^T F once per round; damping retries rescale the
        // diagonal only.
        let k = &lin.k;
        let mut ktk = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut acc = 0.0;
                for r in 0..dim {
                    acc += k[r * dim + i] * k[r * dim + j];
                }
                ktk[i * dim + j] = acc;
            }
        }
        let mut ktf = vec![0.0; dim];
        for (i, slot) in ktf.iter_mut().enumerate() {
            let mut acc = 0.0;
            for r in 0..dim {
                acc -= k[r * dim + i] * res[r];
            }
            *slot = acc;
        }

        let mut accepted = false;
        for _ in 0..4 {
            let mut m = ktk.clone();
            for i in 0..dim {
                m[i * dim + i] *= 1.0 + lambda;
            }
            let mut delta = ktf.clone();
            if crate::gradient::solve_dense(&mut m, &mut delta, dim, 1).is_none()
                || delta.iter().any(|v| !v.is_finite())
            {
                break 'rounds;
            }
            let mut trial = f.to_vec();
            for c in 0..n {
                let base = 3 * c;
                let f_n = (trial[c].x + delta[base]).max(0.0);
                let (mut t_u, mut t_w) =
                    (trial[c].y + delta[base + 1], trial[c].z + delta[base + 2]);
                let limit = problem.rows[c].mu * f_n;
                let t_norm = (t_u * t_u + t_w * t_w).sqrt();
                if t_norm > limit {
                    let s = if t_norm > 0.0 { limit / t_norm } else { 0.0 };
                    t_u *= s;
                    t_w *= s;
                }
                trial[c] = Vec3::new(f_n, t_u, t_w);
            }
            // Merit: the residual with the slip directions re-derived at the
            // trial. The Newton direction descends the rotating-t_hat
            // residual, so the merit must rotate too.
            let trial_that = tangential_dirs(problem, config, &lin.regimes, &trial);
            let trial_res = staged_residual(problem, config, &lin.regimes, &trial_that, &trial);
            let trial_norm = trial_res.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            if trial_norm < fnorm * 0.999 {
                f.copy_from_slice(&trial);
                lambda = (lambda * 0.25).max(1e-12);
                accepted = true;
                break;
            }
            lambda = (lambda * 10.0).min(1e8);
        }
        if census_enabled() {
            eprintln!(
                "  accel: |F|={fnorm:.3e} lambda={lambda:.1e} accepted={accepted} regimes={:?}",
                lin.regimes
            );
        }
        if !accepted {
            break;
        }
    }

    // Certify or revert. The best iterate by sweep movement is only accepted
    // if one more sweep no longer moves it — i.e. the accelerator actually
    // delivered the staged fixed point. Anything less is discarded outright:
    // a near-miss can sit in the basin of the wrong attractor (the
    // borderline-cone case above), where it reads as \"small movement\" while
    // being far from the point the sweeps converge to, and seeding the
    // recorded solve there is strictly worse than not having run.
    if let Some((_, b)) = best {
        f.copy_from_slice(&b);
        let mv = sweep(problem, config, blocks, f, false, None);
        if mv < config.tolerance {
            return;
        }
    }
    f.copy_from_slice(&checkpoint);
}

/// Whether the preconditioned solve path is on (`PHYZ_CONTACT_PRECOND=1`).
///
/// Default off: with this unset the solver behaves byte-for-byte as shipped.
/// See the module doc of the Newton stage for what it changes.
fn precond_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PHYZ_CONTACT_PRECOND").is_ok_and(|v| v == "1" || v == "true"))
}

/// Minimum contact count for `PHYZ_PROBLEM_DUMP` (default 4), via
/// `PHYZ_PROBLEM_DUMP_MIN`.
fn dump_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PHYZ_PROBLEM_DUMP_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
    })
}

/// Per-solve convergence census on stderr (`PHYZ_SOLVE_CENSUS=1`).
///
/// One line per contact solve: contact count, iteration units, final sweep
/// residual, whether tolerance was reached, and where the Newton attempts
/// went — accepted by the line search, rejected by it, or never constructed
/// (`newton_none`, a singular KKT solve). Diagnostic only; default off.
fn census_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PHYZ_SOLVE_CENSUS").is_ok_and(|v| v == "1" || v == "true"))
}

/// Whether to spend the whole iteration budget rather than exiting on
/// stagnation.
///
/// Off by default: with this unset the solver stops exactly where it always
/// has and every number it reports is unchanged.
///
/// The stagnation exit is a throughput decision — a block that removed less
/// than `STALL_RATIO` of the residual is on a rate that cannot reach the
/// tolerance inside the cap, so the remaining sweeps are pure cost. It gives up
/// honestly (`converged: false`) and [`crate::gradient`] refuses the step, so
/// nothing is silently wrong.
///
/// But it gives up *early*, and "on a linear rate" is a local estimate: a solve
/// that stalls for a few blocks and would then have found its footing is
/// abandoned. On a redundant contact manifold — the case active-set Newton
/// exists for — that is a real possibility, and every abandoned step is a step
/// the adjoint cannot differentiate and therefore a whole window's gradient
/// lost, since the adjoint walks backwards and one refusal kills everything
/// behind it. Set this when a gradient matters more than a step time.
fn no_stall_exit() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PHYZ_NO_STALL_EXIT").is_ok_and(|v| v == "1" || v == "true"))
}

/// PGS sweeps spent establishing the active set before Newton takes over.
///
/// Small on purpose. Its job is to decide which contacts are loaded and which
/// way they slide, not to converge; a determined contact set converges inside
/// it anyway and never pays for the factorization.
const WARMUP_SWEEPS: usize = 8;

/// Cap on active-set changes. Each one is a dense `3n x 3n` solve, and an
/// active set that has not settled in this many revisions is oscillating
/// between two nearly-degenerate assignments — a case for the PGS fallback,
/// not for more Newton steps.
const NEWTON_ATTEMPTS: usize = 24;

/// Backtracking steps allowed per Newton proposal, halving each time.
const LINE_SEARCH_STEPS: usize = 3;

/// PGS sweeps between Newton attempts. Enough to move the slip directions to
/// where the next linearization wants them, few enough that a problem Newton
/// can finish does not pay for a long tail of sweeps first.
const PGS_BLOCK: usize = 4;

/// Residual reduction a Newton-plus-PGS block must achieve to count as
/// progress. Anything above this is a linear rate too slow to reach tolerance
/// inside any sane iteration cap.
const STALL_RATIO: f64 = 0.99;

/// Consecutive stalled blocks before the solve gives up and reports
/// `converged: false`. More than one so a single unlucky block — a Newton
/// proposal rejected by the line search, say — does not end the solve.
const STALL_BLOCKS: usize = 3;

/// One projected Gauss-Seidel sweep with the staged Coulomb update.
///
/// Returns the largest per-contact movement, which is the fixed-point residual
/// the solve terminates on. With `normals_only`, the tangential impulses are
/// held at their current values rather than re-solved.
#[allow(clippy::needless_range_loop)]
/// Do two contacts act on a common body?
///
/// `usize::MAX` is the static world. Every ground contact names it, and it
/// couples nothing — it is immovable, so an impulse on one ground contact
/// induces no velocity at another through the world. Counting it as shared
/// would collapse `PerBody` back into `Full` for the most common case there
/// is, so it is excluded explicitly.
fn shares_body(a: (usize, usize), b: (usize, usize)) -> bool {
    let real = |x: usize| x != usize::MAX;
    (real(a.0) && (a.0 == b.0 || a.0 == b.1)) || (real(a.1) && (a.1 == b.0 || a.1 == b.1))
}

/// The parameter differential a sweep carries when the solver-level adjoint
/// re-executes it.
///
/// # Why this rides *inside* the sweep
///
/// The whole point of a solver-level adjoint is to differentiate the branch the
/// forward pass actually took. Every branch in the staged update is decided at
/// **zero tolerance** — `f_n = max(0.0, .)` and `if t_norm > limit` — so the
/// only way to be sure the derivative agrees with the primal is to compute both
/// from the same values in the same pass. A separate backward routine that
/// re-derived the branches from a band (which is what
/// [`crate::gradient::classify_impulses`] does) is exactly the mismatch this
/// crate has shipped twice.
///
/// So the differential is threaded through the sweep rather than recorded and
/// replayed. That also makes the "tape" free: re-executing the solve is
/// deterministic, so nothing per-sweep needs storing.
///
/// The parameter channel is expressed in the two combinations the staged update
/// actually reads:
///
/// - `d_apr` — the differential of `A + R`, row-major `3n x 3n`. The
///   regularizer is diagonal, so off-diagonal blocks are just `dA`.
/// - `dc` — the differential of `b - e_n * bias`, length `3n`. The normal
///   numerator only ever sees `bias - b_n`, so folding the two together is not
///   a shortcut; it is the exact quantity the arithmetic uses.
pub(crate) struct SweepDiff<'a> {
    /// `d(A + R)`, row-major `3n x 3n`.
    pub d_apr: &'a [f64],
    /// `d(b - e_n bias)`, length `3n`.
    pub dc: &'a [f64],
    /// The running differential of the impulses. Updated in place, exactly
    /// where `f` is.
    pub df: Vec<Vec3>,
}

fn sweep(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
    blocks: &[[[f64; 3]; 3]],
    f: &mut [Vec3],
    normals_only: bool,
    mut diff: Option<&mut SweepDiff>,
) -> f64 {
    let n = problem.n;
    let dim = 3 * n;
    let a = &problem.delassus;
    let at = |i: usize, j: usize| a[i * dim + j];
    let mut max_move: f64 = 0.0;

    for c in 0..n {
        let base = 3 * c;

        // r = b_c + sum_{k != c} A_ck f_k  (Gauss-Seidel: uses updated f)
        //
        // Under `ContactCoupling::BlockDiagonal` the sum is skipped entirely
        // and `r` is just the free velocity: that *is* the approximation, and
        // it is the whole of it. Everything below — the staged normal solve,
        // the tangential 2x2, the isotropic disc clamp — is untouched, so the
        // restricted mode remains the same contact model rather than becoming
        // a second one.
        let mut r = [0.0f64; 3];
        // `dr` mirrors `r` term for term. It accumulates `dc` where `r`
        // accumulates `b`, and the product rule where `r` accumulates
        // `A_ck f_k` — under exactly the same coupling mask, so a restricted
        // operator differentiates as the restricted operator it is.
        let mut dr = [0.0f64; 3];
        for (row, r_row) in r.iter_mut().enumerate() {
            let mut acc = problem.free_velocity[base + row];
            let mut dacc = diff.as_ref().map_or(0.0, |d| d.dc[base + row]);
            if config.coupling != ContactCoupling::BlockDiagonal {
                // `k` indexes `f`, `df` and `d_apr`'s column block together;
                // enumerating one would hide that they must stay in step.
                #[allow(clippy::needless_range_loop)]
                for k in 0..n {
                    if k == c {
                        continue;
                    }
                    // PerBody keeps only the blocks the GPU can form without
                    // an articulated solve: those between contacts on a
                    // shared body.
                    if config.coupling == ContactCoupling::PerBody
                        && problem.bodies.len() == n
                        && !shares_body(problem.bodies[c], problem.bodies[k])
                    {
                        continue;
                    }
                    let kb = 3 * k;
                    let f_k = f[k];
                    acc += at(base + row, kb) * f_k.x
                        + at(base + row, kb + 1) * f_k.y
                        + at(base + row, kb + 2) * f_k.z;
                    if let Some(d) = diff.as_ref() {
                        let dfk = d.df[k];
                        let ri = (base + row) * dim;
                        dacc += d.d_apr[ri + kb] * f_k.x
                            + d.d_apr[ri + kb + 1] * f_k.y
                            + d.d_apr[ri + kb + 2] * f_k.z
                            + at(base + row, kb) * dfk.x
                            + at(base + row, kb + 1) * dfk.y
                            + at(base + row, kb + 2) * dfk.z;
                    }
                }
            }
            *r_row = acc;
            dr[row] = dacc;
        }
        // The contact's own regularized block, and its differential. `blocks`
        // is `A_cc + diag(reg)`, so `d_apr`'s own diagonal block is its exact
        // differential — the regularizer's dependence on depth included.
        let db: [[f64; 3]; 3] = match diff.as_ref() {
            None => [[0.0; 3]; 3],
            Some(d) => {
                let mut m = [[0.0; 3]; 3];
                for (i, mi) in m.iter_mut().enumerate() {
                    for (j, e) in mi.iter_mut().enumerate() {
                        *e = d.d_apr[(base + i) * dim + base + j];
                    }
                }
                m
            }
        };
        let dfc = diff.as_ref().map_or(Vec3::zeros(), |d| d.df[c]);

        // Restitution is already folded into `free_velocity` as a target
        // normal velocity (see `point_mass_problem`) rather than applied
        // as a post-solve velocity reset. That keeps it differentiable in
        // `e` and in the approach speed, and stops it fighting the solver.
        let row = problem.rows[c];

        // Coulomb's law is *staged*, not a Euclidean cone projection of
        // the unconstrained impulse. Projecting `(f_n, f_t)` onto the cone
        // moves the normal component too: for a fast-sliding contact the
        // boundary projection sets `f_n = (mu||f_t|| + f_n)/(mu^2+1)`,
        // which inflates the normal impulse far above the value that
        // actually cancels the approach velocity, and the block then
        // decelerates too hard. (Measured: 2.04 m/s^2 instead of the
        // analytic 2.55 on a 40-degree incline.)
        //
        // The physical statement is conditional: the normal impulse is
        // whatever enforces non-penetration, and *given* that, friction is
        // bounded by `mu * f_n`. So solve the normal first, then clamp the
        // tangential to the disc it admits. Stiction is the case where the
        // unconstrained tangential impulse already lies inside that disc.
        // Non-penetration *plus* position stabilization: drive the
        // post-step normal velocity to `row.bias` (a separating velocity
        // proportional to the current penetration) rather than to zero.
        // Driving it to zero freezes whatever penetration the previous
        // steps accumulated, which is exactly how a stack creeps.
        // The residual above deliberately excludes the whole of contact
        // `c`'s own block, so the *intra*-block coupling has to be put
        // back by hand — and it has to be, or this stops being coordinate
        // descent on the objective and its fixed point stops being the
        // KKT point. `A_nu`/`A_nw` are the lever-arm coupling between this
        // contact's own normal and tangential rows; they are non-zero
        // whenever the contact is off the body's centre of mass, which is
        // every contact in a real support polygon. Dropping them made the
        // normal solve ignore the slip its own friction causes and the
        // tangential solve ignore the load its own normal carries, so the
        // sweep converged to a point where friction saturated at `mu*f_n`
        // with no slip to oppose. On a redundant (rank-deficient) manifold
        // those spurious tangential impulses feed back through the
        // off-diagonal blocks and the sweep walks off along the null
        // space instead of terminating.
        let a_nn = blocks[c][0][0];
        let (a_nu, a_nw) = (blocks[c][0][1], blocks[c][0][2]);
        let unclamped = if a_nn > 0.0 {
            (row.bias - r[0] - a_nu * f[c].y - a_nw * f[c].z) / a_nn
        } else {
            0.0
        };
        let f_n = if a_nn > 0.0 { unclamped.max(0.0) } else { 0.0 };
        // `d(bias - r_0) = -dr_0`: `dc` folds `db - e_n dbias` into one vector
        // precisely so this stays a single term. The quotient rule's second
        // term carries `d(A+R)_nn`, and it is the one that vanishes only at a
        // fixed point — dropping it is what makes an IFT gradient of a
        // finitely-swept iterate wrong.
        let d_f_n = if a_nn > 0.0 && unclamped > 0.0 {
            let d_num = -dr[0] - (db[0][1] * f[c].y + a_nu * dfc.y)
                - (db[0][2] * f[c].z + a_nw * dfc.z);
            (d_num - unclamped * db[0][0]) / a_nn
        } else {
            // The forward took the `max(0.0)` branch (or the degenerate
            // `a_nn <= 0` one): the impulse is pinned, so its differential is.
            0.0
        };

        // Tangential 2x2 solve at the fixed normal impulse — including the
        // velocity that normal impulse itself induces on the tangent rows.
        let (a_un, a_wn) = (blocks[c][1][0], blocks[c][2][0]);
        let r_u = r[1] + a_un * f_n;
        let r_w = r[2] + a_wn * f_n;
        let d_r_u = dr[1] + db[1][0] * f_n + a_un * d_f_n;
        let d_r_w = dr[2] + db[2][0] * f_n + a_wn * d_f_n;
        let (m00, m01) = (blocks[c][1][1], blocks[c][1][2]);
        let (m10, m11) = (blocks[c][2][1], blocks[c][2][2]);
        let det = m00 * m11 - m01 * m10;
        let (mut t_u, mut t_w) = if !normals_only {
            if det.abs() > 1e-18 {
                (
                    -(m11 * r_u - m01 * r_w) / det,
                    -(m00 * r_w - m10 * r_u) / det,
                )
            } else {
                (0.0, 0.0)
            }
        } else {
            (f[c].y, f[c].z)
        };
        // Same three branches, same order. `normals_only` holds the tangential
        // impulses, so their differential is held too — a warm start's friction
        // seed carries its sensitivity through the normal-equilibration phase
        // exactly as it carries its value.
        let (mut d_t_u, mut d_t_w) = if !normals_only {
            if det.abs() > 1e-18 {
                let d_det = db[1][1] * m11 + m00 * db[2][2] - db[1][2] * m10 - m01 * db[2][1];
                (
                    -(db[2][2] * r_u + m11 * d_r_u - db[1][2] * r_w - m01 * d_r_w) / det
                        - t_u * d_det / det,
                    -(db[1][1] * r_w + m00 * d_r_w - db[2][1] * r_u - m10 * d_r_u) / det
                        - t_w * d_det / det,
                )
            } else {
                (0.0, 0.0)
            }
        } else {
            (dfc.y, dfc.z)
        };

        // Clamp into the friction disc of radius mu*f_n. The clamp is
        // isotropic, so a block sliding at any heading loses speed
        // identically — the property a pyramidal cone gives up.
        let limit = row.mu * f_n;
        let t_norm = (t_u * t_u + t_w * t_w).sqrt();
        if t_norm > limit {
            if t_norm > 0.0 {
                let s = limit / t_norm;
                // Differentiate before overwriting: `t_u`/`t_w` below are the
                // pre-scale values this derivative is taken at. Expanded, this
                // is `s (I - t_hat t_hat^T) dt* + mu t_hat df_n` — the same
                // projector `FixedPointSensitivity` carries for a converged
                // sliding contact, here evaluated at whichever iterate the
                // sweep is on rather than at an assumed fixed point.
                if diff.is_some() {
                    let d_t_norm = (t_u * d_t_u + t_w * d_t_w) / t_norm;
                    let d_s = (row.mu * d_f_n - s * d_t_norm) / t_norm;
                    let (nu, nw) = (s * d_t_u + t_u * d_s, s * d_t_w + t_w * d_s);
                    d_t_u = nu;
                    d_t_w = nw;
                }
                t_u *= s;
                t_w *= s;
            } else {
                t_u = 0.0;
                t_w = 0.0;
                d_t_u = 0.0;
                d_t_w = 0.0;
            }
        }

        let target = Vec3::new(f_n, t_u, t_w);
        // Relative tolerance, not absolute. The clamp above is exact in
        // exact arithmetic, so the only slack needed is rounding in
        // `limit / t_norm`, which scales with `|f|`. A fixed `1e-9` is
        // fine at impulses of order one and spuriously fires at order 1e6
        // — turning a numerical problem into an aborted process in any
        // debug build, which is the least useful moment to lose the state.
        debug_assert!(
            crate::cone::in_cone(target, row.mu, 1e-9 * (1.0 + target.norm())),
            "staged solve must land in the friction cone"
        );
        let next = f[c] + (target - f[c]) * config.relaxation;
        max_move = max_move.max((next - f[c]).norm());
        f[c] = next;
        if let Some(d) = diff.as_mut() {
            let d_target = Vec3::new(d_f_n, d_t_u, d_t_w);
            d.df[c] = dfc + (d_target - dfc) * config.relaxation;
        }
    }

    max_move
}



/// One active-set Newton step: classify the current iterate, then solve the
/// resulting linear KKT system exactly.
///
/// The regime is read with [`crate::gradient::classify_impulses`] and the
/// matrix built with [`crate::gradient::kkt_matrix`], so the system solved here
/// is the same one the sensitivity differentiates. The right-hand side is the
/// constant part of the same equations:
///
/// - separating: `f_c = 0`;
/// - sticking: `(A + R) f + b = bias` on the normal row, `= 0` on the
///   tangential rows;
/// - sliding: the normal row as above, tangential rows the homogeneous
///   `f_t - mu t_hat f_n = 0`.
///
/// Returns `None` if the KKT matrix is singular at this active set — the
/// caller falls back to PGS, which needs no such assumption.
///
/// # Differentiated
///
/// With `diff`, both halves are differentiated at the branches *this call*
/// takes:
///
/// - `d raw = K^-1 (drhs - dK raw)`, with `drhs` from `dc` and `dK` from
///   `d_apr` on the stationarity rows. The sliding pin rows carry the slip
///   direction's own rotation, `d t_hat = (I - t_hat t_hat^T) df_t / ||f_t||`,
///   because `t_hat` is read off the iterate and the iterate moves with the
///   parameters. Holding it fixed is the approximation the solver is entitled
///   to (its next sweep re-derives the direction) and a derivative is not.
/// - the clamp, exactly as in [`sweep`].
///
/// The regimes are classified from `f` by the same call the primal makes, so
/// the differentiated system is the system solved — there is no second
/// classification and no band.
// `c` indexes three flat arrays at stride 3 as well as `regimes`; enumerating
// one of them would hide the correspondence the others depend on.
#[allow(clippy::needless_range_loop)]
fn newton_step_diff(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
    f: &[Vec3],
    diff: Option<&SweepDiff>,
) -> Option<(Vec<Vec3>, Vec<Vec3>)> {
    use crate::gradient::ContactRegime;

    let n = problem.n;
    let dim = 3 * n;
    // The classification tolerance is relative to the normal impulse and
    // matches `impulse_sensitivity`'s, so a solve that lands here lands in the
    // regime the gradient will assume.
    let regimes = crate::gradient::classify_impulses(problem, f, 1e-7);
    let mut k = crate::gradient::kkt_matrix(problem, config, &regimes, f);

    let mut rhs = vec![0.0; dim];
    for c in 0..n {
        let base = 3 * c;
        match regimes[c] {
            ContactRegime::Separating => {}
            ContactRegime::Sticking => {
                rhs[base] = problem.rows[c].bias - problem.free_velocity[base];
                rhs[base + 1] = -problem.free_velocity[base + 1];
                rhs[base + 2] = -problem.free_velocity[base + 2];
            }
            ContactRegime::Sliding => {
                rhs[base] = problem.rows[c].bias - problem.free_velocity[base];
            }
        }
    }

    // `K` is consumed by the elimination, so keep a copy for `dK raw` and
    // solve the differential in the same factorization pass: appending `drhs`
    // as a second column is what keeps the two systems provably identical.
    let k_nominal = if diff.is_some() { k.clone() } else { Vec::new() };
    crate::gradient::solve_dense(&mut k, &mut rhs, dim, 1)?;
    if rhs.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // `d raw`, solved against the same `K` the primal just used.
    let d_raw = match diff {
        None => vec![0.0; dim],
        Some(d) => {
            let raw = &rhs;
            let df = &d.df;
            // drhs: the constant part of the same equations.
            let mut drhs = vec![0.0; dim];
            for c in 0..n {
                let base = 3 * c;
                match regimes[c] {
                    ContactRegime::Separating => {}
                    ContactRegime::Sticking => {
                        for r in 0..3 {
                            drhs[base + r] = -d.dc[base + r];
                        }
                    }
                    ContactRegime::Sliding => drhs[base] = -d.dc[base],
                }
            }
            // -dK raw, row by row, mirroring `kkt_matrix`'s row structure.
            for c in 0..n {
                let base = 3 * c;
                match regimes[c] {
                    // Identity rows: dK = 0.
                    ContactRegime::Separating => {}
                    ContactRegime::Sticking => {
                        for r in 0..3 {
                            let ri = (base + r) * dim;
                            let mut acc = 0.0;
                            for col in 0..dim {
                                acc += d.d_apr[ri + col] * raw[col];
                            }
                            drhs[base + r] -= acc;
                        }
                    }
                    ContactRegime::Sliding => {
                        let ri = base * dim;
                        let mut acc = 0.0;
                        for col in 0..dim {
                            acc += d.d_apr[ri + col] * raw[col];
                        }
                        drhs[base] -= acc;
                        // Pin rows: `df_t - mu t_hat df_n = 0` with `t_hat`
                        // read off `f`, so `dK` here is `-mu d(t_hat)`.
                        let fc = f[c];
                        let ft = (fc.y * fc.y + fc.z * fc.z).sqrt();
                        if ft > 1e-14 {
                            let that = [fc.y / ft, fc.z / ft];
                            let dft = [df[c].y, df[c].z];
                            let dot = that[0] * dft[0] + that[1] * dft[1];
                            let mu = problem.rows[c].mu;
                            for i in 0..2 {
                                let dthat = (dft[i] - that[i] * dot) / ft;
                                drhs[base + 1 + i] -= -mu * dthat * raw[base];
                            }
                        }
                    }
                }
            }
            let mut kk = k_nominal;
            let mut sol = drhs;
            crate::gradient::solve_dense(&mut kk, &mut sol, dim, 1)?;
            if sol.iter().any(|v| !v.is_finite()) {
                return None;
            }
            sol
        }
    };

    // Project back into the cone with the *staged* clamp, not a Euclidean cone
    // projection: an infeasible proposal must not be allowed to inflate a
    // normal impulse (see the long note in `sweep`).
    let mut cand = Vec::with_capacity(n);
    let mut dcand = Vec::with_capacity(n);
    for c in 0..n {
        let base = 3 * c;
        let f_n = rhs[base].max(0.0);
        let d_f_n = if rhs[base] > 0.0 { d_raw[base] } else { 0.0 };
        let (mut t_u, mut t_w) = (rhs[base + 1], rhs[base + 2]);
        let (mut d_t_u, mut d_t_w) = (d_raw[base + 1], d_raw[base + 2]);
        let limit = problem.rows[c].mu * f_n;
        let t_norm = (t_u * t_u + t_w * t_w).sqrt();
        if t_norm > limit {
            let s = if t_norm > 0.0 { limit / t_norm } else { 0.0 };
            if t_norm > 0.0 {
                let d_t_norm = (t_u * d_t_u + t_w * d_t_w) / t_norm;
                let d_s = (problem.rows[c].mu * d_f_n - s * d_t_norm) / t_norm;
                let (nu, nw) = (s * d_t_u + t_u * d_s, s * d_t_w + t_w * d_s);
                d_t_u = nu;
                d_t_w = nw;
            } else {
                d_t_u = 0.0;
                d_t_w = 0.0;
            }
            t_u *= s;
            t_w *= s;
        }
        cand.push(Vec3::new(f_n, t_u, t_w));
        dcand.push(Vec3::new(d_f_n, d_t_u, d_t_w));
    }
    Some((cand, dcand))
}

/// Convenience: build a single-contact problem for a point mass of effective
/// mass `m` against an immovable surface.
pub fn point_mass_problem(
    mass: f64,
    free_vel: Vec3,
    material: &ContactMaterial,
    depth: f64,
    dt: f64,
    restitution_threshold: f64,
) -> ContactProblem {
    let inv_m = 1.0 / mass;
    let mut delassus = vec![0.0; 9];
    for i in 0..3 {
        delassus[i * 3 + i] = inv_m;
    }
    let e = ContactProblem::effective_restitution(
        material.restitution,
        free_vel.x.min(0.0).abs(),
        restitution_threshold,
    );
    ContactProblem {
        n: 1,
        delassus,
        // Restitution: the target normal velocity is `-e * v_approach`
        // rather than 0, folded into b.
        free_velocity: vec![free_vel.x * (1.0 + e), free_vel.y, free_vel.z],
        rows: vec![ContactRow::from_material(material, depth, dt, e)],
        // A single point mass against the static world: nothing to couple to.
        bodies: vec![(0, usize::MAX)],
    }
}

/// Is the solver-level adjoint enabled?
///
/// Default off: unset, nothing in this crate calls
/// [`solve_contacts_warm_diff`] with a differential and every number the crate
/// reports is byte-identical to what shipped. `PHYZ_SOLVER_ADJOINT=1` switches
/// [`crate::gradient`]'s consumers over to differentiating the algorithm
/// instead of an assumed fixed point.
///
/// It is a knob rather than the default because the two answer different
/// questions. The IFT gradient is the derivative of *the contact model*, and is
/// what a caller wants when the solve converges — it is independent of the
/// solver's schedule, so it does not move when the tolerance or the iteration
/// cap is retuned. The solver-level gradient is the derivative of *this
/// solver's output*, which is what a caller optimizing through a truncated
/// solve is actually climbing. Where the solve converges they agree; where it
/// does not, only the second is the gradient of the function being evaluated.
pub fn solver_adjoint_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PHYZ_SOLVER_ADJOINT").is_ok_and(|v| v == "1" || v == "true"))
}

/// Differentiate the contact solve by re-executing it alongside a parameter
/// differential — the solver-level adjoint.
///
/// `d_apr` is `d(A + R)` (row-major `3n x 3n`) and `dc` is `d(b - e_n bias)`
/// (length `3n`); `d_initial` is the differential of the warm-start seed, which
/// callers that cannot track it across steps should pass empty (see
/// [`solve_contacts_warm_diff`] for exactly what that costs). Returns `df`, the
/// differential of the impulses the solve produced — the same object
/// [`crate::gradient::FixedPointSensitivity::apply`] returns, and a drop-in for
/// it, except that it is correct at an unconverged iterate.
///
/// The `ContactSolution` comes back too, and callers should check it matches
/// the recorded forward solve: a mismatch means the re-execution diverged from
/// the original, which would void the branch-following guarantee.
pub fn contact_solve_differential(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
    initial: &[Vec3],
    d_initial: &[Vec3],
    d_apr: &[f64],
    dc: &[f64],
) -> (ContactSolution, Vec<Vec3>) {
    let dim = 3 * problem.n;
    debug_assert_eq!(d_apr.len(), dim * dim, "d(A+R) is 3n x 3n");
    debug_assert_eq!(dc.len(), dim, "d(b - e_n bias) is 3n");
    let mut df = vec![Vec3::zeros(); problem.n];
    for (slot, seed) in df.iter_mut().zip(d_initial) {
        *slot = *seed;
    }
    let (sol, out) = solve_contacts_warm_diff(
        problem,
        config,
        initial,
        Some(SweepDiff { d_apr, dc, df }),
        None,
    );
    // Preconditioned mode, converged solve: the recorded iteration is a
    // couple of sweeps from an accelerated seed, so a tangent *riding* it has
    // only that many terms of its Neumann series — the honest tangent at a
    // fixed point is the implicit one. Same map, same linearization
    // ([`crate::gradient::FixedPointSensitivity`]); the ridden tangent is
    // kept for any solve that did not converge, which is exactly the case
    // the ridden form exists for.
    if precond_enabled()
        && sol.converged
        && let Some(s) = crate::gradient::FixedPointSensitivity::at(problem, &sol, config)
    {
        let df = ift_forward(problem, &sol, &s, d_apr, dc);
        return (sol, df);
    }
    (sol, out.unwrap_or_default())
}

/// The IFT tangent at a converged solve, from the raw `d(A+R)`/`d(b - e_n
/// bias)` differentials [`SweepDiff`] carries.
///
/// The two inputs of [`crate::gradient::FixedPointSensitivity::apply`] are
/// assembled here: the stationarity differential at frozen impulses,
/// `d_apr f* + dc`, and per sliding contact the tangential-block
/// differential applied to `t* - f_t`.
#[allow(clippy::needless_range_loop)]
fn ift_forward(
    problem: &ContactProblem,
    sol: &ContactSolution,
    s: &crate::gradient::FixedPointSensitivity,
    d_apr: &[f64],
    dc: &[f64],
) -> Vec<Vec3> {
    let n = problem.n;
    let dim = 3 * n;
    let mut flat = vec![0.0; dim];
    for c in 0..n {
        let base = 3 * c;
        flat[base] = sol.impulses[c].x;
        flat[base + 1] = sol.impulses[c].y;
        flat[base + 2] = sol.impulses[c].z;
    }
    let mut d_stat = vec![0.0; dim];
    for row in 0..dim {
        let mut acc = dc[row];
        for col in 0..dim {
            acc += d_apr[row * dim + col] * flat[col];
        }
        d_stat[row] = acc;
    }
    let mut d_mt = vec![[0.0; 2]; n];
    for c in 0..n {
        if let Some(st) = s.slide_tangent(c) {
            let base = 3 * c;
            for i in 0..2 {
                d_mt[c][i] = d_apr[(base + 1 + i) * dim + base + 1] * st.t_rel[0]
                    + d_apr[(base + 1 + i) * dim + base + 2] * st.t_rel[1];
            }
        }
    }
    s.apply(&d_stat, &d_mt)
}

/// Each contact's own regularized `3x3` block, `A_cc + diag(reg)`.
///
/// Extracted so the forward solve and the reverse pass read the *same* blocks
/// rather than two transcriptions of the same three lines. The transpose is
/// only valid if it linearizes the arithmetic the primal actually performed,
/// and the diagonal block is where most of that arithmetic lives — the normal
/// solve's `a_nn`, the tangential `2x2` and its determinant all come from here.
/// A drifted copy would be a silent, small, everywhere-plausible error, which
/// is the worst kind to have in a gradient.
fn regularized_blocks(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
) -> Vec<[[f64; 3]; 3]> {
    let n = problem.n;
    let dim = 3 * n;
    (0..n)
        .map(|c| {
            let base = 3 * c;
            let reg = regularization_diag(problem, c, config);
            let mut m = [[0.0; 3]; 3];
            for (r, row) in m.iter_mut().enumerate() {
                for (col, e) in row.iter_mut().enumerate() {
                    *e = problem.delassus[(base + r) * dim + base + col];
                    if r == col {
                        *e += reg[r];
                    }
                }
            }
            m
        })
        .collect()
}

/// One entry in the reverse-mode tape: a tangent map the forward pass applied,
/// recorded by the primal state it was linearized at.
///
/// # Why a state tape and not a coefficient tape
///
/// The forward-mode adjoint in this file records nothing at all: it re-executes
/// the solve carrying one tangent, so every branch it takes is the branch the
/// primal took *because it is the same expression*. Reverse mode cannot do
/// that — it has to visit the iterations backwards — so something has to be
/// stored. The cheapest honest thing to store is the primal iterate at the
/// entry of each step, because from it every branch and every coefficient can
/// be recomputed by re-running the primal arithmetic, and recomputing them is
/// exactly as branch-faithful as the forward mode's re-execution is.
///
/// Storing the *coefficients* instead — the `s`, the `t_hat`, the clamp flags —
/// would be a few times larger and would introduce a second place where the
/// staged update's branch structure is written down. This crate has shipped
/// that mismatch twice already (see [`SweepDiff`]), so the tape deliberately
/// holds the one thing that cannot disagree with itself.
///
/// # Size, measured
///
/// One `Vec3` per contact per recorded step: `24n` bytes. The stances this
/// solver is built for run 112–4000 sweeps at 15 contacts, so the tape is
/// `40 KB` at the low end and `1.4 MB` at the pathological high end — a
/// one-off allocation next to the `O(n^2)` sweep kernel and the `O(n^3)`
/// Newton factorizations it sits beside. Nothing here justifies a
/// checkpointing scheme.
///
/// Only steps that *survived* are taped. A Newton proposal rejected by the
/// line search leaves both `f` and `df` untouched, so it is not a link in the
/// tangent chain and must not appear.
pub(crate) enum TapeStep {
    /// One projected Gauss-Seidel sweep, recorded by the impulses it started
    /// from. `normals_only` distinguishes the normal-equilibration warm-up
    /// (phase 1a), whose tangential rows are a pass-through, from the full
    /// staged sweep.
    Sweep {
        entry: Vec<Vec3>,
        normals_only: bool,
    },
    /// One *accepted* Newton proposal and the line-search step length that was
    /// taken with it, recorded by the impulses the KKT system was built at.
    /// The sweep that verified the proposal is a separate `Sweep` entry
    /// immediately after this one.
    Newton { f: Vec<Vec3>, alpha: f64 },
}

/// The reverse-mode counterpart of [`contact_solve_differential`]'s output: one
/// covector per parameter channel.
///
/// The contract is the dot-product identity, and it is worth stating precisely
/// because it is also the acceptance test:
///
/// ```text
/// <bar_f, contact_solve_differential(.., d_initial, d_apr, dc).1>
///     == <bar_apr, d_apr> + <bar_c, dc> + <bar_initial, d_initial>
/// ```
///
/// for *every* `(d_initial, d_apr, dc)`. Forward mode is a perfect oracle for
/// this: it is the same linear map, so any disagreement beyond rounding is a
/// bug in one of the two, and the identity holds at an unconverged iterate for
/// exactly the same reason the forward mode does — both differentiate the
/// finite composition of sweeps that ran, not a fixed point nobody reached.
///
/// # Why this exists when forward mode already does
///
/// Cost. The forward differential carries one tangent per call, so a caller who
/// wants the sensitivity of a scalar loss to all of `A + R` pays `9n^2` solves.
/// The transpose pays one, for the same information. On the fifteen-contact
/// skate stance that is the difference between `2025` re-executions of a
/// four-thousand-sweep solve and one.
pub struct TransposedDifferential {
    /// `dL/d(A + R)`, row-major `3n x 3n` — the covector paired with `d_apr`.
    pub bar_apr: Vec<f64>,
    /// `dL/d(b - e_n bias)`, length `3n` — the covector paired with `dc`.
    pub bar_c: Vec<f64>,
    /// `dL/d(initial)`, length `n`. This falls out for free: the reverse walk
    /// ends holding the adjoint of whatever the tangent chain started from, and
    /// what it started from is the warm-start seed. Under a warm start that is
    /// the previous step's impulses, so this is the channel that carries a
    /// gradient across a whole trajectory rather than one step.
    pub bar_initial: Vec<Vec3>,
}

/// Reverse-mode (transposed) solver-level adjoint: propagate one covector over
/// the solve's *output* impulses back to covectors over its parameters.
///
/// Same solve, same branches, same truncation as
/// [`contact_solve_differential`] — this is that function's linear map applied
/// on the other side, not a second model of the solver. It runs the primal
/// once (taping the iterate at each surviving step), then walks the tape
/// backwards applying `P_k^T` to the running covector and accumulating
/// `Q_k^T` into the parameter covectors.
///
/// # The one structural thing to know about the reverse pass
///
/// Gauss-Seidel is sequential and in place: sweeping contacts `0..n`, contact
/// `c` reads the *already updated* `f[k]` for `k < c` and the *stale* `f[k]`
/// for `k > c`. The forward tangent inherits that ordering exactly. Its
/// transpose therefore has to run contacts in reverse, `n-1..0`, and — this is
/// the part that is easy to get wrong — contact `c`'s residual reads `df[k]`
/// for every `k != c`, so its transpose *scatters* the covector from `c` back
/// onto all of them. It is not a per-contact diagonal map in either direction.
///
/// The primal state each contact was linearized at is reconstructed rather than
/// stored: each contact is updated exactly once per sweep, so its pre-update
/// value *is* the sweep's entry value. Restoring `state[c] = entry[c]` on the
/// way down the reverse walk leaves the array holding post-sweep values for
/// `k < c` and entry values for `k >= c`, which is precisely the state contact
/// `c` saw.
///
/// `bar_f` may be shorter than `problem.n`; missing entries are zero.
///
/// # What is *not* differentiated, and why that is right
///
/// Every discrete choice the solve made — the iteration count, the line-search
/// `alpha`, which contacts clamped, the active-set classification, the
/// stagnation exit — enters as a recorded constant. These are piecewise
/// constant functions of the parameters, so their derivative is zero almost
/// everywhere, and following the branch the forward pass took is the entire
/// contract of a solver-level adjoint. The measure-zero set where a branch
/// flips is exactly where the solver's output is non-differentiable, and no
/// amount of smoothing here would change that.
// Stride arithmetic into flat, row-major arrays (base = 3*c), as
// throughout this module.
#[allow(clippy::needless_range_loop)]
pub fn contact_solve_differential_transpose(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
    initial: &[Vec3],
    bar_f: &[Vec3],
) -> (ContactSolution, TransposedDifferential) {
    let n = problem.n;
    let dim = 3 * n;
    let mut tape: Vec<TapeStep> = Vec::new();
    let (solution, _) = solve_contacts_warm_diff(problem, config, initial, None, Some(&mut tape));

    let mut out = TransposedDifferential {
        bar_apr: vec![0.0; dim * dim],
        bar_c: vec![0.0; dim],
        bar_initial: vec![Vec3::zeros(); n],
    };
    if n == 0 {
        return (solution, out);
    }

    // Preconditioned mode, converged solve: transpose of the same implicit
    // tangent [`contact_solve_differential`] returns — see the note there.
    // `bar_initial` stays zero: at a fixed point the answer is
    // seed-independent, which is the forward mode's statement `df/d(seed) = 0`
    // transposed.
    if precond_enabled()
        && solution.converged
        && let Some(s) = crate::gradient::FixedPointSensitivity::at(problem, &solution, config)
    {
        let (bar_stat, bar_mt) = s.apply_transpose(bar_f);
        let mut flat = vec![0.0; dim];
        for c in 0..n {
            let base = 3 * c;
            flat[base] = solution.impulses[c].x;
            flat[base + 1] = solution.impulses[c].y;
            flat[base + 2] = solution.impulses[c].z;
        }
        for row in 0..dim {
            let b = bar_stat[row];
            out.bar_c[row] += b;
            if b != 0.0 {
                for col in 0..dim {
                    out.bar_apr[row * dim + col] += b * flat[col];
                }
            }
        }
        for c in 0..n {
            if let Some(st) = s.slide_tangent(c) {
                let base = 3 * c;
                for i in 0..2 {
                    for j in 0..2 {
                        out.bar_apr[(base + 1 + i) * dim + base + 1 + j] +=
                            bar_mt[c][i] * st.t_rel[j];
                    }
                }
            }
        }
        return (solution, out);
    }

    let blocks = regularized_blocks(problem, config);
    let mut bar = vec![Vec3::zeros(); n];
    for (slot, seed) in bar.iter_mut().zip(bar_f) {
        *slot = *seed;
    }

    for step in tape.iter().rev() {
        match step {
            TapeStep::Sweep {
                entry,
                normals_only,
            } => sweep_transpose(
                problem,
                config,
                &blocks,
                entry,
                *normals_only,
                &mut bar,
                &mut out,
            ),
            TapeStep::Newton { f, alpha } => {
                newton_transpose(problem, config, f, *alpha, &mut bar, &mut out)
            }
        }
    }

    out.bar_initial = bar;
    (solution, out)
}

/// Transpose of one [`sweep`]'s tangent map.
///
/// The forward per-contact chain, in the order [`sweep`] performs it, is:
///
/// 1. `dr = dc_c + sum_{k != c} (dA_ck f_k + A_ck df_k)` — the off-block residual;
/// 2. `db = dA_cc` — the contact's own regularized block;
/// 3. `d f_n = (-dr_0 - dA_nu f_u - A_nu df_u - dA_nw f_w - A_nw df_w
///    - unclamped * dA_nn) / A_nn`, or zero on the `max(0, .)` branch;
/// 4. `d r_t = dr_t + dA_tn f_n + A_tn d f_n` — the load the normal puts on the
///    tangent rows;
/// 5. the `2x2` tangential solve, quotient rule and all, or a pass-through of
///    `df_c`'s tangential part under `normals_only`;
/// 6. the isotropic disc clamp, `s (I - t_hat t_hat^T) dt + mu t_hat d f_n`;
/// 7. relaxation, `df_c <- (1 - w) df_c + w d target`.
///
/// Each step is transposed in place below, walked 7 down to 1. Steps 4 and 5
/// are transposed only inside the branch that consumed them: `d r_t` is
/// *computed* unconditionally in the forward pass but only *read* by the
/// non-degenerate `!normals_only` branch, so accumulating its adjoint outside
/// that branch would invent a dependency the forward map does not have.
///
/// The primal scalars are recomputed here rather than taped. That is a
/// transcription of [`sweep`]'s arithmetic and the one real duplication in this
/// file; it is deliberate, because the alternative — taping `f_n`, `s`,
/// `t_norm`, `det` and the three branch bits per contact per sweep — is both
/// larger and a second, independently-drifting statement of the staged update's
/// branch structure.
// Stride arithmetic into flat row-major arrays, exactly as in `sweep`.
#[allow(clippy::needless_range_loop)]
fn sweep_transpose(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
    blocks: &[[[f64; 3]; 3]],
    entry: &[Vec3],
    normals_only: bool,
    bar: &mut [Vec3],
    out: &mut TransposedDifferential,
) {
    let n = problem.n;
    let dim = 3 * n;
    let a = &problem.delassus;
    let at = |i: usize, j: usize| a[i * dim + j];
    let w = config.relaxation;

    // Run the primal sweep once to get the post-sweep impulses. Combined with
    // the restore below, this reconstructs the exact state every contact was
    // linearized at, at the cost of one extra sweep per taped step.
    let mut state = entry.to_vec();
    sweep(problem, config, blocks, &mut state, normals_only, None);

    // Whether contact `k` contributes to contact `c`'s residual. Kept as one
    // predicate so the forward and reverse passes cannot mask differently — a
    // coupling mismatch would look like a plausible-but-wrong gradient rather
    // than a failure.
    let couples = |c: usize, k: usize| {
        k != c
            && config.coupling != ContactCoupling::BlockDiagonal
            && !(config.coupling == ContactCoupling::PerBody
                && problem.bodies.len() == n
                && !shares_body(problem.bodies[c], problem.bodies[k]))
    };

    for c in (0..n).rev() {
        // Undo contact `c`'s update: `state` now holds post-sweep values for
        // `k < c` and entry values for `k >= c`, which is what contact `c` saw.
        state[c] = entry[c];
        let base = 3 * c;
        let row = problem.rows[c];
        let fc = state[c];

        // ------------------------------------------------------------ primal
        let mut r = [0.0f64; 3];
        for r_row in 0..3 {
            let mut acc = problem.free_velocity[base + r_row];
            for k in 0..n {
                if !couples(c, k) {
                    continue;
                }
                let kb = 3 * k;
                let f_k = state[k];
                acc += at(base + r_row, kb) * f_k.x
                    + at(base + r_row, kb + 1) * f_k.y
                    + at(base + r_row, kb + 2) * f_k.z;
            }
            r[r_row] = acc;
        }
        let a_nn = blocks[c][0][0];
        let (a_nu, a_nw) = (blocks[c][0][1], blocks[c][0][2]);
        let unclamped = if a_nn > 0.0 {
            (row.bias - r[0] - a_nu * fc.y - a_nw * fc.z) / a_nn
        } else {
            0.0
        };
        let f_n = if a_nn > 0.0 { unclamped.max(0.0) } else { 0.0 };
        let (a_un, a_wn) = (blocks[c][1][0], blocks[c][2][0]);
        let r_u = r[1] + a_un * f_n;
        let r_w = r[2] + a_wn * f_n;
        let (m00, m01) = (blocks[c][1][1], blocks[c][1][2]);
        let (m10, m11) = (blocks[c][2][1], blocks[c][2][2]);
        let det = m00 * m11 - m01 * m10;
        let solvable = det.abs() > 1e-18;
        // Pre-clamp tangential impulses: the derivative in step 6 is taken at
        // these, not at the scaled ones.
        let (t_u, t_w) = if !normals_only {
            if solvable {
                (
                    -(m11 * r_u - m01 * r_w) / det,
                    -(m00 * r_w - m10 * r_u) / det,
                )
            } else {
                (0.0, 0.0)
            }
        } else {
            (fc.y, fc.z)
        };
        let limit = row.mu * f_n;
        let t_norm = (t_u * t_u + t_w * t_w).sqrt();
        let clamped = t_norm > limit;

        // ----------------------------------------------------------- reverse
        let g = bar[c];
        // Step 7. `w` is a config constant, so relaxation is a plain convex
        // combination and its transpose is the same combination.
        let mut bar_dfc = g * (1.0 - w);
        let mut a_fn = w * g.x;
        let (mut a_tu, mut a_tw) = (w * g.y, w * g.z);

        // Step 6, the disc clamp. `t_norm == 0` with `limit == 0` is the
        // pinned-solid case the forward zeroes outright; its transpose is the
        // zero map, not a projector with a `0/0` in it.
        if clamped {
            if t_norm > 0.0 {
                let s = limit / t_norm;
                let bar_ds = t_u * a_tu + t_w * a_tw;
                let (mut p_u, mut p_w) = (s * a_tu, s * a_tw);
                a_fn += row.mu / t_norm * bar_ds;
                let bar_dtnorm = -s / t_norm * bar_ds;
                p_u += t_u / t_norm * bar_dtnorm;
                p_w += t_w / t_norm * bar_dtnorm;
                a_tu = p_u;
                a_tw = p_w;
            } else {
                a_tu = 0.0;
                a_tw = 0.0;
            }
        }

        // `db`'s adjoint, accumulated across steps 5, 4 and 3 before being
        // scattered into `bar_apr`'s diagonal block once.
        let mut bar_db = [[0.0f64; 3]; 3];
        let mut bar_dr = [0.0f64; 3];

        // Step 5, the tangential solve.
        if normals_only {
            // Held, not solved: the tangential differential passes straight
            // through from the incoming `df_c`, so its adjoint does too.
            bar_dfc.y += a_tu;
            bar_dfc.z += a_tw;
        } else if solvable {
            let p = a_tu / det;
            let q = a_tw / det;
            let bar_dru = -m11 * p + m10 * q;
            let bar_drw = m01 * p - m00 * q;
            bar_db[2][2] += -r_u * p;
            bar_db[1][2] += r_w * p;
            bar_db[1][1] += -r_w * q;
            bar_db[2][1] += r_u * q;
            // `d det` feeds both rows; its own transpose puts four more terms
            // on the block. Dropping it is the classic quotient-rule miss and
            // it is invisible at a fixed point, which is why it has to be
            // tested at a truncated one.
            let bar_ddet = -t_u * p - t_w * q;
            bar_db[1][1] += m11 * bar_ddet;
            bar_db[2][2] += m00 * bar_ddet;
            bar_db[1][2] += -m10 * bar_ddet;
            bar_db[2][1] += -m01 * bar_ddet;

            // Step 4. Only reachable from here: see the note above.
            bar_dr[1] += bar_dru;
            bar_db[1][0] += f_n * bar_dru;
            a_fn += a_un * bar_dru;
            bar_dr[2] += bar_drw;
            bar_db[2][0] += f_n * bar_drw;
            a_fn += a_wn * bar_drw;
        }

        // Step 3, the normal solve. The `else` branch of the forward is the
        // pinned `max(0, .)` (or degenerate `a_nn <= 0`) case: the impulse does
        // not move, so nothing flows back through it at all.
        if a_nn > 0.0 && unclamped > 0.0 {
            let h = a_fn / a_nn;
            bar_db[0][0] += -unclamped * h;
            bar_dr[0] += -h;
            bar_db[0][1] += -fc.y * h;
            bar_db[0][2] += -fc.z * h;
            bar_dfc.y += -a_nu * h;
            bar_dfc.z += -a_nw * h;
        }

        // Step 2.
        for i in 0..3 {
            for j in 0..3 {
                out.bar_apr[(base + i) * dim + base + j] += bar_db[i][j];
            }
        }

        // Step 1, the scatter. This is where the Gauss-Seidel coupling shows
        // up: contact `c`'s residual read every other contact's impulse, so its
        // adjoint lands on every other contact's covector.
        for r_row in 0..3 {
            let br = bar_dr[r_row];
            out.bar_c[base + r_row] += br;
            if br == 0.0 {
                continue;
            }
            let ri = (base + r_row) * dim;
            for k in 0..n {
                if !couples(c, k) {
                    continue;
                }
                let kb = 3 * k;
                let f_k = state[k];
                out.bar_apr[ri + kb] += br * f_k.x;
                out.bar_apr[ri + kb + 1] += br * f_k.y;
                out.bar_apr[ri + kb + 2] += br * f_k.z;
                bar[k].x += at(base + r_row, kb) * br;
                bar[k].y += at(base + r_row, kb + 1) * br;
                bar[k].z += at(base + r_row, kb + 2) * br;
            }
        }

        // The output overwrote the input, so the incoming covector is replaced
        // rather than accumulated. `couples` excludes `k == c`, so the scatter
        // above cannot have touched this slot.
        bar[c] = bar_dfc;
    }
}

/// Transpose of one accepted Newton proposal, line-search interpolation
/// included.
///
/// The forward map, given the primal `f` the system was built at:
///
/// ```text
/// d raw   = K^-1 drhs(d_apr, dc, df)
/// d cand  = clamp'(raw) d raw
/// df_new  = (1 - alpha) df + alpha d cand
/// ```
///
/// so the transpose is that read upwards: split the covector by `alpha`, push
/// the `d cand` half back through the clamp, solve `K^T y = bar_d_raw`, and
/// scatter `y` through `drhs`'s construction.
///
/// # `K^T` is built explicitly
///
/// [`crate::gradient::solve_dense`] is a destructive Gaussian elimination with
/// partial pivoting and no transposed-solve entry point. Adding a flag would
/// mean a second index convention inside a routine that every gradient in this
/// crate goes through; forming `K^T` costs one `9n^2` transposition against an
/// `O(n^3)` factorization that was going to happen anyway. At the thirty-two
/// contact redundant manifold that is `9216` stores against `~10^6` flops.
///
/// # The sliding pin rows
///
/// `K`'s tangential rows for a sliding contact pin `f_t` to `mu t_hat f_n`,
/// with `t_hat` read off the iterate — so `t_hat` *moves* with the parameters
/// and `dK` is non-zero there. The forward carries
/// `d t_hat = (I - t_hat t_hat^T) df_t / ||f_t||`. That projector is symmetric,
/// which is the only reason this transpose is as short as it is: the covector
/// comes back through the same `(I - t_hat t_hat^T)`. Below `||f_t|| = 1e-14`
/// the direction is not defined and the forward drops the term; the transpose
/// drops it on the same test rather than on a band of its own.
// Stride arithmetic into flat row-major arrays, as elsewhere in this file.
#[allow(clippy::needless_range_loop)]
fn newton_transpose(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
    f: &[Vec3],
    alpha: f64,
    bar: &mut [Vec3],
    out: &mut TransposedDifferential,
) {
    use crate::gradient::ContactRegime;

    let n = problem.n;
    let dim = 3 * n;

    // Rebuild exactly what `newton_step_diff` built, at the same tolerance and
    // from the same iterate, so the regimes the transpose assumes are the
    // regimes the primal solved.
    let regimes = crate::gradient::classify_impulses(problem, f, 1e-7);
    let mut k = crate::gradient::kkt_matrix(problem, config, &regimes, f);
    let k_nominal = k.clone();
    let mut rhs = vec![0.0; dim];
    for c in 0..n {
        let base = 3 * c;
        match regimes[c] {
            ContactRegime::Separating => {}
            ContactRegime::Sticking => {
                rhs[base] = problem.rows[c].bias - problem.free_velocity[base];
                rhs[base + 1] = -problem.free_velocity[base + 1];
                rhs[base + 2] = -problem.free_velocity[base + 2];
            }
            ContactRegime::Sliding => {
                rhs[base] = problem.rows[c].bias - problem.free_velocity[base];
            }
        }
    }
    // The forward already succeeded here — this step is on the tape only
    // because its proposal was accepted — so a failure would mean the primal
    // was re-executed differently, and dropping the contribution is strictly
    // better than scattering nonsense into the covectors.
    if crate::gradient::solve_dense(&mut k, &mut rhs, dim, 1).is_none() {
        return;
    }
    let raw = rhs;

    // The line-search interpolation. `alpha` is a recorded constant, so this is
    // a plain convex combination in both directions.
    let mut bar_cand = vec![Vec3::zeros(); n];
    for c in 0..n {
        bar_cand[c] = bar[c] * alpha;
        bar[c] *= 1.0 - alpha;
    }

    // Transpose of the staged clamp applied to the raw Newton iterate. Same
    // three branches as `sweep`'s, at the same zero tolerance — note the
    // normal's `max(0, .)` is tested on `raw[base]`, the *unclamped* value,
    // exactly as the forward tests it.
    let mut bar_draw = vec![0.0; dim];
    for c in 0..n {
        let base = 3 * c;
        let mu = problem.rows[c].mu;
        let f_n = raw[base].max(0.0);
        let (t_u, t_w) = (raw[base + 1], raw[base + 2]);
        let limit = mu * f_n;
        let t_norm = (t_u * t_u + t_w * t_w).sqrt();
        let g = bar_cand[c];
        let mut a_fn = g.x;
        let (mut a_tu, mut a_tw) = (g.y, g.z);
        if t_norm > limit {
            if t_norm > 0.0 {
                let s = limit / t_norm;
                let bar_ds = t_u * a_tu + t_w * a_tw;
                let (mut p_u, mut p_w) = (s * a_tu, s * a_tw);
                a_fn += mu / t_norm * bar_ds;
                let bar_dtnorm = -s / t_norm * bar_ds;
                p_u += t_u / t_norm * bar_dtnorm;
                p_w += t_w / t_norm * bar_dtnorm;
                a_tu = p_u;
                a_tw = p_w;
            } else {
                a_tu = 0.0;
                a_tw = 0.0;
            }
        }
        bar_draw[base] = if raw[base] > 0.0 { a_fn } else { 0.0 };
        bar_draw[base + 1] = a_tu;
        bar_draw[base + 2] = a_tw;
    }

    // `K^T y = bar_d_raw`.
    let mut kt = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            kt[j * dim + i] = k_nominal[i * dim + j];
        }
    }
    let mut y = bar_draw;
    if crate::gradient::solve_dense(&mut kt, &mut y, dim, 1).is_none() {
        return;
    }

    // Scatter `y` through `drhs`'s construction. Separating contacts are
    // identity rows with a zero right-hand side: nothing flows back through
    // them, which is the same structural zero the forward carries.
    for c in 0..n {
        let base = 3 * c;
        match regimes[c] {
            ContactRegime::Separating => {}
            ContactRegime::Sticking => {
                for r in 0..3 {
                    let yb = y[base + r];
                    out.bar_c[base + r] += -yb;
                    if yb == 0.0 {
                        continue;
                    }
                    let ri = (base + r) * dim;
                    for col in 0..dim {
                        out.bar_apr[ri + col] += -yb * raw[col];
                    }
                }
            }
            ContactRegime::Sliding => {
                let yb = y[base];
                out.bar_c[base] += -yb;
                if yb != 0.0 {
                    let ri = base * dim;
                    for col in 0..dim {
                        out.bar_apr[ri + col] += -yb * raw[col];
                    }
                }
                let fc = f[c];
                let ft = (fc.y * fc.y + fc.z * fc.z).sqrt();
                if ft > 1e-14 {
                    let that = [fc.y / ft, fc.z / ft];
                    let coef = problem.rows[c].mu * raw[base] / ft;
                    let v = [y[base + 1] * coef, y[base + 2] * coef];
                    let dot = that[0] * v[0] + that[1] * v[1];
                    bar[c].y += v[0] - that[0] * dot;
                    bar[c].z += v[1] - that[1] * dot;
                }
            }
        }
    }
}

/// Dev-only dissection bench for a dumped stance problem
/// (`PHYZ_PROBLEM_DUMP`). Run with:
/// `PHYZ_STANCE_LAB=/path cargo test -p phyz-contact stance_lab -- --ignored --nocapture`
#[allow(clippy::needless_range_loop)]
#[cfg(test)]
mod stance_lab {
    use super::*;

    fn load(path: &str) -> (ContactProblem, Vec<Vec3>) {
        let text = std::fs::read_to_string(path).expect("dump file");
        let mut n = 0usize;
        let mut delassus = Vec::new();
        let mut free_velocity = Vec::new();
        let mut rows = Vec::new();
        let mut bodies = Vec::new();
        let mut seed = Vec::new();
        for line in text.lines() {
            let mut it = line.split_whitespace();
            match it.next() {
                Some("n") => n = it.next().unwrap().parse().unwrap(),
                Some("delassus") => delassus = it.map(|v| v.parse().unwrap()).collect(),
                Some("free_velocity") => free_velocity = it.map(|v| v.parse().unwrap()).collect(),
                Some("row") => {
                    let v: Vec<f64> = it.map(|x| x.parse().unwrap()).collect();
                    rows.push(ContactRow {
                        mu: v[0],
                        restitution: v[1],
                        depth: v[2],
                        bias: v[3],
                        impedance: v[4],
                        dimpedance_ddepth: v[5],
                    });
                }
                Some("body") => bodies.push((
                    it.next().unwrap().parse().unwrap(),
                    it.next().unwrap().parse().unwrap(),
                )),
                Some("seed") => {
                    let v: Vec<f64> = it.map(|x| x.parse().unwrap()).collect();
                    seed = v.chunks(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect();
                }
                _ => {}
            }
        }
        (
            ContactProblem {
                n,
                delassus,
                free_velocity,
                rows,
                bodies,
            },
            seed,
        )
    }

    #[test]
    #[ignore = "dev bench, needs PHYZ_STANCE_LAB pointing at a dump"]
    fn stance_lab() {
        let path = std::env::var("PHYZ_STANCE_LAB").expect("set PHYZ_STANCE_LAB");
        let (problem, seed) = load(&path);
        let config = ContactSolverConfig::simulation();
        let n = problem.n;
        eprintln!("n = {n}");

        // Shipped solve from the recorded seed.
        let sol = solve_contacts_warm(&problem, &config, &seed);
        eprintln!(
            "shipped: iters={} residual={:.3e} converged={} E={:.12e}",
            sol.iterations,
            sol.residual,
            sol.converged,
            qp_objective(&problem, &config, &sol.impulses)
        );

        // Regimes at the shipped terminus.
        let regimes = crate::gradient::classify_impulses(&problem, &sol.impulses, 1e-7);
        for c in 0..n {
            let f = sol.impulses[c];
            let ft = (f.y * f.y + f.z * f.z).sqrt();
            let lim = problem.rows[c].mu * f.x;
            eprintln!(
                "  c{c:2} {:?} f_n={:.3e} ft/lim={:.6} depth={:.2e}",
                regimes[c],
                f.x,
                if lim > 0.0 { ft / lim } else { -1.0 },
                problem.rows[c].depth
            );
        }

        // One Newton proposal from the terminus, dissected.
        if let Some((cand, _)) = newton_step_diff(&problem, &config, &sol.impulses, None) {
            let e0 = qp_objective(&problem, &config, &sol.impulses);
            let e_cand = qp_objective(&problem, &config, &cand);
            let blocks = regularized_blocks(&problem, &config);
            let mut swept = cand.clone();
            let mv = sweep(&problem, &config, &blocks, &mut swept, false, None);
            let e_swept = qp_objective(&problem, &config, &swept);
            eprintln!(
                "newton: E0={e0:.12e} E(cand)={e_cand:.12e} E(swept)={e_swept:.12e} sweep_move={mv:.3e}"
            );
            let dmax = cand
                .iter()
                .zip(&sol.impulses)
                .map(|(a, b)| (*a - *b).norm())
                .fold(0.0f64, f64::max);
            eprintln!("newton proposal max |df| = {dmax:.3e}");
        } else {
            eprintln!("newton: proposal construction FAILED (singular K)");
        }

        // The accelerator from the seed.
        {
            let blocks = regularized_blocks(&problem, &config);
            let mut f = seed.clone();
            f.resize(n, Vec3::zeros());
            accelerate(&problem, &config, &blocks, &mut f);
            let mv = sweep(&problem, &config, &blocks, &mut f, false, None);
            eprintln!(
                "accelerate(seed): sweep_move={mv:.3e} E={:.12e}",
                qp_objective(&problem, &config, &f)
            );
        }

        // Dissect the raw Newton linear solve at the terminus: rebuild the
        // system exactly as `newton_step_diff` does, solve, and report the
        // pre-clamp solution per contact plus the linear-solve residual.
        {
            use crate::gradient::ContactRegime;
            let dim = 3 * n;
            let f = &sol.impulses;
            let regimes = crate::gradient::classify_impulses(&problem, f, 1e-7);
            let k = crate::gradient::kkt_matrix(&problem, &config, &regimes, f);
            let mut rhs = vec![0.0; dim];
            for c in 0..n {
                let base = 3 * c;
                match regimes[c] {
                    ContactRegime::Separating => {}
                    ContactRegime::Sticking => {
                        rhs[base] = problem.rows[c].bias - problem.free_velocity[base];
                        rhs[base + 1] = -problem.free_velocity[base + 1];
                        rhs[base + 2] = -problem.free_velocity[base + 2];
                    }
                    ContactRegime::Sliding => {
                        rhs[base] = problem.rows[c].bias - problem.free_velocity[base]
                    }
                }
            }
            let mut kk = k.clone();
            let mut x = rhs.clone();
            crate::gradient::solve_dense(&mut kk, &mut x, dim, 1).unwrap();
            // ||K x - rhs||_inf
            let mut linres = 0.0f64;
            for r in 0..dim {
                let mut acc = 0.0;
                for cidx in 0..dim {
                    acc += k[r * dim + cidx] * x[cidx];
                }
                linres = linres.max((acc - rhs[r]).abs());
            }
            eprintln!("raw newton solve: ||Kx - rhs||_inf = {linres:.3e}");
            for c in 0..n {
                let base = 3 * c;
                let fx = f[c];
                let lim = problem.rows[c].mu * x[base];
                let ft = (x[base + 1] * x[base + 1] + x[base + 2] * x[base + 2]).sqrt();
                eprintln!(
                    "  c{c:2} {:?} raw=({:+.4e},{:+.4e},{:+.4e}) cur=({:+.4e},{:+.4e},{:+.4e}) viol_n={} ft-lim={:+.2e}",
                    regimes[c],
                    x[base],
                    x[base + 1],
                    x[base + 2],
                    fx.x,
                    fx.y,
                    fx.z,
                    x[base] < 0.0,
                    ft - lim.max(0.0),
                );
            }
        }

        // Sweep-vs-stationarity dissection at the accelerator's terminus.
        {
            let blocks = regularized_blocks(&problem, &config);
            let mut f = seed.clone();
            f.resize(n, Vec3::zeros());
            accelerate(&problem, &config, &blocks, &mut f);
            let regimes = crate::gradient::classify_impulses(&problem, &f, 1e-7);
            let lin = crate::gradient::complete_kkt(&problem, &config, &f, &regimes);
            let res = staged_residual(&problem, &config, &lin.regimes, &lin.that, &f);
            let rn = res.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            let before = f.clone();
            let mv = sweep(&problem, &config, &blocks, &mut f, false, None);
            eprintln!("terminus: |F|={rn:.3e} then sweep_move={mv:.3e}");
            for c in 0..n {
                let d = (f[c] - before[c]).norm();
                if d > mv * 0.5 {
                    let base = 3 * c;
                    eprintln!(
                        "  c{c} {:?} moved {d:.3e}: ({:+.9e},{:+.9e},{:+.9e}) -> ({:+.9e},{:+.9e},{:+.9e})",
                        lin.regimes[c],
                        before[c].x, before[c].y, before[c].z,
                        f[c].x, f[c].y, f[c].z
                    );
                    eprintln!(
                        "     F rows: {:+.3e} {:+.3e} {:+.3e}  a_nn={:.3e} mu={} depth={:+.3e} bias={:+.3e} imp={:.3e}",
                        res[base], res[base + 1], res[base + 2],
                        blocks[c][0][0],
                        problem.rows[c].mu, problem.rows[c].depth, problem.rows[c].bias,
                        problem.rows[c].impedance
                    );
                }
            }
        }

        // How far can plain sweeps go with an unlimited budget?
        let mut cfg2 = config;
        cfg2.max_iterations = 200_000;
        cfg2.newton = false;
        let sol2 = solve_contacts_warm(&problem, &cfg2, &seed);
        eprintln!(
            "pgs-only 200k: iters={} residual={:.3e} E={:.12e}",
            sol2.iterations,
            sol2.residual,
            qp_objective(&problem, &config, &sol2.impulses)
        );
    }
}
