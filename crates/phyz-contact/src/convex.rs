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
    solve_contacts_warm_diff(problem, config, initial, None).0
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

    let a = &problem.delassus;
    let at = |i: usize, j: usize| a[i * dim + j];

    // Per-contact 3x3 diagonal block, regularized and inverted once.
    let blocks: Vec<[[f64; 3]; 3]> = (0..n)
        .map(|c| {
            let base = 3 * c;
            let reg = regularization_diag(problem, c, config);
            let mut m = [[0.0; 3]; 3];
            for (r, row) in m.iter_mut().enumerate() {
                for (col, e) in row.iter_mut().enumerate() {
                    *e = at(base + r, base + col);
                    if r == col {
                        *e += reg[r];
                    }
                }
            }
            m
        })
        .collect();

    let mut iterations = 0;

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
        if sweep(problem, config, &blocks, &mut f, true, diff.as_mut()) < config.tolerance {
            break;
        }
    }

    let mut residual = f64::INFINITY;
    while iterations < config.max_iterations.min(2 * WARMUP_SWEEPS) {
        iterations += 1;
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
    let mut stalls = 0;
    while iterations < config.max_iterations {
        let entry_residual = residual;
        if config.newton
            && newton_solves < NEWTON_ATTEMPTS
            && let Some((candidate, d_candidate)) =
                newton_step_diff(problem, config, &f, diff.as_ref())
        {
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
                    residual = trial_residual;
                    f = trial;
                    if let (Some(d), Some(dt)) = (diff.as_mut(), d_trial) {
                        d.df = dt;
                    }
                    break;
                }
                alpha *= 0.5;
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
        if residual > entry_residual * STALL_RATIO {
            stalls += 1;
            if stalls >= STALL_BLOCKS && !no_stall_exit() {
                break;
            }
        } else {
            stalls = 0;
        }
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
    let (sol, out) =
        solve_contacts_warm_diff(problem, config, initial, Some(SweepDiff { d_apr, dc, df }));
    (sol, out.unwrap_or_default())
}
