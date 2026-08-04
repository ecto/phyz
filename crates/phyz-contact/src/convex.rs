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

/// Tuning for the contact solve.
#[derive(Debug, Clone, Copy)]
pub struct ContactSolverConfig {
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
}

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
    let n = problem.n;
    let mut f = vec![Vec3::zeros(); n];
    for (slot, seed) in f.iter_mut().zip(initial) {
        *slot = *seed;
    }
    if n == 0 {
        return ContactSolution {
            impulses: f,
            iterations: 0,
            residual: 0.0,
            converged: true,
        };
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
        if sweep(problem, config, &blocks, &mut f, true) < config.tolerance {
            break;
        }
    }

    let mut residual = f64::INFINITY;
    while iterations < config.max_iterations.min(2 * WARMUP_SWEEPS) {
        iterations += 1;
        residual = sweep(problem, config, &blocks, &mut f, false);
        if residual < config.tolerance {
            return ContactSolution {
                impulses: f,
                iterations,
                residual,
                converged: true,
            };
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
        if newton_solves < NEWTON_ATTEMPTS
            && let Some(candidate) = newton_step(problem, config, &f)
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
                iterations += 1;
                let trial_residual = sweep(problem, config, &blocks, &mut trial, false);
                if trial_residual < residual {
                    residual = trial_residual;
                    f = trial;
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
            residual = sweep(problem, config, &blocks, &mut f, false);
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
            if stalls >= STALL_BLOCKS {
                break;
            }
        } else {
            stalls = 0;
        }
    }

    ContactSolution {
        impulses: f,
        iterations,
        residual,
        converged: residual < config.tolerance,
    }
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
fn sweep(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
    blocks: &[[[f64; 3]; 3]],
    f: &mut [Vec3],
    normals_only: bool,
) -> f64 {
    let n = problem.n;
    let dim = 3 * n;
    let a = &problem.delassus;
    let at = |i: usize, j: usize| a[i * dim + j];
    let mut max_move: f64 = 0.0;

    for c in 0..n {
        let base = 3 * c;

        // r = b_c + sum_{k != c} A_ck f_k  (Gauss-Seidel: uses updated f)
        let mut r = [0.0f64; 3];
        for (row, r_row) in r.iter_mut().enumerate() {
            let mut acc = problem.free_velocity[base + row];
            for k in 0..n {
                if k == c {
                    continue;
                }
                let kb = 3 * k;
                acc += at(base + row, kb) * f[k].x
                    + at(base + row, kb + 1) * f[k].y
                    + at(base + row, kb + 2) * f[k].z;
            }
            *r_row = acc;
        }

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
        let f_n = if a_nn > 0.0 {
            ((row.bias - r[0] - a_nu * f[c].y - a_nw * f[c].z) / a_nn).max(0.0)
        } else {
            0.0
        };

        // Tangential 2x2 solve at the fixed normal impulse — including the
        // velocity that normal impulse itself induces on the tangent rows.
        let (a_un, a_wn) = (blocks[c][1][0], blocks[c][2][0]);
        let r_u = r[1] + a_un * f_n;
        let r_w = r[2] + a_wn * f_n;
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

        // Clamp into the friction disc of radius mu*f_n. The clamp is
        // isotropic, so a block sliding at any heading loses speed
        // identically — the property a pyramidal cone gives up.
        let limit = row.mu * f_n;
        let t_norm = (t_u * t_u + t_w * t_w).sqrt();
        if t_norm > limit {
            if t_norm > 0.0 {
                let s = limit / t_norm;
                t_u *= s;
                t_w *= s;
            } else {
                t_u = 0.0;
                t_w = 0.0;
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
// `c` indexes three flat arrays at stride 3 as well as `regimes`; enumerating
// one of them would hide the correspondence the others depend on.
#[allow(clippy::needless_range_loop)]
fn newton_step(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
    f: &[Vec3],
) -> Option<Vec<Vec3>> {
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

    crate::gradient::solve_dense(&mut k, &mut rhs, dim, 1)?;
    if rhs.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // Project back into the cone with the *staged* clamp, not a Euclidean cone
    // projection: an infeasible proposal must not be allowed to inflate a
    // normal impulse (see the long note in `sweep`).
    Some(
        (0..n)
            .map(|c| {
                let base = 3 * c;
                let f_n = rhs[base].max(0.0);
                let (mut t_u, mut t_w) = (rhs[base + 1], rhs[base + 2]);
                let limit = problem.rows[c].mu * f_n;
                let t_norm = (t_u * t_u + t_w * t_w).sqrt();
                if t_norm > limit {
                    let s = if t_norm > 0.0 { limit / t_norm } else { 0.0 };
                    t_u *= s;
                    t_w *= s;
                }
                Vec3::new(f_n, t_u, t_w)
            })
            .collect(),
    )
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
    }
}
