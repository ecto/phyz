//! Proximal-ADMM constrained forward dynamics.
//!
//! # The problem
//!
//! Constrained forward dynamics over a cut-loop tree is the equality-QP
//!
//! ```text
//! minimize_a   1/2 (a - a_free)^T M (a - a_free)
//! subject to   J a = b
//! ```
//!
//! whose KKT conditions are `M (a - a_free) = J^T mu` and `J a = b`. That is
//! Gauss's principle of least constraint: among all accelerations satisfying
//! the constraint, the physical one is the closest to the unconstrained one in
//! the kinetic-energy metric. `a_free` comes from ABA (so it already carries
//! gravity, Coriolis, actuation and any `qfrc_external`), `M` from CRBA, and
//! `b` from [`crate::assemble`] plus stabilization.
//!
//! # Why not just solve the KKT system
//!
//! Because `J M^-1 J^T` is singular the moment the constraints are redundant,
//! and redundancy is *normal* in mechanisms, not pathological: a planar
//! four-bar closed with a 3-row point constraint has one row that is
//! identically dependent on the others; a Sarrus linkage or a symmetric delta
//! is redundant by construction; and a parallel manipulator held by six 6-row
//! welds is redundant by a lot. A direct factorization of the KKT matrix
//! either fails outright or returns a multiplier of size `1/eps` that then
//! propagates into the accelerations. That failure mode — a confident, huge,
//! wrong number — is exactly what the redundancy test in
//! `tests/redundant.rs` pins down.
//!
//! # The method
//!
//! Proximal ADMM on the augmented Lagrangian, which on an equality constraint
//! is the proximal method of multipliers (the `z`-update of a general ADMM
//! splitting collapses to `z = 0` when the second block is the indicator of
//! `{0}`; the interesting part is the *proximal* regularization on both
//! blocks). Per iteration:
//!
//! ```text
//! a^{k+1}  = (M + sigma I + rho J^T J)^-1 (M a_free + J^T mu^k + rho J^T b + sigma a^k)
//! mu^{k+1} = (mu^k - rho (J a^{k+1} - b)) / (1 + rho * epsilon)
//! ```
//!
//! Three things earn their keep:
//!
//! - **`rho J^T J`** is the augmentation. It makes the primal subproblem see
//!   the constraint, so the iteration converges linearly rather than at the
//!   rate of plain dual ascent.
//! - **`sigma I`** is the primal proximal term. `M` from CRBA is positive
//!   definite for a well-posed model, but a model with a zero-inertia link (a
//!   massless coupler, which is a *reasonable* way to write a four-bar) makes
//!   it only positive semi-definite, and then `M + rho J^T J` can still be
//!   singular in a direction no constraint touches. `sigma > 0` removes that
//!   case unconditionally, at the cost of biasing `a` toward the previous
//!   iterate — a bias that vanishes as the iteration converges, because at the
//!   fixed point `a^{k+1} = a^k`.
//! - **`epsilon`** is the dual proximal term, and it is what makes redundancy
//!   graceful. With `epsilon = 0` and an *inconsistent* redundant set (which
//!   is what numerical drift turns a consistent one into) the multiplier grows
//!   without bound: every iteration adds `-rho r` and `r` never reaches zero.
//!   With `epsilon > 0` the multiplier is capped at `|r| / epsilon` and the
//!   solve returns the minimum-norm-ish multiplier that best satisfies the set
//!   in a least-squares sense. The price is honest and quantifiable: the
//!   converged residual is `O(epsilon * |mu|)` rather than zero, so a stiff
//!   loop under load sits at a small non-zero constraint acceleration. The
//!   crate reports that residual rather than hiding it.
//!
//! The system matrix `M + sigma I + rho J^T J` is symmetric positive definite
//! and **constant across iterations**, so it is factored once (Cholesky) and
//! back-substituted per iteration. That is what makes 50 iterations cheap.
//!
//! # Stabilization: Baumgarte, and why
//!
//! Solving `J a = -Jdot v` holds the constraint at the *acceleration* level
//! only. Position and velocity error then integrate as an undamped double
//! integrator of the truncation error, i.e. drift grows without bound. Two
//! standard fixes: Baumgarte feedback folded into `b`, or a post-step
//! nonlinear projection of `q` back onto the constraint manifold.
//!
//! This crate implements **Baumgarte** ([`Stabilization`]):
//!
//! ```text
//! b = -Jdot v - 2 alpha (J v) - beta^2 c
//! ```
//!
//! Reasons, in order of weight:
//!
//! 1. It keeps the whole step *one* linear solve. A position projection is a
//!    nonlinear least-squares solve on `q` per step, with its own convergence
//!    story to report and its own failure mode on redundant sets — a second
//!    solver's worth of honesty budget.
//! 2. It composes with the existing semi-implicit stepper unchanged; nothing
//!    in `phyz-rigid` has to learn about constraints.
//! 3. It is differentiable in the same sense the rest of the acceleration path
//!    is, which matters if this ever feeds `phyz-diff`. A projection with an
//!    inner iteration count is not.
//!
//! What it costs, stated plainly: Baumgarte does not eliminate drift, it makes
//! the error a damped oscillator with rate `beta` and damping ratio
//! `alpha/beta`, so the error settles to a small non-zero value set by the
//! integrator's truncation error rather than to zero. The measured value for
//! the crate's four-bar over 10 s is in the crate README and in
//! `tests/four_bar.rs`. Gains are not free either: `beta` much above `1/dt`
//! makes the feedback stiffer than the integrator can follow and the loop goes
//! unstable — the defaults are chosen against `dt`, not in the abstract.

use crate::constraint::{LoopConstraintSet, LoopSystem, assemble};
use phyz_math::{DMat, DVec};
use phyz_model::{Model, State};
use phyz_rigid::{aba, crba};
use tang_la::Cholesky;

/// Baumgarte stabilization gains, in the acceleration target
/// `b = -Jdot v - 2 alpha (J v) - beta^2 c`.
#[derive(Debug, Clone, Copy)]
pub struct Stabilization {
    /// Velocity-error gain `alpha` (1/s). Damps `cdot`.
    pub alpha: f64,
    /// Position-error gain `beta` (1/s). Pulls `c` back to zero.
    pub beta: f64,
}

impl Stabilization {
    /// No stabilization: hold the constraint at the acceleration level only.
    ///
    /// Useful to *measure* how bad unstabilized drift is; not a setting to
    /// simulate with.
    pub const NONE: Self = Self {
        alpha: 0.0,
        beta: 0.0,
    };

    /// Critically damped gains (`alpha = beta`) with `beta` scaled to the
    /// timestep: `beta = relaxation_steps^-1 / dt`.
    ///
    /// Critical damping is the right default because an under-damped choice
    /// makes the constraint error *ring*, which reads as a physical vibration
    /// of the mechanism and is much harder to attribute than a slow pull.
    /// `relaxation_steps` is how many timesteps the error takes to decay by
    /// `1/e`; below about 5 the feedback outruns the integrator and the loop
    /// destabilizes at large `dt`.
    pub fn critically_damped(dt: f64, relaxation_steps: f64) -> Self {
        let beta = 1.0 / (relaxation_steps * dt);
        Self { alpha: beta, beta }
    }
}

/// Tuning for [`solve`].
#[derive(Debug, Clone, Copy)]
pub struct LoopSolverConfig {
    /// Augmentation weight `rho`. Larger enforces the constraint harder per
    /// iteration and converges faster on well-conditioned sets, but worsens the
    /// conditioning of the (constant) primal factorization.
    pub rho: f64,
    /// Primal proximal weight `sigma`. Small and positive; its only job is to
    /// keep the primal system SPD when `M` is singular.
    pub sigma: f64,
    /// Dual proximal weight `epsilon`, the redundancy regularizer. Zero gives
    /// exact constraints on a full-rank set and unbounded multipliers on an
    /// inconsistent redundant one. See the module docs.
    ///
    /// The default (`1e-9`) is sized for **full-rank** sets: it puts the
    /// residual floor `epsilon * |mu|` far below [`Self::tolerance`], and its
    /// multiplier cap `|r| / epsilon` correspondingly far above anything
    /// reachable inside [`Self::max_iterations`]. On a knowingly redundant
    /// mechanism, raise it to a real compliance — `1e-6` to `1e-4` — so the cap
    /// binds within the iteration budget. Measured effect on the crate's
    /// inconsistent-redundant fixture: `|mu|_inf` 1.0e3 at `1e-9` against 5.0e1
    /// at `1e-4`, with the accelerations differing by 1.2e-5 rad/s².
    pub epsilon: f64,
    /// Convergence tolerance on `|J a - b|_inf`, in m/s² (rad/s² on weld rows).
    pub tolerance: f64,
    /// Iteration cap. Reaching it means `converged: false`, and the returned
    /// acceleration is the last iterate — feasible to integrate, but not a
    /// solution of the constrained problem.
    pub max_iterations: usize,
    /// Baumgarte gains.
    pub stabilization: Stabilization,
}

impl LoopSolverConfig {
    /// Defaults tuned against a model's own timestep.
    ///
    /// There is no timestep-independent good `beta`, so there is no `Default`
    /// impl — a constructor that silently picks gains for the wrong `dt` is
    /// worse than one the caller has to think about for a second.
    pub fn for_model(model: &Model) -> Self {
        Self {
            rho: 1e3,
            sigma: 1e-8,
            epsilon: 1e-9,
            tolerance: 1e-6,
            max_iterations: 200,
            stabilization: Stabilization::critically_damped(model.dt, 10.0),
        }
    }
}

/// Result of a constrained forward-dynamics solve.
#[derive(Debug, Clone)]
pub struct LoopSolution {
    /// Constrained generalized acceleration, `nv`.
    pub qdd: DVec,
    /// Constraint multipliers `mu`, `m`. The generalized constraint force is
    /// `J^T mu` — that sign, and not the other one, because the KKT stationary
    /// condition here is `M (a - a_free) = J^T mu`.
    pub multipliers: DVec,
    /// Iterations actually performed.
    pub iterations: usize,
    /// Final `|J a - b|_inf`, in m/s² (rad/s² on weld rows).
    pub residual: f64,
    /// Whether [`LoopSolverConfig::tolerance`] was reached. **False means the
    /// solve did not converge** — it is not a "close enough" flag.
    pub converged: bool,
    /// The linearized system that was solved, kept for callers that want the
    /// position residual or the Jacobian without re-assembling.
    pub system: LoopSystem,
}

/// Solve constrained forward dynamics for `constraints` at `state`.
///
/// With an empty constraint set this is ABA and nothing else, reported as
/// converged in zero iterations.
pub fn solve(
    model: &Model,
    state: &State,
    constraints: &LoopConstraintSet,
    cfg: &LoopSolverConfig,
) -> LoopSolution {
    let a_free = aba(model, state);
    let system = assemble(model, state, constraints);
    let m = system.nrows();

    if m == 0 {
        return LoopSolution {
            qdd: a_free,
            multipliers: DVec::zeros(0),
            iterations: 0,
            residual: 0.0,
            converged: true,
            system,
        };
    }

    let mass = crba(model, state);

    // Stabilized acceleration target: b = -Jdot v - 2 alpha cdot - beta^2 c.
    // Every term is A-minus-B oriented, matching `J` (see constraint.rs).
    let st = cfg.stabilization;
    let b = DVec::from_fn(m, |i| {
        -system.bias[i]
            - 2.0 * st.alpha * system.velocity_error[i]
            - st.beta * st.beta * system.position_error[i]
    });

    let out = proximal_admm(&mass, &system.jacobian, &a_free, &b, cfg);
    LoopSolution {
        qdd: out.x,
        multipliers: out.mu,
        iterations: out.iterations,
        residual: out.residual,
        converged: out.converged,
        system,
    }
}

/// Least-squares projection of the current generalized velocity onto the
/// constraint manifold's tangent space: the `v` closest to `state.v` in the
/// mass metric with `J v = 0`.
///
/// Needed because a hand-written initial state is almost never *velocity*
/// consistent. Setting "the crank spins at 4 rad/s" and leaving the other
/// joints at zero describes a mechanism that is being torn apart at 8 m/s, and
/// an acceleration-level constraint cannot undo that — Baumgarte will bleed it
/// off over `1/alpha`, leaving a position error orders of magnitude larger than
/// anything the solver itself contributes. Project once at setup and the
/// measured drift is the solver's, not the initial condition's.
///
/// Uses the same Proximal-ADMM machinery as [`solve`], so it degrades on a
/// redundant set in exactly the same, documented way.
pub fn project_velocity(
    model: &Model,
    state: &State,
    constraints: &LoopConstraintSet,
    cfg: &LoopSolverConfig,
) -> DVec {
    let system = assemble(model, state, constraints);
    let m = system.nrows();
    if m == 0 {
        return state.v.clone();
    }
    let mass = crba(model, state);
    // Target is exact tangency, with no stabilization terms: position error is
    // not a velocity-level quantity and folding it in here would inject a
    // spurious closing velocity.
    let b = DVec::zeros(m);
    proximal_admm(&mass, &system.jacobian, &state.v, &b, cfg).x
}

/// Output of the inner Proximal-ADMM iteration.
struct AdmmOutput {
    x: DVec,
    mu: DVec,
    iterations: usize,
    residual: f64,
    converged: bool,
}

/// `min_x 1/2 (x - x_free)^T M (x - x_free)  s.t.  J x = b`, by Proximal ADMM.
///
/// Shared by [`solve`] (where `x` is an acceleration) and
/// [`project_velocity`] (where it is a velocity). The two are the same QP in
/// the same metric; writing the iteration twice is how a solver and its
/// projection end up disagreeing.
fn proximal_admm(
    mass: &DMat,
    j: &DMat,
    x_free: &DVec,
    b: &DVec,
    cfg: &LoopSolverConfig,
) -> AdmmOutput {
    let nv = mass.nrows();
    let m = j.nrows();
    let jt = j.transpose();

    // Constant primal system: K = M + sigma I + rho J^T J, SPD.
    let mut k = mass.clone();
    let jtj = jt.mul_mat(j);
    for r in 0..nv {
        for c in 0..nv {
            let v = k.get(r, c) + cfg.rho * jtj.get(r, c);
            k.set(r, c, v);
        }
        k.set(r, r, k.get(r, r) + cfg.sigma);
    }
    // Symmetrize before factoring. CRBA fills both triangles, but the product
    // above accumulates them in different orders, so the two halves can differ
    // in the last bit — and Cholesky reads only one triangle, which would make
    // the result depend on which one. Averaging costs nothing and removes the
    // question.
    for r in 0..nv {
        for c in (r + 1)..nv {
            let avg = 0.5 * (k.get(r, c) + k.get(c, r));
            k.set(r, c, avg);
            k.set(c, r, avg);
        }
    }

    let chol = match Cholesky::new(&k) {
        Some(c) => c,
        None => {
            // Only reachable if the model has a genuinely indefinite mass
            // matrix (negative inertia parameters). Report failure rather than
            // returning something plausible-looking.
            return AdmmOutput {
                x: x_free.clone(),
                mu: DVec::zeros(m),
                iterations: 0,
                residual: f64::INFINITY,
                converged: false,
            };
        }
    };

    let m_xfree = mass.mul_vec(x_free);
    let rho_jt_b = {
        let t = jt.mul_vec(b);
        DVec::from_fn(nv, |i| cfg.rho * t[i])
    };

    let mut x = x_free.clone();
    let mut mu = DVec::zeros(m);
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    let mut converged = false;
    let scale = 1.0 / (1.0 + cfg.rho * cfg.epsilon);

    for it in 1..=cfg.max_iterations {
        iterations = it;

        // Primal: K x = M x_free + J^T mu + rho J^T b + sigma x^k
        let jt_mu = jt.mul_vec(&mu);
        let rhs = DVec::from_fn(nv, |i| {
            m_xfree[i] + jt_mu[i] + rho_jt_b[i] + cfg.sigma * x[i]
        });
        x = chol.solve(&rhs);

        // Residual, then the proximal dual update.
        let jx = j.mul_vec(&x);
        let mut worst = 0.0_f64;
        for i in 0..m {
            let r = jx[i] - b[i];
            if r.abs() > worst {
                worst = r.abs();
            }
            mu[i] = (mu[i] - cfg.rho * r) * scale;
        }
        residual = worst;

        if residual <= cfg.tolerance {
            converged = true;
            break;
        }
        if !residual.is_finite() {
            break;
        }
    }

    AdmmOutput {
        x,
        mu,
        iterations,
        residual,
        converged,
    }
}

/// One constrained timestep: solve, then semi-implicit Euler.
///
/// The integrator is `phyz_rigid::semi_implicit_euler`, unmodified — the whole
/// point of an acceleration-level constraint is that the stepper does not need
/// to know constraints exist.
pub fn step(
    model: &Model,
    state: &mut State,
    constraints: &LoopConstraintSet,
    cfg: &LoopSolverConfig,
) -> LoopSolution {
    let sol = solve(model, state, constraints, cfg);
    phyz_rigid::semi_implicit_euler(model, state, sol.qdd.as_slice(), model.dt);
    sol
}

/// Numerical rank of a constraint Jacobian, by singular values relative to the
/// largest one.
///
/// `rank < m` means the set is redundant: some rows say nothing the others do
/// not. That is the input to [`crate::mobility()`] and the thing the redundancy
/// test measures.
pub fn constraint_rank(j: &DMat, rel_tol: f64) -> usize {
    if j.nrows() == 0 || j.ncols() == 0 {
        return 0;
    }
    let svd = tang_la::Svd::new(j);
    let smax = svd.s.iter().fold(0.0_f64, |a, &x| a.max(x));
    if smax == 0.0 {
        return 0;
    }
    svd.s.iter().filter(|&&x| x > rel_tol * smax).count()
}
