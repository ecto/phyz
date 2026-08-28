//! Gradients of the contact solve, by the implicit function theorem.
//!
//! # What is differentiated
//!
//! The converged solution of the convex program, **not** the solver
//! iterations. Given the KKT conditions `F(f*; theta) = 0` at the optimum, the
//! implicit function theorem gives
//!
//! ```text
//! df*/dtheta = -(dF/df)^-1 (dF/dtheta)
//! ```
//!
//! Unrolling the solver instead would make the gradient depend on the
//! iteration count and the initial guess — it would be the derivative of the
//! *algorithm*, not of the physics — and would cost memory linear in the
//! iterations at every step of a trajectory adjoint. See
//! `docs/design/differentiable-contact.md` §2.1.
//!
//! # The active set
//!
//! At the optimum each contact is in one of three regimes, and `dF/df` takes a
//! different form in each:
//!
//! - **Separating** (`f = 0`): the contact carries no impulse, and a small
//!   parameter change does not switch that on. `df/dtheta = 0`.
//! - **Sticking** (`f` strictly inside the cone): the constraint is active as
//!   an equality in all three directions, so `(A + R) df = -db`.
//! - **Sliding** (`f` on the cone boundary): the normal direction is an
//!   equality and the tangential impulse rides the cone boundary along the
//!   slip direction. Both the magnitude (`mu * f_n`) and the **direction**
//!   respond to a perturbation — see [`FixedPointSensitivity`] for the exact
//!   linearization, including the slip-rotation channel.
//!
//! The regime is read off the primal solution and **held fixed** while
//! differentiating. That is what makes the derivative well-defined: at the
//! exact transition between regimes the true derivative does not exist, and no
//! amount of care recovers it.
//!
//! # Honesty about what this returns
//!
//! This is the exact gradient of the **regularized** contact model that was
//! simulated, which is a smooth relaxation of rigid contact. It is *not* the
//! gradient of ideal rigid contact, because ideal rigid contact is not
//! differentiable. Raising `regularization` smooths the dynamics and the
//! gradient together; lowering it sharpens both. At a genuine
//! contact-making/breaking event the gradient is biased, and the bias does not
//! vanish as the regularization shrinks — it grows. Suh et al. (ICML 2022)
//! call this the empirical-bias regime, and it is why
//! [`ContactSolverConfig::gradients`] exists as a separate preset from
//! [`ContactSolverConfig::simulation`].
//!
//! [`ContactSolverConfig::gradients`]: crate::ContactSolverConfig::gradients
//! [`ContactSolverConfig::simulation`]: crate::ContactSolverConfig::simulation

use crate::convex::{ContactProblem, ContactSolution, ContactSolverConfig};
use phyz_math::Vec3;

/// Which branch of the contact law a contact sits on at the solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContactRegime {
    /// No impulse: the surfaces are separating.
    Separating,
    /// Impulse strictly inside the friction cone: no relative sliding.
    Sticking,
    /// Impulse on the cone boundary: sliding at `||f_t|| = mu * f_n`.
    Sliding,
}

/// Classify each contact at the converged solution.
///
/// `tol` is relative to the normal impulse; a contact within `tol` of the cone
/// boundary counts as sliding, since that is the branch the solve took.
pub fn classify(
    problem: &ContactProblem,
    solution: &ContactSolution,
    tol: f64,
) -> Vec<ContactRegime> {
    classify_impulses(problem, &solution.impulses, tol)
}

/// [`classify`] on a bare impulse vector.
///
/// The solver needs this on an *iterate*, before there is a
/// [`ContactSolution`] to wrap it in: [`crate::convex::solve_contacts_warm`]
/// drives an active-set Newton whose linear system is exactly the `K` built
/// below, so it has to read the regime off the current iterate with precisely
/// the rule the gradient will later use. Sharing one function is the point —
/// a solver that converges to a different active set than the gradient
/// linearizes around is the mismatch that has bitten this crate before.
/// The band [`FixedPointSensitivity::at`] classifies with, `1e-7` unless
/// `PHYZ_CLASSIFY_BAND` overrides it.
///
/// It exists to *measure* the boundary channel, not to tune it. The forward
/// sweep branches on the cone exactly — `f_n = max(0, .)` and
/// `if t_norm > limit`, both at zero tolerance — while this backward
/// classification uses a band. Any contact sitting inside that band is
/// linearized on a branch the forward pass need not have taken, and that error
/// is convergence-independent: no amount of extra sweeps removes it.
///
/// Sweeping this knob is what tells the two apart. If the adjoint is unchanged
/// across decades of band, no contact is ambiguously placed and whatever
/// disagreement remains against finite differences is truncation — the finite
/// iteration count, not the boundary. Measured on the K1 13-contact skate
/// stance, that is exactly what happens, which is why the fix that followed is
/// a solver-level replay rather than a softened complementarity function.
///
/// Default-off in the sense that matters: unset, this returns the same `1e-7`
/// the constant always was, so shipped behaviour is byte-identical.
fn classify_band() -> f64 {
    static BAND: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *BAND.get_or_init(|| {
        std::env::var("PHYZ_CLASSIFY_BAND")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|b: &f64| b.is_finite() && *b >= 0.0)
            .unwrap_or(1e-7)
    })
}

pub(crate) fn classify_impulses(
    problem: &ContactProblem,
    impulses: &[Vec3],
    tol: f64,
) -> Vec<ContactRegime> {
    impulses
        .iter()
        .zip(&problem.rows)
        .map(|(f, row)| {
            if f.x <= tol {
                return ContactRegime::Separating;
            }
            let ft = (f.y * f.y + f.z * f.z).sqrt();
            let limit = row.mu * f.x;
            // `limit <= 0` is a frictionless contact carrying load. Its
            // tangential impulse is pinned to zero, which is the *same*
            // algebraic row as the sliding case with a zero slip direction —
            // not a sticking row, which would let `f_t` respond to `db`. A
            // frictionless contact transmits no tangential impulse and no
            // tangential sensitivity.
            if limit <= 0.0 || ft >= limit - tol * f.x.max(1.0) {
                ContactRegime::Sliding
            } else {
                ContactRegime::Sticking
            }
        })
        .collect()
}

/// Exact linearization of the staged fixed-point map at a converged solution.
///
/// # Why this exists alongside `kkt_matrix`
///
/// The staged solve's fixed point pins a **sliding** contact's tangential
/// impulse to the cone boundary along the direction of the *unconstrained*
/// tangential minimizer `t* = -M_t^-1 r_t`:
///
/// ```text
/// f_t = mu * f_n * t_hat,     t_hat = t* / ||t*||
/// ```
///
/// `kkt_matrix` (crate-private, shared with the solver's Newton step)
/// linearizes this holding `t_hat` fixed — the right system for
/// the *solver* (whose iterates re-derive the direction every sweep), but an
/// incomplete derivative of the *map*: under a parameter change the slip
/// direction rotates, `d t_hat = (I - t_hat t_hat^T) dt* / ||t*||`, and that
/// rotation couples every contact's impulse into every sliding contact's
/// tangential rows. On a redundant manifold (a box face on the plane) the
/// dropped channel is a few parts in 1e4 of the total step Jacobian —
/// invisible on the axis-aligned single-contact cases the original validation
/// used, and a systematic bias on anything that tilts.
///
/// This struct carries `-K'^-1` for the **complete** linearization `K'`,
/// plus the per-contact tangential data a caller needs to close the chain
/// rule for parameters that enter through `A` and `b`:
///
/// - sticking rows and sliding normal rows: the stationarity residual
///   `(A + R) f + b - e_n bias`, as before;
/// - sliding tangential rows: `F_t = f_t - mu f_n t_hat(t*)`, whose
///   differential in the parameters is
///   `C * (dRes_t + dM_t (t* - f_t))` with `C = s P M_t^-1`,
///   `s = ||f_t|| / ||t*||`, `P = I - t_hat t_hat^T`, and `dRes_t` the
///   tangential rows of the stationarity differential.
///
/// The solver keeps using `kkt_matrix`; the two agree about the fixed point
/// itself, and disagree only about its derivative — where this one is the
/// exact one.
pub struct FixedPointSensitivity {
    n: usize,
    /// `-K'^-1`, row-major `3n x 3n`: maps a residual differential to `df`.
    inv: Vec<f64>,
    /// Per contact: sliding tangential data, `None` unless the contact is
    /// sliding with a well-defined slip direction.
    slide: Vec<Option<SlideTangent>>,
    /// The regime each contact was linearized in.
    regimes: Vec<ContactRegime>,
}

/// Tangential linearization data of one sliding contact.
#[derive(Debug, Clone, Copy)]
pub struct SlideTangent {
    /// `C = s P M_t^-1` (2x2, rows/cols in `[u, w]`).
    pub map: [[f64; 2]; 2],
    /// `t* - f_t` (in `[u, w]`) — the vector `dM_t` acts on in the parameter
    /// channel.
    pub t_rel: [f64; 2],
}

/// The complete linearization `K'` of the staged fixed-point map at an
/// iterate, with the per-contact tangential data that goes with it.
///
/// Extracted from [`FixedPointSensitivity::at`] so the forward solver's
/// preconditioned accelerator can Newton-iterate on the *same* linearization
/// the gradient assumes — solving and differentiating the same object is the
/// invariant that keeps the two from drifting apart. Unlike `at`, this does
/// not require a converged solution: the linearization is well-defined at any
/// iterate, it is simply only *the derivative* at a fixed point.
pub(crate) struct CompleteKkt {
    /// `K'`, row-major `3n x 3n`.
    pub k: Vec<f64>,
    /// Per-contact sliding tangential data, as in [`FixedPointSensitivity`].
    pub slide: Vec<Option<SlideTangent>>,
    /// The regime each contact was linearized in.
    pub regimes: Vec<ContactRegime>,
    /// Per sliding contact, the slip direction `t_hat` of the unconstrained
    /// tangential minimizer at the iterate (`None` where undefined).
    pub that: Vec<Option<[f64; 2]>>,
}

/// Assemble [`CompleteKkt`] at an arbitrary impulse iterate.
// The loop indices drive stride arithmetic into flat, row-major arrays
// (base = 3*c), matching the rest of this module.
#[allow(clippy::needless_range_loop)]
pub(crate) fn complete_kkt(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
    impulses: &[Vec3],
    regimes: &[ContactRegime],
) -> CompleteKkt {
    let n = problem.n;
    let dim = 3 * n;
    let mut that_dirs: Vec<Option<[f64; 2]>> = vec![None; n];
            // Start from the solver's pinned-direction system and correct the
        // sliding tangential rows.
        let mut k = kkt_matrix(problem, config, regimes, impulses);
        let a = &problem.delassus;
        let mut slide: Vec<Option<SlideTangent>> = vec![None; n];

        for c in 0..n {
            if regimes[c] != ContactRegime::Sliding {
                continue;
            }
            let base = 3 * c;
            let f = impulses[c];
            let ft = (f.y * f.y + f.z * f.z).sqrt();
            if ft <= 1e-14 {
                // Frictionless-but-loaded (or exactly zero slip): the pin rows
                // already say `df_t = 0`, and there is no direction to rotate.
                continue;
            }
            let reg = crate::convex::regularization_diag(problem, c, config);
            // M_t: the contact's own regularized tangential 2x2 block.
            let m = [
                [
                    a[(base + 1) * dim + base + 1] + reg[1],
                    a[(base + 1) * dim + base + 2],
                ],
                [
                    a[(base + 2) * dim + base + 1],
                    a[(base + 2) * dim + base + 2] + reg[2],
                ],
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
            let flat = flat_impulses(impulses);
            let mut r = [0.0f64; 2];
            for (i, ri) in r.iter_mut().enumerate() {
                let row = base + 1 + i;
                let mut acc = problem.free_velocity[row];
                for (col, fc) in flat.iter().enumerate() {
                    acc += a[row * dim + col] * fc;
                }
                acc += reg[1 + i] * flat[row];
                acc -= m[i][0] * f.y + m[i][1] * f.z;
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
            let that = [t_star[0] / t_norm, t_star[1] / t_norm];
            // s = ||f_t|| / ||t*||, clamped: at the clamp boundary the two
            // coincide and s = 1; s > 1 would mean the contact was not
            // actually clamped, i.e. a borderline-sticking classification.
            let s = (ft / t_norm).min(1.0);
            let p = [
                [1.0 - that[0] * that[0], -that[0] * that[1]],
                [-that[1] * that[0], 1.0 - that[1] * that[1]],
            ];
            let mut cmap = [[0.0f64; 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    cmap[i][j] = s * (p[i][0] * minv[0][j] + p[i][1] * minv[1][j]);
                }
            }

            // Rewrite the two tangential rows of K:
            //   df_t - mu t_hat df_n + C (sum_{cols != own t} A_{t,col} df_col) = -dF_theta
            let mu = problem.rows[c].mu;
            for i in 0..2 {
                let row = base + 1 + i;
                for col in 0..dim {
                    let in_own_t = col == base + 1 || col == base + 2;
                    k[row * dim + col] = if in_own_t {
                        if col == row { 1.0 } else { 0.0 }
                    } else {
                        cmap[i][0] * a[(base + 1) * dim + col]
                            + cmap[i][1] * a[(base + 2) * dim + col]
                    };
                }
                k[row * dim + base] -= mu * that[i];
            }
            slide[c] = Some(SlideTangent {
                map: cmap,
                t_rel: [t_star[0] - f.y, t_star[1] - f.z],
            });
            that_dirs[c] = Some(that);
        }


    let _ = dim;
    CompleteKkt {
        k,
        slide,
        regimes: regimes.to_vec(),
        that: that_dirs,
    }
}

impl FixedPointSensitivity {
    /// Build the exact map linearization at a converged solution.
    ///
    /// Returns `None` when the solve did not converge (not a fixed point) or
    /// the linearized system is singular (the derivative genuinely does not
    /// exist at a degenerate active set).
    pub fn at(
        problem: &ContactProblem,
        solution: &ContactSolution,
        config: &ContactSolverConfig,
    ) -> Option<Self> {
        if !solution.converged {
            return None;
        }
        let n = problem.n;
        let dim = 3 * n;
        if n == 0 {
            return Some(Self {
                n,
                inv: Vec::new(),
                slide: Vec::new(),
                regimes: Vec::new(),
            });
        }
        let regimes = classify(problem, solution, classify_band());
        let lin = complete_kkt(problem, config, &solution.impulses, &regimes);
        let (mut k, slide, regimes) = (lin.k, lin.slide, lin.regimes);
        // inv = -K'^-1: solve K X = -I.
        let mut rhs = vec![0.0; dim * dim];
        for i in 0..dim {
            rhs[i * dim + i] = -1.0;
        }
        solve_dense(&mut k, &mut rhs, dim, dim)?;
        Some(Self {
            n,
            inv: rhs,
            slide,
            regimes,
        })
    }

    /// The tangential data of contact `c`, if it is sliding with a defined
    /// slip direction.
    pub fn slide_tangent(&self, c: usize) -> Option<&SlideTangent> {
        self.slide[c].as_ref()
    }

    /// Chain a parameter differential through the map: given the differential
    /// of the stationarity residual `(A + R) f + b - e_n bias` on every row
    /// (`d_stationarity`, length `3n`, computed with the impulses held fixed)
    /// and, per sliding contact, `dM_t (t* - f_t)` (`d_mt_rel`, the
    /// contact's own tangential-block differential applied to
    /// [`SlideTangent::t_rel`]; entries for non-sliding contacts are ignored),
    /// return `df`.
    ///
    /// Rows the map does not respond on (separating contacts) are ignored by
    /// construction: their rows of `-K'^-1` produce zero.
    // The loop indices drive stride arithmetic into flat, row-major arrays
    // (base = 3*c), matching the rest of this module.
    #[allow(clippy::needless_range_loop)]
    pub fn apply(&self, d_stationarity: &[f64], d_mt_rel: &[[f64; 2]]) -> Vec<Vec3> {
        let n = self.n;
        let dim = 3 * n;
        let mut dres = vec![0.0; dim];
        for c in 0..n {
            let base = 3 * c;
            if self.regimes[c] == ContactRegime::Separating {
                // The map's rows for a separating contact are `f = 0`,
                // which carries no parameter dependence at all.
                continue;
            }
            match &self.slide[c] {
                None => {
                    dres[base] = d_stationarity[base];
                    if self.regimes[c] != ContactRegime::Sliding {
                        // Sticking: all three rows are stationarity rows. A
                        // sliding contact with no defined slip direction
                        // (frictionless under load) keeps the tangential pin
                        // `f_t = 0`, which has no parameter channel.
                        dres[base + 1] = d_stationarity[base + 1];
                        dres[base + 2] = d_stationarity[base + 2];
                    }
                }
                Some(st) => {
                    dres[base] = d_stationarity[base];
                    let dr = [
                        d_stationarity[base + 1] + d_mt_rel[c][0],
                        d_stationarity[base + 2] + d_mt_rel[c][1],
                    ];
                    dres[base + 1] = st.map[0][0] * dr[0] + st.map[0][1] * dr[1];
                    dres[base + 2] = st.map[1][0] * dr[0] + st.map[1][1] * dr[1];
                }
            }
        }
        let mut out = vec![Vec3::zeros(); n];
        for c in 0..n {
            let mut d = [0.0; 3];
            for (r, dr) in d.iter_mut().enumerate() {
                let row = 3 * c + r;
                let mut acc = 0.0;
                for (col, dc) in dres.iter().enumerate() {
                    acc += self.inv[row * dim + col] * dc;
                }
                *dr = acc;
            }
            out[c] = Vec3::new(d[0], d[1], d[2]);
        }
        out
    }

    /// Transpose of [`FixedPointSensitivity::apply`]: given a covector on
    /// the impulses, return the covector on the stationarity residual (per
    /// row) and, per sliding contact, the covector on `dM_t (t* - f_t)`.
    ///
    /// This is the reverse-mode counterpart the solver-level adjoint uses at
    /// a converged solve: the two are transposes of the same `-K'^-1`-based
    /// map, so a forward/reverse dot-product identity holds to rounding.
    // Stride arithmetic into flat, row-major arrays, as above.
    #[allow(clippy::needless_range_loop)]
    pub fn apply_transpose(&self, bar_f: &[Vec3]) -> (Vec<f64>, Vec<[f64; 2]>) {
        let n = self.n;
        let dim = 3 * n;
        // bar_dres = inv^T bar_out.
        let mut bar_out = vec![0.0; dim];
        for c in 0..n {
            let base = 3 * c;
            bar_out[base] = bar_f[c].x;
            bar_out[base + 1] = bar_f[c].y;
            bar_out[base + 2] = bar_f[c].z;
        }
        let mut bar_dres = vec![0.0; dim];
        for col in 0..dim {
            let mut acc = 0.0;
            for row in 0..dim {
                acc += self.inv[row * dim + col] * bar_out[row];
            }
            bar_dres[col] = acc;
        }
        // Invert the dres construction of `apply`, row block by row block.
        let mut bar_stat = vec![0.0; dim];
        let mut bar_mt = vec![[0.0; 2]; n];
        for c in 0..n {
            let base = 3 * c;
            if self.regimes[c] == ContactRegime::Separating {
                continue;
            }
            match &self.slide[c] {
                None => {
                    bar_stat[base] = bar_dres[base];
                    if self.regimes[c] != ContactRegime::Sliding {
                        bar_stat[base + 1] = bar_dres[base + 1];
                        bar_stat[base + 2] = bar_dres[base + 2];
                    }
                }
                Some(st) => {
                    bar_stat[base] = bar_dres[base];
                    // Forward: dres_t = map * (d_stat_t + d_mt_rel), so both
                    // channels receive map^T bar_dres_t.
                    let dr = [
                        st.map[0][0] * bar_dres[base + 1] + st.map[1][0] * bar_dres[base + 2],
                        st.map[0][1] * bar_dres[base + 1] + st.map[1][1] * bar_dres[base + 2],
                    ];
                    bar_stat[base + 1] = dr[0];
                    bar_stat[base + 2] = dr[1];
                    bar_mt[c] = dr;
                }
            }
        }
        (bar_stat, bar_mt)
    }

    /// `df/db` under the exact map linearization, same layout as
    /// [`impulse_sensitivity`].
    // Stride arithmetic into flat, row-major arrays, as above.
    #[allow(clippy::needless_range_loop)]
    pub fn free_velocity_sensitivity(&self) -> Vec<f64> {
        let n = self.n;
        let dim = 3 * n;
        // df/db = inv * D, D block-diagonal: sticking I3, sliding
        // [[1, 0], [0, C]], separating 0 (its inv rows are zero anyway, and
        // its stationarity rows do not respond).
        let mut out = vec![0.0; dim * dim];
        for row in 0..dim {
            for c in 0..n {
                let base = 3 * c;
                if self.regimes[c] == ContactRegime::Separating {
                    // `b` does not enter a separating contact's rows.
                    continue;
                }
                match &self.slide[c] {
                    None => {
                        // Sticking: all three rows respond to `db`. A sliding
                        // contact with no defined slip direction (frictionless
                        // under load) responds on the normal row only — its
                        // tangential rows are the algebraic pin `f_t = 0`.
                        let rows = if self.regimes[c] == ContactRegime::Sliding {
                            1
                        } else {
                            3
                        };
                        for r in 0..rows {
                            out[row * dim + base + r] += self.inv[row * dim + base + r];
                        }
                    }
                    Some(st) => {
                        out[row * dim + base] += self.inv[row * dim + base];
                        for i in 0..2 {
                            for j in 0..2 {
                                out[row * dim + base + 1 + j] +=
                                    self.inv[row * dim + base + 1 + i] * st.map[i][j];
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

fn flat_impulses(impulses: &[Vec3]) -> Vec<f64> {
    let mut out = Vec::with_capacity(3 * impulses.len());
    for f in impulses {
        out.push(f.x);
        out.push(f.y);
        out.push(f.z);
    }
    out
}

/// The sensitivity of the solved impulses to the free contact-space velocity.
///
/// Returns `df/db` as a dense row-major `3n x 3n` matrix. This is the block
/// the trajectory adjoint contracts against: `b = J * qd_free`, so
/// `df/dqd_free = (df/db) J`, and the whole contact channel of the adjoint
/// follows from it by the chain rule.
///
/// The linearized system is, per contact:
/// - separating: `df = 0`;
/// - sticking:   `sum_k (A + R)_ck df_k = -db_c` in all three rows;
/// - sliding:    the normal row as above; the tangential rows follow the cone
///   boundary along the slip direction, **including its rotation** — the
///   `d t_hat` channel [`FixedPointSensitivity`] documents. (An earlier form
///   held `t_hat` fixed; the box-on-plane trajectory adjoint measured that as
///   a systematic few-parts-in-1e4 bias of the step Jacobian.)
// The loop indices here drive stride arithmetic into flat, row-major arrays
// (base = 3*c, k*dim + ...). Iterator form would hide the linear algebra, so
// the explicit ranges stay.
#[allow(clippy::needless_range_loop)]
pub fn impulse_sensitivity(
    problem: &ContactProblem,
    solution: &ContactSolution,
    config: &ContactSolverConfig,
) -> Option<Vec<f64>> {
    if !solution.converged {
        // The IFT anchors on a fixed point. An unconverged iterate is not
        // one, and silently differentiating it would produce a confident
        // wrong number — the worst failure mode for a gradient.
        return None;
    }
    if problem.n == 0 {
        return Some(Vec::new());
    }
    // Delegate to the exact map linearization: for sticking/separating
    // contacts it coincides with the historical KKT form, and for sliding
    // contacts it additionally carries the slip-direction rotation the
    // pinned-`t_hat` form dropped.
    Some(FixedPointSensitivity::at(problem, solution, config)?.free_velocity_sensitivity())
}

/// The KKT matrix of the contact problem at a fixed active set.
///
/// This is the single definition of "the linear system the contact solve is
/// solving", and both directions of the crate go through it: the active-set
/// Newton in [`crate::convex`] *solves* it for the impulses, and
/// [`impulse_sensitivity`] *differentiates* through it. Deriving them from one
/// function is deliberate. Twice now this crate has shipped a solver and a
/// gradient that disagreed about the system being solved, and both times the
/// symptom was a confidently wrong number rather than a failure.
///
/// Row structure, per contact and per regime:
///
/// - **Separating**: `f_c = 0`, so the three rows are the identity.
/// - **Sticking**: the three rows of `A + R`. The constraint is active as an
///   equality in all directions.
/// - **Sliding**: the normal row of `A + R`, plus two algebraic rows pinning
///   `f_t` to `mu * f_n * t_hat` along the (fixed) slip direction read off
///   `impulses`.
///
/// `impulses` supplies only the slip directions; the matrix is otherwise a
/// function of the problem, the config and the regimes.
#[allow(clippy::needless_range_loop)]
pub(crate) fn kkt_matrix(
    problem: &ContactProblem,
    config: &ContactSolverConfig,
    regimes: &[ContactRegime],
    impulses: &[Vec3],
) -> Vec<f64> {
    let n = problem.n;
    let dim = 3 * n;
    let mut k = vec![0.0; dim * dim];
    for c in 0..n {
        let base = 3 * c;
        match regimes[c] {
            ContactRegime::Separating => {
                // df_c = 0 regardless of db.
                for r in 0..3 {
                    k[(base + r) * dim + base + r] = 1.0;
                }
            }
            ContactRegime::Sticking => {
                let reg = crate::convex::regularization_diag(problem, c, config);
                for r in 0..3 {
                    for col in 0..dim {
                        k[(base + r) * dim + col] = problem.delassus[(base + r) * dim + col];
                    }
                    k[(base + r) * dim + base + r] += reg[r];
                }
            }
            ContactRegime::Sliding => {
                // Normal row: the non-penetration equality.
                let reg = crate::convex::regularization_diag(problem, c, config);
                for col in 0..dim {
                    k[base * dim + col] = problem.delassus[base * dim + col];
                }
                k[base * dim + base] += reg[0];

                // Tangential rows: f_t is pinned to the cone boundary along
                // the slip direction, so d f_t = mu * t_hat * d f_n.
                let f = impulses[c];
                let ft = (f.y * f.y + f.z * f.z).sqrt();
                let (tu, tw) = if ft > 1e-14 {
                    (f.y / ft, f.z / ft)
                } else {
                    (0.0, 0.0)
                };
                let mu = problem.rows[c].mu;
                // d f_u - mu*tu*d f_n = 0
                k[(base + 1) * dim + base + 1] = 1.0;
                k[(base + 1) * dim + base] = -mu * tu;
                // d f_w - mu*tw*d f_n = 0
                k[(base + 2) * dim + base + 2] = 1.0;
                k[(base + 2) * dim + base] = -mu * tw;
            }
        }
    }
    k
}

/// Sensitivity of the solved impulses to each contact's friction coefficient.
///
/// Returns `df/dmu` as `3n x n`, row-major. Only sliding contacts have a
/// non-zero column: a sticking contact is strictly inside the cone, so a small
/// change in `mu` moves the constraint boundary without moving the solution,
/// and a separating one carries no impulse at all.
///
/// That structural zero is a real property of Coulomb friction, not an
/// approximation — and it is a good check on the whole derivation, because a
/// finite difference reproduces it exactly.
// The loop indices here drive stride arithmetic into flat, row-major arrays
// (base = 3*c, k*dim + ...). Iterator form would hide the linear algebra, so
// the explicit ranges stay.
#[allow(clippy::needless_range_loop)]
pub fn friction_sensitivity(
    problem: &ContactProblem,
    solution: &ContactSolution,
    config: &ContactSolverConfig,
) -> Option<Vec<f64>> {
    if !solution.converged {
        return None;
    }
    let n = problem.n;
    let dim = 3 * n;
    if n == 0 {
        return Some(Vec::new());
    }

    let regimes = classify(problem, solution, 1e-7);
    let sens = impulse_sensitivity(problem, solution, config)?;

    // For a sliding contact, raising mu raises the tangential impulse it can
    // carry: d f_t = f_n * t_hat * d mu, holding f_n fixed to first order.
    // Propagate that through the coupled system via df/db.
    let mut out = vec![0.0; dim * n];
    for c in 0..n {
        if regimes[c] != ContactRegime::Sliding {
            continue;
        }
        let base = 3 * c;
        let f = solution.impulses[c];
        let ft = (f.y * f.y + f.z * f.z).sqrt();
        let (tu, tw) = if ft > 1e-14 {
            (f.y / ft, f.z / ft)
        } else {
            continue;
        };

        // Direct term on this contact's tangential rows.
        let direct = [0.0, f.x * tu, f.x * tw];
        // The extra tangential impulse acts like an equivalent change in b on
        // every other contact through A, so route it through df/db.
        let mut db = vec![0.0; dim];
        for r in 0..3 {
            for k in 0..dim {
                db[k] += problem.delassus[k * dim + base + r] * direct[r];
            }
        }
        for row in 0..dim {
            let mut acc = 0.0;
            for k in 0..dim {
                acc += sens[row * dim + k] * db[k];
            }
            // The direct term is already the response on this contact.
            out[row * n + c] = acc
                + if row >= base && row < base + 3 {
                    direct[row - base]
                } else {
                    0.0
                };
        }
    }
    Some(out)
}

/// Sensitivity of the solved impulses to each contact's penetration depth.
///
/// Returns `df/ddepth` as `3n x n`, row-major.
///
/// Position stabilization made the solve depend on the state through a second
/// channel: alongside `b = J qd_free` there is now a per-contact bias
/// `bias_c = d * erp * depth_c / dt`, and a trajectory adjoint that only
/// contracted through `df/db` would be missing that channel entirely — it
/// would report that penetration has no effect on the impulses, which is
/// exactly backwards now that penetration is what repays itself.
///
/// The bias enters the normal row of the KKT system with the opposite sign to
/// `b`, so `df/dbias = -df/db` on the normal rows, and the chain rule closes
/// with `dbias/ddepth = d * erp / dt`. Pass the same `dt` the assembly used;
/// `erp` comes from the pair material's [`crate::SolRef`].
///
/// # Two channels, not one
///
/// Depth reaches the impulses two ways, and both are included.
///
/// **The bias.** `bias_c = d * erp * violation_c / dt` with
/// `violation = max(depth, 0)`. Since `d` is itself a function of depth, the
/// product rule gives `dbias/ddepth = erp/dt * (d + d' * depth)` while
/// penetrating, and exactly zero once separated.
///
/// **The regularizer.** `R_c = (1-d)/d * A_nn` moves with `d` too, and it is a
/// coefficient *on the impulse itself*, so it perturbs the KKT residual by
/// `f_c * dR/ddepth`. This term used to be dropped, on the argument that it
/// moves the smoothing parameter of the relaxation rather than the physics
/// being relaxed, and was described as bounded by `(dmax-dmin)/dmin ~ 5%`.
///
/// That argument does not survive the contact margin. Inside the margin band a
/// contact is separated, so the bias is identically zero and the regularizer is
/// the **only** channel from depth to force. Dropping it would report a depth
/// sensitivity of exactly zero across the entire band — while the force in fact
/// ramps from 17.66 N to 0 across a 1 mm band, about `1.8e4 N/m`. And the band
/// is not an edge case for this crate: it is precisely the region where contact
/// activation is differentiable at all, which is where a contact-timing
/// gradient for system identification would live.
///
/// Including it on the penetrating side as well is the same fix rather than a
/// separate one. It restores the `~5%` the old form knowingly gave up, and it
/// means there is one expression valid everywhere instead of two regimes with
/// different accuracy.
///
/// # The derivation
///
/// The KKT residual for a loaded contact is `F = (A + R) f + b - e_n bias`.
/// Differentiating in `depth_c`, holding the active set fixed as everywhere
/// else in this module:
///
/// ```text
/// dF/ddepth_c = f_c ⊙ dR_c/ddepth_c  -  e_n dbias_c/ddepth_c
/// ```
///
/// supported only on contact `c`'s own rows, because `R` and `bias` are both
/// per-contact. Then `df/ddepth = -K^-1 dF/ddepth`, and `-K^-1` restricted to
/// the responding rows is exactly the `df/db` that [`impulse_sensitivity`]
/// already computed — so the whole thing is a contraction of `sens` against a
/// three-entry vector, with no second factorization.
///
/// Separating contacts fall out correctly with no special case: they carry
/// `f = 0`, so the `R` term vanishes, and their columns of `df/db` are zero.
///
/// Pass the same `dt` the assembly used; `erp` comes from the pair material's
/// [`crate::SolRef`].
pub fn depth_sensitivity(
    problem: &ContactProblem,
    solution: &ContactSolution,
    config: &ContactSolverConfig,
    solref: &crate::material::SolRef,
    dt: f64,
) -> Option<Vec<f64>> {
    let n = problem.n;
    let dim = 3 * n;
    let sens = impulse_sensitivity(problem, solution, config)?;
    if n == 0 {
        return Some(Vec::new());
    }
    let erp = solref.error_reduction(dt);
    let mut out = vec![0.0; dim * n];
    for c in 0..n {
        let base = 3 * c;
        let contact_row = &problem.rows[c];
        let f = solution.impulses[c];

        // Channel 1: the stabilization bias, which exists only while the
        // surfaces actually overlap. Product rule over `d * violation`.
        let dbias = if contact_row.depth > 0.0 && dt > 0.0 {
            erp / dt * (contact_row.impedance + contact_row.dimpedance_ddepth * contact_row.depth)
        } else {
            0.0
        };

        // Channel 2: the regularizer, which is a coefficient on the impulse.
        // Live on both sides of zero depth, and the only live channel inside
        // the margin band.
        let dreg = crate::convex::regularization_depth_derivative(problem, c, config);
        let f_components = [f.x, f.y, f.z];

        // dF/ddepth_c, supported on contact c's own three rows.
        let mut dresidual = [0.0; 3];
        for r in 0..3 {
            dresidual[r] = f_components[r] * dreg[r];
        }
        dresidual[0] -= dbias;

        // df/ddepth_c = (df/db) * dF/ddepth_c.
        for row in 0..dim {
            let mut acc = 0.0;
            for (r, dres) in dresidual.iter().enumerate() {
                acc += sens[row * dim + base + r] * dres;
            }
            out[row * n + c] = acc;
        }
    }
    Some(out)
}

/// Contract a cotangent on the impulses back to a cotangent on `b`.
///
/// This is the operation a reverse-mode trajectory adjoint actually needs:
/// given `dJ/df`, produce `dJ/db = (df/db)^T dJ/df`, in one pass and without
/// ever forming a Jacobian per parameter.
pub fn pullback_to_free_velocity(sensitivity: &[f64], dim: usize, cotangent: &[Vec3]) -> Vec<f64> {
    let mut flat = vec![0.0; dim];
    for (c, g) in cotangent.iter().enumerate() {
        flat[3 * c] = g.x;
        flat[3 * c + 1] = g.y;
        flat[3 * c + 2] = g.z;
    }
    let mut out = vec![0.0; dim];
    for col in 0..dim {
        let mut acc = 0.0;
        for row in 0..dim {
            acc += sensitivity[row * dim + col] * flat[row];
        }
        out[col] = acc;
    }
    out
}

/// Gauss-Jordan solve of `A X = B` in place; `B` becomes `X`.
///
/// `A` is `dim x dim` row-major, `B` is `dim x ncols` row-major. Returns `None`
/// if `A` is singular, which for the KKT matrix means the active set is
/// degenerate — a case where the derivative genuinely does not exist.
///
/// `ncols` is a parameter rather than `dim` because the same factorization
/// serves two callers with different right-hand sides: the sensitivity below
/// solves against `-I` (`ncols == dim`), and the active-set Newton step in
/// [`crate::convex`] solves against a single vector (`ncols == 1`).
pub(crate) fn solve_dense(a: &mut [f64], b: &mut [f64], dim: usize, ncols: usize) -> Option<()> {
    for col in 0..dim {
        let mut pivot = col;
        for r in col + 1..dim {
            if a[r * dim + col].abs() > a[pivot * dim + col].abs() {
                pivot = r;
            }
        }
        if a[pivot * dim + col].abs() < 1e-14 {
            return None;
        }
        if pivot != col {
            for k in 0..dim {
                a.swap(col * dim + k, pivot * dim + k);
            }
            for k in 0..ncols {
                b.swap(col * ncols + k, pivot * ncols + k);
            }
        }
        let d = a[col * dim + col];
        for k in 0..dim {
            a[col * dim + k] /= d;
        }
        for k in 0..ncols {
            b[col * ncols + k] /= d;
        }
        for r in 0..dim {
            if r == col {
                continue;
            }
            let factor = a[r * dim + col];
            if factor == 0.0 {
                continue;
            }
            for k in 0..dim {
                a[r * dim + k] -= factor * a[col * dim + k];
            }
            for k in 0..ncols {
                b[r * ncols + k] -= factor * b[col * ncols + k];
            }
        }
    }
    Some(())
}
