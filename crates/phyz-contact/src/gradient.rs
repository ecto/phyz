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
//!   equality and the tangential magnitude is pinned to `mu * f_n`, so the
//!   tangential rows are replaced by the differential of that relation.
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
    solution
        .impulses
        .iter()
        .zip(&problem.rows)
        .map(|(f, row)| {
            if f.x <= tol {
                return ContactRegime::Separating;
            }
            let ft = (f.y * f.y + f.z * f.z).sqrt();
            let limit = row.mu * f.x;
            if ft >= limit - tol * f.x.max(1.0) && limit > 0.0 {
                ContactRegime::Sliding
            } else {
                ContactRegime::Sticking
            }
        })
        .collect()
}

/// The sensitivity of the solved impulses to the free contact-space velocity.
///
/// Returns `df/db` as a dense row-major `3n x 3n` matrix. This is the block
/// the trajectory adjoint contracts against: `b = J * qd_free`, so
/// `df/dqd_free = (df/db) J`, and the whole contact channel of the adjoint
/// follows from it by the chain rule.
///
/// The KKT system differentiated is, per contact:
/// - separating: `df = 0`;
/// - sticking:   `sum_k (A + R)_ck df_k = -db_c` in all three rows;
/// - sliding:    the normal row as above; the tangential rows replaced by the
///   differential of `f_t = -mu * f_n * t_hat`, with `t_hat` the (fixed) slip
///   direction.
pub fn impulse_sensitivity(
    problem: &ContactProblem,
    solution: &ContactSolution,
    config: &ContactSolverConfig,
) -> Option<Vec<f64>> {
    if !solution.converged {
        // The IFT anchors on a KKT point. An unconverged iterate is not one,
        // and silently differentiating it would produce a confident wrong
        // number — the worst failure mode for a gradient.
        return None;
    }

    let n = problem.n;
    let dim = 3 * n;
    if n == 0 {
        return Some(Vec::new());
    }

    let regimes = classify(problem, solution, 1e-7);

    // Build the KKT matrix K with dK = -db, then solve K X = -I.
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
                for r in 0..3 {
                    for col in 0..dim {
                        k[(base + r) * dim + col] = problem.delassus[(base + r) * dim + col];
                    }
                    k[(base + r) * dim + base + r] += config.regularization;
                }
            }
            ContactRegime::Sliding => {
                // Normal row: the non-penetration equality.
                for col in 0..dim {
                    k[base * dim + col] = problem.delassus[base * dim + col];
                }
                k[base * dim + base] += config.regularization;

                // Tangential rows: f_t is pinned to the cone boundary along
                // the slip direction, so d f_t = mu * t_hat * d f_n.
                let f = solution.impulses[c];
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

    // Right-hand side: -I for the rows that respond to db, 0 for rows that are
    // pure algebraic relations (separating, and the sliding tangential rows).
    let mut rhs = vec![0.0; dim * dim];
    for c in 0..n {
        let base = 3 * c;
        match regimes[c] {
            ContactRegime::Separating => {}
            ContactRegime::Sticking => {
                for r in 0..3 {
                    rhs[(base + r) * dim + base + r] = -1.0;
                }
            }
            ContactRegime::Sliding => {
                rhs[base * dim + base] = -1.0;
            }
        }
    }

    solve_dense(&mut k, &mut rhs, dim)?;
    Some(rhs)
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
/// `A` is `dim x dim` and `B` is `dim x dim`, both row-major. Returns `None`
/// if `A` is singular, which for the KKT matrix means the active set is
/// degenerate — a case where the derivative genuinely does not exist.
fn solve_dense(a: &mut [f64], b: &mut [f64], dim: usize) -> Option<()> {
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
                b.swap(col * dim + k, pivot * dim + k);
            }
        }
        let d = a[col * dim + col];
        for k in 0..dim {
            a[col * dim + k] /= d;
            b[col * dim + k] /= d;
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
                b[r * dim + k] -= factor * b[col * dim + k];
            }
        }
    }
    Some(())
}
