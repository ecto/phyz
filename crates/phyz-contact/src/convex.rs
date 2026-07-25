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
//! Projected over-relaxed Gauss–Seidel with a per-contact **staged Coulomb
//! update** — normal impulse first, then the tangential impulse clamped into
//! the friction disc that normal admits — run to a residual tolerance rather
//! than a fixed iteration count. Because the problem is strongly convex this
//! converges to the unique minimizer, and running to tolerance is what makes
//! the converged point a valid IFT anchor — differentiating a truncated
//! iterate would differentiate the algorithm, not the physics.
//!
//! The design doc specifies a primal-dual interior-point SOCP for the final
//! form, whose central-path parameter doubles as the gradient-smoothing knob.
//! This module implements the projected-splitting solve of the *same convex
//! problem*; `regularization` plays the smoothing role for now. Swapping the
//! inner solver does not change the problem being solved, the KKT conditions,
//! or the IFT derivation built on them.

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
    /// Iteration cap. Hitting it means the solve did *not* converge; the
    /// result is still feasible but is not a valid point to differentiate.
    pub max_iterations: usize,
    /// Successive over-relaxation factor in `(0, 2)`.
    pub relaxation: f64,
    /// Below this approach speed, restitution ramps smoothly to zero.
    ///
    /// Without it a resting body micro-bounces forever and a stack never
    /// settles. The ramp is `smoothstep`, not a hard cutoff, so restitution
    /// stays differentiable in both `e` and the approach speed.
    pub restitution_threshold: f64,
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
            max_iterations: 200,
            relaxation: 1.0,
            restitution_threshold: 0.05,
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
            max_iterations: 500,
            relaxation: 1.0,
            restitution_threshold: 0.05,
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

/// Solve the convex contact problem.
pub fn solve_contacts(problem: &ContactProblem, config: &ContactSolverConfig) -> ContactSolution {
    let n = problem.n;
    let mut f = vec![Vec3::zeros(); n];
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
            let mut m = [[0.0; 3]; 3];
            for (r, row) in m.iter_mut().enumerate() {
                for (col, e) in row.iter_mut().enumerate() {
                    *e = at(base + r, base + col);
                    if r == col {
                        *e += config.regularization;
                    }
                }
            }
            m
        })
        .collect();

    let mut residual = f64::INFINITY;
    let mut iterations = 0;

    for it in 0..config.max_iterations {
        iterations = it + 1;
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
            let a_nn = blocks[c][0][0];
            let f_n = if a_nn > 0.0 {
                (-r[0] / a_nn).max(0.0)
            } else {
                0.0
            };

            // Tangential 2x2 solve at the fixed normal impulse.
            let (m00, m01) = (blocks[c][1][1], blocks[c][1][2]);
            let (m10, m11) = (blocks[c][2][1], blocks[c][2][2]);
            let det = m00 * m11 - m01 * m10;
            let (mut t_u, mut t_w) = if det.abs() > 1e-18 {
                (
                    -(m11 * r[1] - m01 * r[2]) / det,
                    -(m00 * r[2] - m10 * r[1]) / det,
                )
            } else {
                (0.0, 0.0)
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
            debug_assert!(
                crate::cone::in_cone(target, row.mu, 1e-9),
                "staged solve must land in the friction cone"
            );
            let next = f[c] + (target - f[c]) * config.relaxation;
            max_move = max_move.max((next - f[c]).norm());
            f[c] = next;
        }

        residual = max_move;
        if residual < config.tolerance {
            break;
        }
    }

    ContactSolution {
        impulses: f,
        iterations,
        residual,
        converged: residual < config.tolerance,
    }
}

/// Convenience: build a single-contact problem for a point mass of effective
/// mass `m` against an immovable surface.
pub fn point_mass_problem(
    mass: f64,
    free_vel: Vec3,
    material: &ContactMaterial,
    depth: f64,
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
        rows: vec![ContactRow {
            mu: material.friction,
            restitution: e,
            depth,
        }],
    }
}
