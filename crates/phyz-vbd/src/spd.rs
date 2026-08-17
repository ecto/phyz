//! The indefinite-Hessian guard: a 3×3 symmetric eigen-clamp, and the solve
//! that uses it.
//!
//! # What breaks without it
//!
//! A vertex's local step is `Δx = −H⁻¹g`. That is a descent direction **only**
//! if `H` is positive definite. Real elastic energies are not convex: a
//! buckling spring, a compressed or inverted tet, and any element in the
//! wrong-side-of-the-barrier regime all produce local Hessians with a negative
//! eigenvalue. Three distinct failures follow if you ignore that.
//!
//! 1. **Along a negative eigendirection, `−H⁻¹g` points *uphill*.** Newton with
//!    an indefinite Hessian converges to whatever stationary point is nearest,
//!    saddle points included. In VBD that shows up as a vertex that climbs the
//!    energy while every other vertex descends, and the outer loop stops being
//!    a descent method at all — the energy graph wanders instead of decreasing.
//! 2. **Near a zero eigenvalue the step is unbounded.** `H` passes through
//!    singular on the way from definite to indefinite, so a vertex takes an
//!    arbitrarily large jump on the step where the sign flips. One such vertex
//!    contaminates its whole neighbourhood on the next sweep.
//! 3. **The failure is silent.** No `NaN`, no assertion: just a simulation that
//!    is slightly wrong, then obviously wrong, several hundred steps later.
//!
//! # What the guard does
//!
//! [`spd_solve`] takes the symmetric eigendecomposition, replaces every
//! eigenvalue `λ` with `max(λ, floor)`, and solves in that basis. Clamping to a
//! positive floor — rather than taking `|λ|`, or adding `σI` until the matrix
//! is definite — is the choice that matters:
//!
//! * `|λ|` would keep a large step in a direction whose curvature the model got
//!   backwards.
//! * Uniform `σI` shifting distorts the *well-conditioned* directions too, so a
//!   single bad eigenvalue slows convergence in all three axes.
//! * Clamping leaves the good directions exactly as Newton would have them and
//!   turns the bad ones into a conservative gradient step of size `g/floor`.
//!
//! This is the projected-Newton idea (Teran et al. 2005) applied per vertex
//! block, and it is what VBD's papers mean by the local solve being "always a
//! descent direction". Note the qualifier: it guarantees descent *for that
//! vertex's block sub-problem at that instant*. Combined with the fact that
//! Gauss–Seidel never raises the objective when each block step descends, the
//! total energy is non-increasing per sweep. It does **not** by itself
//! guarantee a bound on the step size across a whole timestep; the `M/h²`
//! inertia term in [`crate::VbdSolver`] does most of that work, and it weakens
//! as `h` grows, which is exactly why the guard matters more at large `h`, not
//! less.
//!
//! # Determinism
//!
//! The eigensolver is a cyclic Jacobi sweep with a *fixed* iteration count and
//! a fixed rotation order. No convergence-dependent early exit, so the number
//! of floating-point operations depends only on the matrix size — the same
//! input produces the same bits every run.

use phyz_math::{Mat3, Vec3};

/// Fixed number of cyclic Jacobi sweeps. Symmetric 3×3 Jacobi converges
/// cubically; six sweeps is machine precision with room to spare, and a fixed
/// count keeps the operation sequence identical on every call.
const JACOBI_SWEEPS: usize = 6;

/// Symmetric eigendecomposition of a 3×3 matrix: `(eigenvalues, eigenvectors)`
/// with `eigenvectors[k]` belonging to `eigenvalues[k]`.
///
/// The input is assumed symmetric; only the lower triangle plus the diagonal is
/// read, so a slightly asymmetric matrix is silently symmetrised rather than
/// producing complex results.
pub fn sym_eigen(m: &Mat3) -> ([f64; 3], [Vec3; 3]) {
    let mut a = [
        [m.get(0, 0), m.get(0, 1), m.get(0, 2)],
        [m.get(0, 1), m.get(1, 1), m.get(1, 2)],
        [m.get(0, 2), m.get(1, 2), m.get(2, 2)],
    ];
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for _ in 0..JACOBI_SWEEPS {
        for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq == 0.0 {
                continue;
            }
            // Standard Jacobi rotation. The `theta`/`t` form below is the
            // numerically stable branch: computing `tan(2θ)` directly loses
            // precision when a[p][p] ≈ a[q][q].
            let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            let t = if theta >= 0.0 {
                1.0 / (theta + (theta * theta + 1.0).sqrt())
            } else {
                -1.0 / (-theta + (theta * theta + 1.0).sqrt())
            };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;

            // A <- A J, then A <- Jt A. The column pass must complete before
            // the row pass starts: the row pass reads the already-rotated
            // columns, which is what makes this Jt (A J) and not something
            // that only looks like it.
            for row in a.iter_mut() {
                let akp = row[p];
                let akq = row[q];
                row[p] = c * akp - s * akq;
                row[q] = s * akp + c * akq;
            }
            let (rp, rq) = (a[p], a[q]);
            a[p] = core::array::from_fn(|k| c * rp[k] - s * rq[k]);
            a[q] = core::array::from_fn(|k| s * rp[k] + c * rq[k]);
            for row in v.iter_mut() {
                let vp = row[p];
                let vq = row[q];
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }

    (
        [a[0][0], a[1][1], a[2][2]],
        [
            Vec3::new(v[0][0], v[1][0], v[2][0]),
            Vec3::new(v[0][1], v[1][1], v[2][1]),
            Vec3::new(v[0][2], v[1][2], v[2][2]),
        ],
    )
}

/// Solve `H x = b` with every eigenvalue of the symmetric `H` clamped up to at
/// least `floor`.
///
/// `floor` must be strictly positive; it is the reciprocal of the largest step
/// the guard will ever produce per unit gradient, so it doubles as the trust
/// region. Returns `None` only if the inputs contain non-finite values, which
/// the caller should treat as "skip this vertex" rather than propagate.
pub fn spd_solve(h: &Mat3, b: &Vec3, floor: f64) -> Option<Vec3> {
    debug_assert!(floor > 0.0, "the eigenvalue floor must be positive");
    if !h.norm_sq().is_finite() || !b.norm_sq().is_finite() {
        return None;
    }
    let (lambda, q) = sym_eigen(h);
    let mut x = Vec3::zero();
    for k in 0..3 {
        let l = if lambda[k] > floor { lambda[k] } else { floor };
        x += q[k] * (q[k].dot(*b) / l);
    }
    x.norm_sq().is_finite().then_some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reconstruct(lambda: [f64; 3], q: [Vec3; 3]) -> Mat3 {
        let mut m = Mat3::zero();
        for k in 0..3 {
            let v = q[k];
            m = m + Mat3::from_cols(
                v * (v.x * lambda[k]),
                v * (v.y * lambda[k]),
                v * (v.z * lambda[k]),
            );
        }
        m
    }

    #[test]
    fn eigen_reconstructs_a_symmetric_matrix() {
        let m = Mat3::new(4.0, 1.0, -2.0, 1.0, 3.0, 0.5, -2.0, 0.5, 6.0);
        let (lambda, q) = sym_eigen(&m);
        let r = reconstruct(lambda, q);
        for i in 0..3 {
            for j in 0..3 {
                assert!((r.get(i, j) - m.get(i, j)).abs() < 1e-12, "{i},{j}");
            }
        }
    }

    #[test]
    fn eigenvectors_are_orthonormal() {
        let m = Mat3::new(4.0, 1.0, -2.0, 1.0, 3.0, 0.5, -2.0, 0.5, 6.0);
        let (_, q) = sym_eigen(&m);
        for i in 0..3 {
            assert!((q[i].norm() - 1.0).abs() < 1e-13);
            for j in (i + 1)..3 {
                assert!(q[i].dot(q[j]).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn eigen_handles_a_diagonal_matrix() {
        let m = Mat3::diagonal(Vec3::new(2.0, 5.0, -1.0));
        let (lambda, _) = sym_eigen(&m);
        let mut sorted = lambda;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] + 1.0).abs() < 1e-14);
        assert!((sorted[1] - 2.0).abs() < 1e-14);
        assert!((sorted[2] - 5.0).abs() < 1e-14);
    }

    /// On a well-conditioned positive definite matrix the guard must be inert:
    /// it has to reproduce the exact Newton step, or it is paying for safety
    /// with accuracy on the 99% of vertices that never needed it.
    #[test]
    fn definite_matrix_is_untouched() {
        let m = Mat3::new(4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 6.0);
        let b = Vec3::new(1.0, -2.0, 0.5);
        let x = spd_solve(&m, &b, 1e-9).unwrap();
        let r = m.mul_vec(x) - b;
        assert!(r.norm() < 1e-12, "residual {}", r.norm());
    }

    /// The guard's whole job: an indefinite Hessian must still give a descent
    /// direction, `x·b > 0` (since the step taken is `−x` and the gradient is
    /// `b`... see `VbdSolver`, which passes `b = −g`).
    #[test]
    fn indefinite_matrix_still_descends() {
        // Eigenvalues 2, 1, −5: one strongly negative direction.
        let m = Mat3::diagonal(Vec3::new(2.0, 1.0, -5.0));
        let g = Vec3::new(0.3, -0.7, 1.1);
        let step = spd_solve(&m, &(-g), 1e-3).unwrap();
        assert!(
            step.dot(g) < 0.0,
            "step is not a descent direction: g·Δx = {}",
            step.dot(g)
        );
    }

    /// A singular Hessian would give an infinite step. The floor bounds it.
    #[test]
    fn singular_matrix_gives_a_bounded_step() {
        let m = Mat3::diagonal(Vec3::new(1.0, 0.0, 1.0));
        let g = Vec3::new(0.0, 1.0, 0.0);
        let floor = 10.0;
        let step = spd_solve(&m, &(-g), floor).unwrap();
        assert!((step.y + 1.0 / floor).abs() < 1e-12, "step {step:?}");
    }

    #[test]
    fn non_finite_input_is_rejected() {
        let m = Mat3::diagonal(Vec3::new(f64::NAN, 1.0, 1.0));
        assert!(spd_solve(&m, &Vec3::new(1.0, 0.0, 0.0), 1e-6).is_none());
    }

    #[test]
    fn eigen_is_deterministic() {
        let m = Mat3::new(4.0, 1.0, -2.0, 1.0, 3.0, 0.5, -2.0, 0.5, 6.0);
        let first = sym_eigen(&m);
        for _ in 0..16 {
            let again = sym_eigen(&m);
            assert_eq!(first.0, again.0);
            assert_eq!(first.1, again.1);
        }
    }
}
