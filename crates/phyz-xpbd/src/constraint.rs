//! Constraint types and the generic XPBD projection they all share.
//!
//! Every constraint here is a scalar function `C(x)` of a handful of particle
//! positions, with a compliance `α = 1/k` (inverse stiffness, units m/N for a
//! length constraint). The solver projects them all through one routine,
//! [`project`], so adding a constraint type means supplying `C` and
//! `∇C` and nothing else.

use crate::particles::ParticleSystem;
use phyz_math::Vec3;

/// Below this gradient-weighted denominator a projection is skipped.
///
/// The denominator is `Σ wᵢ|∇Cᵢ|² + α̃`. It vanishes when every particle in the
/// constraint is pinned (all `w = 0` and `α = 0`) or when the constraint is at
/// a geometric degeneracy where its gradient is zero — a zero-length spring, a
/// collapsed tetrahedron. Dividing there produces an infinite correction that
/// destroys the whole system, so the constraint is skipped for this iteration
/// instead. Skipping is safe: a degenerate constraint carries no well-defined
/// direction to correct along, and the next substep usually leaves the
/// degeneracy on its own.
const DENOM_EPS: f64 = 1e-12;

/// Below this value of `sin²θ` the dihedral bending constraint is skipped.
///
/// `dC/d(n₁·n₂) = −1/√(1−d²)` blows up as the two triangles become coplanar —
/// which is exactly the flat rest state most cloth starts in. For a rest angle
/// that matches the flat configuration the blow-up is removable in exact
/// arithmetic (`C ≈ √(2(1−d))` cancels the `√(1−d²)`), but in floating point
/// the gradients are evaluated near a `0/0` and the corrections are noise. So
/// the constraint is skipped there.
///
/// This is a real approximation, not an exactness argument: a plate flat to
/// within ~1e-6 rad of a *different* rest angle gets no bending correction on
/// that step. It re-engages as soon as the configuration leaves the singular
/// set, which under any load is immediately.
const BEND_SIN_SQ_EPS: f64 = 1e-12;

/// A scalar constraint with compliance and its accumulated Lagrange multiplier.
///
/// The multiplier `lambda` is solver-owned state, reset at the start of every
/// substep. It is what separates XPBD from PBD: see the crate docs.
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    /// What the constraint is.
    pub kind: ConstraintKind,
    /// Compliance `α = 1/k`, the inverse of stiffness, in units of
    /// (constraint units)² / energy. `0.0` is an infinitely stiff constraint
    /// (classic PBD behaviour); larger values are softer.
    pub compliance: f64,
    /// Accumulated Lagrange multiplier for the current substep. The solver
    /// zeroes this at the start of each substep; do not set it yourself.
    pub lambda: f64,
}

impl Constraint {
    /// A constraint with the given compliance and a zeroed multiplier.
    #[must_use]
    pub fn new(kind: ConstraintKind, compliance: f64) -> Self {
        Self {
            kind,
            compliance,
            lambda: 0.0,
        }
    }

    /// A distance constraint holding `a` and `b` at `rest_length`.
    #[must_use]
    pub fn distance(a: usize, b: usize, rest_length: f64, compliance: f64) -> Self {
        Self::new(ConstraintKind::Distance { a, b, rest_length }, compliance)
    }

    /// A dihedral bending constraint over the triangle pair `(a, b, c)` and
    /// `(a, b, d)` sharing edge `a–b`, with rest dihedral angle `rest_angle`
    /// in radians (`0` for flat).
    #[must_use]
    pub fn bending(
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        rest_angle: f64,
        compliance: f64,
    ) -> Self {
        Self::new(
            ConstraintKind::Bending {
                a,
                b,
                c,
                d,
                rest_angle,
            },
            compliance,
        )
    }

    /// A tetrahedron volume constraint holding the signed volume of
    /// `(a, b, c, d)` at `rest_volume`.
    #[must_use]
    pub fn volume(
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        rest_volume: f64,
        compliance: f64,
    ) -> Self {
        Self::new(
            ConstraintKind::Volume {
                a,
                b,
                c,
                d,
                rest_volume,
            },
            compliance,
        )
    }

    /// An attachment holding particle `p` at a fixed world `target`.
    #[must_use]
    pub fn attachment(p: usize, target: Vec3, compliance: f64) -> Self {
        Self::new(ConstraintKind::Attachment { p, target }, compliance)
    }
}

/// The four constraint families this crate implements.
///
/// Indices refer to positions in a [`ParticleSystem`]. They are validated by
/// [`crate::XpbdSolver::step`] only insofar as Rust bounds-checks them: an
/// out-of-range index panics rather than silently doing nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintKind {
    /// `C = |x_b − x_a| − rest_length`. Cloth stretch, springs, cable segments.
    Distance {
        /// First particle.
        a: usize,
        /// Second particle.
        b: usize,
        /// Target separation, metres.
        rest_length: f64,
    },
    /// `C = arccos(n₁·n₂) − rest_angle` for the two triangles `(a,b,c)` and
    /// `(a,b,d)` that share edge `a–b`. Cloth and shell bending.
    Bending {
        /// First vertex of the shared edge.
        a: usize,
        /// Second vertex of the shared edge.
        b: usize,
        /// Apex of the first triangle.
        c: usize,
        /// Apex of the second triangle.
        d: usize,
        /// Rest dihedral angle in radians; `0` is flat.
        rest_angle: f64,
    },
    /// `C = V(a,b,c,d) − rest_volume` with `V` the signed tetrahedron volume.
    /// Soft-body incompressibility.
    Volume {
        /// Tetrahedron vertex 0.
        a: usize,
        /// Tetrahedron vertex 1.
        b: usize,
        /// Tetrahedron vertex 2.
        c: usize,
        /// Tetrahedron vertex 3.
        d: usize,
        /// Target signed volume, m³.
        rest_volume: f64,
    },
    /// `C = |x_p − target|`. A compliant pin to a world point.
    ///
    /// A *hard* pin is better expressed as `inv_mass = 0` — that is exact and
    /// costs no constraint. Use this when the attachment should stretch, or
    /// when the anchor moves and the particle should follow it elastically.
    Attachment {
        /// The attached particle.
        p: usize,
        /// World-space anchor.
        target: Vec3,
    },
}

/// One XPBD projection of one constraint, in place.
///
/// `h` is the *substep* duration, not the frame duration: compliance enters as
/// `α̃ = α/h²`, so passing the frame `dt` while substepping would make every
/// constraint softer by `substeps²`.
///
/// The update solved here is, from Macklin et al. (2016) eqs. 17–18,
///
/// ```text
/// Δλ = (−C − α̃ λ) / (Σ wᵢ |∇Cᵢ|² + α̃)
/// Δxᵢ = wᵢ ∇Cᵢ Δλ
/// λ  += Δλ
/// ```
///
/// The `−α̃ λ` term is the whole point. Without it this reduces to PBD, whose
/// effective stiffness depends on the iteration count and the timestep; with
/// it, `λ` converges to the true constraint force multiplier and the
/// deformation converges to `α · f`, which is a material property and nothing
/// else. [`crate::XpbdSolver`]'s docs spell out the consequence.
pub fn project(constraint: &mut Constraint, particles: &mut ParticleSystem, h: f64) {
    let alpha_tilde = constraint.compliance / (h * h);
    match constraint.kind {
        ConstraintKind::Distance { a, b, rest_length } => {
            let d = particles.positions[b] - particles.positions[a];
            let len = d.norm();
            if len < DENOM_EPS {
                // Coincident particles give no direction to separate along.
                return;
            }
            let n = d / len;
            let c = len - rest_length;
            project_generic(constraint, particles, alpha_tilde, c, &[(a, -n), (b, n)]);
        }
        ConstraintKind::Attachment { p, target } => {
            let d = particles.positions[p] - target;
            let len = d.norm();
            if len < DENOM_EPS {
                return;
            }
            let n = d / len;
            project_generic(constraint, particles, alpha_tilde, len, &[(p, n)]);
        }
        ConstraintKind::Volume {
            a,
            b,
            c,
            d,
            rest_volume,
        } => {
            let (p0, p1, p2, p3) = (
                particles.positions[a],
                particles.positions[b],
                particles.positions[c],
                particles.positions[d],
            );
            // ∇V with respect to each vertex is 1/6 of the cross product of the
            // two edges of the *opposite* face, oriented so the four gradients
            // sum to zero — which is why g0 is computed as the negated sum
            // rather than from its own cross product. Deriving g0 independently
            // is algebraically identical but not bit-identical, and a nonzero
            // gradient sum would let a free-floating tet translate itself while
            // conserving volume.
            let g1 = (p2 - p0).cross(p3 - p0) / 6.0;
            let g2 = (p3 - p0).cross(p1 - p0) / 6.0;
            let g3 = (p1 - p0).cross(p2 - p0) / 6.0;
            let g0 = -(g1 + g2 + g3);
            let vol = (p1 - p0).cross(p2 - p0).dot(p3 - p0) / 6.0;
            project_generic(
                constraint,
                particles,
                alpha_tilde,
                vol - rest_volume,
                &[(a, g0), (b, g1), (c, g2), (d, g3)],
            );
        }
        ConstraintKind::Bending {
            a,
            b,
            c,
            d,
            rest_angle,
        } => project_bending(constraint, particles, alpha_tilde, a, b, c, d, rest_angle),
    }
}

/// The XPBD update itself, given `C` and the per-particle gradients.
///
/// Split out so every constraint type shares one implementation of the
/// compliance and multiplier bookkeeping — the part that is easy to get subtly
/// wrong and hard to notice, because a wrong `α̃` still looks like a plausible
/// simulation, just at the wrong stiffness.
fn project_generic(
    constraint: &mut Constraint,
    particles: &mut ParticleSystem,
    alpha_tilde: f64,
    c: f64,
    grads: &[(usize, Vec3)],
) {
    let mut denom = alpha_tilde;
    for &(i, g) in grads {
        denom += particles.inv_mass[i] * g.dot(g);
    }
    if denom < DENOM_EPS {
        return;
    }
    let delta_lambda = (-c - alpha_tilde * constraint.lambda) / denom;
    constraint.lambda += delta_lambda;
    for &(i, g) in grads {
        let w = particles.inv_mass[i];
        if w > 0.0 {
            particles.positions[i] += g * (w * delta_lambda);
        }
    }
}

/// Dihedral bending, following the gradients in Müller et al. (2007) §4.
///
/// `C = arccos(n₁·n₂) − θ₀` where `n₁`, `n₂` are the unit normals of the two
/// triangles sharing edge `a–b`.
///
/// See [`BEND_SIN_SQ_EPS`] for the coplanar-configuration caveat.
#[allow(clippy::too_many_arguments)]
fn project_bending(
    constraint: &mut Constraint,
    particles: &mut ParticleSystem,
    alpha_tilde: f64,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    rest_angle: f64,
) {
    let Some((angle, g)) = bending_angle_and_gradients(
        particles.positions[a],
        particles.positions[b],
        particles.positions[c],
        particles.positions[d],
    ) else {
        return;
    };
    project_generic(
        constraint,
        particles,
        alpha_tilde,
        angle - rest_angle,
        &[(a, g[0]), (b, g[1]), (c, g[2]), (d, g[3])],
    );
}

/// The dihedral angle across edge `pa–pb` and its gradient at each of the four
/// vertices, or `None` at a degeneracy where the angle is undefined.
///
/// Kept separate from the projection so it can be checked against central
/// differences directly; see the `bending_gradients_match_finite_differences`
/// test at the bottom of this file. Hand-derived gradients of a formula with
/// two normalisations and a cross product are the single easiest thing in this
/// crate to get wrong by a sign, and a sign error there does not crash — it
/// produces cloth that folds instead of flattening, which looks like a tuning
/// problem.
fn bending_angle_and_gradients(pa: Vec3, pb: Vec3, pc: Vec3, pd: Vec3) -> Option<(f64, [Vec3; 4])> {
    // Work relative to `pa`, which is the origin in the reference derivation.
    // p2 is the shared edge; p3 and p4 the two apexes.
    let p2 = pb - pa;
    let p3 = pc - pa;
    let p4 = pd - pa;

    let cross13 = p2.cross(p3);
    let cross14 = p2.cross(p4);
    let l13 = cross13.norm();
    let l14 = cross14.norm();
    // A zero-area triangle has no normal, hence no dihedral.
    if l13 < DENOM_EPS || l14 < DENOM_EPS {
        return None;
    }
    let n1 = cross13 / l13;
    let n2 = cross14 / l14;
    let dot = n1.dot(n2).clamp(-1.0, 1.0);
    let sin_sq = 1.0 - dot * dot;
    if sin_sq < BEND_SIN_SQ_EPS {
        return None;
    }
    let inv_sin = 1.0 / sin_sq.sqrt();

    let q3 = (p2.cross(n2) + n1.cross(p2) * dot) / l13;
    let q4 = (p2.cross(n1) + n2.cross(p2) * dot) / l14;
    let q2 = -(p3.cross(n2) + n1.cross(p3) * dot) / l13 - (p4.cross(n1) + n2.cross(p4) * dot) / l14;
    let q1 = -(q2 + q3 + q4);

    // dC/dpᵢ = qᵢ/√(1−d²). The `arccos` contributes −1/√(1−d²), and the qᵢ as
    // written in the reference derivation are −d(n₁·n₂)/dpᵢ, not +: the two
    // minus signs cancel. Both conventions appear in the literature and the
    // finite-difference test below is what settles which one this code uses.
    let s = inv_sin;
    Some((dot.acos(), [q1 * s, q2 * s, q3 * s, q4 * s]))
}

/// Signed volume of the tetrahedron `(p0, p1, p2, p3)`.
///
/// Handy for computing a [`ConstraintKind::Volume`] rest volume from the
/// initial mesh, which is what almost every caller wants.
#[must_use]
pub fn tet_volume(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> f64 {
    (p1 - p0).cross(p2 - p0).dot(p3 - p0) / 6.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hand-derived dihedral gradients agree with central differences.
    ///
    /// This is the test that catches a sign or a missing term in
    /// `bending_angle_and_gradients` — the code there is a transcription of a
    /// paper's algebra, and transcription errors in it produce a plausible-
    /// looking but wrong simulation rather than a crash.
    #[test]
    fn bending_gradients_match_finite_differences() {
        let pts = [
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.5, 0.1, 0.0),
            Vec3::new(0.1, 0.0, -1.0),
            Vec3::new(0.0, 0.6, 0.8),
        ];
        let (_, grads) = bending_angle_and_gradients(pts[0], pts[1], pts[2], pts[3]).unwrap();

        let angle_of = |p: [Vec3; 4]| {
            bending_angle_and_gradients(p[0], p[1], p[2], p[3])
                .unwrap()
                .0
        };
        let eps = 1e-6;
        let mut worst: f64 = 0.0;
        for i in 0..4 {
            for k in 0..3 {
                let bump = |sign: f64| {
                    let mut q = pts;
                    let mut v = q[i];
                    match k {
                        0 => v.x += sign * eps,
                        1 => v.y += sign * eps,
                        2 => v.z += sign * eps,
                        _ => unreachable!(),
                    }
                    q[i] = v;
                    angle_of(q)
                };
                let numeric = (bump(1.0) - bump(-1.0)) / (2.0 * eps);
                let analytic = match k {
                    0 => grads[i].x,
                    1 => grads[i].y,
                    2 => grads[i].z,
                    _ => unreachable!(),
                };
                let err = (numeric - analytic).abs();
                worst = worst.max(err);
                assert!(
                    err < 1e-6,
                    "vertex {i} component {k}: analytic {analytic:.12}, numeric {numeric:.12}"
                );
            }
        }
        assert!(worst < 1e-6, "worst gradient error {worst:.3e}");
    }

    /// The four volume gradients sum to zero, so a volume-preserving
    /// correction never translates the tetrahedron.
    #[test]
    fn volume_gradients_sum_to_zero() {
        let (p0, p1, p2, p3) = (
            Vec3::new(0.1, -0.2, 0.3),
            Vec3::new(1.1, 0.0, 0.2),
            Vec3::new(0.0, 0.1, 1.4),
            Vec3::new(0.2, 1.3, 0.0),
        );
        let g1 = (p2 - p0).cross(p3 - p0) / 6.0;
        let g2 = (p3 - p0).cross(p1 - p0) / 6.0;
        let g3 = (p1 - p0).cross(p2 - p0) / 6.0;
        let g0 = -(g1 + g2 + g3);
        let sum = g0 + g1 + g2 + g3;
        assert!(sum.norm() < 1e-15, "gradients sum to {sum:?}");

        // And they are the gradients of `tet_volume`: check one by differences.
        let eps = 1e-7;
        let mut q = p3;
        q.y += eps;
        let up = tet_volume(p0, p1, p2, q);
        q.y -= 2.0 * eps;
        let down = tet_volume(p0, p1, p2, q);
        let numeric = (up - down) / (2.0 * eps);
        assert!(
            (numeric - g3.y).abs() < 1e-9,
            "analytic {}, numeric {numeric}",
            g3.y
        );
    }
}
