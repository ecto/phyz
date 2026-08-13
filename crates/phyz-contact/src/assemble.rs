//! Assemble a [`ContactProblem`] from a model, a state, and a contact set.
//!
//! This is the bridge the convex solve needs and the penalty model never did:
//! a per-contact force law only ever looked at one contact, so it could not
//! represent the fact that pressing on one corner of a box unloads another.
//! The Delassus operator `A = J M^-1 J^T` is exactly that coupling.

use crate::convex::{ContactProblem, ContactRow, ContactSolverConfig};
use crate::material::ContactMaterial;
use phyz_collision::Collision;
use phyz_math::{DMat, DVec, SpatialVec, Vec3};
use phyz_model::{Model, State};

/// Build the convex contact problem for `contacts` at the current state.
///
/// `free_qd` is the generalized velocity the system would have after this step
/// *without* contact — i.e. after gravity, actuation and every other force.
/// The solve then finds the impulses that correct it.
/// `free_qd` is the generalized velocity the system would have after this step
/// *without* contact; `dt` is the step those impulses will act over, which the
/// solref position-stabilization bias needs to convert a penetration depth into
/// a separating velocity.
pub fn assemble(
    model: &Model,
    state: &State,
    contacts: &[Collision],
    materials: &[ContactMaterial],
    free_qd: &DVec,
    dt: f64,
    config: &ContactSolverConfig,
) -> ContactAssembly {
    let n = contacts.len();
    let nv = model.nv;
    let (xforms, _) = phyz_rigid::forward_kinematics(model, state);

    // Mass matrix and its inverse, shared by every contact.
    let mass = phyz_rigid::crba(model, state);
    let inv_mass = invert_symmetric(&mass);

    // Per-contact 3 x nv constraint Jacobian, rows ordered [normal, u, w].
    let mut jacobians: Vec<DMat> = Vec::with_capacity(n);
    for c in contacts {
        let point_j = phyz_rigid::relative_point_jacobian(
            model,
            &xforms,
            c.body_i,
            c.body_j,
            c.contact_point,
        );
        let (nrm, u, w) = crate::cone::contact_frame(&c.contact_normal);
        let mut rows = DMat::zeros(3, nv);
        for col in 0..nv {
            let v = Vec3::new(point_j[(0, col)], point_j[(1, col)], point_j[(2, col)]);
            rows[(0, col)] = v.dot(nrm);
            rows[(1, col)] = v.dot(u);
            rows[(2, col)] = v.dot(w);
        }
        jacobians.push(rows);
    }

    // A = J M^-1 J^T, assembled block by block.
    let dim = 3 * n;
    let mut delassus = vec![0.0; dim * dim];
    // Precompute M^-1 J_c^T for each contact (nv x 3).
    let mut minv_jt: Vec<DMat> = Vec::with_capacity(n);
    for jc in &jacobians {
        let mut m = DMat::zeros(nv, 3);
        for r in 0..nv {
            for k in 0..3 {
                let mut acc = 0.0;
                for col in 0..nv {
                    acc += inv_mass[(r, col)] * jc[(k, col)];
                }
                m[(r, k)] = acc;
            }
        }
        minv_jt.push(m);
    }
    for a in 0..n {
        for b in 0..n {
            for r in 0..3 {
                for k in 0..3 {
                    let mut acc = 0.0;
                    for col in 0..nv {
                        acc += jacobians[a][(r, col)] * minv_jt[b][(col, k)];
                    }
                    delassus[(3 * a + r) * dim + 3 * b + k] = acc;
                }
            }
        }
    }

    // b = J * free_qd, with restitution folded into the normal row.
    let mut free_velocity = vec![0.0; dim];
    let mut rows = Vec::with_capacity(n);
    for (ci, c) in contacts.iter().enumerate() {
        for r in 0..3 {
            let mut acc = 0.0;
            for col in 0..nv {
                acc += jacobians[ci][(r, col)] * free_qd[col];
            }
            free_velocity[3 * ci + r] = acc;
        }

        let material = material_for(materials, c.body_i, c.body_j);
        let approach = (-free_velocity[3 * ci]).max(0.0);
        let e = ContactProblem::effective_restitution(
            material.restitution,
            approach,
            config.restitution_threshold,
        );
        // Target normal velocity is `+e * approach` instead of 0.
        free_velocity[3 * ci] *= 1.0 + e;

        // `from_material` applies the margin ramp itself (and records the
        // impedance's depth derivative alongside it, which the gradient needs),
        // so a separated-but-detected contact comes out with a tapering
        // impedance rather than the full `dmin`.
        rows.push(ContactRow::from_material(
            &material,
            c.penetration_depth,
            dt,
            e,
        ));
    }

    ContactAssembly {
        problem: ContactProblem {
            n,
            delassus,
            free_velocity,
            rows,
        },
        jacobians,
        inv_mass,
    }
}

/// Everything the assembly produced: the convex problem, the per-contact
/// constraint Jacobians, and the inverse mass matrix they were built from.
///
/// The inverse mass is kept because the caller needs it again to turn solved
/// impulses into a velocity change, and recomputing it would mean a second
/// `crba` and inversion per step.
pub struct ContactAssembly {
    /// The convex problem to hand to [`crate::solve_contacts`].
    pub problem: ContactProblem,
    /// Per-contact `3 x nv` constraint Jacobian, rows `[normal, u, w]`.
    pub jacobians: Vec<DMat>,
    /// `M^-1` at the assembly configuration.
    pub inv_mass: DMat,
}

impl ContactAssembly {
    /// The generalized velocity change `M^-1 J^T f` produced by `impulses`.
    pub fn velocity_delta(&self, impulses: &[Vec3]) -> DVec {
        let nv = self.inv_mass.nrows();
        let tau = generalized_impulse_inner(nv, &self.jacobians, impulses);
        let mut out = DVec::zeros(nv);
        for r in 0..nv {
            let mut acc = 0.0;
            for c in 0..nv {
                acc += self.inv_mass[(r, c)] * tau[c];
            }
            out[r] = acc;
        }
        out
    }
}

fn generalized_impulse_inner(nv: usize, jacobians: &[DMat], impulses: &[Vec3]) -> DVec {
    let mut out = DVec::zeros(nv);
    for (j, f) in jacobians.iter().zip(impulses) {
        let fv = [f.x, f.y, f.z];
        for col in 0..nv {
            let mut acc = 0.0;
            for (r, fr) in fv.iter().enumerate() {
                acc += j[(r, col)] * fr;
            }
            out[col] += acc;
        }
    }
    out
}

/// Map solved contact impulses back to generalized impulses `J^T f`.
pub fn generalized_impulse(model: &Model, jacobians: &[DMat], impulses: &[Vec3]) -> DVec {
    generalized_impulse_inner(model.nv, jacobians, impulses)
}

/// Distribute solved contact impulses onto per-body spatial wrenches, for
/// callers that still want a force-style interface.
pub fn contact_wrenches(
    state: &State,
    contacts: &[Collision],
    impulses: &[Vec3],
    dt: f64,
) -> Vec<SpatialVec> {
    let nbodies = state.body_xform.len();
    let mut out = vec![SpatialVec::zero(); nbodies];
    if dt <= 0.0 {
        return out;
    }
    for (c, f) in contacts.iter().zip(impulses) {
        let (nrm, u, w) = crate::cone::contact_frame(&c.contact_normal);
        // Impulse -> force over the step.
        let force = (nrm * f.x + u * f.y + w * f.z) / dt;

        let i = c.body_i;
        let r_i = c.contact_point - state.body_xform[i].pos;
        out[i] = out[i] + SpatialVec::new(r_i.cross(force), force);

        if c.body_j != usize::MAX {
            let j = c.body_j;
            let r_j = c.contact_point - state.body_xform[j].pos;
            out[j] = out[j] + SpatialVec::new(r_j.cross(-force), -force);
        }
    }
    out
}

/// The material a contact pair is solved with.
///
/// This used to read `body_i`'s material and ignore `body_j`'s entirely, which
/// made the physics depend on which body the narrow phase listed first: the
/// same rubber-on-ice contact gripped or slid depending on collision ordering.
/// Both sides now go through [`ContactMaterial::combine`], whose rules are
/// commutative, so ordering cannot matter.
///
/// `usize::MAX` in `body_j` is the world/ground, which has no entry in
/// `materials`; the other body's material stands in for the pair, which is the
/// same convention the ground contact always had.
fn material_for(materials: &[ContactMaterial], body_i: usize, body_j: usize) -> ContactMaterial {
    if materials.is_empty() {
        return ContactMaterial::default();
    }
    let pick = |b: usize| materials[b.min(materials.len() - 1)].clone();
    let a = pick(body_i);
    if body_j == usize::MAX {
        return a;
    }
    ContactMaterial::combine(&a, &pick(body_j))
}

/// Invert a symmetric positive-definite matrix by Gauss-Jordan with partial
/// pivoting. The mass matrix is small (nv x nv) and this runs once per step.
fn invert_symmetric(m: &DMat) -> DMat {
    let n = m.nrows();
    let mut a = vec![0.0; n * n];
    let mut inv = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            a[r * n + c] = m[(r, c)];
        }
        inv[r * n + r] = 1.0;
    }

    for col in 0..n {
        // Partial pivot.
        let mut pivot = col;
        for r in col + 1..n {
            if a[r * n + col].abs() > a[pivot * n + col].abs() {
                pivot = r;
            }
        }
        if a[pivot * n + col].abs() < 1e-14 {
            // Singular (a fully constrained or massless DOF); leave the row as
            // identity so the contact simply sees infinite inertia there.
            continue;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
                inv.swap(col * n + k, pivot * n + k);
            }
        }

        let d = a[col * n + col];
        for k in 0..n {
            a[col * n + k] /= d;
            inv[col * n + k] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r * n + col];
            if factor == 0.0 {
                continue;
            }
            for k in 0..n {
                a[r * n + k] -= factor * a[col * n + k];
                inv[r * n + k] -= factor * inv[col * n + k];
            }
        }
    }

    DMat::from_fn(n, n, |r, c| inv[r * n + c])
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform};
    use phyz_model::{Geometry, ModelBuilder};

    #[test]
    fn inverse_mass_round_trips() {
        let model = ModelBuilder::new()
            .dt(1e-3)
            .add_free_body(
                "b",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::sphere(2.0, 0.3),
            )
            .build();
        let state = model.default_state();
        let m = phyz_rigid::crba(&model, &state);
        let inv = invert_symmetric(&m);
        for r in 0..model.nv {
            for c in 0..model.nv {
                let mut acc = 0.0;
                for k in 0..model.nv {
                    acc += m[(r, k)] * inv[(k, c)];
                }
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((acc - want).abs() < 1e-9, "M*M^-1 [{r},{c}] = {acc}");
            }
        }
    }

    /// A free sphere resting on the ground with zero penetration: the solve
    /// must exactly cancel the gravitational approach velocity, and nothing
    /// else. Zero depth is deliberate — it isolates the non-penetration part
    /// from the solref stabilization bias, which
    /// `penetration_adds_a_recovery_impulse` covers separately.
    #[test]
    fn free_body_on_ground_cancels_gravity() {
        let mass = 2.0;
        let dt = 1e-3;
        let model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
            .dt(dt)
            .add_free_body(
                "ball",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::sphere(mass, 0.1),
            )
            .build();
        let mut state = model.default_state();
        state.body_xform[0] = SpatialTransform::new(Mat3::identity(), Vec3::zeros());

        let contacts = vec![Collision {
            body_i: 0,
            body_j: usize::MAX,
            contact_point: Vec3::new(0.0, 0.0, -0.1),
            contact_normal: Vec3::z(),
            penetration_depth: 0.0,
        }];

        // Free velocity after one step of gravity.
        let mut free_qd = DVec::zeros(model.nv);
        free_qd[5] = -GRAVITY * dt;

        let cfg = ContactSolverConfig::simulation();
        let asm = assemble(
            &model,
            &state,
            &contacts,
            &[ContactMaterial::default()],
            &free_qd,
            dt,
            &cfg,
        );
        let sol = crate::solve_contacts(&asm.problem, &cfg);
        assert!(sol.converged);

        // Normal impulse is `d * m*g*dt`, not `m*g*dt`: at zero penetration
        // the solimp impedance is `dmin`, so the constraint is deliberately
        // soft and removes only that fraction of the approach velocity. The
        // remaining 10% is what drives the body to its (small, bounded)
        // steady-state penetration, where the recovery bias makes up the
        // difference exactly — see `penetration_adds_a_recovery_impulse` and
        // the stack test in `tests/stabilization.rs`.
        let d = ContactMaterial::default().solimp.impedance(0.0);
        let expected = d * mass * GRAVITY * dt;
        assert!(
            (sol.impulses[0].x - expected).abs() / expected < 1e-3,
            "normal impulse {} vs d*m*g*dt {expected}",
            sol.impulses[0].x
        );

        // And it maps back to a generalized impulse purely along +z.
        let gen_impulse = generalized_impulse(&model, &asm.jacobians, &sol.impulses);
        assert!(
            (gen_impulse[5] - expected).abs() / expected < 1e-3,
            "generalized z impulse {}",
            gen_impulse[5]
        );
    }

    /// The same sphere, but penetrating: the solve must carry *more* than the
    /// weight, because it is also paying back the penetration. Before position
    /// stabilization `ContactRow::depth` was stored and never read, so this
    /// extra impulse did not exist and a resting body kept whatever
    /// penetration it had accumulated forever.
    #[test]
    fn penetration_adds_a_recovery_impulse() {
        let mass = 2.0;
        let dt = 1e-3;
        let depth = 1e-4;
        let model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
            .dt(dt)
            .add_free_body(
                "ball",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::sphere(mass, 0.1),
            )
            .build();
        let mut state = model.default_state();
        state.body_xform[0] = SpatialTransform::new(Mat3::identity(), Vec3::zeros());

        let contacts = vec![Collision {
            body_i: 0,
            body_j: usize::MAX,
            contact_point: Vec3::new(0.0, 0.0, -0.1),
            contact_normal: Vec3::z(),
            penetration_depth: depth,
        }];
        let mut free_qd = DVec::zeros(model.nv);
        free_qd[5] = -GRAVITY * dt;

        let cfg = ContactSolverConfig::simulation();
        let material = ContactMaterial::default();
        let asm = assemble(
            &model,
            &state,
            &contacts,
            std::slice::from_ref(&material),
            &free_qd,
            dt,
            &cfg,
        );
        let sol = crate::solve_contacts(&asm.problem, &cfg);
        assert!(sol.converged);

        let weight = mass * GRAVITY * dt;
        assert!(
            sol.impulses[0].x > weight,
            "penetrating contact must push harder than the weight: {} vs {weight}",
            sol.impulses[0].x
        );

        // The post-solve normal velocity should be the bias, i.e. separating.
        let row = asm.problem.rows[0];
        assert!(row.bias > 0.0, "a penetrating row must carry a bias");
        let dv = asm.velocity_delta(&sol.impulses);
        let v_after = free_qd[5] + dv[5];
        // Separating, not merely stopped — and short of the full bias,
        // because the impedance keeps the constraint soft.
        assert!(
            v_after > 0.0 && v_after < row.bias,
            "normal velocity after solve {v_after} should separate, under the \
             bias {}",
            row.bias
        );
        // And the recovery is gentle: one step never removes more than the
        // penetration itself.
        assert!(row.bias * dt <= depth + 1e-15);
    }

    /// Two contacts on one body are coupled through `A`: the off-diagonal
    /// blocks must be non-zero. A per-contact force law has no way to see this.
    #[test]
    fn two_contacts_on_one_body_are_coupled() {
        let model = ModelBuilder::new()
            .dt(1e-3)
            .add_free_body(
                "box",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(
                    1.0,
                    Vec3::zeros(),
                    Mat3::from_diagonal(&Vec3::new(0.1, 0.1, 0.1)),
                ),
            )
            .build();
        let mut state = model.default_state();
        state.body_xform[0] = SpatialTransform::identity();
        let _ = Geometry::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        };

        let contacts = vec![
            Collision {
                body_i: 0,
                body_j: usize::MAX,
                contact_point: Vec3::new(0.5, 0.0, -0.5),
                contact_normal: Vec3::z(),
                penetration_depth: 1e-4,
            },
            Collision {
                body_i: 0,
                body_j: usize::MAX,
                contact_point: Vec3::new(-0.5, 0.0, -0.5),
                contact_normal: Vec3::z(),
                penetration_depth: 1e-4,
            },
        ];

        let cfg = ContactSolverConfig::simulation();
        let asm = assemble(
            &model,
            &state,
            &contacts,
            &[ContactMaterial::default()],
            &DVec::zeros(model.nv),
            1e-3,
            &cfg,
        );
        let _dim = 6;
        // Normal-normal off-diagonal block entry between contact 0 and 1.
        // Row 0 (contact 0's normal) x column 3 (contact 1's normal).
        let off = asm.problem.delassus[3];
        assert!(
            off.abs() > 1e-9,
            "contacts on the same body must couple; A[0,3] = {off}"
        );
    }
}
