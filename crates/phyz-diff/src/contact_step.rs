//! One step with ground contact, as an expression graph.
//!
//! [`ContactStep`] extends [`crate::forward::Step`] with soft contact between a
//! fixed set of body-attached points (foot corners) and the flat ground
//! `z = 0`. Per step it mirrors the CPU pipeline of
//! `phyz_contact::assemble` + `solve_contacts` + `velocity_delta`:
//!
//! - detection: a point is a contact when its depth exceeds `-margin`,
//!   located at the midsurface `p + n depth / 2`, as
//!   `find_heightfield_contacts_model` does on flat ground;
//! - assembly: `A = J M⁻¹ Jᵀ` from a traced CRBA (armature included) and an
//!   unrolled Cholesky, `b = J qd_free`, solref bias and solimp impedance
//!   with the margin ramp, and `regularization_diag`;
//! - solve: staged-Coulomb projected Gauss-Seidel, the same per-contact update
//!   as `convex::sweep` including the anisotropic friction-disc step.
//!
//! What differs, by construction: the graph runs a **fixed** number of sweeps
//! (and a fixed number of disc Newton iterations) from a warm start supplied
//! as an input, where the CPU solver alternates active-set Newton with PGS
//! until a residual tolerance. On a redundant manifold that leaves impulse
//! error in `null(Jᵀ)`, which does not move the velocity; the parity test
//! measures `v'` and `q'` against the converged CPU solve, not impulses.
//!
//! Restrictions (asserted): restitution 0 (no impact rows), not
//! `mujoco_compat`, full coupling, flat ground.
//!
//! # Layout
//!
//! Inputs `[q (nq), v (nv), ctrl (nu), f_warm (3 n_points)]`; outputs
//! `[q' (nq), v' (nv), f (3 n_points)]`, with `f` in each point's
//! `[normal, u, w]` frame and zero for a point not in contact.

use phyz_contact::{ContactMaterial, ContactSolverConfig, convex::ContactCoupling};
use phyz_model::{Model, State};
use tang::Scalar;
use tang::{SpatialMat, SpatialTransform, SpatialVec, Vec3};
use tang_expr::codegen::{ComputeKernel, Dialect};
use tang_expr::compile::CompiledMany;
use tang_expr::{ExprGraph, ExprId, trace};

use crate::forward::{
    ancestors_fixed, motion_subspace_1, motion_subspace_n, multi_joint_transform, traced_aba,
    traced_finish_step, traced_free_velocity, traced_generalized_forces,
};
use crate::symbolic::{
    lift_spatial_inertia, lift_spatial_transform, lift_vec3, sym_joint_transform,
};

/// A contact candidate: a point fixed in a body's frame.
#[derive(Clone, Copy, Debug)]
pub struct ContactPoint {
    /// Body index in the model.
    pub body: usize,
    /// Point in that body's frame.
    pub local: phyz_math::Vec3,
}

/// Newton iterations in the friction-disc step (the CPU path runs to 1e-14).
pub(crate) const DISC_ITERS: usize = 8;

/// One contact step for a model and a fixed set of contact points.
pub struct ContactStep {
    /// The traced graph.
    pub graph: ExprGraph,
    /// Outputs: `q'`, `v'`, then `f`.
    pub out: Vec<ExprId>,
    /// Position DOFs.
    pub nq: usize,
    /// Velocity DOFs.
    pub nv: usize,
    /// Controls.
    pub nu: usize,
    /// Contact points.
    pub n_points: usize,
}

impl ContactStep {
    /// Trace one step with `sweeps` PGS sweeps.
    pub fn build(
        model: &Model,
        points: &[ContactPoint],
        material: &ContactMaterial,
        config: &ContactSolverConfig,
        sweeps: usize,
    ) -> Self {
        assert!(
            material.restitution <= 0.0,
            "contact step: restitution must be 0"
        );
        assert!(
            !config.mujoco_compat,
            "contact step: mujoco_compat is not traced"
        );
        assert!(
            config.coupling == ContactCoupling::Full,
            "contact step: only full coupling is traced"
        );
        let (nq, nv) = (model.nq, model.nv);
        let nu = model.default_state().ctrl.len();
        let np = points.len();
        let (graph, out) = trace(|| {
            let var = |i: usize| ExprId::var(i as u16);
            let q: Vec<ExprId> = (0..nq).map(var).collect();
            let v: Vec<ExprId> = (nq..nq + nv).map(var).collect();
            let ctrl: Vec<ExprId> = (nq + nv..nq + nv + nu).map(var).collect();
            let warm: Vec<ExprId> = (nq + nv + nu..nq + nv + nu + 3 * np).map(var).collect();
            traced_contact_step(
                model, points, material, config, sweeps, &q, &v, &ctrl, &warm,
            )
        });
        let (graph, out) = crate::forward::fold_constants(&graph, &out);
        Self {
            graph,
            out,
            nq,
            nv,
            nu,
            n_points: np,
        }
    }

    /// Inputs per evaluation.
    pub fn n_inputs(&self) -> usize {
        self.nq + self.nv + self.nu + 3 * self.n_points
    }

    /// Outputs per evaluation.
    pub fn n_outputs(&self) -> usize {
        self.nq + self.nv + 3 * self.n_points
    }

    /// Pack a state and a warm start into the input layout.
    pub fn inputs(&self, state: &State, warm: &[f64]) -> Vec<f64> {
        let mut x = Vec::with_capacity(self.n_inputs());
        x.extend_from_slice(state.q.as_slice());
        x.extend_from_slice(state.v.as_slice());
        x.extend_from_slice(state.ctrl.as_slice());
        x.extend_from_slice(warm);
        x
    }

    /// Live nodes in the graph.
    pub fn live_nodes(&self) -> usize {
        self.graph.live_set(&self.out).len()
    }

    /// The step's reverse-mode adjoint, folded: inputs `[x, λ]` (this
    /// step's inputs, then an adjoint of its outputs `q', v', f`), outputs
    /// `λᵀ ∂out/∂x` (`n_inputs`). Gradient flows through the fixed PGS
    /// sweeps as traced, including the warm start, and only down the taken
    /// branch of each clamp.
    pub fn adjoint(&self) -> (ExprGraph, Vec<ExprId>) {
        let ni = self.n_inputs();
        let mut g = self.graph.clone();
        let seeds: Vec<ExprId> = (0..self.n_outputs())
            .map(|i| g.var((ni + i) as u16))
            .collect();
        let grad = g.vjp(&self.out, &seeds, ni);
        crate::forward::fold_constants(&g, &grad)
    }

    /// Compile to an f64 closure.
    pub fn compile(&self) -> CompiledMany {
        self.graph.compile_many(&self.out)
    }

    /// Emit a batched kernel.
    pub fn kernel(&self, dialect: Dialect) -> ComputeKernel {
        self.graph.to_kernel(&self.out, self.n_inputs(), dialect)
    }

    /// Emit a batched plain-C kernel in f64.
    pub fn kernel_c_f64(&self) -> ComputeKernel {
        self.graph.to_c_f64(&self.out, self.n_inputs())
    }
}

/// Everything the contact solve reads, per environment: the contact-free
/// velocity, `A = J M⁻¹ Jᵀ` (row-major `3n x 3n`), `b = J qd_free`, the
/// stabilization bias and normal regularizer per point, the activity mask, and
/// `M⁻¹ Jᵀ` as one `nv`-vector per contact row (`3c + r`).
pub(crate) struct Assembly {
    pub free_qd: Vec<ExprId>,
    pub a: Vec<ExprId>,
    pub b: Vec<ExprId>,
    pub bias: Vec<ExprId>,
    pub r_n: Vec<ExprId>,
    pub active: Vec<ExprId>,
    pub minv_jt: Vec<Vec<ExprId>>,
}

/// The tree-structured half of a contact step (see [`traced_kinematics`]).
pub(crate) struct Kinematics {
    pub free_qd: Vec<ExprId>,
    /// The joint-space mass matrix (CRBA, armature included).
    pub mass: Vec<Vec<ExprId>>,
    pub chol: Cholesky,
    /// Contact rows of `J`, `3c + r` for point `c` and frame row `r`, each `nv`.
    pub jac: Vec<Vec<ExprId>>,
    pub bias: Vec<ExprId>,
    pub dcoef: Vec<ExprId>,
    pub active: Vec<ExprId>,
}

/// Trace the tree-structured half of a contact step: contact-free velocity,
/// the mass matrix's Cholesky factor, contact rows of `J`, and per-point
/// bias, impedance coefficient `(1-d)/d` and activity.
pub(crate) fn traced_kinematics(
    model: &Model,
    points: &[ContactPoint],
    material: &ContactMaterial,
    config: &ContactSolverConfig,
    q: &[ExprId],
    v: &[ExprId],
    ctrl: &[ExprId],
) -> Kinematics {
    let lit = ExprId::from_f64;
    let (np, dt) = (points.len(), model.dt);

    let qfrc = traced_generalized_forces(model, q, v, ctrl);
    let qdd = traced_aba(model, q, v, &qfrc);
    let free_qd = traced_free_velocity(model, v, &qdd, dt);

    let x_tree = traced_x_tree(model, q);
    let x_world = world_transforms(model, &x_tree);
    let mass = traced_crba(model, &x_tree);
    let chol = Cholesky::factor(&mass);

    // Contact frame of the flat ground, constant.
    let (n_c, u_c, w_c) = phyz_contact::contact_frame(&phyz_math::Vec3::new(0.0, 0.0, 1.0));
    let frame = [lift_vec3(n_c), lift_vec3(u_c), lift_vec3(w_c)];

    // Per point: depth, activity mask, 3 x nv Jacobian rows.
    let mut depth = Vec::with_capacity(np);
    let mut active = Vec::with_capacity(np);
    let mut jac: Vec<[Vec<ExprId>; 3]> = Vec::with_capacity(np);
    for pt in points {
        let x = &x_world[pt.body];
        let p = x.pos + x.rot.transpose() * lift_vec3(pt.local);
        let d = -p.z; // n.z * (0 - p.z)
        let at = p + Vec3::new(ExprId::ZERO, ExprId::ZERO, d * lit(0.5));
        let cols = point_jacobian(model, &x_world, pt.body, at);
        jac.push(std::array::from_fn(|r| {
            cols.iter().map(|c| c.dot(frame[r])).collect()
        }));
        active.push(ExprId::select(
            d + lit(material.margin),
            ExprId::ONE,
            ExprId::ZERO,
        ));
        depth.push(d);
    }

    let erp = material.solref.error_reduction(dt);
    let mut bias = Vec::with_capacity(np);
    let mut dcoef = Vec::with_capacity(np);
    for d_i in &depth {
        let d = impedance_at(material, *d_i);
        bias.push(d * lit(erp) * (*d_i).max(ExprId::ZERO) * lit(1.0 / dt));
        let dc = d.clamp(lit(1e-6), ExprId::ONE);
        dcoef.push((ExprId::ONE - dc) / dc);
    }
    let _ = config;
    Kinematics {
        free_qd,
        mass,
        chol,
        jac: jac.into_iter().flatten().collect(),
        bias,
        dcoef,
        active,
    }
}

/// Trace contact detection and assembly for one step (no solve).
pub(crate) fn traced_assembly(
    model: &Model,
    points: &[ContactPoint],
    material: &ContactMaterial,
    config: &ContactSolverConfig,
    q: &[ExprId],
    v: &[ExprId],
    ctrl: &[ExprId],
) -> Assembly {
    let lit = ExprId::from_f64;
    let np = points.len();
    let Kinematics {
        free_qd,
        chol,
        jac,
        bias,
        dcoef,
        active,
        ..
    } = traced_kinematics(model, points, material, config, q, v, ctrl);
    // M⁻¹ Jᵀ, then A = J M⁻¹ Jᵀ and b = J qd_free.
    let minv_jt: Vec<Vec<ExprId>> = jac.iter().map(|j| chol.solve(j)).collect();
    let dot = |a: &[ExprId], b: &[ExprId]| {
        let mut s = ExprId::ZERO;
        for (x, y) in a.iter().zip(b) {
            s += *x * *y;
        }
        s
    };
    let dim = 3 * np;
    let mut a = vec![ExprId::ZERO; dim * dim];
    for i in 0..dim {
        for k in 0..dim {
            a[i * dim + k] = dot(&jac[i], &minv_jt[k]);
        }
    }
    let b: Vec<ExprId> = jac.iter().map(|j| dot(j, &free_qd)).collect();

    let r_n = (0..np)
        .map(|i| (dcoef[i] * a[(3 * i) * dim + 3 * i]).max(lit(config.regularization)))
        .collect();
    Assembly {
        free_qd,
        a,
        b,
        bias,
        r_n,
        active,
        minv_jt,
    }
}

#[allow(clippy::too_many_arguments)]
fn traced_contact_step(
    model: &Model,
    points: &[ContactPoint],
    material: &ContactMaterial,
    config: &ContactSolverConfig,
    sweeps: usize,
    q: &[ExprId],
    v: &[ExprId],
    ctrl: &[ExprId],
    warm: &[ExprId],
) -> Vec<ExprId> {
    let lit = ExprId::from_f64;
    let (np, dt) = (points.len(), model.dt);
    let dim = 3 * np;
    let Assembly {
        free_qd,
        a,
        b,
        bias,
        r_n,
        active,
        minv_jt,
    } = traced_assembly(model, points, material, config, q, v, ctrl);
    let reg = lit(config.regularization);
    let blocks: Vec<[[ExprId; 3]; 3]> = (0..np)
        .map(|i| {
            std::array::from_fn(|r| {
                std::array::from_fn(|c| {
                    let e = a[(3 * i + r) * dim + 3 * i + c];
                    match (r == c, r) {
                        (true, 0) => e + r_n[i],
                        (true, _) => e + reg,
                        _ => e,
                    }
                })
            })
        })
        .collect();

    // Staged-Coulomb PGS from the warm start.
    let mu = lit(material.friction);
    let relax = lit(config.relaxation);
    let mut f: Vec<[ExprId; 3]> = (0..np)
        .map(|i| std::array::from_fn(|r| warm[3 * i + r] * active[i]))
        .collect();
    for _ in 0..sweeps {
        for c in 0..np {
            let base = 3 * c;
            let r: [ExprId; 3] = std::array::from_fn(|row| {
                let mut acc = b[base + row];
                for (k, fk) in f.iter().enumerate() {
                    if k != c {
                        for (col, fkc) in fk.iter().enumerate() {
                            acc += a[(base + row) * dim + 3 * k + col] * *fkc;
                        }
                    }
                }
                acc
            });
            let m = &blocks[c];
            let unclamped = (bias[c] - r[0] - m[0][1] * f[c][1] - m[0][2] * f[c][2]) / m[0][0];
            let f_n = unclamped.max(ExprId::ZERO);
            let r_u = r[1] + m[1][0] * f_n;
            let r_w = r[2] + m[2][0] * f_n;
            let det = m[1][1] * m[2][2] - m[1][2] * m[2][1];
            let t_u = -(m[2][2] * r_u - m[1][2] * r_w) / det;
            let t_w = -(m[1][1] * r_w - m[2][1] * r_u) / det;
            let limit = mu * f_n;
            let t_norm = (t_u * t_u + t_w * t_w).sqrt();
            let safe_limit = ExprId::select(limit, limit, ExprId::ONE);
            let disc = disc_block_step(
                [[m[1][1], m[1][2]], [m[2][1], m[2][2]]],
                [r_u, r_w],
                safe_limit,
            );
            let slide = t_norm - limit;
            let pick = |inside: ExprId, on_disc: ExprId| {
                ExprId::select(slide, ExprId::select(limit, on_disc, ExprId::ZERO), inside)
            };
            let target = [f_n, pick(t_u, disc[0]), pick(t_w, disc[1])];
            f[c] = std::array::from_fn(|k| (f[c][k] + (target[k] - f[c][k]) * relax) * active[c]);
        }
    }

    // v1 = qd_free + M⁻¹ Jᵀ f, then the shared end of the step.
    let mut v1 = free_qd;
    for (col, fr) in minv_jt.iter().zip(f.iter().flatten()) {
        for (vk, x) in v1.iter_mut().zip(col) {
            *vk += *x * *fr;
        }
    }
    let (q1, v1) = traced_finish_step(model, q, v, v1, dt);
    q1.into_iter()
        .chain(v1)
        .chain(f.into_iter().flatten())
        .collect()
}

/// `ContactMaterial::impedance_at`, traced: solimp inside, a smoothstep ramp
/// across the margin outside.
fn impedance_at(m: &ContactMaterial, depth: ExprId) -> ExprId {
    let lit = ExprId::from_f64;
    let s = &m.solimp;
    let dmin = s.dmin.clamp(1e-4, 1.0 - 1e-9);
    let dmax = s.dmax.clamp(1e-4, 1.0 - 1e-9);
    let inside = if s.width <= 0.0 {
        lit(dmax)
    } else {
        let x = (depth.abs() * lit(1.0 / s.width)).clamp(ExprId::ZERO, ExprId::ONE);
        let mid = s.midpoint.clamp(1e-6, 1.0 - 1e-6);
        let p = s.power.max(1.0);
        let pow = |b: ExprId| b.powf(lit(p));
        let lo = pow(x) * lit(1.0 / mid.powf(p - 1.0));
        let hi = ExprId::ONE - pow(ExprId::ONE - x) * lit(1.0 / (1.0 - mid).powf(p - 1.0));
        let y = ExprId::select(x - lit(mid), hi, lo);
        lit(dmin) + y * lit(dmax - dmin)
    };
    let d0 = s.impedance(0.0);
    let outside = if m.margin > 0.0 {
        let sm = ExprId::ONE + depth * lit(1.0 / m.margin);
        let ramp = sm * sm * (lit(3.0) - lit(2.0) * sm);
        ExprId::select(sm, ramp * lit(d0), ExprId::ZERO)
    } else {
        ExprId::ZERO
    };
    // depth >= 0 takes solimp: `-depth > 0` is exactly `depth < 0`.
    ExprId::select(-depth, outside, inside)
}

/// `convex::disc_block_step` with a fixed Newton count: the minimizer of the
/// 2x2 tangential block on the friction disc `|t| = limit`.
fn disc_block_step(m: [[ExprId; 2]; 2], r: [ExprId; 2], limit: ExprId) -> [ExprId; 2] {
    let solve = |k: ExprId, rhs: [ExprId; 2]| {
        let (a, b, c, d) = (m[0][0] + k, m[0][1], m[1][0], m[1][1] + k);
        let det = a * d - b * c;
        [
            (d * rhs[0] - b * rhs[1]) / det,
            (a * rhs[1] - c * rhs[0]) / det,
        ]
    };
    let mut k = ExprId::ZERO;
    let mut t = solve(k, [-r[0], -r[1]]);
    for _ in 0..DISC_ITERS {
        let nrm = (t[0] * t[0] + t[1] * t[1]).sqrt();
        let st = solve(k, t);
        let dn = -(t[0] * st[0] + t[1] * st[1]) / nrm;
        let phi = nrm.recip() - limit.recip();
        let dphi = -dn / (nrm * nrm);
        k = (k - phi / dphi).max(ExprId::ZERO);
        t = solve(k, [-r[0], -r[1]]);
    }
    let s = limit / (t[0] * t[0] + t[1] * t[1]).sqrt();
    [t[0] * s, t[1] * s]
}

/// Parent→child Plücker transforms, as `traced_aba` builds them.
fn traced_x_tree(model: &Model, q: &[ExprId]) -> Vec<SpatialTransform<ExprId>> {
    model
        .bodies
        .iter()
        .map(|body| {
            let joint = &model.joints[body.joint_idx];
            let qi = model.q_offsets[body.joint_idx];
            let x_joint = match joint.ndof() {
                0 => SpatialTransform::identity(),
                1 => sym_joint_transform(joint.joint_type, &lift_vec3(joint.axis), q[qi]),
                n => multi_joint_transform(joint.joint_type, &q[qi..qi + n]),
            };
            x_joint.compose(&lift_spatial_transform(&joint.parent_to_joint))
        })
        .collect()
}

/// World→body transforms (`phyz_rigid::forward_kinematics`).
fn world_transforms(
    model: &Model,
    x_tree: &[SpatialTransform<ExprId>],
) -> Vec<SpatialTransform<ExprId>> {
    let mut out: Vec<SpatialTransform<ExprId>> = Vec::with_capacity(x_tree.len());
    for (i, body) in model.bodies.iter().enumerate() {
        let x = if body.parent < 0 {
            x_tree[i]
        } else {
            x_tree[i].compose(&out[body.parent as usize])
        };
        out.push(x);
    }
    out
}

/// `phyz_rigid::point_jacobian`: one world-frame column per velocity DOF.
fn point_jacobian(
    model: &Model,
    x_world: &[SpatialTransform<ExprId>],
    body: usize,
    point: Vec3<ExprId>,
) -> Vec<Vec3<ExprId>> {
    use phyz_model::JointType;
    let mut cols = vec![Vec3::zero(); model.nv];
    let mut cur = body as i32;
    while cur >= 0 {
        let b = &model.bodies[cur as usize];
        let joint = &model.joints[b.joint_idx];
        let vi = model.v_offsets[b.joint_idx];
        let xf = &x_world[cur as usize];
        let to_world = |d: Vec3<ExprId>| xf.rot.transpose() * d;
        let arm = point - xf.pos;
        let unit = |k: usize| {
            let mut e = [ExprId::ZERO; 3];
            e[k] = ExprId::ONE;
            to_world(Vec3::new(e[0], e[1], e[2]))
        };
        for d in 0..joint.ndof() {
            cols[vi + d] = match joint.joint_type {
                JointType::Revolute | JointType::Hinge => {
                    to_world(lift_vec3(joint.axis)).cross(arm)
                }
                JointType::Prismatic | JointType::Slide => to_world(lift_vec3(joint.axis)),
                JointType::Spherical | JointType::Ball => unit(d).cross(arm),
                JointType::Free if d < 3 => unit(d).cross(arm),
                JointType::Free => unit(d - 3),
                JointType::Fixed => Vec3::zero(),
            };
        }
        cur = b.parent;
    }
    cols
}

/// `phyz_rigid::crba` (with armature), traced. Multi-DOF joints only at the root.
fn traced_crba(model: &Model, x_tree: &[SpatialTransform<ExprId>]) -> Vec<Vec<ExprId>> {
    let nb = model.nbodies();
    let nv = model.nv;
    let mut m = vec![vec![ExprId::ZERO; nv]; nv];
    let mut i_c: Vec<SpatialMat<ExprId>> = model
        .bodies
        .iter()
        .map(|b| lift_spatial_inertia(&b.inertia).to_matrix())
        .collect();
    for i in (0..nb).rev() {
        let p = model.bodies[i].parent;
        if p >= 0 {
            let x_mot = x_tree[i].to_motion_matrix();
            let add = x_mot.transpose().mul_mat(&i_c[i]).mul_mat(&x_mot);
            i_c[p as usize] = i_c[p as usize] + add;
        }
    }
    // Columns of S for joint `j`, as spatial vectors.
    let subspace = |j: &phyz_model::Joint| -> Vec<SpatialVec<ExprId>> {
        match j.ndof() {
            0 => vec![],
            1 => vec![motion_subspace_1(j)],
            _ => motion_subspace_n(j)
                .iter()
                .map(|c| {
                    let e = |k: usize| ExprId::from_f64(c[k]);
                    SpatialVec::new(Vec3::new(e(0), e(1), e(2)), Vec3::new(e(3), e(4), e(5)))
                })
                .collect(),
        }
    };
    for i in 0..nb {
        let joint_i = &model.joints[model.bodies[i].joint_idx];
        let vi = model.v_offsets[model.bodies[i].joint_idx];
        let s_i = subspace(joint_i);
        if s_i.len() > 1 {
            assert!(
                ancestors_fixed(model, i),
                "CRBA trace: multi-DOF joint must be at the root"
            );
        }
        for (ki, s) in s_i.iter().enumerate() {
            let f_i = i_c[i].mul_vec(s);
            for (kj, s2) in s_i.iter().enumerate() {
                m[vi + ki][vi + kj] = s2.dot(&f_i);
            }
            let mut f = x_tree[i].inv_apply_force(&f_i);
            let mut j = model.bodies[i].parent;
            while j >= 0 {
                let ju = j as usize;
                let joint_j = &model.joints[model.bodies[ju].joint_idx];
                let vj = model.v_offsets[model.bodies[ju].joint_idx];
                for (kj, s_j) in subspace(joint_j).iter().enumerate() {
                    let val = s_j.dot(&f);
                    m[vi + ki][vj + kj] = val;
                    m[vj + kj][vi + ki] = val;
                }
                f = x_tree[ju].inv_apply_force(&f);
                j = model.bodies[ju].parent;
            }
        }
        for k in 0..s_i.len() {
            m[vi + k][vi + k] += ExprId::from_f64(joint_i.armature);
        }
    }
    m
}

/// Parent DOF of each velocity DOF in the kinematic tree (`-1` at a root):
/// the previous DOF of the same joint, else the last DOF of the nearest
/// ancestor body that has any. Parents always precede children.
pub(crate) fn dof_parents(model: &Model) -> Vec<i32> {
    let mut lambda = vec![-1i32; model.nv];
    let mut last_dof = vec![-1i32; model.nbodies()];
    for (i, body) in model.bodies.iter().enumerate() {
        let joint = &model.joints[body.joint_idx];
        let vi = model.v_offsets[body.joint_idx];
        let inherited = if body.parent < 0 {
            -1
        } else {
            last_dof[body.parent as usize]
        };
        let mut prev = inherited;
        for k in 0..joint.ndof() {
            lambda[vi + k] = prev;
            prev = (vi + k) as i32;
        }
        last_dof[i] = prev;
    }
    lambda
}

/// Featherstone's LTL factorization `M = Lᵀ L` (RBDA §6.5), traced. In this
/// DOF order `L` has no fill-in: `L[k][i] ≠ 0` only for `i` an ancestor of
/// `k`. Returns `L` (lower triangle; other entries zero) and `1 / L[k][k]`.
pub(crate) fn traced_ltl(mass: &[Vec<ExprId>], lambda: &[i32]) -> (Vec<Vec<ExprId>>, Vec<ExprId>) {
    let n = mass.len();
    let mut h = mass.to_vec();
    let mut inv = vec![ExprId::ZERO; n];
    for k in (0..n).rev() {
        h[k][k] = h[k][k].sqrt();
        inv[k] = h[k][k].recip();
        let mut i = lambda[k];
        while i >= 0 {
            let iu = i as usize;
            h[k][iu] *= inv[k];
            i = lambda[iu];
        }
        let mut i = lambda[k];
        while i >= 0 {
            let iu = i as usize;
            let mut j = i;
            while j >= 0 {
                let ju = j as usize;
                let d = h[k][iu] * h[k][ju];
                h[iu][ju] -= d;
                j = lambda[ju];
            }
            i = lambda[iu];
        }
    }
    let l = (0..n)
        .map(|k| {
            let mut row = vec![ExprId::ZERO; n];
            row[k] = h[k][k];
            let mut i = lambda[k];
            while i >= 0 {
                row[i as usize] = h[k][i as usize];
                i = lambda[i as usize];
            }
            row
        })
        .collect();
    (l, inv)
}

/// Unrolled Cholesky `M = L Lᵀ`, reused across right-hand sides.
pub(crate) struct Cholesky {
    pub l: Vec<Vec<ExprId>>,
    pub inv_diag: Vec<ExprId>,
}

impl Cholesky {
    fn factor(d: &[Vec<ExprId>]) -> Self {
        let n = d.len();
        let mut l = vec![vec![ExprId::ZERO; n]; n];
        let mut inv_diag = vec![ExprId::ZERO; n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = d[i][j];
                for (lik, ljk) in l[i][..j].iter().zip(&l[j][..j]) {
                    sum -= *lik * *ljk;
                }
                if i == j {
                    l[i][i] = sum.sqrt();
                    inv_diag[i] = l[i][i].recip();
                } else {
                    l[i][j] = sum * inv_diag[j];
                }
            }
        }
        Self { l, inv_diag }
    }

    // Triangular solves read `l` and the partial result at the same index.
    #[allow(clippy::needless_range_loop)]
    fn solve(&self, b: &[ExprId]) -> Vec<ExprId> {
        let n = b.len();
        let mut y = vec![ExprId::ZERO; n];
        for i in 0..n {
            let mut sum = b[i];
            for k in 0..i {
                sum -= self.l[i][k] * y[k];
            }
            y[i] = sum * self.inv_diag[i];
        }
        let mut x = vec![ExprId::ZERO; n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for k in i + 1..n {
                sum -= self.l[k][i] * x[k];
            }
            x[i] = sum * self.inv_diag[i];
        }
        x
    }
}
