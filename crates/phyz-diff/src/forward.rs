//! Forward dynamics (ABA) traced into a tang-expr graph.
//!
//! [`ForwardDynamics::build`] unrolls the Articulated Body Algorithm for one
//! model's fixed kinematic tree into an [`ExprGraph`]. The graph maps
//! `(q, v, qfrc)` to `qdd` and can be evaluated in f64
//! ([`ForwardDynamics::compile`]) or emitted as a WGSL/MSL/CUDA/C kernel
//! ([`ForwardDynamics::kernel`]) that steps one environment per work item.
//!
//! Unlike [`crate::symbolic`], which traces a simplified ABA for Jacobians, this
//! mirrors `phyz_rigid::aba` term for term:
//!
//! - generalized forces come in as an input (`qfrc`, from
//!   `phyz_rigid::actuation::generalized_forces`), so actuator clamping and
//!   passive terms stay on the host;
//! - armature and the implicit-damping term `dt * damping` enter `D`;
//! - multi-DOF joints (free, spherical) are supported at the root of the tree,
//!   which covers floating-base robots. `D` is solved by an unrolled Cholesky
//!   rather than `phyz_rigid`'s LU inverse, so results agree to round-off.
//!
//! # Input layout
//!
//! `inputs = [q (nq), v (nv), qfrc (nv)]`; outputs are `qdd (nv)`.

use phyz_model::{JointType, Model, State};
use tang::Scalar;
use tang::{Mat3, Quat, SpatialMat, SpatialTransform, SpatialVec, Vec3};
use tang_expr::codegen::{ComputeKernel, Dialect};
use tang_expr::compile::CompiledMany;
use tang_expr::{ExprGraph, ExprId, trace};

use crate::symbolic::{
    lift_spatial_inertia, lift_spatial_transform, lift_vec3, sym_joint_transform,
    sym_outer_product_6,
};

/// ABA for one model, as an expression graph.
pub struct ForwardDynamics {
    /// The traced graph.
    pub graph: ExprGraph,
    /// Output expressions, one per velocity DOF.
    pub qdd: Vec<ExprId>,
    /// Position DOFs.
    pub nq: usize,
    /// Velocity DOFs.
    pub nv: usize,
}

impl ForwardDynamics {
    /// Trace ABA for `model`.
    ///
    /// Panics if a multi-DOF joint has an ancestor with DOFs: only a
    /// floating (or ball) base is supported.
    pub fn build(model: &Model) -> Self {
        let (nq, nv) = (model.nq, model.nv);
        let (graph, qdd) = trace(|| {
            let q: Vec<ExprId> = (0..nq).map(|i| ExprId::var(i as u16)).collect();
            let v: Vec<ExprId> = (0..nv).map(|i| ExprId::var((nq + i) as u16)).collect();
            let qfrc: Vec<ExprId> = (0..nv).map(|i| ExprId::var((nq + nv + i) as u16)).collect();
            traced_aba(model, &q, &v, &qfrc)
        });
        let (graph, qdd) = fold_constants(&graph, &qdd);
        Self { graph, qdd, nq, nv }
    }

    /// Number of inputs per evaluation: `nq + 2 nv`.
    pub fn n_inputs(&self) -> usize {
        self.nq + 2 * self.nv
    }

    /// Pack `(q, v, qfrc)` into the graph's input layout.
    pub fn inputs(&self, state: &State, qfrc: &[f64]) -> Vec<f64> {
        let mut x = Vec::with_capacity(self.n_inputs());
        x.extend_from_slice(state.q.as_slice());
        x.extend_from_slice(state.v.as_slice());
        x.extend_from_slice(qfrc);
        x
    }

    /// Live nodes in the graph (the work per evaluation).
    pub fn live_nodes(&self) -> usize {
        self.graph.live_set(&self.qdd).len()
    }

    /// Compile to an f64 closure `(inputs, qdd)`.
    pub fn compile(&self) -> CompiledMany {
        self.graph.compile_many(&self.qdd)
    }

    /// Emit a batched kernel: one environment per work item.
    pub fn kernel(&self, dialect: Dialect) -> ComputeKernel {
        self.graph.to_kernel(&self.qdd, self.n_inputs(), dialect)
    }

    /// Emit a batched plain-C kernel in f64.
    pub fn kernel_c_f64(&self) -> ComputeKernel {
        self.graph.to_c_f64(&self.qdd, self.n_inputs())
    }
}

/// One full rigid-body step, contact-free, as an expression graph.
///
/// Mirrors `phyz_rigid::actuation::generalized_forces` → `phyz_rigid::aba` →
/// `phyz_rigid::semi_implicit_euler`: actuator laws with their clamps, passive
/// joint terms (damping, spring, soft limits, smoothed friction), forward
/// dynamics, then the joint-aware integrator (free-joint Coriolis strip, frame
/// turn, and the Lie-group update of exp coordinates).
///
/// Inputs `[q (nq), v (nv), ctrl (nu)]`; outputs `[q' (nq), v' (nv)]`.
pub struct Step {
    /// The traced graph.
    pub graph: ExprGraph,
    /// Outputs: `q'` then `v'`.
    pub out: Vec<ExprId>,
    /// Position DOFs.
    pub nq: usize,
    /// Velocity DOFs.
    pub nv: usize,
    /// Controls (`State::ctrl` length).
    pub nu: usize,
}

impl Step {
    /// Trace one step of `model` at `model.dt`.
    pub fn build(model: &Model) -> Self {
        let (nq, nv) = (model.nq, model.nv);
        let nu = model.default_state().ctrl.len();
        let (graph, out) = trace(|| {
            let var = |i: usize| ExprId::var(i as u16);
            let q: Vec<ExprId> = (0..nq).map(var).collect();
            let v: Vec<ExprId> = (nq..nq + nv).map(var).collect();
            let ctrl: Vec<ExprId> = (nq + nv..nq + nv + nu).map(var).collect();
            let qfrc = traced_generalized_forces(model, &q, &v, &ctrl);
            let qdd = traced_aba(model, &q, &v, &qfrc);
            let (q1, v1) = traced_semi_implicit_euler(model, &q, &v, &qdd, model.dt);
            q1.into_iter().chain(v1).collect::<Vec<_>>()
        });
        let (graph, out) = fold_constants(&graph, &out);
        Self {
            graph,
            out,
            nq,
            nv,
            nu,
        }
    }

    /// Inputs per evaluation: `nq + nv + nu`.
    pub fn n_inputs(&self) -> usize {
        self.nq + self.nv + self.nu
    }

    /// Pack a state into the graph's input layout.
    pub fn inputs(&self, state: &State) -> Vec<f64> {
        let mut x = Vec::with_capacity(self.n_inputs());
        x.extend_from_slice(state.q.as_slice());
        x.extend_from_slice(state.v.as_slice());
        x.extend_from_slice(state.ctrl.as_slice());
        x
    }

    /// Live nodes in the graph.
    pub fn live_nodes(&self) -> usize {
        self.graph.live_set(&self.out).len()
    }

    /// Compile to an f64 closure `(inputs, [q', v'])`.
    pub fn compile(&self) -> CompiledMany {
        self.graph.compile_many(&self.out)
    }

    /// Emit a batched kernel: one environment per work item.
    pub fn kernel(&self, dialect: Dialect) -> ComputeKernel {
        self.graph.to_kernel(&self.out, self.n_inputs(), dialect)
    }

    /// Emit a batched plain-C kernel in f64.
    pub fn kernel_c_f64(&self) -> ComputeKernel {
        self.graph.to_c_f64(&self.out, self.n_inputs())
    }
}

/// `phyz_rigid::actuation::generalized_forces`, traced.
pub(crate) fn traced_generalized_forces(
    model: &Model,
    q: &[ExprId],
    v: &[ExprId],
    ctrl: &[ExprId],
) -> Vec<ExprId> {
    let lit = ExprId::from_f64;
    let mut qfrc = vec![ExprId::ZERO; model.nv];
    if model.actuators.is_empty() {
        let n = model.nv.min(ctrl.len());
        qfrc[..n].copy_from_slice(&ctrl[..n]);
    } else {
        for (a, act) in model.actuators.iter().enumerate().take(ctrl.len()) {
            let (Some(&vi), Some(&qi)) = (
                model.v_offsets.get(act.joint_idx),
                model.q_offsets.get(act.joint_idx),
            ) else {
                continue;
            };
            if vi >= model.nv || qi >= q.len() {
                continue;
            }
            let c = match act.ctrl_range {
                Some([lo, hi]) => ctrl[a].clamp(lit(lo.min(hi)), lit(hi.max(lo))),
                None => ctrl[a],
            };
            let f = lit(act.gear)
                * (lit(act.gain) * c + lit(act.bias_q) * q[qi] + lit(act.bias_v) * v[vi]);
            let f = match act.force_range {
                Some([lo, hi]) => f.clamp(lit(lo.min(hi)), lit(hi.max(lo))),
                None => f,
            };
            qfrc[vi] += f;
        }
    }
    for (j, joint) in model.joints.iter().enumerate() {
        let (qo, vo) = (model.q_offsets[j], model.v_offsets[j]);
        for k in 0..joint.ndof() {
            let (qk, vk) = (q[qo + k], v[vo + k]);
            let mut f = -(lit(joint.damping) * vk);
            if joint.ndof() == 1 {
                f += -(lit(joint.stiffness) * (qk - lit(joint.spring_ref)));
                if let Some([lo, hi]) = joint.limits {
                    let (ks, kd) = (lit(joint.limit_stiffness), lit(joint.limit_damping));
                    let below = lit(lo) - qk;
                    let above = qk - lit(hi);
                    let push_up = ks * below + kd * (-vk).max(ExprId::ZERO);
                    let push_down = -(ks * above + kd * vk.max(ExprId::ZERO));
                    f += ExprId::select(
                        below,
                        push_up,
                        ExprId::select(above, push_down, ExprId::ZERO),
                    );
                }
                if joint.friction_loss > 0.0 {
                    f += -(lit(joint.friction_loss) * (vk * lit(1e3)).tanh());
                }
            }
            qfrc[vo + k] += f;
        }
    }
    qfrc
}

/// `phyz_rigid::semi_implicit_euler`, traced (without `PHYZ_INTEG_CURRENT`).
fn traced_semi_implicit_euler(
    model: &Model,
    q: &[ExprId],
    v: &[ExprId],
    qdd: &[ExprId],
    dt: f64,
) -> (Vec<ExprId>, Vec<ExprId>) {
    let v1 = traced_free_velocity(model, v, qdd, dt);
    traced_finish_step(model, q, v, v1, dt)
}

/// `v + dt * qdd` with the free-joint Coriolis term stripped first
/// (`phyz_rigid::strip_free_joint_coriolis`): the contact-free velocity.
pub(crate) fn traced_free_velocity(
    model: &Model,
    v: &[ExprId],
    qdd: &[ExprId],
    dt: f64,
) -> Vec<ExprId> {
    let lit = ExprId::from_f64;
    let mut qdd = qdd.to_vec();
    for (j, joint) in model.joints.iter().enumerate() {
        if joint.joint_type == JointType::Free {
            let o = model.v_offsets[j];
            let w = Vec3::new(v[o], v[o + 1], v[o + 2]);
            let c = w.cross(Vec3::new(v[o + 3], v[o + 4], v[o + 5]));
            qdd[o + 3] += c.x;
            qdd[o + 4] += c.y;
            qdd[o + 5] += c.z;
        }
    }
    (0..model.nv).map(|i| v[i] + lit(dt) * qdd[i]).collect()
}

/// `rotate_free_joint_velocities` then `integrate_configuration`: from the
/// step's updated velocity `v1` (in the start frame) to `(q', v')`.
pub(crate) fn traced_finish_step(
    model: &Model,
    q: &[ExprId],
    v: &[ExprId],
    mut v1: Vec<ExprId>,
    dt: f64,
) -> (Vec<ExprId>, Vec<ExprId>) {
    let lit = ExprId::from_f64;
    for (j, joint) in model.joints.iter().enumerate() {
        if joint.joint_type == JointType::Free {
            let o = model.v_offsets[j];
            let w = Vec3::new(v[o], v[o + 1], v[o + 2]);
            let theta = (w.x * w.x + w.y * w.y + w.z * w.z).sqrt() * lit(dt);
            let lin = Vec3::new(v1[o + 3], v1[o + 4], v1[o + 5]);
            let turned = quat_exp(w * lit(-dt)).rotate(lin);
            let on = theta - lit(1e-12);
            v1[o + 3] = ExprId::select(on, turned.x, lin.x);
            v1[o + 4] = ExprId::select(on, turned.y, lin.y);
            v1[o + 5] = ExprId::select(on, turned.z, lin.z);
        }
    }
    let mut q1 = q.to_vec();
    for (j, joint) in model.joints.iter().enumerate() {
        let (qo, vo) = (model.q_offsets[j], model.v_offsets[j]);
        match joint.joint_type {
            JointType::Fixed => {}
            JointType::Revolute | JointType::Hinge | JointType::Prismatic | JointType::Slide => {
                q1[qo] = q[qo] + lit(dt) * v1[vo];
            }
            JointType::Spherical | JointType::Ball | JointType::Free => {
                let w = Vec3::new(v1[vo], v1[vo + 1], v1[vo + 2]);
                let cur = quat_exp(Vec3::new(q[qo], q[qo + 1], q[qo + 2]));
                let next = cur.mul(&quat_exp(w * lit(dt))).normalize();
                if joint.joint_type == JointType::Free {
                    let lin = next.rotate(Vec3::new(v1[vo + 3], v1[vo + 4], v1[vo + 5]));
                    q1[qo + 3] = q[qo + 3] + lit(dt) * lin.x;
                    q1[qo + 4] = q[qo + 4] + lit(dt) * lin.y;
                    q1[qo + 5] = q[qo + 5] + lit(dt) * lin.z;
                }
                let log = quat_log(next);
                q1[qo] = log.x;
                q1[qo + 1] = log.y;
                q1[qo + 2] = log.z;
            }
        }
    }
    (q1, v1)
}

/// `phyz_math::quat_log` with its small-angle branch as a select.
pub(crate) fn quat_log(q: Quat<ExprId>) -> Vec3<ExprId> {
    let n = (q.v.x * q.v.x + q.v.y * q.v.y + q.v.z * q.v.z).sqrt();
    let big = n - ExprId::from_f64(f64::EPSILON);
    let safe = ExprId::select(big, n, ExprId::ONE);
    let k = ExprId::from_f64(2.0) * safe.atan2(q.w) / safe;
    q.v * ExprId::select(big, k, ExprId::from_f64(2.0))
}

type Sym6 = [[ExprId; 6]; 6];

pub(crate) fn traced_aba(
    model: &Model,
    q: &[ExprId],
    v: &[ExprId],
    qfrc: &[ExprId],
) -> Vec<ExprId> {
    let nb = model.nbodies();
    let zero_sv: SpatialVec<ExprId> = SpatialVec::zero();

    let mut x_tree: Vec<SpatialTransform<ExprId>> = vec![SpatialTransform::identity(); nb];
    let mut vel = vec![zero_sv; nb];
    let mut c_bias = vec![zero_sv; nb];
    let mut p_a = vec![zero_sv; nb];
    let mut i_a: Vec<SpatialMat<ExprId>> = vec![SpatialMat::zero(); nb];

    let a0: SpatialVec<ExprId> = SpatialVec::new(Vec3::zero(), -lift_vec3(model.gravity));

    // Constant per-DOF diagonal added to D: armature + dt * damping.
    let d_diag = |j: &phyz_model::Joint| ExprId::from_f64(j.armature + model.dt * j.damping);

    // -- Pass 1: velocities and bias --
    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let q_idx = model.q_offsets[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let ndof = joint.ndof();

        if ndof > 1 {
            assert!(
                ancestors_fixed(model, i),
                "forward dynamics graph: multi-DOF joint `{}` must be at the tree root",
                joint.name
            );
        }

        let x_joint = match ndof {
            0 => SpatialTransform::identity(),
            1 => sym_joint_transform(joint.joint_type, &lift_vec3(joint.axis), q[q_idx]),
            _ => multi_joint_transform(joint.joint_type, &q[q_idx..q_idx + ndof]),
        };
        x_tree[i] = x_joint.compose(&lift_spatial_transform(&joint.parent_to_joint));

        let v_joint = joint_velocity(joint, &v[v_idx..v_idx + ndof]);
        if body.parent < 0 {
            vel[i] = v_joint;
        } else {
            let pi = body.parent as usize;
            vel[i] = x_tree[i].apply_motion(&vel[pi]) + v_joint;
            c_bias[i] = vel[i].cross_motion(&v_joint);
        }

        i_a[i] = lift_spatial_inertia(&body.inertia).to_matrix();
        p_a[i] = vel[i].cross_force(&i_a[i].mul_vec(&vel[i]));
    }

    // -- Pass 2: articulated inertias and bias forces --
    for i in (0..nb).rev() {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let ndof = joint.ndof();
        if body.parent < 0 {
            continue;
        }
        let pi = body.parent as usize;

        let (ia_new, p_new) = match ndof {
            0 => (i_a[i], p_a[i]),
            1 => {
                let s = motion_subspace_1(joint);
                let ia = i_a[i];
                let ia_s = ia.mul_vec(&s);
                let d = s.dot(&ia_s) + d_diag(joint);
                let d_inv = d.recip();
                let u = qfrc[v_idx] - s.dot(&p_a[i]);
                let ia_new = ia - sym_outer_product_6(&ia_s, &ia_s) * d_inv;
                let p_new = p_a[i] + ia_new.mul_vec(&c_bias[i]) + ia_s * (u * d_inv);
                (ia_new, p_new)
            }
            // A root multi-DOF joint's parent is fixed to the world: nothing
            // downstream reads what would be propagated.
            _ => continue,
        };

        let x_mot = x_tree[i].to_motion_matrix();
        i_a[pi] = i_a[pi] + x_mot.transpose().mul_mat(&ia_new).mul_mat(&x_mot);
        p_a[pi] = p_a[pi] + x_tree[i].inv_apply_force(&p_new);
    }

    // -- Pass 3: accelerations --
    let mut acc = vec![zero_sv; nb];
    let mut qdd = vec![ExprId::ZERO; model.nv];
    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let ndof = joint.ndof();

        let a_parent = if body.parent < 0 {
            x_tree[i].apply_motion(&a0)
        } else {
            x_tree[i].apply_motion(&acc[body.parent as usize])
        };
        let a_total = a_parent + c_bias[i];

        match ndof {
            0 => acc[i] = a_total,
            1 => {
                let s = motion_subspace_1(joint);
                let ia = i_a[i];
                let d = s.dot(&ia.mul_vec(&s)) + d_diag(joint);
                let u = qfrc[v_idx] - s.dot(&p_a[i]);
                let qdd_i = (u - ia.mul_vec(&a_total).dot(&s)) * d.recip();
                qdd[v_idx] = qdd_i;
                acc[i] = a_total + s * qdd_i;
            }
            _ => {
                let s = motion_subspace_n(joint); // 6 x ndof, constant
                let ia = to_sym6(&i_a[i]);
                let pa = sv_arr(&p_a[i]);
                let ia_a = mat6_vec(&ia, &sv_arr(&a_total));
                // U = I_a S (6 x ndof)
                let u_mat: Vec<[ExprId; 6]> = (0..ndof)
                    .map(|k| std::array::from_fn(|r| dot6_const(&ia[r], &s[k])))
                    .collect();
                let mut d = vec![vec![ExprId::ZERO; ndof]; ndof];
                for a in 0..ndof {
                    for b in 0..ndof {
                        d[a][b] = dot6_const(&u_mat[b], &s[a]);
                    }
                    d[a][a] += d_diag(joint);
                }
                let rhs: Vec<ExprId> = (0..ndof)
                    .map(|k| qfrc[v_idx + k] - dot6_const(&pa, &s[k]) - dot6_const(&ia_a, &s[k]))
                    .collect();
                let x = cholesky_solve(&d, &rhs);
                let mut acc_arr = sv_arr(&a_total);
                for k in 0..ndof {
                    qdd[v_idx + k] = x[k];
                    for r in 0..6 {
                        if s[k][r] != 0.0 {
                            acc_arr[r] += x[k] * ExprId::from_f64(s[k][r]);
                        }
                    }
                }
                acc[i] = arr_sv(&acc_arr);
            }
        }
    }
    qdd
}

pub(crate) fn ancestors_fixed(model: &Model, i: usize) -> bool {
    let mut p = model.bodies[i].parent;
    while p >= 0 {
        let b = &model.bodies[p as usize];
        if model.joints[b.joint_idx].ndof() != 0 {
            return false;
        }
        p = b.parent;
    }
    true
}

pub(crate) fn motion_subspace_1(joint: &phyz_model::Joint) -> SpatialVec<ExprId> {
    let axis = lift_vec3(joint.axis);
    match joint.joint_type {
        JointType::Revolute | JointType::Hinge => SpatialVec::new(axis, Vec3::zero()),
        JointType::Prismatic | JointType::Slide => SpatialVec::new(Vec3::zero(), axis),
        t => panic!("not a single-DOF joint: {t:?}"),
    }
}

/// Columns of the constant motion subspace, each a 6-vector `[ang; lin]`.
pub(crate) fn motion_subspace_n(joint: &phyz_model::Joint) -> Vec<[f64; 6]> {
    let s = joint.motion_subspace_matrix();
    (0..s.ncols())
        .map(|k| std::array::from_fn(|r| s[(r, k)]))
        .collect()
}

pub(crate) fn joint_velocity(joint: &phyz_model::Joint, qd: &[ExprId]) -> SpatialVec<ExprId> {
    let axis = lift_vec3(joint.axis);
    match joint.joint_type {
        JointType::Revolute | JointType::Hinge => SpatialVec::new(axis * qd[0], Vec3::zero()),
        JointType::Prismatic | JointType::Slide => SpatialVec::new(Vec3::zero(), axis * qd[0]),
        JointType::Spherical | JointType::Ball => {
            SpatialVec::new(Vec3::new(qd[0], qd[1], qd[2]), Vec3::zero())
        }
        JointType::Free => SpatialVec::new(
            Vec3::new(qd[0], qd[1], qd[2]),
            Vec3::new(qd[3], qd[4], qd[5]),
        ),
        JointType::Fixed => SpatialVec::zero(),
    }
}

/// Parent→child coordinate map of a ball or free joint: `exp(w)ᵀ`, plus the
/// free joint's translation. Mirrors `Joint::joint_transform_slice`.
pub(crate) fn multi_joint_transform(
    joint_type: JointType,
    q: &[ExprId],
) -> SpatialTransform<ExprId> {
    let rot = quat_exp(Vec3::new(q[0], q[1], q[2]))
        .to_matrix()
        .transpose();
    match joint_type {
        JointType::Spherical | JointType::Ball => SpatialTransform::new(rot, Vec3::zero()),
        JointType::Free => SpatialTransform::new(rot, Vec3::new(q[3], q[4], q[5])),
        t => panic!("not a multi-DOF joint: {t:?}"),
    }
}

/// `phyz_math::quat_exp` with its small-angle branch as a select.
pub(crate) fn quat_exp(w: Vec3<ExprId>) -> Quat<ExprId> {
    let angle = (w.x * w.x + w.y * w.y + w.z * w.z).sqrt();
    let big = angle - ExprId::from_f64(f64::EPSILON);
    let safe = ExprId::select(big, angle, ExprId::ONE);
    let (s, c) = (safe * ExprId::from_f64(0.5)).sin_cos();
    let k = ExprId::select(big, s / safe, ExprId::from_f64(0.5));
    let wq = ExprId::select(big, c, ExprId::ONE);
    Quat { w: wq, v: w * k }
}

/// Solve `D x = b` for symmetric positive definite `D`, unrolled.
pub(crate) fn cholesky_solve(d: &[Vec<ExprId>], b: &[ExprId]) -> Vec<ExprId> {
    let n = b.len();
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
    let mut y = vec![ExprId::ZERO; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum * inv_diag[i];
    }
    let mut x = vec![ExprId::ZERO; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in i + 1..n {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum * inv_diag[i];
    }
    x
}

fn mat3_get(m: &Mat3<ExprId>, r: usize, c: usize) -> ExprId {
    let col = match c {
        0 => m.c0,
        1 => m.c1,
        _ => m.c2,
    };
    match r {
        0 => col.x,
        1 => col.y,
        _ => col.z,
    }
}

pub(crate) fn to_sym6(m: &SpatialMat<ExprId>) -> Sym6 {
    std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            let blk = match (r < 3, c < 3) {
                (true, true) => &m.upper_left,
                (true, false) => &m.upper_right,
                (false, true) => &m.lower_left,
                (false, false) => &m.lower_right,
            };
            mat3_get(blk, r % 3, c % 3)
        })
    })
}

pub(crate) fn sv_arr(v: &SpatialVec<ExprId>) -> [ExprId; 6] {
    [
        v.angular.x,
        v.angular.y,
        v.angular.z,
        v.linear.x,
        v.linear.y,
        v.linear.z,
    ]
}

pub(crate) fn arr_sv(a: &[ExprId; 6]) -> SpatialVec<ExprId> {
    SpatialVec::new(Vec3::new(a[0], a[1], a[2]), Vec3::new(a[3], a[4], a[5]))
}

fn mat6_vec(m: &Sym6, x: &[ExprId; 6]) -> [ExprId; 6] {
    std::array::from_fn(|r| {
        let mut s = ExprId::ZERO;
        for c in 0..6 {
            s += m[r][c] * x[c];
        }
        s
    })
}

/// `a · s` for a constant `s`, skipping its zeros.
pub(crate) fn dot6_const(a: &[ExprId; 6], s: &[f64; 6]) -> ExprId {
    let mut sum = ExprId::ZERO;
    for r in 0..6 {
        if s[r] == 1.0 {
            sum += a[r];
        } else if s[r] != 0.0 {
            sum += a[r] * ExprId::from_f64(s[r]);
        }
    }
    sum
}

/// Rebuild `graph` keeping only what `outputs` need, folding constants on
/// the way: literal arithmetic, `x + 0`, `x * 0`, `x * 1`, `-(-x)`, and a
/// select on a literal condition. One forward pass over the node list, so it
/// is linear and does not recurse (the traced graphs are ~10⁵ nodes deep).
///
/// `x * 0 → 0` drops NaN/inf propagation through structurally zero terms
/// (a Jacobian column of a non-ancestor DOF, say); every other rewrite is
/// exact in IEEE arithmetic up to the sign of a zero.
// Float comparisons as guards rather than literal patterns, for clarity.
#[allow(clippy::redundant_guards)]
pub fn fold_constants(graph: &ExprGraph, outputs: &[ExprId]) -> (ExprGraph, Vec<ExprId>) {
    use tang_expr::Node;
    let live = graph.live_set(outputs);
    let n = graph.len();
    let mut g = ExprGraph::new();
    let mut map: Vec<ExprId> = vec![ExprId::ZERO; n];
    let lit_of = |g: &ExprGraph, id: ExprId| match g.node(id) {
        Node::Lit(bits) => Some(f64::from_bits(bits)),
        _ => None,
    };
    for i in 0..n {
        if !live.contains(&i) {
            continue;
        }
        let m = |e: ExprId| map[e.index() as usize];
        let id = match graph.node(ExprId::from_index(i as u32)) {
            Node::Var(v) => g.var(v),
            Node::Lit(bits) => g.lit(f64::from_bits(bits)),
            Node::Add(a, b) => {
                let (a, b) = (m(a), m(b));
                match (lit_of(&g, a), lit_of(&g, b)) {
                    (Some(x), Some(y)) => g.lit(x + y),
                    (Some(x), _) if x == 0.0 => b,
                    (_, Some(y)) if y == 0.0 => a,
                    _ => g.add(a, b),
                }
            }
            Node::Mul(a, b) => {
                let (a, b) = (m(a), m(b));
                match (lit_of(&g, a), lit_of(&g, b)) {
                    (Some(x), Some(y)) => g.lit(x * y),
                    (Some(x), _) | (_, Some(x)) if x == 0.0 => g.lit(0.0),
                    (Some(x), _) if x == 1.0 => b,
                    (_, Some(y)) if y == 1.0 => a,
                    (Some(x), _) if x == -1.0 => g.neg(b),
                    (_, Some(y)) if y == -1.0 => g.neg(a),
                    _ => g.mul(a, b),
                }
            }
            Node::Neg(a) => {
                let a = m(a);
                match (lit_of(&g, a), g.node(a)) {
                    (Some(x), _) => g.lit(-x),
                    (_, Node::Neg(inner)) => inner,
                    _ => g.neg(a),
                }
            }
            Node::Recip(a) => {
                let a = m(a);
                match lit_of(&g, a) {
                    Some(x) => g.lit(1.0 / x),
                    None => g.recip(a),
                }
            }
            Node::Sqrt(a) => {
                let a = m(a);
                match lit_of(&g, a) {
                    Some(x) => g.lit(x.sqrt()),
                    None => g.sqrt(a),
                }
            }
            Node::Sin(a) => {
                let a = m(a);
                match lit_of(&g, a) {
                    Some(x) => g.lit(x.sin()),
                    None => g.sin(a),
                }
            }
            Node::Atan2(y, x) => g.atan2(m(y), m(x)),
            Node::Exp2(a) => g.exp2(m(a)),
            Node::Log2(a) => g.log2(m(a)),
            Node::Select(c, a, b) => {
                let (c, a, b) = (m(c), m(a), m(b));
                match lit_of(&g, c) {
                    Some(x) if x > 0.0 => a,
                    Some(_) => b,
                    None if a == b => a,
                    None => g.select(c, a, b),
                }
            }
        };
        map[i] = id;
    }
    let outs = outputs.iter().map(|o| map[o.index() as usize]).collect();
    (g, outs)
}

/// The Jacobian `d outputs / d inputs` of a graph, as a new graph: outputs
/// row-major (`outputs.len() x n_inputs`), folded.
///
/// Forward mode in one linear pass over the node list (no recursion), with
/// each node's tangent kept sparse: only the inputs it actually depends on.
/// `Select` differentiates as `select(c, da, db)`: piecewise, with the
/// condition's own derivative zero, which is what a clamp or a contact switch
/// means away from its kink.
pub fn jacobian(
    graph: &ExprGraph,
    outputs: &[ExprId],
    n_inputs: usize,
) -> (ExprGraph, Vec<ExprId>) {
    use std::collections::BTreeMap;
    use tang_expr::Node;
    let live = graph.live_set(outputs);
    let n = graph.len();
    let mut g = ExprGraph::new();
    let mut map: Vec<ExprId> = vec![ExprId::ZERO; n];
    let mut tan: Vec<BTreeMap<u16, ExprId>> = vec![BTreeMap::new(); n];
    let ln2 = std::f64::consts::LN_2;
    for i in 0..n {
        if !live.contains(&i) {
            continue;
        }
        let m = |e: ExprId| map[e.index() as usize];
        let t = |e: ExprId| &tan[e.index() as usize];
        // Combine two sparse tangents: ka * ta + kb * tb (None = coefficient 1).
        let combine = |g: &mut ExprGraph,
                       ta: &BTreeMap<u16, ExprId>,
                       ka: Option<ExprId>,
                       tb: &BTreeMap<u16, ExprId>,
                       kb: Option<ExprId>| {
            let mut out = BTreeMap::new();
            let scale = |g: &mut ExprGraph, x: ExprId, k: Option<ExprId>| match k {
                Some(k) => g.mul(k, x),
                None => x,
            };
            for (v, d) in ta {
                let x = scale(g, *d, ka);
                out.insert(*v, x);
            }
            for (v, d) in tb {
                let x = scale(g, *d, kb);
                let y = match out.get(v) {
                    Some(prev) => g.add(*prev, x),
                    None => x,
                };
                out.insert(*v, y);
            }
            out
        };
        let (id, d) = match graph.node(ExprId::from_index(i as u32)) {
            Node::Var(v) => {
                let id = g.var(v);
                let mut d = BTreeMap::new();
                if (v as usize) < n_inputs {
                    d.insert(v, ExprId::ONE);
                }
                (id, d)
            }
            Node::Lit(bits) => (g.lit(f64::from_bits(bits)), BTreeMap::new()),
            Node::Add(a, b) => {
                let id = g.add(m(a), m(b));
                let d = combine(&mut g, t(a), None, t(b), None);
                (id, d)
            }
            Node::Mul(a, b) => {
                let id = g.mul(m(a), m(b));
                let d = combine(&mut g, t(a), Some(m(b)), t(b), Some(m(a)));
                (id, d)
            }
            Node::Neg(a) => {
                let id = g.neg(m(a));
                let k = g.lit(-1.0);
                let d = combine(&mut g, t(a), Some(k), &BTreeMap::new(), None);
                (id, d)
            }
            Node::Recip(a) => {
                // d(1/a) = -(1/a)^2 da
                let id = g.recip(m(a));
                let sq = g.mul(id, id);
                let k = g.neg(sq);
                let d = combine(&mut g, t(a), Some(k), &BTreeMap::new(), None);
                (id, d)
            }
            Node::Sqrt(a) => {
                // d sqrt(a) = da / (2 sqrt(a))
                let id = g.sqrt(m(a));
                let two = g.mul(ExprId::TWO, id);
                let k = g.recip(two);
                let d = combine(&mut g, t(a), Some(k), &BTreeMap::new(), None);
                (id, d)
            }
            Node::Sin(a) => {
                // d sin(a) = cos(a) da, cos(a) = sin(a + pi/2)
                let id = g.sin(m(a));
                let h = g.lit(std::f64::consts::FRAC_PI_2);
                let shifted = g.add(m(a), h);
                let k = g.sin(shifted);
                let d = combine(&mut g, t(a), Some(k), &BTreeMap::new(), None);
                (id, d)
            }
            Node::Atan2(y, x) => {
                // d atan2(y, x) = (x dy - y dx) / (x^2 + y^2)
                let id = g.atan2(m(y), m(x));
                let xx = g.mul(m(x), m(x));
                let yy = g.mul(m(y), m(y));
                let r2 = g.add(xx, yy);
                let inv = g.recip(r2);
                let ky = g.mul(m(x), inv);
                let ny = g.neg(m(y));
                let kx = g.mul(ny, inv);
                let d = combine(&mut g, t(y), Some(ky), t(x), Some(kx));
                (id, d)
            }
            Node::Exp2(a) => {
                // d 2^a = ln2 2^a da
                let id = g.exp2(m(a));
                let l = g.lit(ln2);
                let k = g.mul(l, id);
                let d = combine(&mut g, t(a), Some(k), &BTreeMap::new(), None);
                (id, d)
            }
            Node::Log2(a) => {
                // d log2(a) = da / (a ln2)
                let id = g.log2(m(a));
                let l = g.lit(ln2);
                let al = g.mul(m(a), l);
                let k = g.recip(al);
                let d = combine(&mut g, t(a), Some(k), &BTreeMap::new(), None);
                (id, d)
            }
            Node::Select(c, a, b) => {
                let id = g.select(m(c), m(a), m(b));
                let mut d = BTreeMap::new();
                let keys: std::collections::BTreeSet<u16> =
                    t(a).keys().chain(t(b).keys()).copied().collect();
                for v in keys {
                    let da = t(a).get(&v).copied().unwrap_or(ExprId::ZERO);
                    let db = t(b).get(&v).copied().unwrap_or(ExprId::ZERO);
                    let x = g.select(m(c), da, db);
                    d.insert(v, x);
                }
                (id, d)
            }
        };
        map[i] = id;
        tan[i] = d;
    }
    let mut outs = Vec::with_capacity(outputs.len() * n_inputs);
    for o in outputs {
        let to = &tan[o.index() as usize];
        for v in 0..n_inputs {
            outs.push(to.get(&(v as u16)).copied().unwrap_or(ExprId::ZERO));
        }
    }
    fold_constants(&g, &outs)
}
