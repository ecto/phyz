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
}

type Sym6 = [[ExprId; 6]; 6];

fn traced_aba(model: &Model, q: &[ExprId], v: &[ExprId], qfrc: &[ExprId]) -> Vec<ExprId> {
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

fn ancestors_fixed(model: &Model, i: usize) -> bool {
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

fn motion_subspace_1(joint: &phyz_model::Joint) -> SpatialVec<ExprId> {
    let axis = lift_vec3(joint.axis);
    match joint.joint_type {
        JointType::Revolute | JointType::Hinge => SpatialVec::new(axis, Vec3::zero()),
        JointType::Prismatic | JointType::Slide => SpatialVec::new(Vec3::zero(), axis),
        t => panic!("not a single-DOF joint: {t:?}"),
    }
}

/// Columns of the constant motion subspace, each a 6-vector `[ang; lin]`.
fn motion_subspace_n(joint: &phyz_model::Joint) -> Vec<[f64; 6]> {
    let s = joint.motion_subspace_matrix();
    (0..s.ncols())
        .map(|k| std::array::from_fn(|r| s[(r, k)]))
        .collect()
}

fn joint_velocity(joint: &phyz_model::Joint, qd: &[ExprId]) -> SpatialVec<ExprId> {
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
fn multi_joint_transform(joint_type: JointType, q: &[ExprId]) -> SpatialTransform<ExprId> {
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
fn quat_exp(w: Vec3<ExprId>) -> Quat<ExprId> {
    let angle = (w.x * w.x + w.y * w.y + w.z * w.z).sqrt();
    let big = angle - ExprId::from_f64(f64::EPSILON);
    let safe = ExprId::select(big, angle, ExprId::ONE);
    let (s, c) = (safe * ExprId::from_f64(0.5)).sin_cos();
    let k = ExprId::select(big, s / safe, ExprId::from_f64(0.5));
    let wq = ExprId::select(big, c, ExprId::ONE);
    Quat { w: wq, v: w * k }
}

/// Solve `D x = b` for symmetric positive definite `D`, unrolled.
fn cholesky_solve(d: &[Vec<ExprId>], b: &[ExprId]) -> Vec<ExprId> {
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

fn to_sym6(m: &SpatialMat<ExprId>) -> Sym6 {
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

fn sv_arr(v: &SpatialVec<ExprId>) -> [ExprId; 6] {
    [
        v.angular.x,
        v.angular.y,
        v.angular.z,
        v.linear.x,
        v.linear.y,
        v.linear.z,
    ]
}

fn arr_sv(a: &[ExprId; 6]) -> SpatialVec<ExprId> {
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
fn dot6_const(a: &[ExprId; 6], s: &[f64; 6]) -> ExprId {
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
