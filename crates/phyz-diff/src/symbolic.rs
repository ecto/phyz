//! Symbolic Jacobian computation via tang-expr.
//!
//! Traces the Articulated Body Algorithm (ABA) and semi-implicit Euler step
//! symbolically using `ExprId` as the scalar type, then differentiates the
//! output expressions w.r.t. all input variables. The resulting derivatives
//! are simplified and compiled to fast closures.
//!
//! # How it works
//!
//! 1. Model topology (parent indices, joint types, axes, inertias) is read
//!    from the concrete `Model` at compile time.
//! 2. State variables (q, v, ctrl) become `ExprId::var(i)` in the expression
//!    graph. Gravity and timestep become `ExprId::from_f64(...)` literals.
//! 3. We re-implement the single-DOF ABA path using tang's generic spatial
//!    algebra types (`SpatialVec<ExprId>`, `SpatialMat<ExprId>`, etc.).
//! 4. The output accelerations are combined with semi-implicit Euler to
//!    produce (q', v') expressions. We differentiate each output w.r.t.
//!    each input, simplify, and compile to closures.
//!
//! # Limitations
//!
//! - Only single-DOF joints (revolute, prismatic) are supported. Multi-DOF
//!   joints (spherical, free) require symbolic matrix inversion which is not
//!   yet implemented.
//! - The compiled closures are specialized to a specific model topology.
//!   If the model changes, recompile.

use phyz_model::{JointType, Model};
use tang::Scalar;
use tang::{Mat3, SpatialInertia, SpatialMat, SpatialTransform, SpatialVec, Vec3};
use tang_expr::{ExprId, trace};
use tang_la::DMat;

use crate::StepJacobians;

/// Compiled multi-output closure: maps input params to output values.
type CompiledFn = Box<dyn Fn(&[f64], &mut [f64])>;

/// Compiled symbolic step Jacobians.
///
/// Caches compiled closures for evaluating all 5 Jacobian matrices.
/// Reusable across different states for the same model topology.
pub struct CompiledStepJacobians {
    /// Number of position DOFs.
    nq: usize,
    /// Number of velocity DOFs.
    nv: usize,
    /// Total number of input variables: nq + nv + nv (q, v, ctrl).
    num_inputs: usize,
    /// Compiled closure for all Jacobian entries.
    /// Layout: [dqnext_dq (nq*nq), dqnext_dv (nq*nv), dvnext_dq (nv*nq), dvnext_dv (nv*nv), dvnext_dctrl (nv*nv)]
    jac_fn: CompiledFn,
    /// Total number of Jacobian entries.
    num_outputs: usize,
}

impl CompiledStepJacobians {
    /// Build compiled Jacobians for the given model.
    ///
    /// This traces ABA + semi-implicit Euler symbolically, differentiates
    /// all outputs w.r.t. all inputs, simplifies, and compiles.
    ///
    /// Only supports models with single-DOF joints (revolute/prismatic).
    /// Panics if any joint is multi-DOF.
    pub fn build(model: &Model) -> Self {
        let nq = model.nq;
        let nv = model.nv;
        let num_inputs = nq + nv + nv; // q, v, ctrl

        // Trace the symbolic computation
        let (mut graph, (qnext_exprs, vnext_exprs)) = trace(|| symbolic_step(model));

        // Differentiate each output w.r.t. each input
        let num_jac_entries = nq * nq + nq * nv + nv * nq + nv * nv + nv * nv;
        let mut jac_exprs = Vec::with_capacity(num_jac_entries);

        // dqnext_dq: nq x nq
        for &qn in &qnext_exprs {
            for j in 0..nq {
                let d = graph.diff(qn, j as u16);
                let d = graph.simplify(d);
                jac_exprs.push(d);
            }
        }

        // dqnext_dv: nq x nv
        for &qn in &qnext_exprs {
            for j in 0..nv {
                let var_idx = (nq + j) as u16;
                let d = graph.diff(qn, var_idx);
                let d = graph.simplify(d);
                jac_exprs.push(d);
            }
        }

        // dvnext_dq: nv x nq
        for &vn in &vnext_exprs {
            for j in 0..nq {
                let d = graph.diff(vn, j as u16);
                let d = graph.simplify(d);
                jac_exprs.push(d);
            }
        }

        // dvnext_dv: nv x nv
        for &vn in &vnext_exprs {
            for j in 0..nv {
                let var_idx = (nq + j) as u16;
                let d = graph.diff(vn, var_idx);
                let d = graph.simplify(d);
                jac_exprs.push(d);
            }
        }

        // dvnext_dctrl: nv x nv
        for &vn in &vnext_exprs {
            for j in 0..nv {
                let var_idx = (nq + nv + j) as u16;
                let d = graph.diff(vn, var_idx);
                let d = graph.simplify(d);
                jac_exprs.push(d);
            }
        }

        let jac_fn = graph.compile_many(&jac_exprs);

        Self {
            nq,
            nv,
            num_inputs,
            jac_fn,
            num_outputs: num_jac_entries,
        }
    }

    /// Evaluate the Jacobians at a given state.
    ///
    /// `inputs` layout: [q_0..q_nq, v_0..v_nv, ctrl_0..ctrl_nv]
    pub fn eval(&self, inputs: &[f64]) -> StepJacobians {
        assert_eq!(inputs.len(), self.num_inputs);

        let mut flat = vec![0.0; self.num_outputs];
        (self.jac_fn)(inputs, &mut flat);

        let nq = self.nq;
        let nv = self.nv;
        let mut offset = 0;

        // dqnext_dq: nq x nq
        let dqnext_dq = DMat::from_fn(nq, nq, |i, j| flat[offset + i * nq + j]);
        offset += nq * nq;

        // dqnext_dv: nq x nv
        let dqnext_dv = DMat::from_fn(nq, nv, |i, j| flat[offset + i * nv + j]);
        offset += nq * nv;

        // dvnext_dq: nv x nq
        let dvnext_dq = DMat::from_fn(nv, nq, |i, j| flat[offset + i * nq + j]);
        offset += nv * nq;

        // dvnext_dv: nv x nv
        let dvnext_dv = DMat::from_fn(nv, nv, |i, j| flat[offset + i * nv + j]);
        offset += nv * nv;

        // dvnext_dctrl: nv x nv
        let dvnext_dctrl = DMat::from_fn(nv, nv, |i, j| flat[offset + i * nv + j]);

        StepJacobians {
            dqnext_dq,
            dqnext_dv,
            dvnext_dq,
            dvnext_dv,
            dvnext_dctrl,
        }
    }
}

/// Convenience: compute symbolic step Jacobians for a given model and state.
///
/// Builds the compiled system, evaluates at the given state, and returns.
/// For repeated evaluations on the same model, prefer building once with
/// `CompiledStepJacobians::build()` and calling `eval()` multiple times.
pub fn symbolic_step_jacobians(model: &Model, state: &phyz_model::State) -> StepJacobians {
    let compiled = CompiledStepJacobians::build(model);
    let mut inputs = Vec::with_capacity(model.nq + model.nv + model.nv);
    inputs.extend_from_slice(state.q.as_slice());
    inputs.extend_from_slice(state.v.as_slice());
    inputs.extend_from_slice(state.ctrl.as_slice());
    compiled.eval(&inputs)
}

// ---------------------------------------------------------------------------
// Symbolic ABA + semi-implicit Euler step
// ---------------------------------------------------------------------------

/// Symbolically trace ABA + semi-implicit Euler.
///
/// Variable layout:
///   vars [0..nq)           = q (joint positions)
///   vars [nq..nq+nv)       = v (joint velocities)
///   vars [nq+nv..nq+2*nv)  = ctrl (control inputs)
///
/// Returns (q_next, v_next) as vectors of ExprId.
fn symbolic_step(model: &Model) -> (Vec<ExprId>, Vec<ExprId>) {
    let nq = model.nq;
    let nv = model.nv;

    // Create symbolic variables
    let q: Vec<ExprId> = (0..nq).map(|i| ExprId::var(i as u16)).collect();
    let v: Vec<ExprId> = (0..nv).map(|i| ExprId::var((nq + i) as u16)).collect();
    let ctrl: Vec<ExprId> = (0..nv).map(|i| ExprId::var((nq + nv + i) as u16)).collect();

    // Run symbolic ABA to get accelerations
    let qdd = symbolic_aba(model, &q, &v, &ctrl);

    // Semi-implicit Euler:
    //   v' = v + dt * qdd
    //   q' = q + dt * v'
    let dt = ExprId::from_f64(model.dt);

    let v_next: Vec<ExprId> = (0..nv).map(|i| v[i] + dt * qdd[i]).collect();

    let q_next: Vec<ExprId> = (0..nq).map(|i| q[i] + dt * v_next[i]).collect();

    (q_next, v_next)
}

/// Symbolic ABA implementation for single-DOF joints.
///
/// This mirrors the concrete ABA in phyz-rigid but uses ExprId as scalar.
fn symbolic_aba(model: &Model, q: &[ExprId], v: &[ExprId], ctrl: &[ExprId]) -> Vec<ExprId> {
    let nb = model.nbodies();
    let nv = model.nv;
    let zero_sv: SpatialVec<ExprId> = SpatialVec::zero();

    // Per-body storage
    let mut x_tree: Vec<SpatialTransform<ExprId>> = vec![SpatialTransform::identity(); nb];
    let mut vel: Vec<SpatialVec<ExprId>> = vec![zero_sv; nb];
    let mut c_bias: Vec<SpatialVec<ExprId>> = vec![zero_sv; nb];
    let mut p_a: Vec<SpatialVec<ExprId>> = vec![zero_sv; nb];
    let mut i_a: Vec<SpatialMat<ExprId>> = vec![SpatialMat::zero(); nb];

    // Gravity as spatial acceleration (base acceleration trick)
    let grav = lift_vec3(model.gravity);
    let a0: SpatialVec<ExprId> = SpatialVec::new(Vec3::zero(), -grav);

    // -- Pass 1: Forward -- velocities and bias --
    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let q_idx = model.q_offsets[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let ndof = joint.ndof();

        assert!(
            ndof <= 1,
            "symbolic ABA only supports single-DOF joints (revolute/prismatic/fixed), \
             found joint with ndof={ndof}"
        );

        // Compute joint transform symbolically
        let x_joint = if ndof == 0 {
            SpatialTransform::identity()
        } else {
            sym_joint_transform(joint.joint_type, &lift_vec3(joint.axis), q[q_idx])
        };

        let parent_xf = lift_spatial_transform(&joint.parent_to_joint);
        x_tree[i] = x_joint.compose(&parent_xf);

        let v_joint = if ndof == 0 {
            zero_sv
        } else {
            sym_joint_velocity(joint.joint_type, &lift_vec3(joint.axis), v[v_idx])
        };

        if body.parent < 0 {
            vel[i] = v_joint;
            c_bias[i] = zero_sv;
        } else {
            let pi = body.parent as usize;
            let v_parent = x_tree[i].apply_motion(&vel[pi]);
            vel[i] = v_parent + v_joint;
            c_bias[i] = vel[i].cross_motion(&v_joint);
        }

        // Initialize articulated inertia with rigid body inertia
        let inertia = lift_spatial_inertia(&body.inertia);
        i_a[i] = inertia.to_matrix();

        // Bias force: v x* (I*v) (gyroscopic)
        p_a[i] = vel[i].cross_force(&i_a[i].mul_vec(&vel[i]));
    }

    // -- Pass 2: Backward -- articulated inertias and forces --
    for i in (0..nb).rev() {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let ndof = joint.ndof();

        if ndof == 0 {
            // Fixed joint: propagate to parent
            if body.parent >= 0 {
                let pi = body.parent as usize;
                let x_mot = x_tree[i].to_motion_matrix();
                let ia_parent = x_mot.transpose().mul_mat(&i_a[i]).mul_mat(&x_mot);
                let p_parent = x_tree[i].inv_apply_force(&p_a[i]);
                i_a[pi] = i_a[pi] + ia_parent;
                p_a[pi] = p_a[pi] + p_parent;
            }
            continue;
        }

        // Single-DOF path
        let s_i = sym_motion_subspace(joint.joint_type, &lift_vec3(joint.axis));

        let damping_lit = ExprId::from_f64(joint.damping);
        let tau_i = ctrl[v_idx] - damping_lit * v[v_idx];

        let ia = &i_a[i];
        let u_i = tau_i - s_i.dot(&p_a[i]);
        let d_i = s_i.dot(&ia.mul_vec(&s_i));

        let d_inv = d_i.recip();
        let u_inv_d = u_i * d_inv;
        let ia_s = ia.mul_vec(&s_i);

        if body.parent >= 0 {
            let pi = body.parent as usize;
            let outer = sym_outer_product_6(&ia_s, &ia.mul_vec(&s_i));
            let ia_new = *ia - outer * d_inv;
            let p_new = p_a[i] + ia_new.mul_vec(&c_bias[i]) + ia_s * u_inv_d;

            let x_mot = x_tree[i].to_motion_matrix();
            let ia_parent = x_mot.transpose().mul_mat(&ia_new).mul_mat(&x_mot);
            let p_parent = x_tree[i].inv_apply_force(&p_new);

            i_a[pi] = i_a[pi] + ia_parent;
            p_a[pi] = p_a[pi] + p_parent;
        }
    }

    // -- Pass 3: Forward -- accelerations --
    let mut acc: Vec<SpatialVec<ExprId>> = vec![zero_sv; nb];
    let mut qdd = vec![ExprId::ZERO; nv];

    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let ndof = joint.ndof();

        let a_parent = if body.parent < 0 {
            x_tree[i].apply_motion(&a0)
        } else {
            let pi = body.parent as usize;
            x_tree[i].apply_motion(&acc[pi])
        };

        if ndof == 0 {
            acc[i] = a_parent + c_bias[i];
            continue;
        }

        let s_i = sym_motion_subspace(joint.joint_type, &lift_vec3(joint.axis));
        let ia = &i_a[i];
        let d_i = s_i.dot(&ia.mul_vec(&s_i));
        let d_inv = d_i.recip();

        let damping_lit = ExprId::from_f64(joint.damping);
        let tau_i = ctrl[v_idx] - damping_lit * v[v_idx];
        let u_i = tau_i - s_i.dot(&p_a[i]);

        let qdd_i = (u_i - ia.mul_vec(&(a_parent + c_bias[i])).dot(&s_i)) * d_inv;
        qdd[v_idx] = qdd_i;
        acc[i] = a_parent + c_bias[i] + s_i * qdd_i;
    }

    qdd
}

// ---------------------------------------------------------------------------
// Helpers: lift f64 values to ExprId
// ---------------------------------------------------------------------------

/// Lift a concrete Vec3<f64> to Vec3<ExprId> (as literals).
fn lift_vec3(v: Vec3<f64>) -> Vec3<ExprId> {
    Vec3::new(
        ExprId::from_f64(v.x),
        ExprId::from_f64(v.y),
        ExprId::from_f64(v.z),
    )
}

/// Lift a concrete Mat3<f64> to Mat3<ExprId> (as literals).
fn lift_mat3(m: Mat3<f64>) -> Mat3<ExprId> {
    Mat3::new(
        ExprId::from_f64(m.c0.x),
        ExprId::from_f64(m.c1.x),
        ExprId::from_f64(m.c2.x),
        ExprId::from_f64(m.c0.y),
        ExprId::from_f64(m.c1.y),
        ExprId::from_f64(m.c2.y),
        ExprId::from_f64(m.c0.z),
        ExprId::from_f64(m.c1.z),
        ExprId::from_f64(m.c2.z),
    )
}

/// Lift a concrete SpatialTransform<f64> to SpatialTransform<ExprId>.
fn lift_spatial_transform(xf: &tang::SpatialTransform<f64>) -> SpatialTransform<ExprId> {
    SpatialTransform::new(lift_mat3(xf.rot), lift_vec3(xf.pos))
}

/// Lift a concrete SpatialInertia<f64> to SpatialInertia<ExprId>.
fn lift_spatial_inertia(si: &tang::SpatialInertia<f64>) -> SpatialInertia<ExprId> {
    SpatialInertia::new(
        ExprId::from_f64(si.mass),
        lift_vec3(si.com),
        lift_mat3(si.inertia),
    )
}

// ---------------------------------------------------------------------------
// Helpers: symbolic joint operations
// ---------------------------------------------------------------------------

/// Compute the symbolic joint transform for a single-DOF joint.
fn sym_joint_transform(
    joint_type: JointType,
    axis: &Vec3<ExprId>,
    q: ExprId,
) -> SpatialTransform<ExprId> {
    match joint_type {
        JointType::Revolute | JointType::Hinge => {
            // Rodrigues formula: negate angle for coordinate transform
            let (s, c) = (-q).sin_cos();
            let ax = tang::skew(axis);
            let rot = Mat3::identity() + ax * s + ax.mul_mat(&ax) * (ExprId::ONE - c);
            SpatialTransform::new(rot, Vec3::zero())
        }
        JointType::Prismatic | JointType::Slide => {
            let pos = *axis * q;
            SpatialTransform::new(Mat3::identity(), pos)
        }
        JointType::Fixed => SpatialTransform::identity(),
        _ => panic!(
            "symbolic joint transform not supported for {:?}",
            joint_type
        ),
    }
}

/// Compute the symbolic joint velocity contribution S * qd.
fn sym_joint_velocity(
    joint_type: JointType,
    axis: &Vec3<ExprId>,
    qd: ExprId,
) -> SpatialVec<ExprId> {
    match joint_type {
        JointType::Revolute | JointType::Hinge => SpatialVec::new(*axis * qd, Vec3::zero()),
        JointType::Prismatic | JointType::Slide => SpatialVec::new(Vec3::zero(), *axis * qd),
        JointType::Fixed => SpatialVec::zero(),
        _ => panic!("symbolic joint velocity not supported for {:?}", joint_type),
    }
}

/// Motion subspace vector for a single-DOF joint.
fn sym_motion_subspace(joint_type: JointType, axis: &Vec3<ExprId>) -> SpatialVec<ExprId> {
    match joint_type {
        JointType::Revolute | JointType::Hinge => SpatialVec::new(*axis, Vec3::zero()),
        JointType::Prismatic | JointType::Slide => SpatialVec::new(Vec3::zero(), *axis),
        _ => panic!("motion_subspace not supported for {:?}", joint_type),
    }
}

/// Outer product of two spatial vectors, returning a SpatialMat.
fn sym_outer_product_6(a: &SpatialVec<ExprId>, b: &SpatialVec<ExprId>) -> SpatialMat<ExprId> {
    fn vec3_outer(a: Vec3<ExprId>, b: Vec3<ExprId>) -> Mat3<ExprId> {
        Mat3::new(
            a.x * b.x,
            a.x * b.y,
            a.x * b.z,
            a.y * b.x,
            a.y * b.y,
            a.y * b.z,
            a.z * b.x,
            a.z * b.y,
            a.z * b.z,
        )
    }

    SpatialMat::new(
        vec3_outer(a.angular, b.angular),
        vec3_outer(a.angular, b.linear),
        vec3_outer(a.linear, b.angular),
        vec3_outer(a.linear, b.linear),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{GRAVITY, SpatialInertia as ConcreteSI, SpatialTransform as ConcreteXF};
    use phyz_model::ModelBuilder;

    fn make_pendulum() -> Model {
        ModelBuilder::new()
            .gravity(phyz_math::Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "link1",
                -1,
                ConcreteXF::identity(),
                ConcreteSI::new(
                    1.0,
                    phyz_math::Vec3::new(0.0, -0.5, 0.0),
                    phyz_math::Mat3::from_diagonal(&phyz_math::Vec3::new(
                        1.0 / 12.0,
                        0.0,
                        1.0 / 12.0,
                    )),
                ),
            )
            .build()
    }

    fn make_double_pendulum() -> Model {
        let length = 1.0;
        let mass = 1.0;
        ModelBuilder::new()
            .gravity(phyz_math::Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "link1",
                -1,
                ConcreteXF::identity(),
                ConcreteSI::new(
                    mass,
                    phyz_math::Vec3::new(0.0, -length / 2.0, 0.0),
                    phyz_math::Mat3::from_diagonal(&phyz_math::Vec3::new(
                        mass * length * length / 12.0,
                        0.0,
                        mass * length * length / 12.0,
                    )),
                ),
            )
            .add_revolute_body(
                "link2",
                0,
                ConcreteXF::from_translation(phyz_math::Vec3::new(0.0, -length, 0.0)),
                ConcreteSI::new(
                    mass,
                    phyz_math::Vec3::new(0.0, -length / 2.0, 0.0),
                    phyz_math::Mat3::from_diagonal(&phyz_math::Vec3::new(
                        mass * length * length / 12.0,
                        0.0,
                        mass * length * length / 12.0,
                    )),
                ),
            )
            .build()
    }

    #[test]
    fn test_symbolic_vs_fd_pendulum() {
        let model = make_pendulum();
        let mut state = model.default_state();
        state.q[0] = 0.3;
        state.v[0] = 0.1;

        let sym = symbolic_step_jacobians(&model, &state);
        let fd = crate::finite_diff_jacobians(&model, &state, 1e-6);

        let eps = 1e-4;
        assert!(
            (&fd.dqnext_dq - &sym.dqnext_dq).norm() < eps,
            "dqnext_dq mismatch: fd={:?}, sym={:?}",
            fd.dqnext_dq,
            sym.dqnext_dq
        );
        assert!(
            (&fd.dqnext_dv - &sym.dqnext_dv).norm() < eps,
            "dqnext_dv mismatch: fd={:?}, sym={:?}",
            fd.dqnext_dv,
            sym.dqnext_dv
        );
        assert!(
            (&fd.dvnext_dq - &sym.dvnext_dq).norm() < eps,
            "dvnext_dq mismatch: fd={:?}, sym={:?}",
            fd.dvnext_dq,
            sym.dvnext_dq
        );
        assert!(
            (&fd.dvnext_dv - &sym.dvnext_dv).norm() < eps,
            "dvnext_dv mismatch: fd={:?}, sym={:?}",
            fd.dvnext_dv,
            sym.dvnext_dv
        );
        assert!(
            (&fd.dvnext_dctrl - &sym.dvnext_dctrl).norm() < eps,
            "dvnext_dctrl mismatch: fd={:?}, sym={:?}",
            fd.dvnext_dctrl,
            sym.dvnext_dctrl
        );
    }

    #[test]
    fn test_symbolic_vs_fd_double_pendulum() {
        let model = make_double_pendulum();
        let mut state = model.default_state();
        state.q[0] = 0.3;
        state.q[1] = -0.2;
        state.v[0] = 0.1;
        state.v[1] = -0.05;

        let sym = symbolic_step_jacobians(&model, &state);
        let fd = crate::finite_diff_jacobians(&model, &state, 1e-6);

        let eps = 1e-4;
        assert!(
            (&fd.dqnext_dq - &sym.dqnext_dq).norm() < eps,
            "dqnext_dq mismatch: norm={}",
            (&fd.dqnext_dq - &sym.dqnext_dq).norm()
        );
        assert!(
            (&fd.dqnext_dv - &sym.dqnext_dv).norm() < eps,
            "dqnext_dv mismatch: norm={}",
            (&fd.dqnext_dv - &sym.dqnext_dv).norm()
        );
        assert!(
            (&fd.dvnext_dq - &sym.dvnext_dq).norm() < eps,
            "dvnext_dq mismatch: norm={}",
            (&fd.dvnext_dq - &sym.dvnext_dq).norm()
        );
        assert!(
            (&fd.dvnext_dv - &sym.dvnext_dv).norm() < eps,
            "dvnext_dv mismatch: norm={}",
            (&fd.dvnext_dv - &sym.dvnext_dv).norm()
        );
        assert!(
            (&fd.dvnext_dctrl - &sym.dvnext_dctrl).norm() < eps,
            "dvnext_dctrl mismatch: norm={}",
            (&fd.dvnext_dctrl - &sym.dvnext_dctrl).norm()
        );
    }

    #[test]
    fn test_compiled_reuse() {
        let model = make_pendulum();

        // Build once, evaluate at different states
        let compiled = CompiledStepJacobians::build(&model);

        // State 1
        let inputs1 = vec![0.3, 0.1, 0.0]; // q, v, ctrl
        let jac1 = compiled.eval(&inputs1);

        // State 2
        let inputs2 = vec![0.5, -0.2, 1.0];
        let jac2 = compiled.eval(&inputs2);

        // Jacobians should be different at different states
        assert!(
            (&jac1.dvnext_dq - &jac2.dvnext_dq).norm() > 1e-6,
            "Jacobians should differ at different states"
        );
    }

    #[test]
    fn test_symbolic_vs_analytical() {
        // Symbolic should match the existing analytical_step_jacobians
        let model = make_pendulum();
        let mut state = model.default_state();
        state.q[0] = 0.3;
        state.v[0] = 0.1;

        let sym = symbolic_step_jacobians(&model, &state);
        let anal = crate::analytical_step_jacobians(&model, &state);

        let eps = 1e-4;
        assert!(
            (&anal.dqnext_dq - &sym.dqnext_dq).norm() < eps,
            "dqnext_dq: analytical vs symbolic mismatch"
        );
        assert!(
            (&anal.dvnext_dq - &sym.dvnext_dq).norm() < eps,
            "dvnext_dq: analytical vs symbolic mismatch"
        );
        assert!(
            (&anal.dvnext_dv - &sym.dvnext_dv).norm() < eps,
            "dvnext_dv: analytical vs symbolic mismatch"
        );
        assert!(
            (&anal.dvnext_dctrl - &sym.dvnext_dctrl).norm() < eps,
            "dvnext_dctrl: analytical vs symbolic mismatch"
        );
    }
}
