//! Scalar-generic mirrors of the **model-layout** smooth dynamics: forward
//! kinematics, ABA (with the full actuator/passive force law), CRBA, contact
//! assembly, and the quaternion-aware configuration update.
//!
//! # Why these exist alongside `rollout::step`
//!
//! [`crate::rollout::step`] is already generic, but over its **own** coordinate
//! layout (quaternion sub-blocks, `nq ≠ nv`) and over a *simplified* force law
//! (`τ = ctrl − damping·v`, no actuators, no armature, no limits). The convex
//! contact adjoint ([`crate::contact_adjoint`]) differentiates the dynamics
//! `phyz::Simulator::step_with_contacts` integrates, and those live on the
//! **model** layout (`nq == nv`, exponential coordinates) with the full
//! `generalized_forces` law. Differencing was how the adjoint bridged that gap;
//! these mirrors close it analytically: instantiated at `f64` they reproduce
//! the concrete `phyz_rigid`/`phyz_contact` evaluations (pinned by tests
//! below), instantiated at [`crate::rev::Rev`] they are the reverse-mode
//! derivative of exactly those evaluations.
//!
//! The configuration update mirrors [`phyz_rigid::integrate_configuration`] —
//! the quaternion-aware Lie-group step — **not** the flat `q += dt·v` that
//! `symbolic_step_jacobians` assumes. That distinction is the recorded trap
//! this module exists to respect: a derivative of the wrong integrator is a
//! wrong derivative on every model with a ball or free joint.
//!
//! Branch discipline is the workspace's usual one: every comparison (pivots,
//! clamps, `max(0)` gates, small-angle branches) reads the primal, so a `Rev`
//! evaluation differentiates the branch the `f64` computation takes.

use phyz_model::{ContactMaterial, Joint, JointType, Model};
use tang::{Mat3, Quat, Scalar, SpatialInertia, SpatialMat, SpatialTransform, SpatialVec, Vec3};

// ---------------------------------------------------------------------------
// Lifts
// ---------------------------------------------------------------------------

pub(crate) fn lift_v3<T: Scalar>(v: Vec3<f64>) -> Vec3<T> {
    Vec3::new(T::from_f64(v.x), T::from_f64(v.y), T::from_f64(v.z))
}

fn lift_m3<T: Scalar>(m: &Mat3<f64>) -> Mat3<T> {
    Mat3::from_cols(lift_v3(m.col(0)), lift_v3(m.col(1)), lift_v3(m.col(2)))
}

fn lift_xf<T: Scalar>(xf: &SpatialTransform<f64>) -> SpatialTransform<T> {
    SpatialTransform::new(lift_m3(&xf.rot), lift_v3(xf.pos))
}

// ---------------------------------------------------------------------------
// Quaternion exp/log (mirrors of phyz_math::quat_exp / quat_log)
// ---------------------------------------------------------------------------

/// Mirror of `phyz_math::quat_exp`: same small-angle branch, on the primal.
pub(crate) fn quat_exp_gen<T: Scalar>(omega: Vec3<T>) -> Quat<T> {
    let angle = omega.norm();
    if angle.to_f64() < f64::EPSILON {
        return Quat {
            w: T::ONE,
            v: omega * T::HALF,
        };
    }
    let (s, c) = (angle * T::HALF).sin_cos();
    Quat {
        w: c,
        v: omega * (s / angle),
    }
}

/// Mirror of `phyz_math::quat_log`.
pub(crate) fn quat_log_gen<T: Scalar>(q: &Quat<T>) -> Vec3<T> {
    let norm_v = q.v.norm();
    if norm_v.to_f64() < f64::EPSILON {
        return q.v * T::TWO;
    }
    let angle = T::TWO * norm_v.atan2(q.w);
    q.v * (angle / norm_v)
}

// ---------------------------------------------------------------------------
// Joint transform / motion subspace (model layout)
// ---------------------------------------------------------------------------

/// Mirror of `Joint::joint_transform_slice` (model layout: exponential
/// coordinates, free joint `q = [w(3), pos(3)]`).
pub(crate) fn joint_transform_gen<T: Scalar>(joint: &Joint, q: &[T]) -> SpatialTransform<T> {
    match joint.joint_type {
        JointType::Revolute | JointType::Hinge => {
            let (s, c) = (-q[0]).sin_cos();
            let ax = tang::skew(&lift_v3::<T>(joint.axis));
            let rot = Mat3::identity() + ax * s + ax.mul_mat(&ax) * (T::ONE - c);
            SpatialTransform::new(rot, Vec3::zero())
        }
        JointType::Prismatic | JointType::Slide => {
            SpatialTransform::new(Mat3::identity(), lift_v3::<T>(joint.axis) * q[0])
        }
        JointType::Spherical | JointType::Ball => {
            let w = Vec3::new(q[0], q[1], q[2]);
            let rot = quat_exp_gen(w).to_matrix().transpose();
            SpatialTransform::new(rot, Vec3::zero())
        }
        JointType::Free => {
            let w = Vec3::new(q[0], q[1], q[2]);
            let pos = Vec3::new(q[3], q[4], q[5]);
            let rot = quat_exp_gen(w).to_matrix().transpose();
            SpatialTransform::new(rot, pos)
        }
        JointType::Fixed => SpatialTransform::identity(),
    }
}

const MAX_DOF: usize = 6;

fn unit<T: Scalar>(k: usize) -> Vec3<T> {
    let mut e = Vec3::zero();
    match k {
        0 => e.x = T::ONE,
        1 => e.y = T::ONE,
        _ => e.z = T::ONE,
    }
    e
}

/// Columns of the motion subspace `S` (6×ndof); mirrors
/// `Joint::motion_subspace_matrix`.
fn motion_cols<T: Scalar>(joint: &Joint) -> [SpatialVec<T>; MAX_DOF] {
    let mut s = [SpatialVec::zero(); MAX_DOF];
    let axis = lift_v3::<T>(joint.axis);
    match joint.joint_type {
        JointType::Revolute | JointType::Hinge => s[0] = SpatialVec::new(axis, Vec3::zero()),
        JointType::Prismatic | JointType::Slide => s[0] = SpatialVec::new(Vec3::zero(), axis),
        JointType::Spherical | JointType::Ball => {
            for (k, s_k) in s.iter_mut().enumerate().take(3) {
                *s_k = SpatialVec::new(unit(k), Vec3::zero());
            }
        }
        JointType::Free => {
            for k in 0..3 {
                s[k] = SpatialVec::new(unit(k), Vec3::zero());
                s[3 + k] = SpatialVec::new(Vec3::zero(), unit(k));
            }
        }
        JointType::Fixed => {}
    }
    s
}

/// Mirror of `phyz_rigid`'s `joint_velocity`: `S·qd` over the joint's DOFs.
fn joint_vel_gen<T: Scalar>(joint: &Joint, qd: &[T]) -> SpatialVec<T> {
    let ndof = joint.ndof();
    if ndof == 0 {
        return SpatialVec::zero();
    }
    let s = motion_cols::<T>(joint);
    let mut out = SpatialVec::zero();
    for k in 0..ndof {
        out = out + SpatialVec::new(s[k].angular * qd[k], s[k].linear * qd[k]);
    }
    out
}

// ---------------------------------------------------------------------------
// Forward kinematics
// ---------------------------------------------------------------------------

/// Mirror of [`phyz_rigid::forward_kinematics`] on the model layout.
pub(crate) fn fk_gen<T: Scalar>(
    model: &Model,
    q: &[T],
    v: &[T],
) -> (Vec<SpatialTransform<T>>, Vec<SpatialVec<T>>) {
    let nb = model.nbodies();
    let mut xforms: Vec<SpatialTransform<T>> = vec![SpatialTransform::identity(); nb];
    let mut vels: Vec<SpatialVec<T>> = vec![SpatialVec::zero(); nb];
    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let q_idx = model.q_offsets[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let ndof = joint.ndof();

        let x_joint = if ndof == 0 {
            SpatialTransform::identity()
        } else {
            joint_transform_gen(joint, &q[q_idx..q_idx + ndof])
        };
        let x_tree = x_joint.compose(&lift_xf(&joint.parent_to_joint));
        let v_joint = joint_vel_gen(joint, &v[v_idx..v_idx + ndof]);

        if body.parent < 0 {
            xforms[i] = x_tree;
            vels[i] = v_joint;
        } else {
            let pi = body.parent as usize;
            xforms[i] = x_tree.compose(&xforms[pi]);
            vels[i] = x_tree.apply_motion(&vels[pi]) + v_joint;
        }
    }
    (xforms, vels)
}

// ---------------------------------------------------------------------------
// Generalized forces (actuators + passive), armature
// ---------------------------------------------------------------------------

/// Mirror of `phyz_rigid::actuation::generalized_forces`: the full actuator
/// law (`gear·(gain·clamp(ctrl) + bias_q·q + bias_v·v)` with force-range
/// clamp) plus every passive joint term (damping, spring, dry friction with
/// the smoothed-Coulomb `tanh`, gated soft limits).
// Mirrors the concrete loop shape (`actuator_forces_at`), copy suggestion and
// all — drift-resistance beats idiom here.
#[allow(clippy::manual_memcpy)]
fn qfrc_gen<T: Scalar>(model: &Model, q: &[T], v: &[T], ctrl: &[T]) -> Vec<T> {
    let nv = model.nv;
    let mut qfrc = vec![T::ZERO; nv];

    // Actuator forces (`Model::actuator_forces_at`).
    if model.actuators.is_empty() {
        for i in 0..nv.min(ctrl.len()) {
            qfrc[i] = ctrl[i];
        }
    } else {
        for (a, act) in model.actuators.iter().enumerate() {
            if a >= ctrl.len() {
                break;
            }
            let (Some(&v_idx), Some(&q_idx)) = (
                model.v_offsets.get(act.joint_idx),
                model.q_offsets.get(act.joint_idx),
            ) else {
                continue;
            };
            if v_idx < nv && q_idx < q.len() {
                // `Actuator::force_at`, with the ctrl clamp branching on the
                // primal (a saturated actuator has a genuinely zero gradient).
                let c = match act.ctrl_range {
                    Some([lo, hi]) => {
                        ctrl[a].clamp(T::from_f64(lo.min(hi)), T::from_f64(hi.max(lo)))
                    }
                    None => ctrl[a],
                };
                let mut f = T::from_f64(act.gear)
                    * (T::from_f64(act.gain) * c
                        + T::from_f64(act.bias_q) * q[q_idx]
                        + T::from_f64(act.bias_v) * v[v_idx]);
                if let Some([lo, hi]) = act.force_range {
                    f = f.clamp(T::from_f64(lo.min(hi)), T::from_f64(hi.max(lo)));
                }
                qfrc[v_idx] += f;
            }
        }
    }

    // Passive forces (`Joint::passive_force`).
    for (j, joint) in model.joints.iter().enumerate() {
        let ndof = joint.ndof();
        if ndof == 0 {
            continue;
        }
        let q_idx = model.q_offsets[j];
        let v_idx = model.v_offsets[j];
        for k in 0..ndof {
            let qk = q[q_idx + k];
            let qd = v[v_idx + k];
            let mut f = T::from_f64(-joint.damping) * qd;
            if ndof == 1 && k == 0 {
                f += T::from_f64(-joint.stiffness) * (qk - T::from_f64(joint.spring_ref));
                // `Joint::limit_force`: branch on the primal violation, keep
                // the smooth spring/damper inside the chosen branch.
                if let Some([lo, hi]) = joint.limits {
                    let (depth, sign) = if qk.to_f64() < lo {
                        (T::from_f64(lo) - qk, 1.0)
                    } else if qk.to_f64() > hi {
                        (qk - T::from_f64(hi), -1.0)
                    } else {
                        (T::ZERO, 0.0)
                    };
                    if sign != 0.0 {
                        let inward = (qd * T::from_f64(-sign)).max(T::ZERO);
                        f += (T::from_f64(joint.limit_stiffness) * depth
                            + T::from_f64(joint.limit_damping) * inward)
                            * T::from_f64(sign);
                    }
                }
                if joint.friction_loss > 0.0 {
                    const FRICTION_VEL_SCALE: f64 = 1e-3;
                    f += T::from_f64(-joint.friction_loss)
                        * (qd / T::from_f64(FRICTION_VEL_SCALE)).tanh();
                }
            }
            qfrc[v_idx + k] += f;
        }
    }

    qfrc
}

// ---------------------------------------------------------------------------
// Small dense helpers
// ---------------------------------------------------------------------------

fn outer6<T: Scalar>(a: &SpatialVec<T>, b: &SpatialVec<T>) -> SpatialMat<T> {
    fn v3_outer<T: Scalar>(a: Vec3<T>, b: Vec3<T>) -> Mat3<T> {
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
        v3_outer(a.angular, b.angular),
        v3_outer(a.angular, b.linear),
        v3_outer(a.linear, b.angular),
        v3_outer(a.linear, b.linear),
    )
}

/// Invert a small `n×n` matrix (row-major) by Gauss-Jordan with partial
/// pivoting on the primal. Returns `None` on a numerically singular pivot —
/// the same refusal `DMat::try_inverse` makes in the concrete ABA.
fn invert_small<T: Scalar>(a: &mut [T], n: usize) -> Option<Vec<T>> {
    let mut inv = vec![T::ZERO; n * n];
    for r in 0..n {
        inv[r * n + r] = T::ONE;
    }
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col * n + col].to_f64().abs();
        for r in col + 1..n {
            let mag = a[r * n + col].to_f64().abs();
            if mag > best {
                best = mag;
                piv = r;
            }
        }
        if best < 1e-14 {
            return None;
        }
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
                inv.swap(col * n + k, piv * n + k);
            }
        }
        let d = a[col * n + col].recip();
        for k in 0..n {
            a[col * n + k] *= d;
            inv[col * n + k] *= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r * n + col];
            if factor.to_f64() == 0.0 {
                continue;
            }
            for k in 0..n {
                let t1 = factor * a[col * n + k];
                a[r * n + k] -= t1;
                let t2 = factor * inv[col * n + k];
                inv[r * n + k] -= t2;
            }
        }
    }
    Some(inv)
}

// ---------------------------------------------------------------------------
// ABA (model layout, full force law)
// ---------------------------------------------------------------------------

/// Mirror of [`phyz_rigid::aba_with_external_forces`], with the body inertias
/// supplied separately so a caller can seed them.
pub(crate) fn aba_gen<T: Scalar>(
    model: &Model,
    inertias: &[SpatialInertia<T>],
    q: &[T],
    v: &[T],
    ctrl: &[T],
    ext: Option<&[SpatialVec<T>]>,
) -> Vec<T> {
    let nb = model.nbodies();
    let mut qdd = vec![T::ZERO; model.nv];

    let qfrc = qfrc_gen(model, q, v, ctrl);
    // Armature is a per-DOF model constant (`actuation::armature_diag`).
    let mut armature = vec![0.0f64; model.nv];
    for (j, joint) in model.joints.iter().enumerate() {
        let v_idx = model.v_offsets[j];
        for k in 0..joint.ndof() {
            armature[v_idx + k] = joint.armature;
        }
    }

    let mut x_tree: Vec<SpatialTransform<T>> = vec![SpatialTransform::identity(); nb];
    let mut vel: Vec<SpatialVec<T>> = vec![SpatialVec::zero(); nb];
    let mut c_bias: Vec<SpatialVec<T>> = vec![SpatialVec::zero(); nb];
    let mut p_a: Vec<SpatialVec<T>> = vec![SpatialVec::zero(); nb];
    let mut i_a: Vec<SpatialMat<T>> = vec![SpatialMat::zero(); nb];

    let a0: SpatialVec<T> = SpatialVec::new(Vec3::zero(), -lift_v3(model.gravity));

    // -- Pass 1 --
    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let q_idx = model.q_offsets[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let ndof = joint.ndof();

        let x_joint = if ndof == 0 {
            SpatialTransform::identity()
        } else {
            joint_transform_gen(joint, &q[q_idx..q_idx + ndof])
        };
        x_tree[i] = x_joint.compose(&lift_xf(&joint.parent_to_joint));

        let v_joint = joint_vel_gen(joint, &v[v_idx..v_idx + ndof]);
        if body.parent < 0 {
            vel[i] = v_joint;
            c_bias[i] = SpatialVec::zero();
        } else {
            let pi = body.parent as usize;
            vel[i] = x_tree[i].apply_motion(&vel[pi]) + v_joint;
            c_bias[i] = vel[i].cross_motion(&v_joint);
        }

        i_a[i] = inertias[i].to_matrix();
        p_a[i] = vel[i].cross_force(&i_a[i].mul_vec(&vel[i]));
        if let Some(ext) = ext {
            p_a[i] = p_a[i] - ext[i];
        }
    }

    let idamp = |joint: &Joint| T::from_f64(model.dt * joint.damping);

    // -- Pass 2 --
    for i in (0..nb).rev() {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let ndof = joint.ndof();

        if ndof == 0 {
            if body.parent >= 0 {
                let pi = body.parent as usize;
                let x_mot = x_tree[i].to_motion_matrix();
                i_a[pi] = i_a[pi] + x_mot.transpose().mul_mat(&i_a[i]).mul_mat(&x_mot);
                p_a[pi] = p_a[pi] + x_tree[i].inv_apply_force(&p_a[i]);
            }
            continue;
        }

        if ndof == 1 {
            let s = motion_cols::<T>(joint);
            let s_i = s[0];
            let ia = &i_a[i];
            let u_i = qfrc[v_idx] - s_i.dot(&p_a[i]);
            let d_i = s_i.dot(&ia.mul_vec(&s_i)) + T::from_f64(armature[v_idx]) + idamp(joint);
            if d_i.to_f64().abs() < 1e-20 {
                continue;
            }
            let ia_s = ia.mul_vec(&s_i);
            if body.parent >= 0 {
                let pi = body.parent as usize;
                let d_inv = d_i.recip();
                let ia_new = *ia - outer6(&ia_s, &ia_s) * d_inv;
                let p_new = p_a[i] + ia_new.mul_vec(&c_bias[i]) + ia_s * (u_i * d_inv);
                let x_mot = x_tree[i].to_motion_matrix();
                i_a[pi] = i_a[pi] + x_mot.transpose().mul_mat(&ia_new).mul_mat(&x_mot);
                p_a[pi] = p_a[pi] + x_tree[i].inv_apply_force(&p_new);
            }
        } else {
            let s = motion_cols::<T>(joint);
            let ia = i_a[i];
            let mut u_cols = [SpatialVec::<T>::zero(); MAX_DOF];
            for k in 0..ndof {
                u_cols[k] = ia.mul_vec(&s[k]);
            }
            let mut d = vec![T::ZERO; ndof * ndof];
            for k in 0..ndof {
                for l in 0..ndof {
                    d[k * ndof + l] = s[k].dot(&u_cols[l]);
                }
                d[k * ndof + k] += T::from_f64(armature[v_idx + k]) + idamp(joint);
            }
            let Some(d_inv) = invert_small(&mut d.clone(), ndof) else {
                continue;
            };
            let mut u_vec = vec![T::ZERO; ndof];
            for k in 0..ndof {
                u_vec[k] = qfrc[v_idx + k] - s[k].dot(&p_a[i]);
            }
            if body.parent >= 0 {
                let pi = body.parent as usize;
                // Iᴬ_new = Iᴬ − U D⁻¹ Uᵀ, column by column.
                let mut ia_new = ia;
                for k in 0..ndof {
                    // y_k = Σ_l D⁻¹[k,l] · U[:,l]  (row k of D⁻¹Uᵀ).
                    let mut y = SpatialVec::<T>::zero();
                    for l in 0..ndof {
                        let w = d_inv[k * ndof + l];
                        y = y + SpatialVec::new(u_cols[l].angular * w, u_cols[l].linear * w);
                    }
                    ia_new = ia_new - outer6(&u_cols[k], &y);
                }
                // pᴬ_new = pᴬ + Iᴬ_new·c + U D⁻¹ u
                let mut p_new = p_a[i] + ia_new.mul_vec(&c_bias[i]);
                for k in 0..ndof {
                    let mut w = T::ZERO;
                    for l in 0..ndof {
                        w += d_inv[k * ndof + l] * u_vec[l];
                    }
                    p_new = p_new + u_cols[k] * w;
                }
                let x_mot = x_tree[i].to_motion_matrix();
                i_a[pi] = i_a[pi] + x_mot.transpose().mul_mat(&ia_new).mul_mat(&x_mot);
                p_a[pi] = p_a[pi] + x_tree[i].inv_apply_force(&p_new);
            }
        }
    }

    // -- Pass 3 --
    let mut acc: Vec<SpatialVec<T>> = vec![SpatialVec::zero(); nb];
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

        if ndof == 0 {
            acc[i] = a_parent + c_bias[i];
            continue;
        }

        if ndof == 1 {
            let s = motion_cols::<T>(joint);
            let s_i = s[0];
            let ia = &i_a[i];
            let d_i = s_i.dot(&ia.mul_vec(&s_i)) + T::from_f64(armature[v_idx]) + idamp(joint);
            if d_i.to_f64().abs() < 1e-20 {
                acc[i] = a_parent + c_bias[i];
                continue;
            }
            let u_i = qfrc[v_idx] - s_i.dot(&p_a[i]);
            let qdd_i = (u_i - ia.mul_vec(&(a_parent + c_bias[i])).dot(&s_i)) * d_i.recip();
            qdd[v_idx] = qdd_i;
            acc[i] =
                a_parent + c_bias[i] + SpatialVec::new(s_i.angular * qdd_i, s_i.linear * qdd_i);
        } else {
            let s = motion_cols::<T>(joint);
            let ia = i_a[i];
            let a_total = a_parent + c_bias[i];
            let ia_a = ia.mul_vec(&a_total);
            let mut d = vec![T::ZERO; ndof * ndof];
            let mut rhs = vec![T::ZERO; ndof];
            for k in 0..ndof {
                for l in 0..ndof {
                    d[k * ndof + l] = s[k].dot(&ia.mul_vec(&s[l]));
                }
                d[k * ndof + k] += T::from_f64(armature[v_idx + k]) + idamp(joint);
                rhs[k] = qfrc[v_idx + k] - s[k].dot(&p_a[i]) - s[k].dot(&ia_a);
            }
            let Some(d_inv) = invert_small(&mut d, ndof) else {
                acc[i] = a_total;
                continue;
            };
            let mut s_qdd = SpatialVec::zero();
            for k in 0..ndof {
                let mut x = T::ZERO;
                for l in 0..ndof {
                    x += d_inv[k * ndof + l] * rhs[l];
                }
                qdd[v_idx + k] = x;
                s_qdd = s_qdd + SpatialVec::new(s[k].angular * x, s[k].linear * x);
            }
            acc[i] = a_total + s_qdd;
        }
    }

    qdd
}

// ---------------------------------------------------------------------------
// CRBA (model layout)
// ---------------------------------------------------------------------------

/// Mirror of [`phyz_rigid::crba`]: the `nv × nv` mass matrix (row-major),
/// armature on the diagonal.
// Indexed loops mirror `phyz_rigid::crba`'s tree walk one-for-one.
#[allow(clippy::needless_range_loop)]
pub(crate) fn crba_gen<T: Scalar>(
    model: &Model,
    inertias: &[SpatialInertia<T>],
    q: &[T],
) -> Vec<T> {
    let nb = model.nbodies();
    let nv = model.nv;
    let mut m = vec![T::ZERO; nv * nv];

    let mut x_tree: Vec<SpatialTransform<T>> = vec![SpatialTransform::identity(); nb];
    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let q_idx = model.q_offsets[body.joint_idx];
        let ndof = joint.ndof();
        let x_joint = if ndof == 0 {
            SpatialTransform::identity()
        } else {
            joint_transform_gen(joint, &q[q_idx..q_idx + ndof])
        };
        x_tree[i] = x_joint.compose(&lift_xf(&joint.parent_to_joint));
    }

    let mut i_c: Vec<SpatialMat<T>> = inertias.iter().map(|si| si.to_matrix()).collect();
    for i in (0..nb).rev() {
        let body = &model.bodies[i];
        if body.parent >= 0 {
            let pi = body.parent as usize;
            let x_mot = x_tree[i].to_motion_matrix();
            i_c[pi] = i_c[pi] + x_mot.transpose().mul_mat(&i_c[i]).mul_mat(&x_mot);
        }
    }

    for i in 0..nb {
        let joint_i = &model.joints[model.bodies[i].joint_idx];
        let v_i = model.v_offsets[model.bodies[i].joint_idx];
        let ndof_i = joint_i.ndof();
        if ndof_i == 0 {
            continue;
        }
        let s_i = motion_cols::<T>(joint_i);

        // Diagonal block.
        for ki in 0..ndof_i {
            let f = i_c[i].mul_vec(&s_i[ki]);
            for kj in 0..ndof_i {
                m[(v_i + kj) * nv + v_i + ki] = s_i[kj].dot(&f);
            }
        }

        // Off-diagonal: walk up the tree, one column of F_i at a time.
        for col in 0..ndof_i {
            let mut f = x_tree[i].inv_apply_force(&i_c[i].mul_vec(&s_i[col]));
            let mut j = model.bodies[i].parent;
            while j >= 0 {
                let ju = j as usize;
                let joint_j = &model.joints[model.bodies[ju].joint_idx];
                let v_j = model.v_offsets[model.bodies[ju].joint_idx];
                let ndof_j = joint_j.ndof();
                if ndof_j > 0 {
                    let s_j = motion_cols::<T>(joint_j);
                    for kj in 0..ndof_j {
                        let val = s_j[kj].dot(&f);
                        m[(v_i + col) * nv + v_j + kj] = val;
                        m[(v_j + kj) * nv + v_i + col] = val;
                    }
                }
                f = x_tree[ju].inv_apply_force(&f);
                j = model.bodies[ju].parent;
            }
        }
    }

    for (j, joint) in model.joints.iter().enumerate() {
        let v_idx = model.v_offsets[j];
        for k in 0..joint.ndof() {
            m[(v_idx + k) * nv + v_idx + k] += T::from_f64(joint.armature);
        }
    }

    m
}

/// Mirror of `phyz_contact::assemble`'s `invert_symmetric`: Gauss-Jordan with
/// partial pivoting, singular columns left as identity rows.
pub(crate) fn invert_sym_gen<T: Scalar>(m: &[T], n: usize) -> Vec<T> {
    let mut a = m.to_vec();
    let mut inv = vec![T::ZERO; n * n];
    for r in 0..n {
        inv[r * n + r] = T::ONE;
    }
    for col in 0..n {
        let mut pivot = col;
        for r in col + 1..n {
            if a[r * n + col].to_f64().abs() > a[pivot * n + col].to_f64().abs() {
                pivot = r;
            }
        }
        if a[pivot * n + col].to_f64().abs() < 1e-14 {
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
            if factor.to_f64() == 0.0 {
                continue;
            }
            for k in 0..n {
                let t1 = factor * a[col * n + k];
                a[r * n + k] -= t1;
                let t2 = factor * inv[col * n + k];
                inv[r * n + k] -= t2;
            }
        }
    }
    inv
}

// ---------------------------------------------------------------------------
// Point Jacobians
// ---------------------------------------------------------------------------

/// Mirror of `phyz_rigid::point_jacobian` (`3 × nv`, row-major).
pub(crate) fn point_jacobian_gen<T: Scalar>(
    model: &Model,
    xforms: &[SpatialTransform<T>],
    body: usize,
    point: Vec3<T>,
) -> Vec<T> {
    let nv = model.nv;
    let mut j = vec![T::ZERO; 3 * nv];
    let mut cur = body as i32;
    while cur >= 0 {
        let b = cur as usize;
        let bodyref = &model.bodies[b];
        let joint = &model.joints[bodyref.joint_idx];
        let ndof = joint.ndof();
        if ndof > 0 {
            let v_idx = model.v_offsets[bodyref.joint_idx];
            let xf = &xforms[b];
            let origin = xf.pos;
            // `rot` is world→body; a body direction goes to world through the
            // transpose (`body_to_world_dir`).
            let to_world = xf.rot.transpose();
            let axis_world = to_world.mul_vec(lift_v3::<T>(joint.axis));
            let body_axis = |k: usize| to_world.mul_vec(unit::<T>(k));
            let arm = point - origin;
            for d in 0..ndof {
                let contribution = match joint.joint_type {
                    JointType::Revolute | JointType::Hinge => axis_world.cross(arm),
                    JointType::Prismatic | JointType::Slide => axis_world,
                    JointType::Spherical | JointType::Ball => body_axis(d).cross(arm),
                    JointType::Free if d < 3 => body_axis(d).cross(arm),
                    JointType::Free => body_axis(d - 3),
                    JointType::Fixed => Vec3::zero(),
                };
                j[v_idx + d] = contribution.x;
                j[nv + v_idx + d] = contribution.y;
                j[2 * nv + v_idx + d] = contribution.z;
            }
        }
        cur = bodyref.parent;
    }
    j
}

// ---------------------------------------------------------------------------
// Contact frame, material laws
// ---------------------------------------------------------------------------

/// Mirror of `phyz_contact::cone::contact_frame`.
pub(crate) fn contact_frame_gen<T: Scalar>(normal: &Vec3<T>) -> (Vec3<T>, Vec3<T>, Vec3<T>) {
    let n = normal.normalize();
    let (nx, ny, nz) = (n.x.to_f64().abs(), n.y.to_f64().abs(), n.z.to_f64().abs());
    let a: Vec3<T> = if nx <= ny && nx <= nz {
        unit(0)
    } else if ny <= nz {
        unit(1)
    } else {
        unit(2)
    };
    let u = n.cross(a).normalize();
    let w = n.cross(u);
    (n, u, w)
}

/// Mirror of `SolImp::impedance`.
fn solimp_impedance_gen<T: Scalar>(si: &phyz_model::SolImp, r: T) -> T {
    let dmin = si.dmin.clamp(1e-4, 1.0 - 1e-9);
    let dmax = si.dmax.clamp(1e-4, 1.0 - 1e-9);
    if si.width <= 0.0 {
        return T::from_f64(dmax);
    }
    let x = (r.abs() / T::from_f64(si.width)).clamp(T::ZERO, T::ONE);
    let mid = si.midpoint.clamp(1e-6, 1.0 - 1e-6);
    let p = si.power.max(1.0);
    let y = if x.to_f64() <= mid {
        x.powf(T::from_f64(p)) / T::from_f64(mid.powf(p - 1.0))
    } else {
        T::ONE - (T::ONE - x).powf(T::from_f64(p)) / T::from_f64((1.0 - mid).powf(p - 1.0))
    };
    T::from_f64(dmin) + y * T::from_f64(dmax - dmin)
}

/// Mirror of `ContactMaterial::impedance_at`.
pub(crate) fn impedance_at_gen<T: Scalar>(mat: &ContactMaterial, depth: T) -> T {
    if depth.to_f64() >= 0.0 {
        return solimp_impedance_gen(&mat.solimp, depth);
    }
    let gap = -depth;
    if mat.margin <= 0.0 || gap.to_f64() >= mat.margin {
        return T::ZERO;
    }
    let s = T::ONE - gap / T::from_f64(mat.margin);
    let ramp = s * s * (T::from_f64(3.0) - T::TWO * s);
    ramp * T::from_f64(mat.solimp.impedance(0.0))
}

/// Mirror of `ContactProblem::effective_restitution`.
pub(crate) fn effective_restitution_gen<T: Scalar>(e: T, approach: T, v_rest: f64) -> T {
    if v_rest <= 0.0 {
        return e;
    }
    let s = approach.abs();
    if s.to_f64() <= v_rest {
        return T::ZERO;
    }
    if s.to_f64() >= 2.0 * v_rest {
        return e;
    }
    let t = (s - T::from_f64(v_rest)) / T::from_f64(v_rest);
    e * t * t * (T::from_f64(3.0) - T::TWO * t)
}

// ---------------------------------------------------------------------------
// Configuration update (model layout)
// ---------------------------------------------------------------------------

/// Mirror of [`phyz_rigid::integrate_configuration`]: the quaternion-aware
/// Lie-group step on the model layout. This is the integrator the convex
/// adjoint's `Φ` actually is — **not** `q + dt·v`.
pub(crate) fn integrate_configuration_gen<T: Scalar>(
    model: &Model,
    q: &[T],
    v: &[T],
    dt: f64,
) -> Vec<T> {
    let mut out = q.to_vec();
    let dt = T::from_f64(dt);
    for (jidx, joint) in model.joints.iter().enumerate() {
        let q_off = model.q_offsets[jidx];
        let v_off = model.v_offsets[jidx];
        match joint.joint_type {
            JointType::Fixed => {}
            JointType::Revolute | JointType::Hinge | JointType::Prismatic | JointType::Slide => {
                out[q_off] = q[q_off] + dt * v[v_off];
            }
            JointType::Spherical | JointType::Ball => {
                let omega = Vec3::new(v[v_off], v[v_off + 1], v[v_off + 2]);
                let current = quat_exp_gen(Vec3::new(q[q_off], q[q_off + 1], q[q_off + 2]));
                let next = current.mul(&quat_exp_gen(omega * dt)).normalize();
                let log = quat_log_gen(&next);
                out[q_off] = log.x;
                out[q_off + 1] = log.y;
                out[q_off + 2] = log.z;
            }
            JointType::Free => {
                let omega = Vec3::new(v[v_off], v[v_off + 1], v[v_off + 2]);
                let lin = Vec3::new(v[v_off + 3], v[v_off + 4], v[v_off + 5]);
                let current = quat_exp_gen(Vec3::new(q[q_off], q[q_off + 1], q[q_off + 2]));
                let world_lin = current.rotate(lin);
                out[q_off + 3] = q[q_off + 3] + dt * world_lin.x;
                out[q_off + 4] = q[q_off + 4] + dt * world_lin.y;
                out[q_off + 5] = q[q_off + 5] + dt * world_lin.z;
                let next = current.mul(&quat_exp_gen(omega * dt)).normalize();
                let log = quat_log_gen(&next);
                out[q_off] = log.x;
                out[q_off + 1] = log.y;
                out[q_off + 2] = log.z;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{DVec, GRAVITY};
    use phyz_model::{Actuator, ModelBuilder, State};

    fn skate_like_model() -> Model {
        // A small tree exercising every generic path: a free base, a revolute
        // child with limits/spring/friction/armature, and an actuator with
        // gear and ctrl range.
        let mut model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
            .dt(2e-3)
            .add_free_body(
                "base",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(
                    3.0,
                    Vec3::new(0.02, -0.01, 0.05),
                    Mat3::new(0.09, 0.001, 0.002, 0.001, 0.11, 0.003, 0.002, 0.003, 0.05),
                ),
            )
            .add_revolute_body(
                "leg",
                0,
                SpatialTransform::from_translation(Vec3::new(0.1, 0.0, -0.2)),
                SpatialInertia::new(
                    1.2,
                    Vec3::new(0.0, 0.0, -0.15),
                    Mat3::from_diagonal(&Vec3::new(0.02, 0.02, 0.004)),
                ),
            )
            .build();
        {
            let j = &mut model.joints[1];
            j.damping = 0.7;
            j.armature = 0.011;
            j.stiffness = 2.0;
            j.spring_ref = 0.1;
            j.limits = Some([-1.0, 1.0]);
            j.limit_stiffness = 40.0;
            j.limit_damping = 3.0;
            j.friction_loss = 0.05;
        }
        let mut act = Actuator::motor("m", "leg", 1, 2.5);
        act.ctrl_range = Some([-3.0, 3.0]);
        model.actuators.push(act);
        model
    }

    fn rand_state(model: &Model, seed: u64) -> State {
        let mut s = model.default_state();
        let mut x = seed;
        let mut next = || {
            // xorshift
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x as f64 / u64::MAX as f64) - 0.5
        };
        for i in 0..model.nq {
            s.q[i] = 0.6 * next();
        }
        for i in 0..model.nv {
            s.v[i] = 0.8 * next();
        }
        s.ctrl = DVec::zeros(model.nv);
        for i in 0..model.nv {
            s.ctrl[i] = next();
        }
        s
    }

    fn lift_inertias(model: &Model) -> Vec<SpatialInertia<f64>> {
        model.bodies.iter().map(|b| b.inertia).collect()
    }

    /// The f64 instantiation must reproduce the concrete engine evaluations —
    /// this is the anti-drift pin that makes the mirrors legitimate.
    #[test]
    fn f64_mirrors_match_concrete_engine() {
        let model = skate_like_model();
        for seed in [3u64, 17, 91] {
            let state = rand_state(&model, seed);
            let inertias = lift_inertias(&model);

            // FK.
            let (xf_gen, vel_gen) = fk_gen::<f64>(&model, state.q.as_slice(), state.v.as_slice());
            let (xf, vel) = phyz_rigid::forward_kinematics(&model, &state);
            for i in 0..model.nbodies() {
                assert!((xf_gen[i].pos - xf[i].pos).norm() < 1e-14, "fk pos {i}");
                for c in 0..3 {
                    assert!(
                        (xf_gen[i].rot.col(c) - xf[i].rot.col(c)).norm() < 1e-14,
                        "fk rot {i}"
                    );
                }
                assert!(
                    (vel_gen[i].angular - vel[i].angular).norm() < 1e-14
                        && (vel_gen[i].linear - vel[i].linear).norm() < 1e-14,
                    "fk vel {i}"
                );
            }

            // ABA, with an external wrench to exercise that channel.
            let ext: Vec<SpatialVec<f64>> = (0..model.nbodies())
                .map(|i| {
                    SpatialVec::new(
                        Vec3::new(0.1 * i as f64, -0.2, 0.05),
                        Vec3::new(0.3, 0.1 * i as f64, -0.4),
                    )
                })
                .collect();
            let qdd_gen = aba_gen::<f64>(
                &model,
                &inertias,
                state.q.as_slice(),
                state.v.as_slice(),
                state.ctrl.as_slice(),
                Some(&ext),
            );
            let qdd = phyz_rigid::aba_with_external_forces(&model, &state, Some(&ext));
            for i in 0..model.nv {
                assert!(
                    (qdd_gen[i] - qdd[i]).abs() < 1e-9 * qdd[i].abs().max(1.0),
                    "aba[{i}]: {} vs {}",
                    qdd_gen[i],
                    qdd[i]
                );
            }

            // CRBA.
            let m_gen = crba_gen::<f64>(&model, &inertias, state.q.as_slice());
            let m = phyz_rigid::crba(&model, &state);
            for r in 0..model.nv {
                for c in 0..model.nv {
                    assert!(
                        (m_gen[r * model.nv + c] - m[(r, c)]).abs() < 1e-11,
                        "crba[{r},{c}]"
                    );
                }
            }

            // Point Jacobian.
            let p = Vec3::new(0.3, -0.2, 0.15);
            let j_gen = point_jacobian_gen::<f64>(&model, &xf, 1, p);
            let j = phyz_rigid::point_jacobian(&model, &xf, 1, p);
            for r in 0..3 {
                for c in 0..model.nv {
                    assert!(
                        (j_gen[r * model.nv + c] - j[(r, c)]).abs() < 1e-13,
                        "jac[{r},{c}]"
                    );
                }
            }

            // Configuration update.
            let q_gen = integrate_configuration_gen::<f64>(
                &model,
                state.q.as_slice(),
                state.v.as_slice(),
                model.dt,
            );
            let mut q_ref = state.q.clone();
            phyz_rigid::integrate_configuration(
                &model,
                q_ref.as_mut_slice(),
                state.v.as_slice(),
                model.dt,
            );
            for i in 0..model.nq {
                assert!((q_gen[i] - q_ref[i]).abs() < 1e-14, "phi[{i}]");
            }
        }
    }

    /// Material laws at f64 match the concrete ones across both sides of the
    /// margin and the restitution ramp's three regions.
    #[test]
    fn material_mirrors_match() {
        let mat = ContactMaterial::default();
        for depth in [-2e-3, -5e-4, -1e-9, 0.0, 1e-9, 2e-4, 5e-3] {
            let a = impedance_at_gen::<f64>(&mat, depth);
            let b = mat.impedance_at(depth);
            assert!((a - b).abs() < 1e-14, "impedance_at({depth}): {a} vs {b}");
        }
        for (e, s, vr) in [
            (0.6, 0.0, 1e-3),
            (0.6, 5e-4, 1e-3),
            (0.6, 1.5e-3, 1e-3),
            (0.6, 3e-3, 1e-3),
            (0.6, 0.5, 0.0),
        ] {
            let a = effective_restitution_gen::<f64>(e, s, vr);
            let b = phyz_contact::ContactProblem::effective_restitution(e, s, vr);
            assert!((a - b).abs() < 1e-14, "restitution({e},{s},{vr})");
        }
    }

    /// Reverse-mode gradients of the generic ABA agree with forward-mode
    /// duals of the same mirror — forward and reverse are independent
    /// implementations of the same derivative.
    #[test]
    fn rev_aba_matches_dual_aba() {
        use crate::rev::{Rev, backward, tape_scope};
        use tang::Dual;

        let model = skate_like_model();
        let state = rand_state(&model, 7);
        let inertias = lift_inertias(&model);

        // Reverse: gradient of a fixed contraction of qdd wrt every (q, v, u).
        let weights: Vec<f64> = (0..model.nv).map(|i| 0.3 + 0.1 * i as f64).collect();
        let _g = tape_scope();
        let q: Vec<Rev> = state.q.as_slice().iter().map(|&x| Rev::var(x)).collect();
        let v: Vec<Rev> = state.v.as_slice().iter().map(|&x| Rev::var(x)).collect();
        let u: Vec<Rev> = state.ctrl.as_slice().iter().map(|&x| Rev::var(x)).collect();
        let inert_rev: Vec<SpatialInertia<Rev>> = inertias
            .iter()
            .map(|si| {
                SpatialInertia::new(
                    Rev::constant(si.mass),
                    lift_v3(si.com),
                    super::lift_m3(&si.inertia),
                )
            })
            .collect();
        let qdd = aba_gen::<Rev>(&model, &inert_rev, &q, &v, &u, None);
        let mut s = Rev::constant(0.0);
        for (w, qi) in weights.iter().zip(&qdd) {
            s += Rev::constant(*w) * *qi;
        }
        let g = backward(s);

        // Forward: one dual lane per input, same contraction.
        let check = |kind: usize, idx: usize, got: f64| {
            let lift = |vals: &[f64], k: usize| -> Vec<Dual<f64>> {
                vals.iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        if kind == k && i == idx {
                            Dual::var(x)
                        } else {
                            Dual::constant(x)
                        }
                    })
                    .collect()
            };
            let qd = lift(state.q.as_slice(), 0);
            let vd = lift(state.v.as_slice(), 1);
            let ud = lift(state.ctrl.as_slice(), 2);
            let inert_d: Vec<SpatialInertia<Dual<f64>>> = inertias
                .iter()
                .map(|si| {
                    SpatialInertia::new(
                        Dual::constant(si.mass),
                        lift_v3(si.com),
                        super::lift_m3(&si.inertia),
                    )
                })
                .collect();
            let qdd_d = aba_gen::<Dual<f64>>(&model, &inert_d, &qd, &vd, &ud, None);
            let mut want = 0.0;
            for (w, qi) in weights.iter().zip(&qdd_d) {
                want += w * qi.dual;
            }
            assert!(
                (got - want).abs() < 1e-9 * want.abs().max(1.0),
                "kind {kind} idx {idx}: rev {got} vs dual {want}"
            );
        };
        for (i, qi) in q.iter().enumerate() {
            check(0, i, g.of(*qi));
        }
        for i in 0..model.nv {
            check(1, i, g.of(v[i]));
            check(2, i, g.of(u[i]));
        }
    }
}
