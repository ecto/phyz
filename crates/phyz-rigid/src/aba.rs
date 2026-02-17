//! Articulated Body Algorithm (ABA) — O(n) forward dynamics.
//!
//! Given (q, v, tau), compute qdd (joint accelerations).
//! Three passes over the kinematic tree:
//! 1. Forward pass: compute velocities, bias forces
//! 2. Backward pass: compute articulated inertias, bias forces
//! 3. Forward pass: compute accelerations
//!
//! Supports single-DOF (revolute, prismatic) and multi-DOF (spherical, free) joints.

use phyz_math::SpatialTransform;
use phyz_math::{DVec, Mat6, SpatialMat, SpatialVec, Vec3, Vec6};
use phyz_model::{JointType, Model, State};

/// Compute joint velocity contribution S * qd for any joint type.
fn joint_velocity(joint: &phyz_model::Joint, qd: &[f64]) -> SpatialVec {
    match joint.joint_type {
        JointType::Revolute | JointType::Hinge => {
            SpatialVec::new(joint.axis * qd[0], Vec3::zeros())
        }
        JointType::Prismatic | JointType::Slide => {
            SpatialVec::new(Vec3::zeros(), joint.axis * qd[0])
        }
        JointType::Spherical | JointType::Ball => {
            SpatialVec::new(Vec3::new(qd[0], qd[1], qd[2]), Vec3::zeros())
        }
        JointType::Free => SpatialVec::new(
            Vec3::new(qd[0], qd[1], qd[2]),
            Vec3::new(qd[3], qd[4], qd[5]),
        ),
        JointType::Fixed => SpatialVec::zero(),
    }
}

/// Run the Articulated Body Algorithm.
///
/// Returns generalized accelerations `qdd` of dimension `model.nv`.
/// Accepts optional external forces per body (in body frame).
pub fn aba(model: &Model, state: &State) -> DVec {
    aba_with_external_forces(model, state, None)
}

/// Run ABA with optional external spatial forces applied to each body.
pub fn aba_with_external_forces(
    model: &Model,
    state: &State,
    external_forces: Option<&[SpatialVec]>,
) -> DVec {
    let nb = model.nbodies();
    let mut qdd = DVec::zeros(model.nv);

    // Per-body storage
    let mut x_tree = vec![SpatialTransform::identity(); nb];
    let mut vel = vec![SpatialVec::zero(); nb];
    let mut c_bias = vec![SpatialVec::zero(); nb];
    let mut p_a = vec![SpatialVec::zero(); nb]; // articulated bias force
    let mut i_a = vec![SpatialMat::zero(); nb]; // articulated body inertia

    // Gravity as spatial acceleration (base acceleration trick)
    let a0 = SpatialVec::new(Vec3::zeros(), -model.gravity);

    // ── Pass 1: Forward — velocities and bias ──
    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let q_idx = model.q_offsets[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let ndof = joint.ndof();

        // Compute tree transform
        let x_joint = if ndof == 0 {
            SpatialTransform::identity()
        } else {
            let q_slice = &state.q.as_slice()[q_idx..q_idx + ndof];
            joint.joint_transform_slice(q_slice)
        };
        x_tree[i] = x_joint.compose(&joint.parent_to_joint);

        let v_joint = joint_velocity(joint, &state.v.as_slice()[v_idx..v_idx + ndof]);

        if body.parent < 0 {
            vel[i] = v_joint;
            c_bias[i] = SpatialVec::zero();
        } else {
            let pi = body.parent as usize;
            let v_parent = x_tree[i].apply_motion(&vel[pi]);
            vel[i] = v_parent + v_joint;
            c_bias[i] = vel[i].cross_motion(&v_joint);
        }

        // Initialize articulated inertia with rigid body inertia
        i_a[i] = body.inertia.to_matrix();
        // Bias force: v ×* (I*v) (gyroscopic)
        p_a[i] = vel[i].cross_force(&i_a[i].mul_vec(&vel[i]));

        // Subtract external forces from bias (external forces reduce the bias)
        if let Some(ext) = external_forces {
            p_a[i] = p_a[i] - ext[i];
        }
    }

    // ── Pass 2: Backward — articulated inertias and forces ──
    for i in (0..nb).rev() {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let ndof = joint.ndof();

        if ndof == 0 {
            // Fixed joint: just propagate to parent
            if body.parent >= 0 {
                let pi = body.parent as usize;
                let x_mot = x_tree[i].to_motion_matrix();
                let ia_parent = SpatialMat::from_mat6(x_mot.transpose() * i_a[i].data * x_mot);
                let p_parent = x_tree[i].inv_apply_force(&p_a[i]);
                i_a[pi] = i_a[pi] + ia_parent;
                p_a[pi] = p_a[pi] + p_parent;
            }
            continue;
        }

        if ndof == 1 {
            // Single-DOF: scalar operations (faster path)
            let s_i = joint.motion_subspace();
            let phyz_i = state.ctrl[v_idx] - joint.damping * state.v[v_idx];

            let ia = &i_a[i];
            let u_i = phyz_i - s_i.dot(&p_a[i]);
            let d_i = s_i.dot(&ia.mul_vec(&s_i));

            if d_i.abs() < 1e-20 {
                continue;
            }

            let u_inv_d = u_i / d_i;
            let ia_s = ia.mul_vec(&s_i);

            if body.parent >= 0 {
                let pi = body.parent as usize;
                let outer = outer_product_6(&ia_s, &ia.mul_vec(&s_i));
                let ia_new = SpatialMat::from_mat6(ia.data - outer.data / d_i);
                let p_new = p_a[i] + ia_new.mul_vec(&c_bias[i]) + ia_s * u_inv_d;

                let x_mot = x_tree[i].to_motion_matrix();
                let ia_parent = SpatialMat::from_mat6(x_mot.transpose() * ia_new.data * x_mot);
                let p_parent = x_tree[i].inv_apply_force(&p_new);

                i_a[pi] = i_a[pi] + ia_parent;
                p_a[pi] = p_a[pi] + p_parent;
            }
        } else {
            // Multi-DOF: matrix operations
            let s_mat = joint.motion_subspace_matrix(); // 6 x ndof
            let ia = &i_a[i];

            // tau vector for this joint
            let mut phyz_vec = nalgebra::DVector::<f64>::zeros(ndof);
            for k in 0..ndof {
                phyz_vec[k] = state.ctrl[v_idx + k] - joint.damping * state.v[v_idx + k];
            }

            // U = I_a * S  (6 x ndof)
            let ia_mat = &ia.data;
            let u_mat = ia_mat * &s_mat; // 6 x ndof

            // D = S^T * I_a * S  (ndof x ndof)
            let d_mat = s_mat.transpose() * &u_mat; // ndof x ndof

            // u_vec = tau - S^T * p_a  (ndof x 1)
            let pa_vec = nalgebra::DVector::from_column_slice(p_a[i].data.as_slice());
            let u_vec = &phyz_vec - &(s_mat.transpose() * &pa_vec); // ndof x 1

            // D_inv
            let d_inv = match nalgebra::DMatrix::from_iterator(
                ndof,
                ndof,
                d_mat.iter().cloned(),
            )
            .try_inverse()
            {
                Some(inv) => inv,
                None => continue,
            };

            if body.parent >= 0 {
                let pi = body.parent as usize;

                // I_a^A = I_a - U * D^{-1} * U^T
                let u_dinv = &u_mat * &d_inv; // 6 x ndof
                let ia_new_data = ia_mat - &u_dinv * u_mat.transpose(); // 6 x 6
                let ia_new = SpatialMat::from_mat6(Mat6::from_iterator(ia_new_data.iter().cloned()));

                // p_a^A = p_a + I_a^A * c + U * D^{-1} * u
                let dinv_u = &d_inv * &u_vec; // ndof x 1
                let u_dinv_u = &u_mat * &dinv_u; // 6 x 1
                let c_vec = c_bias[i].data;
                let p_new_data = p_a[i].data + ia_new.data * c_vec + Vec6::from_iterator(u_dinv_u.iter().cloned());
                let p_new = SpatialVec { data: p_new_data };

                let x_mot = x_tree[i].to_motion_matrix();
                let ia_parent = SpatialMat::from_mat6(x_mot.transpose() * ia_new.data * x_mot);
                let p_parent = x_tree[i].inv_apply_force(&p_new);

                i_a[pi] = i_a[pi] + ia_parent;
                p_a[pi] = p_a[pi] + p_parent;
            }
        }
    }

    // ── Pass 3: Forward — accelerations ──
    let mut acc = vec![SpatialVec::zero(); nb];

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

        if ndof == 1 {
            let s_i = joint.motion_subspace();
            let ia = &i_a[i];
            let d_i = s_i.dot(&ia.mul_vec(&s_i));

            if d_i.abs() < 1e-20 {
                acc[i] = a_parent + c_bias[i];
                continue;
            }

            let phyz_i = state.ctrl[v_idx] - joint.damping * state.v[v_idx];
            let u_i = phyz_i - s_i.dot(&p_a[i]);
            let qdd_i = (u_i - ia.mul_vec(&(a_parent + c_bias[i])).dot(&s_i)) / d_i;
            qdd[v_idx] = qdd_i;
            acc[i] = a_parent + c_bias[i] + s_i * qdd_i;
        } else {
            let s_mat = joint.motion_subspace_matrix();
            let ia = &i_a[i];
            let ia_mat = &ia.data;

            let mut phyz_vec = nalgebra::DVector::<f64>::zeros(ndof);
            for k in 0..ndof {
                phyz_vec[k] = state.ctrl[v_idx + k] - joint.damping * state.v[v_idx + k];
            }

            let u_mat = ia_mat * &s_mat;
            let d_mat = s_mat.transpose() * &u_mat;

            let d_inv = match nalgebra::DMatrix::from_iterator(
                ndof,
                ndof,
                d_mat.iter().cloned(),
            )
            .try_inverse()
            {
                Some(inv) => inv,
                None => {
                    acc[i] = a_parent + c_bias[i];
                    continue;
                }
            };

            let pa_vec = nalgebra::DVector::from_column_slice(p_a[i].data.as_slice());
            let u_vec = &phyz_vec - &(s_mat.transpose() * &pa_vec);

            let a_total = a_parent + c_bias[i];
            let a_vec = nalgebra::DVector::from_column_slice(a_total.data.as_slice());
            let st_ia_a = s_mat.transpose() * (ia_mat * &a_vec);

            let qdd_vec = &d_inv * (&u_vec - &st_ia_a);

            for k in 0..ndof {
                qdd[v_idx + k] = qdd_vec[k];
            }

            // a_i = a_parent + c + S * qdd
            let s_qdd = &s_mat * &qdd_vec;
            acc[i] = a_total + SpatialVec {
                data: Vec6::from_iterator(s_qdd.iter().cloned()),
            };
        }
    }

    qdd
}

/// Outer product of two 6D spatial vectors.
fn outer_product_6(a: &SpatialVec, b: &SpatialVec) -> SpatialMat {
    SpatialMat::from_mat6(a.data * b.data.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{GRAVITY, SpatialInertia};
    use phyz_model::ModelBuilder;

    fn make_double_pendulum() -> phyz_model::Model {
        let length = 1.0;
        let mass = 1.0;
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "link1",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(
                    mass,
                    Vec3::new(0.0, -length / 2.0, 0.0),
                    phyz_math::Mat3::from_diagonal(&Vec3::new(
                        mass * length * length / 12.0,
                        0.0,
                        mass * length * length / 12.0,
                    )),
                ),
            )
            .add_revolute_body(
                "link2",
                0,
                SpatialTransform::translation(Vec3::new(0.0, -length, 0.0)),
                SpatialInertia::new(
                    mass,
                    Vec3::new(0.0, -length / 2.0, 0.0),
                    phyz_math::Mat3::from_diagonal(&Vec3::new(
                        mass * length * length / 12.0,
                        0.0,
                        mass * length * length / 12.0,
                    )),
                ),
            )
            .build()
    }

    #[test]
    fn test_double_pendulum_equilibrium() {
        let model = make_double_pendulum();
        let state = model.default_state();
        let qdd = aba(&model, &state);
        assert!(qdd[0].abs() < 1e-10, "qdd[0] = {} at equilibrium", qdd[0]);
        assert!(qdd[1].abs() < 1e-10, "qdd[1] = {} at equilibrium", qdd[1]);
    }

    #[test]
    fn test_aba_rnea_consistency() {
        let model = make_double_pendulum();
        let mut state = model.default_state();
        state.q[0] = 0.3;
        state.q[1] = 0.2;
        state.v[0] = 0.1;
        state.v[1] = -0.1;

        let qdd = aba(&model, &state);
        let tau = crate::rnea(&model, &state, &qdd);

        // ABA with ctrl=0 should give qdd such that RNEA(qdd) = 0
        assert!(tau[0].abs() < 1e-10, "RNEA tau[0] = {}", tau[0]);
        assert!(tau[1].abs() < 1e-10, "RNEA tau[1] = {}", tau[1]);
    }

    #[test]
    fn test_single_pendulum_aba() {
        let length = 1.0;
        let mass = 1.0;

        let model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .add_revolute_body(
                "link1",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(
                    mass,
                    Vec3::new(0.0, -length / 2.0, 0.0),
                    phyz_math::Mat3::from_diagonal(&Vec3::new(
                        mass * length * length / 12.0,
                        0.0,
                        mass * length * length / 12.0,
                    )),
                ),
            )
            .build();

        let mut state = model.default_state();
        state.q[0] = std::f64::consts::FRAC_PI_2;

        let qdd = aba(&model, &state);

        let i_total = mass * length * length / 3.0;
        let expected_qdd = -(mass * GRAVITY * length / 2.0) / i_total;

        assert!(
            (qdd[0] - expected_qdd).abs() < 1e-6,
            "qdd = {}, expected = {}",
            qdd[0],
            expected_qdd
        );
    }

    #[test]
    fn test_free_joint_freefall() {
        // A free body in gravity should accelerate downward at g
        let model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
            .add_free_body(
                "ball",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::sphere(1.0, 0.1),
            )
            .build();

        let state = model.default_state();
        let qdd = aba(&model, &state);

        // Free joint DOFs: [wx, wy, wz, vx, vy, vz] mapping to q = [x, y, z, wx, wy, wz]
        // Accelerations: angular should be 0, linear z should be -g
        assert!(qdd[0].abs() < 1e-10, "ang_x accel = {}", qdd[0]);
        assert!(qdd[1].abs() < 1e-10, "ang_y accel = {}", qdd[1]);
        assert!(qdd[2].abs() < 1e-10, "ang_z accel = {}", qdd[2]);
        assert!(qdd[3].abs() < 1e-10, "lin_x accel = {}", qdd[3]);
        assert!(qdd[4].abs() < 1e-10, "lin_y accel = {}", qdd[4]);
        assert!(
            (qdd[5] - (-GRAVITY)).abs() < 1e-6,
            "lin_z accel = {}, expected = {}",
            qdd[5],
            -GRAVITY
        );
    }
}
