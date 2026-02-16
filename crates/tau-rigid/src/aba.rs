//! Articulated Body Algorithm (ABA) — O(n) forward dynamics.
//!
//! Given (q, v, tau), compute qdd (joint accelerations).
//! Three passes over the kinematic tree:
//! 1. Forward pass: compute velocities, bias forces
//! 2. Backward pass: compute articulated inertias, bias forces
//! 3. Forward pass: compute accelerations

use tau_math::SpatialTransform;
use tau_math::{DVec, SpatialMat, SpatialVec, Vec3};
use tau_model::{Model, State};

/// Run the Articulated Body Algorithm.
///
/// Returns generalized accelerations `qdd` of dimension `model.nv`.
pub fn aba(model: &Model, state: &State) -> DVec {
    let nb = model.nbodies();
    let mut qdd = DVec::zeros(model.nv);

    // Per-body storage
    let mut x_tree = vec![SpatialTransform::identity(); nb]; // parent-to-child transform
    let mut vel = vec![SpatialVec::zero(); nb]; // spatial velocity in body frame
    let mut c_bias = vec![SpatialVec::zero(); nb]; // Coriolis/centrifugal bias
    let mut p_a = vec![SpatialVec::zero(); nb]; // articulated bias force
    let mut i_a = vec![SpatialMat::zero(); nb]; // articulated body inertia

    // Gravity as spatial acceleration (expressed as base acceleration trick)
    // Instead of adding gravity forces to each body, we pretend the base accelerates upward.
    let a0 = SpatialVec::new(Vec3::zeros(), -model.gravity);

    // ── Pass 1: Forward — velocities and bias ──
    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let q_idx = model.q_offsets[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];

        let q = state.q[q_idx];
        let qd = state.v[v_idx];

        // Compute tree transform
        let x_joint = joint.joint_transform(q);
        x_tree[i] = x_joint.compose(&joint.parent_to_joint);

        let s_i = joint.motion_subspace();

        if body.parent < 0 {
            vel[i] = s_i * qd;
            c_bias[i] = SpatialVec::zero(); // no Coriolis for root with zero parent velocity
        } else {
            let pi = body.parent as usize;
            let v_parent = x_tree[i].apply_motion(&vel[pi]);
            vel[i] = v_parent + s_i * qd;
            c_bias[i] = vel[i].cross_motion(&(s_i * qd));
        }

        // Initialize articulated inertia with rigid body inertia
        i_a[i] = body.inertia.to_matrix();
        // Bias force: v × I*v (gyroscopic)
        p_a[i] = vel[i].cross_force(&i_a[i].mul_vec(&vel[i]));
    }

    // ── Pass 2: Backward — articulated inertias and forces ──
    for i in (0..nb).rev() {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let s_i = joint.motion_subspace();

        // Applied torque
        let tau_i = state.ctrl[v_idx] - joint.damping * state.v[v_idx];

        // Articulated quantities for this body
        let ia = &i_a[i];
        let u_i = tau_i - s_i.dot(&p_a[i]);
        let d_i = s_i.dot(&ia.mul_vec(&s_i));

        if d_i.abs() < 1e-20 {
            continue;
        }

        let u_inv_d = u_i / d_i;
        let ia_s = ia.mul_vec(&s_i);

        if body.parent >= 0 {
            let pi = body.parent as usize;

            // I_a^A = I_a - I_a * s * (s^T * I_a * s)^-1 * s^T * I_a
            let outer = outer_product_6(&ia_s, &ia.mul_vec(&s_i));
            let ia_new = SpatialMat::from_mat6(ia.data - outer.data / d_i);

            // p_a^A = p_a + I_a^A * c + I_a * s * u / d
            let p_new = p_a[i] + ia_new.mul_vec(&c_bias[i]) + ia_s * u_inv_d;

            // Transform articulated inertia from body to parent frame: I_parent = X^T I X
            let x_mot = x_tree[i].to_motion_matrix();
            let ia_parent = SpatialMat::from_mat6(x_mot.transpose() * ia_new.data * x_mot);
            let p_parent = x_tree[i].inv_apply_force(&p_new);

            i_a[pi] = i_a[pi] + ia_parent;
            p_a[pi] = p_a[pi] + p_parent;
        }
    }

    // ── Pass 3: Forward — accelerations ──
    let mut acc = vec![SpatialVec::zero(); nb];

    for i in 0..nb {
        let body = &model.bodies[i];
        let joint = &model.joints[body.joint_idx];
        let v_idx = model.v_offsets[body.joint_idx];
        let s_i = joint.motion_subspace();

        let a_parent = if body.parent < 0 {
            x_tree[i].apply_motion(&a0)
        } else {
            let pi = body.parent as usize;
            x_tree[i].apply_motion(&acc[pi])
        };

        let ia = &i_a[i];
        let d_i = s_i.dot(&ia.mul_vec(&s_i));

        if d_i.abs() < 1e-20 {
            acc[i] = a_parent + c_bias[i];
            continue;
        }

        let tau_i = state.ctrl[v_idx] - joint.damping * state.v[v_idx];
        let u_i = tau_i - s_i.dot(&p_a[i]);

        let qdd_i = (u_i - ia.mul_vec(&(a_parent + c_bias[i])).dot(&s_i)) / d_i;
        qdd[v_idx] = qdd_i;

        acc[i] = a_parent + c_bias[i] + s_i * qdd_i;
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
    use tau_math::{GRAVITY, SpatialInertia};
    use tau_model::ModelBuilder;

    fn make_double_pendulum() -> tau_model::Model {
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
                    tau_math::Mat3::from_diagonal(&Vec3::new(
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
                    tau_math::Mat3::from_diagonal(&Vec3::new(
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
        // Simple pendulum: revolute about Z, gravity along -Y, rod hangs in -Y at q=0.
        // At q=π/2, rod points along +X → max gravity torque about Z.
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
                    Vec3::new(0.0, -length / 2.0, 0.0), // CoM halfway down -Y
                    tau_math::Mat3::from_diagonal(&Vec3::new(
                        mass * length * length / 12.0,
                        0.0,
                        mass * length * length / 12.0,
                    )),
                ),
            )
            .build();

        // At q = π/2 (rod pointing along +X), gravity torque is maximal
        let mut state = model.default_state();
        state.q[0] = std::f64::consts::FRAC_PI_2;

        let qdd = aba(&model, &state);

        // qdd = -(m*g*L/2) / I_total where I_total = mL²/3
        let i_total = mass * length * length / 3.0;
        let expected_qdd = -(mass * GRAVITY * length / 2.0) / i_total;

        assert!(
            (qdd[0] - expected_qdd).abs() < 1e-6,
            "qdd = {}, expected = {}",
            qdd[0],
            expected_qdd
        );
    }
}
