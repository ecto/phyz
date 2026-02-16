//! Differentiation utilities for tau: Jacobians of dynamics.
//!
//! Provides both finite-difference and (future) analytical derivatives
//! of the equations of motion.

use tau_math::{DMat, DVec};
use tau_model::{Model, State};
use tau_rigid::aba;

/// Jacobians of a single simulation step.
///
/// For state (q, v) → (q', v') after one step:
/// - `dqnext_dq`: ∂q'/∂q
/// - `dqnext_dv`: ∂q'/∂v
/// - `dvnext_dq`: ∂v'/∂q
/// - `dvnext_dv`: ∂v'/∂v
/// - `dvnext_dctrl`: ∂v'/∂ctrl
#[derive(Debug, Clone)]
pub struct StepJacobians {
    pub dqnext_dq: DMat,
    pub dqnext_dv: DMat,
    pub dvnext_dq: DMat,
    pub dvnext_dv: DMat,
    pub dvnext_dctrl: DMat,
}

/// Compute step Jacobians via finite differences.
///
/// Perturbs each component of (q, v, ctrl) and measures the change in (q', v')
/// after a semi-implicit Euler step.
pub fn finite_diff_jacobians(model: &Model, state: &State, eps: f64) -> StepJacobians {
    let nq = model.nq;
    let nv = model.nv;
    let dt = model.dt;

    // Nominal step (used as reference for central differences)
    let (_q_nom, _v_nom) = semi_implicit_euler_step(model, state, dt);

    let mut dqnext_dq = DMat::zeros(nq, nq);
    let mut dqnext_dv = DMat::zeros(nq, nv);
    let mut dvnext_dq = DMat::zeros(nv, nq);
    let mut dvnext_dv = DMat::zeros(nv, nv);
    let mut dvnext_dctrl = DMat::zeros(nv, nv);

    // Perturb q
    for j in 0..nq {
        let mut s_plus = state.clone();
        s_plus.q[j] += eps;
        let (qp, vp) = semi_implicit_euler_step(model, &s_plus, dt);

        let mut s_minus = state.clone();
        s_minus.q[j] -= eps;
        let (qm, vm) = semi_implicit_euler_step(model, &s_minus, dt);

        let inv_2eps = 1.0 / (2.0 * eps);
        dqnext_dq.set_column(j, &((&qp - &qm) * inv_2eps));
        dvnext_dq.set_column(j, &((&vp - &vm) * inv_2eps));
    }

    // Perturb v
    for j in 0..nv {
        let mut s_plus = state.clone();
        s_plus.v[j] += eps;
        let (qp, vp) = semi_implicit_euler_step(model, &s_plus, dt);

        let mut s_minus = state.clone();
        s_minus.v[j] -= eps;
        let (qm, vm) = semi_implicit_euler_step(model, &s_minus, dt);

        let inv_2eps = 1.0 / (2.0 * eps);
        dqnext_dv.set_column(j, &((&qp - &qm) * inv_2eps));
        dvnext_dv.set_column(j, &((&vp - &vm) * inv_2eps));
    }

    // Perturb ctrl
    for j in 0..nv {
        let mut s_plus = state.clone();
        s_plus.ctrl[j] += eps;
        let (_, vp) = semi_implicit_euler_step(model, &s_plus, dt);

        let mut s_minus = state.clone();
        s_minus.ctrl[j] -= eps;
        let (_, vm) = semi_implicit_euler_step(model, &s_minus, dt);

        let inv_2eps = 1.0 / (2.0 * eps);
        dvnext_dctrl.set_column(j, &((&vp - &vm) * inv_2eps));
    }

    StepJacobians {
        dqnext_dq,
        dqnext_dv,
        dvnext_dq,
        dvnext_dv,
        dvnext_dctrl,
    }
}

/// Compute analytical step Jacobians for semi-implicit Euler.
///
/// Semi-implicit Euler:
///   qdd = ABA(q, v, ctrl)
///   v' = v + dt * qdd
///   q' = q + dt * v'
///
/// Derivatives via chain rule through ABA.
/// Uses finite differences on ABA itself for now (analytical ABA derivatives in future PR).
pub fn analytical_step_jacobians(model: &Model, state: &State) -> StepJacobians {
    let nq = model.nq;
    let nv = model.nv;
    let dt = model.dt;
    let eps = 1e-7;

    // Compute ABA Jacobians via finite differences on the acceleration function
    let _qdd_nom = aba(model, state);

    let mut dqdd_dq = DMat::zeros(nv, nq);
    let mut dqdd_dv = DMat::zeros(nv, nv);
    let mut dqdd_dctrl = DMat::zeros(nv, nv);

    for j in 0..nq {
        let mut sp = state.clone();
        sp.q[j] += eps;
        let qddp = aba(model, &sp);

        let mut sm = state.clone();
        sm.q[j] -= eps;
        let qddm = aba(model, &sm);

        dqdd_dq.set_column(j, &((&qddp - &qddm) / (2.0 * eps)));
    }

    for j in 0..nv {
        let mut sp = state.clone();
        sp.v[j] += eps;
        let qddp = aba(model, &sp);

        let mut sm = state.clone();
        sm.v[j] -= eps;
        let qddm = aba(model, &sm);

        dqdd_dv.set_column(j, &((&qddp - &qddm) / (2.0 * eps)));
    }

    for j in 0..nv {
        let mut sp = state.clone();
        sp.ctrl[j] += eps;
        let qddp = aba(model, &sp);

        let mut sm = state.clone();
        sm.ctrl[j] -= eps;
        let qddm = aba(model, &sm);

        dqdd_dctrl.set_column(j, &((&qddp - &qddm) / (2.0 * eps)));
    }

    // Semi-implicit Euler derivatives:
    // v' = v + dt * qdd(q, v, ctrl)
    // q' = q + dt * v'
    //
    // dv'/dq = dt * dqdd/dq
    // dv'/dv = I + dt * dqdd/dv
    // dv'/dctrl = dt * dqdd/dctrl
    //
    // dq'/dq = I + dt * dv'/dq = I + dt² * dqdd/dq
    // dq'/dv = dt * (I + dt * dqdd/dv) = dt * dv'/dv
    let i_nv = DMat::identity(nv, nv);
    let i_nq = DMat::identity(nq, nq);

    let dvnext_dq = &dqdd_dq * dt;
    let dvnext_dv = &i_nv + &dqdd_dv * dt;
    let dvnext_dctrl = &dqdd_dctrl * dt;
    let dqnext_dq = &i_nq + &dvnext_dq * dt;
    let dqnext_dv = &dvnext_dv * dt;

    StepJacobians {
        dqnext_dq,
        dqnext_dv,
        dvnext_dq,
        dvnext_dv,
        dvnext_dctrl,
    }
}

/// Semi-implicit Euler step (returns new q, v).
fn semi_implicit_euler_step(model: &Model, state: &State, dt: f64) -> (DVec, DVec) {
    let qdd = aba(model, state);
    let v_new = &state.v + &qdd * dt;
    let q_new = &state.q + &v_new * dt;
    (q_new, v_new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tau_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
    use tau_model::ModelBuilder;

    fn make_pendulum() -> Model {
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "link1",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(
                    1.0,
                    Vec3::new(0.0, -0.5, 0.0),
                    Mat3::from_diagonal(&Vec3::new(1.0 / 12.0, 0.0, 1.0 / 12.0)),
                ),
            )
            .build()
    }

    #[test]
    fn test_jacobians_match() {
        let model = make_pendulum();
        let mut state = model.default_state();
        state.q[0] = 0.3;
        state.v[0] = 0.1;

        let fd = finite_diff_jacobians(&model, &state, 1e-6);
        let an = analytical_step_jacobians(&model, &state);

        // Check that analytical matches finite-diff
        let eps = 1e-4;
        assert!(
            (&fd.dqnext_dq - &an.dqnext_dq).norm() < eps,
            "dqnext_dq mismatch: fd={:.6}, an={:.6}",
            fd.dqnext_dq,
            an.dqnext_dq
        );
        assert!(
            (&fd.dvnext_dq - &an.dvnext_dq).norm() < eps,
            "dvnext_dq mismatch"
        );
        assert!(
            (&fd.dvnext_dv - &an.dvnext_dv).norm() < eps,
            "dvnext_dv mismatch"
        );
        assert!(
            (&fd.dvnext_dctrl - &an.dvnext_dctrl).norm() < eps,
            "dvnext_dctrl mismatch"
        );
    }
}
