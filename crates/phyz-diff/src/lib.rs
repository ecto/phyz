//! Differentiation utilities for phyz: Jacobians of the equations of motion.
//!
//! Three routines compute the same object — the Jacobians of one semi-implicit
//! Euler step — by different means, trading accuracy against cost:
//!
//! | Function | Method | Exact? |
//! | --- | --- | --- |
//! | [`finite_diff_jacobians`] | central differences over the whole step | no |
//! | [`semi_implicit_step_jacobians`] | chain rule through the integrator, central differences on ABA | no (ABA block is FD) |
//! | [`symbolic::symbolic_step_jacobians`] | symbolic differentiation of the dynamics | yes |
//!
//! Only the symbolic path is derivative-exact. The other two are finite
//! differences and inherit the usual step-size/round-off tradeoff. For exact
//! *parameter* and contact-surface gradients over a whole trajectory, see the
//! dual-number adjoint in [`phyz::diff`](https://docs.rs/phyz) instead.
//!
//! # Example
//!
//! ```
//! use phyz_diff::semi_implicit_step_jacobians;
//! use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
//! use phyz_model::ModelBuilder;
//!
//! let model = ModelBuilder::new()
//!     .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
//!     .dt(0.002)
//!     .add_revolute_body(
//!         "pendulum",
//!         -1,
//!         SpatialTransform::identity(),
//!         SpatialInertia::new(
//!             1.0,
//!             Vec3::new(0.0, -0.5, 0.0),
//!             Mat3::from_diagonal(&Vec3::new(0.083, 0.0, 0.083)),
//!         ),
//!     )
//!     .build();
//!
//! let mut state = model.default_state();
//! state.q[0] = 0.3;
//!
//! let jac = semi_implicit_step_jacobians(&model, &state);
//! assert_eq!(jac.dvnext_dq.nrows(), model.nv);
//! ```

#![warn(missing_docs)]

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

pub mod rollout;
pub mod symbolic;

// The rollout adjoint API is the crate's headline surface; hoist it to the
// root so callers can write `phyz_diff::AdjointRollout` (and so the `phyz`
// facade's `phyz::diff::…` spelling keeps working).
pub use rollout::{
    AdjointGradients, AdjointRollout, CollisionMesh, ContactSetup, FinalStateObjective,
    GroundContact, N_INERTIA_PARAMS, adjoint_rollout_gradient, inertia_params, rollout_objective,
};

use phyz_math::{DMat, DVec};
use phyz_model::{Model, State};
use phyz_rigid::aba;

/// Jacobians of a single simulation step.
///
/// For state (q, v) -> (q', v') after one step:
/// - `dqnext_dq`: dq'/dq
/// - `dqnext_dv`: dq'/dv
/// - `dvnext_dq`: dv'/dq
/// - `dvnext_dv`: dv'/dv
/// - `dvnext_dctrl`: dv'/dctrl
#[derive(Debug, Clone)]
pub struct StepJacobians {
    /// dq'/dq — sensitivity of next positions to current positions.
    pub dqnext_dq: DMat,
    /// dq'/dv — sensitivity of next positions to current velocities.
    pub dqnext_dv: DMat,
    /// dv'/dq — sensitivity of next velocities to current positions.
    pub dvnext_dq: DMat,
    /// dv'/dv — sensitivity of next velocities to current velocities.
    pub dvnext_dv: DMat,
    /// dv'/dctrl — sensitivity of next velocities to applied joint torques.
    pub dvnext_dctrl: DMat,
}

/// Set column `j` of matrix `m` from DVec `col`.
fn set_col(m: &mut DMat, j: usize, col: &DVec) {
    for i in 0..col.len() {
        m[(i, j)] = col[i];
    }
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

    let inv_2eps = 1.0 / (2.0 * eps);

    // Perturb q
    for j in 0..nq {
        let mut s_plus = state.clone();
        s_plus.q[j] += eps;
        let (qp, vp) = semi_implicit_euler_step(model, &s_plus, dt);

        let mut s_minus = state.clone();
        s_minus.q[j] -= eps;
        let (qm, vm) = semi_implicit_euler_step(model, &s_minus, dt);

        set_col(&mut dqnext_dq, j, &(&(&qp - &qm) * inv_2eps));
        set_col(&mut dvnext_dq, j, &(&(&vp - &vm) * inv_2eps));
    }

    // Perturb v
    for j in 0..nv {
        let mut s_plus = state.clone();
        s_plus.v[j] += eps;
        let (qp, vp) = semi_implicit_euler_step(model, &s_plus, dt);

        let mut s_minus = state.clone();
        s_minus.v[j] -= eps;
        let (qm, vm) = semi_implicit_euler_step(model, &s_minus, dt);

        set_col(&mut dqnext_dv, j, &(&(&qp - &qm) * inv_2eps));
        set_col(&mut dvnext_dv, j, &(&(&vp - &vm) * inv_2eps));
    }

    // Perturb ctrl
    for j in 0..nv {
        let mut s_plus = state.clone();
        s_plus.ctrl[j] += eps;
        let (_, vp) = semi_implicit_euler_step(model, &s_plus, dt);

        let mut s_minus = state.clone();
        s_minus.ctrl[j] -= eps;
        let (_, vm) = semi_implicit_euler_step(model, &s_minus, dt);

        set_col(&mut dvnext_dctrl, j, &(&(&vp - &vm) * inv_2eps));
    }

    StepJacobians {
        dqnext_dq,
        dqnext_dv,
        dvnext_dq,
        dvnext_dv,
        dvnext_dctrl,
    }
}

/// Deprecated alias for [`semi_implicit_step_jacobians`].
///
/// The old name claimed the result was analytical; the ABA block is computed
/// by finite differences.
#[deprecated(
    since = "0.1.0",
    note = "this is not analytical — the ABA block uses finite differences; use `semi_implicit_step_jacobians`, or `symbolic::symbolic_step_jacobians` for exact derivatives"
)]
pub fn analytical_step_jacobians(model: &Model, state: &State) -> StepJacobians {
    semi_implicit_step_jacobians(model, state)
}

/// Compute step Jacobians for semi-implicit Euler by chain rule through the
/// integrator.
///
/// Semi-implicit Euler:
///   qdd = ABA(q, v, ctrl)
///   v' = v + dt * qdd
///   q' = q + dt * v'
///
/// The integrator layer is differentiated exactly; the ABA block
/// (∂qdd/∂q, ∂qdd/∂v, ∂qdd/∂ctrl) is still obtained by **central finite
/// differences** on `aba`, so this routine is *not* derivative-exact. It is
/// cheaper and better-conditioned than [`finite_diff_jacobians`], which
/// perturbs the whole step, but it carries the same truncation/round-off
/// tradeoff. For exact derivatives use
/// [`symbolic::symbolic_step_jacobians`].
pub fn semi_implicit_step_jacobians(model: &Model, state: &State) -> StepJacobians {
    let nq = model.nq;
    let nv = model.nv;
    let dt = model.dt;
    let eps = 1e-7;
    let inv_2eps = 1.0 / (2.0 * eps);

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

        set_col(&mut dqdd_dq, j, &(&(&qddp - &qddm) * inv_2eps));
    }

    for j in 0..nv {
        let mut sp = state.clone();
        sp.v[j] += eps;
        let qddp = aba(model, &sp);

        let mut sm = state.clone();
        sm.v[j] -= eps;
        let qddm = aba(model, &sm);

        set_col(&mut dqdd_dv, j, &(&(&qddp - &qddm) * inv_2eps));
    }

    for j in 0..nv {
        let mut sp = state.clone();
        sp.ctrl[j] += eps;
        let qddp = aba(model, &sp);

        let mut sm = state.clone();
        sm.ctrl[j] -= eps;
        let qddm = aba(model, &sm);

        set_col(&mut dqdd_dctrl, j, &(&(&qddp - &qddm) * inv_2eps));
    }

    // Semi-implicit Euler derivatives:
    // v' = v + dt * qdd(q, v, ctrl)
    // q' = q + dt * v'
    //
    // dv'/dq = dt * dqdd/dq
    // dv'/dv = I + dt * dqdd/dv
    // dv'/dctrl = dt * dqdd/dctrl
    //
    // dq'/dq = I + dt * dv'/dq = I + dt^2 * dqdd/dq
    // dq'/dv = dt * (I + dt * dqdd/dv) = dt * dv'/dv
    let i_nv = DMat::identity(nv);
    let i_nq = DMat::identity(nq);

    let dvnext_dq = dqdd_dq.scale(dt);
    let dvnext_dv = &i_nv + &dqdd_dv.scale(dt);
    let dvnext_dctrl = dqdd_dctrl.scale(dt);
    let dqnext_dq = &i_nq + &dvnext_dq.scale(dt);
    let dqnext_dv = dvnext_dv.scale(dt);

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
    let v_new = &state.v + &(&qdd * dt);
    let q_new = &state.q + &(&v_new * dt);
    (q_new, v_new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
    use phyz_model::ModelBuilder;

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
        let an = semi_implicit_step_jacobians(&model, &state);

        // Check that analytical matches finite-diff
        let eps = 1e-4;
        assert!(
            (&fd.dqnext_dq - &an.dqnext_dq).norm() < eps,
            "dqnext_dq mismatch: fd={:?}, an={:?}",
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
