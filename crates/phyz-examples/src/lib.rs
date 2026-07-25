//! Harness crate for the repository-root `examples/` directory.
//!
//! Two jobs:
//!
//! 1. Every file in `examples/` is registered here as a cargo target, so
//!    `cargo build --workspace --examples` type-checks all of them in CI and
//!    the documented API cannot silently rot.
//! 2. It provides the small [`Simulator`] / [`Solver`] shim the older
//!    examples were written against.
//!
//! **The shim is not part of the phyz public API.** `phyz` today exposes the
//! dynamics (`aba`, `rnea`, `crba`) and the differentiable rollout, but no
//! stepper type — callers write their own integrator loop, as
//! `examples/gpu_batch.rs` and `examples/coupled_em_rigid.rs` do. This shim
//! exists only so the pre-existing examples keep building; it is deliberately
//! minimal and lives outside any published crate.

#![warn(missing_docs)]

use phyz_contact::{ContactMaterial, contact_forces, find_ground_contacts};
use phyz_diff::{StepJacobians, semi_implicit_step_jacobians};
use phyz_math::SpatialVec;
use phyz_model::{Model, State};
use phyz_rigid::{aba, aba_with_external_forces, forward_kinematics};

/// Integration scheme used by [`Simulator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Integrator {
    /// Semi-implicit (symplectic) Euler: velocity first, then position.
    SemiImplicitEuler,
    /// Classical explicit fourth-order Runge-Kutta.
    Rk4,
}

/// A minimal fixed-step simulator over [`phyz_rigid::aba()`].
#[derive(Debug, Clone, Copy)]
pub struct Simulator {
    /// The integration scheme to use.
    pub integrator: Integrator,
}

impl Default for Simulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Simulator {
    /// A semi-implicit Euler simulator.
    pub fn new() -> Self {
        Self {
            integrator: Integrator::SemiImplicitEuler,
        }
    }

    /// An RK4 simulator.
    pub fn rk4() -> Self {
        Self {
            integrator: Integrator::Rk4,
        }
    }

    /// Advance `state` by `model.dt`.
    pub fn step(&self, model: &Model, state: &mut State) {
        match self.integrator {
            Integrator::SemiImplicitEuler => semi_implicit_euler(model, state, None),
            Integrator::Rk4 => rk4(model, state),
        }
        state.time += model.dt;
    }

    /// Advance `state` by one step and return the Jacobians of that step.
    ///
    /// The Jacobians come from [`phyz_diff::semi_implicit_step_jacobians`],
    /// which is finite-difference based inside ABA — see that function's docs.
    pub fn step_with_jacobians(&self, model: &Model, state: &mut State) -> StepJacobians {
        let jac = semi_implicit_step_jacobians(model, state);
        self.step(model, state);
        jac
    }

    /// Advance `state` by one step under penalty ground contact at
    /// `ground_height`.
    pub fn step_with_contacts(
        &self,
        model: &Model,
        state: &mut State,
        ground_height: f64,
        material: &ContactMaterial,
    ) {
        forward_kinematics(model, state);

        let geometries: Vec<_> = model.bodies.iter().map(|b| b.geometry.clone()).collect();
        let contacts = find_ground_contacts(state, &geometries, ground_height);
        let materials = vec![material.clone(); contacts.len()];
        let wrenches = contact_forces(&contacts, state, &materials, None);

        semi_implicit_euler(model, state, Some(&wrenches));
        state.time += model.dt;
    }
}

fn semi_implicit_euler(model: &Model, state: &mut State, external: Option<&[SpatialVec]>) {
    let qdd = match external {
        Some(f) => aba_with_external_forces(model, state, Some(f)),
        None => aba(model, state),
    };
    let dt = model.dt;
    for k in 0..model.nv {
        state.v[k] += dt * qdd[k];
    }
    for k in 0..model.nq {
        state.q[k] += dt * state.v[k];
    }
}

fn rk4(model: &Model, state: &mut State) {
    let dt = model.dt;
    let (nq, nv) = (model.nq, model.nv);

    // Derivative of (q, v) at a probe offset from the current state.
    let deriv = |dq: &[f64], dv: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let mut probe = state.clone();
        for k in 0..nq {
            probe.q[k] = state.q[k] + dq[k];
        }
        for k in 0..nv {
            probe.v[k] = state.v[k] + dv[k];
        }
        let qdd = aba(model, &probe);
        (
            (0..nq).map(|k| probe.v[k]).collect(),
            (0..nv).map(|k| qdd[k]).collect(),
        )
    };

    let zero_q = vec![0.0; nq];
    let zero_v = vec![0.0; nv];
    let scale = |a: &[f64], f: f64| -> Vec<f64> { a.iter().map(|x| x * f).collect() };

    let (k1q, k1v) = deriv(&zero_q, &zero_v);
    let (k2q, k2v) = deriv(&scale(&k1q, dt / 2.0), &scale(&k1v, dt / 2.0));
    let (k3q, k3v) = deriv(&scale(&k2q, dt / 2.0), &scale(&k2v, dt / 2.0));
    let (k4q, k4v) = deriv(&scale(&k3q, dt), &scale(&k3v, dt));

    for k in 0..nq {
        state.q[k] += dt / 6.0 * (k1q[k] + 2.0 * k2q[k] + 2.0 * k3q[k] + k4q[k]);
    }
    for k in 0..nv {
        state.v[k] += dt / 6.0 * (k1v[k] + 2.0 * k2v[k] + 2.0 * k3v[k] + k4v[k]);
    }
}

/// A one-step integrator.
pub trait Solver {
    /// Advance `state` by `model.dt`.
    fn step(&self, model: &Model, state: &mut State);
}

/// Semi-implicit Euler as a [`Solver`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SemiImplicitEulerSolver;

impl Solver for SemiImplicitEulerSolver {
    fn step(&self, model: &Model, state: &mut State) {
        semi_implicit_euler(model, state, None);
        state.time += model.dt;
    }
}
