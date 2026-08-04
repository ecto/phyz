//! [`Solver`] adapter for the Featherstone rigid-body solver in `phyz-rigid`.

use phyz_math::{SpatialVec, Vec3};
use phyz_model::{Model, State};
use phyz_rigid::{aba_with_external_forces, forward_kinematics, total_energy};

use crate::coupling::SolverType;
use crate::solver::{CouplingSite, ExternalInput, Solver};

/// Per-body coupling attributes that the rigid model itself does not carry.
#[derive(Clone, Copy, Debug)]
pub struct BodyCoupling {
    /// Index into `model.bodies`.
    pub body: usize,
    /// Electric charge on the body (C).
    pub charge: f64,
}

/// A rigid-body domain exposed to the coupling layer.
///
/// Wraps a [`Model`] + [`State`] pair and integrates with semi-implicit Euler
/// on top of `aba_with_external_forces`, which is what the rest of the
/// workspace uses. Coupling forces arrive as world-frame [`ExternalInput::Force`]
/// and are converted to body-frame spatial forces for ABA.
pub struct RigidSolver {
    /// The kinematic model.
    pub model: Model,
    /// Current state.
    pub state: State,
    /// Bodies exposed as coupling sites, with their charges.
    pub coupled_bodies: Vec<BodyCoupling>,
    /// Queued external spatial forces, one per body, in body frame.
    pending: Vec<SpatialVec>,
}

impl RigidSolver {
    /// Create a rigid solver from a model and state.
    ///
    /// No bodies are exposed for coupling until [`Self::couple_body`] is called.
    pub fn new(model: Model, state: State) -> Self {
        let nb = model.nbodies();
        Self {
            model,
            state,
            coupled_bodies: Vec::new(),
            pending: vec![SpatialVec::zero(); nb],
        }
    }

    /// Expose `body` to the coupling layer with the given charge.
    ///
    /// Returns the site id to use in [`ExternalInput::Force`].
    pub fn couple_body(&mut self, body: usize, charge: f64) -> usize {
        assert!(body < self.model.nbodies(), "body index out of range");
        self.coupled_bodies.push(BodyCoupling { body, charge });
        self.coupled_bodies.len() - 1
    }

    /// World-frame position and linear velocity of a body's origin.
    fn body_kinematics(&self, body: usize) -> (Vec3, Vec3) {
        let (xforms, vels) = forward_kinematics(&self.model, &self.state);
        let xf = &xforms[body];
        // `xf` is world→body; its translation is the body origin in world
        // coordinates and `rot` maps world vectors into the body frame.
        let position = xf.pos;
        let velocity = xf.rot.transpose() * vels[body].linear;
        (position, velocity)
    }

    /// Semi-implicit Euler step: `v += qdd·dt`, then a configuration step.
    ///
    /// Delegates the `q` update to [`phyz_rigid::integrate_configuration`],
    /// which is the one place that knows each joint type's parameterisation
    /// (ball and free joints store rotation as exponential coordinates, and a
    /// free joint's linear velocity is body-frame).
    fn integrate(&mut self, dt: f64) {
        let qdd = aba_with_external_forces(&self.model, &self.state, Some(&self.pending));
        self.state.v += &(&qdd * dt);
        let v = self.state.v.clone();
        phyz_rigid::integrate_configuration(
            &self.model,
            self.state.q.as_mut_slice(),
            v.as_slice(),
            dt,
        );
        self.state.time += dt;
    }
}

impl Solver for RigidSolver {
    fn solver_type(&self) -> SolverType {
        SolverType::RigidBody
    }

    fn natural_dt(&self) -> f64 {
        self.model.dt
    }

    fn time(&self) -> f64 {
        self.state.time
    }

    fn apply_external(&mut self, input: ExternalInput) {
        match input {
            ExternalInput::Force {
                site,
                force,
                torque,
            } => {
                let Some(bc) = self.coupled_bodies.get(site) else {
                    return;
                };
                let body = bc.body;
                let (xforms, _) = forward_kinematics(&self.model, &self.state);
                // ABA takes external forces as body-frame spatial forces
                // (torque; force). `xf.rot` maps world → body.
                let rot = xforms[body].rot;
                let f_body = rot * force;
                let t_body = rot * torque;
                self.pending[body] = self.pending[body] + SpatialVec::new(t_body, f_body);
            }
            // A rigid domain has an explicit momentum integral, so a Reaction
            // entry is pure bookkeeping here and needs no state change.
            ExternalInput::Reaction { .. } => {}
            // Rigid bodies are not a current sink.
            ExternalInput::Current { .. } => {}
        }
    }

    fn advance(&mut self, dt: f64) {
        self.integrate(dt);
        for f in &mut self.pending {
            *f = SpatialVec::zero();
        }
    }

    fn energy(&self) -> f64 {
        total_energy(&self.model, &self.state)
    }

    fn sites(&self) -> Vec<CouplingSite> {
        self.coupled_bodies
            .iter()
            .enumerate()
            .map(|(id, bc)| {
                let (position, velocity) = self.body_kinematics(bc.body);
                CouplingSite {
                    id,
                    position,
                    velocity,
                    mass: self.model.bodies[bc.body].inertia.mass,
                    charge: bc.charge,
                }
            })
            .collect()
    }

    fn momentum(&self) -> Vec3 {
        phyz_guardian::total_momentum(&self.model, &self.state)
    }
}
