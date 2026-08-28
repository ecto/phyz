//! phyz's differentiable step, as a tang autograd node.
//!
//! Gradients used to cross the phyz→training boundary through hand-rolled
//! glue: a caller assembled a [`ConvexContactRollout`] by hand, named an
//! objective, and read the trajectory-level derivatives back out. That works
//! once. It does not compose, and it does not put physics inside a training
//! loop next to a network.
//!
//! This crate turns one simulation step into an ordinary differentiable op:
//!
//! ```text
//!   forward:   (state_in, ctrl) -> state_out
//!   backward:  cotangent^T · d state_out  ->  (d state_in, d ctrl)
//! ```
//!
//! [`PhysicsStep`] is that op. [`PhysicsTape`] chains N of them and pulls a
//! cotangent back through the chain, which is the whole point: short-horizon
//! analytic policy gradients (SHAC), gradient-based sysid, and trajectory
//! optimisation stop being bespoke programs and become tang training loops.
//! The gradients land in [`tang_train::Parameter`] buffers, so
//! [`tang_train::ModuleAdam`] and friends consume them unchanged.
//!
//! # Which side of the boundary this lives on
//!
//! `phyz-diff` already depends on `tang` (for `Scalar`, `DVec`, `ExprGraph`).
//! A `tang-phyz` crate in the tang workspace would therefore close a
//! dependency cycle. The bridge lives phyz-side; tang stays the leaf.
//!
//! # Where the derivative comes from
//!
//! [`phyz_diff::convex_adjoint_gradient`] is trajectory-level and
//! objective-scoped: it answers `dJ/dq0`, `dJ/dv0`, `dJ/du_t` for a whole
//! rollout under one [`FinalStateObjective`]. A VJP is exactly that with
//! `steps = 1` and a *linear* objective — set `J = w·(q_out, v_out)` and the
//! reported `d_q0`, `d_v0`, `d_ctrl[0]` **are** `wᵀ ∂state_out/∂·`, in one
//! call rather than one call per output component.
//!
//! With `PHYZ_SOLVER_ADJOINT=1` the contact channel is closed by
//! differentiating the solver's own executed sweeps rather than applying the
//! implicit function theorem at a fixed point the solver never reached, so
//! truncated solves still yield a gradient instead of a refusal. This crate
//! does not read the variable — it is a process-wide `OnceLock` inside
//! `phyz-contact` — but everything here is written to work in either mode.
//!
//! # Determinism, and the price of a pure step
//!
//! A custom-VJP node needs its forward to be a *function* of its declared
//! inputs. `Simulator`'s warm-start cache is per-simulator state that a
//! `(state_in, ctrl)` signature does not name, so this op's forward is a
//! fresh-cache one-step rollout — byte-identical to the forward the backward
//! then differentiates, by construction, since it is literally
//! [`phyz_diff::convex_rollout_objective`] with a capturing objective.
//!
//! The consequence is honest and worth stating: an N-step [`PhysicsTape`] is
//! *not* bit-identical to `Simulator::step_with_contacts` called N times with
//! warm starting on. It is identical to the same simulator with
//! `with_warm_start(false)`. Cold-starting every step costs solver iterations
//! and buys a step function whose gradient is the gradient of what ran.

use std::cell::RefCell;

use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_diff::{
    ConvexAdjointError, ConvexContactRollout, FinalStateObjective, convex_adjoint_gradient,
    convex_rollout_objective,
};
use phyz_math::DVec;
use phyz_model::Model;
use tang_tensor::{Shape, Tensor};

/// What can go wrong crossing the boundary.
#[derive(Debug)]
pub enum PhysicsGradError {
    /// A tensor arrived with the wrong number of elements.
    Shape {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    /// The physics adjoint refused. Carries phyz's own reason.
    Adjoint(ConvexAdjointError),
    /// `backward` was called on a tape with no recorded steps.
    EmptyTape,
}

impl core::fmt::Display for PhysicsGradError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Shape {
                what,
                expected,
                got,
            } => write!(f, "{what}: expected {expected} elements, got {got}"),
            Self::Adjoint(e) => write!(f, "physics adjoint refused: {e:?}"),
            Self::EmptyTape => write!(f, "backward on a tape with no recorded steps"),
        }
    }
}

impl std::error::Error for PhysicsGradError {}

impl From<ConvexAdjointError> for PhysicsGradError {
    fn from(e: ConvexAdjointError) -> Self {
        Self::Adjoint(e)
    }
}

/// The cotangents a single step's VJP returns.
#[derive(Clone, Debug)]
pub struct StepVjp {
    /// `wᵀ ∂state_out/∂state_in`, length `nq + nv`.
    pub d_state_in: Tensor<f64>,
    /// `wᵀ ∂state_out/∂ctrl`, length `nv`.
    pub d_ctrl: Tensor<f64>,
}

/// One phyz simulation step as a differentiable op.
///
/// The op is the *scene* — model, ground, material, solver config. It holds
/// no trajectory state, so it is a plain immutable value that any number of
/// forwards and backwards can share. State flows through the arguments:
/// `state = [q ; v]`, a `Tensor<f64>` of length `nq + nv`; `ctrl` is length
/// `nv` (raw generalised forces when the model has no actuators, actuator
/// commands when it does).
///
/// ```no_run
/// # use phyz_tang::PhysicsStep;
/// # use tang_tensor::Tensor;
/// # fn demo(op: &PhysicsStep, s0: Tensor<f64>, u: Tensor<f64>) {
/// let s1 = op.forward(&s0, &u).unwrap();
/// // seed a cotangent on the output and pull it back
/// let mut w = Tensor::<f64>::zeros(s1.shape().clone());
/// w.data_mut()[5] = 1.0; // d(height)/d(inputs)
/// let g = op.vjp(&s0, &u, &w).unwrap();
/// # let _ = g.d_state_in;
/// # }
/// ```
pub struct PhysicsStep<'m> {
    model: &'m Model,
    ground_height: f64,
    material: ContactMaterial,
    config: ContactSolverConfig,
}

impl<'m> PhysicsStep<'m> {
    /// A step against `model` with a ground plane at `z = 0`, default
    /// material, and the shipped simulation solver settings — i.e. the
    /// derivative of the physics that actually runs.
    pub fn new(model: &'m Model) -> Self {
        Self {
            model,
            ground_height: 0.0,
            material: ContactMaterial::default(),
            config: ContactSolverConfig::simulation(),
        }
    }

    /// Move the ground plane.
    pub fn with_ground_height(mut self, h: f64) -> Self {
        self.ground_height = h;
        self
    }

    /// Friction, restitution, compliance.
    pub fn with_material(mut self, material: ContactMaterial) -> Self {
        self.material = material;
        self
    }

    /// Solver settings. [`ContactSolverConfig::simulation`] differentiates the
    /// shipped physics; [`ContactSolverConfig::gradients`] trades a little
    /// fidelity for smoother sensitivities.
    pub fn with_config(mut self, config: ContactSolverConfig) -> Self {
        self.config = config;
        self
    }

    /// Length of the state vector, `nq + nv`.
    pub fn state_dim(&self) -> usize {
        self.model.nq + self.model.nv
    }

    /// Length of the control vector, `nv`.
    pub fn ctrl_dim(&self) -> usize {
        self.model.nv
    }

    /// The model this op steps.
    pub fn model(&self) -> &Model {
        self.model
    }

    fn split(&self, state: &Tensor<f64>) -> Result<(DVec, DVec), PhysicsGradError> {
        let want = self.state_dim();
        let got = state.numel();
        if got != want {
            return Err(PhysicsGradError::Shape {
                what: "state",
                expected: want,
                got,
            });
        }
        let d = state.data();
        Ok((
            DVec::from_slice(&d[..self.model.nq]),
            DVec::from_slice(&d[self.model.nq..]),
        ))
    }

    fn ctrl_vec(&self, ctrl: &Tensor<f64>) -> Result<DVec, PhysicsGradError> {
        let want = self.ctrl_dim();
        let got = ctrl.numel();
        if got != want {
            return Err(PhysicsGradError::Shape {
                what: "ctrl",
                expected: want,
                got,
            });
        }
        Ok(DVec::from_slice(ctrl.data()))
    }

    fn rollout<'a>(
        &'a self,
        q0: DVec,
        v0: DVec,
        ctrl: &'a dyn Fn(usize) -> DVec,
    ) -> ConvexContactRollout<'a> {
        ConvexContactRollout {
            model: self.model,
            ground_height: self.ground_height,
            material: self.material.clone(),
            config: self.config,
            q0,
            v0,
            steps: 1,
            ctrl,
        }
    }

    /// Advance one step: `(state_in, ctrl) -> state_out`.
    ///
    /// This is `phyz_diff::convex_rollout_objective` with a capturing
    /// objective, so it is the same code path — not merely the same physics —
    /// that [`Self::vjp`] differentiates.
    pub fn forward(
        &self,
        state_in: &Tensor<f64>,
        ctrl: &Tensor<f64>,
    ) -> Result<Tensor<f64>, PhysicsGradError> {
        let (q0, v0) = self.split(state_in)?;
        let u = self.ctrl_vec(ctrl)?;

        let out: RefCell<Vec<f64>> = RefCell::new(Vec::new());
        let objective = FinalStateObjective {
            value: &|q: &[f64], v: &[f64]| {
                let mut o = out.borrow_mut();
                o.clear();
                o.extend_from_slice(q);
                o.extend_from_slice(v);
                0.0
            },
            gradient: &|q: &[f64], v: &[f64]| (vec![0.0; q.len()], vec![0.0; v.len()]),
        };
        let hold = |_t: usize| u.clone();
        let _ = convex_rollout_objective(&self.rollout(q0, v0, &hold), &objective);

        let data = out.into_inner();
        Ok(Tensor::new(data, Shape::from_slice(&[self.state_dim()])))
    }

    /// Pull a cotangent on `state_out` back to `state_in` and `ctrl`.
    ///
    /// `cotangent` is `w = dL/d state_out`, length `nq + nv`. The result is
    /// `wᵀ ∂state_out/∂state_in` and `wᵀ ∂state_out/∂ctrl`.
    ///
    /// One call, not one per output component: the underlying trajectory
    /// adjoint is objective-scoped, and a linear objective `J = w·(q,v)` makes
    /// its reported `dJ/dq0`, `dJ/dv0`, `dJ/du_0` the VJP by definition.
    pub fn vjp(
        &self,
        state_in: &Tensor<f64>,
        ctrl: &Tensor<f64>,
        cotangent: &Tensor<f64>,
    ) -> Result<StepVjp, PhysicsGradError> {
        let (q0, v0) = self.split(state_in)?;
        let u = self.ctrl_vec(ctrl)?;
        let want = self.state_dim();
        if cotangent.numel() != want {
            return Err(PhysicsGradError::Shape {
                what: "cotangent",
                expected: want,
                got: cotangent.numel(),
            });
        }
        let (wq, wv) = cotangent.data().split_at(self.model.nq);
        let (wq, wv) = (wq.to_vec(), wv.to_vec());

        let objective = FinalStateObjective {
            value: &|q: &[f64], v: &[f64]| {
                let mut j = 0.0;
                for (a, b) in wq.iter().zip(q) {
                    j += a * b;
                }
                for (a, b) in wv.iter().zip(v) {
                    j += a * b;
                }
                j
            },
            gradient: &|_q: &[f64], _v: &[f64]| (wq.clone(), wv.clone()),
        };
        let hold = |_t: usize| u.clone();
        let g = convex_adjoint_gradient(&self.rollout(q0, v0, &hold), &objective)?;

        let mut d_state = Vec::with_capacity(want);
        d_state.extend_from_slice(g.d_q0.as_slice());
        d_state.extend_from_slice(g.d_v0.as_slice());
        let d_ctrl = g
            .d_ctrl
            .first()
            .map(|c| c.as_slice().to_vec())
            .unwrap_or_else(|| vec![0.0; self.ctrl_dim()]);

        Ok(StepVjp {
            d_state_in: Tensor::new(d_state, Shape::from_slice(&[want])),
            d_ctrl: Tensor::new(d_ctrl, Shape::from_slice(&[self.ctrl_dim()])),
        })
    }
}

/// Gradients of a scalar loss w.r.t. a whole recorded trajectory.
#[derive(Clone, Debug)]
pub struct TapeGrads {
    /// `dL/d state_0`, length `nq + nv`.
    pub d_state0: Tensor<f64>,
    /// `dL/d ctrl_t`, one entry per recorded step, each length `nv`.
    pub d_ctrl: Vec<Tensor<f64>>,
}

/// N chained [`PhysicsStep`]s, and the reverse sweep through them.
///
/// # The one-forward-caching question
///
/// tang-train's `Module` caches its forward's input in a single
/// `Option<Tensor>` field that the next forward overwrites, so
/// forward-forward-backward silently returns the *second* forward's gradient.
/// That hazard is a property of a one-slot cache, not of custom VJPs, and
/// this tape does not have one: [`Self::step`] **appends** `(state_in, ctrl)`
/// to a `Vec`, so a second forward extends the trajectory rather than
/// invalidating the first. There is no staleness to get: every step's VJP is
/// re-derived from the inputs recorded for *that* step.
///
/// That is what phyz's solver-level adjoint buys us. Because the contact
/// solve is deterministic, the backward re-executes it and reproduces every
/// branch bit for bit, so the tape needs only the step's inputs — no
/// per-sweep trace, and nothing that can go stale.
///
/// Start a new trajectory with [`Self::reset`]; that is the explicit,
/// visible operation that a `Module`'s overwrite does behind your back.
pub struct PhysicsTape<'a, 'm> {
    op: &'a PhysicsStep<'m>,
    state: Tensor<f64>,
    records: Vec<(Tensor<f64>, Tensor<f64>)>,
}

impl<'a, 'm> PhysicsTape<'a, 'm> {
    /// Begin a trajectory at `state0`.
    pub fn new(op: &'a PhysicsStep<'m>, state0: Tensor<f64>) -> Self {
        Self {
            op,
            state: state0,
            records: Vec::new(),
        }
    }

    /// Advance one step under `ctrl`, recording what backward will need.
    /// Returns the new state.
    pub fn step(&mut self, ctrl: &Tensor<f64>) -> Result<&Tensor<f64>, PhysicsGradError> {
        let next = self.op.forward(&self.state, ctrl)?;
        let prev = core::mem::replace(&mut self.state, next);
        self.records.push((prev, ctrl.clone()));
        Ok(&self.state)
    }

    /// The current state.
    pub fn state(&self) -> &Tensor<f64> {
        &self.state
    }

    /// Number of recorded steps.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether nothing has been stepped yet.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Drop the trajectory and restart at `state0`.
    pub fn reset(&mut self, state0: Tensor<f64>) {
        self.records.clear();
        self.state = state0;
    }

    /// Pull `cotangent = dL/d state_N` back through every recorded step.
    ///
    /// Reverse mode, exactly as it reads: the running cotangent enters step
    /// `t`'s VJP, `dL/d ctrl_t` falls out, and `dL/d state_t` becomes the
    /// cotangent for step `t-1`. Takes `&self` — backward does not consume
    /// or disturb the tape, so it can be run again with a different seed.
    pub fn backward(&self, cotangent: &Tensor<f64>) -> Result<TapeGrads, PhysicsGradError> {
        if self.records.is_empty() {
            return Err(PhysicsGradError::EmptyTape);
        }
        let mut w = cotangent.clone();
        let mut d_ctrl = vec![Tensor::zeros(Shape::from_slice(&[0])); self.records.len()];
        for (t, (state_in, ctrl)) in self.records.iter().enumerate().rev() {
            let g = self.op.vjp(state_in, ctrl, &w)?;
            d_ctrl[t] = g.d_ctrl;
            w = g.d_state_in;
        }
        Ok(TapeGrads {
            d_state0: w,
            d_ctrl,
        })
    }
}
