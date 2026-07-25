//! Reward and termination specification.
//!
//! Rewards are a weighted sum of named terms. Keeping them declarative (rather
//! than a `Box<dyn Fn>`) is what lets the same task definition be evaluated on
//! the CPU backend in f64 and, later, compiled into a GPU kernel that never
//! round-trips state to the host.

use crate::obs::Kinematics;
use phyz_model::Model;

/// A world axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// +X.
    X,
    /// +Y.
    Y,
    /// +Z.
    Z,
}

impl Axis {
    fn pick(self, v: phyz_math::Vec3) -> f64 {
        match self {
            Axis::X => v.x,
            Axis::Y => v.y,
            Axis::Z => v.z,
        }
    }
}

/// One additive reward term.
#[derive(Debug, Clone, PartialEq)]
pub enum RewardTerm {
    /// `weight` per step the episode stays alive.
    Alive {
        /// Scale factor.
        weight: f64,
    },
    /// `weight *` body-frame linear velocity along `axis`.
    Velocity {
        /// Body index.
        body: usize,
        /// Which velocity component to reward.
        axis: Axis,
        /// Scale factor.
        weight: f64,
    },
    /// `-weight * sum(action^2)`.
    CtrlCost {
        /// Scale factor.
        weight: f64,
    },
    /// `-weight * (world height of body - target)^2`.
    HeightPenalty {
        /// Body index.
        body: usize,
        /// Desired world-frame Z.
        target: f64,
        /// Scale factor.
        weight: f64,
    },
}

/// How an episode may end.
#[derive(Debug, Clone, PartialEq)]
pub struct Termination {
    /// Terminate when a body's world-frame Z leaves `(min, max)`, given as
    /// `(body index, min, max)`. The standard "has it fallen over" check.
    pub healthy_z: Option<(usize, f64, f64)>,
    /// Terminate when any state entry becomes non-finite. Always worth leaving
    /// on: in f32 a diverging environment otherwise poisons the batch forever.
    pub terminate_on_nonfinite: bool,
    /// Terminate when any generalized velocity exceeds this magnitude.
    ///
    /// A divergence guard, not a task rule. Numerical blow-up passes through
    /// enormous-but-finite values long before it reaches NaN, and by then the
    /// observation clip has been saturated for many steps, feeding the policy
    /// garbage. Catching it early converts an unrecoverable batch into one
    /// terminated episode. Isaac Gym and Brax both do this; it is the
    /// difference between a training run that survives a bad contact and one
    /// that silently learns from noise.
    pub max_velocity: Option<f64>,
}

impl Default for Termination {
    fn default() -> Self {
        Self {
            healthy_z: None,
            terminate_on_nonfinite: true,
            max_velocity: Some(1.0e3),
        }
    }
}

/// The task layered on top of a [`Model`].
#[derive(Debug, Clone, PartialEq)]
pub struct TaskSpec {
    /// Reward terms, summed.
    pub reward: Vec<RewardTerm>,
    /// Termination conditions.
    pub termination: Termination,
    /// Episode length in control steps; exceeding it *truncates* (not
    /// terminates), matching Gymnasium semantics.
    pub max_episode_steps: usize,
    /// Uniform noise half-width applied to `q` on reset.
    pub reset_qpos_noise: f64,
    /// Uniform noise half-width applied to `v` on reset.
    pub reset_qvel_noise: f64,
}

impl Default for TaskSpec {
    fn default() -> Self {
        Self {
            reward: vec![RewardTerm::Alive { weight: 1.0 }],
            termination: Termination::default(),
            max_episode_steps: 1000,
            reset_qpos_noise: 0.0,
            reset_qvel_noise: 0.0,
        }
    }
}

impl TaskSpec {
    /// Evaluate the reward for one environment.
    pub fn reward(&self, _model: &Model, view: &Kinematics<'_>, action: &[f32]) -> f64 {
        self.reward
            .iter()
            .map(|t| match t {
                RewardTerm::Alive { weight } => *weight,
                RewardTerm::Velocity { body, axis, weight } => {
                    weight * axis.pick(view.vel[*body].linear)
                }
                RewardTerm::CtrlCost { weight } => {
                    -weight * action.iter().map(|a| (*a as f64) * (*a as f64)).sum::<f64>()
                }
                RewardTerm::HeightPenalty {
                    body,
                    target,
                    weight,
                } => {
                    let d = view.world_pos(*body).z - target;
                    -weight * d * d
                }
            })
            .sum()
    }

    /// Whether this environment has terminated (an MDP-level failure, distinct
    /// from time-limit truncation).
    pub fn terminated(&self, view: &Kinematics<'_>) -> bool {
        if self.termination.terminate_on_nonfinite
            && (view.state.q.as_slice().iter().any(|x| !x.is_finite())
                || view.state.v.as_slice().iter().any(|x| !x.is_finite()))
        {
            return true;
        }
        if let Some(limit) = self.termination.max_velocity
            && view.state.v.as_slice().iter().any(|x| x.abs() > limit)
        {
            return true;
        }
        if let Some((body, lo, hi)) = self.termination.healthy_z {
            let z = view.world_pos(body).z;
            if !(z > lo && z < hi) {
                return true;
            }
        }
        false
    }
}
