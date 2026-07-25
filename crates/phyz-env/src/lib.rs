//! Batched reinforcement-learning environments for phyz.
//!
//! A [`VecEnv`] is `N` copies of one physics model stepped in lockstep, with
//! Gymnasium-compatible semantics: `reset(seed)`, `step(actions)`, per-env
//! `reward` / `terminated` / `truncated`, and automatic reset on episode end.
//!
//! ```no_run
//! use phyz_env::{BatchEnv, EnvConfig, VecEnv};
//! use phyz_mjcf::MjcfLoader;
//!
//! let model = MjcfLoader::from_file("models/ant.xml").unwrap().build_model();
//! let mut env = BatchEnv::new(model.clone(), EnvConfig::new(&model, 4096)).unwrap();
//!
//! let batch = env.reset(Some(0));
//! let nu = env.action_space().dim();
//! let actions = vec![0.0f32; env.num_envs() * nu];
//! let batch = env.step(&actions);
//! println!("{:?}", &batch.rewards[..4]);
//! ```
//!
//! Design rationale, precision policy, determinism guarantees and the Python
//! packaging plan live in `docs/design/batched-envs.md`.

#![warn(missing_docs)]

pub mod batch;
pub mod contact;
pub mod integrate;
pub mod obs;
pub mod presets;
pub mod rng;
pub mod space;
pub mod task;
pub mod tensor;

pub use batch::{BatchEnv, EnvConfig};
pub use contact::GroundContact;
pub use obs::{ObsSpec, ObsTerm};
pub use presets::{Benchmark, make};
pub use space::BoxSpace;
pub use task::{Axis, RewardTerm, TaskSpec, Termination};

/// Errors from environment construction and stepping.
#[derive(Debug, thiserror::Error)]
pub enum EnvError {
    /// The configuration is internally inconsistent.
    #[error("invalid environment config: {0}")]
    Config(String),

    /// The model could not be loaded.
    #[error("model load failed: {0}")]
    Mjcf(#[from] phyz_mjcf::MjcfError),

    /// A backend refused to run this model. Carries the reason so callers can
    /// fall back to the CPU backend rather than guessing.
    #[error("backend unsupported: {0}")]
    Unsupported(String),
}

/// When a finished environment is reset.
///
/// The default matches Gymnasium >= 1.0's vector default. Under
/// [`AutoresetMode::NextStep`] a terminal step returns the *terminal*
/// observation, and the following `step()` ignores its action, resets, and
/// returns the reset observation with `reward = 0` and both flags false. This
/// is the mode that avoids the "which observation does `obs` hold?" ambiguity
/// that bit everyone in Gym <= 0.26.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AutoresetMode {
    /// Reset on the step after termination. Gymnasium's default.
    #[default]
    NextStep,
    /// Reset within the same step; the terminal observation is reported in
    /// [`StepBatch::final_obs`]. Matches Gym <= 0.26 and EnvPool.
    SameStep,
    /// Never auto-reset; the caller is responsible.
    Disabled,
}

/// One step's worth of results for the whole batch.
///
/// Every field is a flat, contiguous, row-major array with the environment as
/// the leading axis, so it can be handed to numpy or to a `tang` tensor without
/// a copy or a transpose. See [`tensor`] for the zero-copy views.
#[derive(Debug, Clone)]
pub struct StepBatch {
    /// `num_envs × obs_dim` observations.
    pub obs: Vec<f32>,
    /// `num_envs` rewards.
    pub rewards: Vec<f32>,
    /// `num_envs` MDP-level termination flags.
    pub terminated: Vec<bool>,
    /// `num_envs` time-limit truncation flags.
    pub truncated: Vec<bool>,
    /// `num_envs × obs_dim`. Only meaningful where [`Self::final_valid`] is
    /// set, and only ever written under [`AutoresetMode::SameStep`].
    pub final_obs: Vec<f32>,
    /// Which entries of [`Self::final_obs`] are populated.
    pub final_valid: Vec<bool>,
    /// Undiscounted return of the episode that just ended.
    pub episode_return: Vec<f32>,
    /// Length of the episode that just ended, in control steps.
    pub episode_length: Vec<u32>,
    /// Which entries of the two `episode_*` arrays are populated. Equivalent
    /// to Gymnasium's `info["episode"]["_r"]` mask.
    pub episode_done: Vec<bool>,
}

impl StepBatch {
    /// Allocate zeroed buffers.
    pub fn new(num_envs: usize, obs_dim: usize) -> Self {
        Self {
            obs: vec![0.0; num_envs * obs_dim],
            rewards: vec![0.0; num_envs],
            terminated: vec![false; num_envs],
            truncated: vec![false; num_envs],
            final_obs: vec![0.0; num_envs * obs_dim],
            final_valid: vec![false; num_envs],
            episode_return: vec![0.0; num_envs],
            episode_length: vec![0; num_envs],
            episode_done: vec![false; num_envs],
        }
    }

    /// Reset the per-step scalars without reallocating. `obs` is left alone
    /// because every environment overwrites it unconditionally.
    pub fn clear(&mut self) {
        self.rewards.fill(0.0);
        self.terminated.fill(false);
        self.truncated.fill(false);
        self.final_valid.fill(false);
        self.episode_return.fill(0.0);
        self.episode_length.fill(0);
        self.episode_done.fill(false);
    }

    /// Number of environments.
    pub fn num_envs(&self) -> usize {
        self.rewards.len()
    }
}

/// A batch of environments with Gymnasium vector-env semantics.
pub trait VecEnv {
    /// Number of parallel environments.
    fn num_envs(&self) -> usize;

    /// Per-environment observation space.
    fn observation_space(&self) -> &BoxSpace;

    /// Per-environment action space.
    fn action_space(&self) -> &BoxSpace;

    /// Reset every environment. Passing a seed makes the entire subsequent
    /// rollout reproducible; see the determinism section of the design doc.
    fn reset(&mut self, seed: Option<u64>) -> &StepBatch;

    /// Advance every environment by one control step.
    ///
    /// `actions` is `num_envs × action_dim`, row-major. Values outside the
    /// action space are clamped, never rejected — an RL algorithm sampling
    /// from an unbounded Gaussian should not crash the simulator.
    fn step(&mut self, actions: &[f32]) -> &StepBatch;

    /// The current observation buffer, without stepping.
    fn observations(&self) -> &[f32];
}
