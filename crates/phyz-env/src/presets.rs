//! Ready-made configurations for the standard benchmark models.
//!
//! These make `ant`, `half_cheetah`, `humanoid` and the hand approximation
//! one call away, which is what a new user tries first.
//!
//! The reward shaping follows the usual MuJoCo-Gym recipe (forward velocity +
//! alive bonus − control cost) but the coefficients are **not** tuned to
//! reproduce published return curves — the models themselves are re-authored
//! (see `models/*.xml`). Treat returns as internally comparable only.

use crate::contact::GroundContact;
use crate::obs::{ObsSpec, ObsTerm};
use crate::task::{Axis, RewardTerm, TaskSpec, Termination};
use crate::{AutoresetMode, BatchEnv, EnvConfig, EnvError};
use phyz_model::Model;

/// Which benchmark to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Benchmark {
    /// Quadruped locomotion, 8 actuators.
    Ant,
    /// Planar running, 6 actuators.
    HalfCheetah,
    /// Bipedal locomotion, 17 actuators.
    Humanoid,
    /// High-DOF dexterous hand approximation, 20 position servos.
    ShadowHand,
}

impl Benchmark {
    /// The MJCF filename under `models/`.
    pub fn model_file(&self) -> &'static str {
        match self {
            Benchmark::Ant => "ant.xml",
            Benchmark::HalfCheetah => "half_cheetah.xml",
            Benchmark::Humanoid => "humanoid.xml",
            Benchmark::ShadowHand => "shadow_hand.xml",
        }
    }

    /// The task definition, given the loaded model.
    pub fn task(&self, _model: &Model) -> TaskSpec {
        match self {
            Benchmark::Ant => TaskSpec {
                reward: vec![
                    RewardTerm::Velocity {
                        body: 0,
                        axis: Axis::X,
                        weight: 1.0,
                    },
                    RewardTerm::Alive { weight: 1.0 },
                    RewardTerm::CtrlCost { weight: 0.005 },
                ],
                termination: Termination {
                    healthy_z: Some((0, 0.26, 1.2)),
                    ..Termination::default()
                },
                max_episode_steps: 1000,
                reset_qpos_noise: 0.1,
                reset_qvel_noise: 0.1,
            },
            Benchmark::HalfCheetah => TaskSpec {
                reward: vec![
                    RewardTerm::Velocity {
                        body: 0,
                        axis: Axis::X,
                        weight: 1.0,
                    },
                    RewardTerm::CtrlCost { weight: 0.1 },
                ],
                // Half-cheetah famously never terminates on a task condition;
                // it only truncates. The divergence guard stays on, because a
                // blown-up environment is not a task outcome.
                termination: Termination {
                    healthy_z: None,
                    ..Termination::default()
                },
                max_episode_steps: 1000,
                reset_qpos_noise: 0.1,
                reset_qvel_noise: 0.1,
            },
            Benchmark::Humanoid => TaskSpec {
                reward: vec![
                    RewardTerm::Velocity {
                        body: 0,
                        axis: Axis::X,
                        weight: 1.25,
                    },
                    RewardTerm::Alive { weight: 5.0 },
                    RewardTerm::CtrlCost { weight: 0.1 },
                ],
                termination: Termination {
                    healthy_z: Some((0, 1.0, 2.0)),
                    ..Termination::default()
                },
                max_episode_steps: 1000,
                reset_qpos_noise: 0.01,
                reset_qvel_noise: 0.01,
            },
            Benchmark::ShadowHand => TaskSpec {
                // No manipulation objective without an object in the scene, so
                // the default task is "stay finite and stay near the reference
                // pose". Real manipulation tasks need equality constraints and
                // a free-floating object; see the design doc.
                reward: vec![
                    RewardTerm::Alive { weight: 1.0 },
                    RewardTerm::CtrlCost { weight: 0.01 },
                ],
                termination: Termination::default(),
                max_episode_steps: 500,
                reset_qpos_noise: 0.02,
                reset_qvel_noise: 0.0,
            },
        }
    }

    /// The observation layout.
    pub fn obs(&self, model: &Model) -> ObsSpec {
        match self {
            // Locomotion: drop the root x/y so the policy sees a
            // translation-invariant state, then add root orientation and
            // velocities explicitly.
            Benchmark::Ant | Benchmark::HalfCheetah | Benchmark::Humanoid => ObsSpec {
                terms: vec![
                    ObsTerm::Qpos {
                        start: 2,
                        len: model.nq - 2,
                    },
                    ObsTerm::Qvel {
                        start: 0,
                        len: model.nv,
                    },
                    ObsTerm::BodyQuat { body: 0 },
                    ObsTerm::LastAction,
                ],
                clip: Some(10.0),
            },
            Benchmark::ShadowHand => ObsSpec {
                terms: vec![
                    ObsTerm::Qpos {
                        start: 0,
                        len: model.nq,
                    },
                    ObsTerm::Qvel {
                        start: 0,
                        len: model.nv,
                    },
                    ObsTerm::LastAction,
                ],
                clip: Some(10.0),
            },
        }
    }

    /// Physics substeps per control step.
    pub fn frame_skip(&self) -> usize {
        match self {
            Benchmark::Ant | Benchmark::HalfCheetah => 5,
            Benchmark::Humanoid => 5,
            Benchmark::ShadowHand => 10,
        }
    }

    /// The full config for `num_envs` copies.
    pub fn config(&self, model: &Model, num_envs: usize) -> EnvConfig {
        EnvConfig {
            num_envs,
            obs: self.obs(model),
            task: self.task(model),
            frame_skip: self.frame_skip(),
            ground: match self {
                // The hand is a fixed-base manipulator with no floor.
                Benchmark::ShadowHand => None,
                _ => Some(GroundContact::default()),
            },
            autoreset: AutoresetMode::NextStep,
            seed: 0,
        }
    }
}

/// Build a benchmark environment, loading its MJCF from `models_dir`.
pub fn make(
    benchmark: Benchmark,
    models_dir: &str,
    num_envs: usize,
) -> Result<BatchEnv, EnvError> {
    let path = format!("{models_dir}/{}", benchmark.model_file());
    let loader = phyz_mjcf::MjcfLoader::from_file(&path)?;

    // Loudly surface anything the parser dropped: training on a model that
    // quietly lost its tendons is worse than not training at all.
    if !loader.unsupported().is_empty() {
        return Err(EnvError::Unsupported(format!(
            "{path} uses MJCF features phyz does not implement: {:?}",
            loader.unsupported()
        )));
    }

    let model = loader.build_model();

    // A model whose rest pose violates its own joint limits simulates
    // plausibly for a few steps and then catapults. Refuse it up front rather
    // than letting someone debug the physics.
    let fatal = model.fatal_issues();
    if !fatal.is_empty() {
        return Err(EnvError::Config(format!(
            "{path} is not physically valid: {}",
            fatal
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join("; ")
        )));
    }

    let config = benchmark.config(&model, num_envs);
    BatchEnv::new(model, config)
}
