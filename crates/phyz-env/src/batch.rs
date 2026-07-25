//! The CPU reference backend: `N` independent environments stepped in lockstep.
//!
//! This is the semantic specification of a phyz vector env. It runs in f64 and
//! makes no approximations, so the GPU backend can be tested against it
//! directly (`assert_backends_agree`-style tests) rather than against a
//! hand-written oracle.

use crate::contact::GroundContact;
use crate::integrate::semi_implicit_euler;
use crate::obs::{Kinematics, ObsSpec};
use crate::rng::Stream;
use crate::space::BoxSpace;
use crate::task::TaskSpec;
use crate::{AutoresetMode, EnvError, StepBatch, VecEnv};
use phyz_math::{SpatialTransform, SpatialVec};
use phyz_model::{Model, State};
use phyz_rigid::{aba_with_external_forces, forward_kinematics};

/// Everything needed to instantiate a batch of environments.
#[derive(Debug, Clone)]
pub struct EnvConfig {
    /// Number of parallel environments.
    pub num_envs: usize,
    /// Observation layout.
    pub obs: ObsSpec,
    /// Reward / termination / reset-noise definition.
    pub task: TaskSpec,
    /// Physics substeps per `step()` call (MuJoCo's `frame_skip`).
    pub frame_skip: usize,
    /// Ground plane, or `None` for free space.
    pub ground: Option<GroundContact>,
    /// Autoreset semantics.
    pub autoreset: AutoresetMode,
    /// Base seed. `reset(Some(seed))` overrides it.
    pub seed: u64,
}

impl EnvConfig {
    /// A config with sensible defaults for `num_envs` copies of `model`.
    pub fn new(model: &Model, num_envs: usize) -> Self {
        Self {
            num_envs,
            obs: ObsSpec::full_state(model, false),
            task: TaskSpec::default(),
            frame_skip: 1,
            ground: None,
            autoreset: AutoresetMode::NextStep,
            seed: 0,
        }
    }
}

/// Per-environment scratch, kept out of the hot loop's allocation path.
struct EnvSlot {
    state: State,
    xform: Vec<SpatialTransform>,
    vel: Vec<SpatialVec>,
    ext: Vec<SpatialVec>,
    episode_index: u64,
    step_count: u32,
    ret: f32,
    /// Set when the env ended on the previous step and `NextStep` autoreset
    /// owes it a reset before the next physics advance.
    pending_reset: bool,
}

/// `N` copies of one model, stepped together. See [`VecEnv`].
pub struct BatchEnv {
    model: Model,
    config: EnvConfig,
    slots: Vec<EnvSlot>,
    obs_space: BoxSpace,
    act_space: BoxSpace,
    last_action: Vec<f32>,
    batch: StepBatch,
    seed: u64,
}

impl BatchEnv {
    /// Build a batch from a model and config.
    pub fn new(model: Model, config: EnvConfig) -> Result<Self, EnvError> {
        if config.num_envs == 0 {
            return Err(EnvError::Config("num_envs must be > 0".into()));
        }
        if config.frame_skip == 0 {
            return Err(EnvError::Config("frame_skip must be > 0".into()));
        }

        // Actuators drive velocity DOFs. With no `<actuator>` block, fall back
        // to direct torque control on every DOF, which is what the CPU
        // dynamics already assume (`state.ctrl` is nv-long).
        let act_space = if model.actuators.is_empty() {
            BoxSpace::uniform(model.nv, -1.0, 1.0)
        } else {
            let mut low = Vec::with_capacity(model.actuators.len());
            let mut high = Vec::with_capacity(model.actuators.len());
            for a in &model.actuators {
                model
                    .v_offsets
                    .get(a.joint_idx)
                    .ok_or_else(|| EnvError::Config(format!("actuator {} has no joint", a.name)))?;
                let [lo, hi] = a.ctrl_range.unwrap_or([-1.0, 1.0]);
                low.push(lo as f32);
                high.push(hi as f32);
            }
            BoxSpace::new(low, high)
        };

        let obs_dim = config.obs.dim(&model);
        let nu = act_space.dim();
        let n = config.num_envs;
        let nb = model.nbodies();

        let slots = (0..n)
            .map(|_| EnvSlot {
                state: model.default_state(),
                xform: vec![SpatialTransform::identity(); nb],
                vel: vec![SpatialVec::zero(); nb],
                ext: vec![SpatialVec::zero(); nb],
                episode_index: 0,
                step_count: 0,
                ret: 0.0,
                pending_reset: false,
            })
            .collect();

        // The contact impulse bound depends on the timestep; keep them in sync
        // rather than making the caller remember.
        let mut config = config;
        if let Some(g) = config.ground.as_mut() {
            g.dt = model.dt;
        }

        let seed = config.seed;
        Ok(Self {
            obs_space: BoxSpace::unbounded(obs_dim),
            act_space,
            last_action: vec![0.0; n * nu],
            batch: StepBatch::new(n, obs_dim),
            slots,
            model,
            config,
            seed,
        })
    }

    /// Load a model from an MJCF file and wrap it in a batch.
    pub fn from_mjcf(path: &str, config_fn: impl FnOnce(&Model) -> EnvConfig) -> Result<Self, EnvError> {
        let loader = phyz_mjcf::MjcfLoader::from_file(path)?;
        let model = loader.build_model();
        let config = config_fn(&model);
        Self::new(model, config)
    }

    /// The underlying physics model.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Per-environment simulation state, for checkpointing or debugging.
    pub fn states(&self) -> impl Iterator<Item = &State> {
        self.slots.iter().map(|s| &s.state)
    }

    /// Reset one environment to a freshly sampled initial state.
    fn reset_slot(&mut self, i: usize) {
        let slot = &mut self.slots[i];
        slot.state = self.model.default_state();
        slot.episode_index += 1;
        slot.step_count = 0;
        slot.ret = 0.0;
        slot.pending_reset = false;

        let mut rng = Stream::new(self.seed, i as u64, slot.episode_index);
        let qn = self.config.task.reset_qpos_noise;
        let vn = self.config.task.reset_qvel_noise;
        if qn != 0.0 {
            for k in 0..self.model.nq {
                slot.state.q[k] += rng.uniform_sym(qn);
            }
        }
        if vn != 0.0 {
            for k in 0..self.model.nv {
                slot.state.v[k] += rng.uniform_sym(vn);
            }
        }

        let nu = self.act_space.dim();
        self.last_action[i * nu..(i + 1) * nu].fill(0.0);
        Self::refresh_kinematics(&self.model, &mut self.slots[i]);
    }

    /// Write one environment's actions into `State::ctrl`.
    ///
    /// `ctrl` is **actuator space** (length `nu`), not DOF space: the dynamics
    /// map it through `gear`, `ctrlrange` and the affine servo law in
    /// `phyz_rigid::actuation`. Doing the mapping here as well would apply the
    /// gear ratio twice. Models with no `<actuator>` block keep the legacy
    /// meaning, where `ctrl` is a raw per-DOF generalized force.
    fn apply_actions(&mut self, i: usize, act: &[f32]) {
        let ctrl = &mut self.slots[i].state.ctrl;
        for c in ctrl.as_mut_slice().iter_mut() {
            *c = 0.0;
        }
        for (a_idx, value) in act.iter().enumerate() {
            if a_idx < ctrl.len() {
                ctrl[a_idx] = *value as f64;
            }
        }
    }

    fn refresh_kinematics(model: &Model, slot: &mut EnvSlot) {
        let (x, v) = forward_kinematics(model, &slot.state);
        slot.xform = x;
        slot.vel = v;
    }

    /// Advance one environment by `frame_skip` physics substeps.
    fn advance(model: &Model, ground: Option<&GroundContact>, slot: &mut EnvSlot, substeps: usize) {
        for _ in 0..substeps {
            Self::refresh_kinematics(model, slot);
            let ext = match ground {
                Some(g) => {
                    g.forces(model, &slot.xform, &slot.vel, &mut slot.ext);
                    Some(slot.ext.as_slice())
                }
                None => None,
            };
            let qdd = aba_with_external_forces(model, &slot.state, ext);
            semi_implicit_euler(model, &mut slot.state, qdd.as_slice(), model.dt);
        }
        Self::refresh_kinematics(model, slot);
    }

    /// Write env `i`'s current observation into `dst`.
    fn write_obs(&self, i: usize, dst: &mut [f32]) {
        let nu = self.act_space.dim();
        let view = Kinematics {
            state: &self.slots[i].state,
            xform: &self.slots[i].xform,
            vel: &self.slots[i].vel,
        };
        self.config
            .obs
            .write(&self.model, &view, &self.last_action[i * nu..(i + 1) * nu], dst);
    }
}

impl VecEnv for BatchEnv {
    fn num_envs(&self) -> usize {
        self.config.num_envs
    }

    fn observation_space(&self) -> &BoxSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &BoxSpace {
        &self.act_space
    }

    fn reset(&mut self, seed: Option<u64>) -> &StepBatch {
        if let Some(s) = seed {
            self.seed = s;
            for slot in &mut self.slots {
                slot.episode_index = 0;
            }
        }
        self.batch.clear();
        let obs_dim = self.obs_space.dim();
        for i in 0..self.config.num_envs {
            self.reset_slot(i);
            let mut buf = vec![0.0f32; obs_dim];
            self.write_obs(i, &mut buf);
            self.batch.obs[i * obs_dim..(i + 1) * obs_dim].copy_from_slice(&buf);
        }
        &self.batch
    }

    fn step(&mut self, actions: &[f32]) -> &StepBatch {
        let n = self.config.num_envs;
        let nu = self.act_space.dim();
        let obs_dim = self.obs_space.dim();
        assert_eq!(
            actions.len(),
            n * nu,
            "expected {} action scalars ({n} envs × {nu}), got {}",
            n * nu,
            actions.len()
        );

        self.batch.clear();
        let mut clamped = actions.to_vec();
        self.act_space.clamp_batch(&mut clamped);

        for i in 0..n {
            let act = &clamped[i * nu..(i + 1) * nu];

            // Gymnasium >= 1.0 `NextStep` autoreset: the step *after* a
            // terminal one ignores its action, resets, and reports reward 0
            // with both flags false.
            if self.config.autoreset == AutoresetMode::NextStep && self.slots[i].pending_reset {
                self.reset_slot(i);
                let mut buf = vec![0.0f32; obs_dim];
                self.write_obs(i, &mut buf);
                self.batch.obs[i * obs_dim..(i + 1) * obs_dim].copy_from_slice(&buf);
                continue;
            }

            self.last_action[i * nu..(i + 1) * nu].copy_from_slice(act);
            self.apply_actions(i, act);

            Self::advance(
                &self.model,
                self.config.ground.as_ref(),
                &mut self.slots[i],
                self.config.frame_skip,
            );
            self.slots[i].step_count += 1;

            let view = Kinematics {
                state: &self.slots[i].state,
                xform: &self.slots[i].xform,
                vel: &self.slots[i].vel,
            };
            let reward = self.config.task.reward(&self.model, &view, act) as f32;
            let terminated = self.config.task.terminated(&view);
            let truncated = !terminated
                && self.slots[i].step_count as usize >= self.config.task.max_episode_steps;

            self.slots[i].ret += reward;
            self.batch.rewards[i] = reward;
            self.batch.terminated[i] = terminated;
            self.batch.truncated[i] = truncated;

            let done = terminated || truncated;
            if done {
                self.batch.episode_return[i] = self.slots[i].ret;
                self.batch.episode_length[i] = self.slots[i].step_count;
                self.batch.episode_done[i] = true;
            }

            match (done, self.config.autoreset) {
                (true, AutoresetMode::SameStep) => {
                    // Report the terminal observation in `final_obs`, and the
                    // reset observation in `obs`.
                    let mut fin = vec![0.0f32; obs_dim];
                    self.write_obs(i, &mut fin);
                    self.batch.final_obs[i * obs_dim..(i + 1) * obs_dim].copy_from_slice(&fin);
                    self.batch.final_valid[i] = true;
                    self.reset_slot(i);
                }
                (true, AutoresetMode::NextStep) => {
                    self.slots[i].pending_reset = true;
                }
                _ => {}
            }

            let mut buf = vec![0.0f32; obs_dim];
            self.write_obs(i, &mut buf);
            self.batch.obs[i * obs_dim..(i + 1) * obs_dim].copy_from_slice(&buf);
        }

        &self.batch
    }

    fn observations(&self) -> &[f32] {
        &self.batch.obs
    }
}
