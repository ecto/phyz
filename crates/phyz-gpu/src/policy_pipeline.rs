//! Device-resident observation + policy pass for the CUDA batch simulator.
//!
//! The PD pass moved the control *law* on-device; this moves the control
//! *loop* — observe, run a small MLP, sample an action, write the PD target
//! — so an RL collector never has to read state back between control
//! steps. It is deliberately narrow: a fixed op-table observation, a
//! two-hidden-layer tanh MLP, a diagonal Gaussian. That is the whole of what
//! `ipse-sim`'s PPO collector needs, and it is enough to keep the GPU busy;
//! anything the table cannot express stays on the host, fed by
//! [`crate::cuda::BatchSim::readback_state_history`] once per rollout instead of
//! once per control step.
//!
//! Layouts here are mirrored verbatim in `cuda/phyz_kernels.cu`
//! (`OBS_OP_STRIDE`, `XF_STRIDE`, the op kinds, the weight order); the CPU
//! references at the bottom of this file are what the parity tests hold the
//! kernels to.

use phyz_math::Vec3;
use phyz_model::{Model, State};
use phyz_rigid::forward_kinematics;

/// Floats per body in the FK readout buffer (`XF_STRIDE` in the kernels).
///
/// ```text
/// [0..9]   rotation, world -> body, row-major (State::body_xform convention)
/// [9..12]  body origin, world
/// [12..15] angular velocity, body frame
/// [15..18] linear velocity, body frame
/// ```
pub const XF_STRIDE: usize = 18;

/// Floats per observation feature in the op table (`OBS_OP_STRIDE`).
pub const OBS_OP_STRIDE: usize = 4;

/// Per-thread capacity of the policy kernel's scratch arrays. A spec that
/// exceeds these is rejected on the host (`PolicySpec::validate`) rather than
/// silently skipped on device.
pub const POLICY_MAX_IN: usize = 128;
/// See [`POLICY_MAX_IN`].
pub const POLICY_MAX_H: usize = 256;
/// See [`POLICY_MAX_IN`].
pub const POLICY_MAX_OUT: usize = 32;

/// One observation feature. Kinds are mirrored in `obs_thread`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObsOp {
    /// A constant — a zeroed slot, or a command channel.
    Const(f64),
    /// `q[i] - reference`.
    QMinus(usize, f64),
    /// `v[i]`.
    V(usize),
    /// Pitch of a body: `atan2(-r02, r22)` of its world->body rotation.
    BodyPitch(usize),
    /// Roll of a body: `atan2(r12, r22)`.
    BodyRoll(usize),
    /// Heading error `wrap(cmd - yaw)` with `yaw = atan2(r01, r00)`.
    BodyYawError(usize, f64),
    /// Body origin height, world z.
    BodyPosZ(usize),
}

impl ObsOp {
    fn pack(self) -> [f32; OBS_OP_STRIDE] {
        match self {
            ObsOp::Const(c) => [0.0, 0.0, 0.0, c as f32],
            ObsOp::QMinus(i, c) => [1.0, i as f32, 0.0, c as f32],
            ObsOp::V(i) => [2.0, i as f32, 0.0, 0.0],
            ObsOp::BodyPitch(b) => [3.0, b as f32, 0.0, 0.0],
            ObsOp::BodyRoll(b) => [4.0, b as f32, 0.0, 0.0],
            ObsOp::BodyYawError(b, c) => [5.0, b as f32, 0.0, c as f32],
            ObsOp::BodyPosZ(b) => [6.0, b as f32, 0.0, 0.0],
        }
    }

    fn check(self, nq: usize, nv: usize, nb: usize) -> Result<(), String> {
        let ok = match self {
            ObsOp::Const(_) => true,
            ObsOp::QMinus(i, _) => i < nq,
            ObsOp::V(i) => i < nv,
            ObsOp::BodyPitch(b)
            | ObsOp::BodyRoll(b)
            | ObsOp::BodyYawError(b, _)
            | ObsOp::BodyPosZ(b) => b < nb,
        };
        if ok {
            Ok(())
        } else {
            Err(format!(
                "observation op {self:?} out of range (nq {nq}, nv {nv}, nbodies {nb})"
            ))
        }
    }

    /// The feature on the CPU, from a state whose `body_xform` is current.
    /// This is the reference the device pass is held to.
    pub fn eval(self, state: &State) -> f64 {
        match self {
            ObsOp::Const(c) => c,
            ObsOp::QMinus(i, c) => state.q[i] - c,
            ObsOp::V(i) => state.v[i],
            ObsOp::BodyPitch(b) => {
                let r = &state.body_xform[b].rot;
                (-r[(0, 2)]).atan2(r[(2, 2)])
            }
            ObsOp::BodyRoll(b) => {
                let r = &state.body_xform[b].rot;
                r[(1, 2)].atan2(r[(2, 2)])
            }
            ObsOp::BodyYawError(b, cmd) => {
                let r = &state.body_xform[b].rot;
                let yaw = r[(0, 1)].atan2(r[(0, 0)]);
                let mut e = cmd - yaw;
                while e > std::f64::consts::PI {
                    e -= 2.0 * std::f64::consts::PI;
                }
                while e < -std::f64::consts::PI {
                    e += 2.0 * std::f64::consts::PI;
                }
                e
            }
            ObsOp::BodyPosZ(b) => state.body_xform[b].pos.z,
        }
    }
}

/// Pack an op table for the observation kernel.
pub fn pack_obs_ops(ops: &[ObsOp]) -> Vec<f32> {
    ops.iter().flat_map(|o| o.pack()).collect()
}

/// What the policy pass runs: observation table, MLP shape, action wiring.
#[derive(Debug, Clone)]
pub struct PolicySpec {
    /// The observation row, in order — its length is the MLP input width.
    pub obs: Vec<ObsOp>,
    /// Hidden width of both hidden layers.
    pub hidden: usize,
    /// For each action `k`, the PD servo slot (registration order, as in
    /// `set_position_targets`) that receives `base + clamp(action_k)`.
    /// Its length is the MLP output width.
    pub act_slots: Vec<usize>,
    /// Applied-action clamp.
    pub act_clamp: f64,
    /// AR(1) coefficient of the exploration noise (0 = white).
    pub noise_rho: f64,
    /// Per-input Gaussian noise scale added in place before the forward
    /// (and therefore recorded); zero entries draw nothing.
    pub input_noise: Vec<f64>,
    /// Control steps of observation/action history to keep on device.
    pub history_steps: usize,
}

impl PolicySpec {
    /// Input width.
    pub fn n_in(&self) -> usize {
        self.obs.len()
    }
    /// Output width.
    pub fn n_out(&self) -> usize {
        self.act_slots.len()
    }
    /// Number of weights the MLP takes, in the kernel's flat order.
    pub fn n_weights(&self) -> usize {
        let (i, h, o) = (self.n_in(), self.hidden, self.n_out());
        i * h + h + h * h + h + h * o + o
    }
    /// Reject shapes the kernels cannot hold or wiring that indexes past
    /// the PD table.
    pub fn validate(
        &self,
        nq: usize,
        nv: usize,
        nb: usize,
        n_pd_dofs: usize,
    ) -> Result<(), String> {
        if self.n_in() == 0 || self.n_in() > POLICY_MAX_IN {
            return Err(format!(
                "policy input width {} not in 1..={POLICY_MAX_IN}",
                self.n_in()
            ));
        }
        if self.hidden == 0 || self.hidden > POLICY_MAX_H {
            return Err(format!(
                "policy hidden width {} not in 1..={POLICY_MAX_H}",
                self.hidden
            ));
        }
        if self.n_out() == 0 || self.n_out() > POLICY_MAX_OUT {
            return Err(format!(
                "policy output width {} not in 1..={POLICY_MAX_OUT}",
                self.n_out()
            ));
        }
        if self.input_noise.len() != self.n_in() {
            return Err(format!(
                "input_noise has {} entries for {} inputs",
                self.input_noise.len(),
                self.n_in()
            ));
        }
        if self.history_steps == 0 {
            return Err("policy history_steps must be at least 1".into());
        }
        for op in &self.obs {
            op.check(nq, nv, nb)?;
        }
        for &s in &self.act_slots {
            if s >= n_pd_dofs {
                return Err(format!(
                    "action slot {s} exceeds the {n_pd_dofs} registered PD servos"
                ));
            }
        }
        Ok(())
    }
}

/// The per-world xorshift64 seed the policy kernel starts from — the same
/// mixing on the host and in tests, so a stream can be replayed exactly.
pub fn world_seed(seed: u64, world: usize) -> u64 {
    // SplitMix64 of (seed, world): decorrelates neighbouring worlds; the
    // `.max(1)` is xorshift's zero-state guard, as in `XorShift::new`.
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(world as u64 + 1));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)).max(1)
}

/// The kernel's random stream, on the host: xorshift64 + Box–Muller, the
/// same arithmetic as `rng_normal` in `phyz_kernels.cu` (and as
/// `ipse_dojo::search::XorShift`).
#[derive(Debug, Clone, Copy)]
pub struct KernelRng(pub u64);

impl KernelRng {
    /// Start from a world's seed.
    pub fn for_world(seed: u64, world: usize) -> Self {
        Self(world_seed(seed, world))
    }
    /// Next raw draw.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform on [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal via Box–Muller.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.max(1e-300).ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Two-hidden-layer tanh MLP forward on the host, `f64`, from the flat
/// weight vector in the kernel's order. The reference for `policy_thread`.
pub fn mlp_forward(n_in: usize, n_h: usize, n_out: usize, weights: &[f64], x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), n_in);
    let (w1, rest) = weights.split_at(n_h * n_in);
    let (b1, rest) = rest.split_at(n_h);
    let (w2, rest) = rest.split_at(n_h * n_h);
    let (b2, rest) = rest.split_at(n_h);
    let (w3, rest) = rest.split_at(n_out * n_h);
    let (b3, _) = rest.split_at(n_out);
    let h1: Vec<f64> = (0..n_h)
        .map(|j| (b1[j] + (0..n_in).map(|i| w1[j * n_in + i] * x[i]).sum::<f64>()).tanh())
        .collect();
    let h2: Vec<f64> = (0..n_h)
        .map(|j| (b2[j] + (0..n_h).map(|i| w2[j * n_h + i] * h1[i]).sum::<f64>()).tanh())
        .collect();
    (0..n_out)
        .map(|k| b3[k] + (0..n_h).map(|i| w3[k * n_h + i] * h2[i]).sum::<f64>())
        .collect()
}

/// One world's policy step on the host, exactly as the kernel does it:
/// input noise in place, forward, AR(1) Gaussian sample, log-prob, clamped
/// action onto the base target. Returns `(actions, logp)`; `obs`, `z` and
/// `targets` are updated in place. `targets` is this world's PD target row.
#[allow(clippy::too_many_arguments)]
pub fn policy_reference(
    spec: &PolicySpec,
    weights: &[f64],
    std: &[f64],
    rng: &mut KernelRng,
    obs: &mut [f64],
    z: &mut [f64],
    base_targets: &[f64],
    targets: &mut [f64],
) -> (Vec<f64>, f64) {
    for (i, o) in obs.iter_mut().enumerate() {
        if spec.input_noise[i] != 0.0 {
            *o += spec.input_noise[i] * rng.normal();
        }
    }
    let mean = mlp_forward(spec.n_in(), spec.hidden, spec.n_out(), weights, obs);
    let rho = spec.noise_rho;
    let keep = (1.0 - rho * rho).max(0.0).sqrt();
    let mut logp = 0.0;
    let mut act = Vec::with_capacity(spec.n_out());
    for k in 0..spec.n_out() {
        z[k] = rho * z[k] + keep * rng.normal();
        let a = mean[k] + std[k] * z[k];
        let zi = (a - mean[k]) / std[k];
        logp += -0.5 * zi * zi - std[k].ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
        let slot = spec.act_slots[k];
        targets[slot] = base_targets[slot] + a.clamp(-spec.act_clamp, spec.act_clamp);
        act.push(a);
    }
    (act, logp)
}

/// The observation row on the host for one state: FK, then the ops.
pub fn observe_reference(model: &Model, state: &mut State, ops: &[ObsOp]) -> Vec<f64> {
    let (xforms, _) = forward_kinematics(model, state);
    state.body_xform = xforms;
    ops.iter().map(|o| o.eval(state)).collect()
}

/// A body's world position from an FK readout row — for hosts that want
/// the device kinematics without a `State`.
pub fn xform_pos(row: &[f32]) -> Vec3 {
    Vec3::new(row[9] as f64, row[10] as f64, row[11] as f64)
}
