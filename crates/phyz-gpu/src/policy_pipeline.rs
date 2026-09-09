//! Device-resident observation + policy pass for the CUDA batch simulator.
//!
//! The PD pass moved the control *law* on-device; this moves the control
//! *loop* — observe, run a small MLP, sample an action, write the PD target
//! — so an RL collector never has to read state back between control
//! steps. It is deliberately narrow: a fixed op-table observation, a
//! two-hidden-layer tanh MLP, a diagonal Gaussian. That is the whole of what
//! `ipse-sim`'s PPO collector needs, and it is enough to keep the GPU busy;
//! anything the table cannot express stays on the host, fed by
//! `BatchSim::readback_state_history` (the `cuda` feature) once per rollout instead of
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
pub const POLICY_MAX_OUT: usize = 64;

/// Floats per `ObsOp::ComOverSupport` support point in the aux buffer:
/// body index then the body-frame offset.
pub const SUPPORT_STRIDE: usize = 4;

/// One observation feature. Kinds are mirrored in `obs_thread`.
///
/// Most variants are one scalar from one body or one DOF and pack into the
/// four floats of [`OBS_OP_STRIDE`]. [`ObsOp::ComOverSupport`] does not — it
/// reduces over every body and over a caller-supplied set of support points
/// — so its payload lives in the *aux* buffer [`pack_obs_ops`] returns
/// alongside the table, and its table row is a `(offset, len)` window into
/// it.
#[derive(Debug, Clone, PartialEq)]
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
    /// Body origin, world, on `axis` (0 = x, 1 = y, 2 = z). The general
    /// form of [`ObsOp::BodyPosZ`], which stays for the callers that read
    /// better with a name.
    BodyPos {
        /// Body index.
        body: usize,
        /// World axis: 0 = x, 1 = y, 2 = z.
        axis: u8,
    },
    /// Per-world constant `k` — the value `set_policy_world_consts` wrote
    /// into slot `k` of *this* world's row.
    ///
    /// [`ObsOp::Const`] is baked into the op table, which is one table for
    /// every world and is uploaded with the spec; this reads a row that is
    /// per-world and writable between control steps, so one batch can carry
    /// a different command (or a different reference) in every world.
    WorldSlot(usize),
    /// `q[i] -` per-world constant `k`. The per-world form of
    /// [`ObsOp::QMinus`], whose reference is baked into the shared table.
    QMinusWorld(usize, usize),
    /// The whole-model centre of mass minus a support reference, one axis,
    /// optionally in a body's heading frame.
    ///
    /// ```text
    /// com = (Σ_b m_b · X_b · c_b) / Σ_b m_b          // every body with mass
    /// tgt = mean over `support` of  X_body · offset  // body-frame points
    /// e   = com − tgt                                // world frame
    /// ```
    ///
    /// With `heading_body: Some(h)`, the horizontal pair is rotated into
    /// `h`'s yaw (`yaw = atan2(r01, r00)`, the same yaw
    /// [`ObsOp::BodyYawError`] reads) before the axis is selected:
    /// `[cos·e.x + sin·e.y, −sin·e.x + cos·e.y, e.z]`. With `None` the
    /// world-frame `e` is used directly.
    ///
    /// An empty `support` evaluates to zero on both paths, so a rig with no
    /// registered feet reads a defined value rather than a NaN.
    ComOverSupport {
        /// The support points: `(body, offset in that body's frame)`. The
        /// reference is their mean — two soles for a biped, four for a
        /// quadruped.
        support: Vec<(usize, [f64; 3])>,
        /// Body whose yaw defines the horizontal frame, or `None` for
        /// world axes.
        heading_body: Option<usize>,
        /// Which component of the (possibly rotated) difference: 0 = along
        /// heading, 1 = across, 2 = height.
        axis: u8,
    },
}

impl ObsOp {
    /// The table row, given where this op's payload starts in the aux
    /// buffer. Ops without a payload ignore `aux_off` and consume nothing.
    fn pack(&self, aux_off: usize) -> [f32; OBS_OP_STRIDE] {
        match *self {
            ObsOp::Const(c) => [0.0, 0.0, 0.0, c as f32],
            ObsOp::QMinus(i, c) => [1.0, i as f32, 0.0, c as f32],
            ObsOp::V(i) => [2.0, i as f32, 0.0, 0.0],
            ObsOp::BodyPitch(b) => [3.0, b as f32, 0.0, 0.0],
            ObsOp::BodyRoll(b) => [4.0, b as f32, 0.0, 0.0],
            ObsOp::BodyYawError(b, c) => [5.0, b as f32, 0.0, c as f32],
            ObsOp::BodyPosZ(b) => [6.0, b as f32, 0.0, 0.0],
            ObsOp::BodyPos { body, axis } => [7.0, body as f32, axis as f32, 0.0],
            ObsOp::ComOverSupport {
                ref support, axis, ..
            } => [8.0, aux_off as f32, support.len() as f32, axis as f32],
            ObsOp::WorldSlot(k) => [9.0, k as f32, 0.0, 0.0],
            ObsOp::QMinusWorld(i, k) => [10.0, i as f32, k as f32, 0.0],
        }
    }

    /// This op's aux payload — empty for everything but
    /// [`ObsOp::ComOverSupport`], whose layout is
    /// `[heading_flag, heading_body, (body, ox, oy, oz) × n]`.
    fn aux(&self) -> Vec<f32> {
        match *self {
            ObsOp::ComOverSupport {
                ref support,
                heading_body,
                ..
            } => {
                let mut a = Vec::with_capacity(2 + support.len() * SUPPORT_STRIDE);
                a.push(f32::from(heading_body.is_some()));
                a.push(heading_body.unwrap_or(0) as f32);
                for &(b, o) in support {
                    a.push(b as f32);
                    a.extend(o.iter().map(|&x| x as f32));
                }
                a
            }
            _ => Vec::new(),
        }
    }

    fn check(&self, nq: usize, nv: usize, nb: usize) -> Result<(), String> {
        let ok = match *self {
            ObsOp::Const(_) => true,
            ObsOp::WorldSlot(_) => true,
            ObsOp::QMinusWorld(i, _) => i < nq,
            ObsOp::QMinus(i, _) => i < nq,
            ObsOp::V(i) => i < nv,
            ObsOp::BodyPitch(b)
            | ObsOp::BodyRoll(b)
            | ObsOp::BodyYawError(b, _)
            | ObsOp::BodyPosZ(b) => b < nb,
            ObsOp::BodyPos { body, axis } => body < nb && axis < 3,
            ObsOp::ComOverSupport {
                ref support,
                heading_body,
                axis,
            } => {
                axis < 3
                    && heading_body.is_none_or(|h| h < nb)
                    && support.iter().all(|&(b, _)| b < nb)
            }
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
    ///
    /// `model` is read only for the mass distribution
    /// [`ObsOp::ComOverSupport`] reduces over; every other variant ignores
    /// it.
    pub fn eval(&self, model: &Model, state: &State) -> f64 {
        self.eval_with(model, state, &[])
    }

    /// As [`ObsOp::eval`], with this world's per-world constant row
    /// ([`ObsOp::WorldSlot`], [`ObsOp::QMinusWorld`]). A slot past the end
    /// of `consts` reads zero, exactly as the kernel does when the row is
    /// shorter than the slot it is asked for.
    pub fn eval_with(&self, model: &Model, state: &State, consts: &[f64]) -> f64 {
        let slot = |k: usize| consts.get(k).copied().unwrap_or(0.0);
        match *self {
            ObsOp::WorldSlot(k) => slot(k),
            ObsOp::QMinusWorld(i, k) => state.q[i] - slot(k),
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
            ObsOp::BodyPos { body, axis } => axis_of(state.body_xform[body].pos, axis),
            ObsOp::ComOverSupport {
                ref support,
                heading_body,
                axis,
            } => {
                if support.is_empty() {
                    return 0.0;
                }
                let mut tgt = Vec3::zeros();
                for &(b, o) in support {
                    let x = &state.body_xform[b];
                    tgt += x.pos + x.rot.transpose().mul_vec(Vec3::new(o[0], o[1], o[2]));
                }
                let tgt = tgt / support.len() as f64;
                let e = centre_of_mass(model, state) - tgt;
                match heading_body {
                    None => axis_of(e, axis),
                    Some(h) => {
                        let r = &state.body_xform[h].rot;
                        let (sy, cy) = r[(0, 1)].atan2(r[(0, 0)]).sin_cos();
                        match axis {
                            0 => cy * e.x + sy * e.y,
                            1 => -sy * e.x + cy * e.y,
                            _ => e.z,
                        }
                    }
                }
            }
        }
    }
}

fn axis_of(v: Vec3, axis: u8) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// The model's mass-weighted centre of mass, world frame, from a state whose
/// `body_xform` is current. Bodies with non-positive mass are skipped, as
/// they are in the kernel.
pub fn centre_of_mass(model: &Model, state: &State) -> Vec3 {
    let mut num = Vec3::zeros();
    let mut den = 0.0;
    for (i, b) in model.bodies.iter().enumerate() {
        if b.inertia.mass <= 0.0 || i >= state.body_xform.len() {
            continue;
        }
        let x = &state.body_xform[i];
        num += (x.pos + x.rot.transpose().mul_vec(b.inertia.com)) * b.inertia.mass;
        den += b.inertia.mass;
    }
    if den > 0.0 { num / den } else { Vec3::zeros() }
}

/// The per-body mass table the observation kernel reduces over:
/// `[mass, com.x, com.y, com.z]` per body, in model order.
pub fn pack_com_table(model: &Model) -> Vec<f32> {
    model
        .bodies
        .iter()
        .flat_map(|b| {
            let c = b.inertia.com;
            [b.inertia.mass as f32, c.x as f32, c.y as f32, c.z as f32]
        })
        .collect()
}

/// Pack an op table for the observation kernel: `(table, aux)`, where the
/// table is `OBS_OP_STRIDE` floats per feature and `aux` is the
/// variable-length payload the reducing ops index into.
pub fn pack_obs_ops(ops: &[ObsOp]) -> (Vec<f32>, Vec<f32>) {
    let mut table = Vec::with_capacity(ops.len() * OBS_OP_STRIDE);
    let mut aux = Vec::new();
    for op in ops {
        table.extend(op.pack(aux.len()));
        aux.extend(op.aux());
    }
    (table, aux)
}

/// The **feedforward-torque channel**: the outputs past the PD-target ones.
///
/// A robot whose only actuation seam is a position servo is limited by the
/// servo, not by its motors — a commanded target reaches the joint through
/// `kp`, and a fast move needs a target the tracker can never be far enough
/// from. The channel adds a second seam: outputs `act_slots.len()..n_out`
/// are torques in **normalized** units, scaled by [`TauChannel::scale`] into
/// newton-metres and summed with the servo term *inside* the effort clamp,
/// so a joint can never be commanded past its rating however the two
/// channels split the demand.
///
/// `slots` and `scale` are parallel and index the PD registration order, as
/// [`PolicySpec::act_slots`] does. A spec without this field is what it
/// always was: `n_out == act_slots.len()`, and no line of the torque path
/// executes on either the host or the device.
#[derive(Debug, Clone)]
pub struct TauChannel {
    /// The PD servo slot each torque output drives, in output order.
    pub slots: Vec<usize>,
    /// Newton-metres per unit of the (clamped) output, in output order —
    /// the joint's own effort limit is the honest default, because then a
    /// saturated output asks for exactly the torque the motor is rated for.
    pub scale: Vec<f64>,
}

/// What the policy pass runs: observation table, MLP shape, action wiring.
#[derive(Debug, Clone)]
pub struct PolicySpec {
    /// The observation row, in order — its length is the MLP input width.
    pub obs: Vec<ObsOp>,
    /// Hidden width of both hidden layers.
    pub hidden: usize,
    /// For each POSITION action `k`, the PD servo slot (registration
    /// order, as in `set_position_targets`) that receives
    /// `base + clamp(action_k)`. Its length is the MLP output width unless
    /// [`PolicySpec::tau`] adds a torque channel on the end.
    pub act_slots: Vec<usize>,
    /// The feedforward-torque outputs, appended after the position ones.
    /// `None` — the default — is a pure position policy, byte for byte.
    pub tau: Option<TauChannel>,
    /// Applied-action clamp, one value for every action slot.
    pub act_clamp: f64,
    /// Per-action-slot applied-action clamp. When `Some`, action `k` is
    /// clamped to `±act_clamp_slots[k]` and `act_clamp` is unused; its
    /// length must be `n_out()`. `None` is the scalar `act_clamp`, and
    /// filling every entry with `act_clamp` reproduces it exactly.
    pub act_clamp_slots: Option<Vec<f64>>,
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
    /// Output width: the position slots, then the torque ones.
    pub fn n_out(&self) -> usize {
        self.act_slots.len() + self.n_tau()
    }
    /// How many outputs are PD-target offsets — the first `n_pos` of them.
    pub fn n_pos(&self) -> usize {
        self.act_slots.len()
    }
    /// How many outputs are feedforward torques, 0 without a channel.
    pub fn n_tau(&self) -> usize {
        self.tau.as_ref().map_or(0, |t| t.slots.len())
    }
    /// Width of the per-world constant row: one past the highest slot any
    /// [`ObsOp::WorldSlot`] / [`ObsOp::QMinusWorld`] reads, and 0 when the
    /// spec uses neither. This is what
    /// `BatchSim::set_policy_world_consts` expects per world, and the row
    /// the buffer is preallocated for — it is fixed by the spec, so the
    /// buffer never moves while the spec stands.
    pub fn n_world_consts(&self) -> usize {
        self.obs
            .iter()
            .filter_map(|o| match *o {
                ObsOp::WorldSlot(k) | ObsOp::QMinusWorld(_, k) => Some(k + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }
    /// The applied-action clamp for action `k`.
    pub fn clamp_at(&self, k: usize) -> f64 {
        match &self.act_clamp_slots {
            Some(c) => c[k],
            None => self.act_clamp,
        }
    }
    /// The per-slot clamp row the kernel reads — the scalar broadcast when
    /// no per-slot vector is set.
    pub fn clamp_row(&self) -> Vec<f64> {
        match &self.act_clamp_slots {
            Some(c) => c.clone(),
            None => vec![self.act_clamp; self.n_out()],
        }
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
        if let Some(c) = &self.act_clamp_slots
            && c.len() != self.n_out()
        {
            return Err(format!(
                "act_clamp_slots has {} entries for {} actions",
                c.len(),
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
        if let Some(t) = &self.tau {
            if t.slots.len() != t.scale.len() {
                return Err(format!(
                    "torque channel has {} slots and {} scales",
                    t.slots.len(),
                    t.scale.len()
                ));
            }
            if t.slots.is_empty() {
                return Err(
                    "an empty torque channel is not a policy without one — use `tau: None`".into(),
                );
            }
            for &s in &t.slots {
                if s >= n_pd_dofs {
                    return Err(format!(
                        "torque slot {s} exceeds the {n_pd_dofs} registered PD servos"
                    ));
                }
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
    let mut tau = vec![0.0; targets.len()];
    policy_reference_tau(
        spec,
        weights,
        std,
        rng,
        obs,
        z,
        base_targets,
        targets,
        &mut tau,
    )
}

/// [`policy_reference`], also filling this world's feedforward-torque row —
/// the reference for the [`TauChannel`] half of `policy_thread`. `tau_ff` is
/// the PD target row's width and is fully rewritten (zeros on every slot the
/// channel does not drive), exactly as the kernel rewrites it.
#[allow(clippy::too_many_arguments)]
pub fn policy_reference_tau(
    spec: &PolicySpec,
    weights: &[f64],
    std: &[f64],
    rng: &mut KernelRng,
    obs: &mut [f64],
    z: &mut [f64],
    base_targets: &[f64],
    targets: &mut [f64],
    tau_ff: &mut [f64],
) -> (Vec<f64>, f64) {
    for (i, o) in obs.iter_mut().enumerate() {
        if spec.input_noise[i] != 0.0 {
            *o += spec.input_noise[i] * rng.normal();
        }
    }
    targets.copy_from_slice(base_targets);
    if spec.tau.is_some() {
        tau_ff.fill(0.0);
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
        let lim = spec.clamp_at(k);
        let applied = a.clamp(-lim, lim);
        if k < spec.n_pos() {
            let slot = spec.act_slots[k];
            targets[slot] = base_targets[slot] + applied;
        } else {
            let t = spec.tau.as_ref().expect("n_out past n_pos needs a channel");
            let j = k - spec.n_pos();
            tau_ff[t.slots[j]] = applied * t.scale[j];
        }
        act.push(a);
    }
    (act, logp)
}

/// The observation row on the host for one state: FK, then the ops.
pub fn observe_reference(model: &Model, state: &mut State, ops: &[ObsOp]) -> Vec<f64> {
    observe_reference_with(model, state, ops, &[])
}

/// [`observe_reference`] with this world's per-world constant row.
pub fn observe_reference_with(
    model: &Model,
    state: &mut State,
    ops: &[ObsOp],
    consts: &[f64],
) -> Vec<f64> {
    let (xforms, _) = forward_kinematics(model, state);
    state.body_xform = xforms;
    ops.iter()
        .map(|o| o.eval_with(model, state, consts))
        .collect()
}

/// A body's world position from an FK readout row — for hosts that want
/// the device kinematics without a `State`.
pub fn xform_pos(row: &[f32]) -> Vec3 {
    Vec3::new(row[9] as f64, row[10] as f64, row[11] as f64)
}
