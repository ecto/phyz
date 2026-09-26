//! Declarative observation assembly.
//!
//! This deliberately does **not** go through `phyz_world::sensor::Sensor`.
//! That type returns hard-coded zeros for `BodyAccel`, `ForceTorque` and
//! `Rangefinder` (see `crates/phyz-world/src/sensor.rs:59,90,98`), which would
//! silently feed a policy dead channels. The terms below are restricted to
//! quantities the engine can actually produce today; anything that would need
//! a placeholder is simply not offered. See the "Blockers" section of
//! `docs/design/batched-envs.md` for the plan to converge the two.

use phyz_math::{Mat3, SpatialTransform, SpatialVec, Vec3};
use phyz_model::{Model, State};

/// One contiguous slice of the observation vector.
#[derive(Debug, Clone, PartialEq)]
pub enum ObsTerm {
    /// `len` generalized positions starting at `start`.
    Qpos {
        /// First index into `q`.
        start: usize,
        /// Number of entries.
        len: usize,
    },
    /// `len` generalized velocities starting at `start`.
    Qvel {
        /// First index into `v`.
        start: usize,
        /// Number of entries.
        len: usize,
    },
    /// World-frame position of a body origin (3).
    BodyPos {
        /// Body index.
        body: usize,
    },
    /// World-frame orientation of a body as a `(w, x, y, z)` quaternion (4).
    BodyQuat {
        /// Body index.
        body: usize,
    },
    /// Body-frame linear velocity of the body origin (3).
    BodyLinVel {
        /// Body index.
        body: usize,
    },
    /// Body-frame angular velocity (3).
    BodyAngVel {
        /// Body index.
        body: usize,
    },
    /// The action applied on the previous step (`nu`). Common in locomotion
    /// observations and free to compute, so it is offered directly.
    LastAction,
    /// Elapsed episode time in seconds (1).
    Time,
}

impl ObsTerm {
    /// Width of this term in the flattened observation.
    pub fn dim(&self, model: &Model) -> usize {
        match self {
            ObsTerm::Qpos { len, .. } | ObsTerm::Qvel { len, .. } => *len,
            ObsTerm::BodyPos { .. } | ObsTerm::BodyLinVel { .. } | ObsTerm::BodyAngVel { .. } => 3,
            ObsTerm::BodyQuat { .. } => 4,
            ObsTerm::LastAction => model.actuators.len().max(model.nv),
            ObsTerm::Time => 1,
        }
    }
}

/// The ordered list of terms making up one observation vector.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ObsSpec {
    /// Terms, concatenated in order.
    pub terms: Vec<ObsTerm>,
    /// Optional symmetric clip applied after assembly; MuJoCo-Gym locomotion
    /// tasks conventionally clip at 10.
    pub clip: Option<f32>,
}

impl ObsSpec {
    /// Total observation width.
    pub fn dim(&self, model: &Model) -> usize {
        self.terms.iter().map(|t| t.dim(model)).sum()
    }

    /// The default "everything the engine can honestly report" observation:
    /// all generalized positions and velocities. Matches the shape of the
    /// classic MuJoCo-Gym locomotion observations minus the root x/y position,
    /// which is excluded because it is not translation-invariant.
    pub fn full_state(model: &Model, exclude_root_xy: bool) -> Self {
        let skip = if exclude_root_xy { 2 } else { 0 };
        Self {
            terms: vec![
                ObsTerm::Qpos {
                    start: skip,
                    len: model.nq - skip,
                },
                ObsTerm::Qvel {
                    start: 0,
                    len: model.nv,
                },
            ],
            clip: Some(10.0),
        }
    }

    /// Write one environment's observation into `out` (length == [`Self::dim`]).
    pub fn write(
        &self,
        model: &Model,
        view: &Kinematics<'_>,
        last_action: &[f32],
        out: &mut [f32],
    ) {
        let mut off = 0;
        for term in &self.terms {
            let n = term.dim(model);
            let dst = &mut out[off..off + n];
            match term {
                ObsTerm::Qpos { start, len } => {
                    for (i, d) in dst.iter_mut().enumerate().take(*len) {
                        *d = view.state.q[start + i] as f32;
                    }
                }
                ObsTerm::Qvel { start, len } => {
                    for (i, d) in dst.iter_mut().enumerate().take(*len) {
                        *d = view.state.v[start + i] as f32;
                    }
                }
                ObsTerm::BodyPos { body } => {
                    let p = view.world_pos(*body);
                    dst.copy_from_slice(&[p.x as f32, p.y as f32, p.z as f32]);
                }
                ObsTerm::BodyQuat { body } => {
                    let (w, x, y, z) = mat3_to_quat(&view.world_rot(*body));
                    dst.copy_from_slice(&[w as f32, x as f32, y as f32, z as f32]);
                }
                ObsTerm::BodyLinVel { body } => {
                    let v = view.vel[*body].linear;
                    dst.copy_from_slice(&[v.x as f32, v.y as f32, v.z as f32]);
                }
                ObsTerm::BodyAngVel { body } => {
                    let w = view.vel[*body].angular;
                    dst.copy_from_slice(&[w.x as f32, w.y as f32, w.z as f32]);
                }
                ObsTerm::LastAction => {
                    for (i, d) in dst.iter_mut().enumerate() {
                        *d = last_action.get(i).copied().unwrap_or(0.0);
                    }
                }
                ObsTerm::Time => dst[0] = view.state.time as f32,
            }
            off += n;
        }
        if let Some(c) = self.clip {
            for x in out.iter_mut() {
                *x = x.clamp(-c, c);
            }
        }
    }
}

/// Cached kinematics for one environment at one instant.
///
/// `xform[i]` is the world→body Plücker transform produced by
/// [`phyz_rigid::forward_kinematics`]; `vel[i]` is the body-frame spatial
/// velocity. World-frame quantities are derived on demand rather than stored,
/// since most observations never ask for them.
pub struct Kinematics<'a> {
    /// The state these kinematics were computed from.
    pub state: &'a State,
    /// World→body transforms, one per body.
    pub xform: &'a [SpatialTransform],
    /// Body-frame spatial velocities, one per body.
    pub vel: &'a [SpatialVec],
}

impl Kinematics<'_> {
    /// World-frame position of body `i`'s origin.
    ///
    /// `xform[i].pos` is already the world position despite the transform being
    /// named world→body; taking `.inverse()` here would mirror every model
    /// through the origin. Locked down by
    /// `phyz-rigid/tests/frame_conventions.rs`.
    pub fn world_pos(&self, i: usize) -> Vec3 {
        self.xform[i].pos
    }

    /// Body→world rotation of body `i`.
    pub fn world_rot(&self, i: usize) -> Mat3 {
        self.xform[i].rot.transpose()
    }
}

/// Rotation matrix → `(w, x, y, z)` quaternion, Shepperd's method.
pub fn mat3_to_quat(m: &Mat3) -> (f64, f64, f64, f64) {
    let trace = m[(0, 0)] + m[(1, 1)] + m[(2, 2)];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        (
            0.25 * s,
            (m[(2, 1)] - m[(1, 2)]) / s,
            (m[(0, 2)] - m[(2, 0)]) / s,
            (m[(1, 0)] - m[(0, 1)]) / s,
        )
    } else if m[(0, 0)] > m[(1, 1)] && m[(0, 0)] > m[(2, 2)] {
        let s = (1.0 + m[(0, 0)] - m[(1, 1)] - m[(2, 2)]).sqrt() * 2.0;
        (
            (m[(2, 1)] - m[(1, 2)]) / s,
            0.25 * s,
            (m[(0, 1)] + m[(1, 0)]) / s,
            (m[(0, 2)] + m[(2, 0)]) / s,
        )
    } else if m[(1, 1)] > m[(2, 2)] {
        let s = (1.0 + m[(1, 1)] - m[(0, 0)] - m[(2, 2)]).sqrt() * 2.0;
        (
            (m[(0, 2)] - m[(2, 0)]) / s,
            (m[(0, 1)] + m[(1, 0)]) / s,
            0.25 * s,
            (m[(1, 2)] + m[(2, 1)]) / s,
        )
    } else {
        let s = (1.0 + m[(2, 2)] - m[(0, 0)] - m[(1, 1)]).sqrt() * 2.0;
        (
            (m[(1, 0)] - m[(0, 1)]) / s,
            (m[(0, 2)] + m[(2, 0)]) / s,
            (m[(1, 2)] + m[(2, 1)]) / s,
            0.25 * s,
        )
    }
}
