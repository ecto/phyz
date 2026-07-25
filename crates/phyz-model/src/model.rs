//! Model definition — static description of a physical system.

use crate::{Body, Joint, State};
use phyz_math::{GRAVITY, SpatialInertia, SpatialTransform, Vec3};

/// How an actuator turns its control signal into joint force.
///
/// All variants share MuJoCo's affine transmission model:
///
/// ```text
/// force = gear * (gain * ctrl + bias[0] + bias[1] * q + bias[2] * qdot)
/// ```
///
/// which makes `position`/`velocity` servos a special case of `general` rather
/// than a separate code path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActuatorType {
    /// Direct force/torque source: `gain = 1`, no bias.
    Motor,
    /// Position servo: `gain = kp`, `bias = [0, -kp, -kv]`.
    Position,
    /// Velocity servo: `gain = kv`, `bias = [0, 0, -kv]`.
    Velocity,
    /// Arbitrary affine actuator with explicit gain/bias parameters.
    General,
}

/// An actuator attached to a joint.
#[derive(Debug, Clone)]
pub struct Actuator {
    /// Actuator name, as declared in the source model.
    pub name: String,
    /// Name of the joint this actuator drives.
    pub joint_name: String,
    /// Index into [`Model::joints`] of the driven joint.
    pub joint_idx: usize,
    /// Gear ratio applied to the control signal to produce joint torque.
    pub gear: f64,
    /// Optional `[lo, hi]` clamp on the control signal, before `gear`.
    pub ctrl_range: Option<[f64; 2]>,
    /// Transmission model for this actuator.
    pub actuator_type: ActuatorType,
    /// Multiplier on the control signal.
    pub gain: f64,
    /// Affine bias terms `[constant, position, velocity]`.
    pub bias: [f64; 3],
    /// Clamp on the produced force, if any.
    pub force_range: Option<[f64; 2]>,
}

impl Actuator {
    /// A direct force/torque actuator.
    pub fn motor(name: impl Into<String>, joint_name: impl Into<String>, joint_idx: usize) -> Self {
        Self {
            name: name.into(),
            joint_name: joint_name.into(),
            joint_idx,
            gear: 1.0,
            ctrl_range: None,
            actuator_type: ActuatorType::Motor,
            gain: 1.0,
            bias: [0.0; 3],
            force_range: None,
        }
    }

    /// Clamp a raw control value to this actuator's control range.
    pub fn clamp_ctrl(&self, ctrl: f64) -> f64 {
        match self.ctrl_range {
            Some([lo, hi]) => ctrl.clamp(lo.min(hi), hi.max(lo)),
            None => ctrl,
        }
    }

    /// Generalized force produced for a raw control value at joint state
    /// `(q, qdot)`.
    ///
    /// `ctrl` is clamped to `ctrl_range` and the result to `force_range`. For a
    /// [`ActuatorType::Motor`] the bias terms are zero, so this reduces to
    /// `gear * clamp(ctrl)` and `q`/`qdot` are unused.
    pub fn force(&self, ctrl: f64, q: f64, qdot: f64) -> f64 {
        let ctrl = self.clamp_ctrl(ctrl);
        let f =
            self.gear * (self.gain * ctrl + self.bias[0] + self.bias[1] * q + self.bias[2] * qdot);
        match self.force_range {
            Some([lo, hi]) => f.clamp(lo.min(hi), hi.max(lo)),
            None => f,
        }
    }
}

/// Static model describing the topology and parameters of a physical system.
#[derive(Debug, Clone)]
pub struct Model {
    /// Bodies in the kinematic tree (index 0 = first body, no world body).
    pub bodies: Vec<Body>,
    /// Joints connecting bodies.
    pub joints: Vec<Joint>,
    /// Gravity vector in world frame.
    pub gravity: Vec3,
    /// Integration timestep.
    pub dt: f64,
    /// Total number of position DOFs.
    pub nq: usize,
    /// Total number of velocity DOFs.
    pub nv: usize,
    /// Position DOF offset for each joint.
    pub q_offsets: Vec<usize>,
    /// Velocity DOF offset for each joint.
    pub v_offsets: Vec<usize>,
    /// Actuators (motors) acting on joints.
    pub actuators: Vec<Actuator>,
}

impl Model {
    /// Create a default empty state for this model.
    pub fn default_state(&self) -> State {
        State::new_with_nu(self.nq, self.nv, self.nu(), self.bodies.len())
    }

    /// Number of actuators (MuJoCo's `nu`).
    pub fn nu(&self) -> usize {
        self.actuators.len()
    }

    /// Map actuator-space controls to generalized forces of dimension `nv`.
    ///
    /// Each actuator's control is clamped to its `ctrl_range` and turned into a
    /// force by [`Actuator::force`], which is added to the first velocity DOF of
    /// the joint it drives. `q` and `v` supply that joint's state, which
    /// position and velocity servos need for their bias terms; a plain motor
    /// ignores them.
    ///
    /// When the model has no actuators, `ctrl` is interpreted as a raw per-DOF
    /// generalized force and passed through unchanged — this keeps hand-built
    /// models (which set `state.ctrl` per DOF) working.
    pub fn actuator_forces(
        &self,
        ctrl: &phyz_math::DVec,
        q: &phyz_math::DVec,
        v: &phyz_math::DVec,
    ) -> phyz_math::DVec {
        let mut qfrc = phyz_math::DVec::zeros(self.nv);
        if self.actuators.is_empty() {
            for i in 0..self.nv.min(ctrl.len()) {
                qfrc[i] = ctrl[i];
            }
            return qfrc;
        }
        for (a, act) in self.actuators.iter().enumerate() {
            if a >= ctrl.len() {
                break;
            }
            let Some(&v_idx) = self.v_offsets.get(act.joint_idx) else {
                continue;
            };
            if v_idx < self.nv {
                let q_idx = self.q_offsets.get(act.joint_idx).copied().unwrap_or(v_idx);
                let qj = if q_idx < q.len() { q[q_idx] } else { 0.0 };
                let vj = if v_idx < v.len() { v[v_idx] } else { 0.0 };
                qfrc[v_idx] += act.force(ctrl[a], qj, vj);
            }
        }
        qfrc
    }

    /// Number of bodies.
    pub fn nbodies(&self) -> usize {
        self.bodies.len()
    }

    /// Look up a body index by name.
    pub fn body_index(&self, name: &str) -> Option<usize> {
        self.bodies.iter().position(|b| b.name == name)
    }

    /// Look up a joint index by name.
    pub fn joint_index(&self, name: &str) -> Option<usize> {
        self.joints.iter().position(|j| j.name == name)
    }
}

/// Builder for constructing models.
pub struct ModelBuilder {
    bodies: Vec<Body>,
    joints: Vec<Joint>,
    gravity: Vec3,
    dt: f64,
}

impl ModelBuilder {
    /// Start building a new model.
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            joints: Vec::new(),
            gravity: Vec3::new(0.0, 0.0, -GRAVITY),
            dt: 0.001,
        }
    }

    /// Set the gravity vector.
    pub fn gravity(mut self, g: Vec3) -> Self {
        self.gravity = g;
        self
    }

    /// Set the timestep.
    pub fn dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Add a body with a revolute joint attached to the given parent.
    ///
    /// `parent` is the index of the parent body, or -1 for world.
    /// `parent_to_joint` is the transform from parent body frame to joint frame.
    /// `inertia` is the body's spatial inertia in its own frame.
    pub fn add_revolute_body(
        mut self,
        name: &str,
        parent: i32,
        parent_to_joint: SpatialTransform,
        inertia: SpatialInertia,
    ) -> Self {
        let joint_idx = self.joints.len();
        self.joints.push(Joint::revolute(parent_to_joint));
        self.bodies
            .push(Body::new(name, inertia, parent, joint_idx));
        self
    }

    /// Add a body with a prismatic joint attached to the given parent.
    pub fn add_prismatic_body(
        mut self,
        name: &str,
        parent: i32,
        parent_to_joint: SpatialTransform,
        axis: Vec3,
        inertia: SpatialInertia,
    ) -> Self {
        let joint_idx = self.joints.len();
        self.joints.push(Joint::prismatic(parent_to_joint, axis));
        self.bodies
            .push(Body::new(name, inertia, parent, joint_idx));
        self
    }

    /// Add a body with a spherical (ball) joint attached to the given parent.
    pub fn add_spherical_body(
        mut self,
        name: &str,
        parent: i32,
        parent_to_joint: SpatialTransform,
        inertia: SpatialInertia,
    ) -> Self {
        let joint_idx = self.joints.len();
        self.joints.push(Joint::spherical(parent_to_joint));
        self.bodies
            .push(Body::new(name, inertia, parent, joint_idx));
        self
    }

    /// Add a body with a free joint (6 DOF) attached to the given parent.
    pub fn add_free_body(
        mut self,
        name: &str,
        parent: i32,
        parent_to_joint: SpatialTransform,
        inertia: SpatialInertia,
    ) -> Self {
        let joint_idx = self.joints.len();
        self.joints.push(Joint::free(parent_to_joint));
        self.bodies
            .push(Body::new(name, inertia, parent, joint_idx));
        self
    }

    /// Add a body with a fixed joint (0 DOF) attached to the given parent.
    pub fn add_fixed_body(
        mut self,
        name: &str,
        parent: i32,
        parent_to_joint: SpatialTransform,
        inertia: SpatialInertia,
    ) -> Self {
        let joint_idx = self.joints.len();
        self.joints.push(Joint::fixed(parent_to_joint));
        self.bodies
            .push(Body::new(name, inertia, parent, joint_idx));
        self
    }

    /// Add a generic joint and body.
    pub fn add_body(
        mut self,
        name: &str,
        parent: i32,
        joint: Joint,
        inertia: SpatialInertia,
    ) -> Self {
        let joint_idx = self.joints.len();
        self.joints.push(joint);
        self.bodies
            .push(Body::new(name, inertia, parent, joint_idx));
        self
    }

    /// Add a free body with collision geometry (useful for dropping objects).
    pub fn add_free_body_with_geometry(
        mut self,
        name: &str,
        parent: i32,
        parent_to_joint: SpatialTransform,
        inertia: SpatialInertia,
        geometry: crate::Body,
    ) -> Self {
        let joint_idx = self.joints.len();
        self.joints.push(Joint::free(parent_to_joint));
        let mut body = Body::new(name, inertia, parent, joint_idx);
        body.geometry = geometry.geometry;
        body.collisions = geometry.collisions;
        body.visuals = geometry.visuals;
        self.bodies.push(body);
        self
    }

    /// Build the model.
    pub fn build(self) -> Model {
        let mut nq = 0;
        let mut nv = 0;
        let mut q_offsets = Vec::new();
        let mut v_offsets = Vec::new();

        for joint in &self.joints {
            q_offsets.push(nq);
            v_offsets.push(nv);
            nq += joint.ndof();
            nv += joint.ndof();
        }

        Model {
            bodies: self.bodies,
            joints: self.joints,
            gravity: self.gravity,
            dt: self.dt,
            nq,
            nv,
            q_offsets,
            v_offsets,
            actuators: Vec::new(),
        }
    }
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}
