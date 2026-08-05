//! Model definition — static description of a physical system.

use crate::{Body, Joint, JointType, State};
use phyz_math::{GRAVITY, SpatialInertia, SpatialTransform, Vec3};

/// An actuator attached to a joint.
///
/// Follows MuJoCo's affine actuator model: the generalized force applied to the
/// joint's first DOF is
///
/// ```text
/// f = gear * (gain * ctrl + bias_q * q + bias_v * v)
/// ```
///
/// which expresses `<motor>`, `<position>`, `<velocity>` and the affine subset
/// of `<general>` with one parameter set, so downstream code never branches on
/// actuator type. A plain motor is `gain = 1, bias_q = bias_v = 0`, for which
/// this reduces to `gear * clamp(ctrl)`.
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
    /// Coefficient on `ctrl`. 1 for a motor, `kp` for a position servo.
    pub gain: f64,
    /// Coefficient on joint position. `-kp` for a position servo, else 0.
    pub bias_q: f64,
    /// Coefficient on joint velocity. `-kv` for a servo, else 0.
    pub bias_v: f64,
    /// Force limits applied after the affine law, or `None` if unlimited.
    pub force_range: Option<[f64; 2]>,
}

impl Default for Actuator {
    fn default() -> Self {
        Self {
            name: String::new(),
            joint_name: String::new(),
            joint_idx: 0,
            gear: 1.0,
            ctrl_range: None,
            gain: 1.0,
            bias_q: 0.0,
            bias_v: 0.0,
            force_range: None,
        }
    }
}

impl Actuator {
    /// Clamp a raw control value to this actuator's control range.
    pub fn clamp_ctrl(&self, ctrl: f64) -> f64 {
        match self.ctrl_range {
            Some([lo, hi]) => ctrl.clamp(lo.min(hi), hi.max(lo)),
            None => ctrl,
        }
    }

    /// Generalized force produced by this actuator for a raw control value,
    /// ignoring any state feedback.
    ///
    /// Correct for motors. Position and velocity servos additionally depend on
    /// `q` and `v`; use [`Self::force_at`] for those.
    pub fn force(&self, ctrl: f64) -> f64 {
        self.force_at(ctrl, 0.0, 0.0)
    }

    /// Evaluate the full affine actuator law at the joint's current state.
    pub fn force_at(&self, ctrl: f64, q: f64, v: f64) -> f64 {
        let f = self.gear * (self.gain * self.clamp_ctrl(ctrl) + self.bias_q * q + self.bias_v * v);
        match self.force_range {
            Some([lo, hi]) => f.clamp(lo.min(hi), hi.max(lo)),
            None => f,
        }
    }

    /// A direct-torque motor: `f = gear * ctrl`.
    pub fn motor(name: &str, joint_name: &str, joint_idx: usize, gear: f64) -> Self {
        Self {
            name: name.to_string(),
            joint_name: joint_name.to_string(),
            joint_idx,
            gear,
            ..Default::default()
        }
    }

    /// A position servo: `f = gear * (kp * (ctrl - q) - kv * v)`.
    pub fn position(
        name: &str,
        joint_name: &str,
        joint_idx: usize,
        gear: f64,
        kp: f64,
        kv: f64,
    ) -> Self {
        Self {
            gain: kp,
            bias_q: -kp,
            bias_v: -kv,
            ..Self::motor(name, joint_name, joint_idx, gear)
        }
    }

    /// A velocity servo: `f = gear * kv * (ctrl - v)`.
    pub fn velocity(name: &str, joint_name: &str, joint_idx: usize, gear: f64, kv: f64) -> Self {
        Self {
            gain: kv,
            bias_v: -kv,
            ..Self::motor(name, joint_name, joint_idx, gear)
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
    /// Body pairs that must never produce a contact, beyond the structural
    /// exclusions [`Model::may_collide`] applies on its own.
    ///
    /// The escape hatch for authored geometry that overlaps across a *moving*
    /// joint — common in URDFs, where a shoulder capsule is drawn long enough
    /// to intersect the trunk box at some arm angles. Such a pair is a
    /// modelling artifact, not an event, but no rule derivable from the tree
    /// can tell it apart from a genuine self-touch, so it is data.
    ///
    /// Order within a pair does not matter.
    pub contact_exclude: Vec<(usize, usize)>,
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
    /// Each actuator's control is clamped to its `ctrl_range`, scaled by `gear`,
    /// and added to the first velocity DOF of the joint it drives.
    ///
    /// When the model has no actuators, `ctrl` is interpreted as a raw per-DOF
    /// generalized force and passed through unchanged — this keeps hand-built
    /// models (which set `state.ctrl` per DOF) working.
    pub fn actuator_forces(&self, ctrl: &phyz_math::DVec) -> phyz_math::DVec {
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
                qfrc[v_idx] += act.force(ctrl[a]);
            }
        }
        qfrc
    }

    /// Map actuator-space controls to generalized forces, evaluating the full
    /// affine actuator law against the joint's current state.
    ///
    /// Identical to [`Self::actuator_forces`] for plain motors. Position and
    /// velocity servos need `q` and `v`, so this is the variant the dynamics
    /// call; the state-free one remains for callers that only have controls.
    pub fn actuator_forces_at(
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
            let (Some(&v_idx), Some(&q_idx)) = (
                self.v_offsets.get(act.joint_idx),
                self.q_offsets.get(act.joint_idx),
            ) else {
                continue;
            };
            if v_idx < self.nv && q_idx < q.len() {
                qfrc[v_idx] += act.force_at(ctrl[a], q[q_idx], v[v_idx]);
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

    /// Weld-group id per body: bodies joined only by [`JointType::Fixed`]
    /// joints share an id, because no configuration can ever move them
    /// relative to one another.
    ///
    /// Used by contact filtering — see [`Model::may_collide`]. A welded pair
    /// overlaps by construction (that is what a fixed joint usually
    /// *expresses*: one physical part split across two links), and the
    /// overlap is constant, so a contact between them is a permanent
    /// unresolvable penetration rather than an event.
    ///
    /// Transitive, so a chain `A —fixed— B —fixed— C` puts all three in one
    /// group even though `A` and `C` are not parent and child.
    pub fn weld_groups(&self) -> Vec<usize> {
        let n = self.bodies.len();
        let mut group: Vec<usize> = (0..n).collect();

        // Union-find with path halving; the tree is shallow so this is
        // effectively linear.
        fn find(group: &mut [usize], mut i: usize) -> usize {
            while group[i] != i {
                group[i] = group[group[i]];
                i = group[i];
            }
            i
        }

        for (i, body) in self.bodies.iter().enumerate() {
            if body.parent < 0 {
                continue;
            }
            let Some(joint) = self.joints.get(body.joint_idx) else {
                continue;
            };
            if joint.joint_type != JointType::Fixed {
                continue;
            }
            let p = body.parent as usize;
            if p >= n {
                continue;
            }
            let (a, b) = (find(&mut group, i), find(&mut group, p));
            if a != b {
                group[a] = b;
            }
        }

        (0..n).map(|i| find(&mut group, i)).collect()
    }

    /// May bodies `i` and `j` produce a contact with each other?
    ///
    /// `welds` is [`Model::weld_groups`], passed in so a detection loop
    /// computes it once rather than per pair.
    ///
    /// Three structural exclusions, none of them tunable, because each one
    /// describes a pair that is in contact in *every* configuration and so
    /// carries no information:
    ///
    /// 1. **A body with itself.** Two shapes on one link cannot move relative
    ///    to each other.
    /// 2. **Parent and child.** The two links a joint connects overlap at the
    ///    joint by construction — that is what a joint *is*. A humanoid has
    ///    one such pair per joint, and left unfiltered they are the only
    ///    contacts the narrow phase ever reports.
    /// 3. **Welded bodies.** As (2), for links a fixed joint chain has made
    ///    rigid with respect to each other.
    ///
    /// Anything else — a hand reaching its own thigh, an arm crossing the
    /// trunk — is a real event and is allowed through. Models needing a
    /// further exclusion (geometry that overlaps across a *moving* joint, a
    /// common URDF authoring artifact) list it in [`Model::contact_exclude`].
    pub fn may_collide(&self, i: usize, j: usize, welds: &[usize]) -> bool {
        if i == j {
            return false;
        }
        if welds.get(i).is_some() && welds.get(i) == welds.get(j) {
            return false;
        }
        let parent_child = self.bodies.get(i).map(|b| b.parent) == Some(j as i32)
            || self.bodies.get(j).map(|b| b.parent) == Some(i as i32);
        if parent_child {
            return false;
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        !self
            .contact_exclude
            .iter()
            .any(|&(a, b)| (a.min(b), a.max(b)) == (lo, hi))
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
            contact_exclude: Vec::new(),
        }
    }
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}
