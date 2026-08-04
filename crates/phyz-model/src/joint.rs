//! Joint types and definitions.

use phyz_math::{DMat, Mat3, Quat, SpatialTransform, SpatialVec, Vec3};

/// Joint type enumeration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointType {
    /// Single rotational DOF about an axis.
    Revolute,
    /// Single translational DOF along an axis.
    Prismatic,
    /// 3 DOF spherical joint (ball joint) using quaternions.
    Spherical,
    /// 6 DOF free joint (3 translation + 3 rotation).
    Free,
    /// 0 DOF fixed joint (rigid attachment).
    Fixed,
    /// Alias for Revolute (MuJoCo compatibility).
    Hinge,
    /// Alias for Prismatic (MuJoCo compatibility).
    Slide,
    /// Alias for Spherical (MuJoCo compatibility).
    Ball,
}

/// A joint connecting two bodies.
#[derive(Debug, Clone)]
pub struct Joint {
    /// Joint type.
    pub joint_type: JointType,
    /// Transform from parent body frame to joint frame (constant).
    ///
    /// `pos` is the joint origin expressed in parent-body coordinates and
    /// `rot` is the coordinate transform *parent → joint*, i.e. the joint's
    /// orientation in the parent frame is `rot.transpose()`.
    pub parent_to_joint: SpatialTransform,
    /// Joint axis, expressed in the joint frame (for revolute: typically Z).
    pub axis: Vec3,
    /// Damping coefficient (force = -damping * qd).
    pub damping: f64,
    /// Joint position limits [lower, upper] (None = unlimited).
    ///
    /// Only meaningful for single-DOF joints (revolute/prismatic). Limits on
    /// ball and free joints are ignored — MuJoCo expresses those as a cone
    /// limit, which this model does not represent yet.
    pub limits: Option<[f64; 2]>,
    /// Stiffness of the soft joint-limit force (force per unit of violation).
    pub limit_stiffness: f64,
    /// Damping of the soft joint-limit force (force per unit of inward velocity).
    pub limit_damping: f64,
    /// Rotor / armature inertia added to this joint's diagonal of the mass matrix.
    ///
    /// MuJoCo's `armature`. Physically it is the reflected inertia of the motor
    /// rotor; numerically it also regularises the mass matrix.
    pub armature: f64,
    /// Passive spring stiffness (force = -stiffness * (q - spring_ref)).
    pub stiffness: f64,
    /// Rest position of the passive spring (MuJoCo's `springref`).
    pub spring_ref: f64,
    /// Dry (Coulomb) friction magnitude — MuJoCo's `frictionloss`, and where
    /// URDF's `<dynamics friction="...">` lands.
    pub friction_loss: f64,
    /// Actuation effort limit (torque or force), if the source declares one.
    ///
    /// Descriptive only: nothing in the solver clamps to it yet.
    pub effort_limit: Option<f64>,
    /// Velocity limit, if the source declares one. Descriptive only.
    pub velocity_limit: Option<f64>,
    /// Name of the joint in the source file (empty if unnamed).
    pub name: String,
}

impl Default for Joint {
    fn default() -> Self {
        Self {
            joint_type: JointType::Fixed,
            parent_to_joint: SpatialTransform::identity(),
            axis: Vec3::zeros(),
            damping: 0.0,
            limits: None,
            limit_stiffness: DEFAULT_LIMIT_STIFFNESS,
            limit_damping: DEFAULT_LIMIT_DAMPING,
            armature: 0.0,
            stiffness: 0.0,
            spring_ref: 0.0,
            friction_loss: 0.0,
            effort_limit: None,
            velocity_limit: None,
            name: String::new(),
        }
    }
}

/// Default stiffness of the soft joint-limit force.
///
/// Chosen to be stiff enough that a limit is visibly hard at the ~1 ms
/// timesteps this engine targets, but soft enough to stay explicit-integrator
/// stable for unit-scale inertias. Models with very different scales should set
/// [`Joint::limit_stiffness`] explicitly.
pub const DEFAULT_LIMIT_STIFFNESS: f64 = 1000.0;

/// Default damping of the soft joint-limit force. Critical-ish for a unit
/// inertia against [`DEFAULT_LIMIT_STIFFNESS`], which kills limit chatter.
pub const DEFAULT_LIMIT_DAMPING: f64 = 60.0;

impl Joint {
    /// Create a revolute joint with the given parent-to-joint transform.
    pub fn revolute(parent_to_joint: SpatialTransform) -> Self {
        Self {
            joint_type: JointType::Revolute,
            parent_to_joint,
            axis: Vec3::new(0.0, 0.0, 1.0), // revolute about Z
            ..Default::default()
        }
    }

    /// Create a prismatic joint with the given parent-to-joint transform and axis.
    pub fn prismatic(parent_to_joint: SpatialTransform, axis: Vec3) -> Self {
        Self {
            joint_type: JointType::Prismatic,
            parent_to_joint,
            axis,
            ..Default::default()
        }
    }

    /// Create a spherical (ball) joint with the given parent-to-joint transform.
    pub fn spherical(parent_to_joint: SpatialTransform) -> Self {
        Self {
            joint_type: JointType::Spherical,
            parent_to_joint,
            axis: Vec3::zeros(), // not used for spherical
            ..Default::default()
        }
    }

    /// Create a free joint with the given parent-to-joint transform.
    pub fn free(parent_to_joint: SpatialTransform) -> Self {
        Self {
            joint_type: JointType::Free,
            parent_to_joint,
            axis: Vec3::zeros(), // not used for free
            ..Default::default()
        }
    }

    /// Create a fixed joint (rigid attachment).
    pub fn fixed(parent_to_joint: SpatialTransform) -> Self {
        Self {
            joint_type: JointType::Fixed,
            parent_to_joint,
            axis: Vec3::zeros(),
            ..Default::default()
        }
    }

    /// Set the joint name (builder style).
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Number of degrees of freedom for this joint type.
    pub fn ndof(&self) -> usize {
        match self.joint_type {
            JointType::Revolute | JointType::Hinge => 1,
            JointType::Prismatic | JointType::Slide => 1,
            JointType::Spherical | JointType::Ball => 3,
            JointType::Free => 6,
            JointType::Fixed => 0,
        }
    }

    /// Soft joint-limit force for a single-DOF joint at position `q`, velocity `qd`.
    ///
    /// Returns a generalized force along the joint's DOF (positive = pushes `q`
    /// up). Zero when the joint is unlimited, multi-DOF, or inside its range.
    ///
    /// # Why a penalty force
    ///
    /// The contact layer is currently a penalty method, so limits are modelled
    /// the same way for consistency: a one-sided spring-damper on the violation
    /// depth. The damping term is *gated* — it only ever resists motion deeper
    /// into the limit, never pulls the joint back toward the violation — so this
    /// force is strictly dissipative and cannot inject energy at the boundary
    /// (which is what makes naive `-d * qd` limit damping oscillate).
    ///
    /// # Migration to a constraint
    ///
    /// This is deliberately expressed as a pure function of `(q, qd)` returning
    /// a force, with the violation depth and its sign computed by
    /// [`Joint::limit_violation`]. When the solver gains unilateral constraints,
    /// `limit_violation` becomes the constraint function `g(q) >= 0` and its
    /// Jacobian row is the joint's motion subspace; only the force assembly here
    /// is replaced, not the model or the parsing that feeds it.
    pub fn limit_force(&self, q: f64, qd: f64) -> f64 {
        let Some((depth, sign)) = self.limit_violation(q) else {
            return 0.0;
        };
        // `sign` is the direction the restoring force must push (+1 at the lower
        // limit, -1 at the upper). Damping engages only while the joint is still
        // moving into the limit, i.e. when qd opposes `sign`.
        let inward_speed = (-sign * qd).max(0.0);
        sign * (self.limit_stiffness * depth + self.limit_damping * inward_speed)
    }

    /// Limit violation at position `q`, as `(depth, restoring_direction)`.
    ///
    /// `depth` is a non-negative penetration past the limit; the direction is
    /// `+1.0` past the lower bound and `-1.0` past the upper bound. Returns
    /// `None` when the joint is unlimited, multi-DOF, or within range.
    ///
    /// This is the constraint function a future unilateral solver would use.
    pub fn limit_violation(&self, q: f64) -> Option<(f64, f64)> {
        if self.ndof() != 1 {
            return None;
        }
        let [lo, hi] = self.limits?;
        if q < lo {
            Some((lo - q, 1.0))
        } else if q > hi {
            Some((q - hi, -1.0))
        } else {
            None
        }
    }

    /// Total passive generalized force on DOF `k` of this joint.
    ///
    /// Combines viscous damping, the passive spring, dry friction and the soft
    /// joint limit. `q`/`qd` are this DOF's position and velocity.
    pub fn passive_force(&self, k: usize, q: f64, qd: f64) -> f64 {
        let mut f = -self.damping * qd;
        if self.ndof() == 1 && k == 0 {
            f += -self.stiffness * (q - self.spring_ref);
            f += self.limit_force(q, qd);
            if self.friction_loss > 0.0 {
                // Smoothed Coulomb friction: -mu * sign(qd), regularised so a
                // joint at rest does not chatter under an explicit integrator.
                // A proper implementation is a box constraint on the friction
                // force, which lands with the constraint solver.
                const FRICTION_VEL_SCALE: f64 = 1e-3;
                f += -self.friction_loss * (qd / FRICTION_VEL_SCALE).tanh();
            }
        }
        f
    }

    /// Compute the joint transform for the given joint position(s).
    ///
    /// Returns the Plücker transform from predecessor to successor frame.
    /// `q` slice should have length >= ndof().
    pub fn joint_transform_slice(&self, q: &[f64]) -> SpatialTransform {
        match self.joint_type {
            JointType::Revolute | JointType::Hinge => {
                // Passive rotation: negate angle for coordinate transform
                let angle = q[0];
                let (s, c) = (-angle).sin_cos();
                let a = &self.axis;
                let ax = phyz_math::skew(a);
                let rot = Mat3::identity() + ax * s + ax * ax * (1.0 - c);
                SpatialTransform::new(rot, Vec3::zeros())
            }
            JointType::Prismatic | JointType::Slide => {
                let distance = q[0];
                let pos = self.axis * distance;
                SpatialTransform::new(Mat3::identity(), pos)
            }
            JointType::Spherical | JointType::Ball => {
                // q = exponential coordinates (3) of the child→parent
                // rotation R, integrated as R ← R·exp(ω·dt) (body angular
                // velocity composed on the right — see
                // phyz_rigid::integrate). The joint transform is the
                // *coordinate map* parent→child, i.e. Rᵀ: same negation the
                // hinge branch applies to its angle. Using R here made the
                // dynamics (which see q only through this map) disagree with
                // the integrator by an inverse, which pumped energy into any
                // passive spherical joint the moment its axis moved.
                let w = Vec3::new(q[0], q[1], q[2]);
                let rot = Quat::exp(&w).to_matrix().transpose();
                SpatialTransform::new(rot, Vec3::zeros())
            }
            JointType::Free => {
                // ### FREE-JOINT DOF ORDER — ANGULAR FIRST. ###
                //
                // q = [wx, wy, wz, x, y, z]: exp-coords of the child→parent
                // rotation, *then* position in parent coords. This matches
                // `SpatialVec`'s `[angular; linear]` order, which is what the
                // free joint's motion subspace (the 6×6 identity, see
                // `motion_subspace_matrix`) and therefore `v`, `qdd`, `tau`,
                // the Jacobians and the contact solver all use.
                //
                // It did NOT used to. Until this was fixed, `q` was
                // `[x, y, z, wx, wy, wz]` — translation first — while `v` was
                // angular first, so the flat `q += v·dt` that most callers
                // performed fed the vertical acceleration into the yaw
                // exponential coordinate: a free body released under gravity
                // never fell, it spun up at 9.81 rad/s². If you are changing
                // this, change `phyz_rigid::integrate_configuration` and the
                // GPU `INTEGRATE_SHADER` in the same commit.
                //
                // Note this is *not* `phyz_diff`'s rollout layout, which packs
                // a free joint as `[x, y, z, quat(4)]` in its own private
                // `DofLayout` and never shares indices with `State::q`.
                let w = Vec3::new(q[0], q[1], q[2]);
                let pos = Vec3::new(q[3], q[4], q[5]);
                let rot = Quat::exp(&w).to_matrix().transpose();
                SpatialTransform::new(rot, pos)
            }
            JointType::Fixed => {
                // No DOF, return identity
                SpatialTransform::identity()
            }
        }
    }

    /// Compute the joint transform for a single-DOF joint (backward compat).
    /// For multi-DOF joints, use joint_transform_slice instead.
    pub fn joint_transform(&self, q: f64) -> SpatialTransform {
        self.joint_transform_slice(&[q])
    }

    /// Motion subspace matrix S for this joint.
    /// Returns a matrix of size 6 × ndof.
    /// For single-DOF joints, returns a 6×1 column vector.
    pub fn motion_subspace_matrix(&self) -> DMat {
        match self.joint_type {
            JointType::Revolute | JointType::Hinge => {
                let s = SpatialVec::new(self.axis, Vec3::zeros());
                let arr = s.as_array();
                DMat::from_row_slice(6, 1, &arr)
            }
            JointType::Prismatic | JointType::Slide => {
                let s = SpatialVec::new(Vec3::zeros(), self.axis);
                let arr = s.as_array();
                DMat::from_row_slice(6, 1, &arr)
            }
            JointType::Spherical | JointType::Ball => {
                // 3 DOF: angular velocity in body frame
                // S = [I_3×3; 0_3×3] (angular part is identity, linear is zero)
                DMat::from_row_slice(
                    6,
                    3,
                    &[
                        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                    ],
                )
            }
            JointType::Free => {
                // 6 DOF: [angular; linear] velocity
                // S = I_6×6
                DMat::identity(6)
            }
            JointType::Fixed => {
                // 0 DOF: empty 6×0 matrix
                DMat::zeros(6, 0)
            }
        }
    }

    /// Motion subspace for single-DOF joints (backward compat).
    /// For multi-DOF joints, use motion_subspace_matrix instead.
    /// Fixed joints return a zero vector.
    pub fn motion_subspace(&self) -> SpatialVec {
        match self.joint_type {
            JointType::Revolute | JointType::Hinge => SpatialVec::new(self.axis, Vec3::zeros()),
            JointType::Prismatic | JointType::Slide => SpatialVec::new(Vec3::zeros(), self.axis),
            JointType::Fixed => SpatialVec::zero(), // 0 DOF
            _ => panic!(
                "motion_subspace() only valid for single-DOF joints; use motion_subspace_matrix() for multi-DOF joints"
            ),
        }
    }
}
