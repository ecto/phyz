//! Sensor models for extracting observations from simulation state.
//!
//! Every sensor here reports a real, computed quantity. Sensors that once
//! returned hard-coded zeros (`BodyAccel`, `ForceTorque`, the accelerometer
//! half of `Imu`) now read the per-body spatial accelerations and joint
//! wrenches that the dynamics recursions already produce — see
//! [`phyz_rigid::aba_dynamics`] and [`phyz_rigid::rnea_with_wrenches`].
//!
//! A sensor that cannot be computed does not silently return zeros. It returns
//! [`SensorError`], because a policy trained against dead observation channels
//! fails in a way nothing surfaces.

use phyz_math::{Mat3, SpatialVec, Vec3};
use phyz_model::{Model, State};
use phyz_rigid::{Dynamics, aba_dynamics};

/// Sensor types for extracting observations from simulation.
#[derive(Debug, Clone, PartialEq)]
pub enum Sensor {
    /// Joint position and velocity for one joint (`2 * ndof` values).
    JointState {
        /// Index into `Model::joints`.
        joint_idx: usize,
    },
    /// Proper linear acceleration of a body origin, in the body frame (3).
    ///
    /// This is what an accelerometer measures, so it includes gravity: a body
    /// at rest reads `+g` along its local up axis, not zero.
    BodyAccel {
        /// Index into `Model::bodies`.
        body_idx: usize,
    },
    /// Body-frame angular velocity (3).
    BodyAngularVel {
        /// Index into `Model::bodies`.
        body_idx: usize,
    },
    /// Spatial force transmitted through a body's inboard joint (6:
    /// torque then force, body frame).
    ForceTorque {
        /// Index into `Model::bodies`.
        body_idx: usize,
    },
    /// IMU: proper acceleration (3) then angular velocity (3), body frame.
    Imu {
        /// Index into `Model::bodies`.
        body_idx: usize,
    },
    /// Body pose as `[x, y, z, qw, qx, qy, qz]` in world coordinates (7).
    FrameCapture {
        /// Index into `Model::bodies`.
        body_idx: usize,
    },
    /// Distance along a body's local axis to the nearest obstacle.
    ///
    /// **Not implemented.** `phyz-collision` provides GJK distance queries but
    /// no ray cast, so there is nothing honest to return. Reading this sensor
    /// yields [`SensorError::NotImplemented`] rather than `max_dist`, which
    /// would be indistinguishable from "nothing in range".
    Rangefinder {
        /// Index into `Model::bodies`.
        body_idx: usize,
        /// Maximum range in metres.
        max_dist: f64,
        /// Ray direction in the body frame.
        direction: Vec3,
    },
}

/// Why a sensor could not produce a reading.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SensorError {
    /// The sensor refers to a body or joint that does not exist.
    #[error("sensor target index {index} is out of range ({count} available)")]
    OutOfRange {
        /// The offending index.
        index: usize,
        /// How many exist.
        count: usize,
    },
    /// The engine cannot compute this quantity yet.
    #[error("sensor '{kind}' is not implemented: {reason}")]
    NotImplemented {
        /// Sensor variant name.
        kind: &'static str,
        /// What is missing.
        reason: &'static str,
    },
}

/// Output from a sensor reading.
#[derive(Debug, Clone)]
pub struct SensorOutput {
    /// Sensor identifier (index in sensor array).
    pub sensor_id: usize,
    /// Simulation time when reading was taken.
    pub timestamp: f64,
    /// Flattened sensor data.
    pub data: Vec<f64>,
}

/// Dynamics quantities shared by every sensor in one reading.
///
/// Computing this once per step and passing it to each sensor avoids running
/// ABA per sensor, which for an IMU-per-body setup would be quadratic.
pub struct SensorContext {
    dynamics: Dynamics,
    wrenches: Option<Vec<SpatialVec>>,
}

impl SensorContext {
    /// Compute the dynamics quantities the sensors will read.
    ///
    /// `need_wrenches` additionally runs RNEA; skip it unless a
    /// [`Sensor::ForceTorque`] is present.
    pub fn new(model: &Model, state: &State, need_wrenches: bool) -> Self {
        let dynamics = aba_dynamics(model, state, None);
        let wrenches = if need_wrenches {
            Some(phyz_rigid::rnea_with_wrenches(model, state, &dynamics.qdd).1)
        } else {
            None
        };
        Self {
            dynamics,
            wrenches,
        }
    }

    /// Whether any of `sensors` needs the RNEA pass.
    pub fn wrenches_needed(sensors: &[Sensor]) -> bool {
        sensors
            .iter()
            .any(|s| matches!(s, Sensor::ForceTorque { .. }))
    }
}

impl Sensor {
    /// Read this sensor.
    pub fn read(
        &self,
        model: &Model,
        state: &State,
        ctx: &SensorContext,
        sensor_id: usize,
    ) -> Result<SensorOutput, SensorError> {
        let nb = model.nbodies();
        let check = |i: usize, count: usize| -> Result<(), SensorError> {
            if i >= count {
                Err(SensorError::OutOfRange { index: i, count })
            } else {
                Ok(())
            }
        };

        let data = match self {
            Sensor::JointState { joint_idx } => {
                check(*joint_idx, model.joints.len())?;
                let q_off = model.q_offsets[*joint_idx];
                let v_off = model.v_offsets[*joint_idx];
                let ndof = model.joints[*joint_idx].ndof();
                let mut out = Vec::with_capacity(ndof * 2);
                out.extend((0..ndof).map(|i| state.q[q_off + i]));
                out.extend((0..ndof).map(|i| state.v[v_off + i]));
                out
            }

            Sensor::BodyAccel { body_idx } => {
                check(*body_idx, nb)?;
                let a = proper_accel(&ctx.dynamics, *body_idx);
                vec![a.x, a.y, a.z]
            }

            Sensor::BodyAngularVel { body_idx } => {
                check(*body_idx, nb)?;
                let w = ctx.dynamics.vel[*body_idx].angular;
                vec![w.x, w.y, w.z]
            }

            Sensor::ForceTorque { body_idx } => {
                check(*body_idx, nb)?;
                let w = ctx.wrenches.as_ref().ok_or(SensorError::NotImplemented {
                    kind: "ForceTorque",
                    reason: "SensorContext was built without wrenches; pass need_wrenches = true",
                })?;
                let f = w[*body_idx];
                vec![
                    f.angular.x,
                    f.angular.y,
                    f.angular.z,
                    f.linear.x,
                    f.linear.y,
                    f.linear.z,
                ]
            }

            Sensor::Imu { body_idx } => {
                check(*body_idx, nb)?;
                let a = proper_accel(&ctx.dynamics, *body_idx);
                let w = ctx.dynamics.vel[*body_idx].angular;
                vec![a.x, a.y, a.z, w.x, w.y, w.z]
            }

            Sensor::FrameCapture { body_idx } => {
                check(*body_idx, nb)?;
                // `xform.pos` is already the world position and `xform.rot` is
                // world→body; see phyz-rigid/tests/frame_conventions.rs.
                let world = &ctx.dynamics.xform[*body_idx];
                let q = mat3_to_quat(&world.rot.transpose());
                vec![
                    world.pos.x,
                    world.pos.y,
                    world.pos.z,
                    q.0,
                    q.1,
                    q.2,
                    q.3,
                ]
            }

            Sensor::Rangefinder { .. } => {
                return Err(SensorError::NotImplemented {
                    kind: "Rangefinder",
                    reason: "phyz-collision has no ray cast; returning max_dist would be \
                             indistinguishable from an empty scene",
                });
            }
        };

        Ok(SensorOutput {
            sensor_id,
            timestamp: state.time,
            data,
        })
    }

    /// Output width for this sensor on `model`.
    pub fn output_dim(&self, model: &Model) -> usize {
        match self {
            Sensor::JointState { joint_idx } => model
                .joints
                .get(*joint_idx)
                .map(|j| j.ndof() * 2)
                .unwrap_or(0),
            Sensor::BodyAccel { .. } | Sensor::BodyAngularVel { .. } => 3,
            Sensor::ForceTorque { .. } | Sensor::Imu { .. } => 6,
            Sensor::FrameCapture { .. } => 7,
            Sensor::Rangefinder { .. } => 1,
        }
    }

    /// Whether this sensor can produce a reading at all on this build.
    pub fn is_implemented(&self) -> bool {
        !matches!(self, Sensor::Rangefinder { .. })
    }
}

/// Proper (accelerometer-measured) linear acceleration of a body origin.
///
/// ABA's pass 3 uses the base-acceleration trick: gravity enters as a fictitious
/// base acceleration of `-g`, so the spatial acceleration it produces is already
/// the *proper* acceleration. The classical-to-spatial correction `ω × v` turns
/// the spatial acceleration's linear part into the acceleration of the material
/// point currently at the body origin.
fn proper_accel(d: &Dynamics, i: usize) -> Vec3 {
    let a = d.acc[i];
    let v = d.vel[i];
    a.linear + v.angular.cross(v.linear)
}

/// Rotation matrix to `(w, x, y, z)` quaternion, Shepperd's method.
fn mat3_to_quat(mat: &Mat3) -> (f64, f64, f64, f64) {
    let trace = mat[(0, 0)] + mat[(1, 1)] + mat[(2, 2)];

    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        (
            0.25 * s,
            (mat[(2, 1)] - mat[(1, 2)]) / s,
            (mat[(0, 2)] - mat[(2, 0)]) / s,
            (mat[(1, 0)] - mat[(0, 1)]) / s,
        )
    } else if mat[(0, 0)] > mat[(1, 1)] && mat[(0, 0)] > mat[(2, 2)] {
        let s = (1.0 + mat[(0, 0)] - mat[(1, 1)] - mat[(2, 2)]).sqrt() * 2.0;
        (
            (mat[(2, 1)] - mat[(1, 2)]) / s,
            0.25 * s,
            (mat[(0, 1)] + mat[(1, 0)]) / s,
            (mat[(0, 2)] + mat[(2, 0)]) / s,
        )
    } else if mat[(1, 1)] > mat[(2, 2)] {
        let s = (1.0 + mat[(1, 1)] - mat[(0, 0)] - mat[(2, 2)]).sqrt() * 2.0;
        (
            (mat[(0, 2)] - mat[(2, 0)]) / s,
            (mat[(0, 1)] + mat[(1, 0)]) / s,
            0.25 * s,
            (mat[(1, 2)] + mat[(2, 1)]) / s,
        )
    } else {
        let s = (1.0 + mat[(2, 2)] - mat[(0, 0)] - mat[(1, 1)]).sqrt() * 2.0;
        (
            (mat[(1, 0)] - mat[(0, 1)]) / s,
            (mat[(0, 2)] + mat[(2, 0)]) / s,
            (mat[(1, 2)] + mat[(2, 1)]) / s,
            0.25 * s,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{SpatialInertia, SpatialTransform};
    use phyz_model::{Joint, ModelBuilder};

    fn pendulum() -> Model {
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -9.81))
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build()
    }

    fn free_body() -> Model {
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -9.81))
            .add_body(
                "body",
                -1,
                Joint::free(SpatialTransform::identity()),
                SpatialInertia::new(
                    1.0,
                    Vec3::zeros(),
                    Mat3::from_diagonal(&Vec3::new(0.1, 0.1, 0.1)),
                ),
            )
            .build()
    }

    fn read(model: &Model, state: &State, s: &Sensor) -> Vec<f64> {
        let ctx = SensorContext::new(model, state, SensorContext::wrenches_needed(std::slice::from_ref(s)));
        s.read(model, state, &ctx, 0).unwrap().data
    }

    #[test]
    fn joint_state_reports_q_and_v() {
        let model = pendulum();
        let mut state = model.default_state();
        state.q[0] = 0.5;
        state.v[0] = 1.0;
        assert_eq!(read(&model, &state, &Sensor::JointState { joint_idx: 0 }), vec![0.5, 1.0]);
    }

    #[test]
    fn angular_velocity_follows_the_joint_axis() {
        let model = pendulum();
        let mut state = model.default_state();
        state.v[0] = 2.0;
        let w = read(&model, &state, &Sensor::BodyAngularVel { body_idx: 0 });
        assert!((w[2] - 2.0).abs() < 1e-10, "{w:?}");
    }

    /// The regression that matters: this used to be hard-coded `[0, 0, 0]`.
    /// A free body in free fall is weightless, so its accelerometer reads zero;
    /// but that must be a *computed* zero, so check a body that is not in free
    /// fall as well.
    #[test]
    fn body_accel_is_computed_not_zero() {
        let model = pendulum();
        let mut state = model.default_state();
        state.q[0] = std::f64::consts::FRAC_PI_2; // horizontal, max gravity torque
        let a = read(&model, &state, &Sensor::BodyAccel { body_idx: 0 });
        assert!(
            a.iter().any(|x| x.abs() > 1e-6),
            "accelerometer on a swinging pendulum must not read zero: {a:?}"
        );
        assert!(a.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn free_falling_body_is_weightless() {
        let model = free_body();
        let state = model.default_state();
        let a = read(&model, &state, &Sensor::BodyAccel { body_idx: 0 });
        for x in &a {
            assert!(x.abs() < 1e-9, "free fall should read ~0, got {a:?}");
        }
    }

    #[test]
    fn imu_reports_accel_then_gyro() {
        let model = pendulum();
        let mut state = model.default_state();
        state.q[0] = 0.7;
        state.v[0] = 1.5;
        let d = read(&model, &state, &Sensor::Imu { body_idx: 0 });
        assert_eq!(d.len(), 6);
        assert!((d[5] - 1.5).abs() < 1e-10, "gyro z should be qd: {d:?}");
    }

    /// Also previously hard-coded zeros. A loaded pendulum transmits a real
    /// wrench through its joint.
    #[test]
    fn force_torque_is_computed_not_zero() {
        let model = pendulum();
        let mut state = model.default_state();
        state.q[0] = std::f64::consts::FRAC_PI_2;
        let f = read(&model, &state, &Sensor::ForceTorque { body_idx: 0 });
        assert_eq!(f.len(), 6);
        assert!(
            f.iter().any(|x| x.abs() > 1e-6),
            "joint wrench must not be zero: {f:?}"
        );
    }

    #[test]
    fn force_torque_without_wrenches_errors_rather_than_lying() {
        let model = pendulum();
        let state = model.default_state();
        let ctx = SensorContext::new(&model, &state, false);
        let err = Sensor::ForceTorque { body_idx: 0 }
            .read(&model, &state, &ctx, 0)
            .unwrap_err();
        assert!(matches!(err, SensorError::NotImplemented { .. }));
    }

    #[test]
    fn rangefinder_errors_instead_of_returning_max_dist() {
        let model = pendulum();
        let state = model.default_state();
        let ctx = SensorContext::new(&model, &state, false);
        let s = Sensor::Rangefinder {
            body_idx: 0,
            max_dist: 10.0,
            direction: Vec3::new(0.0, 0.0, -1.0),
        };
        assert!(!s.is_implemented());
        assert!(matches!(
            s.read(&model, &state, &ctx, 0),
            Err(SensorError::NotImplemented { .. })
        ));
    }

    #[test]
    fn out_of_range_target_is_an_error() {
        let model = pendulum();
        let state = model.default_state();
        let ctx = SensorContext::new(&model, &state, false);
        assert!(matches!(
            Sensor::BodyAccel { body_idx: 99 }.read(&model, &state, &ctx, 0),
            Err(SensorError::OutOfRange { .. })
        ));
    }

    #[test]
    fn frame_capture_reports_world_pose() {
        let model = free_body();
        let mut state = model.default_state();
        state.q[2] = 2.5;
        let d = read(&model, &state, &Sensor::FrameCapture { body_idx: 0 });
        assert_eq!(d.len(), 7);
        assert!((d[2] - 2.5).abs() < 1e-9, "world z: {d:?}");
        assert!((d[3] - 1.0).abs() < 1e-9, "identity quaternion: {d:?}");
    }
}
