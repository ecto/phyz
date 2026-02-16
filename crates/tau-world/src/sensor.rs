//! Sensor models for extracting observations from simulation state.

use tau_model::{Model, State};

/// Sensor types for extracting observations from simulation.
#[derive(Debug, Clone)]
pub enum Sensor {
    /// Joint position and velocity sensor.
    JointState { joint_idx: usize },
    /// Body linear acceleration in world frame.
    BodyAccel { body_idx: usize },
    /// Body angular velocity in body frame.
    BodyAngularVel { body_idx: usize },
    /// Reaction force/torque at body.
    ForceTorque { body_idx: usize },
    /// Distance to nearest obstacle (placeholder - requires collision detection).
    Rangefinder { body_idx: usize, max_dist: f64 },
    /// IMU: acceleration + angular velocity in body frame.
    Imu { body_idx: usize },
    /// Snapshot of body transform.
    FrameCapture { body_idx: usize },
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

impl Sensor {
    /// Read sensor output from current state.
    ///
    /// Returns a SensorOutput with sensor_id set to 0 (caller should update).
    pub fn read(&self, model: &Model, state: &State, sensor_id: usize) -> SensorOutput {
        let data = match self {
            Sensor::JointState { joint_idx } => {
                let q_off = model.q_offsets[*joint_idx];
                let v_off = model.v_offsets[*joint_idx];
                let ndof = model.joints[*joint_idx].ndof();

                let mut out = Vec::with_capacity(ndof * 2);
                for i in 0..ndof {
                    out.push(state.q[q_off + i]);
                }
                for i in 0..ndof {
                    out.push(state.v[v_off + i]);
                }
                out
            }
            Sensor::BodyAccel { body_idx: _ } => {
                // Acceleration = d²x/dt² ≈ (v_next - v_curr) / dt
                // This requires storing previous velocity or computing numerically.
                // For now, return zero (placeholder).
                vec![0.0, 0.0, 0.0]
            }
            Sensor::BodyAngularVel { body_idx } => {
                // Extract angular velocity from body spatial velocity
                // v_spatial = S * qd where S is motion subspace
                // For now, approximate from joint velocities
                if *body_idx >= model.nbodies() {
                    vec![0.0, 0.0, 0.0]
                } else {
                    let joint_idx = model.bodies[*body_idx].joint_idx;
                    let v_off = model.v_offsets[joint_idx];
                    let joint = &model.joints[joint_idx];

                    match joint.ndof() {
                        1 => {
                            // Single revolute joint: angular velocity = axis * qd
                            let qd = state.v[v_off];
                            let w = joint.axis * qd;
                            vec![w.x, w.y, w.z]
                        }
                        3 | 6 => {
                            // Spherical or free joint: angular velocity is directly in state
                            vec![state.v[v_off], state.v[v_off + 1], state.v[v_off + 2]]
                        }
                        _ => vec![0.0, 0.0, 0.0],
                    }
                }
            }
            Sensor::ForceTorque { body_idx: _ } => {
                // Reaction forces require inverse dynamics (RNEA).
                // Placeholder: return zeros.
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
            Sensor::Rangefinder {
                body_idx: _,
                max_dist,
            } => {
                // Requires collision detection to find nearest obstacle.
                // Placeholder: return max distance.
                vec![*max_dist]
            }
            Sensor::Imu { body_idx } => {
                // IMU = acceleration (3) + angular velocity (3)
                // Placeholder: return zeros for acceleration, compute angular velocity
                let mut imu_data = vec![0.0, 0.0, 0.0]; // acceleration

                if *body_idx < model.nbodies() {
                    let joint_idx = model.bodies[*body_idx].joint_idx;
                    let v_off = model.v_offsets[joint_idx];
                    let joint = &model.joints[joint_idx];

                    let angular_vel = match joint.ndof() {
                        1 => {
                            let qd = state.v[v_off];
                            let w = joint.axis * qd;
                            vec![w.x, w.y, w.z]
                        }
                        3 | 6 => {
                            vec![state.v[v_off], state.v[v_off + 1], state.v[v_off + 2]]
                        }
                        _ => vec![0.0, 0.0, 0.0],
                    };
                    imu_data.extend(angular_vel);
                } else {
                    imu_data.extend(vec![0.0, 0.0, 0.0]);
                }

                imu_data
            }
            Sensor::FrameCapture { body_idx } => {
                // Return body transform as [x, y, z, qw, qx, qy, qz]
                if *body_idx < state.body_xform.len() {
                    let xform = &state.body_xform[*body_idx];
                    let pos = xform.pos;

                    // Convert rotation matrix to quaternion
                    // This is a simplified conversion - should use proper matrix->quat
                    let quat = mat3_to_quat(&xform.rot);

                    vec![pos.x, pos.y, pos.z, quat.0, quat.1, quat.2, quat.3]
                } else {
                    vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                }
            }
        };

        SensorOutput {
            sensor_id,
            timestamp: state.time,
            data,
        }
    }

    /// Get expected output dimension for this sensor.
    pub fn output_dim(&self) -> usize {
        match self {
            Sensor::JointState { .. } => 2, // q + v (multiplied by ndof at runtime)
            Sensor::BodyAccel { .. } => 3,
            Sensor::BodyAngularVel { .. } => 3,
            Sensor::ForceTorque { .. } => 6,
            Sensor::Rangefinder { .. } => 1,
            Sensor::Imu { .. } => 6,          // accel (3) + gyro (3)
            Sensor::FrameCapture { .. } => 7, // pos (3) + quat (4)
        }
    }
}

/// Helper function to convert rotation matrix to quaternion.
/// Returns (w, x, y, z).
fn mat3_to_quat(mat: &tau_math::Mat3) -> (f64, f64, f64, f64) {
    // Shepperd's method for numerical stability
    let trace = mat[(0, 0)] + mat[(1, 1)] + mat[(2, 2)];

    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        let w = 0.25 * s;
        let x = (mat[(2, 1)] - mat[(1, 2)]) / s;
        let y = (mat[(0, 2)] - mat[(2, 0)]) / s;
        let z = (mat[(1, 0)] - mat[(0, 1)]) / s;
        (w, x, y, z)
    } else if mat[(0, 0)] > mat[(1, 1)] && mat[(0, 0)] > mat[(2, 2)] {
        let s = (1.0 + mat[(0, 0)] - mat[(1, 1)] - mat[(2, 2)]).sqrt() * 2.0;
        let w = (mat[(2, 1)] - mat[(1, 2)]) / s;
        let x = 0.25 * s;
        let y = (mat[(0, 1)] + mat[(1, 0)]) / s;
        let z = (mat[(0, 2)] + mat[(2, 0)]) / s;
        (w, x, y, z)
    } else if mat[(1, 1)] > mat[(2, 2)] {
        let s = (1.0 + mat[(1, 1)] - mat[(0, 0)] - mat[(2, 2)]).sqrt() * 2.0;
        let w = (mat[(0, 2)] - mat[(2, 0)]) / s;
        let x = (mat[(0, 1)] + mat[(1, 0)]) / s;
        let y = 0.25 * s;
        let z = (mat[(1, 2)] + mat[(2, 1)]) / s;
        (w, x, y, z)
    } else {
        let s = (1.0 + mat[(2, 2)] - mat[(0, 0)] - mat[(1, 1)]).sqrt() * 2.0;
        let w = (mat[(1, 0)] - mat[(0, 1)]) / s;
        let x = (mat[(0, 2)] + mat[(2, 0)]) / s;
        let y = (mat[(1, 2)] + mat[(2, 1)]) / s;
        let z = 0.25 * s;
        (w, x, y, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tau_math::{SpatialInertia, SpatialTransform, Vec3};
    use tau_model::ModelBuilder;

    #[test]
    fn test_joint_state_sensor() {
        let model = ModelBuilder::new()
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build();

        let mut state = model.default_state();
        state.q[0] = 0.5;
        state.v[0] = 1.0;

        let sensor = Sensor::JointState { joint_idx: 0 };
        let output = sensor.read(&model, &state, 0);

        assert_eq!(output.data.len(), 2);
        assert_eq!(output.data[0], 0.5);
        assert_eq!(output.data[1], 1.0);
    }

    #[test]
    fn test_body_angular_vel_sensor() {
        let model = ModelBuilder::new()
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build();

        let mut state = model.default_state();
        state.v[0] = 2.0;

        let sensor = Sensor::BodyAngularVel { body_idx: 0 };
        let output = sensor.read(&model, &state, 0);

        assert_eq!(output.data.len(), 3);
        // Angular velocity should be along Z axis (joint axis) with magnitude 2.0
        assert!((output.data[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_imu_sensor() {
        let model = ModelBuilder::new()
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build();

        let state = model.default_state();
        let sensor = Sensor::Imu { body_idx: 0 };
        let output = sensor.read(&model, &state, 0);

        assert_eq!(output.data.len(), 6); // 3 accel + 3 gyro
    }

    #[test]
    fn test_frame_capture_sensor() {
        let model = ModelBuilder::new()
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build();

        let state = model.default_state();
        let sensor = Sensor::FrameCapture { body_idx: 0 };
        let output = sensor.read(&model, &state, 0);

        assert_eq!(output.data.len(), 7); // 3 pos + 4 quat
        // Quaternion should be approximately identity [1, 0, 0, 0]
        assert!((output.data[3] - 1.0).abs() < 0.1); // qw close to 1
    }
}
