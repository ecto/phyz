//! Observation adapters for extracting different types of data from simulation.

use phyz_model::{Model, State};

/// Trait for adapters that extract observations from simulation state.
pub trait ObservationAdapter {
    /// Extract observation data from state.
    fn extract(&self, state: &State, model: &Model) -> Vec<f64>;

    /// Get expected output dimension.
    fn output_dim(&self) -> usize;
}

/// Observer for joint angles.
#[derive(Debug, Clone)]
pub struct JointAngleObserver {
    /// Joint index to observe.
    pub joint_idx: usize,
}

impl JointAngleObserver {
    /// Create a new joint angle observer.
    pub fn new(joint_idx: usize) -> Self {
        Self { joint_idx }
    }
}

impl ObservationAdapter for JointAngleObserver {
    fn extract(&self, state: &State, model: &Model) -> Vec<f64> {
        if self.joint_idx >= model.joints.len() {
            return vec![0.0];
        }

        let q_off = model.q_offsets[self.joint_idx];
        vec![state.q[q_off]]
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Observer for end-effector pose (position in world frame).
#[derive(Debug, Clone)]
pub struct EndEffectorPoseObserver {
    /// Body index to observe.
    pub body_idx: usize,
}

impl EndEffectorPoseObserver {
    /// Create a new end-effector pose observer.
    pub fn new(body_idx: usize) -> Self {
        Self { body_idx }
    }
}

impl ObservationAdapter for EndEffectorPoseObserver {
    fn extract(&self, state: &State, _model: &Model) -> Vec<f64> {
        if self.body_idx >= state.body_xform.len() {
            return vec![0.0, 0.0, 0.0];
        }

        let pos = state.body_xform[self.body_idx].pos;
        vec![pos.x, pos.y, pos.z]
    }

    fn output_dim(&self) -> usize {
        3
    }
}

/// Observer for body velocity.
#[derive(Debug, Clone)]
pub struct BodyVelocityObserver {
    /// Body index to observe.
    pub body_idx: usize,
}

impl BodyVelocityObserver {
    /// Create a new body velocity observer.
    pub fn new(body_idx: usize) -> Self {
        Self { body_idx }
    }
}

impl ObservationAdapter for BodyVelocityObserver {
    fn extract(&self, state: &State, model: &Model) -> Vec<f64> {
        if self.body_idx >= model.nbodies() {
            return vec![0.0, 0.0, 0.0];
        }

        let joint_idx = model.bodies[self.body_idx].joint_idx;
        let v_off = model.v_offsets[joint_idx];
        let joint = &model.joints[joint_idx];

        match joint.ndof() {
            1 => {
                // Single DOF joint
                let qd = state.v[v_off];
                let w = joint.axis * qd;
                vec![w.x, w.y, w.z]
            }
            3 | 6 => {
                // Multi-DOF joint
                vec![state.v[v_off], state.v[v_off + 1], state.v[v_off + 2]]
            }
            _ => vec![0.0, 0.0, 0.0],
        }
    }

    fn output_dim(&self) -> usize {
        3
    }
}

/// Observer for center of mass position.
#[derive(Debug, Clone)]
pub struct CenterOfMassObserver;

impl ObservationAdapter for CenterOfMassObserver {
    fn extract(&self, state: &State, model: &Model) -> Vec<f64> {
        let mut total_mass = 0.0;
        let mut com = [0.0, 0.0, 0.0];

        for (i, body) in model.bodies.iter().enumerate() {
            if i < state.body_xform.len() {
                let mass = body.inertia.mass;
                // Transform COM from body frame to world frame
                let body_xform = &state.body_xform[i];
                let com_world = body_xform.rot * body.inertia.com + body_xform.pos;

                total_mass += mass;
                com[0] += mass * com_world.x;
                com[1] += mass * com_world.y;
                com[2] += mass * com_world.z;
            }
        }

        if total_mass > 1e-10 {
            com[0] /= total_mass;
            com[1] /= total_mass;
            com[2] /= total_mass;
        }

        vec![com[0], com[1], com[2]]
    }

    fn output_dim(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{GRAVITY, SpatialInertia, SpatialTransform, Vec3};
    use phyz_model::ModelBuilder;

    fn make_pendulum() -> Model {
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, -0.5, 0.0)),
            )
            .build()
    }

    #[test]
    fn test_joint_angle_observer() {
        let model = make_pendulum();
        let mut state = model.default_state();
        state.q[0] = 0.5;

        let observer = JointAngleObserver::new(0);
        let obs = observer.extract(&state, &model);

        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0], 0.5);
        assert_eq!(observer.output_dim(), 1);
    }

    #[test]
    fn test_end_effector_pose_observer() {
        let model = make_pendulum();
        let state = model.default_state();

        let observer = EndEffectorPoseObserver::new(0);
        let obs = observer.extract(&state, &model);

        assert_eq!(obs.len(), 3);
        assert_eq!(observer.output_dim(), 3);
    }

    #[test]
    fn test_body_velocity_observer() {
        let model = make_pendulum();
        let mut state = model.default_state();
        state.v[0] = 1.0;

        let observer = BodyVelocityObserver::new(0);
        let obs = observer.extract(&state, &model);

        assert_eq!(obs.len(), 3);
        assert_eq!(observer.output_dim(), 3);
        // Angular velocity should be along Z axis
        assert!((obs[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_center_of_mass_observer() {
        let model = make_pendulum();
        let mut state = model.default_state();

        // Need to compute forward kinematics to update body transforms
        phyz_rigid::forward_kinematics(&model, &mut state);

        let observer = CenterOfMassObserver;
        let obs = observer.extract(&state, &model);

        assert_eq!(obs.len(), 3);
        assert_eq!(observer.output_dim(), 3);
        // COM should be somewhere reasonable (not at origin)
        let com_magnitude = (obs[0] * obs[0] + obs[1] * obs[1] + obs[2] * obs[2]).sqrt();
        assert!(com_magnitude > 0.1); // At least 0.1 units from origin
    }
}
