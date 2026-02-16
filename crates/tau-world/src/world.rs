//! World container that combines model, state, sensors, and tendons.

use crate::{Sensor, SensorOutput, Tendon};
use tau_model::{Model, State};

/// A complete simulation world with sensors and actuators.
pub struct World {
    /// The kinematic model.
    pub model: Model,
    /// Current simulation state.
    pub state: State,
    /// Sensors attached to the world.
    pub sensors: Vec<Sensor>,
    /// History of sensor readings (one vec per timestep).
    pub sensor_history: Vec<Vec<SensorOutput>>,
    /// Tendons (cable actuators) in the world.
    pub tendons: Vec<Tendon>,
}

impl World {
    /// Create a new world from a model.
    pub fn new(model: Model) -> Self {
        let state = model.default_state();
        Self {
            model,
            state,
            sensors: Vec::new(),
            sensor_history: Vec::new(),
            tendons: Vec::new(),
        }
    }

    /// Create a new world with initial state.
    pub fn with_state(model: Model, state: State) -> Self {
        Self {
            model,
            state,
            sensors: Vec::new(),
            sensor_history: Vec::new(),
            tendons: Vec::new(),
        }
    }

    /// Add a sensor to the world.
    pub fn add_sensor(&mut self, sensor: Sensor) {
        self.sensors.push(sensor);
    }

    /// Add a tendon to the world.
    pub fn add_tendon(&mut self, tendon: Tendon) {
        if !tendon.is_valid(&self.model) {
            panic!("Invalid tendon: path contains out-of-bounds body indices");
        }
        self.tendons.push(tendon);
    }

    /// Step the simulation forward using a custom step function.
    ///
    /// The step_fn should advance the state by one timestep.
    /// Records sensor outputs after the step.
    pub fn step<F>(&mut self, step_fn: F)
    where
        F: FnOnce(&Model, &mut State),
    {
        // Apply tendon forces as external forces
        self.apply_tendon_forces();

        // Step simulation
        step_fn(&self.model, &mut self.state);

        // Record sensor outputs
        let readings = self.read_sensors();
        self.sensor_history.push(readings);
    }

    /// Read all sensors at the current state.
    fn read_sensors(&self) -> Vec<SensorOutput> {
        self.sensors
            .iter()
            .enumerate()
            .map(|(id, sensor)| sensor.read(&self.model, &self.state, id))
            .collect()
    }

    /// Apply tendon forces to state as external forces.
    ///
    /// Modifies state.ctrl to include tendon contributions.
    fn apply_tendon_forces(&mut self) {
        for tendon in &self.tendons {
            let forces = tendon.compute_forces(&self.state);

            // Convert spatial forces to joint torques
            // This is a simplified approach - proper implementation would use
            // Jacobian transpose: tau = J^T * f
            for (body_idx, _force) in forces {
                if body_idx < self.model.nbodies() {
                    let joint_idx = self.model.bodies[body_idx].joint_idx;
                    let v_off = self.model.v_offsets[joint_idx];

                    // For now, just add a small contribution
                    // Proper implementation needs geometric Jacobian
                    if v_off < self.state.ctrl.len() {
                        // Placeholder: proportional to stretch
                        let current_len = tendon.current_length(&self.state);
                        let stretch = current_len - tendon.rest_length;
                        let torque = -tendon.stiffness * stretch * 0.1; // scaled contribution
                        self.state.ctrl[v_off] += torque;
                    }
                }
            }
        }
    }

    /// Get the most recent sensor readings.
    pub fn latest_sensor_readings(&self) -> Option<&Vec<SensorOutput>> {
        self.sensor_history.last()
    }

    /// Clear sensor history.
    pub fn clear_history(&mut self) {
        self.sensor_history.clear();
    }

    /// Get number of timesteps recorded.
    pub fn history_len(&self) -> usize {
        self.sensor_history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tau_math::{SpatialInertia, SpatialTransform, Vec3};
    use tau_model::ModelBuilder;

    #[test]
    fn test_world_creation() {
        let model = ModelBuilder::new()
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build();

        let world = World::new(model);
        assert_eq!(world.sensors.len(), 0);
        assert_eq!(world.tendons.len(), 0);
        assert_eq!(world.history_len(), 0);
    }

    #[test]
    fn test_world_with_sensor() {
        let model = ModelBuilder::new()
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build();

        let mut world = World::new(model);
        world.add_sensor(Sensor::JointState { joint_idx: 0 });

        assert_eq!(world.sensors.len(), 1);
    }

    #[test]
    fn test_world_step_with_sensors() {
        let model = ModelBuilder::new()
            .dt(0.01)
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build();

        let mut world = World::new(model);
        world.add_sensor(Sensor::JointState { joint_idx: 0 });

        // Step a few times with a dummy step function
        for _ in 0..5 {
            world.step(|_model, state| {
                state.time += 0.01;
            });
        }

        assert_eq!(world.history_len(), 5);
        assert!(world.latest_sensor_readings().is_some());
    }

    #[test]
    fn test_world_with_tendon() {
        let model = ModelBuilder::new()
            .add_revolute_body(
                "link1",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .add_revolute_body(
                "link2",
                0,
                SpatialTransform::translation(Vec3::new(0.0, 0.0, -1.0)),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build();

        let mut world = World::new(model);
        let tendon = Tendon::new(vec![0, 1], 100.0, 1.0, 10.0);
        world.add_tendon(tendon);

        assert_eq!(world.tendons.len(), 1);
    }

    #[test]
    #[should_panic(expected = "Invalid tendon")]
    fn test_world_invalid_tendon() {
        let model = ModelBuilder::new()
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build();

        let mut world = World::new(model);
        // Try to add tendon with invalid body index
        let tendon = Tendon::new(vec![0, 99], 100.0, 1.0, 10.0);
        world.add_tendon(tendon);
    }
}
