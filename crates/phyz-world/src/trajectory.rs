//! Trajectory recording and export for machine learning training data.

use crate::SensorOutput;
use phyz_model::State;
use std::collections::HashMap;

/// Records trajectories of (q, v, ctrl, time) for ML training.
pub struct TrajectoryRecorder {
    /// Recorded positions at each timestep.
    pub q_history: Vec<Vec<f64>>,
    /// Recorded velocities at each timestep.
    pub v_history: Vec<Vec<f64>>,
    /// Recorded control inputs at each timestep.
    pub ctrl_history: Vec<Vec<f64>>,
    /// Timestamps for each step.
    pub time_history: Vec<f64>,
    /// Sensor outputs at each timestep.
    pub sensor_history: Vec<Vec<SensorOutput>>,
}

impl TrajectoryRecorder {
    /// Create a new empty trajectory recorder.
    pub fn new() -> Self {
        Self {
            q_history: Vec::new(),
            v_history: Vec::new(),
            ctrl_history: Vec::new(),
            time_history: Vec::new(),
            sensor_history: Vec::new(),
        }
    }

    /// Record the current state.
    pub fn record(&mut self, state: &State) {
        self.q_history.push(state.q.as_slice().to_vec());
        self.v_history.push(state.v.as_slice().to_vec());
        self.ctrl_history.push(state.ctrl.as_slice().to_vec());
        self.time_history.push(state.time);
    }

    /// Record state with sensor outputs.
    pub fn record_with_sensors(&mut self, state: &State, sensors: Vec<SensorOutput>) {
        self.record(state);
        self.sensor_history.push(sensors);
    }

    /// Number of timesteps recorded.
    pub fn len(&self) -> usize {
        self.time_history.len()
    }

    /// Check if recorder is empty.
    pub fn is_empty(&self) -> bool {
        self.time_history.is_empty()
    }

    /// Clear all recorded data.
    pub fn clear(&mut self) {
        self.q_history.clear();
        self.v_history.clear();
        self.ctrl_history.clear();
        self.time_history.clear();
        self.sensor_history.clear();
    }

    /// Export to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let mut data = HashMap::new();
        data.insert("q", &self.q_history);
        data.insert("v", &self.v_history);
        data.insert("ctrl", &self.ctrl_history);

        // Convert time history to nested vec for consistency
        let time_nested: Vec<Vec<f64>> = self.time_history.iter().map(|&t| vec![t]).collect();
        data.insert("time", &time_nested);

        serde_json::to_string_pretty(&data)
    }

    /// Export to JSON file.
    pub fn to_json_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Convert to a dictionary-like structure suitable for numpy/ML frameworks.
    ///
    /// Returns:
    /// - "q": (nsteps, nq) flattened data
    /// - "v": (nsteps, nv) flattened data
    /// - "ctrl": (nsteps, nv) flattened data
    /// - "time": (nsteps,) flattened data
    pub fn to_flat_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut dict = HashMap::new();

        // Flatten q, v, ctrl
        let q_flat: Vec<f64> = self.q_history.iter().flatten().copied().collect();
        let v_flat: Vec<f64> = self.v_history.iter().flatten().copied().collect();
        let ctrl_flat: Vec<f64> = self.ctrl_history.iter().flatten().copied().collect();

        dict.insert("q".to_string(), q_flat);
        dict.insert("v".to_string(), v_flat);
        dict.insert("ctrl".to_string(), ctrl_flat);
        dict.insert("time".to_string(), self.time_history.clone());

        dict
    }

    /// Get trajectory statistics.
    pub fn stats(&self) -> TrajectoryStats {
        if self.is_empty() {
            return TrajectoryStats::default();
        }

        let nsteps = self.len();
        let nq = self.q_history[0].len();
        let nv = self.v_history[0].len();
        let duration = self.time_history.last().unwrap() - self.time_history[0];

        TrajectoryStats {
            nsteps,
            nq,
            nv,
            duration,
        }
    }
}

impl Default for TrajectoryRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about a recorded trajectory.
#[derive(Debug, Clone, Default)]
pub struct TrajectoryStats {
    /// Number of timesteps.
    pub nsteps: usize,
    /// Number of position DOFs.
    pub nq: usize,
    /// Number of velocity DOFs.
    pub nv: usize,
    /// Total duration (seconds).
    pub duration: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::DVec;

    fn make_test_state(t: f64) -> State {
        let mut state = State::new(2, 2, 1);
        state.q = DVec::from_vec(vec![t, t * 2.0]);
        state.v = DVec::from_vec(vec![t * 3.0, t * 4.0]);
        state.ctrl = DVec::from_vec(vec![t * 5.0, t * 6.0]);
        state.time = t;
        state
    }

    #[test]
    fn test_trajectory_recording() {
        let mut recorder = TrajectoryRecorder::new();

        for i in 0..10 {
            let t = i as f64 * 0.1;
            let state = make_test_state(t);
            recorder.record(&state);
        }

        assert_eq!(recorder.len(), 10);
        assert_eq!(recorder.q_history.len(), 10);
        assert_eq!(recorder.v_history.len(), 10);
        assert_eq!(recorder.time_history.len(), 10);
    }

    #[test]
    fn test_trajectory_stats() {
        let mut recorder = TrajectoryRecorder::new();

        for i in 0..5 {
            let state = make_test_state(i as f64);
            recorder.record(&state);
        }

        let stats = recorder.stats();
        assert_eq!(stats.nsteps, 5);
        assert_eq!(stats.nq, 2);
        assert_eq!(stats.nv, 2);
        assert_eq!(stats.duration, 4.0);
    }

    #[test]
    fn test_trajectory_to_json() {
        let mut recorder = TrajectoryRecorder::new();

        for i in 0..3 {
            let state = make_test_state(i as f64);
            recorder.record(&state);
        }

        let json = recorder.to_json();
        assert!(json.is_ok());
        let json_str = json.unwrap();
        assert!(json_str.contains("\"q\""));
        assert!(json_str.contains("\"v\""));
        assert!(json_str.contains("\"ctrl\""));
    }

    #[test]
    fn test_trajectory_to_flat_dict() {
        let mut recorder = TrajectoryRecorder::new();

        for i in 0..3 {
            let state = make_test_state(i as f64);
            recorder.record(&state);
        }

        let dict = recorder.to_flat_dict();
        assert_eq!(dict.get("q").unwrap().len(), 6); // 3 steps * 2 DOF
        assert_eq!(dict.get("v").unwrap().len(), 6);
        assert_eq!(dict.get("time").unwrap().len(), 3);
    }

    #[test]
    fn test_trajectory_clear() {
        let mut recorder = TrajectoryRecorder::new();
        recorder.record(&make_test_state(0.0));
        recorder.record(&make_test_state(1.0));

        assert_eq!(recorder.len(), 2);

        recorder.clear();
        assert_eq!(recorder.len(), 0);
        assert!(recorder.is_empty());
    }
}
