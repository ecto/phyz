//! The benchmark suites.

pub mod energy;
pub mod gpu_batch;
pub mod gradient;
pub mod rapier;
pub mod single_sim;

/// Scenes measured by the single-sim and comparison suites, in report order.
pub fn standard_scenes() -> Vec<crate::Scene> {
    use crate::Scene::*;
    vec![Pendulum, DoublePendulum, Ant, BoxStack(8)]
}
