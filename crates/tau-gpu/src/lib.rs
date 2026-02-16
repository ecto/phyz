//! GPU-accelerated batch simulation using wgpu compute shaders.
//!
//! Implements parallel simulation of multiple independent environments
//! using GPU compute shaders for ABA and integration.

pub mod gpu_simulator;
pub mod gpu_state;
pub mod shaders;

pub use gpu_simulator::GpuSimulator;
pub use gpu_state::GpuState;
