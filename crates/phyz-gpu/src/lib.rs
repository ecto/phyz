//! GPU-accelerated batch simulation using wgpu compute shaders.
//!
//! Implements parallel simulation of multiple independent environments
//! using GPU compute shaders for ABA and integration.

pub mod contact_pipeline;
pub mod gpu_batch_simulator;
pub mod gpu_simulator;
pub mod gpu_state;
pub mod shaders;
pub mod sparse;
pub mod sparse_shaders;

pub use contact_pipeline::ContactPipeline;
pub use gpu_batch_simulator::GpuBatchSimulator;
pub use gpu_simulator::GpuSimulator;
pub use gpu_state::GpuState;
