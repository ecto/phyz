//! GPU-accelerated batch simulation using wgpu compute shaders.
//!
//! Implements parallel simulation of multiple independent environments
//! using GPU compute shaders for ABA and integration.

#![warn(missing_docs)]

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

pub mod contact_pipeline;
pub mod gpu_batch_simulator;
pub mod gpu_simulator;
pub mod gpu_state;
pub mod interop;
pub mod pd_pipeline;
pub mod shaders;
pub mod sparse;
pub mod sparse_shaders;

pub use contact_pipeline::ContactPipeline;
pub use gpu_batch_simulator::GpuBatchSimulator;
pub use gpu_simulator::GpuSimulator;
pub use gpu_state::GpuState;
pub use pd_pipeline::{PdDof, PdPipeline};

pub use interop::GpuInterop;
