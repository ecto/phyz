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
#[cfg(any(feature = "cuda", feature = "cuda-host"))]
pub mod cuda;
pub mod gpu_batch_simulator;
pub mod gpu_simulator;
pub mod gpu_state;
pub mod interop;
pub mod layout;
pub mod pd_pipeline;
pub mod policy_pipeline;
pub mod shaders;
pub mod sparse;
pub mod sparse_shaders;

pub use contact_pipeline::{
    BodyContactGains, BodyContactState, BodyPlane, ContactPipeline, GroundContactParams,
};
pub use gpu_batch_simulator::{DEFAULT_CONTACT_SWEEPS, GpuBatchSimulator};
pub use gpu_simulator::GpuSimulator;
pub use gpu_state::GpuState;
pub use pd_pipeline::{PdDof, PdPipeline};
pub use policy_pipeline::{ObsOp, PolicySpec};

pub use interop::GpuInterop;

#[cfg(feature = "cuda")]
pub use cuda::CudaBatchSimulator;
#[cfg(feature = "cuda-host")]
pub use cuda::HostBatchSimulator;

#[cfg(any(feature = "cuda", feature = "cuda-host"))]
pub use cuda::train::{
    AdamCfg, KlMode, NetDims, PpoUpdateCfg, SampleBatch, TrainBackend, TrainPipeline, UpdateStats,
};
