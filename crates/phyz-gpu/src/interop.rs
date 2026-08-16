//! Raw GPU handles for zero-copy interop with an ML runtime.
//!
//! This is phyz's half of the sim↔learning contract described in
//! `docs/design/batched-envs.md` §3. Everything here is implemented; the
//! remaining work is on the `tang` side.
//!
//! # The contract
//!
//! A rollout loop that never leaves the GPU looks like:
//!
//! 1. The physics compute pass writes observations into a `wgpu::Buffer`.
//! 2. That **same buffer** is wrapped as a `tang` GPU tensor — no copy.
//! 3. The policy's kernels read it and write an action tensor.
//! 4. The action tensor's buffer is bound directly as the simulator's `ctrl`
//!    buffer.
//!
//! One device, one queue, one submission per step, zero host round-trips.
//!
//! # What `tang` still needs
//!
//! Three additive changes, none of which phyz can make on its behalf:
//!
//! 1. **Align `wgpu` versions.** `phyz-gpu` is on `wgpu` 23, `tang-gpu` on 24.
//!    Two different `wgpu::Device` types cannot share a buffer, so nothing else
//!    matters until this is resolved. Best fixed by hoisting `wgpu` into a
//!    single shared workspace dependency.
//! 2. **`tang_gpu::GpuDevice::from_raw(Arc<Device>, Arc<Queue>)`.** Today
//!    `GpuDevice` owns its device by value and can only be built by
//!    `GpuDevice::new()`, which requests its own adapter. phyz already stores
//!    both as `Arc` and already accepts a caller-supplied pair via
//!    [`crate::GpuBatchSimulator::with_device_queue`], so this is the only
//!    missing direction.
//! 3. **`tang_gpu::GpuBuffer::from_wgpu(wgpu::Buffer, len)`.** `GpuBuffer`'s
//!    fields are private with no constructor from an existing buffer.
//!    `tang_compute::ComputeTensor::from_buffer` is the precedent to follow.
//!
//! Sharing a device is also a *correctness* fix, not only a performance one:
//! today `GpuBatchSimulator::new` and `GpuDevice::new` each request their own
//! adapter, so on a multi-GPU machine the simulator and the policy can silently
//! land on different GPUs.

use crate::GpuBatchSimulator;
use std::sync::Arc;

/// Borrowed handles to a batch simulator's device and state buffers.
///
/// Every buffer is `num_envs`-major and tightly packed, so it maps onto a
/// `[num_envs, dim]` row-major tensor with no reshape or transpose.
pub struct GpuInterop<'a> {
    /// The device the simulation runs on. An ML runtime must use *this* device,
    /// not one it created itself.
    pub device: &'a Arc<wgpu::Device>,
    /// The queue the simulation submits to.
    pub queue: &'a Arc<wgpu::Queue>,
    /// Generalized positions, `num_envs × nq` f32.
    pub q: &'a wgpu::Buffer,
    /// Generalized velocities, `num_envs × nv` f32.
    pub v: &'a wgpu::Buffer,
    /// Control inputs, `num_envs × nv` f32. Writable by the policy.
    pub ctrl: &'a wgpu::Buffer,
    /// Number of parallel environments.
    pub num_envs: usize,
    /// Position DOFs per environment.
    pub nq: usize,
    /// Velocity DOFs per environment.
    pub nv: usize,
}

impl GpuInterop<'_> {
    /// Shape of the `q` buffer as a tensor.
    pub fn q_shape(&self) -> [usize; 2] {
        [self.num_envs, self.nq]
    }

    /// Shape of the `v` and `ctrl` buffers as a tensor.
    pub fn v_shape(&self) -> [usize; 2] {
        [self.num_envs, self.nv]
    }

    /// Element count of the `q` buffer.
    pub fn q_len(&self) -> usize {
        self.num_envs * self.nq
    }

    /// Element count of the `v` and `ctrl` buffers.
    pub fn v_len(&self) -> usize {
        self.num_envs * self.nv
    }
}

impl GpuBatchSimulator {
    /// Borrow the raw device, queue and state buffers.
    ///
    /// Intended for binding simulator memory directly into another compute
    /// pipeline. The returned buffers are live simulator state: reading them
    /// after a [`GpuBatchSimulator::step`] submission without synchronising
    /// gives whatever the queue has reached.
    pub fn interop(&self) -> GpuInterop<'_> {
        GpuInterop {
            device: &self.device,
            queue: &self.queue,
            q: &self.state.q_buffer,
            v: &self.state.v_buffer,
            ctrl: &self.state.ctrl_buffer,
            num_envs: self.state.nworld,
            nq: self.state.nq,
            nv: self.state.nv,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
    use phyz_model::ModelBuilder;

    #[test]
    fn interop_reports_the_batch_shapes() {
        let model = ModelBuilder::new()
            .add_revolute_body(
                "l",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.1),
            )
            .build();

        let Ok(sim) = GpuBatchSimulator::new(model, 128) else {
            eprintln!("skipping: no GPU adapter");
            return;
        };

        let io = sim.interop();
        assert_eq!(io.num_envs, 128);
        assert_eq!(io.q_shape(), [128, 1]);
        assert_eq!(io.v_len(), 128);
        // The whole point: the buffers are big enough to be bound as tensors.
        assert!(io.q.size() >= (io.q_len() * 4) as u64);
        assert!(io.ctrl.size() >= (io.v_len() * 4) as u64);
    }
}
