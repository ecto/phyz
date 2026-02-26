//! GPU state buffer management.
//!
//! Handles allocation and synchronization of GPU buffers for batch simulation.

use bytemuck::{Pod, Zeroable};
use phyz_model::{Model, State};
use std::sync::Arc;

/// GPU-backed batch simulation state.
///
/// Stores state for `nworld` parallel environments on GPU memory.
pub struct GpuState {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub nworld: usize,
    pub nq: usize,
    pub nv: usize,

    // State buffers (nworld × ndof)
    pub q_buffer: wgpu::Buffer,
    pub v_buffer: wgpu::Buffer,
    pub ctrl_buffer: wgpu::Buffer,

    // Scratch buffers for computation
    pub qdd_buffer: wgpu::Buffer,

    // External forces buffer (nbodies × 6 per env, spatial force per body)
    pub ext_forces_buffer: wgpu::Buffer,
    pub nbodies: usize,

    // Staging buffers for CPU ↔ GPU transfer
    pub q_staging: wgpu::Buffer,
    pub v_staging: wgpu::Buffer,
}

/// GPU-friendly packed state data for a single environment.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PackedState {
    q: [f32; 16], // max 16 DOF for simplicity
    v: [f32; 16],
    ctrl: [f32; 16],
}

impl GpuState {
    /// Create GPU buffers for batch simulation.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        model: &Model,
        nworld: usize,
    ) -> Self {
        let nq = model.nq;
        let nv = model.nv;
        let nbodies = model.nbodies();

        // Create buffers with STORAGE usage for compute shaders
        let q_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q_buffer"),
            size: (nworld * nq * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let v_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v_buffer"),
            size: (nworld * nv * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let ctrl_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ctrl_buffer"),
            size: (nworld * nv * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let qdd_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qdd_buffer"),
            size: (nworld * nv * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // External forces: 6 floats per body per environment
        let ext_forces_size = (nworld * nbodies * 6 * std::mem::size_of::<f32>()).max(4) as u64;
        let ext_forces_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ext_forces_buffer"),
            size: ext_forces_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Staging buffers for readback
        let q_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q_staging"),
            size: (nworld * nq * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let v_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v_staging"),
            size: (nworld * nv * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            nworld,
            nq,
            nv,
            q_buffer,
            v_buffer,
            ctrl_buffer,
            qdd_buffer,
            ext_forces_buffer,
            nbodies,
            q_staging,
            v_staging,
        }
    }

    /// Upload states from CPU to GPU.
    pub fn upload_states(&self, states: &[State]) {
        assert_eq!(states.len(), self.nworld);

        // Pack states into flat f32 arrays
        let mut q_data = vec![0.0f32; self.nworld * self.nq];
        let mut v_data = vec![0.0f32; self.nworld * self.nv];
        let mut ctrl_data = vec![0.0f32; self.nworld * self.nv];

        for (i, state) in states.iter().enumerate() {
            for j in 0..self.nq {
                q_data[i * self.nq + j] = state.q[j] as f32;
            }
            for j in 0..self.nv {
                v_data[i * self.nv + j] = state.v[j] as f32;
                ctrl_data[i * self.nv + j] = state.ctrl[j] as f32;
            }
        }

        // Upload to GPU
        self.queue
            .write_buffer(&self.q_buffer, 0, bytemuck::cast_slice(&q_data));
        self.queue
            .write_buffer(&self.v_buffer, 0, bytemuck::cast_slice(&v_data));
        self.queue
            .write_buffer(&self.ctrl_buffer, 0, bytemuck::cast_slice(&ctrl_data));
    }

    /// Download states from GPU to CPU.
    pub async fn download_states(&self) -> Result<(Vec<f32>, Vec<f32>), String> {
        // Copy from storage buffers to staging buffers
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download_encoder"),
            });

        encoder.copy_buffer_to_buffer(
            &self.q_buffer,
            0,
            &self.q_staging,
            0,
            (self.nworld * self.nq * std::mem::size_of::<f32>()) as u64,
        );

        encoder.copy_buffer_to_buffer(
            &self.v_buffer,
            0,
            &self.v_staging,
            0,
            (self.nworld * self.nv * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map and read staging buffers
        let q_slice = self.q_staging.slice(..);
        let v_slice = self.v_staging.slice(..);

        let (q_tx, q_rx) = futures_intrusive::channel::shared::oneshot_channel();
        let (v_tx, v_rx) = futures_intrusive::channel::shared::oneshot_channel();

        q_slice.map_async(wgpu::MapMode::Read, move |result| {
            q_tx.send(result).ok();
        });

        v_slice.map_async(wgpu::MapMode::Read, move |result| {
            v_tx.send(result).ok();
        });

        self.device.poll(wgpu::Maintain::Wait);

        q_rx.receive()
            .await
            .ok_or("Failed to map q buffer")?
            .map_err(|e| format!("GPU buffer mapping failed: {:?}", e))?;
        v_rx.receive()
            .await
            .ok_or("Failed to map v buffer")?
            .map_err(|e| format!("GPU buffer mapping failed: {:?}", e))?;

        let q_data = q_slice.get_mapped_range();
        let v_data = v_slice.get_mapped_range();

        let q_vec: Vec<f32> = bytemuck::cast_slice(&q_data).to_vec();
        let v_vec: Vec<f32> = bytemuck::cast_slice(&v_data).to_vec();

        drop(q_data);
        drop(v_data);
        self.q_staging.unmap();
        self.v_staging.unmap();

        Ok((q_vec, v_vec))
    }
}
