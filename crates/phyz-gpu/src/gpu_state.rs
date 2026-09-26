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
    /// The wgpu device the buffers were allocated on.
    pub device: Arc<wgpu::Device>,
    /// The queue uploads and readbacks are submitted to.
    pub queue: Arc<wgpu::Queue>,
    /// Number of independent environments held in these buffers.
    pub nworld: usize,
    /// Generalised position count per world.
    pub nq: usize,
    /// Generalised velocity count per world.
    pub nv: usize,

    // State buffers (nworld × ndof)
    /// Joint positions, `nworld * nq` f32.
    pub q_buffer: wgpu::Buffer,
    /// Joint velocities, `nworld * nv` f32.
    pub v_buffer: wgpu::Buffer,
    /// Control inputs, `nworld * nv` f32.
    pub ctrl_buffer: wgpu::Buffer,

    // Scratch buffers for computation
    /// Joint accelerations written by the ABA pass.
    pub qdd_buffer: wgpu::Buffer,

    // External forces buffer (nbodies × 6 per env, spatial force per body)
    /// Per-body external wrenches written by the contact pass.
    pub ext_forces_buffer: wgpu::Buffer,
    /// Per-body contact state written by the contact pass,
    /// `nworld * nbodies * CONTACT_STATE_STRIDE` f32
    /// (see [`crate::layout::CONTACT_STATE_STRIDE`] for the layout).
    pub contact_state_buffer: wgpu::Buffer,
    /// Body count per world.
    pub nbodies: usize,

    // Staging buffers for CPU ↔ GPU transfer
    /// Host-visible staging buffer for reading `q` back.
    pub q_staging: wgpu::Buffer,
    /// Host-visible staging buffer for reading `v` back.
    pub v_staging: wgpu::Buffer,
    /// Host-visible staging buffer for reading contact state back.
    pub contact_staging: wgpu::Buffer,
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

        // Contact state: CONTACT_STATE_STRIDE floats per body per environment.
        let contact_state_size =
            (nworld * nbodies * crate::layout::CONTACT_STATE_STRIDE * std::mem::size_of::<f32>())
                .max(4) as u64;
        let contact_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("contact_state_buffer"),
            size: contact_state_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
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

        let contact_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("contact_staging"),
            size: contact_state_size,
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
            contact_state_buffer,
            nbodies,
            q_staging,
            v_staging,
            contact_staging,
        }
    }

    /// Upload states from CPU to GPU.
    pub fn upload_states(&self, states: &[State]) {
        assert_eq!(states.len(), self.nworld);

        let (q_data, v_data, ctrl_data) = crate::layout::pack_states(states, self.nq, self.nv);

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

        self.device.poll(wgpu::PollType::Wait).expect("device poll");

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

    /// Download the per-body contact state written by the contact pass.
    ///
    /// Returns `nworld * nbodies * CONTACT_STATE_STRIDE` f32 in body-major
    /// order within each world (see
    /// [`crate::layout::CONTACT_STATE_STRIDE`] for the layout).
    pub async fn download_contact_states(&self) -> Result<Vec<f32>, String> {
        let size = (self.nworld
            * self.nbodies
            * crate::layout::CONTACT_STATE_STRIDE
            * std::mem::size_of::<f32>())
        .max(4) as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("contact_download_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.contact_state_buffer,
            0,
            &self.contact_staging,
            0,
            size,
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = self.contact_staging.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });

        self.device.poll(wgpu::PollType::Wait).expect("device poll");

        rx.receive()
            .await
            .ok_or("Failed to map contact buffer")?
            .map_err(|e| format!("GPU buffer mapping failed: {:?}", e))?;

        let data = slice.get_mapped_range();
        let vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.contact_staging.unmap();

        Ok(vec)
    }
}
