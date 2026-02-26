//! GPU-accelerated batch simulator for arbitrary articulated bodies.
//!
//! Unlike [`GpuSimulator`](crate::GpuSimulator) which only handles single-body
//! pendulums, `GpuBatchSimulator` supports arbitrary tree topologies with
//! revolute, prismatic, and fixed joints.

use crate::contact_pipeline::ContactPipeline;
use crate::gpu_state::GpuState;
use crate::shaders::{ABA_GENERAL_SHADER, INTEGRATE_SHADER};
use bytemuck::{Pod, Zeroable};
use phyz_model::{Model, State};
use std::sync::Arc;

/// Simulation parameters passed to the general ABA shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BatchSimParams {
    nworld: u32,
    nv: u32,
    dt: f32,
    nbodies: u32,
    gx: f32,
    gy: f32,
    gz: f32,
    _padding: u32,
}

/// Packed body data for GPU upload (32 f32 values per body).
///
/// Layout matches the WGSL shader's `BODY_STRIDE` of 32:
/// ```text
/// [0]  parent (bitcast i32)
/// [1]  joint_type (0=revolute, 1=prismatic, 2=fixed)
/// [2]  q_offset
/// [3]  v_offset
/// [4]  mass
/// [5..8]  com (x,y,z)
/// [8..14] inertia (xx,yy,zz,xy,xz,yz)
/// [14..23] ptj rotation (row-major 3x3)
/// [23..26] ptj translation (x,y,z)
/// [26..29] axis (x,y,z)
/// [29] damping
/// [30..32] padding
/// ```
const BODY_STRIDE: usize = 32;

/// GPU-accelerated batch simulator for general articulated bodies.
pub struct GpuBatchSimulator {
    /// The wgpu device.
    pub device: Arc<wgpu::Device>,
    /// The wgpu queue.
    pub queue: Arc<wgpu::Queue>,
    /// GPU state buffers.
    pub state: GpuState,
    /// The physics model.
    pub model: Model,

    // Compute pipelines
    aba_pipeline: wgpu::ComputePipeline,
    integrate_pipeline: wgpu::ComputePipeline,

    // Bind groups
    aba_bind_group: wgpu::BindGroup,
    integrate_bind_group: wgpu::BindGroup,

    // Optional contact pipeline
    contact_pipeline: Option<ContactPipeline>,

    // Buffers (kept alive)
    _sim_params_buffer: wgpu::Buffer,
    bodies_buffer: wgpu::Buffer,
}

impl GpuBatchSimulator {
    /// Create a new batch simulator from a phyz Model.
    ///
    /// `nworld` is the number of parallel environments to simulate.
    pub fn new(model: Model, nworld: usize) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or("Failed to find GPU adapter")?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("phyz-gpu-batch-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .map_err(|e| format!("Failed to create device: {e}"))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        Self::with_device_queue(model, nworld, device, queue)
    }

    /// Create with an existing device and queue.
    pub fn with_device_queue(
        model: Model,
        nworld: usize,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Result<Self, String> {
        let state = GpuState::new(device.clone(), queue.clone(), &model, nworld);

        // Pack simulation params
        let sim_params = BatchSimParams {
            nworld: nworld as u32,
            nv: model.nv as u32,
            dt: model.dt as f32,
            nbodies: model.nbodies() as u32,
            gx: model.gravity.x as f32,
            gy: model.gravity.y as f32,
            gz: model.gravity.z as f32,
            _padding: 0,
        };

        let sim_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_sim_params"),
            size: std::mem::size_of::<BatchSimParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&sim_params_buffer, 0, bytemuck::bytes_of(&sim_params));

        // Pack body data
        let bodies_data = pack_bodies(&model);
        let bodies_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bodies_buffer"),
            size: (bodies_data.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&bodies_buffer, 0, bytemuck::cast_slice(&bodies_data));

        // Create shader modules
        let aba_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aba_general_shader"),
            source: wgpu::ShaderSource::Wgsl(ABA_GENERAL_SHADER.into()),
        });

        let integrate_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("integrate_shader"),
            source: wgpu::ShaderSource::Wgsl(INTEGRATE_SHADER.into()),
        });

        // ABA bind group layout (6 bindings: params, bodies, q, v, ctrl, qdd)
        let aba_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aba_general_bgl"),
                entries: &[
                    bgl_uniform(0),    // params
                    bgl_storage_ro(1), // bodies
                    bgl_storage_ro(2), // q
                    bgl_storage_ro(3), // v
                    bgl_storage_ro(4), // ctrl
                    bgl_storage_rw(5), // qdd
                    bgl_storage_ro(6), // ext_forces
                ],
            });

        // Integration bind group layout (4 bindings: params, q, v, qdd)
        let integrate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("integrate_bgl"),
                entries: &[
                    bgl_uniform(0),    // params
                    bgl_storage_rw(1), // q
                    bgl_storage_rw(2), // v
                    bgl_storage_ro(3), // qdd
                ],
            });

        // Create pipelines
        let aba_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("aba_general_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("aba_general_pl"),
                    bind_group_layouts: &[&aba_bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &aba_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let integrate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("integrate_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("integrate_pl"),
                    bind_group_layouts: &[&integrate_bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &integrate_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create bind groups
        let aba_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aba_general_bg"),
            layout: &aba_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bodies_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state.q_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state.v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: state.ctrl_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: state.qdd_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: state.ext_forces_buffer.as_entire_binding(),
                },
            ],
        });

        let integrate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("integrate_bg"),
            layout: &integrate_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state.q_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state.v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state.qdd_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            state,
            model,
            aba_pipeline,
            integrate_pipeline,
            aba_bind_group,
            integrate_bind_group,
            contact_pipeline: None,
            _sim_params_buffer: sim_params_buffer,
            bodies_buffer,
        })
    }

    /// Enable ground contact pipeline with penalty forces.
    pub fn enable_ground_contact(
        &mut self,
        ground_height: f64,
        stiffness: f64,
        damping: f64,
        friction: f64,
    ) -> Result<(), String> {
        let pipeline = ContactPipeline::new(
            &self.device,
            &self.queue,
            &self.model,
            &self.state,
            &self.bodies_buffer,
            ground_height,
            stiffness,
            damping,
            friction,
        )?;
        self.contact_pipeline = Some(pipeline);
        Ok(())
    }

    /// Upload initial states to GPU.
    pub fn load_states(&self, states: &[State]) {
        self.state.upload_states(states);
    }

    /// Upload control inputs for all environments.
    pub fn set_controls(&self, controls: &[Vec<f64>]) {
        let nv = self.model.nv;
        let mut ctrl_data = vec![0.0f32; self.state.nworld * nv];
        for (i, ctrl) in controls.iter().enumerate() {
            for (j, &val) in ctrl.iter().enumerate().take(nv) {
                ctrl_data[i * nv + j] = val as f32;
            }
        }
        self.queue
            .write_buffer(&self.state.ctrl_buffer, 0, bytemuck::cast_slice(&ctrl_data));
    }

    /// Run one simulation step on GPU (contact + ABA + integration).
    pub fn step(&self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batch_step_encoder"),
            });

        // Pass 0: Contact detection (writes external forces)
        if let Some(contact) = &self.contact_pipeline {
            contact.encode(&mut encoder);
        }

        // Pass 1: ABA (compute accelerations, reads external forces)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aba_general_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.aba_pipeline);
            pass.set_bind_group(0, &self.aba_bind_group, &[]);
            let workgroups = (self.state.nworld as u32).div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 2: Integration
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("integrate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.integrate_pipeline);
            pass.set_bind_group(0, &self.integrate_bind_group, &[]);
            let total_dofs = (self.state.nworld * self.state.nv) as u32;
            let workgroups = total_dofs.div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Download states from GPU to CPU.
    pub fn readback_states(&self) -> Vec<State> {
        let (q_data, v_data) =
            pollster::block_on(self.state.download_states()).expect("Failed to download states");

        let mut states = Vec::new();
        for i in 0..self.state.nworld {
            let mut state = self.model.default_state();
            for j in 0..self.state.nq {
                state.q[j] = q_data[i * self.state.nq + j] as f64;
            }
            for j in 0..self.state.nv {
                state.v[j] = v_data[i * self.state.nv + j] as f64;
            }
            states.push(state);
        }

        states
    }
}

/// Pack model bodies into a flat f32 array for GPU upload.
fn pack_bodies(model: &Model) -> Vec<f32> {
    let nb = model.nbodies();
    let mut data = vec![0.0f32; nb * BODY_STRIDE];

    for (i, body) in model.bodies.iter().enumerate() {
        let base = i * BODY_STRIDE;
        let joint = &model.joints[body.joint_idx];

        // [0] parent
        data[base] = f32::from_bits(body.parent as u32);
        // [1] joint_type
        let jtype: u32 = match joint.joint_type {
            phyz_model::JointType::Revolute | phyz_model::JointType::Hinge => 0,
            phyz_model::JointType::Prismatic | phyz_model::JointType::Slide => 1,
            phyz_model::JointType::Fixed => 2,
            _ => 2, // treat unsupported as fixed for now
        };
        data[base + 1] = f32::from_bits(jtype);
        // [2] q_offset
        data[base + 2] = f32::from_bits(model.q_offsets[body.joint_idx] as u32);
        // [3] v_offset
        data[base + 3] = f32::from_bits(model.v_offsets[body.joint_idx] as u32);
        // [4] mass
        data[base + 4] = body.inertia.mass as f32;
        // [5..8] com
        data[base + 5] = body.inertia.com.x as f32;
        data[base + 6] = body.inertia.com.y as f32;
        data[base + 7] = body.inertia.com.z as f32;
        // [8..14] inertia (xx,yy,zz,xy,xz,yz)
        data[base + 8] = body.inertia.inertia[(0, 0)] as f32;
        data[base + 9] = body.inertia.inertia[(1, 1)] as f32;
        data[base + 10] = body.inertia.inertia[(2, 2)] as f32;
        data[base + 11] = body.inertia.inertia[(0, 1)] as f32;
        data[base + 12] = body.inertia.inertia[(0, 2)] as f32;
        data[base + 13] = body.inertia.inertia[(1, 2)] as f32;
        // [14..23] ptj rotation (row-major)
        let r = &joint.parent_to_joint.rot;
        for row in 0..3 {
            for col in 0..3 {
                data[base + 14 + row * 3 + col] = r[(row, col)] as f32;
            }
        }
        // [23..26] ptj translation
        data[base + 23] = joint.parent_to_joint.pos.x as f32;
        data[base + 24] = joint.parent_to_joint.pos.y as f32;
        data[base + 25] = joint.parent_to_joint.pos.z as f32;
        // [26..29] axis
        data[base + 26] = joint.axis.x as f32;
        data[base + 27] = joint.axis.y as f32;
        data[base + 28] = joint.axis.z as f32;
        // [29] damping
        data[base + 29] = joint.damping as f32;
        // [30..32] padding
    }

    data
}

// Helper functions for bind group layout entries
fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
    use phyz_model::ModelBuilder;
    use phyz_rigid::aba;

    fn make_single_pendulum() -> Model {
        let length = 1.0;
        let mass = 1.0;
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "link1",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(
                    mass,
                    Vec3::new(0.0, -length / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(
                        mass * length * length / 12.0,
                        0.0,
                        mass * length * length / 12.0,
                    )),
                ),
            )
            .build()
    }

    fn make_double_pendulum() -> Model {
        let length = 1.0;
        let mass = 1.0;
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.001)
            .add_revolute_body(
                "link1",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(
                    mass,
                    Vec3::new(0.0, -length / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(
                        mass * length * length / 12.0,
                        0.0,
                        mass * length * length / 12.0,
                    )),
                ),
            )
            .add_revolute_body(
                "link2",
                0,
                SpatialTransform::from_translation(Vec3::new(0.0, -length, 0.0)),
                SpatialInertia::new(
                    mass,
                    Vec3::new(0.0, -length / 2.0, 0.0),
                    Mat3::from_diagonal(&Vec3::new(
                        mass * length * length / 12.0,
                        0.0,
                        mass * length * length / 12.0,
                    )),
                ),
            )
            .build()
    }

    fn make_6dof_arm() -> Model {
        let length = 0.3;
        let mass = 0.5;
        let inertia = SpatialInertia::new(
            mass,
            Vec3::new(0.0, 0.0, -length / 2.0),
            Mat3::from_diagonal(&Vec3::new(
                mass * length * length / 12.0,
                mass * length * length / 12.0,
                0.001,
            )),
        );

        ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
            .dt(0.001)
            // Joint 1: revolute Z
            .add_revolute_body("j1", -1, SpatialTransform::identity(), inertia.clone())
            // Joint 2: revolute Z, offset along Z
            .add_revolute_body(
                "j2",
                0,
                SpatialTransform::from_translation(Vec3::new(0.0, 0.0, -length)),
                inertia.clone(),
            )
            // Joint 3: revolute Z
            .add_revolute_body(
                "j3",
                1,
                SpatialTransform::from_translation(Vec3::new(0.0, 0.0, -length)),
                inertia.clone(),
            )
            // Joint 4: revolute Z
            .add_revolute_body(
                "j4",
                2,
                SpatialTransform::from_translation(Vec3::new(0.0, 0.0, -length)),
                inertia.clone(),
            )
            // Joint 5: revolute Z
            .add_revolute_body(
                "j5",
                3,
                SpatialTransform::from_translation(Vec3::new(0.0, 0.0, -length)),
                inertia.clone(),
            )
            // Joint 6: revolute Z
            .add_revolute_body(
                "j6",
                4,
                SpatialTransform::from_translation(Vec3::new(0.0, 0.0, -length)),
                inertia,
            )
            .build()
    }

    /// Run GPU ABA on a single environment and compare with CPU ABA.
    fn compare_gpu_vs_cpu(model: &Model, state: &State, tolerance: f64) {
        // CPU reference
        let cpu_qdd = aba(model, state);

        // GPU
        let sim = match GpuBatchSimulator::new(model.clone(), 1) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Skipping GPU test (no adapter): {e}");
                return;
            }
        };

        sim.load_states(&[state.clone()]);
        sim.step();
        let gpu_states = sim.readback_states();

        // After step, GPU has integrated (q, v updated). But we want to compare
        // the qdd before integration. Instead, we'll compare the integrated result.
        // For a single step with semi-implicit Euler:
        //   v_new = v_old + dt * qdd
        //   q_new = q_old + dt * v_new
        let dt = model.dt;
        for j in 0..model.nv {
            let expected_v = state.v[j] + dt * cpu_qdd[j];
            let expected_q = state.q[j] + dt * expected_v;
            let gpu_v = gpu_states[0].v[j];
            let gpu_q = gpu_states[0].q[j];

            assert!(
                (gpu_v - expected_v).abs() < tolerance,
                "DOF {j}: gpu_v={gpu_v:.6}, expected_v={expected_v:.6}, diff={:.2e}",
                (gpu_v - expected_v).abs()
            );
            assert!(
                (gpu_q - expected_q).abs() < tolerance,
                "DOF {j}: gpu_q={gpu_q:.6}, expected_q={expected_q:.6}, diff={:.2e}",
                (gpu_q - expected_q).abs()
            );
        }
    }

    #[test]
    fn test_pendulum_gpu_vs_cpu() {
        let model = make_single_pendulum();
        let mut state = model.default_state();
        state.q[0] = std::f64::consts::FRAC_PI_4;
        compare_gpu_vs_cpu(&model, &state, 1e-3);
    }

    #[test]
    fn test_double_pendulum_gpu_vs_cpu() {
        let model = make_double_pendulum();
        let mut state = model.default_state();
        state.q[0] = 0.3;
        state.q[1] = 0.2;
        state.v[0] = 0.1;
        state.v[1] = -0.1;
        // f32 precision loss accumulates through 2-body tree
        compare_gpu_vs_cpu(&model, &state, 5e-3);
    }

    #[test]
    fn test_6dof_arm_gpu_vs_cpu() {
        let model = make_6dof_arm();
        let mut state = model.default_state();
        state.q[0] = 0.1;
        state.q[1] = -0.2;
        state.q[2] = 0.3;
        state.q[3] = -0.1;
        state.q[4] = 0.2;
        state.q[5] = -0.3;
        compare_gpu_vs_cpu(&model, &state, 1e-2); // f32 tolerance for 6-DOF chain
    }

    #[test]
    fn test_batch_simulation() {
        let model = make_double_pendulum();
        let sim = match GpuBatchSimulator::new(model.clone(), 4) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Skipping GPU test (no adapter): {e}");
                return;
            }
        };

        let mut states = Vec::new();
        for i in 0..4 {
            let mut s = model.default_state();
            s.q[0] = (i as f64) * 0.25;
            s.q[1] = (i as f64) * 0.1;
            states.push(s);
        }

        sim.load_states(&states);
        sim.step();
        let results = sim.readback_states();

        assert_eq!(results.len(), 4);

        // Each env should have different states since they started differently
        let q0_0 = results[0].q[0];
        let q1_0 = results[1].q[0];
        assert!(
            (q0_0 - q1_0).abs() > 1e-6,
            "Environments should diverge: q0={q0_0}, q1={q1_0}"
        );
    }

    #[test]
    fn test_pack_bodies() {
        let model = make_double_pendulum();
        let data = pack_bodies(&model);
        assert_eq!(data.len(), 2 * BODY_STRIDE);

        // First body: parent = -1
        let parent0 = f32::to_bits(data[0]) as i32;
        assert_eq!(parent0, -1);

        // Second body: parent = 0
        let parent1 = f32::to_bits(data[BODY_STRIDE]) as i32;
        assert_eq!(parent1, 0);
    }
}
