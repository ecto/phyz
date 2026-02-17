//! GPU-accelerated batch simulator.
//!
//! Orchestrates compute pipelines for parallel simulation of multiple environments.

use crate::gpu_state::GpuState;
use crate::shaders::{ABA_SIMPLE_SHADER, INTEGRATE_SHADER};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use phyz_model::{Model, State};

/// Parameters passed to compute shaders.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SimParams {
    nworld: u32,
    nv: u32,
    dt: f32,
    _padding: u32,
}

/// Body parameters for simple pendulum dynamics.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BodyParams {
    mass: f32,
    inertia: f32,
    com_y: f32,
    damping: f32,
    gravity_y: f32,
    _padding: [f32; 3],
}

/// GPU-accelerated batch simulator.
pub struct GpuSimulator {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub state: GpuState,
    pub model: Model,

    // Compute pipelines
    aba_pipeline: wgpu::ComputePipeline,
    integrate_pipeline: wgpu::ComputePipeline,

    // Bind groups
    aba_bind_group: wgpu::BindGroup,
    integrate_bind_group: wgpu::BindGroup,

    // Uniform buffers (kept for potential future updates)
    _sim_params_buffer: wgpu::Buffer,
    _body_params_buffer: wgpu::Buffer,
}

impl GpuSimulator {
    /// Create a new GPU simulator for batch simulation.
    pub fn new(model: Model, nworld: usize) -> Result<Self, String> {
        // Initialize wgpu
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
                label: Some("phyz-gpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .map_err(|e| format!("Failed to create device: {}", e))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Create GPU state buffers
        let state = GpuState::new(device.clone(), queue.clone(), &model, nworld);

        // Create uniform buffers
        let sim_params = SimParams {
            nworld: nworld as u32,
            nv: model.nv as u32,
            dt: model.dt as f32,
            _padding: 0,
        };

        let sim_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim_params"),
            size: std::mem::size_of::<SimParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&sim_params_buffer, 0, bytemuck::bytes_of(&sim_params));

        // Extract body parameters from model (assuming single-body pendulum)
        let body_params = if !model.bodies.is_empty() {
            let body = &model.bodies[0];
            let joint = &model.joints[body.joint_idx];

            // For revolute joint about Z axis: total inertia = I_zz + m*(com.x² + com.y²)
            let i_zz = body.inertia.inertia[(2, 2)];
            let com_xy_sq =
                body.inertia.com.x * body.inertia.com.x + body.inertia.com.y * body.inertia.com.y;
            let total_inertia = i_zz + body.inertia.mass * com_xy_sq;

            BodyParams {
                mass: body.inertia.mass as f32,
                inertia: total_inertia as f32,
                com_y: body.inertia.com.y as f32,
                damping: joint.damping as f32,
                gravity_y: model.gravity.y.abs() as f32,
                _padding: [0.0; 3],
            }
        } else {
            BodyParams {
                mass: 1.0,
                inertia: 1.0,
                com_y: -0.5,
                damping: 0.0,
                gravity_y: 9.81,
                _padding: [0.0; 3],
            }
        };

        let body_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("body_params"),
            size: std::mem::size_of::<BodyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&body_params_buffer, 0, bytemuck::bytes_of(&body_params));

        // Create shader modules
        let aba_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aba_shader"),
            source: wgpu::ShaderSource::Wgsl(ABA_SIMPLE_SHADER.into()),
        });

        let integrate_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("integrate_shader"),
            source: wgpu::ShaderSource::Wgsl(INTEGRATE_SHADER.into()),
        });

        // Create bind group layouts
        let aba_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aba_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let integrate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("integrate_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create pipelines
        let aba_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aba_pipeline_layout"),
            bind_group_layouts: &[&aba_bind_group_layout],
            push_constant_ranges: &[],
        });

        let aba_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("aba_pipeline"),
            layout: Some(&aba_pipeline_layout),
            module: &aba_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let integrate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("integrate_pipeline_layout"),
                bind_group_layouts: &[&integrate_bind_group_layout],
                push_constant_ranges: &[],
            });

        let integrate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("integrate_pipeline"),
            layout: Some(&integrate_pipeline_layout),
            module: &integrate_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create bind groups
        let aba_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aba_bind_group"),
            layout: &aba_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: body_params_buffer.as_entire_binding(),
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
            ],
        });

        let integrate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("integrate_bind_group"),
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
            _sim_params_buffer: sim_params_buffer,
            _body_params_buffer: body_params_buffer,
        })
    }

    /// Upload initial states to GPU.
    pub fn load_states(&self, states: &[State]) {
        self.state.upload_states(states);
    }

    /// Run one simulation step on GPU.
    pub fn step(&self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("step_encoder"),
            });

        // Compute pass 1: ABA (compute accelerations)
        {
            let mut aba_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aba_pass"),
                timestamp_writes: None,
            });
            aba_pass.set_pipeline(&self.aba_pipeline);
            aba_pass.set_bind_group(0, &self.aba_bind_group, &[]);
            let workgroups = (self.state.nworld as u32).div_ceil(256);
            aba_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Compute pass 2: Integration
        {
            let mut integrate_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("integrate_pass"),
                timestamp_writes: None,
            });
            integrate_pass.set_pipeline(&self.integrate_pipeline);
            integrate_pass.set_bind_group(0, &self.integrate_bind_group, &[]);
            let total_dofs = (self.state.nworld * self.state.nv) as u32;
            let workgroups = total_dofs.div_ceil(256);
            integrate_pass.dispatch_workgroups(workgroups, 1, 1);
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
