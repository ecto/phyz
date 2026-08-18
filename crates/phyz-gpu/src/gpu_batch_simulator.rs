//! GPU-accelerated batch simulator for arbitrary articulated bodies.
//!
//! Unlike [`GpuSimulator`](crate::GpuSimulator) which only handles single-body
//! pendulums, `GpuBatchSimulator` supports arbitrary tree topologies with
//! revolute, prismatic, and fixed joints.

use crate::contact_pipeline::ContactPipeline;
use crate::gpu_state::GpuState;
use crate::layout::{pack_bodies, pack_rows, unpack_contacts, unpack_states};
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

    // Optional PD position-servo pipeline
    pd_pipeline: Option<crate::pd_pipeline::PdPipeline>,

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
                    bgl_storage_ro(4), // bodies (joint layout)
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bodies_buffer.as_entire_binding(),
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
            pd_pipeline: None,
            _sim_params_buffer: sim_params_buffer,
            bodies_buffer,
        })
    }

    /// Enable ground contact pipeline with penalty forces.
    ///
    /// Every body uses the same `stiffness`/`damping`. Returns the number of
    /// bodies whose geometry the contact pass can collide, and errors when
    /// that count is zero (a contact pass that can see nothing is a model
    /// bug, not a valid configuration).
    ///
    /// **`friction` is accepted and ignored.** The ground contact shader
    /// produces a normal force only, so this path is frictionless and per-body
    /// [`phyz_model::Body::material`] does not reach it. See
    /// [`crate::contact_pipeline::GroundContactParams`]. Use the CPU
    /// `Simulator::step_with_contacts` when friction matters.
    ///
    /// One global stiffness has to be stiff enough to hold the heaviest body
    /// up and soft enough for the lightest to integrate stably — for models
    /// with a wide mass spread no value satisfies both (see
    /// [`crate::contact_pipeline::GroundContactParams::check_stability`]).
    /// Use [`Self::enable_ground_contact_per_body`] for those.
    pub fn enable_ground_contact(
        &mut self,
        ground_height: f64,
        stiffness: f64,
        damping: f64,
        friction: f64,
    ) -> Result<usize, String> {
        let pipeline = ContactPipeline::new(
            &self.device,
            &self.queue,
            &self.model,
            &self.state,
            &self.bodies_buffer,
            crate::contact_pipeline::GroundContactParams {
                ground_height,
                stiffness,
                damping,
                friction,
            },
        )?;
        let collidable = pipeline.collidable_bodies();
        self.contact_pipeline = Some(pipeline);
        Ok(collidable)
    }

    /// Enable ground contact with per-body penalty gains.
    ///
    /// **`friction` is accepted and ignored**, as in
    /// [`Self::enable_ground_contact`].
    ///
    /// `gains` holds one entry per body, in body order — most easily built
    /// with [`crate::BodyContactGains::uniform_frequency`], which gives every
    /// body the same contact frequency so a 1-gram link and a 5-kg torso are
    /// both integrable at the same `dt`. Returns the collidable-body count,
    /// erroring when it is zero.
    pub fn enable_ground_contact_per_body(
        &mut self,
        ground_height: f64,
        friction: f64,
        gains: &[crate::contact_pipeline::BodyContactGains],
    ) -> Result<usize, String> {
        let pipeline = ContactPipeline::with_body_gains(
            &self.device,
            &self.queue,
            &self.model,
            &self.state,
            &self.bodies_buffer,
            crate::contact_pipeline::GroundContactParams {
                ground_height,
                stiffness: 0.0,
                damping: 0.0,
                friction,
            },
            Some(gains),
        )?;
        let collidable = pipeline.collidable_bodies();
        self.contact_pipeline = Some(pipeline);
        Ok(collidable)
    }

    /// Enable PD position servos on the given DOFs.
    ///
    /// Each [`crate::pd_pipeline::PdDof`] names one single-DOF joint's slot in
    /// `q`/`v` plus its gains; the pass writes `clamp(kp*(target-q) - kd*v)`
    /// into `ctrl` before every ABA pass, so [`Self::set_position_targets`]
    /// replaces [`Self::set_controls`] as the action interface. DOFs not
    /// listed keep whatever `set_controls` wrote (zero by default) — a
    /// floating base stays unactuated.
    pub fn enable_pd_control(&mut self, dofs: &[crate::pd_pipeline::PdDof]) -> Result<(), String> {
        let pipeline =
            crate::pd_pipeline::PdPipeline::new(&self.device, &self.queue, &self.state, dofs)?;
        self.pd_pipeline = Some(pipeline);
        Ok(())
    }

    /// Upload per-environment position targets for the PD servos.
    ///
    /// `targets[env]` is indexed in the order the `PdDof` list was passed to
    /// [`Self::enable_pd_control`]. Errors if PD control is not enabled.
    pub fn set_position_targets(&self, targets: &[Vec<f64>]) -> Result<(), String> {
        let pd = self
            .pd_pipeline
            .as_ref()
            .ok_or("PD control not enabled — call enable_pd_control first")?;
        pd.set_targets(&self.queue, targets);
        Ok(())
    }

    /// Upload initial states to GPU.
    pub fn load_states(&self, states: &[State]) {
        self.state.upload_states(states);
    }

    /// Upload control inputs for all environments.
    pub fn set_controls(&self, controls: &[Vec<f64>]) {
        let ctrl_data = pack_rows(controls, self.state.nworld, self.model.nv);
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

        // Pass -1: PD servos (writes ctrl from position targets)
        if let Some(pd) = &self.pd_pipeline {
            pd.encode(&mut encoder);
        }

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
            let threads = (self.state.nworld * self.model.nbodies()) as u32;
            let workgroups = threads.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Download states from GPU to CPU.
    pub fn readback_states(&self) -> Vec<State> {
        let (q_data, v_data) =
            pollster::block_on(self.state.download_states()).expect("Failed to download states");
        unpack_states(&self.model, self.state.nworld, &q_data, &v_data)
    }

    /// Download per-body ground-contact state from the GPU.
    ///
    /// Returns one [`crate::BodyContactState`] per body per environment
    /// (`result[env][body]`), reflecting the most recent [`Self::step`]:
    /// whether the body touches the ground, its penetration depth, and the
    /// contact point and penalty force in world coordinates. This is the
    /// observation channel a contact-bearing RL task needs (touch flags,
    /// normal forces) without recomputing contacts on the CPU.
    ///
    /// Errors when ground contact is not enabled.
    pub fn readback_contacts(&self) -> Result<Vec<Vec<crate::BodyContactState>>, String> {
        if self.contact_pipeline.is_none() {
            return Err(
                "ground contact not enabled — call enable_ground_contact first".to_string(),
            );
        }

        let data = pollster::block_on(self.state.download_contact_states())?;
        Ok(unpack_contacts(
            &data,
            self.state.nworld,
            self.state.nbodies,
        ))
    }
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
    use crate::layout::BODY_STRIDE;
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
            .add_revolute_body("j1", -1, SpatialTransform::identity(), inertia)
            // Joint 2: revolute Z, offset along Z
            .add_revolute_body(
                "j2",
                0,
                SpatialTransform::from_translation(Vec3::new(0.0, 0.0, -length)),
                inertia,
            )
            // Joint 3: revolute Z
            .add_revolute_body(
                "j3",
                1,
                SpatialTransform::from_translation(Vec3::new(0.0, 0.0, -length)),
                inertia,
            )
            // Joint 4: revolute Z
            .add_revolute_body(
                "j4",
                2,
                SpatialTransform::from_translation(Vec3::new(0.0, 0.0, -length)),
                inertia,
            )
            // Joint 5: revolute Z
            .add_revolute_body(
                "j5",
                3,
                SpatialTransform::from_translation(Vec3::new(0.0, 0.0, -length)),
                inertia,
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

        sim.load_states(std::slice::from_ref(state));
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
