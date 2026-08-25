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

/// Contact sweeps per step in impulse mode; mirrors
/// `phyz_contact::GPU_SWEEPS`.
pub const DEFAULT_CONTACT_SWEEPS: usize = 16;

/// [`DEFAULT_CONTACT_SWEEPS`], overridable at run time by
/// `PHYZ_CONTACT_SWEEPS`.
///
/// The sweep count is the single largest term in an impulse step's cost — it
/// is dead linear, because a sweep is `[contact, ABA]` and 16 of them is 17
/// ABA solves. Whether a shorter sweep loop is *physically* the same run is a
/// question about a particular model and a particular task, which only the
/// caller's own ruler can answer; this makes it answerable without a rebuild.
/// Out-of-range or unparseable values are ignored.
pub fn default_contact_sweeps() -> usize {
    std::env::var("PHYZ_CONTACT_SWEEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n >= 1 && *n <= 256)
        .unwrap_or(DEFAULT_CONTACT_SWEEPS)
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

    /// Contact sweeps per step in impulse mode.
    ///
    /// Each sweep costs one extra ABA dispatch, because the interleaving is
    /// what applies the Delassus operator. Defaults to
    /// `phyz_contact::GPU_SWEEPS`, duplicated here rather than depended on —
    /// `phyz-gpu` does not depend on `phyz-contact` — and asserted equal in
    /// `tests/contact_impulse_parity.rs`.
    pub contact_sweeps: usize,

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
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()
        .ok_or("Failed to find GPU adapter")?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            trace: wgpu::Trace::Off,
            label: Some("phyz-gpu-batch-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: Default::default(),
        }))
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
            contact_sweeps: default_contact_sweeps(),
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
                ..Default::default()
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
        self.enable_ground_contact_with_plane(ground_height, friction, gains, &[])
    }

    /// [`Self::enable_ground_contact_per_body`] plus an optional
    /// body-attached contact plane (see [`crate::contact_pipeline::BodyPlane`])
    /// — the deck a rider stands on. One compute pass handles both: the same
    /// FK feeds the ground test and the plane test, and forces land on both
    /// the touching body and the plane's body.
    pub fn enable_ground_contact_with_plane(
        &mut self,
        ground_height: f64,
        friction: f64,
        gains: &[crate::contact_pipeline::BodyContactGains],
        planes: &[crate::contact_pipeline::BodyPlane],
    ) -> Result<usize, String> {
        self.enable_contact_terrain(ground_height, friction, gains, planes, None)
    }

    /// [`Self::enable_ground_contact_with_plane`] over heightfield terrain
    /// instead of the flat plane.
    ///
    /// `ground_height` is ignored while a heightfield is loaded — the
    /// surface comes from the field, and a
    /// [`phyz_model::Heightfield::flat`] field reproduces the plane exactly.
    /// Swap terrain between training iterations with
    /// [`Self::set_heightfield`]; the node buffer is sized to this first
    /// field, so start with the largest grid the run will use.
    pub fn enable_contact_terrain(
        &mut self,
        ground_height: f64,
        friction: f64,
        gains: &[crate::contact_pipeline::BodyContactGains],
        planes: &[crate::contact_pipeline::BodyPlane],
        heightfield: Option<&phyz_model::Heightfield>,
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
                ..Default::default()
            },
            Some(gains),
            planes,
            heightfield,
        )?;
        let collidable = pipeline.collidable_bodies();
        self.contact_pipeline = Some(pipeline);
        Ok(collidable)
    }

    /// Enable contact as a **velocity-level convex impulse solve** — the same
    /// contact problem `phyz_contact::solve_contacts` states, rather than the
    /// penalty approximation of it.
    ///
    /// This is the entry point that makes GPU results transferable to the CPU
    /// path. The penalty model differs from the CPU's in ways that are not
    /// tuning: it sinks by `mg/k` at rest, its friction is a slip-speed
    /// regularization that creeps instead of sticking, and its dampers are
    /// explicit and need caps to stay stable. The impulse solve has none of
    /// those properties because it is not an approximation of the CPU model —
    /// it is the same model, solved with a documented sweep budget.
    ///
    /// Cost: one extra ABA dispatch per sweep (see
    /// [`Self::contact_sweeps`]), because interleaving `[contact, ABA]` is how
    /// the Delassus operator gets applied without being assembled.
    ///
    /// `gains` is still required — it carries per-body geometry packing — but
    /// the stiffness and damping in it are unused in this mode.
    pub fn enable_contact_impulse(
        &mut self,
        ground_height: f64,
        friction: f64,
        gains: &[crate::contact_pipeline::BodyContactGains],
        planes: &[crate::contact_pipeline::BodyPlane],
        heightfield: Option<&phyz_model::Heightfield>,
    ) -> Result<usize, String> {
        let dt = self.model.dt;
        let pipeline = ContactPipeline::with_body_gains(
            &self.device,
            &self.queue,
            &self.model,
            &self.state,
            &self.bodies_buffer,
            crate::contact_pipeline::GroundContactParams {
                ground_height,
                friction,
                impulse_solve: true,
                // MuJoCo's SolRef default, evaluated at this model's dt so the
                // shader never re-derives it.
                solref_erp: crate::contact_pipeline::GroundContactParams::solref_erp_from(
                    0.02, 1.0, dt,
                ),
                ..Default::default()
            },
            Some(gains),
            planes,
            heightfield,
        )?;
        let collidable = pipeline.collidable_bodies();
        self.contact_pipeline = Some(pipeline);
        Ok(collidable)
    }

    /// Replace the contact terrain in place — a buffer write, no pipeline
    /// rebuild. Errors if contact is not enabled or the new field outgrows
    /// the buffer allocated by [`Self::enable_contact_terrain`].
    pub fn set_heightfield(&mut self, hf: &phyz_model::Heightfield) -> Result<(), String> {
        let pipeline = self
            .contact_pipeline
            .as_mut()
            .ok_or("contact not enabled — call enable_contact_terrain first")?;
        pipeline.set_heightfield(&self.queue, hf)
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

    /// Encode one ABA dispatch: accelerations from the current state and
    /// whatever `ext_forces` currently holds.
    fn encode_aba(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aba_general_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.aba_pipeline);
        pass.set_bind_group(0, &self.aba_bind_group, &[]);
        let workgroups = (self.state.nworld as u32).div_ceil(64);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Run one simulation step on GPU (contact + ABA + integration).
    pub fn step(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batch_step_encoder"),
            });

        // Pass -1: PD servos (writes ctrl from position targets)
        if let Some(pd) = &self.pd_pipeline {
            pd.encode(&mut encoder);
        }

        // ── Contact ──
        //
        // Penalty mode is one pass: forces in, ABA once, done.
        //
        // Impulse mode interleaves [contact, ABA] so that each contact sweep
        // reads a `qdd` that already carries the previous sweep's impulses
        // through the full articulated chain. That interleaving IS the
        // Delassus operator application — the residual `A f + b` is evaluated
        // exactly, with the true `M^-1`, without ever assembling `A`. See the
        // IMPULSE MODE block in `shaders.rs`.
        //
        // The leading ABA (before any contact pass) is what gives sweep 0 a
        // free acceleration to measure against; without it the first sweep
        // would size its impulses against a stale `qdd` from the previous
        // step and the solve would lag the state by one frame.
        let sweeps = match &self.contact_pipeline {
            Some(c) if c.impulse_solve() => self.contact_sweeps,
            _ => 0,
        };

        if sweeps > 0 {
            self.encode_aba(&mut encoder);
            // No per-sweep uniform: every sweep runs the same shader with the
            // same parameters, and it could not be otherwise here. All sweeps
            // are encoded into ONE command buffer, and `queue.write_buffer`
            // writes are ordered at submission — so a uniform rewritten between
            // encodes would take its LAST value for every dispatch, silently.
            // Per-sweep state would need separate submits or a dynamic offset;
            // nothing needs it, because the impulses in `contact_state` carry
            // all the state a sweep depends on.
            for _ in 0..sweeps {
                if let Some(contact) = &self.contact_pipeline {
                    contact.encode(&mut encoder);
                }
                self.encode_aba(&mut encoder);
            }
        } else if let Some(contact) = &self.contact_pipeline {
            contact.encode(&mut encoder);
        }

        // Pass 1: ABA (compute accelerations, reads external forces)
        if sweeps == 0 {
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
        let mut sim = match GpuBatchSimulator::new(model.clone(), 1) {
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
        let mut sim = match GpuBatchSimulator::new(model.clone(), 4) {
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
