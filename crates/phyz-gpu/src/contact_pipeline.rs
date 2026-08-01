//! GPU ground contact pipeline.
//!
//! Computes forward kinematics and ground plane contact penalty forces
//! as a compute pass that runs before the ABA shader.

use crate::gpu_state::GpuState;
use crate::shaders::CONTACT_GROUND_SHADER;
use bytemuck::{Pod, Zeroable};
use phyz_model::{Geometry, Model};
use std::sync::Arc;

/// Contact parameters for the ground plane shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ContactParams {
    nworld: u32,
    nbodies: u32,
    nv: u32,
    ground_height: f32,
    time_const: f32,
    damp_ratio: f32,
    friction: f32,
    _padding: f32,
}

/// Packed geometry data per body (8 f32 values).
///
/// ```text
/// [0]  geom_type (0=none, 1=sphere, 2=box, 3=capsule, 4=cylinder)
/// [1]  param0 (radius for sphere/capsule/cylinder, half_x for box)
/// [2]  param1 (length for capsule, half_y for box, height for cylinder)
/// [3]  param2 (half_z for box)
/// [4..8] reserved
/// ```
const GEOM_STRIDE: usize = 8;

/// GPU ground contact pipeline.
pub struct ContactPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    _params_buffer: wgpu::Buffer,
    _geom_buffer: wgpu::Buffer,
    nworld: usize,
}

/// Physical parameters of the penalty ground-contact model.
///
/// Stiffness is *derived* per body from its mass and [`Self::time_const`],
/// following MuJoCo's `solref`, rather than specified directly. A single
/// global stiffness cannot work across a robot: measured on the Booster K1,
/// every value at or above 5e3 diverged to NaN within 400 ticks (the feet)
/// while every value at or below 1e3 was too soft to carry the robot (the
/// torso). Per-body `k = m/tc^2` gives every body the same natural frequency
/// `1/tc`, so stability depends on `dt/tc` and nothing else.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundContactParams {
    /// Height of the ground plane along the gravity axis.
    pub ground_height: f64,
    /// Contact response time constant, seconds. Must be comfortably larger
    /// than the timestep; MuJoCo's default is `0.02`.
    pub time_const: f64,
    /// Damping ratio. `1.0` is critically damped — a foot that neither
    /// bounces nor sinks.
    pub damp_ratio: f64,
    /// Coulomb friction coefficient.
    pub friction: f64,
}

impl ContactPipeline {
    /// Create a contact pipeline for ground plane contacts.
    pub fn new(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        model: &Model,
        state: &GpuState,
        bodies_buffer: &wgpu::Buffer,
        contact: GroundContactParams,
    ) -> Result<Self, String> {
        let GroundContactParams {
            ground_height,
            time_const,
            damp_ratio,
            friction,
        } = contact;
        if time_const <= 0.0 {
            return Err(format!(
                "ground contact time_const must be positive, got {time_const}"
            ));
        }
        let nworld = state.nworld;

        // Pack contact params
        let params = ContactParams {
            nworld: nworld as u32,
            nbodies: model.nbodies() as u32,
            nv: model.nv as u32,
            ground_height: ground_height as f32,
            time_const: time_const as f32,
            damp_ratio: damp_ratio as f32,
            friction: friction as f32,
            _padding: 0.0,
        };

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("contact_params"),
            size: std::mem::size_of::<ContactParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Pack geometry data
        let geom_data = pack_geometries(model)?;
        let geom_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("geometry_buffer"),
            size: (geom_data.len() * std::mem::size_of::<f32>()).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&geom_buffer, 0, bytemuck::cast_slice(&geom_data));

        // Create shader module
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("contact_ground_shader"),
            source: wgpu::ShaderSource::Wgsl(CONTACT_GROUND_SHADER.into()),
        });

        // Bind group layout (7 bindings)
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("contact_bgl"),
            entries: &[
                bgl_uniform(0),    // contact_params
                bgl_storage_ro(1), // bodies
                bgl_storage_ro(2), // geometry
                bgl_storage_ro(3), // q
                bgl_storage_ro(4), // v
                bgl_storage_rw(5), // ext_forces (output)
            ],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("contact_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("contact_pl"),
                    bind_group_layouts: &[&bgl],
                    push_constant_ranges: &[],
                }),
            ),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("contact_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bodies_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: geom_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state.q_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: state.v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: state.ext_forces_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            pipeline,
            bind_group,
            _params_buffer: params_buffer,
            _geom_buffer: geom_buffer,
            nworld,
        })
    }

    /// Encode the contact compute pass into a command encoder.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("contact_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let workgroups = (self.nworld as u32).div_ceil(64);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}

/// Pack per-body collision geometry for the contact shader.
///
/// Fails when a body carries a shape the shader cannot represent (a mesh or
/// a plane) rather than packing it as "no collision". A dropped shape is
/// invisible at runtime — the body simply falls through the ground — and
/// that failure looks like a physics bug rather than a missing feature. The
/// caller converts explicitly with
/// [`phyz_model::Geometry::to_box_approximation`] and thereby records that
/// it accepted the approximation.
fn pack_geometries(model: &Model) -> Result<Vec<f32>, String> {
    let nb = model.nbodies();
    let mut data = vec![0.0f32; nb * GEOM_STRIDE];

    for (i, body) in model.bodies.iter().enumerate() {
        let base = i * GEOM_STRIDE;
        match &body.geometry {
            None => {
                data[base] = 0.0; // type = none
            }
            Some(Geometry::Sphere { radius }) => {
                data[base] = 1.0;
                data[base + 1] = *radius as f32;
            }
            Some(Geometry::Box { half_extents }) => {
                data[base] = 2.0;
                data[base + 1] = half_extents.x as f32;
                data[base + 2] = half_extents.y as f32;
                data[base + 3] = half_extents.z as f32;
            }
            Some(Geometry::Capsule { radius, length }) => {
                data[base] = 3.0;
                data[base + 1] = *radius as f32;
                data[base + 2] = *length as f32;
            }
            Some(Geometry::Cylinder { radius, height }) => {
                data[base] = 4.0;
                data[base + 1] = *radius as f32;
                data[base + 2] = *height as f32;
            }
            Some(other) => {
                return Err(format!(
                    "body {} ('{}') has collision geometry the GPU contact shader cannot \
                     represent ({}). It would silently never collide. Convert it first \
                     with Geometry::to_box_approximation, or leave the body without \
                     geometry if it is not meant to collide.",
                    i,
                    body.name,
                    match other {
                        Geometry::Mesh { .. } => "mesh",
                        Geometry::Plane { .. } => "plane",
                        _ => "unsupported",
                    }
                ));
            }
        }
    }

    Ok(data)
}

// Bind group layout helpers
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
