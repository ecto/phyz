//! GPU ground contact pipeline.
//!
//! Computes forward kinematics and ground plane contact penalty forces
//! as a compute pass that runs before the ABA shader.

use crate::gpu_state::GpuState;
use crate::shaders::CONTACT_GROUND_SHADER;
use bytemuck::{Pod, Zeroable};
use phyz_math::Vec3;
use phyz_model::{Geometry, Model};
use std::sync::Arc;

/// Contact parameters for the ground plane shader.
///
/// # `friction` is not implemented
///
/// The field is uploaded to the shader and the shader never reads it. The
/// ground contact pass computes a Kelvin-Voigt *normal* force and writes
/// `[0, 0, f_z]`; no tangential force is produced, so GPU ground contact is
/// frictionless regardless of what is passed here. A box given an initial
/// horizontal velocity slides forever.
///
/// This also means per-body [`phyz_model::Body::material`] does not reach the
/// GPU path — nor could it, since the quantity it would set is the one that is
/// missing. The CPU stepper (`Simulator::step_with_contacts`) honours both.
///
/// Implementing it needs the contact point's tangential velocity, which the
/// pass deliberately avoids computing (the normal damper uses a finite
/// difference of penetration precisely so it can skip a velocity FK), so it is
/// a real piece of work rather than an oversight to patch in passing.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ContactParams {
    nworld: u32,
    nbodies: u32,
    nv: u32,
    ground_height: f32,
    dt: f32,
    friction: f32,
    _pad0: f32,
    _pad1: f32,
}

/// Packed geometry data per body (16 f32 values).
///
/// ```text
/// [0]  geom_type (0=none, 1=sphere, 2=box, 3=capsule, 4=cylinder, 5=mesh)
/// [1]  param0 (radius for sphere/capsule/cylinder, half_x for box, aabb min_x for mesh)
/// [2]  param1 (length for capsule, half_y for box, height for cylinder, aabb min_y for mesh)
/// [3]  param2 (half_z for box, aabb min_z for mesh)
/// [4..7] aabb max (x,y,z) for mesh
/// [7]  reserved
/// [8]  contact stiffness (per body)
/// [9]  contact damping (per body)
/// [10..16] reserved
/// ```
const GEOM_STRIDE: usize = 16;

/// Floats per body in the contact-state buffer.
///
/// ```text
/// [0]  touching (1.0 while penetrating, else 0.0)
/// [1]  penetration depth, metres
/// [2..5] contact point, world frame (x, y, z)
/// [5..8] contact force, world frame (x, y, z)
/// ```
pub const CONTACT_STATE_STRIDE: usize = 8;

/// Explicit-integration stability factor: a penalty spring of stiffness `k`
/// on a body of mass `m` has natural frequency `w = sqrt(k/m)`, and the
/// semi-implicit Euler step is only reliably stable while `w * dt` stays
/// under roughly this value.
const OMEGA_DT_LIMIT: f64 = 0.3;

/// GPU ground contact pipeline.
pub struct ContactPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    _params_buffer: wgpu::Buffer,
    _geom_buffer: wgpu::Buffer,
    nworld: usize,
    collidable_bodies: usize,
}

/// Physical parameters of the penalty ground-contact model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundContactParams {
    /// Height of the ground plane along the gravity axis.
    pub ground_height: f64,
    /// Penalty stiffness.
    pub stiffness: f64,
    /// Penalty damping.
    pub damping: f64,
    /// Coulomb friction coefficient.
    pub friction: f64,
}

impl GroundContactParams {
    /// Check this global stiffness against the explicit-integration bound.
    ///
    /// A single stiffness has to serve every body that can touch the ground,
    /// and the stability bound `w * dt <= 0.3` (with `w = sqrt(k/m)`) is set
    /// by the *lightest* collidable body: `k <= m_min * (0.3 / dt)^2`. For a
    /// model with a wide mass spread that bound can sit below the stiffness
    /// needed to hold the total weight up, in which case no global value
    /// works and the simulation reaches NaN within a few hundred steps.
    ///
    /// Returns `Err` describing the empty window when `stiffness` exceeds the
    /// bound; the fix is per-body gains via
    /// [`crate::GpuBatchSimulator::enable_ground_contact_per_body`].
    pub fn check_stability(&self, model: &Model) -> Result<(), String> {
        let Some((name, m_min)) = lightest_collidable_body(model) else {
            return Ok(());
        };
        let k_max = m_min * (OMEGA_DT_LIMIT / model.dt).powi(2);
        if self.stiffness > k_max {
            return Err(format!(
                "ground contact stiffness {:.3e} exceeds the explicit-integration bound \
                 {:.3e} set by the lightest collidable body ('{}', {:.3e} kg) at dt = {:.1e}; \
                 the state will go NaN once that body touches the ground. \
                 Use enable_ground_contact_per_body (e.g. BodyContactGains::uniform_frequency) \
                 so each body gets a stiffness matched to its own mass.",
                self.stiffness, k_max, name, m_min, model.dt
            ));
        }
        Ok(())
    }
}

/// Per-body penalty gains for ground contact.
///
/// A penalty spring's stability under explicit integration depends on
/// `sqrt(stiffness / mass)`, so one global stiffness cannot serve both a
/// 1-gram link and a multi-kilogram torso. Per-body gains let each body get a
/// spring matched to its own mass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyContactGains {
    /// Penalty stiffness for this body, N/m.
    pub stiffness: f64,
    /// Penalty damping for this body, N·s/m.
    pub damping: f64,
}

impl BodyContactGains {
    /// Gains giving every body the same contact frequency `omega` (rad/s)
    /// and damping ratio `zeta`: `k = m * omega^2`, `d = 2 * zeta * m * omega`.
    ///
    /// With mass-proportional stiffness the stability bound `omega * dt <= 0.3`
    /// binds identically for every body, so a mixed-mass model has a single
    /// scalar to tune instead of an empty window. Pick `omega` at or below
    /// `0.3 / dt`; `zeta` around 1.0 gives critically damped, non-bouncy
    /// contact.
    pub fn uniform_frequency(model: &Model, omega: f64, zeta: f64) -> Vec<BodyContactGains> {
        model
            .bodies
            .iter()
            .map(|body| {
                let m = body.inertia.mass;
                BodyContactGains {
                    stiffness: m * omega * omega,
                    damping: 2.0 * zeta * m * omega,
                }
            })
            .collect()
    }
}

/// Per-body ground-contact state read back from the GPU.
///
/// Reported by [`crate::GpuBatchSimulator::readback_contacts`]; everything is
/// in the world frame. The contact point is the body's lowest point on its
/// collision shape, and the force is the penalty normal force applied there
/// (zero when not touching).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyContactState {
    /// True while the body's collision shape penetrates the ground plane.
    pub touching: bool,
    /// Penetration depth in metres (0 when not touching).
    pub penetration: f64,
    /// Contact point in world coordinates.
    pub point: Vec3,
    /// Contact force in world coordinates.
    pub force: Vec3,
}

impl ContactPipeline {
    /// Create a contact pipeline for ground plane contacts.
    ///
    /// Every body uses the global `contact.stiffness` / `contact.damping`.
    /// `contact.friction` is accepted and ignored — see
    /// [`GroundContactParams`].
    /// Errors when no body carries GPU-collidable geometry — a contact pass
    /// that can see nothing is a model bug, not a valid configuration.
    pub fn new(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        model: &Model,
        state: &GpuState,
        bodies_buffer: &wgpu::Buffer,
        contact: GroundContactParams,
    ) -> Result<Self, String> {
        Self::with_body_gains(device, queue, model, state, bodies_buffer, contact, None)
    }

    /// Create a contact pipeline with optional per-body gains.
    ///
    /// When `body_gains` is `Some`, it must hold one entry per body (in body
    /// order) and overrides the global stiffness/damping for every body.
    pub fn with_body_gains(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        model: &Model,
        state: &GpuState,
        bodies_buffer: &wgpu::Buffer,
        contact: GroundContactParams,
        body_gains: Option<&[BodyContactGains]>,
    ) -> Result<Self, String> {
        let nworld = state.nworld;

        if let Some(gains) = body_gains
            && gains.len() != model.nbodies()
        {
            return Err(format!(
                "body_gains has {} entries but the model has {} bodies",
                gains.len(),
                model.nbodies()
            ));
        }

        // Pack contact params
        let params = ContactParams {
            nworld: nworld as u32,
            nbodies: model.nbodies() as u32,
            nv: model.nv as u32,
            ground_height: contact.ground_height as f32,
            dt: model.dt as f32,
            friction: contact.friction as f32,
            _pad0: 0.0,
            _pad1: 0.0,
        };

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("contact_params"),
            size: std::mem::size_of::<ContactParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Pack geometry data
        let (geom_data, collidable_bodies) = pack_geometries(model, &contact, body_gains);
        if collidable_bodies == 0 {
            let skipped: Vec<&str> = model
                .bodies
                .iter()
                .filter(|b| gpu_geom_type(b.geometry.as_ref()) == 0)
                .map(|b| b.name.as_str())
                .collect();
            return Err(format!(
                "ground contact enabled but no body has GPU-collidable geometry \
                 (supported: sphere, box, capsule, cylinder, mesh); \
                 bodies without a supported shape: [{}]",
                skipped.join(", ")
            ));
        }
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
                bgl_storage_rw(6), // contact_state (output + previous-step penetration)
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
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: state.contact_state_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            pipeline,
            bind_group,
            _params_buffer: params_buffer,
            _geom_buffer: geom_buffer,
            nworld,
            collidable_bodies,
        })
    }

    /// Number of bodies whose geometry the contact pass can collide.
    pub fn collidable_bodies(&self) -> usize {
        self.collidable_bodies
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

/// The GPU geometry type code for a body's primary collision shape.
fn gpu_geom_type(geometry: Option<&Geometry>) -> u32 {
    match geometry {
        None | Some(Geometry::Plane { .. }) => 0,
        Some(Geometry::Sphere { .. }) => 1,
        Some(Geometry::Box { .. }) => 2,
        Some(Geometry::Capsule { .. }) => 3,
        Some(Geometry::Cylinder { .. }) => 4,
        Some(Geometry::Mesh { .. }) => 5,
    }
}

/// The lightest body the contact pass can collide, for the stability bound.
fn lightest_collidable_body(model: &Model) -> Option<(&str, f64)> {
    model
        .bodies
        .iter()
        .filter(|b| gpu_geom_type(b.geometry.as_ref()) != 0 && b.inertia.mass > 0.0)
        .map(|b| (b.name.as_str(), b.inertia.mass))
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

/// Pack body geometry data into a flat f32 array.
///
/// Returns the packed data and the number of bodies with collidable geometry.
fn pack_geometries(
    model: &Model,
    contact: &GroundContactParams,
    body_gains: Option<&[BodyContactGains]>,
) -> (Vec<f32>, usize) {
    let nb = model.nbodies();
    let mut data = vec![0.0f32; nb * GEOM_STRIDE];
    let mut collidable = 0;

    for (i, body) in model.bodies.iter().enumerate() {
        let base = i * GEOM_STRIDE;
        match &body.geometry {
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
            Some(Geometry::Mesh { vertices, .. }) if !vertices.is_empty() => {
                // Body-frame AABB; the shader takes the lowest of its eight
                // rotated corners. Coarser than the true hull but it collides,
                // which silence did not.
                let mut mn = *vertices.first().unwrap();
                let mut mx = mn;
                for v in vertices {
                    mn = Vec3::new(mn.x.min(v.x), mn.y.min(v.y), mn.z.min(v.z));
                    mx = Vec3::new(mx.x.max(v.x), mx.y.max(v.y), mx.z.max(v.z));
                }
                data[base] = 5.0;
                data[base + 1] = mn.x as f32;
                data[base + 2] = mn.y as f32;
                data[base + 3] = mn.z as f32;
                data[base + 4] = mx.x as f32;
                data[base + 5] = mx.y as f32;
                data[base + 6] = mx.z as f32;
            }
            // Planes are the ground's own representation and empty meshes have
            // no extent; neither can collide with the ground plane.
            None | Some(Geometry::Plane { .. }) | Some(Geometry::Mesh { .. }) => {
                data[base] = 0.0;
            }
        }

        if data[base] != 0.0 {
            collidable += 1;
        }

        let (k, d) = match body_gains {
            Some(gains) => (gains[i].stiffness, gains[i].damping),
            None => (contact.stiffness, contact.damping),
        };
        data[base + 8] = k as f32;
        data[base + 9] = d as f32;
    }

    (data, collidable)
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

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{Mat3, SpatialInertia, SpatialTransform};
    use phyz_model::ModelBuilder;

    fn ball_inertia(mass: f64, radius: f64) -> SpatialInertia {
        let i = 0.4 * mass * radius * radius;
        SpatialInertia::new(
            mass,
            Vec3::zeros(),
            Mat3::from_diagonal(&Vec3::new(i, i, i)),
        )
    }

    fn geometry_body(geometry: Geometry) -> phyz_model::Body {
        let mut b = phyz_model::Body::new("geom", ball_inertia(1.0, 0.1), -1, 0);
        b.geometry = Some(geometry);
        b
    }

    #[test]
    fn test_pack_mesh_aabb() {
        let model = ModelBuilder::new()
            .add_free_body_with_geometry(
                "hull",
                -1,
                SpatialTransform::identity(),
                ball_inertia(1.0, 0.1),
                geometry_body(Geometry::Mesh {
                    vertices: vec![
                        Vec3::new(-0.1, -0.2, -0.05),
                        Vec3::new(0.3, 0.1, 0.0),
                        Vec3::new(0.0, 0.0, 0.25),
                    ],
                    faces: vec![[0, 1, 2]],
                }),
            )
            .build();

        let params = GroundContactParams {
            ground_height: 0.0,
            stiffness: 100.0,
            damping: 1.0,
            friction: 0.5,
        };
        let (data, collidable) = pack_geometries(&model, &params, None);
        assert_eq!(collidable, 1);
        assert_eq!(data[0], 5.0);
        assert_eq!(&data[1..4], &[-0.1, -0.2, -0.05]);
        assert_eq!(&data[4..7], &[0.3, 0.1, 0.25]);
        assert_eq!(data[8], 100.0);
        assert_eq!(data[9], 1.0);
    }

    #[test]
    fn test_pack_counts_collidable_and_applies_body_gains() {
        let model = ModelBuilder::new()
            .add_free_body_with_geometry(
                "ball",
                -1,
                SpatialTransform::identity(),
                ball_inertia(2.0, 0.1),
                geometry_body(Geometry::Sphere { radius: 0.1 }),
            )
            .add_revolute_body(
                "bare",
                0,
                SpatialTransform::identity(),
                ball_inertia(0.5, 0.1),
            )
            .build();

        let params = GroundContactParams {
            ground_height: 0.0,
            stiffness: 100.0,
            damping: 1.0,
            friction: 0.5,
        };
        let gains = [
            BodyContactGains {
                stiffness: 200.0,
                damping: 4.0,
            },
            BodyContactGains {
                stiffness: 50.0,
                damping: 1.0,
            },
        ];
        let (data, collidable) = pack_geometries(&model, &params, Some(&gains));
        assert_eq!(collidable, 1);
        assert_eq!(data[8], 200.0);
        assert_eq!(data[9], 4.0);
        assert_eq!(data[GEOM_STRIDE], 0.0); // bare body: no geometry
        assert_eq!(data[GEOM_STRIDE + 8], 50.0);
    }

    #[test]
    fn test_uniform_frequency_gains() {
        let model = ModelBuilder::new()
            .add_free_body_with_geometry(
                "heavy",
                -1,
                SpatialTransform::identity(),
                ball_inertia(5.0, 0.1),
                geometry_body(Geometry::Sphere { radius: 0.1 }),
            )
            .add_free_body_with_geometry(
                "light",
                -1,
                SpatialTransform::identity(),
                ball_inertia(0.001, 0.01),
                geometry_body(Geometry::Sphere { radius: 0.01 }),
            )
            .build();

        let omega = 200.0;
        let zeta = 1.0;
        let gains = BodyContactGains::uniform_frequency(&model, omega, zeta);
        assert_eq!(gains.len(), 2);
        for (g, body) in gains.iter().zip(&model.bodies) {
            let m = body.inertia.mass;
            assert!((g.stiffness - m * omega * omega).abs() < 1e-12);
            assert!((g.damping - 2.0 * zeta * m * omega).abs() < 1e-12);
            // Same frequency for every body regardless of mass.
            assert!(((g.stiffness / m).sqrt() - omega).abs() < 1e-9);
        }
    }

    #[test]
    fn test_check_stability_reports_empty_window() {
        // The measured case from issue #53: a 1-gram link at dt = 1e-3 caps
        // the global stiffness at ~90, far below the 1.354e4 needed for
        // support.
        let model = ModelBuilder::new()
            .dt(1e-3)
            .add_free_body_with_geometry(
                "torso",
                -1,
                SpatialTransform::identity(),
                ball_inertia(6.9, 0.2),
                geometry_body(Geometry::Sphere { radius: 0.2 }),
            )
            .add_free_body_with_geometry(
                "tiny_link",
                -1,
                SpatialTransform::identity(),
                ball_inertia(0.001, 0.01),
                geometry_body(Geometry::Sphere { radius: 0.01 }),
            )
            .build();

        let params = GroundContactParams {
            ground_height: 0.0,
            stiffness: 1.354e4,
            damping: 10.0,
            friction: 0.5,
        };
        let err = params.check_stability(&model).unwrap_err();
        assert!(err.contains("tiny_link"), "unexpected message: {err}");

        let safe = GroundContactParams {
            stiffness: 80.0,
            ..params
        };
        assert!(safe.check_stability(&model).is_ok());
    }
}
