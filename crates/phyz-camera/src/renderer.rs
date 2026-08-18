//! The wgpu rasterizer behind [`RgbdCamera`].
//!
//! Rasterization, not ray tracing: at the 64–128 px resolutions a world model
//! actually consumes, a raster pass over a few thousand triangles is orders of
//! magnitude cheaper than tracing rays, and it is the path that extends
//! naturally to rendering a whole batch of environments in one submission.

use std::sync::Arc;

use crate::error::{CameraError, Result};
use crate::frame::{CameraFrame, ColorBuffer, DepthBuffer};
use crate::pose::{CameraPose, projection_matrix};
use crate::scene::RenderScene;
use phyz_math::{SpatialTransform, Vec3};
use phyz_world::{CameraIntrinsics, Sensor, SensorContext};

/// Row pitch alignment wgpu requires for texture→buffer copies.
const COPY_ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

/// Fixed directional lighting. Photorealism is not the goal; a stable, non-flat
/// shading that makes geometry legible is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lighting {
    /// World-space direction *towards* the light. Normalized on upload.
    pub direction: Vec3,
    /// Ambient term, added everywhere.
    pub ambient: f32,
    /// Diffuse (Lambert) weight.
    pub diffuse: f32,
    /// Background colour for pixels with no geometry, as linear RGBA in
    /// `[0, 1]`. Their depth is `0.0` regardless.
    pub background: [f32; 4],
}

impl Default for Lighting {
    fn default() -> Self {
        Self {
            // Over the shoulder and slightly down, so a floor and a wall shade
            // differently.
            direction: Vec3::new(-0.3, -0.4, 0.87),
            ambient: 0.25,
            diffuse: 0.75,
            background: [0.02, 0.02, 0.03, 1.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view: [f32; 16],
    proj: [f32; 16],
    light_dir: [f32; 4],
    shading: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    row0: [f32; 4],
    row1: [f32; 4],
    row2: [f32; 4],
    albedo: [f32; 4],
}

/// A headless pinhole RGBD camera.
///
/// One camera owns one set of render targets, sized by its intrinsics. Rendering
/// is `render(&scene, &pose)`; the camera itself holds no scene state, so the
/// same camera can be pointed at different scenes and the same scene can be seen
/// by several cameras.
///
/// # Conventions
///
/// OpenCV optical frame (+Z forward, +X right, +Y down), `v = 0` is the top row,
/// depth is **linear metres of optical-axis Z** with `0.0` meaning no return.
/// See [`CameraIntrinsics`].
pub struct RgbdCamera {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    intrinsics: CameraIntrinsics,
    lighting: Lighting,

    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,

    color_target: wgpu::Texture,
    depth_target: wgpu::Texture,
    zbuffer: wgpu::Texture,
    color_readback: wgpu::Buffer,
    depth_readback: wgpu::Buffer,
    color_row_bytes: u32,
    depth_row_bytes: u32,

    vertex_buffer: Option<wgpu::Buffer>,
    /// Set by [`Self::invalidate_scene`]; forces the next render to re-upload
    /// vertices instead of trusting the instance-count heuristic.
    scene_dirty: bool,
    instance_buffer: Option<wgpu::Buffer>,
    draws: Vec<DrawRange>,
}

/// One instanced draw: a contiguous vertex span and its instance slot.
struct DrawRange {
    vertices: std::ops::Range<u32>,
    instance: u32,
}

impl RgbdCamera {
    /// Create a camera with its own wgpu device.
    ///
    /// Returns [`CameraError::NoAdapter`] when no GPU (or software fallback) is
    /// available, so headless CI can skip rather than hang or panic.
    pub fn new(intrinsics: CameraIntrinsics) -> Result<Self> {
        let (device, queue) = default_device()?;
        Self::with_device_queue(intrinsics, device, queue)
    }

    /// Create a camera sharing an existing device and queue — the same
    /// `Arc<Device>` / `Arc<Queue>` pair `phyz_gpu::GpuBatchSimulator` was built
    /// with, so simulation and rendering do not fight over two GPU contexts.
    /// This is why the crate pins the same wgpu version phyz-gpu does.
    pub fn with_device_queue(
        intrinsics: CameraIntrinsics,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Result<Self> {
        if !intrinsics.is_valid() {
            return Err(CameraError::InvalidIntrinsics {
                width: intrinsics.width,
                height: intrinsics.height,
                fx: intrinsics.fx,
                fy: intrinsics.fy,
                near: intrinsics.near,
                far: intrinsics.far,
            });
        }

        let (w, h) = (intrinsics.width, intrinsics.height);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("phyz_camera_raster"),
            source: wgpu::ShaderSource::Wgsl(include_str!("raster.wgsl").into()),
        });

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("phyz_camera_bind_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phyz_camera_uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phyz_camera_bind_group"),
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("phyz_camera_pipeline_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<crate::mesh::Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
                // Vertex tint. Location 6 rather than 2 because 2..5 are the
                // instance buffer's transform rows and albedo; locations are
                // per-pipeline, not per-buffer, so they must not collide.
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 24,
                    shader_location: 6,
                },
            ],
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 4,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 48,
                    shader_location: 5,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("phyz_camera_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout, instance_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::RED,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // No back-face culling: URDF meshes and tessellated primitives
                // are not reliably wound, and a ground quad is legitimately seen
                // from one side only. Depth testing sorts out visibility.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let target = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let attach_copy = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        let color_target = target(
            "phyz_camera_color",
            wgpu::TextureFormat::Rgba8Unorm,
            attach_copy,
        );
        let depth_target = target(
            "phyz_camera_depth",
            wgpu::TextureFormat::R32Float,
            attach_copy,
        );
        let zbuffer = target(
            "phyz_camera_zbuffer",
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );

        let color_row_bytes = aligned_row_bytes(w, 4);
        let depth_row_bytes = aligned_row_bytes(w, 4);
        let readback = |label: &str, row: u32| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (row as u64) * (h as u64),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };

        Ok(Self {
            color_readback: readback("phyz_camera_color_readback", color_row_bytes),
            depth_readback: readback("phyz_camera_depth_readback", depth_row_bytes),
            device,
            queue,
            intrinsics,
            lighting: Lighting::default(),
            pipeline,
            bind_group,
            uniform_buffer,
            color_target,
            depth_target,
            zbuffer,
            color_row_bytes,
            depth_row_bytes,
            vertex_buffer: None,
            instance_buffer: None,
            draws: Vec::new(),
            scene_dirty: false,
        })
    }

    /// The intrinsics this camera renders with.
    pub fn intrinsics(&self) -> &CameraIntrinsics {
        &self.intrinsics
    }

    /// The shared device, for handing to `phyz-gpu` or another camera.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// The shared queue.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Current lighting.
    pub fn lighting(&self) -> &Lighting {
        &self.lighting
    }

    /// Replace the lighting. Affects RGB only; depth is untouched.
    pub fn set_lighting(&mut self, lighting: Lighting) {
        self.lighting = lighting;
    }

    /// Declare that mesh *vertices* changed under a live camera.
    ///
    /// The next render re-uploads them. Needed because the cheap path only
    /// notices a change in the number of drawn instances: repaint a mesh, or
    /// hand the same camera a different scene of the same shape, and without
    /// this the old vertices are drawn with no error anywhere. Adding or
    /// removing an instance already forces the upload, so this is only for the
    /// same-shape case.
    pub fn invalidate_scene(&mut self) {
        self.scene_dirty = true;
    }

    /// Upload a scene's geometry. Call once per topology change; per-step pose
    /// updates go through [`Self::render`], which re-uploads only the small
    /// instance buffer.
    pub fn upload_scene(&mut self, scene: &RenderScene) {
        let mut vertices: Vec<crate::mesh::Vertex> = Vec::new();
        let mut draws = Vec::new();
        let mut instances: Vec<InstanceRaw> = Vec::with_capacity(scene.instances.len());

        for (slot, inst) in scene.instances.iter().enumerate() {
            let mesh = &scene.meshes[inst.mesh];
            if mesh.is_empty() {
                continue;
            }
            let start = vertices.len() as u32;
            vertices.extend_from_slice(&mesh.vertices);
            draws.push(DrawRange {
                vertices: start..vertices.len() as u32,
                instance: slot as u32,
            });
            instances.push(instance_raw(inst));
        }
        // Instance slots must line up with what was actually pushed.
        for (i, d) in draws.iter_mut().enumerate() {
            d.instance = i as u32;
        }

        self.vertex_buffer = (!vertices.is_empty()).then(|| {
            self.upload(
                "phyz_camera_vertices",
                bytemuck::cast_slice(&vertices),
                wgpu::BufferUsages::VERTEX,
            )
        });
        self.instance_buffer = (!instances.is_empty()).then(|| {
            self.upload(
                "phyz_camera_instances",
                bytemuck::cast_slice(&instances),
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            )
        });
        self.draws = draws;
    }

    /// Refresh only the per-instance world placements, leaving vertex data
    /// alone. The instance layout must match the last [`Self::upload_scene`].
    fn update_instances(&mut self, scene: &RenderScene) {
        let Some(buffer) = &self.instance_buffer else {
            return;
        };
        let instances: Vec<InstanceRaw> = scene
            .instances
            .iter()
            .filter(|i| !scene.meshes[i.mesh].is_empty())
            .map(instance_raw)
            .collect();
        if instances.len() == self.draws.len() {
            self.queue
                .write_buffer(buffer, 0, bytemuck::cast_slice(&instances));
        }
    }

    /// Render one RGBD frame.
    ///
    /// Re-uploads geometry when the scene's shape changed since the last call,
    /// and otherwise only rewrites the instance buffer — the common case for a
    /// robot whose links move but whose meshes do not.
    pub fn render(&mut self, scene: &RenderScene, pose: &CameraPose) -> Result<CameraFrame> {
        self.render_at(scene, pose, 0.0)
    }

    /// [`Self::render`], stamping the frame with a simulation time.
    pub fn render_at(
        &mut self,
        scene: &RenderScene,
        pose: &CameraPose,
        timestamp: f64,
    ) -> Result<CameraFrame> {
        let drawable = scene
            .instances
            .iter()
            .filter(|i| !scene.meshes[i.mesh].is_empty())
            .count();
        // Vertex data is re-uploaded when the shape of the scene changes or
        // when the caller says it has; otherwise only the instance rows are
        // rewritten, which is what makes a per-step render cheap. The rule
        // matters because it is not conservative: a scene with the same number
        // of drawable instances but *different vertices* — a mesh repainted, a
        // room swapped for another of equal size — keeps drawing the old
        // buffer. That was invisible while meshes were static geometry and
        // became a real way to render a stale room once vertices carried
        // colour, so [`Self::invalidate_scene`] exists to say so explicitly.
        if self.vertex_buffer.is_none() || self.scene_dirty || drawable != self.draws.len() {
            self.upload_scene(scene);
        } else {
            self.update_instances(scene);
        }
        self.scene_dirty = false;

        let light = self
            .lighting
            .direction
            .try_normalize()
            .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
        let uniforms = Uniforms {
            view: pose.view_matrix(),
            proj: projection_matrix(&self.intrinsics),
            light_dir: [light.x as f32, light.y as f32, light.z as f32, 0.0],
            shading: [self.lighting.ambient, self.lighting.diffuse, 0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let color_view = self.color_target.create_view(&Default::default());
        let depth_view = self.depth_target.create_view(&Default::default());
        let z_view = self.zbuffer.create_view(&Default::default());
        let bg = self.lighting.background;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("phyz_camera_encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("phyz_camera_pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: bg[0] as f64,
                                g: bg[1] as f64,
                                b: bg[2] as f64,
                                a: bg[3] as f64,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &depth_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // 0.0 is the documented "no return" depth.
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &z_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if let (Some(vb), Some(ib)) = (&self.vertex_buffer, &self.instance_buffer) {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_vertex_buffer(1, ib.slice(..));
                for d in &self.draws {
                    pass.draw(d.vertices.clone(), d.instance..d.instance + 1);
                }
            }
        }

        copy_to_buffer(
            &mut encoder,
            &self.color_target,
            &self.color_readback,
            self.color_row_bytes,
            self.intrinsics.height,
        );
        copy_to_buffer(
            &mut encoder,
            &self.depth_target,
            &self.depth_readback,
            self.depth_row_bytes,
            self.intrinsics.height,
        );
        self.queue.submit(Some(encoder.finish()));

        let color = self.read_rows(&self.color_readback, self.color_row_bytes, 4)?;
        let depth_bytes = self.read_rows(&self.depth_readback, self.depth_row_bytes, 4)?;
        let depth: Vec<f32> = depth_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Ok(CameraFrame {
            intrinsics: self.intrinsics,
            timestamp,
            color: ColorBuffer::Cpu(color),
            depth: DepthBuffer::Cpu(depth),
        })
    }

    /// Render the view from a [`Sensor::Camera`] descriptor.
    ///
    /// The pose comes from the same [`SensorContext`] the other sensors read, so
    /// the image is consistent with the rest of the observation by construction.
    /// The camera's own mount body is *not* excluded automatically — pass
    /// `SceneOptions::exclude_body` when building the scene if you want that.
    pub fn render_sensor(
        &mut self,
        ctx: &SensorContext<'_>,
        sensor: &Sensor,
        scene: &RenderScene,
        sensor_id: usize,
    ) -> Result<CameraFrame> {
        let pose = sensor_pose(ctx, sensor, sensor_id)?;
        self.render_at(scene, &pose, ctx.state.time)
    }

    fn upload(&self, label: &str, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: data.len() as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, data);
        buffer
    }

    /// Map a readback buffer and strip the row padding wgpu's copy alignment
    /// forces on us.
    fn read_rows(&self, buffer: &wgpu::Buffer, row_bytes: u32, bpp: u32) -> Result<Vec<u8>> {
        let (w, h) = (self.intrinsics.width, self.intrinsics.height);
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| CameraError::Readback(e.to_string()))?
            .map_err(|e| CameraError::Readback(e.to_string()))?;

        let tight = (w * bpp) as usize;
        let mut out = Vec::with_capacity(tight * h as usize);
        {
            let mapped = slice.get_mapped_range();
            for row in 0..h as usize {
                let start = row * row_bytes as usize;
                out.extend_from_slice(&mapped[start..start + tight]);
            }
        }
        buffer.unmap();
        Ok(out)
    }
}

/// The world pose of a [`Sensor::Camera`], from a step's body transforms.
pub fn sensor_pose(
    ctx: &SensorContext<'_>,
    sensor: &Sensor,
    sensor_id: usize,
) -> Result<CameraPose> {
    let Sensor::Camera {
        body_idx, origin, ..
    } = sensor
    else {
        return Err(CameraError::NotACamera { sensor_id });
    };
    body_pose(ctx.xforms(), *body_idx, origin)
}

/// The world pose of a camera mounted on `body_idx` with the given extrinsics.
pub fn body_pose(
    xforms: &[SpatialTransform],
    body_idx: usize,
    extrinsics: &SpatialTransform,
) -> Result<CameraPose> {
    let xform = xforms.get(body_idx).ok_or(CameraError::UnknownBody {
        body_idx,
        nbodies: xforms.len(),
    })?;
    Ok(CameraPose::from_body(xform, extrinsics))
}

fn instance_raw(inst: &crate::scene::Instance) -> InstanceRaw {
    let r = &inst.world_from_local;
    let p = inst.position;
    let row = |i: usize, t: f64| {
        [
            r.get(i, 0) as f32,
            r.get(i, 1) as f32,
            r.get(i, 2) as f32,
            t as f32,
        ]
    };
    InstanceRaw {
        row0: row(0, p.x),
        row1: row(1, p.y),
        row2: row(2, p.z),
        albedo: [inst.albedo[0], inst.albedo[1], inst.albedo[2], 1.0],
    }
}

fn aligned_row_bytes(width: u32, bytes_per_pixel: u32) -> u32 {
    let unpadded = width * bytes_per_pixel;
    unpadded.div_ceil(COPY_ALIGN) * COPY_ALIGN
}

fn copy_to_buffer(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    buffer: &wgpu::Buffer,
    row_bytes: u32,
    height: u32,
) {
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width: texture.width(),
            height,
            depth_or_array_layers: 1,
        },
    );
}

/// Request a wgpu device suitable for headless rendering.
///
/// No surface is created and none is needed, so this works on a machine with no
/// display server at all.
pub fn default_device() -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or(CameraError::NoAdapter)?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("phyz_camera_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: Default::default(),
        },
        None,
    ))
    .map_err(|e| CameraError::DeviceCreation(e.to_string()))?;

    Ok((Arc::new(device), Arc::new(queue)))
}
