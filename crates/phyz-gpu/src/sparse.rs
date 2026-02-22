//! GPU sparse matrix-vector operations for Lanczos eigensolver.
//!
//! Provides CSR SpMV, dot products, axpy, scaling, and fused
//! reorthogonalization primitives. Supports f64 (preferred) with
//! f32 fallback for devices without `SHADER_FLOAT64`.

use crate::sparse_shaders;
use std::sync::Arc;

/// Detected GPU floating-point precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPrecision {
    F64,
    F32,
}

impl GpuPrecision {
    /// Bytes per scalar element.
    pub fn elem_size(self) -> usize {
        match self {
            GpuPrecision::F64 => 8,
            GpuPrecision::F32 => 4,
        }
    }
}

/// CSR sparse matrix stored on GPU.
pub struct GpuSparseMatrix {
    pub nrows: u32,
    pub nnz: u32,
    pub row_ptr_buf: wgpu::Buffer,
    pub col_idx_buf: wgpu::Buffer,
    pub vals_buf: wgpu::Buffer,
}

/// GPU vector operations engine.
///
/// Holds compute pipelines, scratch buffers, and device references.
/// All operations encode into a `CommandEncoder` for batched submission.
pub struct GpuVecOps {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub precision: GpuPrecision,
    pub dim: u32,

    // Pipelines
    spmv_pipeline: wgpu::ComputePipeline,
    axpy_pipeline: wgpu::ComputePipeline,
    scale_pipeline: wgpu::ComputePipeline,
    dot_phase1_pipeline: wgpu::ComputePipeline,
    dot_phase2_pipeline: wgpu::ComputePipeline,
    multi_dot_phase1_pipeline: wgpu::ComputePipeline,
    multi_dot_phase2_pipeline: wgpu::ComputePipeline,
    batch_subtract_pipeline: wgpu::ComputePipeline,

    // Scratch buffers for reductions
    partials_buf: wgpu::Buffer,
    n_workgroups: u32,

    // Staging buffer for scalar readback
    scalar_staging: wgpu::Buffer,
    /// Result buffer for dot/multi_dot (also used as overlaps input for batch_subtract).
    pub scalar_result_buf: wgpu::Buffer,

    // Dim uniform buffer
    dim_uniform: wgpu::Buffer,
}

/// Request a GPU device, preferring f64 support.
pub fn request_device() -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>, GpuPrecision), String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or("No GPU adapter found")?;

    let has_f64 = adapter
        .features()
        .contains(wgpu::Features::SHADER_F64);

    let required_features = if has_f64 {
        wgpu::Features::SHADER_F64
    } else {
        wgpu::Features::empty()
    };

    // Request the adapter's actual limits (not wgpu defaults) to allow large buffers.
    let mut limits = wgpu::Limits::default();
    let adapter_limits = adapter.limits();
    limits.max_buffer_size = adapter_limits.max_buffer_size;
    limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("phyz-sparse-device"),
            required_features,
            required_limits: limits,
            memory_hints: Default::default(),
        },
        None,
    ))
    .map_err(|e| format!("Failed to create device: {e}"))?;

    let precision = if has_f64 {
        GpuPrecision::F64
    } else {
        GpuPrecision::F32
    };

    Ok((Arc::new(device), Arc::new(queue), precision))
}

impl GpuSparseMatrix {
    /// Upload a CSR matrix to GPU. Values are converted to f32 if precision is F32.
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        row_ptr: &[u32],
        col_indices: &[u32],
        values_f64: &[f64],
        precision: GpuPrecision,
    ) -> Self {
        let nrows = (row_ptr.len() - 1) as u32;
        let nnz = col_indices.len() as u32;

        let row_ptr_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("csr_row_ptr"),
            size: (row_ptr.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&row_ptr_buf, 0, bytemuck::cast_slice(row_ptr));

        let col_idx_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("csr_col_idx"),
            size: (col_indices.len() * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&col_idx_buf, 0, bytemuck::cast_slice(col_indices));

        let vals_buf = match precision {
            GpuPrecision::F64 => {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("csr_vals_f64"),
                    size: (values_f64.len() * 8).max(8) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(values_f64));
                buf
            }
            GpuPrecision::F32 => {
                let vals_f32: Vec<f32> = values_f64.iter().map(|&v| v as f32).collect();
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("csr_vals_f32"),
                    size: (vals_f32.len() * 4).max(4) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(&vals_f32));
                buf
            }
        };

        Self {
            nrows,
            nnz,
            row_ptr_buf,
            col_idx_buf,
            vals_buf,
        }
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader_src: &str,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{label}_layout")),
                bind_group_layouts: &[bind_group_layout],
                push_constant_ranges: &[],
            }),
        ),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

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

impl GpuVecOps {
    /// Create GPU vector operations engine.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        precision: GpuPrecision,
        dim: u32,
    ) -> Self {
        let n_workgroups = dim.div_ceil(256).min(65535);
        let elem_size = precision.elem_size();

        // --- SpMV pipeline: 6 bindings ---
        let spmv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spmv_bgl"),
            entries: &[
                bgl_storage_ro(0), // row_ptr
                bgl_storage_ro(1), // col_idx
                bgl_storage_ro(2), // vals
                bgl_storage_ro(3), // x
                bgl_storage_rw(4), // y
                bgl_uniform(5),    // dim
            ],
        });
        let spmv_shader = match precision {
            GpuPrecision::F64 => sparse_shaders::SPMV_F64,
            GpuPrecision::F32 => sparse_shaders::SPMV_F32,
        };
        let spmv_pipeline = create_pipeline(&device, "spmv", spmv_shader, &spmv_bgl);

        // --- AXPY pipeline: 3 bindings ---
        let axpy_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("axpy_bgl"),
            entries: &[
                bgl_uniform(0),    // params (dim + alpha)
                bgl_storage_ro(1), // x
                bgl_storage_rw(2), // y
            ],
        });
        let axpy_shader = match precision {
            GpuPrecision::F64 => sparse_shaders::AXPY_F64,
            GpuPrecision::F32 => sparse_shaders::AXPY_F32,
        };
        let axpy_pipeline = create_pipeline(&device, "axpy", axpy_shader, &axpy_bgl);

        // --- Scale pipeline: 2 bindings ---
        let scale_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scale_bgl"),
            entries: &[
                bgl_uniform(0),    // params (dim + alpha)
                bgl_storage_rw(1), // x
            ],
        });
        let scale_shader = match precision {
            GpuPrecision::F64 => sparse_shaders::SCALE_F64,
            GpuPrecision::F32 => sparse_shaders::SCALE_F32,
        };
        let scale_pipeline = create_pipeline(&device, "scale", scale_shader, &scale_bgl);

        // --- Dot phase1: 4 bindings ---
        let dot1_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dot_phase1_bgl"),
            entries: &[
                bgl_storage_ro(0), // a
                bgl_storage_ro(1), // b
                bgl_storage_rw(2), // partials
                bgl_uniform(3),    // dim
            ],
        });
        let dot1_shader = match precision {
            GpuPrecision::F64 => sparse_shaders::DOT_PHASE1_F64,
            GpuPrecision::F32 => sparse_shaders::DOT_PHASE1_F32,
        };
        let dot_phase1_pipeline = create_pipeline(&device, "dot_phase1", dot1_shader, &dot1_bgl);

        // --- Dot phase2: 3 bindings ---
        let dot2_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dot_phase2_bgl"),
            entries: &[
                bgl_storage_ro(0), // partials
                bgl_storage_rw(1), // result
                bgl_uniform(2),    // n_partials
            ],
        });
        let dot2_shader = match precision {
            GpuPrecision::F64 => sparse_shaders::DOT_PHASE2_F64,
            GpuPrecision::F32 => sparse_shaders::DOT_PHASE2_F32,
        };
        let dot_phase2_pipeline = create_pipeline(&device, "dot_phase2", dot2_shader, &dot2_bgl);

        // --- Multi-dot phase1: 4 bindings ---
        let mdot1_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("multi_dot_phase1_bgl"),
            entries: &[
                bgl_uniform(0),    // params
                bgl_storage_ro(1), // q_bank
                bgl_storage_ro(2), // w
                bgl_storage_rw(3), // partials
            ],
        });
        let mdot1_shader = match precision {
            GpuPrecision::F64 => sparse_shaders::MULTI_DOT_PHASE1_F64,
            GpuPrecision::F32 => sparse_shaders::MULTI_DOT_PHASE1_F32,
        };
        let multi_dot_phase1_pipeline =
            create_pipeline(&device, "multi_dot_phase1", mdot1_shader, &mdot1_bgl);

        // --- Multi-dot phase2: 3 bindings ---
        let mdot2_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("multi_dot_phase2_bgl"),
            entries: &[
                bgl_uniform(0),    // params
                bgl_storage_ro(1), // partials
                bgl_storage_rw(2), // overlaps
            ],
        });
        let mdot2_shader = match precision {
            GpuPrecision::F64 => sparse_shaders::MULTI_DOT_PHASE2_F64,
            GpuPrecision::F32 => sparse_shaders::MULTI_DOT_PHASE2_F32,
        };
        let multi_dot_phase2_pipeline =
            create_pipeline(&device, "multi_dot_phase2", mdot2_shader, &mdot2_bgl);

        // --- Batch subtract: 4 bindings ---
        let bsub_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("batch_subtract_bgl"),
            entries: &[
                bgl_uniform(0),    // params
                bgl_storage_ro(1), // q_bank
                bgl_storage_ro(2), // overlaps
                bgl_storage_rw(3), // w
            ],
        });
        let bsub_shader = match precision {
            GpuPrecision::F64 => sparse_shaders::BATCH_SUBTRACT_F64,
            GpuPrecision::F32 => sparse_shaders::BATCH_SUBTRACT_F32,
        };
        let batch_subtract_pipeline =
            create_pipeline(&device, "batch_subtract", bsub_shader, &bsub_bgl);

        // --- Scratch buffers ---
        // Partials for dot reduction (also reused for multi-dot with larger allocation)
        let partials_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dot_partials"),
            // Allocate enough for multi-dot: n_workgroups * max_vecs (allocate generously)
            size: (n_workgroups as usize * 512 * elem_size).max(elem_size) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let scalar_result_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scalar_result"),
            size: (elem_size * 512) as u64, // room for multi-dot overlaps
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let scalar_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scalar_staging"),
            size: (elem_size * 512) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dim uniform
        let dim_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dim_uniform"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&dim_uniform, 0, bytemuck::bytes_of(&dim));

        Self {
            device,
            queue,
            precision,
            dim,
            spmv_pipeline,
            axpy_pipeline,
            scale_pipeline,
            dot_phase1_pipeline,
            dot_phase2_pipeline,
            multi_dot_phase1_pipeline,
            multi_dot_phase2_pipeline,
            batch_subtract_pipeline,
            partials_buf,
            n_workgroups,
            scalar_staging,
            scalar_result_buf,
            dim_uniform,
        }
    }

    /// Create a GPU vector buffer of the given dimension.
    pub fn create_vec_buffer(&self, label: &str) -> wgpu::Buffer {
        let size = (self.dim as usize * self.precision.elem_size()).max(self.precision.elem_size());
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Create the q_bank buffer for storing Lanczos vectors.
    pub fn create_q_bank(&self, max_vecs: u32) -> wgpu::Buffer {
        let size = max_vecs as usize * self.dim as usize * self.precision.elem_size();
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q_bank"),
            size: size.max(self.precision.elem_size()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Upload f64 data to a GPU buffer (converts to f32 if needed).
    pub fn upload_vec(&self, buf: &wgpu::Buffer, data: &[f64]) {
        match self.precision {
            GpuPrecision::F64 => {
                self.queue
                    .write_buffer(buf, 0, bytemuck::cast_slice(data));
            }
            GpuPrecision::F32 => {
                let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                self.queue
                    .write_buffer(buf, 0, bytemuck::cast_slice(&f32_data));
            }
        }
    }

    /// Upload f64 data into q_bank at vector slot `index`.
    pub fn upload_q_vec(&self, q_bank: &wgpu::Buffer, index: u32, data: &[f64]) {
        let offset = index as u64 * self.dim as u64 * self.precision.elem_size() as u64;
        match self.precision {
            GpuPrecision::F64 => {
                self.queue
                    .write_buffer(q_bank, offset, bytemuck::cast_slice(data));
            }
            GpuPrecision::F32 => {
                let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                self.queue
                    .write_buffer(q_bank, offset, bytemuck::cast_slice(&f32_data));
            }
        }
    }

    /// Encode SpMV: y = matrix * x
    pub fn encode_spmv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        matrix: &GpuSparseMatrix,
        x_buf: &wgpu::Buffer,
        y_buf: &wgpu::Buffer,
    ) {
        let bg = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("spmv_bg"),
                layout: &self.spmv_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: matrix.row_ptr_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: matrix.col_idx_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: matrix.vals_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: y_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.dim_uniform.as_entire_binding(),
                    },
                ],
            });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("spmv_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.spmv_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(self.n_workgroups, 1, 1);
    }

    /// Encode AXPY: y += alpha * x
    pub fn encode_axpy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        alpha: f64,
        x_buf: &wgpu::Buffer,
        y_buf: &wgpu::Buffer,
    ) {
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("axpy_params"),
            size: 16, // enough for both f64 and f32 variants
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        match self.precision {
            GpuPrecision::F64 => {
                // struct { dim: u32, _pad: u32, alpha: vec2<u32> (bitcast f64) }
                let mut data = [0u8; 16];
                data[0..4].copy_from_slice(&self.dim.to_le_bytes());
                data[4..8].copy_from_slice(&0u32.to_le_bytes());
                data[8..16].copy_from_slice(&alpha.to_le_bytes());
                self.queue.write_buffer(&params_buf, 0, &data);
            }
            GpuPrecision::F32 => {
                // struct { dim: u32, alpha: f32 }
                let mut data = [0u8; 8];
                data[0..4].copy_from_slice(&self.dim.to_le_bytes());
                data[4..8].copy_from_slice(&(alpha as f32).to_le_bytes());
                self.queue.write_buffer(&params_buf, 0, &data);
            }
        }

        let bg = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("axpy_bg"),
                layout: &self.axpy_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: y_buf.as_entire_binding(),
                    },
                ],
            });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("axpy_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.axpy_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(self.n_workgroups, 1, 1);
    }

    /// Encode SCALE: x *= alpha
    pub fn encode_scale(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        alpha: f64,
        x_buf: &wgpu::Buffer,
    ) {
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scale_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        match self.precision {
            GpuPrecision::F64 => {
                let mut data = [0u8; 16];
                data[0..4].copy_from_slice(&self.dim.to_le_bytes());
                data[4..8].copy_from_slice(&0u32.to_le_bytes());
                data[8..16].copy_from_slice(&alpha.to_le_bytes());
                self.queue.write_buffer(&params_buf, 0, &data);
            }
            GpuPrecision::F32 => {
                let mut data = [0u8; 8];
                data[0..4].copy_from_slice(&self.dim.to_le_bytes());
                data[4..8].copy_from_slice(&(alpha as f32).to_le_bytes());
                self.queue.write_buffer(&params_buf, 0, &data);
            }
        }

        let bg = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scale_bg"),
                layout: &self.scale_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x_buf.as_entire_binding(),
                    },
                ],
            });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scale_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.scale_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(self.n_workgroups, 1, 1);
    }

    /// Encode dot product: result = a · b
    /// Result is written to internal scalar_result_buf.
    pub fn encode_dot(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a_buf: &wgpu::Buffer,
        b_buf: &wgpu::Buffer,
    ) {
        // Phase 1: partial sums
        let bg1 = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dot_phase1_bg"),
                layout: &self.dot_phase1_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.partials_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.dim_uniform.as_entire_binding(),
                    },
                ],
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dot_phase1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.dot_phase1_pipeline);
            pass.set_bind_group(0, &bg1, &[]);
            pass.dispatch_workgroups(self.n_workgroups, 1, 1);
        }

        // Phase 2: reduce partials
        let n_partials_uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("n_partials_uniform"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(
            &n_partials_uniform,
            0,
            bytemuck::bytes_of(&self.n_workgroups),
        );

        let bg2 = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dot_phase2_bg"),
                layout: &self.dot_phase2_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.partials_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.scalar_result_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: n_partials_uniform.as_entire_binding(),
                    },
                ],
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dot_phase2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.dot_phase2_pipeline);
            pass.set_bind_group(0, &bg2, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    /// Encode multi-dot: overlaps[k] = dot(q_bank[k], w) for k=0..n_vecs.
    /// Result is written to scalar_result_buf (first n_vecs elements).
    pub fn encode_multi_dot(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q_bank: &wgpu::Buffer,
        w_buf: &wgpu::Buffer,
        n_vecs: u32,
    ) {
        if n_vecs == 0 {
            return;
        }

        // Phase 1: 2D dispatch
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mdot_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_data: [u32; 4] = [self.dim, n_vecs, self.dim, 0]; // dim, n_vecs, stride, pad
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

        let bg1 = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mdot_phase1_bg"),
                layout: &self.multi_dot_phase1_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: q_bank.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.partials_buf.as_entire_binding(),
                    },
                ],
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mdot_phase1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.multi_dot_phase1_pipeline);
            pass.set_bind_group(0, &bg1, &[]);
            pass.dispatch_workgroups(self.n_workgroups, n_vecs, 1);
        }

        // Phase 2: reduce per vector
        let params2_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mdot_phase2_params"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params2_data: [u32; 2] = [self.n_workgroups, n_vecs];
        self.queue
            .write_buffer(&params2_buf, 0, bytemuck::cast_slice(&params2_data));

        let bg2 = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mdot_phase2_bg"),
                layout: &self.multi_dot_phase2_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params2_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.partials_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.scalar_result_buf.as_entire_binding(),
                    },
                ],
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mdot_phase2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.multi_dot_phase2_pipeline);
            pass.set_bind_group(0, &bg2, &[]);
            pass.dispatch_workgroups(n_vecs, 1, 1);
        }
    }

    /// Encode batch subtract: w -= sum_k overlaps[k] * q_bank[k]
    /// `overlaps_buf` should contain n_vecs scalars (from multi_dot).
    pub fn encode_batch_subtract(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q_bank: &wgpu::Buffer,
        overlaps_buf: &wgpu::Buffer,
        w_buf: &wgpu::Buffer,
        n_vecs: u32,
    ) {
        if n_vecs == 0 {
            return;
        }

        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bsub_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_data: [u32; 4] = [self.dim, n_vecs, self.dim, 0];
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

        let bg = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bsub_bg"),
                layout: &self.batch_subtract_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: q_bank.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: overlaps_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: w_buf.as_entire_binding(),
                    },
                ],
            });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bsub_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.batch_subtract_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(self.n_workgroups, 1, 1);
    }

    /// Copy w_buf into q_bank at slot `index`.
    pub fn encode_copy_to_q_bank(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        w_buf: &wgpu::Buffer,
        q_bank: &wgpu::Buffer,
        index: u32,
    ) {
        let offset = index as u64 * self.dim as u64 * self.precision.elem_size() as u64;
        let size = self.dim as u64 * self.precision.elem_size() as u64;
        encoder.copy_buffer_to_buffer(w_buf, 0, q_bank, offset, size);
    }

    /// Submit an encoder and read back a scalar from scalar_result_buf.
    pub fn submit_and_read_scalar(&self, encoder: wgpu::CommandEncoder) -> f64 {
        let elem_size = self.precision.elem_size() as u64;

        // Copy result to staging
        let mut encoder = encoder;
        encoder.copy_buffer_to_buffer(
            &self.scalar_result_buf,
            0,
            &self.scalar_staging,
            0,
            elem_size,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map and read
        let slice = self.scalar_staging.slice(0..elem_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("channel closed")
            .expect("buffer map failed");

        let data = slice.get_mapped_range();
        let val = match self.precision {
            GpuPrecision::F64 => {
                let bytes: [u8; 8] = data[0..8].try_into().unwrap();
                f64::from_le_bytes(bytes)
            }
            GpuPrecision::F32 => {
                let bytes: [u8; 4] = data[0..4].try_into().unwrap();
                f32::from_le_bytes(bytes) as f64
            }
        };
        drop(data);
        self.scalar_staging.unmap();
        val
    }

    /// Submit work without reading back.
    pub fn submit(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Download a vector from GPU to CPU (f64).
    pub fn download_vec(&self, buf: &wgpu::Buffer) -> Vec<f64> {
        let size = self.dim as u64 * self.precision.elem_size() as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download_encoder"),
            });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("channel closed")
            .expect("buffer map failed");

        let data = slice.get_mapped_range();
        let result = match self.precision {
            GpuPrecision::F64 => bytemuck::cast_slice::<u8, f64>(&data).to_vec(),
            GpuPrecision::F32 => {
                let f32s: &[f32] = bytemuck::cast_slice(&data);
                f32s.iter().map(|&v| v as f64).collect()
            }
        };
        drop(data);
        staging.unmap();
        result
    }

    /// Download q_bank vector at slot `index` to CPU (f64).
    pub fn download_q_vec(&self, q_bank: &wgpu::Buffer, index: u32) -> Vec<f64> {
        let elem_size = self.precision.elem_size() as u64;
        let vec_size = self.dim as u64 * elem_size;
        let offset = index as u64 * vec_size;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download_q_staging"),
            size: vec_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download_q_encoder"),
            });
        encoder.copy_buffer_to_buffer(q_bank, offset, &staging, 0, vec_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("channel closed")
            .expect("buffer map failed");

        let data = slice.get_mapped_range();
        let result = match self.precision {
            GpuPrecision::F64 => bytemuck::cast_slice::<u8, f64>(&data).to_vec(),
            GpuPrecision::F32 => {
                let f32s: &[f32] = bytemuck::cast_slice(&data);
                f32s.iter().map(|&v| v as f64).collect()
            }
        };
        drop(data);
        staging.unmap();
        result
    }

    /// Run multi-dot for a chunk of q_bank vectors, writing results at an offset
    /// in `scalar_result_buf`.
    ///
    /// Computes `overlaps[k] = dot(q_chunk[k], w)` for `k = 0..n_vecs`, storing
    /// results at `scalar_result_buf[result_offset .. result_offset + n_vecs]`.
    pub fn run_multi_dot_range(
        &self,
        q_buf: &wgpu::Buffer,
        w_buf: &wgpu::Buffer,
        n_vecs: u32,
        result_offset: u32,
    ) {
        if n_vecs == 0 {
            return;
        }

        let elem_size = self.precision.elem_size() as u64;
        let mut encoder = self.encoder();

        // Phase 1: 2D dispatch (same as encode_multi_dot)
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mdot_range_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_data: [u32; 4] = [self.dim, n_vecs, self.dim, 0];
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

        let bg1 = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mdot_range_phase1_bg"),
                layout: &self.multi_dot_phase1_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: q_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.partials_buf.as_entire_binding(),
                    },
                ],
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mdot_range_phase1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.multi_dot_phase1_pipeline);
            pass.set_bind_group(0, &bg1, &[]);
            pass.dispatch_workgroups(self.n_workgroups, n_vecs, 1);
        }

        // Phase 2: reduce per vector, writing at offset in scalar_result_buf
        let params2_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mdot_range_phase2_params"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params2_data: [u32; 2] = [self.n_workgroups, n_vecs];
        self.queue
            .write_buffer(&params2_buf, 0, bytemuck::cast_slice(&params2_data));

        let result_byte_offset = result_offset as u64 * elem_size;
        let result_byte_size = n_vecs as u64 * elem_size;

        let bg2 = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mdot_range_phase2_bg"),
                layout: &self.multi_dot_phase2_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params2_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.partials_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.scalar_result_buf,
                            offset: result_byte_offset,
                            size: std::num::NonZeroU64::new(result_byte_size),
                        }),
                    },
                ],
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mdot_range_phase2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.multi_dot_phase2_pipeline);
            pass.set_bind_group(0, &bg2, &[]);
            pass.dispatch_workgroups(n_vecs, 1, 1);
        }

        self.submit(encoder);
    }

    /// Run batch subtract for a chunk of q_bank vectors, reading overlaps from
    /// an offset in `scalar_result_buf`.
    ///
    /// Computes `w -= sum_k overlaps[k] * q_chunk[k]` for `k = 0..n_vecs`,
    /// reading overlaps from `scalar_result_buf[overlaps_offset..]`.
    pub fn run_batch_subtract_range(
        &self,
        q_buf: &wgpu::Buffer,
        w_buf: &wgpu::Buffer,
        n_vecs: u32,
        overlaps_offset: u32,
    ) {
        if n_vecs == 0 {
            return;
        }

        let elem_size = self.precision.elem_size() as u64;
        let mut encoder = self.encoder();

        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bsub_range_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_data: [u32; 4] = [self.dim, n_vecs, self.dim, 0];
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

        let overlaps_byte_offset = overlaps_offset as u64 * elem_size;
        let overlaps_byte_size = n_vecs as u64 * elem_size;

        let bg = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bsub_range_bg"),
                layout: &self.batch_subtract_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: q_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.scalar_result_buf,
                            offset: overlaps_byte_offset,
                            size: std::num::NonZeroU64::new(overlaps_byte_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: w_buf.as_entire_binding(),
                    },
                ],
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bsub_range_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.batch_subtract_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.n_workgroups, 1, 1);
        }

        self.submit(encoder);
    }

    /// Create a new command encoder.
    pub fn encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vecops_encoder"),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>, GpuPrecision)> {
        match request_device() {
            Ok(triple) => Some(triple),
            Err(e) => {
                eprintln!("Skipping GPU test (no adapter): {e}");
                None
            }
        }
    }

    #[test]
    fn test_spmv_small() {
        let Some((device, queue, precision)) = setup() else {
            return;
        };

        // 5x5 tridiagonal: diag=2, off-diag=-1
        let n = 5u32;
        let mut row_ptr = vec![0u32; 6];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..5u32 {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(-1.0f64);
            }
            col_indices.push(i);
            values.push(2.0);
            if i < 4 {
                col_indices.push(i + 1);
                values.push(-1.0);
            }
            row_ptr[(i + 1) as usize] = col_indices.len() as u32;
        }

        let mat = GpuSparseMatrix::upload(&device, &queue, &row_ptr, &col_indices, &values, precision);
        let ops = GpuVecOps::new(device, queue, precision, n);

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x_buf = ops.create_vec_buffer("x");
        ops.upload_vec(&x_buf, &x);

        let y_buf = ops.create_vec_buffer("y");

        let mut encoder = ops.encoder();
        ops.encode_spmv(&mut encoder, &mat, &x_buf, &y_buf);
        ops.submit(encoder);

        let y = ops.download_vec(&y_buf);

        // Expected: tridiag * [1,2,3,4,5]
        let expected = [0.0, 0.0, 0.0, 0.0, 6.0];
        let tol = if precision == GpuPrecision::F64 {
            1e-10
        } else {
            1e-5
        };
        for i in 0..5 {
            assert!(
                (y[i] - expected[i]).abs() < tol,
                "spmv[{i}]: got={}, expected={}, diff={}",
                y[i],
                expected[i],
                (y[i] - expected[i]).abs()
            );
        }
    }

    #[test]
    fn test_dot_product() {
        let Some((device, queue, precision)) = setup() else {
            return;
        };

        let n = 257u32; // boundary size
        let ops = GpuVecOps::new(device, queue, precision, n);

        let a: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.1).collect();
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.2).collect();

        let a_buf = ops.create_vec_buffer("a");
        let b_buf = ops.create_vec_buffer("b");
        ops.upload_vec(&a_buf, &a);
        ops.upload_vec(&b_buf, &b);

        let mut encoder = ops.encoder();
        ops.encode_dot(&mut encoder, &a_buf, &b_buf);
        let result = ops.submit_and_read_scalar(encoder);

        let expected: f64 = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
        let tol = if precision == GpuPrecision::F64 {
            1e-6
        } else {
            expected.abs() * 1e-4
        };
        assert!(
            (result - expected).abs() < tol,
            "dot: got={result}, expected={expected}, diff={}",
            (result - expected).abs()
        );
    }

    #[test]
    fn test_dot_boundary_sizes() {
        let Some((device, queue, precision)) = setup() else {
            return;
        };

        for &n in &[1u32, 255, 256, 257, 512, 513] {
            let ops = GpuVecOps::new(device.clone(), queue.clone(), precision, n);

            let a: Vec<f64> = (0..n).map(|i| 1.0 / (i as f64 + 1.0)).collect();
            let b: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();

            let a_buf = ops.create_vec_buffer("a");
            let b_buf = ops.create_vec_buffer("b");
            ops.upload_vec(&a_buf, &a);
            ops.upload_vec(&b_buf, &b);

            let mut encoder = ops.encoder();
            ops.encode_dot(&mut encoder, &a_buf, &b_buf);
            let result = ops.submit_and_read_scalar(encoder);

            // a[i]*b[i] = 1.0 for all i, so sum = n
            let expected = n as f64;
            let tol = if precision == GpuPrecision::F64 {
                1e-8
            } else {
                expected * 1e-4
            };
            assert!(
                (result - expected).abs() < tol,
                "dot(n={n}): got={result}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_axpy() {
        let Some((device, queue, precision)) = setup() else {
            return;
        };

        let n = 100u32;
        let ops = GpuVecOps::new(device, queue, precision, n);

        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y_init: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let alpha = 2.5;

        let x_buf = ops.create_vec_buffer("x");
        let y_buf = ops.create_vec_buffer("y");
        ops.upload_vec(&x_buf, &x);
        ops.upload_vec(&y_buf, &y_init);

        let mut encoder = ops.encoder();
        ops.encode_axpy(&mut encoder, alpha, &x_buf, &y_buf);
        ops.submit(encoder);

        let y = ops.download_vec(&y_buf);
        let tol = if precision == GpuPrecision::F64 {
            1e-10
        } else {
            1e-3
        };
        for i in 0..n as usize {
            let expected = y_init[i] + alpha * x[i];
            assert!(
                (y[i] - expected).abs() < tol,
                "axpy[{i}]: got={}, expected={expected}",
                y[i]
            );
        }
    }

    #[test]
    fn test_scale() {
        let Some((device, queue, precision)) = setup() else {
            return;
        };

        let n = 100u32;
        let ops = GpuVecOps::new(device, queue, precision, n);

        let x_init: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let alpha = 0.5;

        let x_buf = ops.create_vec_buffer("x");
        ops.upload_vec(&x_buf, &x_init);

        let mut encoder = ops.encoder();
        ops.encode_scale(&mut encoder, alpha, &x_buf);
        ops.submit(encoder);

        let x = ops.download_vec(&x_buf);
        let tol = if precision == GpuPrecision::F64 {
            1e-10
        } else {
            1e-3
        };
        for i in 0..n as usize {
            let expected = x_init[i] * alpha;
            assert!(
                (x[i] - expected).abs() < tol,
                "scale[{i}]: got={}, expected={expected}",
                x[i]
            );
        }
    }
}
