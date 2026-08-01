//! GPU PD position-servo pipeline.
//!
//! Computes per-joint PD control torques as a compute pass that runs before
//! the ABA shader, writing directly into the `ctrl` buffer. This is what
//! lets a batched RL environment command *position targets* (the action
//! space every locomotion policy uses) instead of raw generalized forces,
//! without a host round-trip per substep.
//!
//! The torque law is exactly the CPU servo's
//! (`MotorTarget::compute_torque`, Position mode, zero feedforward):
//!
//! ```text
//! tau = clamp(kp * (target - q) - kd * v, -max_force, max_force)
//! ```
//!
//! Deliberately **no gravity feedforward**: on a floating base the
//! bolted-root RNEA feedforward injects the ground's support wrench into the
//! joints (measured on the Booster K1: 90° of trunk tilt in 0.22 s), so the
//! CPU path gates it off there too. Plain PD is what MuJoCo/Isaac position
//! actuators run.
//!
//! Only single-DOF joints are servoed. A free or spherical joint has no
//! meaningful scalar target; its DOFs keep whatever `set_controls` wrote
//! (zero by default). This mirrors the RL seam, where the floating base is
//! unactuated by construction.

use crate::gpu_state::GpuState;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// One servoed DOF: where it lives in `q`/`v`, and its gains.
#[derive(Debug, Clone, Copy)]
pub struct PdDof {
    /// Index into an environment's `q` slice.
    pub q_index: usize,
    /// Index into an environment's `v` (and `ctrl`) slice.
    pub v_index: usize,
    /// Position gain (N·m/rad or N/m).
    pub kp: f64,
    /// Damping gain (N·m·s/rad or N·s/m).
    pub kd: f64,
    /// Torque/force clamp. The CPU servo defaults this to `kp·π` when the
    /// joint has no authored effort limit; callers should do the same.
    pub max_force: f64,
}

/// Uniform parameters for the PD shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PdParams {
    nworld: u32,
    nq: u32,
    nv: u32,
    n_dofs: u32,
}

/// Packed per-DOF data (8 f32 per entry).
///
/// ```text
/// [0] q_index (as f32; exact for any index < 2^24)
/// [1] v_index
/// [2] kp
/// [3] kd
/// [4] max_force
/// [5..8] reserved
/// ```
const DOF_STRIDE: usize = 8;

/// WGSL source for the PD servo pass.
///
/// One thread per (environment, servoed DOF). Reads `q`/`v`, writes `ctrl`.
const PD_SHADER: &str = r#"
struct PdParams {
    nworld: u32,
    nq: u32,
    nv: u32,
    n_dofs: u32,
}

@group(0) @binding(0) var<uniform> params: PdParams;
@group(0) @binding(1) var<storage, read> dofs: array<f32>;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> targets: array<f32>;
@group(0) @binding(5) var<storage, read_write> ctrl: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.nworld * params.n_dofs;
    if (idx >= total) {
        return;
    }
    let world = idx / params.n_dofs;
    let d = idx % params.n_dofs;
    let base = d * 8u;
    let q_index = u32(dofs[base]);
    let v_index = u32(dofs[base + 1u]);
    let kp = dofs[base + 2u];
    let kd = dofs[base + 3u];
    let max_force = dofs[base + 4u];

    let qj = q[world * params.nq + q_index];
    let vj = v[world * params.nv + v_index];
    let tgt = targets[world * params.n_dofs + d];
    let tau = kp * (tgt - qj) - kd * vj;
    ctrl[world * params.nv + v_index] = clamp(tau, -max_force, max_force);
}
"#;

/// GPU PD servo pipeline.
pub struct PdPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    /// Per-env position targets, `nworld × n_dofs` f32.
    targets_buffer: wgpu::Buffer,
    _params_buffer: wgpu::Buffer,
    _dofs_buffer: wgpu::Buffer,
    nworld: usize,
    n_dofs: usize,
}

impl PdPipeline {
    /// Create a PD pipeline servoing `dofs`.
    pub fn new(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        state: &GpuState,
        dofs: &[PdDof],
    ) -> Result<Self, String> {
        if dofs.is_empty() {
            return Err("PD pipeline needs at least one servoed DOF".into());
        }
        for d in dofs {
            if d.q_index >= state.nq || d.v_index >= state.nv {
                return Err(format!(
                    "PD DOF out of range: q_index {} (nq {}), v_index {} (nv {})",
                    d.q_index, state.nq, d.v_index, state.nv
                ));
            }
        }
        let nworld = state.nworld;
        let n_dofs = dofs.len();

        let params = PdParams {
            nworld: nworld as u32,
            nq: state.nq as u32,
            nv: state.nv as u32,
            n_dofs: n_dofs as u32,
        };
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pd_params"),
            size: std::mem::size_of::<PdParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        let mut dof_data = vec![0.0f32; n_dofs * DOF_STRIDE];
        for (i, d) in dofs.iter().enumerate() {
            let b = i * DOF_STRIDE;
            dof_data[b] = d.q_index as f32;
            dof_data[b + 1] = d.v_index as f32;
            dof_data[b + 2] = d.kp as f32;
            dof_data[b + 3] = d.kd as f32;
            dof_data[b + 4] = d.max_force as f32;
        }
        let dofs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pd_dofs"),
            size: (dof_data.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&dofs_buffer, 0, bytemuck::cast_slice(&dof_data));

        let targets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pd_targets"),
            size: ((nworld * n_dofs).max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pd_shader"),
            source: wgpu::ShaderSource::Wgsl(PD_SHADER.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pd_bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                bgl_storage_ro(2),
                bgl_storage_ro(3),
                bgl_storage_ro(4),
                bgl_storage_rw(5),
            ],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pd_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pd_pl"),
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
            label: Some("pd_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dofs_buffer.as_entire_binding(),
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
                    resource: targets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: state.ctrl_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            pipeline,
            bind_group,
            targets_buffer,
            _params_buffer: params_buffer,
            _dofs_buffer: dofs_buffer,
            nworld,
            n_dofs,
        })
    }

    /// Upload per-environment position targets (`targets[env][dof]`, in the
    /// order the pipeline's `PdDof` list was given).
    pub fn set_targets(&self, queue: &wgpu::Queue, targets: &[Vec<f64>]) {
        let mut data = vec![0.0f32; self.nworld * self.n_dofs];
        for (w, t) in targets.iter().enumerate().take(self.nworld) {
            for (d, &val) in t.iter().enumerate().take(self.n_dofs) {
                data[w * self.n_dofs + d] = val as f32;
            }
        }
        queue.write_buffer(&self.targets_buffer, 0, bytemuck::cast_slice(&data));
    }

    /// Encode the PD pass. Must run before the ABA pass that consumes `ctrl`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pd_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let threads = (self.nworld * self.n_dofs) as u32;
        pass.dispatch_workgroups(threads.div_ceil(64), 1, 1);
    }

    /// Number of servoed DOFs per environment.
    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }
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

#[cfg(test)]
mod tests {
    use crate::GpuBatchSimulator;
    use crate::pd_pipeline::PdDof;
    use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
    use phyz_model::ModelBuilder;
    use phyz_rigid::aba;

    /// GPU PD trajectory must match the CPU reference: same servo law
    /// (`clamp(kp·(target−q) − kd·v)`), same ABA, same semi-implicit Euler.
    /// A servoed pendulum driven toward a target for 100 steps is the whole
    /// contract in one number — any divergence in the torque law, the clamp,
    /// or the DOF indexing shows up as a growing q gap.
    #[test]
    fn pd_servo_trajectory_matches_cpu_reference() {
        let inertia = SpatialInertia::new(1.0, Vec3::new(0.0, 0.0, -0.5), Mat3::identity() * 0.1);
        let model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -9.81))
            .dt(0.002)
            .add_revolute_body("link", -1, SpatialTransform::identity(), inertia)
            .build();

        let (kp, kd, max_force, target) = (30.0, 2.0, 25.0, 0.8);

        let Ok(mut sim) = GpuBatchSimulator::new(model.clone(), 4) else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        sim.enable_pd_control(&[PdDof {
            q_index: 0,
            v_index: 0,
            kp,
            kd,
            max_force,
        }])
        .unwrap();
        sim.set_position_targets(&vec![vec![target]; 4]).unwrap();

        let mut init = model.default_state();
        init.q[0] = 0.3;
        sim.load_states(&vec![init.clone(); 4]);

        // CPU reference, f64.
        let mut cpu = init.clone();
        for _ in 0..100 {
            let tau = (kp * (target - cpu.q[0]) - kd * cpu.v[0]).clamp(-max_force, max_force);
            cpu.ctrl[0] = tau;
            let qdd = aba(&model, &cpu);
            cpu.v[0] += model.dt * qdd[0];
            cpu.q[0] += model.dt * cpu.v[0];
        }

        for _ in 0..100 {
            sim.step();
        }
        let gpu = sim.readback_states();

        // f32 state + f32 arithmetic accumulates over 100 steps; 1e-3 rad
        // on a trajectory spanning ~0.5 rad is well past any sign or
        // indexing error while leaving room for precision drift.
        for (w, s) in gpu.iter().enumerate() {
            assert!(
                (s.q[0] - cpu.q[0]).abs() < 1e-3,
                "env {w}: gpu q {} vs cpu q {} (diff {:.2e})",
                s.q[0],
                cpu.q[0],
                (s.q[0] - cpu.q[0]).abs()
            );
        }

        // And the servo actually pulled toward the target.
        assert!(
            (gpu[0].q[0] - 0.3).abs() > 0.05,
            "servo did not move the joint"
        );
    }

    /// Out-of-range DOF indices must be a construction error, not a silent
    /// out-of-bounds read in the shader.
    #[test]
    fn pd_rejects_out_of_range_dofs() {
        let inertia = SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.1);
        let model = ModelBuilder::new()
            .add_revolute_body("link", -1, SpatialTransform::identity(), inertia)
            .build();
        let Ok(mut sim) = GpuBatchSimulator::new(model, 1) else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let err = sim.enable_pd_control(&[PdDof {
            q_index: 5,
            v_index: 0,
            kp: 1.0,
            kd: 0.0,
            max_force: 1.0,
        }]);
        assert!(err.is_err(), "out-of-range q_index accepted");
    }
}
