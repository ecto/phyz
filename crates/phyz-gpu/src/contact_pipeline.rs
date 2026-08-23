//! GPU ground contact pipeline.
//!
//! Computes forward kinematics and ground plane contact penalty forces
//! as a compute pass that runs before the ABA shader.

use crate::gpu_state::GpuState;
use crate::shaders::CONTACT_GROUND_SHADER;
use bytemuck::{Pod, Zeroable};
use phyz_math::Vec3;
use phyz_model::{Heightfield, Model};
use std::sync::Arc;

/// Contact parameters for the contact kernels — the WGSL uniform, and the
/// same 32 floats the CUDA pass reads from a buffer. Packed in one place by
/// [`ContactParams::pack`] so the two backends cannot disagree on a slot.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct ContactParams {
    pub(crate) nworld: u32,
    pub(crate) nbodies: u32,
    pub(crate) nv: u32,
    pub(crate) ground_height: f32,
    pub(crate) dt: f32,
    pub(crate) friction: f32,
    /// Number of body-attached contact planes (0 = none).
    pub(crate) nplanes: u32,
    /// Offset, in floats, of the plane table inside the geometry buffer.
    /// Plane records are appended after the collision-instance records so
    /// the pass needs no extra storage binding — the WebGPU baseline allows
    /// eight per stage and this pass already binds eight.
    pub(crate) plane_base: u32,
    pub(crate) _pad_plane0: f32,
    pub(crate) _pad_plane1: f32,
    pub(crate) _pad_plane2: f32,
    // Heightfield terrain header; hf_nx == 0 means "flat plane at
    // ground_height", the pre-heightfield behaviour.
    pub(crate) hf_nx: u32,
    pub(crate) hf_ny: u32,
    pub(crate) hf_ox: f32,
    pub(crate) hf_oy: f32,
    pub(crate) hf_oz: f32,
    pub(crate) hf_cell: f32,
    /// 0 = penalty forces, 1 = velocity-level convex impulse solve.
    pub(crate) solve_mode: u32,
    /// Reserved. Was a sweep index; the shader never read it, and it could not
    /// have worked — all sweeps share one command buffer, and queue writes are
    /// ordered at submission, so the uniform would hold its last value for
    /// every dispatch. Kept as explicit padding rather than removed so the
    /// struct layout the shader expects does not shift.
    pub(crate) _reserved_sweep: u32,
    pub(crate) restitution: f32,
    pub(crate) restitution_threshold: f32,
    pub(crate) solref_erp: f32,
    pub(crate) margin: f32,
    pub(crate) solimp_dmin: f32,
    pub(crate) solimp_dmax: f32,
    pub(crate) solimp_width: f32,
    pub(crate) solimp_mid: f32,
    pub(crate) solimp_power: f32,
    pub(crate) _pad0: f32,
}

impl ContactParams {
    /// Pack the contact parameters for `nworld` worlds of `model`.
    pub(crate) fn pack(
        model: &Model,
        nworld: usize,
        contact: &GroundContactParams,
        planes: &[BodyPlane],
        heightfield: Option<&Heightfield>,
    ) -> Self {
        Self {
            nworld: nworld as u32,
            nbodies: model.nbodies() as u32,
            nv: model.nv as u32,
            ground_height: contact.ground_height as f32,
            dt: model.dt as f32,
            friction: contact.friction as f32,
            nplanes: planes.len() as u32,
            plane_base: (crate::layout::geometry_instance_count(model) * crate::layout::GEOM_STRIDE)
                as u32,
            _pad_plane0: 0.0,
            _pad_plane1: 0.0,
            _pad_plane2: 0.0,
            hf_nx: heightfield.map_or(0, |h| h.nx as u32),
            hf_ny: heightfield.map_or(0, |h| h.ny as u32),
            hf_ox: heightfield.map_or(0.0, |h| h.origin.x as f32),
            hf_oy: heightfield.map_or(0.0, |h| h.origin.y as f32),
            hf_oz: heightfield.map_or(0.0, |h| h.origin.z as f32),
            hf_cell: heightfield.map_or(1.0, |h| h.cell as f32),
            solve_mode: u32::from(contact.impulse_solve),
            _reserved_sweep: 0,
            restitution: contact.restitution as f32,
            restitution_threshold: contact.restitution_threshold as f32,
            solref_erp: contact.solref_erp as f32,
            margin: contact.margin as f32,
            solimp_dmin: contact.solimp_dmin as f32,
            solimp_dmax: contact.solimp_dmax as f32,
            solimp_width: contact.solimp_width as f32,
            solimp_mid: contact.solimp_midpoint as f32,
            solimp_power: contact.solimp_power as f32,
            _pad0: 0.0,
        }
    }

    /// Point the heightfield header at `hf` (the node buffer is separate).
    pub(crate) fn set_heightfield(&mut self, hf: &Heightfield) {
        self.hf_nx = hf.nx as u32;
        self.hf_ny = hf.ny as u32;
        self.hf_ox = hf.origin.x as f32;
        self.hf_oy = hf.origin.y as f32;
        self.hf_oz = hf.origin.z as f32;
        self.hf_cell = hf.cell as f32;
    }

    /// The parameters as the flat f32 buffer the CUDA pass reads.
    #[cfg(any(feature = "cuda", feature = "cuda-host"))]
    pub(crate) fn as_f32s(&self) -> &[f32] {
        bytemuck::cast_slice(std::slice::from_ref(self))
    }
}

pub use crate::layout::{CONTACT_STATE_STRIDE, MAX_CONTACT_PTS};
use crate::layout::{
    GEOM_STRIDE, lightest_collidable_body, no_collidable_geometry_error, pack_geometries,
};

/// Explicit-integration stability factor: a penalty spring of stiffness `k`
/// on a body of mass `m` has natural frequency `w = sqrt(k/m)`, and the
/// semi-implicit Euler step is only reliably stable while `w * dt` stays
/// under roughly this value.
const OMEGA_DT_LIMIT: f64 = 0.3;

/// GPU ground contact pipeline.
pub struct ContactPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: ContactParams,
    params_buffer: wgpu::Buffer,
    _geom_buffer: wgpu::Buffer,
    hf_buffer: wgpu::Buffer,
    /// Node capacity of `hf_buffer`. [`Self::set_heightfield`] can rewrite
    /// terrain up to this many nodes without rebuilding the pipeline —
    /// wgpu buffers cannot grow under a live bind group.
    hf_capacity: usize,
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
    /// Solve the velocity-level convex contact problem instead of applying
    /// penalty forces.
    ///
    /// This is the setting that makes the GPU run the *same contact model* as
    /// `phyz_contact::solve_contacts` rather than a second one: real Coulomb
    /// stiction inside a second-order cone, solref position stabilization,
    /// and no `mg/k` resting sink. `stiffness`/`damping` are unused when it
    /// is on. See the IMPULSE MODE block in `shaders.rs`.
    pub impulse_solve: bool,
    /// Coefficient of restitution, and the approach speed below which it
    /// ramps smoothly to zero. Impulse mode only.
    pub restitution: f64,
    /// Approach speed below which restitution ramps to zero.
    pub restitution_threshold: f64,
    /// solref error-reduction fraction: how much of the current penetration
    /// the reference response removes per step.
    pub solref_erp: f64,
    /// Contact margin: the band above the surface in which a contact is still
    /// detected, its impedance tapering smoothly to zero. Mirrors
    /// `ContactMaterial::margin`.
    pub margin: f64,
    /// SolImp sigmoid, mirroring `phyz_contact::material::SolImp`.
    pub solimp_dmin: f64,
    /// SolImp `dmax`.
    pub solimp_dmax: f64,
    /// SolImp `width`.
    pub solimp_width: f64,
    /// SolImp `midpoint`.
    pub solimp_midpoint: f64,
    /// SolImp `power`.
    pub solimp_power: f64,
}

impl Default for GroundContactParams {
    /// Penalty mode with MuJoCo's contact defaults, so adding the impulse
    /// fields did not change what any existing caller gets.
    ///
    /// `solref_erp` is the *evaluated* `SolRef::error_reduction(dt)` rather
    /// than a time constant, because the shader has no reason to re-derive it
    /// and every reason not to: two independent evaluations of the same
    /// formula is exactly how the CPU and GPU contact models drifted apart in
    /// the first place. Callers that care should set it from their own
    /// `SolRef` at their own `dt` — `GroundContactParams::solref_erp_from` does
    /// that arithmetic.
    fn default() -> Self {
        Self {
            ground_height: 0.0,
            stiffness: 0.0,
            damping: 0.0,
            friction: 0.0,
            impulse_solve: false,
            restitution: 0.0,
            restitution_threshold: 0.05,
            // MuJoCo's SolRef default (timeconst 0.02, dampratio 1.0)
            // evaluated at dt = 1 ms.
            solref_erp: 0.001 / (2.0 * 0.02 + 0.001),
            // MuJoCo's SolImp defaults, and `ContactMaterial`'s margin.
            margin: 0.001,
            solimp_dmin: 0.9,
            solimp_dmax: 0.95,
            solimp_width: 0.001,
            solimp_midpoint: 0.5,
            solimp_power: 2.0,
        }
    }
}

impl GroundContactParams {
    /// `SolRef::error_reduction` — the fraction of the current penetration the
    /// reference response removes in one step of length `dt`.
    ///
    /// Duplicated from `phyz_contact::material::SolRef` because `phyz-gpu`
    /// does not depend on `phyz-contact`, and asserted equal to it in
    /// `tests/contact_impulse_parity.rs` so the duplication cannot rot.
    pub fn solref_erp_from(timeconst: f64, dampratio: f64, dt: f64) -> f64 {
        if dt <= 0.0 {
            return 0.0;
        }
        let tau = timeconst.max(0.0);
        let zeta = dampratio.max(1e-6);
        let denom = 2.0 * tau * zeta * zeta + dt;
        if denom <= 0.0 { 1.0 } else { dt / denom }
    }

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

/// A contact plane welded to a body: the deck of a skateboard, the bed of a
/// truck, any flat top surface other bodies stand on.
///
/// This is deliberately NOT general body-body contact. The plane is the
/// body's local `+Z` face at `offset` along local Z, rectangular; a
/// foot only ever meets a deck's top face, and a moving plane
/// captures that at a fraction of a broad phase's cost. Forces are applied
/// to **both** bodies — the board must feel the rider or it cannot be
/// kicked out from under one.
///
/// # A compound surface is a *set* of these
///
/// The contact pass takes a slice of planes, not one. A skateboard's top is
/// not a single flat face: the kicktail rises at 15 deg beyond the flex
/// hinge, and on this rig the kicktail is a *separate body*. One untilted
/// plane on the deck put every pre-tip stance 6 to 25 mm above the surface it
/// was standing on and on the wrong body (ecto/phyz#82). Two planes — one on
/// the deck, one on the tail body, each with its own [`Self::tilt`] —
/// express it. The CPU reference has no plane at all: it collides the foot's
/// boxes against the deck's boxes through the general narrow phase, and this
/// is the device's cheap stand-in for that.
///
/// Plane contacts do not appear in [`BodyContactState`] readback: those
/// slots are per body and already carry the ground result, and a foot
/// standing on a deck would otherwise overwrite its ground reading with a
/// different surface entirely.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyPlane {
    /// Body index the plane is attached to.
    pub body: usize,
    /// Face offset along the body's local `+Z`, metres (deck half-thickness).
    pub offset: f64,
    /// Penetration beyond which the contact is ignored, metres. Guards a
    /// body approaching from *below* the plane against being captured and
    /// catapulted through it.
    pub max_depth: f64,
    /// Half-extents of the face in the body's own x/y, metres. The plane
    /// only supports what is actually over it.
    ///
    /// A shape's contact points are clamped into the overlap between its own
    /// footprint and this rectangle, and a shape with no overlap at all gets
    /// no contact and falls. Clamping rather than discarding is deliberate: a
    /// deck narrower than a foot still carries that foot, along its edge. The
    /// K1's foot is 22 cm across a 19.7 cm deck — feet point ACROSS a board,
    /// so toes and heel overhang, as a real skater's do — and simply dropping
    /// the overhanging corners deletes the whole roll support. Measured, that
    /// took standing from 3.00 s to 0.8 s in every stance.
    ///
    /// The footprint is an AABB in face coordinates: exact while the shape is
    /// aligned with the face, a slight over-estimate under yaw.
    pub half_x: f64,
    /// Half-extent of the face along the body's own `y`, metres. See
    /// [`Self::half_x`].
    pub half_y: f64,
    /// Bodies that never contact the plane (the plane body itself is always
    /// excluded): wheels and hangers under a deck, for instance.
    ///
    /// Packed as a per-plane bitmask over body indices, so two planes on the
    /// same rig can exclude different bodies. Bounded by
    /// [`crate::layout::MAX_BODIES`], which the contact pass already is.
    pub exclude: Vec<usize>,
    /// Extra rotation of the face inside the body, body -> face, row-major,
    /// in the same sense as [`phyz_model::GeomInstance::origin`]'s `rot`.
    ///
    /// Identity gives the historical behaviour: the face is the body's own
    /// local `+Z`. A tilted face — a kicktail on the deck body, a wedge —
    /// sets this. [`Self::offset`] is measured along the *face's* `+Z`, so
    /// a tilted plane's offset still means "half-thickness".
    pub tilt: phyz_math::Mat3,
    /// In-plane origin of the face inside the body, body coordinates, before
    /// [`Self::offset`] is applied along the face normal. Zero puts the face
    /// centre on the body origin, which is where it used to be.
    pub center: Vec3,
}

impl BodyPlane {
    /// A face on the body's own local `+Z`, `offset` above the body origin —
    /// the shape every caller had before tilted faces existed.
    pub fn flat(body: usize, offset: f64, max_depth: f64, half_x: f64, half_y: f64) -> Self {
        Self {
            body,
            offset,
            max_depth,
            half_x,
            half_y,
            exclude: Vec::new(),
            tilt: phyz_math::Mat3::identity(),
            center: Vec3::zeros(),
        }
    }

    /// Bodies this plane never contacts, as the bitmask the kernels read.
    fn exclude_mask(&self) -> u32 {
        let mut mask = 1u32 << (self.body as u32);
        for &b in &self.exclude {
            mask |= 1u32 << (b as u32);
        }
        mask
    }
}

/// Floats per body-attached plane record, appended to the geometry buffer.
///
/// ```text
/// [0]  body index (bitcast u32)
/// [1]  half_x
/// [2]  half_y
/// [3]  max_depth
/// [4..7] face origin in BODY coordinates (centre + offset along the face normal)
/// [7..16] face rotation, row-major, body -> face
/// [16] exclude bitmask over body indices (bitcast u32)
/// [17..24] reserved
/// ```
///
/// Deliberately the same stride as a geometry record so the two tables share
/// one buffer and the pass keeps its eight storage bindings.
pub const PLANE_STRIDE: usize = GEOM_STRIDE;

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
        Self::with_body_gains(
            device,
            queue,
            model,
            state,
            bodies_buffer,
            contact,
            None,
            &[],
            None,
        )
    }

    /// Create a contact pipeline with optional per-body gains.
    ///
    /// When `body_gains` is `Some`, it must hold one entry per body (in body
    /// order) and overrides the global stiffness/damping for every body.
    #[allow(clippy::too_many_arguments)] // each argument is a distinct GPU resource or contact feature
    pub fn with_body_gains(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        model: &Model,
        state: &GpuState,
        bodies_buffer: &wgpu::Buffer,
        contact: GroundContactParams,
        body_gains: Option<&[BodyContactGains]>,
        planes: &[BodyPlane],
        heightfield: Option<&Heightfield>,
    ) -> Result<Self, String> {
        let nworld = state.nworld;
        let (geom_data, collidable_bodies) =
            pack_contact_geometry(model, &contact, body_gains, planes, heightfield)?;
        let params = ContactParams::pack(model, nworld, &contact, planes, heightfield);

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("contact_params"),
            size: std::mem::size_of::<ContactParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Heightfield node buffer. Always bound (the shader's binding count
        // is fixed); a single placeholder node when there is no terrain,
        // since hf_nx == 0 routes the shader around it anyway. Sized to this
        // first field: `set_heightfield` rewrites in place per training
        // iteration, so enable with the largest grid the run will use.
        let hf_capacity = heightfield.map_or(1, |h| h.heights.len().max(1));
        let hf_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("heightfield_buffer"),
            size: (hf_capacity * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        match heightfield {
            Some(h) => queue.write_buffer(&hf_buffer, 0, bytemuck::cast_slice(&h.heights)),
            None => queue.write_buffer(&hf_buffer, 0, bytemuck::cast_slice(&[0.0f32])),
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
                bgl_storage_rw(6), // contact_state (readback)
                bgl_storage_ro(7), // heightfield nodes
                bgl_storage_ro(8), // qdd (free acceleration from this sweep's ABA)
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
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: hf_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: state.qdd_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            pipeline,
            bind_group,
            params,
            params_buffer,
            _geom_buffer: geom_buffer,
            hf_buffer,
            hf_capacity,
            nworld,
            collidable_bodies,
        })
    }

    /// Replace the terrain without rebuilding the pipeline.
    ///
    /// This is the randomization hook: a training run draws a fresh
    /// heightfield per iteration, and a params write plus a buffer write is
    /// all that costs. The new field may change shape and origin freely as
    /// long as it fits the node capacity allocated at construction — wgpu
    /// buffers cannot grow under a live bind group, so a larger grid needs
    /// the pipeline rebuilt (enable contact with the bigger field first).
    pub fn set_heightfield(&mut self, queue: &wgpu::Queue, hf: &Heightfield) -> Result<(), String> {
        validate_heightfield(hf)?;
        if hf.heights.len() > self.hf_capacity {
            return Err(format!(
                "heightfield has {} nodes but the GPU buffer was sized for {}; \
                 enable contact with the largest grid first, or rebuild the pipeline",
                hf.heights.len(),
                self.hf_capacity
            ));
        }
        self.params.set_heightfield(hf);
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
        queue.write_buffer(&self.hf_buffer, 0, bytemuck::cast_slice(&hf.heights));
        Ok(())
    }

    /// Number of bodies whose geometry the contact pass can collide.
    pub fn collidable_bodies(&self) -> usize {
        self.collidable_bodies
    }

    /// Encode the contact compute pass into a command encoder.
    /// Is this pipeline running the velocity-level impulse solve?
    pub fn impulse_solve(&self) -> bool {
        self.params.solve_mode == 1
    }

    /// Record the contact pass into `encoder`.
    ///
    /// One dispatch is one sweep: the host is expected to interleave this with
    /// the ABA pass, so that the next sweep reads a `qdd` already carrying the
    /// impulses this one wrote.
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

/// Validate a contact configuration and pack its geometry table: per-body
/// shapes and gains, plus the skip-plane flag on the plane's own body and
/// its `exclude` list. Shared by the wgpu and CUDA paths so both reject the
/// same inputs and pack the same bytes. Returns the table and the
/// collidable-body count; errors when that count is zero.
pub(crate) fn pack_contact_geometry(
    model: &Model,
    contact: &GroundContactParams,
    body_gains: Option<&[BodyContactGains]>,
    planes: &[BodyPlane],
    heightfield: Option<&Heightfield>,
) -> Result<(Vec<f32>, usize), String> {
    if let Some(hf) = heightfield {
        validate_heightfield(hf)?;
    }
    if model.nbodies() > crate::layout::MAX_BODIES {
        return Err(format!(
            "contact pass supports at most {} bodies, model has {}",
            crate::layout::MAX_BODIES,
            model.nbodies()
        ));
    }
    for p in planes {
        if p.body >= model.nbodies() {
            return Err(format!(
                "plane body {} out of range for a model with {} bodies",
                p.body,
                model.nbodies()
            ));
        }
        for &b in &p.exclude {
            if b >= model.nbodies() {
                return Err(format!("plane exclude body {b} out of range"));
            }
        }
    }
    if let Some(gains) = body_gains
        && gains.len() != model.nbodies()
    {
        return Err(format!(
            "body_gains has {} entries but the model has {} bodies",
            gains.len(),
            model.nbodies()
        ));
    }
    let (mut geom_data, collidable) = pack_geometries(model, contact, body_gains);
    if collidable == 0 {
        return Err(no_collidable_geometry_error(model));
    }

    // Plane records, appended after the instance records; `plane_base` in
    // ContactParams points here.
    let base = geom_data.len();
    geom_data.resize(base + planes.len() * PLANE_STRIDE, 0.0);
    for (n, p) in planes.iter().enumerate() {
        let b = base + n * PLANE_STRIDE;
        geom_data[b] = f32::from_bits(p.body as u32);
        geom_data[b + 1] = p.half_x as f32;
        geom_data[b + 2] = p.half_y as f32;
        geom_data[b + 3] = p.max_depth as f32;
        // `tilt` is body -> face, so the face's own +Z in body coordinates is
        // the transpose's third column, i.e. `tilt` row 2. Offset rides along
        // that, which is what keeps "offset = half-thickness" true under tilt.
        let nz = Vec3::new(p.tilt[(2, 0)], p.tilt[(2, 1)], p.tilt[(2, 2)]);
        let origin = p.center + nz * p.offset;
        geom_data[b + 4] = origin.x as f32;
        geom_data[b + 5] = origin.y as f32;
        geom_data[b + 6] = origin.z as f32;
        for r in 0..3 {
            for c in 0..3 {
                geom_data[b + 7 + r * 3 + c] = p.tilt[(r, c)] as f32;
            }
        }
        geom_data[b + 16] = f32::from_bits(p.exclude_mask());
    }
    Ok((geom_data, collidable))
}

/// A malformed heightfield fails loudly at upload rather than as an
/// out-of-bounds read in the shader, which wgpu clamps to whatever is in
/// range — i.e. terrain that silently is not what the caller built.
pub(crate) fn validate_heightfield(hf: &Heightfield) -> Result<(), String> {
    if hf.nx == 0 || hf.ny == 0 {
        return Err("heightfield needs at least one node per axis".into());
    }
    if hf.heights.len() != hf.nx * hf.ny {
        return Err(format!(
            "heightfield claims {}x{} nodes but carries {} heights",
            hf.nx,
            hf.ny,
            hf.heights.len()
        ));
    }
    if !(hf.cell.is_finite() && hf.cell > 0.0) {
        return Err(format!(
            "heightfield cell must be positive, got {}",
            hf.cell
        ));
    }
    Ok(())
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
    use phyz_model::Geometry;
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
            ..Default::default()
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
            ..Default::default()
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
        // The table is instance-indexed now: a body with no shapes gets no
        // record at all, and its range in the body table is empty.
        assert_eq!(data.len(), GEOM_STRIDE);
        let ranges = crate::layout::geometry_ranges(&model);
        assert_eq!(ranges, vec![(0, 1), (1, 0)]);
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
            ..Default::default()
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
