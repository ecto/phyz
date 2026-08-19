//! Device-agnostic buffer layouts shared by every kernel backend.
//!
//! The wgpu (WGSL) path and the CUDA path run the same physics on the same
//! flat `f32` buffers; only the kernel language and the launch plumbing
//! differ. Everything that decides *what bytes go where* — how a `Model` is
//! flattened into the body table, how geometry and penalty gains are packed,
//! how `State`s are laid out per world — lives here so both backends read one
//! definition and cannot drift apart.
//!
//! Every stride and slot documented here is mirrored verbatim in
//! [`crate::shaders`] (WGSL) and in `cuda/phyz_kernels.cu` (CUDA C).

use phyz_math::Vec3;
use phyz_model::{Geometry, Model, State};

use crate::contact_pipeline::{BodyContactGains, GroundContactParams};
use crate::pd_pipeline::PdDof;

/// Floats per body in the packed body table.
///
/// ```text
/// [0]  parent (bitcast i32, -1 for root)
/// [1]  joint_type (0=revolute, 1=prismatic, 2=fixed, 3=ball, 4=free)
/// [2]  q_offset (bitcast u32)
/// [3]  v_offset (bitcast u32)
/// [4]  mass
/// [5..8]  com (x,y,z)
/// [8..14] inertia (xx,yy,zz,xy,xz,yz)
/// [14..23] ptj rotation (row-major 3x3)
/// [23..26] ptj translation (x,y,z)
/// [26..29] axis (x,y,z)
/// [29] damping
/// [30] passive spring stiffness (single-DOF joints; see joint.rs passive_force)
/// [31] spring reference angle
/// [32] armature (rotor inertia, added to the D diagonal like the CPU's crba/aba)
/// [33..36] padding
/// ```
pub const BODY_STRIDE: usize = 36;

/// Floats per body in the packed geometry table.
///
/// ```text
/// [0]  geom_type (0=none, 1=sphere, 2=box, 3=capsule, 4=cylinder, 5=mesh)
/// [1]  param0 (radius for sphere/capsule/cylinder, half_x for box, aabb min_x for mesh)
/// [2]  param1 (length for capsule, half_y for box, height for cylinder, aabb min_y for mesh)
/// [3]  param2 (half_z for box, aabb min_z for mesh)
/// [4..7] aabb max (x,y,z) for mesh
/// [7]  skip-plane flag (1.0 = this body never contacts the attached plane)
/// [8]  contact stiffness (per body)
/// [9]  contact damping (per body)
/// [10..13] instance origin position, body frame
/// [13..22] instance origin rotation, row-major (body -> shape coordinates)
/// [22..24] reserved
/// ```
///
/// There is no separate "carried mass" override here, and deliberately so:
/// the penalty gains are already per body (slots 8 and 9), so a foot that holds
/// up a whole robot is expressed by giving that foot the stiffness it needs
/// rather than by naming a mass the kernel then divides. See
/// [`BodyContactGains::uniform_frequency`] for the mass-proportional recipe
/// that makes one setting work across a robot.
pub const GEOM_STRIDE: usize = 24;

/// Contact slots per body, matching `MAX_PTS` in the kernels: a box contacts
/// through all eight corners, every other shape through one.
pub const MAX_CONTACT_PTS: usize = 8;

/// Floats per body in the contact-state buffer.
///
/// ```text
/// [0]  touching (1.0 while penetrating, else 0.0)
/// [1]  penetration depth, metres
/// [2..5] contact point, world frame (x, y, z)
/// [5..8] contact force, world frame (x, y, z)
/// [8..32]  warm-start impulses, ground/terrain contacts (MAX_CONTACT_PTS vec3s)
/// [32..56] warm-start impulses, body-attached face
/// ```
/// Two impulse blocks because a body can touch both the ground and the face
/// in the same step and must not share a warm-start slot. Matches `CS_STRIDE`
/// in the kernels.
pub const CONTACT_STATE_STRIDE: usize = 8 + 2 * MAX_CONTACT_PTS * 3;

/// Floats per servoed DOF in the PD table.
///
/// ```text
/// [0] q_index (as f32; exact for any index < 2^24)
/// [1] v_index
/// [2] kp
/// [3] kd
/// [4] max_force
/// [5..8] reserved
/// ```
pub const PD_DOF_STRIDE: usize = 8;

/// Largest body count a per-world kernel thread can hold in private storage.
///
/// Both kernel languages size their per-thread scratch arrays to this; a
/// model with more bodies must be rejected on the host rather than silently
/// indexing past the arrays on device.
pub const MAX_BODIES: usize = 32;

/// Pack model bodies into a flat f32 array (see [`BODY_STRIDE`]).
pub fn pack_bodies(model: &Model) -> Vec<f32> {
    let nb = model.nbodies();
    let mut data = vec![0.0f32; nb * BODY_STRIDE];

    for (i, body) in model.bodies.iter().enumerate() {
        let base = i * BODY_STRIDE;
        let joint = &model.joints[body.joint_idx];

        data[base] = f32::from_bits(body.parent as u32);
        let jtype: u32 = match joint.joint_type {
            phyz_model::JointType::Revolute | phyz_model::JointType::Hinge => 0,
            phyz_model::JointType::Prismatic | phyz_model::JointType::Slide => 1,
            phyz_model::JointType::Fixed => 2,
            phyz_model::JointType::Spherical | phyz_model::JointType::Ball => 3,
            phyz_model::JointType::Free => 4,
        };
        data[base + 1] = f32::from_bits(jtype);
        data[base + 2] = f32::from_bits(model.q_offsets[body.joint_idx] as u32);
        data[base + 3] = f32::from_bits(model.v_offsets[body.joint_idx] as u32);
        data[base + 4] = body.inertia.mass as f32;
        data[base + 5] = body.inertia.com.x as f32;
        data[base + 6] = body.inertia.com.y as f32;
        data[base + 7] = body.inertia.com.z as f32;
        data[base + 8] = body.inertia.inertia[(0, 0)] as f32;
        data[base + 9] = body.inertia.inertia[(1, 1)] as f32;
        data[base + 10] = body.inertia.inertia[(2, 2)] as f32;
        data[base + 11] = body.inertia.inertia[(0, 1)] as f32;
        data[base + 12] = body.inertia.inertia[(0, 2)] as f32;
        data[base + 13] = body.inertia.inertia[(1, 2)] as f32;
        let r = &joint.parent_to_joint.rot;
        for row in 0..3 {
            for col in 0..3 {
                data[base + 14 + row * 3 + col] = r[(row, col)] as f32;
            }
        }
        data[base + 23] = joint.parent_to_joint.pos.x as f32;
        data[base + 24] = joint.parent_to_joint.pos.y as f32;
        data[base + 25] = joint.parent_to_joint.pos.z as f32;
        data[base + 26] = joint.axis.x as f32;
        data[base + 27] = joint.axis.y as f32;
        data[base + 28] = joint.axis.z as f32;
        data[base + 29] = joint.damping as f32;
        // [30..31] passive spring, the truck-bushing term. Packed here
        // because the ABA pass applies it explicitly, exactly as the CPU's
        // `Joint::passive_force` does — same clause, same sign.
        data[base + 30] = joint.stiffness as f32;
        data[base + 31] = joint.spring_ref as f32;
        data[base + 32] = joint.armature as f32;
        // [33..36] padding
    }

    data
}

/// The kernel geometry type code for a body's primary collision shape.
pub fn gpu_geom_type(geometry: Option<&Geometry>) -> u32 {
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
pub fn lightest_collidable_body(model: &Model) -> Option<(&str, f64)> {
    model
        .bodies
        .iter()
        .filter(|b| gpu_geom_type(b.geometry.as_ref()) != 0 && b.inertia.mass > 0.0)
        .map(|b| (b.name.as_str(), b.inertia.mass))
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

/// Pack body geometry data into a flat f32 array (see [`GEOM_STRIDE`]).
///
/// Returns the packed data and the number of bodies with collidable geometry.
pub fn pack_geometries(
    model: &Model,
    contact: &GroundContactParams,
    body_gains: Option<&[BodyContactGains]>,
) -> (Vec<f32>, usize) {
    let nb = model.nbodies();
    let mut data = vec![0.0f32; nb * GEOM_STRIDE];
    let mut collidable = 0;

    for (i, body) in model.bodies.iter().enumerate() {
        let base = i * GEOM_STRIDE;

        // Identity instance origin unless a collision instance carries one.
        data[base + 13] = 1.0;
        data[base + 17] = 1.0;
        data[base + 21] = 1.0;

        // Prefer `Body::collisions[0]` — offsets included, which is where a
        // K1 rig mounts its foot pads (2.6 cm forward of the ankle) —
        // falling back to the legacy centred `Body::geometry`. Bodies with
        // more than one collision instance contact through their FIRST
        // instance only on the GPU; that is a documented fidelity gap, not a
        // silent one, and the CPU impulse path remains the referee.
        let (geom, origin) = match body.collisions.first() {
            Some(inst) => (Some(&inst.geometry), Some(&inst.origin)),
            None => (body.geometry.as_ref(), None),
        };
        if let Some(o) = origin {
            data[base + 10] = o.pos.x as f32;
            data[base + 11] = o.pos.y as f32;
            data[base + 12] = o.pos.z as f32;
            for r in 0..3 {
                for c in 0..3 {
                    data[base + 13 + r * 3 + c] = o.rot[(r, c)] as f32;
                }
            }
        }

        match geom {
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
                // Body-frame AABB; the kernel takes the lowest of its eight
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

/// The error message for a contact pass that can collide nothing.
pub fn no_collidable_geometry_error(model: &Model) -> String {
    let skipped: Vec<&str> = model
        .bodies
        .iter()
        .filter(|b| gpu_geom_type(b.geometry.as_ref()) == 0)
        .map(|b| b.name.as_str())
        .collect();
    format!(
        "ground contact enabled but no body has GPU-collidable geometry \
         (supported: sphere, box, capsule, cylinder, mesh); \
         bodies without a supported shape: [{}]",
        skipped.join(", ")
    )
}

/// Validate servoed DOFs against the model's `q`/`v` extents.
pub fn check_pd_dofs(dofs: &[PdDof], nq: usize, nv: usize) -> Result<(), String> {
    if dofs.is_empty() {
        return Err("PD pipeline needs at least one servoed DOF".into());
    }
    for d in dofs {
        if d.q_index >= nq || d.v_index >= nv {
            return Err(format!(
                "PD DOF out of range: q_index {} (nq {}), v_index {} (nv {})",
                d.q_index, nq, d.v_index, nv
            ));
        }
    }
    Ok(())
}

/// Pack servoed DOFs into a flat f32 table (see [`PD_DOF_STRIDE`]).
pub fn pack_pd_dofs(dofs: &[PdDof]) -> Vec<f32> {
    let mut dof_data = vec![0.0f32; dofs.len() * PD_DOF_STRIDE];
    for (i, d) in dofs.iter().enumerate() {
        let b = i * PD_DOF_STRIDE;
        dof_data[b] = d.q_index as f32;
        dof_data[b + 1] = d.v_index as f32;
        dof_data[b + 2] = d.kp as f32;
        dof_data[b + 3] = d.kd as f32;
        dof_data[b + 4] = d.max_force as f32;
    }
    dof_data
}

/// Flatten `rows[world][k]` into `nworld * width` f32, zero-padding short or
/// missing rows and truncating long ones.
pub fn pack_rows(rows: &[Vec<f64>], nworld: usize, width: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; nworld * width];
    for (w, row) in rows.iter().enumerate().take(nworld) {
        for (k, &val) in row.iter().enumerate().take(width) {
            data[w * width + k] = val as f32;
        }
    }
    data
}

/// Flatten states into per-world `q`, `v` and `ctrl` arrays.
pub fn pack_states(states: &[State], nq: usize, nv: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let nworld = states.len();
    let mut q_data = vec![0.0f32; nworld * nq];
    let mut v_data = vec![0.0f32; nworld * nv];
    let mut ctrl_data = vec![0.0f32; nworld * nv];
    for (i, state) in states.iter().enumerate() {
        for j in 0..nq {
            q_data[i * nq + j] = state.q[j] as f32;
        }
        for j in 0..nv {
            v_data[i * nv + j] = state.v[j] as f32;
            ctrl_data[i * nv + j] = state.ctrl[j] as f32;
        }
    }
    (q_data, v_data, ctrl_data)
}

/// Rebuild `nworld` states from flat `q`/`v` arrays.
pub fn unpack_states(model: &Model, nworld: usize, q_data: &[f32], v_data: &[f32]) -> Vec<State> {
    let (nq, nv) = (model.nq, model.nv);
    (0..nworld)
        .map(|i| {
            let mut state = model.default_state();
            for j in 0..nq {
                state.q[j] = q_data[i * nq + j] as f64;
            }
            for j in 0..nv {
                state.v[j] = v_data[i * nv + j] as f64;
            }
            state
        })
        .collect()
}

/// Rebuild `result[env][body]` contact states from the flat contact buffer.
pub fn unpack_contacts(
    data: &[f32],
    nworld: usize,
    nbodies: usize,
) -> Vec<Vec<crate::contact_pipeline::BodyContactState>> {
    let stride = CONTACT_STATE_STRIDE;
    (0..nworld)
        .map(|env| {
            (0..nbodies)
                .map(|body| {
                    let base = (env * nbodies + body) * stride;
                    let s = &data[base..base + stride];
                    crate::contact_pipeline::BodyContactState {
                        touching: s[0] != 0.0,
                        penetration: s[1] as f64,
                        point: Vec3::new(s[2] as f64, s[3] as f64, s[4] as f64),
                        force: Vec3::new(s[5] as f64, s[6] as f64, s[7] as f64),
                    }
                })
                .collect()
        })
        .collect()
}
