//! Contact detection and force computation.

use crate::material::ContactMaterial;
use phyz_collision::{AABB, Collision, gjk_distance_rot, sweep_and_prune};
use phyz_math::{SpatialVec, Vec3};
use phyz_model::{Geometry as ModelGeometry, Model, State};

/// Convert phyz_model::Geometry to phyz_collision::Geometry.
fn convert_geometry(g: &ModelGeometry) -> phyz_collision::Geometry {
    match g {
        ModelGeometry::Sphere { radius } => phyz_collision::Geometry::Sphere { radius: *radius },
        ModelGeometry::Capsule { radius, length } => phyz_collision::Geometry::Capsule {
            radius: *radius,
            length: *length,
        },
        ModelGeometry::Box { half_extents } => phyz_collision::Geometry::Box {
            half_extents: *half_extents,
        },
        ModelGeometry::Cylinder { radius, height } => phyz_collision::Geometry::Cylinder {
            radius: *radius,
            height: *height,
        },
        ModelGeometry::Mesh { vertices, faces } => phyz_collision::Geometry::Mesh {
            vertices: vertices.clone(),
            faces: faces.clone(),
        },
        ModelGeometry::Plane { normal } => phyz_collision::Geometry::Plane { normal: *normal },
    }
}

/// Find all contacts in the current state.
pub fn find_contacts(
    _model: &Model,
    state: &State,
    geometries: &[Option<ModelGeometry>],
) -> Vec<Collision> {
    let mut contacts = Vec::new();

    // Build AABBs for broad phase
    let mut aabbs = Vec::new();
    for (i, geom_opt) in geometries.iter().enumerate() {
        if let Some(geom) = geom_opt {
            let xform = &state.body_xform[i];
            let pos = xform.pos;
            let rot = xform.rot;
            let collision_geom = convert_geometry(geom);
            let aabb = AABB::from_geometry(&collision_geom, &pos, &rot);
            aabbs.push(aabb);
        } else {
            // No geometry for this body
            aabbs.push(AABB::new(Vec3::zeros(), Vec3::zeros()));
        }
    }

    // Broad phase: find potentially colliding pairs
    let pairs = sweep_and_prune(&aabbs);

    // Narrow phase: GJK/EPA for each pair
    for (i, j) in pairs {
        if let (Some(geom_i), Some(geom_j)) = (&geometries[i], &geometries[j]) {
            let xform_i = &state.body_xform[i];
            let xform_j = &state.body_xform[j];
            let pos_i = xform_i.pos;
            let pos_j = xform_j.pos;
            let rot_i = xform_i.rot;
            let rot_j = xform_j.rot;

            let collision_geom_i = convert_geometry(geom_i);
            let collision_geom_j = convert_geometry(geom_j);
            let dist = gjk_distance_rot(
                &collision_geom_i,
                &collision_geom_j,
                &pos_i,
                &pos_j,
                &rot_i,
                &rot_j,
            );

            if dist < 0.0 {
                let normal = (pos_j - pos_i).normalize();
                let contact_point = (pos_i + pos_j) * 0.5;
                let penetration_depth = -dist;

                contacts.push(Collision {
                    body_i: i,
                    body_j: j,
                    contact_point,
                    contact_normal: normal,
                    penetration_depth,
                });
            }
        }
    }

    contacts
}

/// Compute contact forces for all contacts using body spatial velocities.
///
/// `body_velocities` should come from forward_kinematics (linear part of spatial velocity).
/// Returns spatial forces for each body in body frame.
pub fn contact_forces(
    contacts: &[Collision],
    state: &State,
    materials: &[ContactMaterial],
    body_velocities: Option<&[SpatialVec]>,
) -> Vec<SpatialVec> {
    let nbodies = state.body_xform.len();
    let mut forces = vec![SpatialVec::zero(); nbodies];

    for contact in contacts {
        let i = contact.body_i;
        let j = contact.body_j;

        // Get material (use first body's material, or default)
        let material = if materials.is_empty() {
            &ContactMaterial::default()
        } else {
            &materials[i.min(materials.len() - 1)]
        };

        // Extract linear velocities from spatial velocities
        let vel_i = body_velocities
            .and_then(|vels| vels.get(i))
            .map(|v| v.linear)
            .unwrap_or(Vec3::zeros());

        let vel_j = if j == usize::MAX {
            // Ground — zero velocity
            Vec3::zeros()
        } else {
            body_velocities
                .and_then(|vels| vels.get(j))
                .map(|v| v.linear)
                .unwrap_or(Vec3::zeros())
        };

        // Compute force
        let force = crate::compute_contact_force(contact, material, &vel_i, &vel_j);

        // Apply equal and opposite forces
        forces[i] = forces[i] + force;
        if j != usize::MAX {
            forces[j] = forces[j] - force;
        }
    }

    forces
}

/// Find ground plane contacts.
pub fn find_ground_contacts(
    state: &State,
    geometries: &[Option<ModelGeometry>],
    ground_height: f64,
) -> Vec<Collision> {
    let mut contacts = Vec::new();

    for (i, geom_opt) in geometries.iter().enumerate() {
        if let Some(geom) = geom_opt {
            let xform = &state.body_xform[i];
            let pos = xform.pos;

            // Check if body is below ground
            let min_z = match geom {
                ModelGeometry::Sphere { radius } => pos.z - radius,
                ModelGeometry::Box { half_extents } => pos.z - half_extents.z,
                ModelGeometry::Capsule { radius, length } => pos.z - length * 0.5 - radius,
                ModelGeometry::Cylinder { height, .. } => pos.z - height * 0.5,
                _ => pos.z,
            };

            if min_z < ground_height {
                let penetration = ground_height - min_z;
                contacts.push(Collision {
                    body_i: i,
                    body_j: usize::MAX, // Ground is not a body
                    contact_point: Vec3::new(pos.x, pos.y, ground_height),
                    contact_normal: Vec3::z(),
                    penetration_depth: penetration,
                });
            }
        }
    }

    contacts
}
