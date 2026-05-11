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

    // Build AABBs for broad phase. If a body's transform contains any
    // non-finite component (NaN/inf — e.g. left over from an upstream blowup),
    // emit a NaN-tagged AABB. The broad phase will skip it via its own
    // finiteness filter, but emitting it keeps the index alignment with
    // `geometries`/`state.body_xform`.
    let mut aabbs = Vec::new();
    for (i, geom_opt) in geometries.iter().enumerate() {
        if let Some(geom) = geom_opt {
            let xform = &state.body_xform[i];
            let pos = xform.pos;
            let rot = xform.rot;
            if !pos_is_finite(&pos) || !rot_is_finite(&rot) {
                aabbs.push(AABB::new(
                    Vec3::new(f64::NAN, f64::NAN, f64::NAN),
                    Vec3::new(f64::NAN, f64::NAN, f64::NAN),
                ));
                continue;
            }
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

            // Defensive: even if the broad phase let through a body with a
            // NaN transform we refuse to produce a contact with a NaN normal.
            if !pos_is_finite(&pos_i) || !pos_is_finite(&pos_j) {
                continue;
            }

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
                // Default: take the normal from the body centres. When the
                // centres coincide (or are within numerical noise) that
                // division gives NaN, so we fall back to an EPA-derived
                // normal, then to +Z. If even EPA can't agree on a
                // direction we drop the contact rather than seeding NaN.
                let center_offset = pos_j - pos_i;
                let normal = if center_offset.norm() > 1e-9 {
                    center_offset.normalize()
                } else if let Some((_, epa_normal)) = phyz_collision::epa_penetration_rot(
                    &collision_geom_i,
                    &collision_geom_j,
                    &pos_i,
                    &pos_j,
                    &rot_i,
                    &rot_j,
                ) {
                    if pos_is_finite(&epa_normal) && epa_normal.norm() > 1e-9 {
                        epa_normal.normalize()
                    } else {
                        Vec3::z()
                    }
                } else {
                    Vec3::z()
                };

                if !pos_is_finite(&normal) {
                    continue;
                }

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

fn pos_is_finite(v: &Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

fn rot_is_finite(m: &phyz_math::Mat3) -> bool {
    for i in 0..3 {
        for j in 0..3 {
            if !m[(i, j)].is_finite() {
                return false;
            }
        }
    }
    true
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

        // Apply equal and opposite forces.
        //
        // `compute_contact_force` returns `normal * magnitude`. Conventions for
        // the normal differ between the two contact sources:
        //
        // * `find_ground_contacts` sets `contact_normal = +z`, which is the
        //   direction we need to push body i (away from the ground). Adding
        //   the returned wrench directly is therefore correct.
        // * `find_contacts` sets `contact_normal = (pos_j - pos_i).normalize()`
        //   — i.e. pointing from i toward j. To separate the pair we must
        //   push body i in the OPPOSITE direction (-normal) and body j along
        //   +normal. Naively adding `+force` to body i and `-force` to body j
        //   pushes them INTO each other, which is what we observe before this
        //   fix.
        if j == usize::MAX {
            forces[i] = forces[i] + force;
        } else {
            forces[i] = forces[i] - force;
            forces[j] = forces[j] + force;
        }
    }

    forces
}

/// Compute contact forces with implicit damping for low-mass-body stability.
///
/// See [`crate::compute_contact_force_implicit`] for the derivation. Pass
/// `f64::INFINITY` in `masses[i]` for fixed/world bodies. For ground contacts
/// (`body_j == usize::MAX`) the ground is treated as having infinite mass.
pub fn contact_forces_implicit(
    contacts: &[Collision],
    state: &State,
    materials: &[ContactMaterial],
    body_velocities: Option<&[SpatialVec]>,
    masses: &[f64],
    dt: f64,
) -> Vec<SpatialVec> {
    let nbodies = state.body_xform.len();
    let mut forces = vec![SpatialVec::zero(); nbodies];

    for contact in contacts {
        let i = contact.body_i;
        let j = contact.body_j;

        let material = if materials.is_empty() {
            &ContactMaterial::default()
        } else {
            &materials[i.min(materials.len() - 1)]
        };

        let vel_i = body_velocities
            .and_then(|vels| vels.get(i))
            .map(|v| v.linear)
            .unwrap_or(Vec3::zeros());

        let vel_j = if j == usize::MAX {
            Vec3::zeros()
        } else {
            body_velocities
                .and_then(|vels| vels.get(j))
                .map(|v| v.linear)
                .unwrap_or(Vec3::zeros())
        };

        let mass_i = masses.get(i).copied().unwrap_or(f64::INFINITY);
        let mass_j = if j == usize::MAX {
            f64::INFINITY
        } else {
            masses.get(j).copied().unwrap_or(f64::INFINITY)
        };

        let force = crate::compute_contact_force_implicit(
            contact, material, &vel_i, &vel_j, mass_i, mass_j, dt,
        );

        if j == usize::MAX {
            forces[i] = forces[i] + force;
        } else {
            forces[i] = forces[i] - force;
            forces[j] = forces[j] + force;
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
