//! Contact detection and force computation.

use crate::material::ContactMaterial;
use phyz_collision::{AABB, Collision, gjk_distance_rot, sweep_and_prune};
use phyz_math::{SpatialTransformExt, SpatialVec, Vec3};
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
///
/// # Known gap: this path has no margin
///
/// The `dist < 0.0` predicate below is the same zero-margin cutoff that
/// [`find_ground_contacts`] used to have, and it has the same consequence: a
/// pair at exactly touching distance still carries `solimp.dmin` of the load
/// and then drops to nothing. It is deliberately *not* fixed here, because the
/// fix is not the same one. Ground contacts have an analytic signed distance
/// per candidate point, so a negative depth is just arithmetic; a general pair
/// gets its manifold from `contact_manifold`, which is EPA-based and only
/// defined on the penetrating side. Producing separated manifolds needs a
/// GJK witness-point/closest-feature path that does not exist yet. Tracked
/// separately; see the `find_ground_contacts` docs for the physics.
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
                // Poisoned transform; degrade gracefully and skip this body
                // by giving it a degenerate, non-finite-tagged AABB.
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
                // Full manifold from the narrow phase: an EPA normal (not the
                // body-centre difference, which is not a contact normal at
                // all), contact points on the surfaces (not the midpoint of
                // the two body centres, which lies inside both bodies), and up
                // to four points per pair so face contacts resist tipping.
                let Some(manifold) = phyz_collision::contact_manifold(
                    &collision_geom_i,
                    &collision_geom_j,
                    &pos_i,
                    &rot_i,
                    &pos_j,
                    &rot_j,
                ) else {
                    continue;
                };

                for point in &manifold.points {
                    contacts.push(Collision {
                        body_i: i,
                        body_j: j,
                        contact_point: point.position,
                        contact_normal: manifold.normal,
                        penetration_depth: point.depth,
                    });
                }
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
#[deprecated(
    note = "penalty contact is superseded by the convex solve (`assemble` + \
            `solve_contacts`), which couples contacts through the Delassus \
            operator instead of treating each in isolation"
)]
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
        #[allow(deprecated)]
        let force = crate::compute_contact_force(contact, material, &vel_i, &vel_j);
        let f_linear = force.linear;

        // Apply equal and opposite forces AT THE CONTACT POINT. Sign
        // convention follows Goal 1; the torque component follows Goal 4
        // (τ = r × F where r is from the body's frame origin to the contact
        // point in world frame). For ground contacts, only body i is updated.
        if j == usize::MAX {
            let r_i = contact.contact_point - state.body_xform[i].pos;
            forces[i] = forces[i] + SpatialVec::new(r_i.cross(f_linear), f_linear);
        } else {
            let r_i = contact.contact_point - state.body_xform[i].pos;
            let r_j = contact.contact_point - state.body_xform[j].pos;
            forces[i] = forces[i] + SpatialVec::new(r_i.cross(-f_linear), -f_linear);
            forces[j] = forces[j] + SpatialVec::new(r_j.cross(f_linear), f_linear);
        }
    }

    forces
}

/// Compute contact forces with implicit damping for low-mass-body stability.
///
/// Like [`contact_forces`], but the per-contact wrench is computed by
/// [`crate::compute_contact_force_implicit`] which uses the body velocity at
/// the END of the step. This makes the contact solve unconditionally stable
/// for any choice of `dt`, contact stiffness, and damping — preventing the
/// "low-mass cube launches off the plate" failure mode of the explicit form.
///
/// `masses[i]` is the effective contact mass of body `i`. Use
/// `f64::INFINITY` for fixed/world bodies. For ground contacts
/// (`body_j == usize::MAX`) the ground is treated as having infinite mass.
#[deprecated(
    note = "penalty contact is superseded by the convex solve (`assemble` + \
            `solve_contacts`), which couples contacts through the Delassus \
            operator instead of treating each in isolation"
)]
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

        #[allow(deprecated)]
        let force = crate::compute_contact_force_implicit(
            contact, material, &vel_i, &vel_j, mass_i, mass_j, dt,
        );
        let f_linear = force.linear;

        // Apply at the contact point (Goal 4). Sign convention from Goal 1.
        if j == usize::MAX {
            let r_i = contact.contact_point - state.body_xform[i].pos;
            forces[i] = forces[i] + SpatialVec::new(r_i.cross(f_linear), f_linear);
        } else {
            let r_i = contact.contact_point - state.body_xform[i].pos;
            let r_j = contact.contact_point - state.body_xform[j].pos;
            forces[i] = forces[i] + SpatialVec::new(r_i.cross(-f_linear), -f_linear);
            forces[j] = forces[j] + SpatialVec::new(r_j.cross(f_linear), f_linear);
        }
    }

    forces
}

/// Find contacts against a horizontal ground plane at `z = ground_height`.
///
/// Shapes with a flat underside (boxes) report a contact per supporting
/// corner, so a resting box has a real support polygon. Curved shapes report
/// the single point they actually touch at.
///
/// The old implementation ignored body orientation entirely — it used the
/// axis-aligned half-extent for depth and put one contact at the body centre's
/// `(x, y)`. A tilted box therefore reported the wrong depth at a point that
/// was not on the box, and no box could resist tipping.
///
/// # Margin
///
/// `margin` (metres) is the band *above* the plane in which a candidate still
/// produces a contact, reported with a **negative** `penetration_depth` equal to
/// minus its gap. Pass [`ContactMaterial::margin`] — that is the knob, and it is
/// what [`ContactMaterial::combine`] already mixes with `max`. Pass `0.0` to get
/// the old hard `p.z < ground_height` predicate back.
///
/// The predicate used to be exactly that hard cutoff, and that was a real bug
/// rather than a cosmetic one. Contact here is *soft*: a candidate at exactly
/// zero depth still has impedance `solimp.dmin` (0.9 by default), so it carries
/// nearly its full share of the load right up to the instant it stops existing.
/// On a K1 humanoid stance the least-loaded foot corner was measured carrying
/// **22.3 N — 11% of body weight — on the step before it vanished**, and 0 N on
/// the next. Total vertical force was conserved (the load redistributed to the
/// remaining corners), so nothing leaked; but the contact *set* jumped, and it
/// jumped underneath a balancing controller.
///
/// With a margin the candidate survives past the surface with a negative depth,
/// and [`ContactMaterial::impedance_at`] tapers its impedance — and hence the
/// normal force, which is linear in impedance — smoothly to zero at the band
/// edge. Separated contacts get no stabilization bias and cannot pull, so this
/// buys continuity without buying adhesion.
///
/// Candidates are still ranked deepest-first before the manifold cap, so the
/// margin never lets a barely-separated point displace a genuinely loaded one.
///
/// [`ContactMaterial::margin`]: crate::material::ContactMaterial::margin
/// [`ContactMaterial::combine`]: crate::material::ContactMaterial::combine
/// [`ContactMaterial::impedance_at`]: crate::material::ContactMaterial::impedance_at
pub fn find_ground_contacts(
    state: &State,
    geometries: &[Option<ModelGeometry>],
    ground_height: f64,
    margin: f64,
) -> Vec<Collision> {
    let mut contacts = Vec::new();
    // A negative margin would mean "ignore contacts that are already
    // penetrating", which is never what a caller wants.
    let margin = if margin.is_finite() {
        margin.max(0.0)
    } else {
        0.0
    };
    let cutoff = ground_height + margin;

    for (i, geom_opt) in geometries.iter().enumerate() {
        let Some(geom) = geom_opt else { continue };
        let xform = &state.body_xform[i];
        // `SpatialTransform::rot` is the *world→body* rotation; the
        // direction-carrying `SpatialTransformExt` methods used below exist so
        // this file never has to hand-roll the transpose again. (Using `rot`
        // directly here once rotated the offsets the wrong way — invisible at
        // identity, inverted as soon as the body tilted.)
        let pos = xform.pos;
        if !pos_is_finite(&pos) || !rot_is_finite(&xform.rot) {
            continue;
        }

        // Candidate support points in world coordinates.
        let candidates: Vec<Vec3> = match geom {
            ModelGeometry::Box { half_extents } => {
                let h = half_extents;
                let mut v = Vec::with_capacity(8);
                for sx in [-1.0, 1.0] {
                    for sy in [-1.0, 1.0] {
                        for sz in [-1.0, 1.0] {
                            v.push(xform.body_to_world_point(Vec3::new(
                                sx * h.x,
                                sy * h.y,
                                sz * h.z,
                            )));
                        }
                    }
                }
                v
            }
            ModelGeometry::Sphere { radius } => {
                vec![pos - Vec3::new(0.0, 0.0, *radius)]
            }
            ModelGeometry::Capsule { radius, length } => {
                // The two hemisphere centres, each dropped by the radius.
                let axis = xform.body_to_world_dir(Vec3::new(0.0, 0.0, length * 0.5));
                vec![
                    pos + axis - Vec3::new(0.0, 0.0, *radius),
                    pos - axis - Vec3::new(0.0, 0.0, *radius),
                ]
            }
            ModelGeometry::Cylinder { radius, height } => {
                // Rim points of both end caps, sampled around the circle.
                let hz = xform.body_to_world_dir(Vec3::new(0.0, 0.0, height * 0.5));
                let (ex, ey) = (
                    xform.body_to_world_dir(Vec3::x()) * *radius,
                    xform.body_to_world_dir(Vec3::y()) * *radius,
                );
                let mut v = Vec::with_capacity(8);
                for k in 0..4 {
                    let t = k as f64 * std::f64::consts::FRAC_PI_2;
                    let r = ex * t.cos() + ey * t.sin();
                    v.push(pos + hz + r);
                    v.push(pos - hz + r);
                }
                v
            }
            ModelGeometry::Mesh { vertices, .. } => vertices
                .iter()
                .map(|v| xform.body_to_world_point(*v))
                .collect(),
            ModelGeometry::Plane { .. } => continue,
        };

        // Keep the points within `margin` of the plane, deepest first, capped
        // at a manifold. `depth` is signed: positive is penetration, negative
        // is a gap inside the margin band. The comparison is strict, so a point
        // at exactly `margin` is already excluded — and by then its impedance
        // has already ramped to zero, which is what makes the exclusion a
        // no-op rather than a step.
        let mut hits: Vec<(f64, Vec3)> = candidates
            .into_iter()
            .filter(|p| p.z.is_finite() && p.z < cutoff)
            .map(|p| (ground_height - p.z, p))
            .collect();
        if hits.is_empty() {
            continue;
        }
        hits.sort_by(|a, b| b.0.total_cmp(&a.0));
        hits.truncate(phyz_collision::MAX_MANIFOLD_POINTS);

        for (depth, p) in hits {
            contacts.push(Collision {
                body_i: i,
                body_j: usize::MAX, // Ground is not a body
                // On the midsurface between the vertex and the plane. This
                // stays correct for a negative depth: the midpoint simply sits
                // above the plane rather than below it.
                contact_point: Vec3::new(p.x, p.y, ground_height - depth * 0.5),
                contact_normal: Vec3::z(),
                penetration_depth: depth,
            });
        }
    }

    contacts
}
