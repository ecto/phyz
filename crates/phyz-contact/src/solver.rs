//! Contact detection and force computation.

use crate::material::ContactMaterial;
use phyz_collision::{AABB, Collision, sweep_and_prune};
use phyz_math::{SpatialTransformExt, SpatialVec, Vec3};
use phyz_model::{Geometry as ModelGeometry, Heightfield, Model, State};

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

/// Find all body-body contacts in the current state.
///
/// Walks every collision shape of every body — [`phyz_model::Body::collisions`]
/// at its own offset and orientation, falling back to the centred
/// [`phyz_model::Body::geometry`] for bodies with an empty collision list — so
/// a humanoid whose URDF hangs a capsule off the shoulder can have that capsule
/// hit something. The `geometries` argument is gone: the shapes come from
/// `model`, which is the only place the offsets live.
///
/// # Which pairs are considered
///
/// Structural exclusions come from [`phyz_model::Model::may_collide`]: a body
/// never collides with itself, with its parent or child, or with anything a
/// fixed-joint chain has welded to it. Those pairs overlap in *every*
/// configuration — a joint's two links intersect at the joint by construction —
/// so reporting them would bury every genuine self-touch under one permanent
/// contact per joint. Model-specific exclusions live in
/// [`phyz_model::Model::contact_exclude`].
///
/// # Margin
///
/// `margin` (metres) is the band outside which a separated pair is ignored.
/// Inside it a pair still produces a manifold, reported with a **negative**
/// `penetration_depth` equal to minus its gap. Pass
/// [`ContactMaterial::margin`] — the same knob the ground path takes.
///
/// This path had no margin for a long time, and the reason it now does is
/// worth stating: contact is *soft*, so a pair at zero distance carries nearly
/// its full share of the load and then vanishes, and the contact set jumps
/// underneath the solver. On the ground path that cost a balancing controller
/// a 22 N discontinuity; here it cost stability outright. A limb pinched
/// between two bodies chattered rather than settling, which put hard bounds on
/// what a humanoid simulation could do — measured on a Booster K1 side-fall:
/// tucked-arm poses diverged, drops above 0.5 m diverged, and `dt = 2 ms`
/// diverged where 1 ms survived, non-monotonically in drop height, which is
/// the signature of a contact set flickering rather than of a stiffness
/// problem.
///
/// [`ContactMaterial::margin`]: crate::material::ContactMaterial::margin
pub fn find_contacts(model: &Model, state: &State, margin: f64) -> Vec<Collision> {
    let shapes = placed_shapes(model, state);
    if shapes.len() < 2 {
        return Vec::new();
    }

    let margin = if margin.is_finite() {
        margin.max(0.0)
    } else {
        0.0
    };

    // Broad phase over *shapes*, not bodies: a body with two far-apart
    // primitives would otherwise present one AABB spanning both and pair with
    // everything in between.
    //
    // Each box is padded by half the margin, because the broad phase decides
    // what the narrow phase is even allowed to see: an AABB gap never exceeds
    // the true surface gap, so a pair inside the margin has AABBs at most
    // `margin` apart, and half each closes exactly that. Culling on unpadded
    // boxes caps the effective margin at zero for any pair whose AABBs do not
    // already touch — which is most of them, and is why the first version of
    // this reported band contacts on only 4 of 158 body-body contacts through
    // a humanoid fall.
    let aabbs: Vec<AABB> = shapes
        .iter()
        .map(|s| AABB::from_geometry(&s.geometry, &s.pos, &s.rot).expanded(0.5 * margin))
        .collect();
    let pairs = sweep_and_prune(&aabbs);

    let welds = model.weld_groups();
    let mut contacts = Vec::new();

    for (si, sj) in pairs {
        let (a, b) = (&shapes[si], &shapes[sj]);
        if !model.may_collide(a.body, b.body, &welds) {
            continue;
        }

        // Full manifold from the narrow phase, penetrating or within the
        // margin band: an EPA normal when overlapping and the GJK closest-point
        // direction when separated (not the body-centre difference, which is
        // not a contact normal at all), contact points on the surfaces (not the
        // midpoint of the two body centres, which lies inside both bodies), and
        // up to four points per pair so face contacts resist tipping.
        let Some(manifold) = phyz_collision::contact_manifold_within(
            &a.geometry,
            &b.geometry,
            &a.pos,
            &a.rot,
            &b.pos,
            &b.rot,
            margin,
        ) else {
            continue;
        };

        for point in &manifold.points {
            contacts.push(Collision {
                body_i: a.body,
                body_j: b.body,
                contact_point: point.position,
                // `Manifold::normal` points from shape `a` toward shape `b`;
                // `Collision::contact_normal` is the direction `body_i` must
                // move to *separate*. Opposite senses — see the field docs for
                // what passing it through unnegated did.
                contact_normal: -manifold.normal,
                penetration_depth: point.depth,
            });
        }
    }

    contacts
}

/// One collision shape, resolved to world coordinates.
struct PlacedShape {
    body: usize,
    geometry: phyz_collision::Geometry,
    pos: Vec3,
    /// **Shape→world** rotation — the convention `phyz_collision` wants, which
    /// is the *transpose* of the one `State::body_xform` carries.
    ///
    /// `Geometry::support` computes a shape's world axis as `rot * ẑ`, and
    /// `box_support_face` takes a world direction into shape coordinates with
    /// `rot.transpose() * dir`; both therefore read `rot` as local→world. FK
    /// hands out the opposite (`SpatialTransform::rot` is world→body), and
    /// this pipeline used to pass it straight through to `AABB::from_geometry`,
    /// GJK and EPA.
    ///
    /// That is an error of `2θ` in every shape's orientation, and it is
    /// invisible at the identity — which is why it survived: body-body
    /// contacts only ever ran on upright test fixtures. The observable
    /// contradiction is that it made contact *non-equivariant*: the K1 lying
    /// on its side reported a 13 mm trunk/upper-arm overlap that vanished when
    /// the same joint angles were evaluated with the base upright, even though
    /// a body-body overlap depends only on the relative configuration.
    rot: phyz_math::Mat3,
}

/// Every collision shape in the model, in world coordinates.
///
/// Bodies with a non-empty `collisions` list contribute all of them at their
/// own offsets; the rest fall back to the centred legacy `geometry` field.
/// Shapes on a body with a non-finite transform are dropped rather than
/// emitted with a NaN pose — the broad phase would skip them anyway, and a
/// contact with a NaN normal is worse than a missing one.
fn placed_shapes(model: &Model, state: &State) -> Vec<PlacedShape> {
    let mut out = Vec::new();
    for (i, body) in model.bodies.iter().enumerate() {
        let Some(xform) = state.body_xform.get(i) else {
            continue;
        };
        if !pos_is_finite(&xform.pos) || !rot_is_finite(&xform.rot) {
            continue;
        }

        let mut push = |geom: &ModelGeometry, origin: &phyz_math::SpatialTransform| {
            let sx = shape_world_xform(xform, origin);
            if !pos_is_finite(&sx.pos) || !rot_is_finite(&sx.rot) {
                return;
            }
            out.push(PlacedShape {
                body: i,
                geometry: convert_geometry(geom),
                pos: sx.pos,
                // `shape_world_xform` returns world→shape, matching FK; the
                // collision crate wants its inverse. See `PlacedShape::rot`.
                rot: sx.rot.transpose(),
            });
        };

        if body.collisions.is_empty() {
            if let Some(g) = &body.geometry {
                push(g, &phyz_math::SpatialTransform::identity());
            }
        } else {
            for inst in &body.collisions {
                push(&inst.geometry, &inst.origin);
            }
        }
    }
    out
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

        // Same convention adaptation as `contact_forces_implicit`:
        // `compute_contact_force` reads `contact_normal` as pointing from `i`
        // toward `j` and returns the force on `j`, while
        // `Collision::contact_normal` is now the direction `i` separates
        // along. The ground branch already agreed and is untouched; the body
        // pair flips the normal back for the call.
        let adapted;
        let (query, on_i) = if j == usize::MAX {
            (contact, true)
        } else {
            adapted = Collision {
                contact_normal: -contact.contact_normal,
                ..contact.clone()
            };
            (&adapted, false)
        };
        #[allow(deprecated)]
        let force = crate::compute_contact_force(query, material, &vel_i, &vel_j);
        let f_linear = force.linear;

        // Apply equal and opposite forces AT THE CONTACT POINT; the torque
        // component is `τ = r × F` with `r` from the body frame origin to the
        // contact point in world frame. For ground contacts only body i is
        // updated.
        let r_i = contact.contact_point - state.body_xform[i].pos;
        if on_i {
            forces[i] = forces[i] + SpatialVec::new(r_i.cross(f_linear), f_linear);
        } else {
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

        // `compute_contact_force_implicit` predates the unified normal
        // convention: it reads `contact_normal` as pointing from `i` toward
        // `j` and returns the force on `j`. For a ground contact `j` is the
        // world and the caller has always applied the result to `i`, which
        // happens to agree with today's convention — so that path is left
        // exactly as it was. For a body pair the two senses are opposite, so
        // the normal is flipped back for the call and the result applied to
        // `j` with its reaction on `i`.
        let adapted;
        let (query, on_i) = if j == usize::MAX {
            (contact, true)
        } else {
            adapted = Collision {
                contact_normal: -contact.contact_normal,
                ..contact.clone()
            };
            (&adapted, false)
        };
        #[allow(deprecated)]
        let force = crate::compute_contact_force_implicit(
            query, material, &vel_i, &vel_j, mass_i, mass_j, dt,
        );
        let f_linear = force.linear;

        let r_i = contact.contact_point - state.body_xform[i].pos;
        if on_i {
            forces[i] = forces[i] + SpatialVec::new(r_i.cross(f_linear), f_linear);
        } else {
            let r_j = contact.contact_point - state.body_xform[j].pos;
            forces[j] = forces[j] + SpatialVec::new(r_j.cross(f_linear), f_linear);
            forces[i] = forces[i] + SpatialVec::new(r_i.cross(-f_linear), -f_linear);
        }
    }

    forces
}

/// Find contacts against a horizontal ground plane at `z = ground_height`.
///
/// Reads one **centred** shape per body. For bodies whose collision shapes
/// live at offsets ([`phyz_model::Body::collisions`]) use
/// [`find_ground_contacts_model`], which walks the full collision set and is
/// what `phyz::Simulator` runs.
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
    let margin = sanitize_margin(margin);
    let cutoff = ground_height + margin;

    for (i, geom_opt) in geometries.iter().enumerate() {
        let Some(geom) = geom_opt else { continue };
        let xform = &state.body_xform[i];
        if !pos_is_finite(&xform.pos) || !rot_is_finite(&xform.rot) {
            continue;
        }
        let Some((candidates, _drop)) = ground_candidates(geom, xform) else {
            continue;
        };
        emit_ground_hits(&mut contacts, i, candidates, ground_height, cutoff);
    }

    contacts
}

/// A negative margin would mean "ignore contacts that are already
/// penetrating", which is never what a caller wants.
fn sanitize_margin(margin: f64) -> f64 {
    if margin.is_finite() {
        margin.max(0.0)
    } else {
        0.0
    }
}

/// Candidate ground-support points for one shape, in world coordinates.
///
/// `xform` follows the body-transform convention (`rot` is world→shape, `pos`
/// is the shape origin in world) — for a shape mounted at an offset inside a
/// body, compose the body transform with the shape's `GeomInstance::origin`
/// first; [`shape_world_xform`] does exactly that.
///
/// The second return is the shape's *world-axis drop*: the constant `r` by
/// which a sphere or capsule support point hangs below its centre along world
/// `−ẑ` (zero for shapes whose candidates are material points). The convex
/// adjoint needs it to pin a contact to a body-frame feature.
///
/// Returns `None` for planes, which cannot rest on the ground plane.
fn ground_candidates(
    geom: &ModelGeometry,
    xform: &phyz_math::SpatialTransform,
) -> Option<(Vec<Vec3>, f64)> {
    // `SpatialTransform::rot` is the *world→shape* rotation; the
    // direction-carrying `SpatialTransformExt` methods used below exist so
    // this file never has to hand-roll the transpose again. (Using `rot`
    // directly here once rotated the offsets the wrong way — invisible at
    // identity, inverted as soon as the body tilted.)
    let pos = xform.pos;
    let (candidates, drop): (Vec<Vec3>, f64) = match geom {
        ModelGeometry::Box { half_extents } => {
            let h = half_extents;
            let mut v = Vec::with_capacity(8);
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    for sz in [-1.0, 1.0] {
                        v.push(xform.body_to_world_point(Vec3::new(sx * h.x, sy * h.y, sz * h.z)));
                    }
                }
            }
            (v, 0.0)
        }
        ModelGeometry::Sphere { radius } => (vec![pos - Vec3::new(0.0, 0.0, *radius)], *radius),
        ModelGeometry::Capsule { radius, length } => {
            // The two hemisphere centres, each dropped by the radius.
            let axis = xform.body_to_world_dir(Vec3::new(0.0, 0.0, length * 0.5));
            (
                vec![
                    pos + axis - Vec3::new(0.0, 0.0, *radius),
                    pos - axis - Vec3::new(0.0, 0.0, *radius),
                ],
                *radius,
            )
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
                let r = ex * phyz_math::fp::cos(t) + ey * phyz_math::fp::sin(t);
                v.push(pos + hz + r);
                v.push(pos - hz + r);
            }
            (v, 0.0)
        }
        ModelGeometry::Mesh { vertices, .. } => (
            vertices
                .iter()
                .map(|v| xform.body_to_world_point(*v))
                .collect(),
            0.0,
        ),
        ModelGeometry::Plane { .. } => return None,
    };
    Some((candidates, drop))
}

/// The world transform of a shape mounted at `origin` inside a body at
/// `body_xform`, in the same convention as the body transform itself
/// (`rot` world→shape, `pos` shape origin in world).
///
/// `origin` follows [`phyz_model::GeomInstance::origin`]: `origin.pos` is the
/// shape origin in body coordinates and `origin.rot` is body→shape.
fn shape_world_xform(
    body_xform: &phyz_math::SpatialTransform,
    origin: &phyz_math::SpatialTransform,
) -> phyz_math::SpatialTransform {
    phyz_math::SpatialTransform::new(
        origin.rot.mul_mat(&body_xform.rot),
        body_xform.body_to_world_point(origin.pos),
    )
}

/// Filter, rank, cap and emit one body's candidate points as ground contacts.
///
/// Keep the points within `margin` of the plane, deepest first, capped at a
/// manifold. `depth` is signed: positive is penetration, negative is a gap
/// inside the margin band. The comparison is strict, so a point at exactly
/// `margin` is already excluded — and by then its impedance has already
/// ramped to zero, which is what makes the exclusion a no-op rather than a
/// step.
fn emit_ground_hits(
    contacts: &mut Vec<Collision>,
    body: usize,
    candidates: Vec<Vec3>,
    ground_height: f64,
    cutoff: f64,
) {
    let mut hits: Vec<(f64, Vec3)> = candidates
        .into_iter()
        .filter(|p| p.z.is_finite() && p.z < cutoff)
        .map(|p| (ground_height - p.z, p))
        .collect();
    hits.sort_by(|a, b| b.0.total_cmp(&a.0));
    hits.truncate(phyz_collision::MAX_MANIFOLD_POINTS);

    for (depth, p) in hits {
        contacts.push(Collision {
            body_i: body,
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

/// [`find_ground_contacts`] over the model's *full* collision set.
///
/// Where `find_ground_contacts` reads one centred shape per body (the legacy
/// [`phyz_model::Body::geometry`] field), this walks
/// [`phyz_model::Body::collisions`] — every shape, each at its own offset and
/// orientation inside the body — falling back to `geometry` only for bodies
/// whose collision list is empty. A humanoid whose URDF places a trunk box
/// 15 cm above the link origin finally *has* a trunk, instead of a phantom
/// shape centred on the hip.
///
/// Candidates from all of a body's shapes are ranked together and capped at
/// the same per-body manifold as before, so a single-shape body produces
/// bit-identical contacts through either entry point.
pub fn find_ground_contacts_model(
    model: &Model,
    state: &State,
    ground_height: f64,
    margin: f64,
) -> Vec<Collision> {
    find_ground_contacts_model_with_drop(model, state, ground_height, margin)
        .into_iter()
        .map(|(c, _)| c)
        .collect()
}

/// [`find_ground_contacts_model`], with each contact's *world-axis drop*: the
/// radius by which a sphere/capsule support point hangs below its centre along
/// world `−ẑ` (zero for box, cylinder and mesh contacts, whose support points
/// are material points of the body). The convex adjoint uses it to pin a
/// contact to the body-frame feature that produced it; with several shapes
/// per body the producing shape is no longer recoverable from the body alone,
/// so detection reports it.
pub fn find_ground_contacts_model_with_drop(
    model: &Model,
    state: &State,
    ground_height: f64,
    margin: f64,
) -> Vec<(Collision, f64)> {
    let margin = sanitize_margin(margin);
    let cutoff = ground_height + margin;
    let mut out = Vec::new();

    for (i, body) in model.bodies.iter().enumerate() {
        let Some(xform) = state.body_xform.get(i) else {
            continue;
        };
        if !pos_is_finite(&xform.pos) || !rot_is_finite(&xform.rot) {
            continue;
        }

        // One candidate pool per body: shapes compete for the same manifold
        // slots, exactly as a single shape's own corners already did.
        let mut pool: Vec<(f64, Vec3, f64)> = Vec::new();
        let mut push_shape = |geom: &ModelGeometry, sx: &phyz_math::SpatialTransform| {
            let Some((candidates, drop)) = ground_candidates(geom, sx) else {
                return;
            };
            pool.extend(
                candidates
                    .into_iter()
                    .filter(|p| p.z.is_finite() && p.z < cutoff)
                    .map(|p| (ground_height - p.z, p, drop)),
            );
        };

        if body.collisions.is_empty() {
            if let Some(geom) = &body.geometry {
                push_shape(geom, xform);
            }
        } else {
            for inst in &body.collisions {
                push_shape(&inst.geometry, &shape_world_xform(xform, &inst.origin));
            }
        }

        pool.sort_by(|a, b| b.0.total_cmp(&a.0));
        pool.truncate(phyz_collision::MAX_MANIFOLD_POINTS);

        for (depth, p, drop) in pool {
            out.push((
                Collision {
                    body_i: i,
                    body_j: usize::MAX, // Ground is not a body
                    contact_point: Vec3::new(p.x, p.y, ground_height - depth * 0.5),
                    contact_normal: Vec3::z(),
                    penetration_depth: depth,
                },
                drop,
            ));
        }
    }

    out
}

/// [`find_ground_contacts_model`] against a [`Heightfield`] instead of a
/// flat plane.
///
/// Each candidate support point `p` is tested against the terrain surface
/// directly below it: with `h = hf.height(p.x, p.y)` and `n` the surface
/// normal there, the signed depth along the normal is `n.z · (h − p.z)` —
/// positive inside the terrain, negative in the margin band above it. The
/// contact normal is the terrain normal, so a box on a uniform slope feels
/// exactly the tilted plane it is standing on.
///
/// # Small-slope assumption
///
/// Support candidates are still selected with world `−ẑ` as the "down"
/// direction (candidate selection is shared with the flat path), not the
/// local terrain normal. For the shallow terrain a walking or skating robot
/// trains on — a few degrees of ramp, centimetre bumps — the two directions
/// select the same feature; on a steep wall a curved shape's reported
/// support point would be off by the slope angle. Vertical-face terrain is
/// out of scope for this detector.
///
/// On a [`Heightfield::flat`] field this reproduces
/// [`find_ground_contacts_model`] bit for bit: the normal is `+ẑ`, the
/// depth reduces to `h − p.z`, and the contact point to the same
/// midsurface.
pub fn find_heightfield_contacts_model(
    model: &Model,
    state: &State,
    hf: &Heightfield,
    margin: f64,
) -> Vec<Collision> {
    find_heightfield_contacts_model_with_drop(model, state, hf, margin)
        .into_iter()
        .map(|(c, _)| c)
        .collect()
}

/// [`find_heightfield_contacts_model`], with each contact's world-axis drop
/// (see [`find_ground_contacts_model_with_drop`]).
pub fn find_heightfield_contacts_model_with_drop(
    model: &Model,
    state: &State,
    hf: &Heightfield,
    margin: f64,
) -> Vec<(Collision, f64)> {
    let margin = sanitize_margin(margin);
    let mut out = Vec::new();

    for (i, body) in model.bodies.iter().enumerate() {
        let Some(xform) = state.body_xform.get(i) else {
            continue;
        };
        if !pos_is_finite(&xform.pos) || !rot_is_finite(&xform.rot) {
            continue;
        }

        // One candidate pool per body, exactly as the flat path: shapes
        // compete for the same manifold slots. Entries carry the terrain
        // normal alongside depth, point and drop.
        let mut pool: Vec<(f64, Vec3, Vec3, f64)> = Vec::new();
        let mut push_shape = |geom: &ModelGeometry, sx: &phyz_math::SpatialTransform| {
            let Some((candidates, drop)) = ground_candidates(geom, sx) else {
                return;
            };
            for p in candidates {
                if !(p.z.is_finite() && p.x.is_finite() && p.y.is_finite()) {
                    continue;
                }
                let n = hf.normal(p.x, p.y);
                let depth = n.z * (hf.height(p.x, p.y) - p.z);
                if depth > -margin {
                    pool.push((depth, p, n, drop));
                }
            }
        };

        if body.collisions.is_empty() {
            if let Some(geom) = &body.geometry {
                push_shape(geom, xform);
            }
        } else {
            for inst in &body.collisions {
                push_shape(&inst.geometry, &shape_world_xform(xform, &inst.origin));
            }
        }

        pool.sort_by(|a, b| b.0.total_cmp(&a.0));
        pool.truncate(phyz_collision::MAX_MANIFOLD_POINTS);

        for (depth, p, n, drop) in pool {
            out.push((
                Collision {
                    body_i: i,
                    body_j: usize::MAX, // Terrain is not a body
                    // Midsurface between the support point and the terrain,
                    // along the normal — the heightfield generalization of
                    // the flat path's `ground_height - depth/2`.
                    contact_point: p + n * (depth * 0.5),
                    contact_normal: n,
                    penetration_depth: depth,
                },
                drop,
            ));
        }
    }

    out
}
