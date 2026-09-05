//! Contact manifold generation.
//!
//! GJK+EPA answers "how deep, and along which normal" with a *single* number
//! and direction. That is not enough to rest a box on a plane: one point has
//! no resistance to tipping, so a stack jitters and falls over. A manifold is
//! the set of contact points sharing that normal — for a face-on-face contact,
//! the polygon where the two surfaces overlap.
//!
//! # Method
//!
//! 1. Run EPA for the separating normal `n` and depth.
//! 2. Ask each shape for its *support face*: the polygon whose outward normal
//!    is most aligned with `±n`. Curved shapes (sphere, and the barrel of a
//!    capsule/cylinder) have no flat face and report `None`.
//! 3. If both shapes gave a polygon, clip the incident polygon against the
//!    side planes of the reference polygon (Sutherland–Hodgman) and keep the
//!    clipped vertices that lie below the reference face.
//! 4. Reduce to at most [`MAX_MANIFOLD_POINTS`] by keeping the deepest point
//!    and then maximizing spread, so the retained patch spans the real contact
//!    area rather than clustering in a corner.
//!
//! When either shape is curved, the single EPA point is the correct and
//! complete answer — a sphere really does touch a plane at one point.
//!
//! # Differentiability
//!
//! Feature selection (which face is the reference, which clipped vertices
//! survive) is **combinatorial** and is deliberately not on the differentiable
//! path: it is decided once per step from the primal configuration and then
//! held fixed. Each surviving point's position and depth *are* smooth
//! functions of the configuration. This is the same approximation MuJoCo and
//! Dojo make; see `docs/design/differentiable-contact.md` §4.4.
//!
//! That paragraph was true of the *clipped* path and false of the fallback,
//! and until `fix/clip-faces-manifolds` the fallback was where 80 % of the
//! body-body pairs of a humanoid standing on a skateboard ended up.
//!
//! Step 3 rejected its whole clipped polygon because of two things it got
//! wrong about the reference face:
//!
//! - **It always took the reference from `A`.** The separating normal is a
//!   face normal of at most one of the two boxes; the other touches on an edge
//!   or a corner, and its "support face" is tilted away from `n` by whatever
//!   the relative orientation is. Whichever face is better aligned with `n` is
//!   the one the manifold should be measured against.
//! - **It measured separation along `n`, not along the reference face's own
//!   normal.** The only direction that measures a distance from a plane is
//!   that plane's normal. Measuring along `n` charges every incident vertex
//!   for the reference face's own extent along `n` — on the K1 skate stance,
//!   up to 45 mm. GJK reported a 0.5 mm gap; `clip_faces` read 3–44 mm,
//!   `sep > margin` rejected every vertex, and the pair fell through.
//!
//! Downstream that mattered because `single_point` picks a box corner by the
//! **sign of each component** of the direction in the box's local frame, and
//! along a face normal those in-plane components are exact-cancellation noise.
//! Their signs are decided by the last bits of the pose, so the witness point
//! walked between corners of the box — measured at up to 24 cm — while the
//! normal and the depth stayed bit-identical. [`stable_witness`] makes that
//! fallback report the supporting *feature's* centroid instead; it is still
//! there, still off by default, and no longer the thing standing between this
//! module and a differentiable contact.
//!
//! [`legacy_clip`] restores both old behaviours together.

use crate::geometry::Geometry;
use crate::gjk::GjkOutcome;
use phyz_math::{Mat3, Vec3};
use std::sync::OnceLock;

/// Whether to build the `single_point` fallback from the *supporting
/// feature's centroid* rather than from a single support vertex.
///
/// Off by default: with this unset every number this module produces is
/// bit-identical to what it has always produced.
///
/// # What it is for
///
/// `Geometry::support` picks a box corner by the **sign of each component** of
/// the direction in the box's local frame. When the contact normal is aligned
/// with a face — a foot flat on a deck, the overwhelmingly common resting case
/// — the two in-plane components are not small, they are *noise*: the exact
/// cancellation is at the 1e-16 level, and their signs are decided by the last
/// bits of the pose. Flip one and the returned vertex is a different corner of
/// the box, a whole half-extent away.
///
/// Measured on the K1 skate stance (13 contacts, two soles on grip tape, four
/// wheels, the deck joint) **before** `fix/clip-faces-manifolds`: the face-clip
/// path failed for 80 % of body-body pairs — the incident face's vertices read
/// 3–44 mm beyond the reference plane while GJK reported a 0.5 mm gap — so
/// nearly every body-body contact took this fallback. Perturbing one torque
/// lane by `1e-9 N·m` moved witness points by up to **24 cm** while the normal
/// and the depth stayed bit-identical, which is the signature exactly: the
/// physics of the contact did not move, the *choice of vertex* did. Downstream
/// that was a different moment arm, a `4.2e-2` change in the solved impulses,
/// and a trunk position that jumped `6.3e-6 m` in one 0.5 ms step — an
/// amplification of `~1e9` that made two consecutive contacted steps
/// non-differentiable even though one step on its own matched central
/// differences to every digit.
///
/// The fallback rate on that stance is 1.8 % now, so this flag is a net under
/// the fix rather than the fix itself — and on that stance turning it *on* is
/// measurably worse (a two-step `1e-9 N·m` response of `1.3e-10 m` against
/// `3.8e-15 m` with it off), because averaging tied vertices reintroduces a
/// tie set that moves.
///
/// With this set, vertices within `SUPPORT_TIE_EPS` of the maximum are
/// averaged instead of raced. A face-on contact returns the face centre, an
/// edge-on contact the edge midpoint, a corner contact the corner — the
/// centroid of whichever feature actually supports the direction. The tie set
/// is then a function of the geometry at millimetre scale rather than of the
/// 16th digit, so a last-bits change to the state no longer moves the contact
/// point at all.
pub fn stable_witness() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PHYZ_STABLE_WITNESS").is_ok_and(|v| v == "1" || v == "true"))
}

/// Restore the pre-fix face clipper: reference face always taken from shape
/// `A`, and separation measured along the contact normal from that face's
/// first vertex.
///
/// Off by default. The old path produced no face manifold at all for 80 % of
/// the body-body pairs of the K1 skate stance, and let a 20:1 box stack fall
/// flat — see `clip_faces` and `tests/contact_physics_benchmarks.rs` — so it
/// is kept only as an escape hatch for reproducing pre-fix numbers.
pub fn legacy_clip() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PHYZ_LEGACY_CLIP").is_ok_and(|v| v == "1" || v == "true"))
}

/// Trace every face-clip decision to stderr. Diagnostic only.
fn clip_debug() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PHYZ_CLIP_DEBUG").is_ok())
}

/// Relative slack for deciding that two vertices support a direction equally.
///
/// Sized to sit far above float cancellation noise (~1e-16 relative) and far
/// below any real geometric distinction: at a 0.3 m box this is 3e-10 m, so a
/// vertex is only ever tied with the maximum when it is coplanar with it to
/// well under a micron.
const SUPPORT_TIE_EPS: f64 = 1e-9;

/// Upper bound on points retained per manifold.
///
/// Four is enough to span any planar contact patch (a quad), and is the
/// standard choice — more points cost solver time without adding constraint
/// rank for convex face contact.
pub const MAX_MANIFOLD_POINTS: usize = 4;

/// One point of a contact manifold.
#[derive(Debug, Clone, Copy)]
pub struct ManifoldPoint {
    /// Contact position in world coordinates, on the midsurface between the
    /// two overlapping surfaces.
    pub position: Vec3,
    /// Penetration depth at this point (positive = overlapping).
    pub depth: f64,
}

/// A set of contact points sharing one normal.
#[derive(Debug, Clone)]
pub struct Manifold {
    /// Shared contact normal, pointing from shape `a` toward shape `b`.
    pub normal: Vec3,
    /// Contact points, at most [`MAX_MANIFOLD_POINTS`], deepest first.
    pub points: Vec<ManifoldPoint>,
}

/// Build a contact manifold between two posed geometries.
///
/// Returns `None` when the shapes are disjoint or the query is degenerate.
pub fn contact_manifold(
    geom_a: &Geometry,
    geom_b: &Geometry,
    pos_a: &Vec3,
    rot_a: &Mat3,
    geom_b_pos: &Vec3,
    rot_b: &Mat3,
) -> Option<Manifold> {
    contact_manifold_within(geom_a, geom_b, pos_a, rot_a, geom_b_pos, rot_b, 0.0)
}

/// Build a contact manifold, including for pairs separated by less than
/// `margin`.
///
/// # Why a separated manifold is worth building
///
/// Contact here is *soft*: a pair at exactly zero distance still carries
/// impedance `solimp.dmin` (0.9 by default), so it holds nearly its full share
/// of the load right up to the instant it stops existing. With a hard cutoff
/// the contact set jumps underneath the solver. `find_ground_contacts` has had
/// a margin for exactly this reason, and the measured consequence of *not*
/// having one on the body-body path was not subtle: a limb pinched between two
/// bodies chattered instead of settling, which bounded the poses, the drop
/// heights and the timestep a humanoid simulation could use.
///
/// Within the band a point is reported with a **negative** `depth` equal to
/// minus its gap, and `phyz_contact::ContactMaterial::impedance_at` then
/// tapers its impedance — and hence its force — smoothly to zero at the band
/// edge. Separated contacts
/// get no stabilization bias and cannot pull, so this buys continuity without
/// buying adhesion.
///
/// # How
///
/// Both branches need one thing: a separating normal pointing from `a` toward
/// `b`. Overlapping, EPA supplies it. Separated, it is `−v̂` where `v` is the
/// closest point of the Minkowski difference to the origin — a quantity GJK
/// already converges on and used to discard ([`GjkOutcome::Separated::closest`]).
/// With the normal in hand the *same* face-clipping path serves both, so a
/// separated box pair gets a real four-point manifold rather than a single
/// witness point, and the manifold does not change shape as the pair crosses
/// into contact.
pub fn contact_manifold_within(
    geom_a: &Geometry,
    geom_b: &Geometry,
    pos_a: &Vec3,
    rot_a: &Mat3,
    geom_b_pos: &Vec3,
    rot_b: &Mat3,
    margin: f64,
) -> Option<Manifold> {
    let pos_b = geom_b_pos;
    let margin = if margin.is_finite() {
        margin.max(0.0)
    } else {
        0.0
    };

    let (depth, normal) = match crate::gjk::gjk_rot(geom_a, geom_b, pos_a, pos_b, rot_a, rot_b) {
        GjkOutcome::Penetrating { simplex } => {
            crate::epa::epa_from_simplex(geom_a, geom_b, pos_a, pos_b, rot_a, rot_b, &simplex)?
        }
        GjkOutcome::Separated { distance, closest } => {
            // Strictly inside the band: at exactly `margin` the impedance has
            // already tapered to zero, so excluding it is a no-op rather than
            // a step. A non-finite distance is refused rather than admitted —
            // spelled out because the negated comparison that says the same
            // thing reads as a typo.
            if distance.is_nan() || distance >= margin {
                return None;
            }
            let n = closest.norm();
            if n < 1e-12 {
                // Touching exactly, with no direction to be had. EPA owns the
                // overlapping side; refuse rather than guess a normal.
                return None;
            }
            (-distance, -closest / n)
        }
        GjkOutcome::Indeterminate => return None,
    };
    if !depth.is_finite() || !is_finite(&normal) || normal.norm() < 0.5 {
        return None;
    }

    // Reference face on A points along +n (toward B); incident face on B
    // points back along -n.
    let face_a = support_face(geom_a, pos_a, rot_a, &normal);
    let face_b = support_face(geom_b, pos_b, rot_b, &(-normal));

    let points = match (face_a, face_b) {
        (Some(fa), Some(fb)) => {
            // Which of the two faces is the *reference* — the one whose plane
            // the manifold is measured against — is not a free choice. The
            // separating normal is a face normal of at most one of the two
            // boxes; the other shape touches it on an edge or a corner, and
            // its "support face" is then tilted away from `n` by whatever the
            // relative orientation happens to be. Measuring depth against that
            // tilted plane charges the incident vertices for the reference
            // face's own tilt: on the K1 skate stance the reference face spans
            // up to 45 mm along `n`, so every incident vertex reads as tens of
            // millimetres of gap while the true gap is half a millimetre.
            //
            // So take the better-aligned face as the reference, and fall back
            // to A only on a tie. `legacy_clip()` restores the old
            // always-A choice.
            let (reference, incident) = if legacy_clip()
                || fa.verts.is_empty()
                || (!fb.verts.is_empty() && fa.normal.dot(normal) >= fb.normal.dot(-normal))
            {
                (fa, fb)
            } else {
                (fb, fa)
            };
            let clipped = clip_faces(&reference, &incident, &normal, margin);
            if clip_debug() {
                eprintln!(
                    "PAIR clipped={} npts={}",
                    clipped.is_some(),
                    clipped.as_ref().map_or(0, |v| v.len())
                );
            }
            clipped.unwrap_or_else(|| {
                vec![single_point(
                    geom_a, geom_b, pos_a, rot_a, pos_b, rot_b, &normal, depth,
                )]
            })
        }
        // At least one surface is curved: one point is the whole story.
        _ => vec![single_point(
            geom_a, geom_b, pos_a, rot_a, pos_b, rot_b, &normal, depth,
        )],
    };

    let points = reduce(points);
    if points.is_empty() {
        return None;
    }
    Some(Manifold { normal, points })
}

/// The witness-point midpoint: the deepest point of A along `+n` and the
/// deepest point of B along `-n` are the two surface points that overlap; the
/// contact sits halfway between them.
///
/// This replaces the old `(pos_i + pos_j) * 0.5`, the midpoint of the two body
/// *centres*, which is generally inside both bodies and on neither surface.
#[allow(clippy::too_many_arguments)]
fn single_point(
    geom_a: &Geometry,
    geom_b: &Geometry,
    pos_a: &Vec3,
    rot_a: &Mat3,
    pos_b: &Vec3,
    rot_b: &Mat3,
    normal: &Vec3,
    depth: f64,
) -> ManifoldPoint {
    let wa = witness(geom_a, normal, pos_a, rot_a);
    let wb = witness(geom_b, &(-*normal), pos_b, rot_b);
    ManifoldPoint {
        position: (wa + wb) * 0.5,
        depth,
    }
}

/// The support point, or — under [`stable_witness`] — the centroid of the
/// supporting *feature*.
fn witness(geom: &Geometry, dir: &Vec3, pos: &Vec3, rot: &Mat3) -> Vec3 {
    if !stable_witness() {
        return geom.support(dir, pos, rot);
    }
    match geom {
        Geometry::Box { half_extents } => box_support_centroid(half_extents, pos, rot, dir),
        // Curved shapes have a unique support point for any direction — the
        // argmax is over a smooth surface, not a vertex set, so there is no
        // tie to break and nothing here to fix.
        _ => geom.support(dir, pos, rot),
    }
}

/// Centroid of the box vertices that support `dir` equally.
///
/// Returns the corner for a corner contact, the edge midpoint for an edge
/// contact, and the face centre for a face contact — with the tie decided at
/// [`SUPPORT_TIE_EPS`] rather than by the sign of a cancelled float.
fn box_support_centroid(half_extents: &Vec3, pos: &Vec3, rot: &Mat3, dir: &Vec3) -> Vec3 {
    let local_dir = rot.transpose() * *dir;
    let h = [half_extents.x, half_extents.y, half_extents.z];
    let d = [local_dir.x, local_dir.y, local_dir.z];

    // The support value separates per axis, so the tie set does too: an axis
    // contributes |d_k| * h_k when its sign is decided and *both* signs when
    // that axis's contribution is within the slack of zero. Enumerating the
    // eight corners would give the same answer at eight times the cost.
    let scale = SUPPORT_TIE_EPS
        * (1.0 + local_dir.norm())
        * (1.0 + h.iter().fold(0.0, |a: f64, x| a.max(*x)));
    let mut local = [0.0; 3];
    for k in 0..3 {
        local[k] = if (d[k] * h[k]).abs() <= scale {
            // Tied: both faces along this axis support `dir` equally, so the
            // feature spans the axis and its centroid sits at the middle.
            0.0
        } else if d[k] >= 0.0 {
            h[k]
        } else {
            -h[k]
        };
    }
    pos + *rot * Vec3::new(local[0], local[1], local[2])
}

/// A planar face of a shape, in world coordinates.
struct Face {
    /// Polygon vertices, ordered around the face.
    verts: Vec<Vec3>,
    /// Outward unit normal.
    normal: Vec3,
    /// A point known to lie on the face's plane. For a bounded polygon this is
    /// just `verts[0]`; for an unbounded plane it is the only datum available,
    /// and without it the clipper had to guess a datum from the *incident*
    /// polygon, which made every depth relative and therefore always zero at
    /// the deepest vertex.
    point: Vec3,
}

/// The face of `geom` whose outward normal is most aligned with `dir`.
///
/// Returns `None` for shapes with no flat face in that direction — the caller
/// then falls back to a single contact point.
fn support_face(geom: &Geometry, pos: &Vec3, rot: &Mat3, dir: &Vec3) -> Option<Face> {
    match geom {
        Geometry::Box { half_extents } => Some(box_support_face(half_extents, pos, rot, dir)),
        Geometry::Plane { normal } => {
            // An infinite plane has no bounded polygon; it is always the
            // reference face, and clipping against it needs no side planes.
            let n = (*rot * *normal).normalize();
            Some(Face {
                verts: Vec::new(),
                normal: n,
                point: *pos,
            })
        }
        // Sphere: no flat face anywhere. Capsule/cylinder: flat only on the
        // end caps, and treating the barrel as flat would fabricate contact
        // area that does not exist. Mesh: needs face topology we do not carry
        // through the narrow phase yet.
        _ => None,
    }
}

/// The box face whose outward normal is most aligned with `dir`, as a quad.
fn box_support_face(half_extents: &Vec3, pos: &Vec3, rot: &Mat3, dir: &Vec3) -> Face {
    // Local axis most aligned with `dir`.
    let local_dir = rot.transpose() * *dir;
    let comps = [local_dir.x, local_dir.y, local_dir.z];
    let (axis, sign) = {
        let mut best = 0usize;
        for (i, c) in comps.iter().enumerate() {
            if c.abs() > comps[best].abs() {
                best = i;
            }
        }
        (best, if comps[best] >= 0.0 { 1.0 } else { -1.0 })
    };

    let h = [half_extents.x, half_extents.y, half_extents.z];
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);

    // Four corners of the face, wound consistently.
    let mut verts = Vec::with_capacity(4);
    for (su, sv) in [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)] {
        let mut local = [0.0; 3];
        local[axis] = sign * h[axis];
        local[u] = su * h[u];
        local[v] = sv * h[v];
        let l = Vec3::new(local[0], local[1], local[2]);
        verts.push(pos + *rot * l);
    }

    let mut n_local = [0.0; 3];
    n_local[axis] = sign;
    let normal = (*rot * Vec3::new(n_local[0], n_local[1], n_local[2])).normalize();

    let point = verts[0];
    Face {
        verts,
        normal,
        point,
    }
}

/// Clip the incident polygon against the reference face's side planes and keep
/// the vertices that lie inside the reference surface.
fn clip_faces(
    reference: &Face,
    incident: &Face,
    normal: &Vec3,
    margin: f64,
) -> Option<Vec<ManifoldPoint>> {
    if incident.verts.is_empty() {
        // Incident is an infinite plane — swap roles is not meaningful here.
        return None;
    }

    let mut poly = incident.verts.clone();

    // Clip against each side plane of the reference polygon. An infinite
    // reference plane (no verts) has no side planes, so the incident face
    // survives whole — which is exactly right for a box resting on ground.
    let n_ref = reference.verts.len();
    // The polygon's winding depends on which face of the box was selected, so
    // `edge × normal` points inward for one sign and outward for the other.
    // Orienting each side plane toward the face centroid is winding-agnostic.
    let centroid = if n_ref > 0 {
        reference.verts.iter().fold(Vec3::zeros(), |a, v| a + *v) / n_ref as f64
    } else {
        Vec3::zeros()
    };
    for i in 0..n_ref {
        let a = reference.verts[i];
        let b = reference.verts[(i + 1) % n_ref];
        let edge = b - a;
        let plane_n = edge.cross(reference.normal);
        let nn = plane_n.norm();
        if nn < 1e-12 {
            continue;
        }
        let mut plane_n = plane_n / nn;
        if (centroid - a).dot(plane_n) < 0.0 {
            plane_n = -plane_n;
        }
        poly = clip_polygon(&poly, &a, &plane_n);
        if poly.is_empty() {
            if clip_debug() {
                eprintln!("CLIP EMPTY at side plane {i}/{n_ref}");
            }
            return None;
        }
    }

    // Reference plane: the point and the *face's own* outward normal.
    //
    // The separation of a clipped vertex is its signed distance from that
    // plane, and the only direction that measures a distance from a plane is
    // the plane's normal. Measuring along the contact normal instead — as this
    // did — is only equivalent when the two agree, and they agree only when
    // the reference face happens to be the face that generated `n`. When they
    // differ by 15°, as the tilted foot boxes on the skate deck do, the
    // reference face's own extent along `n` (up to 45 mm) is added to every
    // vertex's reading, the `sep > margin` test rejects the whole polygon, and
    // the manifold silently drops to the `single_point` fallback.
    let ref_point = reference.point;
    let ref_n = if legacy_clip() {
        *normal
    } else {
        reference.normal
    };

    if clip_debug() {
        let seps: Vec<f64> = poly.iter().map(|p| (p - ref_point).dot(ref_n)).collect();
        let kept = seps.iter().filter(|s| **s <= margin.max(1e-9)).count();
        eprintln!(
            "CLIP align_ref={:.6} align_inc={:.6} margin={:.3e} npoly={} kept={} seps={:?}",
            reference.normal.dot(normal).abs(),
            incident.normal.dot(normal).abs(),
            margin,
            poly.len(),
            kept,
            seps
        );
    }

    let out: Vec<ManifoldPoint> = poly
        .iter()
        .filter_map(|p| {
            // Separation of the incident vertex from the reference plane,
            // measured along the contact normal. Negative separation is
            // penetration; a positive separation up to `margin` is a gap
            // inside the band and is kept, with a negative depth.
            let sep = (p - ref_point).dot(ref_n);
            if sep > margin.max(1e-9) {
                return None;
            }
            Some(ManifoldPoint {
                // Put the point on the midsurface: halfway back toward the
                // reference plane, along that plane's normal.
                position: p - ref_n * (sep * 0.5),
                depth: -sep,
            })
        })
        .collect();

    if out.is_empty() { None } else { Some(out) }
}

/// Sutherland–Hodgman clip of `poly` against the half-space
/// `{x : (x - point)·n >= 0}`.
fn clip_polygon(poly: &[Vec3], point: &Vec3, n: &Vec3) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(poly.len() + 2);
    if poly.is_empty() {
        return out;
    }
    for i in 0..poly.len() {
        let cur = poly[i];
        let next = poly[(i + 1) % poly.len()];
        let d_cur = (cur - point).dot(n);
        let d_next = (next - point).dot(n);
        let cur_in = d_cur >= 0.0;
        let next_in = d_next >= 0.0;

        if cur_in {
            out.push(cur);
        }
        if cur_in != next_in {
            let denom = d_cur - d_next;
            if denom.abs() > 1e-15 {
                let t = d_cur / denom;
                out.push(cur + (next - cur) * t);
            }
        }
    }
    out
}

/// Reduce a candidate set to at most [`MAX_MANIFOLD_POINTS`], keeping the
/// deepest point and then the points that maximize spread.
///
/// Keeping the first N in list order would cluster them along one edge of the
/// patch and leave the contact with far less resistance to tipping than the
/// geometry actually provides.
fn reduce(mut pts: Vec<ManifoldPoint>) -> Vec<ManifoldPoint> {
    pts.retain(|p| p.depth.is_finite() && is_finite(&p.position));
    if pts.len() <= MAX_MANIFOLD_POINTS {
        pts.sort_by(|a, b| b.depth.total_cmp(&a.depth));
        return pts;
    }

    let mut chosen: Vec<ManifoldPoint> = Vec::with_capacity(MAX_MANIFOLD_POINTS);

    // Seed with the deepest point — the one that matters most for the solve.
    let deepest = pts
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.depth.total_cmp(&b.depth))
        .map(|(i, _)| i)
        .unwrap();
    chosen.push(pts.swap_remove(deepest));

    // Then farthest-point sampling.
    while chosen.len() < MAX_MANIFOLD_POINTS && !pts.is_empty() {
        let (idx, _) = pts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let d = chosen
                    .iter()
                    .map(|c| (c.position - p.position).norm())
                    .fold(f64::INFINITY, f64::min);
                (i, d)
            })
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();
        chosen.push(pts.swap_remove(idx));
    }

    chosen.sort_by(|a, b| b.depth.total_cmp(&a.depth));
    chosen
}

fn is_finite(v: &Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// The witness point under [`stable_witness`].
///
/// [`stable_witness`] is read from the environment once per process, so these
/// exercise [`box_support_centroid`] directly — it is the whole of the change,
/// and the env gate only decides whether [`single_point`] calls it.
#[cfg(test)]
mod stable_witness_tests {
    use super::*;

    fn unit_box() -> Vec3 {
        Vec3::new(0.3, 0.1, 0.02)
    }

    /// A direction along a face normal is supported *equally* by all four
    /// corners of that face. `Geometry::support` has to pick one of them; the
    /// centroid is the face centre, which is the honest answer.
    #[test]
    fn a_face_aligned_direction_gives_the_face_centre() {
        let h = unit_box();
        let c = box_support_centroid(&h, &Vec3::zeros(), &Mat3::identity(), &Vec3::z());
        assert!(
            (c - Vec3::new(0.0, 0.0, h.z)).norm() < 1e-12,
            "expected the +z face centre, got {c:?}"
        );
    }

    /// The regression this whole change exists for.
    ///
    /// `Geometry::support` chooses a box corner by the *sign* of each component
    /// of the direction in the box's local frame. Along a face normal the two
    /// in-plane components are exact cancellation noise, so their signs are
    /// decided by the last bits — and flipping one walks the witness point to a
    /// different corner, a whole half-extent away. On the K1 skate stance that
    /// moved contact points by up to 24 cm under a `1e-9 N·m` torque change,
    /// which is what made two consecutive contacted steps non-differentiable.
    #[test]
    fn last_bit_wobble_in_the_direction_does_not_move_the_witness_point() {
        let h = unit_box();
        let (pos, rot) = (Vec3::zeros(), Mat3::identity());
        let geom = Geometry::Box { half_extents: h };

        let base = box_support_centroid(&h, &pos, &rot, &Vec3::z());
        let mut worst_stable = 0.0f64;
        let mut worst_support = 0.0f64;
        let raw0 = geom.support(&Vec3::z(), &pos, &rot);

        // Every sign combination of a last-bits in-plane component: exactly the
        // perturbations a 1e-15 change to the pose delivers.
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                let dir = Vec3::new(sx * 1e-17, sy * 1e-17, 1.0);
                let c = box_support_centroid(&h, &pos, &rot, &dir);
                worst_stable = worst_stable.max((c - base).norm());
                let raw = geom.support(&dir, &pos, &rot);
                worst_support = worst_support.max((raw - raw0).norm());
            }
        }

        assert!(
            worst_stable < 1e-12,
            "the stable witness moved by {worst_stable:.3e} m under last-bit noise"
        );
        // And the same wobble really does throw the raw support function across
        // the box — the test would be vacuous if it did not.
        assert!(
            worst_support > 0.1,
            "expected the raw support point to jump; it moved {worst_support:.3e} m"
        );
    }

    /// An edge-on direction is supported by two corners: the answer is the edge
    /// midpoint, not either end.
    #[test]
    fn an_edge_on_direction_gives_the_edge_midpoint() {
        let h = unit_box();
        let d = Vec3::new(0.0, 1.0, 1.0).normalize();
        let c = box_support_centroid(&h, &Vec3::zeros(), &Mat3::identity(), &d);
        assert!(
            (c - Vec3::new(0.0, h.y, h.z)).norm() < 1e-12,
            "expected the +y+z edge midpoint, got {c:?}"
        );
    }

    /// Where the argmax is genuinely decided, nothing changes: the corner is
    /// still the corner. The fix must not blunt a real corner contact.
    #[test]
    fn a_decided_corner_direction_is_unchanged() {
        let h = unit_box();
        let (pos, rot) = (Vec3::zeros(), Mat3::identity());
        let d = Vec3::new(1.0, 1.0, 1.0).normalize();
        let c = box_support_centroid(&h, &pos, &rot, &d);
        let raw = Geometry::Box { half_extents: h }.support(&d, &pos, &rot);
        assert!(
            (c - raw).norm() < 1e-12,
            "corner contact should match the raw support point: {c:?} vs {raw:?}"
        );
    }

    /// The centroid is taken in the box's own frame, so it rotates with the box
    /// rather than being computed in world axes.
    #[test]
    fn the_centroid_rides_the_box_frame() {
        let h = unit_box();
        let rot =
            phyz_math::Quat::from_axis_angle(Vec3::x(), std::f64::consts::FRAC_PI_2).to_matrix();
        let pos = Vec3::new(1.0, 2.0, 3.0);
        // The box's local +z now points along world −y.
        let c = box_support_centroid(&h, &pos, &rot, &(-Vec3::y()));
        let expect = pos + rot * Vec3::new(0.0, 0.0, h.z);
        assert!(
            (c - expect).norm() < 1e-12,
            "expected {expect:?}, got {c:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box(h: f64) -> Geometry {
        Geometry::Box {
            half_extents: Vec3::new(h, h, h),
        }
    }

    /// A box resting flat on a larger box must produce a four-point manifold —
    /// the property that makes a stack stand up.
    #[test]
    fn box_on_box_gives_four_points() {
        let ground = Geometry::Box {
            half_extents: Vec3::new(5.0, 5.0, 0.5),
        };
        let b = unit_box(0.5);
        let m = contact_manifold(
            &ground,
            &b,
            &Vec3::zeros(),
            &Mat3::identity(),
            // Box bottom at z = 0.49, ground top at 0.5 → 0.01 overlap.
            &Vec3::new(0.0, 0.0, 0.99),
            &Mat3::identity(),
        )
        .expect("overlapping boxes produce a manifold");

        assert_eq!(m.points.len(), 4, "expected a 4-point face manifold");
        assert!(
            (m.normal - Vec3::z()).norm() < 1e-6,
            "normal ~ +z, got {:?}",
            m.normal
        );
        for p in &m.points {
            assert!(
                (p.depth - 0.01).abs() < 1e-6,
                "each corner overlaps by 0.01, got {}",
                p.depth
            );
            // Corners of the upper box, on the midsurface z = 0.495.
            assert!((p.position.z - 0.495).abs() < 1e-6);
            assert!((p.position.x.abs() - 0.5).abs() < 1e-6);
            assert!((p.position.y.abs() - 0.5).abs() < 1e-6);
        }
    }

    /// The four points must actually span the patch, not cluster.
    #[test]
    fn box_on_box_points_are_spread() {
        let ground = Geometry::Box {
            half_extents: Vec3::new(5.0, 5.0, 0.5),
        };
        let b = unit_box(0.5);
        let m = contact_manifold(
            &ground,
            &b,
            &Vec3::zeros(),
            &Mat3::identity(),
            &Vec3::new(0.0, 0.0, 0.99),
            &Mat3::identity(),
        )
        .unwrap();
        let centroid =
            m.points.iter().fold(Vec3::zeros(), |a, p| a + p.position) / m.points.len() as f64;
        for p in &m.points {
            let r = (p.position - centroid).norm();
            assert!(r > 0.5, "point too close to centroid: {r}");
        }
    }

    /// A sphere touching a box really does touch at one point.
    #[test]
    fn sphere_on_box_gives_one_point() {
        let ground = Geometry::Box {
            half_extents: Vec3::new(5.0, 5.0, 0.5),
        };
        let s = Geometry::Sphere { radius: 0.25 };
        let m = contact_manifold(
            &ground,
            &s,
            &Vec3::zeros(),
            &Mat3::identity(),
            &Vec3::new(0.0, 0.0, 0.74),
            &Mat3::identity(),
        )
        .expect("sphere overlaps box");
        assert_eq!(m.points.len(), 1);
        assert!((m.normal - Vec3::z()).norm() < 1e-3);
        assert!((m.points[0].depth - 0.01).abs() < 1e-3);
        // The point sits on the surface, not at a body centre.
        assert!(
            (m.points[0].position.z - 0.495).abs() < 1e-3,
            "contact point z = {}",
            m.points[0].position.z
        );
    }

    /// A box tilted off-axis contacts on an edge, not a face: fewer points,
    /// and they must still lie on the true contact edge.
    #[test]
    fn tilted_box_contacts_on_an_edge() {
        let ground = Geometry::Box {
            half_extents: Vec3::new(5.0, 5.0, 0.5),
        };
        let b = unit_box(0.5);
        let angle = 0.3_f64;
        let (s, c) = angle.sin_cos();
        // Rotate about +x, so the contact edge runs along x.
        let rot = Mat3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c);
        let drop = 0.5 * (c + s) - 0.01;
        let m = contact_manifold(
            &ground,
            &b,
            &Vec3::zeros(),
            &Mat3::identity(),
            &Vec3::new(0.0, 0.0, 0.5 + drop),
            &rot,
        )
        .expect("tilted box overlaps ground");
        assert!((m.normal - Vec3::z()).norm() < 1e-6);
        assert!(
            m.points.len() <= 2,
            "edge contact should not report a full face, got {}",
            m.points.len()
        );
        for p in &m.points {
            assert!(p.depth > 0.0 && p.depth < 0.05, "depth {}", p.depth);
        }
    }

    /// **The tilted-reference regression.** A small box resting on a *tilted*
    /// slab, with its own sole parallel to the slab, is a face-on-face contact
    /// and must produce a face manifold whose depths are the real overlap.
    ///
    /// It did not. `clip_faces` measured each clipped vertex's separation along
    /// the contact normal from a vertex of a reference face it always took from
    /// shape `A` — here the slab, tilted 15° away from that normal and so
    /// spanning 2.6 m along it. Every vertex read metres of gap against a
    /// centimetre of real overlap, `sep > margin` rejected the whole polygon,
    /// and the pair fell through to a single support vertex.
    ///
    /// The scale is the point: the error is the *reference face's own size*,
    /// which is why the same bug read as "3–44 mm beyond the plane" on 0.15 m
    /// foot boxes and as a collapsed stack on 0.1 m ones.
    #[test]
    fn a_tilted_reference_face_still_gives_a_face_manifold() {
        let tilt = 15f64.to_radians();
        // A big slab rotated about +x, so its top face leans in y and its
        // highest feature is the long edge at local y = +5.
        let (c, s) = (tilt.cos(), tilt.sin());
        let slab_rot = Mat3::new(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c);
        let slab = Geometry::Box {
            half_extents: Vec3::new(5.0, 5.0, 0.5),
        };

        // A small axis-aligned box sitting on that edge: its flat underside is
        // the contact plane, so the separating normal is +z — which the slab's
        // own support face misses by the full 15°.
        let rider = unit_box(0.5);
        let edge = slab_rot * Vec3::new(0.0, -5.0, 0.5);
        let rider_pos = Vec3::new(0.0, edge.y, edge.z + 0.5 - 0.01);

        let m = contact_manifold(
            &slab,
            &rider,
            &Vec3::zeros(),
            &slab_rot,
            &rider_pos,
            &Mat3::identity(),
        )
        .expect("a box resting on a slab's edge is a contact");

        assert!(
            (m.normal - Vec3::z()).norm() < 1e-6,
            "the normal is the rider's own face normal, +z; got {:?}",
            m.normal
        );
        assert!(
            m.points.len() >= 2,
            "an edge-on-face contact must give at least the two ends of the \
             edge, got {} point(s) — one point has no resistance to rolling \
             along it, and getting one here means the clipped polygon was \
             rejected and `single_point` took over",
            m.points.len()
        );
        let deepest = m
            .points
            .iter()
            .map(|p| p.depth)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (deepest - 0.01).abs() < 1e-6,
            "the deepest point overlaps by 0.01 m; got {deepest} — an error \
             the size of the slab's own extent along the normal (2.6 m here) \
             is the bug this pins"
        );
    }

    /// Disjoint shapes produce nothing.
    #[test]
    fn separated_boxes_have_no_manifold() {
        let b = unit_box(0.5);
        assert!(
            contact_manifold(
                &b,
                &b,
                &Vec3::zeros(),
                &Mat3::identity(),
                &Vec3::new(0.0, 0.0, 5.0),
                &Mat3::identity(),
            )
            .is_none()
        );
    }
}
