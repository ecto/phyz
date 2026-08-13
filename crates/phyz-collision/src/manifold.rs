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

use crate::geometry::Geometry;
use crate::gjk::GjkOutcome;
use phyz_math::{Mat3, Vec3};

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
        (Some(ref_face), Some(inc_face)) => clip_faces(&ref_face, &inc_face, &normal, margin)
            .unwrap_or_else(|| {
                vec![single_point(
                    geom_a, geom_b, pos_a, rot_a, pos_b, rot_b, &normal, depth,
                )]
            }),
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
    let wa = geom_a.support(normal, pos_a, rot_a);
    let wb = geom_b.support(&(-*normal), pos_b, rot_b);
    ManifoldPoint {
        position: (wa + wb) * 0.5,
        depth,
    }
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
            return None;
        }
    }

    // Reference plane offset along the shared normal.
    let ref_point = reference.point;

    let out: Vec<ManifoldPoint> = poly
        .iter()
        .filter_map(|p| {
            // Separation of the incident vertex from the reference plane,
            // measured along the contact normal. Negative separation is
            // penetration; a positive separation up to `margin` is a gap
            // inside the band and is kept, with a negative depth.
            let sep = (p - ref_point).dot(normal);
            if sep > margin.max(1e-9) {
                return None;
            }
            Some(ManifoldPoint {
                // Put the point on the midsurface.
                position: p - *normal * (sep * 0.5),
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
