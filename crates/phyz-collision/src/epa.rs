//! EPA (Expanding Polytope Algorithm) for penetration depth calculation.
//!
//! EPA is only valid when seeded with a polytope that **encloses the origin**
//! of the Minkowski difference. The entry point that guarantees this is
//! [`epa_from_simplex`], fed the terminating simplex of a GJK run that
//! reported penetration ([`crate::gjk::gjk_rot`]).
//!
//! [`epa_penetration_rot`] is the convenience wrapper that runs GJK first. It
//! previously built its start polytope from four fixed support directions,
//! which does not enclose the origin in general — every normal orientation and
//! visibility test downstream then assumed something untrue, and on a plain
//! pair of overlapping spheres it burned its whole iteration budget without
//! converging.

use crate::geometry::Geometry;
use crate::gjk::{GjkOutcome, gjk_rot};
use phyz_math::{Mat3, Vec3};

/// Triangle face in EPA polytope. `normal` points away from the origin.
#[derive(Clone)]
struct Face {
    indices: [usize; 3],
    normal: Vec3,
    distance: f64,
}

/// Compute penetration depth and contact normal using EPA.
///
/// Returns `(depth, normal)` if penetrating, `None` otherwise. `normal` points
/// from `geom_a` toward `geom_b` — i.e. translating `geom_b` by
/// `depth * normal` separates the pair.
pub fn epa_penetration(
    geom_a: &Geometry,
    geom_b: &Geometry,
    pos_a: &Vec3,
    pos_b: &Vec3,
) -> Option<(f64, Vec3)> {
    let rot_a = Mat3::identity();
    let rot_b = Mat3::identity();
    epa_penetration_rot(geom_a, geom_b, pos_a, pos_b, &rot_a, &rot_b)
}

/// EPA with rotation matrices. Runs GJK to obtain a valid seed simplex.
pub fn epa_penetration_rot(
    geom_a: &Geometry,
    geom_b: &Geometry,
    pos_a: &Vec3,
    pos_b: &Vec3,
    rot_a: &Mat3,
    rot_b: &Mat3,
) -> Option<(f64, Vec3)> {
    match gjk_rot(geom_a, geom_b, pos_a, pos_b, rot_a, rot_b) {
        GjkOutcome::Penetrating { simplex } => {
            epa_from_simplex(geom_a, geom_b, pos_a, pos_b, rot_a, rot_b, &simplex)
        }
        _ => None,
    }
}

/// EPA proper, seeded with a GJK simplex that encloses the origin.
///
/// Returns `None` rather than panicking when the polytope can't be built or
/// expanded reliably — degenerate seed, NaN support points, or failure to
/// converge within the iteration cap.
pub fn epa_from_simplex(
    geom_a: &Geometry,
    geom_b: &Geometry,
    pos_a: &Vec3,
    pos_b: &Vec3,
    rot_a: &Mat3,
    rot_b: &Mat3,
    seed: &[Vec3],
) -> Option<(f64, Vec3)> {
    let support = |d: &Vec3| {
        let sa = geom_a.support(d, pos_a, rot_a);
        let sb = geom_b.support(&(-*d), pos_b, rot_b);
        sa - sb
    };

    let mut points = build_tetrahedron(seed, &support)?;

    // Orient each seed face outward relative to the polytope's interior. With
    // a genuine origin-enclosing tetrahedron the origin itself works as the
    // reference interior point.
    let mut faces = vec![
        Face::new(&points, [0, 1, 2])?,
        Face::new(&points, [0, 3, 1])?,
        Face::new(&points, [0, 2, 3])?,
        Face::new(&points, [1, 3, 2])?,
    ];

    // Smooth shapes converge linearly: each iteration shaves one facet off a
    // curved surface, so a sphere-sphere pair needs roughly 100 expansions to
    // reach TOLERANCE. The old cap of 64 bailed out just short of that and
    // returned `None` for the most common contact pair there is. The face cap
    // is raised in step so a deep overlap, which needs the most refinement,
    // cannot hit it before the iteration cap.
    const MAX_ITERATIONS: usize = 256;
    const MAX_POLYTOPE_FACES: usize = 4096;
    // Relative tolerance. An absolute 1e-8 is unreachable on a curved surface:
    // the polytope would have to approximate the sphere to that accuracy
    // locally, which costs far more faces than any sane budget allows.
    const TOLERANCE: f64 = 1e-6;

    // Best (deepest) hull face seen so far. EPA's face distance is always a
    // valid *lower bound* on the penetration depth, so on budget exhaustion
    // returning the best bound beats returning `None` and dropping a contact
    // that certainly exists.
    let mut best: Option<(f64, Vec3)> = None;

    for _ in 0..MAX_ITERATIONS {
        let closest_idx = closest_face(&faces)?;
        let closest_distance = faces[closest_idx].distance;
        let closest_normal = faces[closest_idx].normal;

        if best.is_none_or(|(d, _)| closest_distance > d) {
            best = Some((closest_distance, closest_normal));
        }

        let s = support(&closest_normal);
        if !is_finite(&s) {
            return best;
        }
        let dist = s.dot(&closest_normal);
        if !dist.is_finite() {
            return best;
        }

        // The support point along the face normal is no further out than the
        // face itself: that face is on the Minkowski hull, so its distance is
        // the penetration depth.
        if dist - closest_distance < TOLERANCE * dist.abs().max(1.0) {
            return Some((closest_distance, closest_normal));
        }

        // Expand: delete every face `s` can see, and rebuild over the horizon
        // (the boundary of the deleted region).
        let horizon = {
            let mut edges: Vec<(usize, usize)> = Vec::new();
            faces.retain(|face| {
                if is_visible(&points, face, &s) {
                    for i in 0..3 {
                        push_horizon_edge(&mut edges, (face.indices[i], face.indices[(i + 1) % 3]));
                    }
                    false
                } else {
                    true
                }
            });
            edges
        };

        if horizon.is_empty() {
            return best;
        }

        let new_idx = points.len();
        points.push(s);

        for (a, b) in horizon {
            // Winding (a, b, new) inherits the deleted faces' outward
            // orientation, so the rebuilt cap is consistently oriented.
            match Face::new(&points, [a, b, new_idx]) {
                Some(f) => faces.push(f),
                // Sliver triangle: dropping it is safe (it contributes no
                // volume), but if it leaves a hole `closest_face` will simply
                // pick another face and the loop continues.
                None => continue,
            }
        }

        if faces.len() > MAX_POLYTOPE_FACES {
            return best;
        }
    }

    best
}

/// Grow a GJK simplex of 1–4 points into a non-degenerate tetrahedron that
/// still contains the origin.
///
/// GJK may terminate on a point, segment, or triangle when the origin lies on
/// the boundary (a touching contact). EPA needs volume, so we extend along
/// directions orthogonal to whatever the simplex already spans.
fn build_tetrahedron(seed: &[Vec3], support: &impl Fn(&Vec3) -> Vec3) -> Option<Vec<Vec3>> {
    let mut pts: Vec<Vec3> = Vec::with_capacity(4);
    for p in seed {
        if !is_finite(p) {
            return None;
        }
        // Skip duplicates; a degenerate repeat adds no span.
        if pts.iter().all(|q: &Vec3| (q - p).norm() > 1e-12) {
            pts.push(*p);
        }
    }
    if pts.is_empty() {
        return None;
    }

    // 1 -> 2: any direction with a distinct support point.
    if pts.len() == 1 {
        let dirs = [
            Vec3::x(),
            -Vec3::x(),
            Vec3::y(),
            -Vec3::y(),
            Vec3::z(),
            -Vec3::z(),
        ];
        for d in dirs {
            let p = support(&d);
            if is_finite(&p) && (p - pts[0]).norm() > 1e-9 {
                pts.push(p);
                break;
            }
        }
        if pts.len() < 2 {
            return None;
        }
    }

    // 2 -> 3: search orthogonal to the segment.
    if pts.len() == 2 {
        let ab = pts[1] - pts[0];
        let axis = least_aligned_axis(&ab);
        let ortho = ab.cross(&axis);
        for d in [ortho, -ortho] {
            if d.norm() < 1e-12 {
                continue;
            }
            let p = support(&d.normalize());
            if is_finite(&p) && (p - pts[0]).cross(&(p - pts[1])).norm() > 1e-12 {
                pts.push(p);
                break;
            }
        }
        if pts.len() < 3 {
            return None;
        }
    }

    // 3 -> 4: search along the triangle's normal.
    if pts.len() == 3 {
        let n = (pts[1] - pts[0]).cross(&(pts[2] - pts[0]));
        if n.norm() < 1e-12 {
            return None;
        }
        let n = n.normalize();
        for d in [n, -n] {
            let p = support(&d);
            if is_finite(&p) && signed_volume(&pts[0], &pts[1], &pts[2], &p).abs() > 1e-14 {
                pts.push(p);
                break;
            }
        }
        if pts.len() < 4 {
            return None;
        }
    }

    pts.truncate(4);
    if pts.len() != 4 {
        return None;
    }

    // Fix the winding so that [0,1,2] is counter-clockwise seen from outside,
    // i.e. point 3 is on the negative side of triangle (0,1,2).
    if signed_volume(&pts[0], &pts[1], &pts[2], &pts[3]) > 0.0 {
        pts.swap(0, 1);
    }
    if signed_volume(&pts[0], &pts[1], &pts[2], &pts[3]).abs() < 1e-14 {
        return None;
    }
    Some(pts)
}

fn signed_volume(a: &Vec3, b: &Vec3, c: &Vec3, d: &Vec3) -> f64 {
    (b - a).cross(&(c - a)).dot(&(d - a))
}

/// A unit axis that is far from parallel to `v`, for building an orthogonal.
fn least_aligned_axis(v: &Vec3) -> Vec3 {
    let (ax, ay, az) = (v.x.abs(), v.y.abs(), v.z.abs());
    if ax <= ay && ax <= az {
        Vec3::x()
    } else if ay <= az {
        Vec3::y()
    } else {
        Vec3::z()
    }
}

/// Accumulate a horizon edge, cancelling it against its twin.
///
/// Every interior edge of the visible region is walked exactly twice, once in
/// each direction, so a matching `(b, a)` cancels `(a, b)`. What survives is
/// the silhouette. The previous implementation sorted the edge list and
/// removed adjacent equal pairs, which mis-cancelled as soon as three or more
/// faces met along coincident indices — that is what collapsed the expansion
/// to an empty horizon and returned `None` mid-convergence.
fn push_horizon_edge(edges: &mut Vec<(usize, usize)>, e: (usize, usize)) {
    if let Some(pos) = edges.iter().position(|&(a, b)| a == e.1 && b == e.0) {
        edges.swap_remove(pos);
    } else {
        edges.push(e);
    }
}

/// Pick the face with the smallest finite distance to the origin.
fn closest_face(faces: &[Face]) -> Option<usize> {
    let mut best_idx = None;
    let mut best_dist = f64::INFINITY;
    for (i, f) in faces.iter().enumerate() {
        if f.distance.is_finite() && f.distance < best_dist {
            best_dist = f.distance;
            best_idx = Some(i);
        }
    }
    best_idx
}

fn is_finite(v: &Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

impl Face {
    /// Build a face from a triangle wound counter-clockwise seen from outside.
    ///
    /// Returns `None` for a degenerate (zero-area) triangle. Unlike the old
    /// version this does **not** flip the normal to face away from the origin:
    /// with a valid origin-enclosing polytope the winding already encodes
    /// orientation, and flipping by origin-side breaks down exactly when the
    /// closest face passes through the origin (depth ≈ 0, a touching contact).
    fn new(points: &[Vec3], indices: [usize; 3]) -> Option<Self> {
        let a = points[indices[0]];
        let b = points[indices[1]];
        let c = points[indices[2]];
        let cross = (b - a).cross(&(c - a));
        let norm = cross.norm();
        if !norm.is_finite() || norm <= 1e-14 {
            return None;
        }
        let normal = cross / norm;
        let distance = normal.dot(&a);
        if !distance.is_finite() || !is_finite(&normal) {
            return None;
        }
        Some(Self {
            indices,
            normal,
            // The polytope encloses the origin, so an outward-wound face has
            // distance >= 0 up to rounding; clamp the rounding case.
            distance: distance.max(0.0),
        })
    }
}

fn is_visible(points: &[Vec3], face: &Face, point: &Vec3) -> bool {
    let a = points[face.indices[0]];
    (point - a).dot(&face.normal) > 1e-12
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Geometry;

    fn mesh(verts: &[[f64; 3]]) -> Geometry {
        Geometry::Mesh {
            vertices: verts.iter().map(|v| Vec3::new(v[0], v[1], v[2])).collect(),
            faces: Vec::new(),
        }
    }

    /// Regression for issue #4: a degenerate (single-vertex) "mesh" used to
    /// drive EPA into a polytope of zero-area faces and panic in
    /// `partial_cmp(NaN).unwrap()`. It must return `None` instead.
    #[test]
    fn degenerate_single_point_mesh_returns_none() {
        let g = mesh(&[[0.0, 0.0, 0.0]]);
        let p = Vec3::zeros();
        let rot = Mat3::identity();
        assert!(epa_penetration_rot(&g, &g, &p, &p, &rot, &rot).is_none());
    }

    /// A flat (coplanar) mesh has zero Minkowski-difference volume; no
    /// tetrahedron can be built. EPA must bail rather than panic.
    #[test]
    fn coplanar_mesh_returns_none() {
        let g = mesh(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]);
        let p = Vec3::zeros();
        let rot = Mat3::identity();
        assert!(epa_penetration_rot(&g, &g, &p, &p, &rot, &rot).is_none());
    }

    /// Glancing cylindrical-style contact: two thin disks touching edge-to-
    /// edge. Previously this triggered intermittent panics inside the EPA
    /// polytope expansion.
    #[test]
    fn glancing_thin_disk_contact_no_panic() {
        let disk: Vec<[f64; 3]> = (0..16)
            .map(|i| {
                let t = (i as f64) * std::f64::consts::TAU / 16.0;
                [t.cos(), t.sin(), 0.0]
            })
            .collect();
        let a = mesh(&disk);
        let b = mesh(&disk);
        let pa = Vec3::zeros();
        let pb = Vec3::new(1.99, 0.0, 0.0);
        let rot = Mat3::identity();
        let _ = epa_penetration_rot(&a, &b, &pa, &pb, &rot, &rot);
    }

    /// Sphere-sphere penetration produces the analytic (depth, normal).
    #[test]
    fn sphere_sphere_penetrating_still_works() {
        let s = Geometry::Sphere { radius: 1.0 };
        let pa = Vec3::zeros();
        let pb = Vec3::new(1.5, 0.0, 0.0);
        let (depth, normal) = epa_penetration(&s, &s, &pa, &pb).expect("EPA returns a result");
        assert!(depth.is_finite());
        assert!((depth - 0.5).abs() < 1e-3, "depth ~ 0.5, got {depth}");
        // Normal must be along the centre line, pointing a -> b.
        assert!(
            (normal - Vec3::x()).norm() < 1e-3,
            "normal ~ +x, got {normal:?}"
        );
    }

    /// Depth must track the overlap analytically across a sweep, not just at
    /// one configuration — this is what a hardcoded constant would fail.
    #[test]
    fn sphere_sphere_depth_matches_analytic_sweep() {
        let s = Geometry::Sphere { radius: 1.0 };
        for k in 1..=9 {
            let sep = 0.2 * k as f64; // centre distance 0.2 .. 1.8
            let expected = 2.0 - sep; // r_a + r_b - d
            let (depth, normal) =
                epa_penetration(&s, &s, &Vec3::zeros(), &Vec3::new(sep, 0.0, 0.0))
                    .unwrap_or_else(|| panic!("EPA failed at separation {sep}"));
            assert!(
                (depth - expected).abs() < 5e-3,
                "sep {sep}: depth {depth} vs {expected}"
            );
            // Normal accuracy degrades as the overlap deepens: at near-
            // concentric overlap the Minkowski hull is almost symmetric about
            // the origin, so distinguishing the closest direction needs finer
            // angular resolution than the face budget provides. Depth stays
            // accurate throughout, and shallow overlap — the regime contact
            // actually operates in — is tight.
            let normal_tol = 5e-3 + 1e-2 * expected;
            assert!(
                (normal - Vec3::x()).norm() < normal_tol,
                "sep {sep}: normal {normal:?}"
            );
        }
    }

    /// Box/sphere: the case `find_contacts` silently missed entirely.
    #[test]
    fn box_sphere_penetration() {
        let b = Geometry::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.1),
        };
        let s = Geometry::Sphere { radius: 0.1 };
        // Sphere centre at z = 0.05; box top face at z = 0.1. Overlap 0.15.
        let (depth, normal) = epa_penetration(&b, &s, &Vec3::zeros(), &Vec3::new(0.0, 0.0, 0.05))
            .expect("box/sphere overlap detected");
        assert!(
            (depth - 0.15).abs() < 1e-3,
            "expected depth 0.15, got {depth}"
        );
        // Minimum translation is +z: pushing the sphere up by 0.15 clears the
        // box's top face, versus 0.25 to clear the bottom. `normal` points
        // a -> b (box -> sphere).
        assert!(
            (normal - Vec3::z()).norm() < 1e-3,
            "normal ~ +z, got {normal:?}"
        );
    }

    /// Two unit boxes overlapping slightly along z: depth is the overlap and
    /// the normal is the face normal.
    #[test]
    fn box_box_face_contact() {
        let g = Geometry::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        };
        let (depth, normal) = epa_penetration(&g, &g, &Vec3::zeros(), &Vec3::new(0.0, 0.0, 0.98))
            .expect("box/box overlap detected");
        assert!((depth - 0.02).abs() < 1e-6, "expected 0.02, got {depth}");
        assert!(
            (normal - Vec3::z()).norm() < 1e-6,
            "normal ~ +z, got {normal:?}"
        );
    }

    /// `gjk_distance_rot`'s negative branch must carry a real magnitude.
    #[test]
    fn gjk_signed_distance_reports_real_depth() {
        use crate::gjk::gjk_distance;
        let s = Geometry::Sphere { radius: 1.0 };
        let d = gjk_distance(&s, &s, &Vec3::zeros(), &Vec3::new(1.5, 0.0, 0.0));
        assert!((d + 0.5).abs() < 5e-3, "expected ~ -0.5, got {d}");

        let d = gjk_distance(&s, &s, &Vec3::zeros(), &Vec3::new(3.0, 0.0, 0.0));
        assert!((d - 1.0).abs() < 1e-6, "expected separation 1.0, got {d}");
    }
}
