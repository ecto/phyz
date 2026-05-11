//! EPA (Expanding Polytope Algorithm) for penetration depth calculation.

use crate::geometry::Geometry;
use phyz_math::{Mat3, Vec3};

/// Triangle face in EPA polytope.
#[derive(Clone)]
struct Face {
    indices: [usize; 3],
    normal: Vec3,
    distance: f64,
}

/// Compute penetration depth and contact normal using EPA.
/// Returns (depth, normal) if penetrating, None otherwise.
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

/// EPA with rotation matrices.
///
/// Returns `None` rather than panicking when the polytope can't be built or
/// expanded reliably — degenerate initial simplex, NaN support points, or
/// failure to converge within the iteration cap.
pub fn epa_penetration_rot(
    geom_a: &Geometry,
    geom_b: &Geometry,
    pos_a: &Vec3,
    pos_b: &Vec3,
    rot_a: &Mat3,
    rot_b: &Mat3,
) -> Option<(f64, Vec3)> {
    let support = |d: &Vec3| {
        let sa = geom_a.support(d, pos_a, rot_a);
        let sb = geom_b.support(&(-*d), pos_b, rot_b);
        sa - sb
    };

    // Initial tetrahedron directions
    let dirs = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(-1.0, -1.0, -1.0).normalize(),
    ];

    let mut points: Vec<Vec3> = Vec::new();
    for d in &dirs {
        let p = support(d);
        if !is_finite(&p) {
            return None;
        }
        points.push(p);
    }

    // Reject a degenerate initial tetrahedron (coplanar / coincident support
    // points). EPA assumes the polytope encloses the origin; with a flat
    // initial simplex the expansion can produce zero-area faces and bogus
    // normals on glancing curved contacts.
    if !is_nondegenerate_tetra(&points) {
        return None;
    }

    let mut faces = vec![
        Face::new(&points, [0, 1, 2]),
        Face::new(&points, [0, 3, 1]),
        Face::new(&points, [0, 2, 3]),
        Face::new(&points, [1, 3, 2]),
    ];

    const MAX_ITERATIONS: usize = 64;
    const MAX_POLYTOPE_FACES: usize = 1024;
    const TOLERANCE: f64 = 1e-6;

    for _ in 0..MAX_ITERATIONS {
        let closest_idx = match closest_face(&faces) {
            Some(idx) => idx,
            None => return None,
        };

        let closest_distance = faces[closest_idx].distance;
        let closest_normal = faces[closest_idx].normal;

        let s = support(&closest_normal);
        if !is_finite(&s) {
            return None;
        }
        let dist = s.dot(&closest_normal);
        if !dist.is_finite() {
            return None;
        }

        if dist - closest_distance < TOLERANCE {
            if !closest_distance.is_finite() || !is_finite(&closest_normal) {
                return None;
            }
            return Some((closest_distance, closest_normal));
        }

        // Expand polytope: remove faces visible from s, collect their edges.
        let mut edges = Vec::new();
        faces.retain(|face| {
            if is_visible(&points, face, &s) {
                for i in 0..3 {
                    let edge = (face.indices[i], face.indices[(i + 1) % 3]);
                    edges.push(edge);
                }
                false
            } else {
                true
            }
        });

        // Drop shared (interior) edges; what remains is the silhouette.
        edges.sort_by_key(|&(a, b)| (a.min(b), a.max(b)));
        let mut unique_edges = Vec::new();
        let mut i = 0;
        while i < edges.len() {
            let e = edges[i];
            let e_sorted = (e.0.min(e.1), e.0.max(e.1));
            if i + 1 < edges.len() {
                let next = edges[i + 1];
                let next_sorted = (next.0.min(next.1), next.0.max(next.1));
                if e_sorted == next_sorted {
                    i += 2;
                    continue;
                }
            }
            unique_edges.push(e);
            i += 1;
        }

        if unique_edges.is_empty() {
            return None;
        }

        let new_idx = points.len();
        points.push(s);

        for (a, b) in unique_edges {
            faces.push(Face::new(&points, [a, b, new_idx]));
        }

        if faces.len() > MAX_POLYTOPE_FACES {
            return None;
        }
    }

    None
}

/// Pick the face with the smallest finite distance to the origin. Faces with
/// non-finite distances (NaN/Inf from degenerate triangles) are skipped — a
/// naive `partial_cmp().unwrap()` would panic on NaN.
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

/// True iff the four points span a non-degenerate tetrahedron (signed volume
/// well above floating-point noise).
fn is_nondegenerate_tetra(points: &[Vec3]) -> bool {
    if points.len() < 4 {
        return false;
    }
    let a = points[0];
    let b = points[1];
    let c = points[2];
    let d = points[3];
    let volume = (b - a).cross(&(c - a)).dot(&(d - a));
    volume.is_finite() && volume.abs() > 1e-12
}

impl Face {
    fn new(points: &[Vec3], indices: [usize; 3]) -> Self {
        let a = points[indices[0]];
        let b = points[indices[1]];
        let c = points[indices[2]];
        let ab = b - a;
        let ac = c - a;
        let cross = ab.cross(&ac);
        let norm = cross.norm();
        // Degenerate (zero-area) triangle: mark distance = +inf so the
        // closest-face search skips it instead of selecting a face whose
        // direction is ill-defined.
        if !norm.is_finite() || norm <= 1e-10 {
            return Self {
                indices,
                normal: Vec3::new(0.0, 0.0, 0.0),
                distance: f64::INFINITY,
            };
        }
        let mut normal = cross / norm;
        let mut distance = normal.dot(&a);
        if distance < 0.0 {
            normal = -normal;
            distance = -distance;
        }
        if !distance.is_finite() || !is_finite(&normal) {
            return Self {
                indices,
                normal: Vec3::new(0.0, 0.0, 0.0),
                distance: f64::INFINITY,
            };
        }
        Self {
            indices,
            normal,
            distance,
        }
    }
}

fn is_visible(points: &[Vec3], face: &Face, point: &Vec3) -> bool {
    let a = points[face.indices[0]];
    let to_point = point - a;
    to_point.dot(&face.normal) > 0.0
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

    /// A flat (coplanar) mesh has zero Minkowski-difference volume; the
    /// initial tetrahedron is degenerate. EPA must bail rather than panic.
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
        // Must not panic; result may be Some or None depending on the
        // degenerate initial simplex check, but the call must complete.
        let _ = epa_penetration_rot(&a, &b, &pa, &pb, &rot, &rot);
    }

    /// Sphere-sphere penetration still produces a sensible (depth, normal).
    #[test]
    fn sphere_sphere_penetrating_still_works() {
        let s = Geometry::Sphere { radius: 1.0 };
        let pa = Vec3::zeros();
        let pb = Vec3::new(1.5, 0.0, 0.0);
        let (depth, _normal) = epa_penetration(&s, &s, &pa, &pb).expect("EPA returns a result");
        assert!(depth.is_finite());
        assert!((depth - 0.5).abs() < 0.1, "depth ~ 0.5, got {}", depth);
    }
}
