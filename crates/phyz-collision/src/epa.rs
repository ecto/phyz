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
pub fn epa_penetration_rot(
    geom_a: &Geometry,
    geom_b: &Geometry,
    pos_a: &Vec3,
    pos_b: &Vec3,
    rot_a: &Mat3,
    rot_b: &Mat3,
) -> Option<(f64, Vec3)> {
    // Start with a simplex tetrahedron from GJK
    // For simplicity, we'll build a small initial tetrahedron
    let support = |d: &Vec3| {
        let sa = geom_a.support(d, pos_a, rot_a);
        let sb = geom_b.support(&-d, pos_b, rot_b);
        sa - sb
    };

    // Initial tetrahedron directions
    let dirs = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(-1.0, -1.0, -1.0).normalize(),
    ];

    let mut points = Vec::new();
    for d in &dirs {
        points.push(support(d));
    }

    // Build initial polytope faces
    let mut faces = vec![
        Face::new(&points, [0, 1, 2]),
        Face::new(&points, [0, 3, 1]),
        Face::new(&points, [0, 2, 3]),
        Face::new(&points, [1, 3, 2]),
    ];

    const MAX_ITERATIONS: usize = 64;
    const TOLERANCE: f64 = 1e-6;

    for _ in 0..MAX_ITERATIONS {
        // Find closest face to origin
        let closest_idx = faces
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.distance.partial_cmp(&b.distance).unwrap())
            .map(|(i, _)| i)?;

        let closest = &faces[closest_idx];

        // Get new support point in direction of closest face normal
        let s = support(&closest.normal);
        let dist = s.dot(&closest.normal);

        // Check convergence
        if dist - closest.distance < TOLERANCE {
            return Some((closest.distance, closest.normal));
        }

        // Expand polytope: remove faces visible from s, add new faces
        let mut edges = Vec::new();
        faces.retain(|face| {
            if is_visible(&points, face, &s) {
                // Add edges to boundary
                for i in 0..3 {
                    let edge = (face.indices[i], face.indices[(i + 1) % 3]);
                    edges.push(edge);
                }
                false
            } else {
                true
            }
        });

        // Remove duplicate edges (internal edges)
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

        // Add new point
        let new_idx = points.len();
        points.push(s);

        // Create new faces from boundary edges
        for (a, b) in unique_edges {
            faces.push(Face::new(&points, [a, b, new_idx]));
        }
    }

    None
}

impl Face {
    fn new(points: &[Vec3], indices: [usize; 3]) -> Self {
        let a = points[indices[0]];
        let b = points[indices[1]];
        let c = points[indices[2]];
        let ab = b - a;
        let ac = c - a;
        let mut normal = ab.cross(&ac);
        let norm = normal.norm();
        if norm > 1e-10 {
            normal /= norm;
        }
        let distance = normal.dot(&a);
        // Ensure normal points towards origin
        if distance < 0.0 {
            normal = -normal;
        }
        Self {
            indices,
            normal,
            distance: distance.abs(),
        }
    }
}

fn is_visible(points: &[Vec3], face: &Face, point: &Vec3) -> bool {
    let a = points[face.indices[0]];
    let to_point = point - a;
    to_point.dot(&face.normal) > 0.0
}
