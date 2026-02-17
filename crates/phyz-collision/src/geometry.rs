//! Geometry primitives for collision detection.

use phyz_math::{Mat3, Vec3};

/// Collision geometry types.
#[derive(Debug, Clone)]
pub enum Geometry {
    /// Sphere with given radius.
    Sphere { radius: f64 },
    /// Capsule: cylinder with hemispheres at ends.
    Capsule { radius: f64, length: f64 },
    /// Box with half-extents (width/2, height/2, depth/2).
    Box { half_extents: Vec3 },
    /// Cylinder along Z axis.
    Cylinder { radius: f64, height: f64 },
    /// Convex mesh (vertices only; convex hull assumed).
    Mesh {
        vertices: Vec<Vec3>,
        faces: Vec<[usize; 3]>,
    },
    /// Infinite plane with outward normal.
    Plane { normal: Vec3 },
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    /// Create AABB from min/max corners.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Compute AABB for a geometry at given position and rotation.
    pub fn from_geometry(geom: &Geometry, pos: &Vec3, rot: &Mat3) -> Self {
        match geom {
            Geometry::Sphere { radius } => {
                let r = Vec3::new(*radius, *radius, *radius);
                AABB::new(pos - r, pos + r)
            }
            Geometry::Capsule { radius, length } => {
                let half_len = length * 0.5;
                let r = Vec3::new(*radius, *radius, half_len + *radius);
                // Transform by rotation
                let corners = [
                    rot * Vec3::new(r.x, r.y, r.z),
                    rot * Vec3::new(-r.x, r.y, r.z),
                    rot * Vec3::new(r.x, -r.y, r.z),
                    rot * Vec3::new(-r.x, -r.y, r.z),
                    rot * Vec3::new(r.x, r.y, -r.z),
                    rot * Vec3::new(-r.x, r.y, -r.z),
                    rot * Vec3::new(r.x, -r.y, -r.z),
                    rot * Vec3::new(-r.x, -r.y, -r.z),
                ];
                let mut min = pos + corners[0];
                let mut max = min;
                for corner in &corners[1..] {
                    let c = pos + corner;
                    min = min.inf(&c);
                    max = max.sup(&c);
                }
                AABB::new(min, max)
            }
            Geometry::Box { half_extents } => {
                let corners = [
                    rot * Vec3::new(half_extents.x, half_extents.y, half_extents.z),
                    rot * Vec3::new(-half_extents.x, half_extents.y, half_extents.z),
                    rot * Vec3::new(half_extents.x, -half_extents.y, half_extents.z),
                    rot * Vec3::new(-half_extents.x, -half_extents.y, half_extents.z),
                    rot * Vec3::new(half_extents.x, half_extents.y, -half_extents.z),
                    rot * Vec3::new(-half_extents.x, half_extents.y, -half_extents.z),
                    rot * Vec3::new(half_extents.x, -half_extents.y, -half_extents.z),
                    rot * Vec3::new(-half_extents.x, -half_extents.y, -half_extents.z),
                ];
                let mut min = pos + corners[0];
                let mut max = min;
                for corner in &corners[1..] {
                    let c = pos + corner;
                    min = min.inf(&c);
                    max = max.sup(&c);
                }
                AABB::new(min, max)
            }
            Geometry::Cylinder { radius, height } => {
                let half_h = height * 0.5;
                let r = Vec3::new(*radius, *radius, half_h);
                let corners = [
                    rot * Vec3::new(r.x, 0.0, r.z),
                    rot * Vec3::new(-r.x, 0.0, r.z),
                    rot * Vec3::new(0.0, r.y, r.z),
                    rot * Vec3::new(0.0, -r.y, r.z),
                    rot * Vec3::new(r.x, 0.0, -r.z),
                    rot * Vec3::new(-r.x, 0.0, -r.z),
                    rot * Vec3::new(0.0, r.y, -r.z),
                    rot * Vec3::new(0.0, -r.y, -r.z),
                ];
                let mut min = pos + corners[0];
                let mut max = min;
                for corner in &corners[1..] {
                    let c = pos + corner;
                    min = min.inf(&c);
                    max = max.sup(&c);
                }
                AABB::new(min, max)
            }
            Geometry::Mesh { vertices, .. } => {
                if vertices.is_empty() {
                    return AABB::new(*pos, *pos);
                }
                let v0 = pos + rot * vertices[0];
                let mut min = v0;
                let mut max = v0;
                for v in vertices.iter().skip(1) {
                    let vt = pos + rot * v;
                    min = min.inf(&vt);
                    max = max.sup(&vt);
                }
                AABB::new(min, max)
            }
            Geometry::Plane { .. } => {
                // Plane is infinite; return large AABB
                let large = 1e6;
                AABB::new(
                    Vec3::new(-large, -large, -large),
                    Vec3::new(large, large, large),
                )
            }
        }
    }

    /// Check if two AABBs overlap.
    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }
}

impl Geometry {
    /// Support function: point on surface farthest in direction `dir`.
    pub fn support(&self, dir: &Vec3, pos: &Vec3, rot: &Mat3) -> Vec3 {
        match self {
            Geometry::Sphere { radius } => {
                let d = dir.normalize();
                pos + d * *radius
            }
            Geometry::Capsule { radius, length } => {
                let half_len = length * 0.5;
                let axis = rot * Vec3::z();
                let p1 = pos + axis * half_len;
                let p2 = pos - axis * half_len;
                let d = dir.normalize();
                if dir.dot(&axis) > 0.0 {
                    p1 + d * *radius
                } else {
                    p2 + d * *radius
                }
            }
            Geometry::Box { half_extents } => {
                let local_dir = rot.transpose() * dir;
                let sx = if local_dir.x >= 0.0 {
                    half_extents.x
                } else {
                    -half_extents.x
                };
                let sy = if local_dir.y >= 0.0 {
                    half_extents.y
                } else {
                    -half_extents.y
                };
                let sz = if local_dir.z >= 0.0 {
                    half_extents.z
                } else {
                    -half_extents.z
                };
                pos + rot * Vec3::new(sx, sy, sz)
            }
            Geometry::Cylinder { radius, height } => {
                let half_h = height * 0.5;
                let axis = rot * Vec3::z();
                let local_dir = rot.transpose() * dir;
                let radial = Vec3::new(local_dir.x, local_dir.y, 0.0);
                let radial_norm = radial.norm();
                let radial_dir = if radial_norm > 1e-10 {
                    radial / radial_norm
                } else {
                    Vec3::x()
                };
                let sz = if local_dir.z >= 0.0 { half_h } else { -half_h };
                pos + rot * (radial_dir * *radius) + axis * sz
            }
            Geometry::Mesh { vertices, .. } => {
                let mut best = pos + rot * vertices[0];
                let mut best_dot = dir.dot(&best);
                for v in vertices.iter().skip(1) {
                    let vt = pos + rot * v;
                    let d = dir.dot(&vt);
                    if d > best_dot {
                        best = vt;
                        best_dot = d;
                    }
                }
                best
            }
            Geometry::Plane { normal } => {
                // Plane support: if dir·normal > 0, project origin onto plane
                // Otherwise, return point at infinity (clamped to large value)
                let dot = dir.dot(normal);
                if dot > 0.0 {
                    pos + normal * 1e6
                } else {
                    pos - normal * 1e6
                }
            }
        }
    }
}
