//! Analytic ray/shape intersection.
//!
//! GJK answers "how far apart are these two convex bodies"; a rangefinder needs
//! "how far along this ray do I first hit something". Marching a ray with GJK is
//! both slow and inexact, so each primitive gets a closed-form test here.
//!
//! Every shape is defined in its own local frame (cylinders and capsules run
//! along local +Z, boxes are axis-aligned about their centre) and placed by a
//! world position `pos` plus a rotation `rot` that maps **local → world**.

use crate::geometry::Geometry;
use phyz_math::{Mat3, Vec3};

/// A ray with a normalized direction.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    /// Ray origin in world coordinates.
    pub origin: Vec3,
    /// Unit direction in world coordinates.
    pub dir: Vec3,
}

impl Ray {
    /// Create a ray, normalizing `dir`.
    ///
    /// Returns `None` if `dir` is (numerically) the zero vector, since a ray
    /// with no direction has no meaningful intersection.
    pub fn new(origin: Vec3, dir: Vec3) -> Option<Self> {
        let n = dir.norm();
        if !n.is_finite() || n < 1e-12 {
            return None;
        }
        Some(Self {
            origin,
            dir: dir / n,
        })
    }

    /// Point at parameter `t` along the ray.
    pub fn at(&self, t: f64) -> Vec3 {
        self.origin + self.dir * t
    }
}

/// A ray/shape intersection.
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    /// Distance along the ray to the first intersection.
    pub distance: f64,
    /// Intersection point in world coordinates.
    pub point: Vec3,
    /// Outward surface normal at the intersection, in world coordinates.
    pub normal: Vec3,
}

/// Rays starting exactly on a surface must not immediately re-hit it.
const T_EPS: f64 = 1e-9;

/// Intersect a ray with a geometry placed at `pos` with orientation `rot`.
///
/// `rot` maps local shape coordinates into world coordinates. Returns the
/// nearest intersection with `distance >= 0`, or `None` if the ray misses.
/// A ray originating inside a solid reports the exit point.
pub fn ray_intersect(geom: &Geometry, pos: &Vec3, rot: &Mat3, ray: &Ray) -> Option<RayHit> {
    // Work in the shape's local frame: world → local is rot^T.
    let inv = rot.transpose();
    let o = inv.mul_vec(ray.origin - *pos);
    let d = inv.mul_vec(ray.dir);

    let (t, local_normal) = match geom {
        Geometry::Sphere { radius } => sphere_hit(&o, &d, *radius)?,
        Geometry::Box { half_extents } => box_hit(&o, &d, half_extents)?,
        Geometry::Cylinder { radius, height } => cylinder_hit(&o, &d, *radius, *height * 0.5)?,
        // A capsule's `length` is the length of the cylindrical section; the
        // hemispherical caps extend `radius` beyond each end.
        Geometry::Capsule { radius, length } => capsule_hit(&o, &d, *radius, *length * 0.5)?,
        Geometry::Plane { normal } => plane_hit(&o, &d, normal)?,
        Geometry::Mesh { vertices, faces } => mesh_hit(&o, &d, vertices, faces)?,
    };

    Some(RayHit {
        distance: t,
        point: ray.at(t),
        normal: rot.mul_vec(local_normal),
    })
}

/// Cast a ray against many placed geometries, returning the nearest hit within
/// `max_distance` along with the index of the geometry that was hit.
pub fn ray_cast<'a, I>(shapes: I, ray: &Ray, max_distance: f64) -> Option<(usize, RayHit)>
where
    I: IntoIterator<Item = (&'a Geometry, Vec3, Mat3)>,
{
    let mut best: Option<(usize, RayHit)> = None;
    for (idx, (geom, pos, rot)) in shapes.into_iter().enumerate() {
        let Some(hit) = ray_intersect(geom, &pos, &rot, ray) else {
            continue;
        };
        if hit.distance > max_distance {
            continue;
        }
        if best.is_none_or(|(_, b)| hit.distance < b.distance) {
            best = Some((idx, hit));
        }
    }
    best
}

/// Pick the nearest non-negative root of a quadratic, given both roots.
fn nearest_root(t0: f64, t1: f64) -> Option<f64> {
    let (lo, hi) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
    if lo >= T_EPS {
        Some(lo)
    } else if hi >= T_EPS {
        Some(hi)
    } else {
        None
    }
}

fn sphere_hit(o: &Vec3, d: &Vec3, radius: f64) -> Option<(f64, Vec3)> {
    // |o + t d|² = r², with |d| = 1
    let b = o.dot(d);
    let c = o.norm_sq() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t = nearest_root(-b - sq, -b + sq)?;
    let p = *o + *d * t;
    Some((t, p / radius))
}

fn box_hit(o: &Vec3, d: &Vec3, half: &Vec3) -> Option<(f64, Vec3)> {
    // Slab method, tracking which axis produced the entry/exit plane.
    let mut t_min = f64::NEG_INFINITY;
    let mut t_max = f64::INFINITY;
    let mut min_axis = 0usize;
    let mut max_axis = 0usize;
    let mut min_sign = 1.0;
    let mut max_sign = 1.0;

    let oa = [o.x, o.y, o.z];
    let da = [d.x, d.y, d.z];
    let ha = [half.x, half.y, half.z];

    for axis in 0..3 {
        if da[axis].abs() < 1e-12 {
            // Parallel to this slab: miss unless already inside it.
            if oa[axis] < -ha[axis] || oa[axis] > ha[axis] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / da[axis];
        let mut t1 = (-ha[axis] - oa[axis]) * inv;
        let mut t2 = (ha[axis] - oa[axis]) * inv;
        let mut s1 = -1.0;
        let mut s2 = 1.0;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
            std::mem::swap(&mut s1, &mut s2);
        }
        if t1 > t_min {
            t_min = t1;
            min_axis = axis;
            min_sign = s1;
        }
        if t2 < t_max {
            t_max = t2;
            max_axis = axis;
            max_sign = s2;
        }
        if t_min > t_max {
            return None;
        }
    }

    let (t, axis, sign) = if t_min >= T_EPS {
        (t_min, min_axis, min_sign)
    } else if t_max >= T_EPS {
        (t_max, max_axis, max_sign)
    } else {
        return None;
    };

    let mut n = Vec3::zeros();
    match axis {
        0 => n.x = sign,
        1 => n.y = sign,
        _ => n.z = sign,
    }
    Some((t, n))
}

/// Candidate hit on the infinite cylinder of radius `r` about local Z,
/// restricted to |z| <= half_h. Returns (t, normal) for the nearest valid root.
fn cylinder_side(o: &Vec3, d: &Vec3, r: f64, half_h: f64) -> Option<(f64, Vec3)> {
    let a = d.x * d.x + d.y * d.y;
    if a < 1e-12 {
        return None; // ray parallel to the axis: only the caps can be hit
    }
    let b = o.x * d.x + o.y * d.y;
    let c = o.x * o.x + o.y * o.y - r * r;
    let disc = b * b - a * c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    for t in [(-b - sq) / a, (-b + sq) / a] {
        if t < T_EPS {
            continue;
        }
        let z = o.z + d.z * t;
        if z.abs() <= half_h {
            let p = *o + *d * t;
            return Some((t, Vec3::new(p.x / r, p.y / r, 0.0)));
        }
    }
    None
}

/// Nearest hit on the two disc caps at z = ±half_h.
fn cylinder_caps(o: &Vec3, d: &Vec3, r: f64, half_h: f64) -> Option<(f64, Vec3)> {
    if d.z.abs() < 1e-12 {
        return None;
    }
    let mut best: Option<(f64, Vec3)> = None;
    for sign in [-1.0, 1.0] {
        let t = (sign * half_h - o.z) / d.z;
        if t < T_EPS {
            continue;
        }
        let p = *o + *d * t;
        if p.x * p.x + p.y * p.y <= r * r && best.is_none_or(|(bt, _)| t < bt) {
            best = Some((t, Vec3::new(0.0, 0.0, sign)));
        }
    }
    best
}

fn cylinder_hit(o: &Vec3, d: &Vec3, r: f64, half_h: f64) -> Option<(f64, Vec3)> {
    match (
        cylinder_side(o, d, r, half_h),
        cylinder_caps(o, d, r, half_h),
    ) {
        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
        (Some(a), None) => Some(a),
        (None, b) => b,
    }
}

fn capsule_hit(o: &Vec3, d: &Vec3, r: f64, half_h: f64) -> Option<(f64, Vec3)> {
    let mut best: Option<(f64, Vec3)> = cylinder_side(o, d, r, half_h);

    // Hemispherical caps: sphere tests, accepted only on the correct side.
    for sign in [-1.0, 1.0] {
        let centre = Vec3::new(0.0, 0.0, sign * half_h);
        let oc = *o - centre;
        let b = oc.dot(d);
        let c = oc.norm_sq() - r * r;
        let disc = b * b - c;
        if disc < 0.0 {
            continue;
        }
        let sq = disc.sqrt();
        for t in [-b - sq, -b + sq] {
            if t < T_EPS {
                continue;
            }
            let p = *o + *d * t;
            if (p.z - sign * half_h) * sign < 0.0 {
                continue; // belongs to the cylindrical section, not this cap
            }
            if best.is_none_or(|(bt, _)| t < bt) {
                best = Some((t, (p - centre) / r));
            }
            break;
        }
    }
    best
}

fn plane_hit(o: &Vec3, d: &Vec3, normal: &Vec3) -> Option<(f64, Vec3)> {
    let n = normal.normalize();
    let denom = d.dot(n);
    if denom.abs() < 1e-12 {
        return None; // parallel
    }
    // Plane passes through the shape origin (already subtracted from `o`).
    let t = -o.dot(n) / denom;
    if t < T_EPS {
        return None;
    }
    // Report the normal facing back along the ray.
    Some((t, if denom < 0.0 { n } else { -n }))
}

fn mesh_hit(o: &Vec3, d: &Vec3, vertices: &[Vec3], faces: &[[usize; 3]]) -> Option<(f64, Vec3)> {
    let mut best: Option<(f64, Vec3)> = None;
    for f in faces {
        let (Some(&v0), Some(&v1), Some(&v2)) =
            (vertices.get(f[0]), vertices.get(f[1]), vertices.get(f[2]))
        else {
            continue;
        };
        if let Some(t) = triangle_hit(o, d, &v0, &v1, &v2)
            && best.is_none_or(|(bt, _)| t < bt)
        {
            let n = (v1 - v0).cross(v2 - v0);
            let n = n.normalize();
            best = Some((t, if n.dot(d) < 0.0 { n } else { -n }));
        }
    }
    best
}

/// Möller–Trumbore, two-sided.
fn triangle_hit(o: &Vec3, d: &Vec3, v0: &Vec3, v1: &Vec3, v2: &Vec3) -> Option<f64> {
    const EPS: f64 = 1e-12;
    let e1 = *v1 - *v0;
    let e2 = *v2 - *v0;
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < EPS {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = *o - *v0;
    let u = tvec.dot(p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = tvec.cross(e1);
    let v = d.dot(q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv_det;
    if t < T_EPS { None } else { Some(t) }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn ray(o: [f64; 3], d: [f64; 3]) -> Ray {
        Ray::new(Vec3::new(o[0], o[1], o[2]), Vec3::new(d[0], d[1], d[2])).unwrap()
    }

    #[test]
    fn zero_direction_rejected() {
        assert!(Ray::new(Vec3::zeros(), Vec3::zeros()).is_none());
    }

    #[test]
    fn sphere_known_distance() {
        // Sphere of radius 1 centred 5 m down +X; ray from the origin along +X
        // must report exactly 4.0 to the near surface.
        let g = Geometry::Sphere { radius: 1.0 };
        let hit = ray_intersect(
            &g,
            &Vec3::new(5.0, 0.0, 0.0),
            &Mat3::identity(),
            &ray([0.0; 3], [1.0, 0.0, 0.0]),
        )
        .expect("ray should hit sphere");
        assert!((hit.distance - 4.0).abs() < EPS, "got {}", hit.distance);
        assert!((hit.point.x - 4.0).abs() < EPS);
        // Outward normal at the near face points back at the ray origin.
        assert!((hit.normal.x + 1.0).abs() < EPS);
    }

    #[test]
    fn sphere_tangent_miss() {
        let g = Geometry::Sphere { radius: 1.0 };
        // Offset by 1.5 in Y: clean miss.
        let hit = ray_intersect(
            &g,
            &Vec3::new(5.0, 1.5, 0.0),
            &Mat3::identity(),
            &ray([0.0; 3], [1.0, 0.0, 0.0]),
        );
        assert!(hit.is_none());
    }

    #[test]
    fn sphere_behind_ray_misses() {
        let g = Geometry::Sphere { radius: 1.0 };
        let hit = ray_intersect(
            &g,
            &Vec3::new(-5.0, 0.0, 0.0),
            &Mat3::identity(),
            &ray([0.0; 3], [1.0, 0.0, 0.0]),
        );
        assert!(hit.is_none());
    }

    #[test]
    fn ray_from_inside_sphere_exits() {
        let g = Geometry::Sphere { radius: 2.0 };
        let hit = ray_intersect(
            &g,
            &Vec3::zeros(),
            &Mat3::identity(),
            &ray([0.0; 3], [1.0, 0.0, 0.0]),
        )
        .unwrap();
        assert!((hit.distance - 2.0).abs() < EPS);
    }

    #[test]
    fn box_face_and_corner() {
        let g = Geometry::Box {
            half_extents: Vec3::new(1.0, 1.0, 1.0),
        };
        let hit = ray_intersect(
            &g,
            &Vec3::new(0.0, 0.0, 10.0),
            &Mat3::identity(),
            &ray([0.0; 3], [0.0, 0.0, 1.0]),
        )
        .unwrap();
        assert!((hit.distance - 9.0).abs() < EPS, "got {}", hit.distance);
        assert!((hit.normal.z + 1.0).abs() < EPS);

        // Rotating the box 45° about Z turns the +X face into an edge that is
        // sqrt(2) from the centre.
        let rot = Mat3::rotation_z(std::f64::consts::FRAC_PI_4);
        let hit = ray_intersect(
            &g,
            &Vec3::new(10.0, 0.0, 0.0),
            &rot,
            &ray([0.0; 3], [1.0, 0.0, 0.0]),
        )
        .unwrap();
        assert!(
            (hit.distance - (10.0 - 2f64.sqrt())).abs() < 1e-9,
            "got {}",
            hit.distance
        );
    }

    #[test]
    fn cylinder_side_and_cap() {
        let g = Geometry::Cylinder {
            radius: 0.5,
            height: 2.0,
        };
        // Side hit.
        let hit = ray_intersect(
            &g,
            &Vec3::new(3.0, 0.0, 0.0),
            &Mat3::identity(),
            &ray([0.0; 3], [1.0, 0.0, 0.0]),
        )
        .unwrap();
        assert!((hit.distance - 2.5).abs() < EPS, "got {}", hit.distance);

        // Cap hit from directly below (cylinder runs along local Z, half-height 1).
        let hit = ray_intersect(
            &g,
            &Vec3::new(0.0, 0.0, 5.0),
            &Mat3::identity(),
            &ray([0.0; 3], [0.0, 0.0, 1.0]),
        )
        .unwrap();
        assert!((hit.distance - 4.0).abs() < EPS, "got {}", hit.distance);
        assert!((hit.normal.z + 1.0).abs() < EPS);

        // Aimed above the end cap: misses entirely.
        assert!(
            ray_intersect(
                &g,
                &Vec3::new(3.0, 0.0, 0.0),
                &Mat3::identity(),
                &ray([0.0, 0.0, 1.5], [1.0, 0.0, 0.0]),
            )
            .is_none()
        );
    }

    #[test]
    fn capsule_cap_is_rounded() {
        // Cylindrical section 2 long (half 1), radius 0.5: the cap tip sits at
        // z = 1.5, so a downward ray from z = 5 travels 3.5.
        let g = Geometry::Capsule {
            radius: 0.5,
            length: 2.0,
        };
        let hit = ray_intersect(
            &g,
            &Vec3::zeros(),
            &Mat3::identity(),
            &ray([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]),
        )
        .unwrap();
        assert!((hit.distance - 3.5).abs() < EPS, "got {}", hit.distance);
    }

    #[test]
    fn plane_hit_distance() {
        let g = Geometry::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
        };
        let hit = ray_intersect(
            &g,
            &Vec3::zeros(),
            &Mat3::identity(),
            &ray([0.0, 0.0, 3.0], [0.0, 0.0, -1.0]),
        )
        .unwrap();
        assert!((hit.distance - 3.0).abs() < EPS);
        assert!((hit.normal.z - 1.0).abs() < EPS);
    }

    #[test]
    fn mesh_triangle_hit() {
        let g = Geometry::Mesh {
            vertices: vec![
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            faces: vec![[0, 1, 2]],
        };
        let hit = ray_intersect(
            &g,
            &Vec3::new(0.0, 0.0, 2.0),
            &Mat3::identity(),
            &ray([0.0, 0.0, 10.0], [0.0, 0.0, -1.0]),
        )
        .unwrap();
        assert!((hit.distance - 8.0).abs() < EPS, "got {}", hit.distance);
    }

    #[test]
    fn ray_cast_picks_nearest_within_range() {
        let near = Geometry::Sphere { radius: 0.5 };
        let far = Geometry::Sphere { radius: 0.5 };
        let shapes = vec![
            (&far, Vec3::new(8.0, 0.0, 0.0), Mat3::identity()),
            (&near, Vec3::new(3.0, 0.0, 0.0), Mat3::identity()),
        ];
        let r = ray([0.0; 3], [1.0, 0.0, 0.0]);

        let (idx, hit) = ray_cast(shapes.clone(), &r, 100.0).unwrap();
        assert_eq!(idx, 1);
        assert!((hit.distance - 2.5).abs() < EPS);

        // Range cut shorter than the near sphere: nothing in range.
        assert!(ray_cast(shapes.clone(), &r, 1.0).is_none());
        // Range that only reaches the far sphere's side of the near one still
        // finds the near sphere.
        let (idx, _) = ray_cast(shapes, &r, 3.0).unwrap();
        assert_eq!(idx, 1);
    }
}
