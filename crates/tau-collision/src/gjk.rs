//! GJK (Gilbert-Johnson-Keerthi) algorithm for distance queries.

use crate::geometry::Geometry;
use tau_math::{Mat3, Vec3};

/// Simplex for GJK algorithm (up to 4 points).
struct Simplex {
    points: Vec<Vec3>,
}

impl Simplex {
    fn new() -> Self {
        Self { points: Vec::new() }
    }

    fn add(&mut self, point: Vec3) {
        self.points.push(point);
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    /// Update simplex to contain origin, return true if origin is contained.
    fn contains_origin(&mut self, dir: &mut Vec3) -> bool {
        match self.len() {
            2 => self.line_case(dir),
            3 => self.triangle_case(dir),
            4 => self.tetrahedron_case(dir),
            _ => false,
        }
    }

    fn line_case(&mut self, dir: &mut Vec3) -> bool {
        let a = self.points[1];
        let b = self.points[0];
        let ab = b - a;
        let ao = -a;

        if ab.dot(&ao) > 0.0 {
            // Origin is between A and B
            *dir = ab.cross(&ao).cross(&ab);
        } else {
            // Origin is past A
            self.points.remove(0);
            *dir = ao;
        }
        false
    }

    fn triangle_case(&mut self, dir: &mut Vec3) -> bool {
        let a = self.points[2];
        let b = self.points[1];
        let c = self.points[0];
        let ab = b - a;
        let ac = c - a;
        let ao = -a;
        let abc = ab.cross(&ac);

        if abc.cross(&ac).dot(&ao) > 0.0 {
            if ac.dot(&ao) > 0.0 {
                // Origin is past AC edge
                self.points = vec![c, a];
                *dir = ac.cross(&ao).cross(&ac);
            } else {
                // Origin is past A
                self.points = vec![a];
                *dir = ao;
            }
        } else if ab.cross(&abc).dot(&ao) > 0.0 {
            if ab.dot(&ao) > 0.0 {
                // Origin is past AB edge
                self.points = vec![b, a];
                *dir = ab.cross(&ao).cross(&ab);
            } else {
                // Origin is past A
                self.points = vec![a];
                *dir = ao;
            }
        } else {
            // Origin is above or below triangle
            if abc.dot(&ao) > 0.0 {
                *dir = abc;
            } else {
                self.points = vec![b, c, a];
                *dir = -abc;
            }
        }
        false
    }

    fn tetrahedron_case(&mut self, dir: &mut Vec3) -> bool {
        let a = self.points[3];
        let b = self.points[2];
        let c = self.points[1];
        let d = self.points[0];
        let ao = -a;

        let ab = b - a;
        let ac = c - a;
        let ad = d - a;
        let abc = ab.cross(&ac);
        let acd = ac.cross(&ad);
        let adb = ad.cross(&ab);

        // Check which face the origin is closest to
        if abc.dot(&ao) > 0.0 {
            // Origin is past ABC face
            self.points = vec![c, b, a];
            return self.triangle_case(dir);
        }
        if acd.dot(&ao) > 0.0 {
            // Origin is past ACD face
            self.points = vec![d, c, a];
            return self.triangle_case(dir);
        }
        if adb.dot(&ao) > 0.0 {
            // Origin is past ADB face
            self.points = vec![b, d, a];
            return self.triangle_case(dir);
        }

        // Origin is inside tetrahedron
        true
    }
}

/// Compute signed distance between two geometries.
/// Returns negative value if penetrating.
pub fn gjk_distance(geom_a: &Geometry, geom_b: &Geometry, pos_a: &Vec3, pos_b: &Vec3) -> f64 {
    let rot_a = Mat3::identity();
    let rot_b = Mat3::identity();
    gjk_distance_rot(geom_a, geom_b, pos_a, pos_b, &rot_a, &rot_b)
}

/// GJK with rotation matrices.
pub fn gjk_distance_rot(
    geom_a: &Geometry,
    geom_b: &Geometry,
    pos_a: &Vec3,
    pos_b: &Vec3,
    rot_a: &Mat3,
    rot_b: &Mat3,
) -> f64 {
    let mut simplex = Simplex::new();
    let mut dir = pos_b - pos_a;
    if dir.norm() < 1e-10 {
        dir = Vec3::x();
    }

    // Support function for Minkowski difference A - B
    let support = |d: &Vec3| {
        let sa = geom_a.support(d, pos_a, rot_a);
        let sb = geom_b.support(&-d, pos_b, rot_b);
        sa - sb
    };

    let mut s = support(&dir);
    simplex.add(s);
    dir = -s;

    for _ in 0..64 {
        // Ensure dir is not zero
        let dir_norm = dir.norm();
        if dir_norm < 1e-10 {
            // Direction is zero, likely at origin
            return 0.0;
        }

        s = support(&dir);
        if s.dot(&dir) < 0.0 {
            // No intersection
            return dir_norm;
        }
        simplex.add(s);
        if simplex.contains_origin(&mut dir) {
            // Penetrating
            return -1.0;
        }
    }

    // Convergence failed; approximate distance
    let dir_norm = dir.norm();
    if dir_norm < 1e-10 { 0.0 } else { dir_norm }
}
