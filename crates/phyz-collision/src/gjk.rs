//! GJK (Gilbert-Johnson-Keerthi) algorithm for distance queries.

use crate::geometry::Geometry;
use phyz_math::{Mat3, Vec3};

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

        if ab.dot(ao) > 0.0 {
            // Origin is between A and B
            *dir = ab.cross(ao).cross(ab);
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
        let abc = ab.cross(ac);

        if abc.cross(ac).dot(ao) > 0.0 {
            if ac.dot(ao) > 0.0 {
                // Origin is past AC edge
                self.points = vec![c, a];
                *dir = ac.cross(ao).cross(ac);
            } else {
                // Origin is past A
                self.points = vec![a];
                *dir = ao;
            }
        } else if ab.cross(abc).dot(ao) > 0.0 {
            if ab.dot(ao) > 0.0 {
                // Origin is past AB edge
                self.points = vec![b, a];
                *dir = ab.cross(ao).cross(ab);
            } else {
                // Origin is past A
                self.points = vec![a];
                *dir = ao;
            }
        } else {
            // Origin is above or below triangle
            if abc.dot(ao) > 0.0 {
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
        let abc = ab.cross(ac);
        let acd = ac.cross(ad);
        let adb = ad.cross(ab);

        // Check which face the origin is closest to
        if abc.dot(ao) > 0.0 {
            // Origin is past ABC face
            self.points = vec![c, b, a];
            return self.triangle_case(dir);
        }
        if acd.dot(ao) > 0.0 {
            // Origin is past ACD face
            self.points = vec![d, c, a];
            return self.triangle_case(dir);
        }
        if adb.dot(ao) > 0.0 {
            // Origin is past ADB face
            self.points = vec![b, d, a];
            return self.triangle_case(dir);
        }

        // Origin is inside tetrahedron
        true
    }
}

/// What GJK concluded about a pair.
#[derive(Debug, Clone)]
pub enum GjkOutcome {
    /// The shapes are disjoint; `distance` is the separation (> 0).
    Separated {
        /// Separation between the two surfaces (> 0).
        distance: f64,
    },
    /// The shapes overlap. `simplex` is the terminating simplex — a set of
    /// Minkowski-difference points enclosing the origin, which is exactly the
    /// seed EPA needs to be valid.
    Penetrating {
        /// Minkowski-difference points enclosing the origin.
        simplex: Vec<Vec3>,
    },
    /// GJK could not decide within its iteration budget (near-tangential
    /// configurations on curved surfaces). Callers should treat this as "no
    /// contact" rather than guessing.
    Indeterminate,
}

/// Compute signed distance between two geometries.
///
/// Positive is the separation distance; negative is the **penetration depth**
/// (see [`gjk_distance_rot`]).
pub fn gjk_distance(geom_a: &Geometry, geom_b: &Geometry, pos_a: &Vec3, pos_b: &Vec3) -> f64 {
    let rot_a = Mat3::identity();
    let rot_b = Mat3::identity();
    gjk_distance_rot(geom_a, geom_b, pos_a, pos_b, &rot_a, &rot_b)
}

/// Signed distance with rotation matrices: `+separation` when disjoint,
/// `-depth` when overlapping.
///
/// The negative branch runs EPA to get a *real* depth. It previously returned
/// a hardcoded `-1.0`, which meant every overlapping pair reported a fake one
/// metre of penetration — and any penalty force built on it was meaningless.
pub fn gjk_distance_rot(
    geom_a: &Geometry,
    geom_b: &Geometry,
    pos_a: &Vec3,
    pos_b: &Vec3,
    rot_a: &Mat3,
    rot_b: &Mat3,
) -> f64 {
    match gjk_rot(geom_a, geom_b, pos_a, pos_b, rot_a, rot_b) {
        GjkOutcome::Separated { distance } => distance,
        GjkOutcome::Indeterminate => 0.0,
        GjkOutcome::Penetrating { simplex } => {
            match crate::epa::epa_from_simplex(geom_a, geom_b, pos_a, pos_b, rot_a, rot_b, &simplex)
            {
                Some((depth, _)) => -depth,
                // Overlap is certain but EPA could not measure it; report a
                // vanishing depth rather than inventing a magnitude.
                None => -0.0,
            }
        }
    }
}

/// Run GJK, reporting separation *or* the origin-enclosing simplex.
pub fn gjk_rot(
    geom_a: &Geometry,
    geom_b: &Geometry,
    pos_a: &Vec3,
    pos_b: &Vec3,
    rot_a: &Mat3,
    rot_b: &Mat3,
) -> GjkOutcome {
    let mut simplex = Simplex::new();
    let mut dir = pos_b - pos_a;
    if dir.norm() < 1e-10 {
        dir = Vec3::x();
    }

    // Support function for Minkowski difference A - B
    let support = |d: &Vec3| {
        let sa = geom_a.support(d, pos_a, rot_a);
        let sb = geom_b.support(&(-*d), pos_b, rot_b);
        sa - sb
    };

    let mut s = support(&dir);
    if !is_finite(&s) {
        return GjkOutcome::Indeterminate;
    }
    simplex.add(s);
    dir = -s;

    for _ in 0..64 {
        let dir_norm = dir.norm();
        if dir_norm < 1e-10 {
            // The search direction collapsed: the origin lies on the current
            // simplex. With a single point that means the two surfaces touch
            // exactly (distance 0). With two or more the origin is enclosed by
            // the simplex, so the shapes are penetrating — reporting separation
            // here would make deep overlaps invisible to `find_contacts`.
            return if simplex.len() >= 2 {
                GjkOutcome::Penetrating {
                    simplex: simplex.points.clone(),
                }
            } else {
                GjkOutcome::Separated { distance: 0.0 }
            };
        }

        s = support(&dir);
        if !is_finite(&s) {
            return GjkOutcome::Indeterminate;
        }
        if s.dot(dir) < 0.0 {
            return GjkOutcome::Separated { distance: dir_norm };
        }
        simplex.add(s);
        if simplex.contains_origin(&mut dir) {
            return GjkOutcome::Penetrating {
                simplex: simplex.points.clone(),
            };
        }
    }

    GjkOutcome::Indeterminate
}

fn is_finite(v: &Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}
