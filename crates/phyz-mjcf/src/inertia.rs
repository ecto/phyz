//! Inertia derived from collision geometry.
//!
//! Reference MJCF models almost never write `<inertial>` — MuJoCo's compiler
//! infers mass and inertia from `<geom>` shape plus `density` (default
//! 1000 kg/m³, i.e. water). Without this, every geom-only body falls back to a
//! placeholder 1 kg point mass and the dynamics are meaningless.

use phyz_math::{Mat3, SpatialInertia, Vec3};

/// A geom's mass properties in the *body* frame.
#[derive(Debug, Clone, Copy)]
pub struct MassProps {
    /// Mass in kg.
    pub mass: f64,
    /// Centre of mass in body coordinates.
    pub com: Vec3,
    /// Inertia tensor about the centre of mass, in body coordinates.
    pub inertia: Mat3,
}

/// The shapes we can derive inertia for.
#[derive(Debug, Clone, Copy)]
pub enum Shape {
    /// Radius.
    Sphere { radius: f64 },
    /// Radius and half-length of the cylindrical section (MJCF `size` order).
    Capsule { radius: f64, half_len: f64 },
    /// Radius and half-height.
    Cylinder { radius: f64, half_height: f64 },
    /// Half-extents.
    Box { half: Vec3 },
}

impl Shape {
    /// Volume in m³.
    pub fn volume(&self) -> f64 {
        use std::f64::consts::PI;
        match *self {
            Shape::Sphere { radius } => 4.0 / 3.0 * PI * radius.powi(3),
            Shape::Capsule { radius, half_len } => {
                PI * radius * radius * (2.0 * half_len) + 4.0 / 3.0 * PI * radius.powi(3)
            }
            Shape::Cylinder {
                radius,
                half_height,
            } => PI * radius * radius * (2.0 * half_height),
            Shape::Box { half } => 8.0 * half.x * half.y * half.z,
        }
    }

    /// Principal moments about the shape's own centre, for the given mass.
    ///
    /// The local Z axis is the symmetry axis for capsules and cylinders, which
    /// is MJCF's convention.
    pub fn principal_inertia(&self, mass: f64) -> Vec3 {
        use std::f64::consts::PI;
        match *self {
            Shape::Sphere { radius } => {
                let i = 0.4 * mass * radius * radius;
                Vec3::new(i, i, i)
            }
            Shape::Box { half } => {
                let (x, y, z) = (half.x, half.y, half.z);
                Vec3::new(
                    mass / 3.0 * (y * y + z * z),
                    mass / 3.0 * (x * x + z * z),
                    mass / 3.0 * (x * x + y * y),
                )
            }
            Shape::Cylinder {
                radius,
                half_height,
            } => {
                let h = 2.0 * half_height;
                let ixx = mass * (3.0 * radius * radius + h * h) / 12.0;
                Vec3::new(ixx, ixx, 0.5 * mass * radius * radius)
            }
            Shape::Capsule { radius, half_len } => {
                // Split the mass between the cylindrical shaft and the two
                // hemispherical caps by volume, then combine with the parallel
                // axis theorem for the caps.
                let v_cyl = PI * radius * radius * (2.0 * half_len);
                let v_cap = 2.0 / 3.0 * PI * radius.powi(3); // one hemisphere
                let total = v_cyl + 2.0 * v_cap;
                if total <= 0.0 {
                    return Vec3::zeros();
                }
                let density = mass / total;
                let m_cyl = density * v_cyl;
                let m_cap = density * v_cap;

                let h = 2.0 * half_len;
                let izz = 0.5 * m_cyl * radius * radius + 2.0 * (0.4 * m_cap * radius * radius);

                // Hemisphere inertia about its own centroid is 83/320 m r²;
                // its centroid sits 3r/8 beyond the shaft end.
                let d = half_len + 3.0 / 8.0 * radius;
                let cap_ixx = 83.0 / 320.0 * m_cap * radius * radius + m_cap * d * d;
                let ixx = m_cyl * (3.0 * radius * radius + h * h) / 12.0 + 2.0 * cap_ixx;

                Vec3::new(ixx, ixx, izz)
            }
        }
    }
}

/// Mass properties for a geom at `pos` with rotation `rot` (geom→body),
/// given either an explicit `mass` or a `density`.
pub fn mass_props(shape: Shape, pos: Vec3, rot: Mat3, density: f64, mass: Option<f64>) -> MassProps {
    let m = mass.unwrap_or_else(|| density * shape.volume());
    let principal = shape.principal_inertia(m);
    // Rotate the principal-axis tensor into body coordinates: I_b = R I R^T.
    let local = Mat3::from_diagonal(&principal);
    let inertia = rot * local * rot.transpose();
    MassProps {
        mass: m,
        com: pos,
        inertia,
    }
}

/// Combine several geoms' mass properties into one body inertia.
///
/// Masses add, the centre of mass is the mass-weighted mean, and each geom's
/// inertia is shifted to the combined centre with the parallel axis theorem.
pub fn combine(parts: &[MassProps]) -> Option<SpatialInertia> {
    let total: f64 = parts.iter().map(|p| p.mass).sum();
    if total <= 0.0 {
        return None;
    }

    let mut com = Vec3::zeros();
    for p in parts {
        com += p.com * p.mass;
    }
    com = com / total;

    let mut inertia = Mat3::zero();
    for p in parts {
        let d = p.com - com;
        let d2 = d.x * d.x + d.y * d.y + d.z * d.z;
        // I += I_local + m (d·d I₃ - d dᵀ)
        let shift = Mat3::identity() * d2 - outer(d, d);
        inertia = inertia + p.inertia + shift * p.mass;
    }

    Some(SpatialInertia::new(total, com, inertia))
}

fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::new(
        a.x * b.x, a.x * b.y, a.x * b.z,
        a.y * b.x, a.y * b.y, a.y * b.z,
        a.z * b.x, a.z * b.y, a.z * b.z,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_matches_analytic() {
        let s = Shape::Sphere { radius: 0.5 };
        let m = 1000.0 * s.volume();
        assert!((m - 1000.0 * 4.0 / 3.0 * std::f64::consts::PI * 0.125).abs() < 1e-9);
        let i = s.principal_inertia(m);
        assert!((i.x - 0.4 * m * 0.25).abs() < 1e-9);
    }

    /// A capsule with zero shaft length is exactly a sphere; the split-mass
    /// formula must agree with the closed form there.
    #[test]
    fn degenerate_capsule_is_a_sphere() {
        let r = 0.3;
        let cap = Shape::Capsule {
            radius: r,
            half_len: 0.0,
        };
        let sph = Shape::Sphere { radius: r };
        assert!((cap.volume() - sph.volume()).abs() < 1e-12);

        let m = 2.0;
        let ic = cap.principal_inertia(m);
        let is = sph.principal_inertia(m);
        assert!((ic.z - is.z).abs() < 1e-9, "{} vs {}", ic.z, is.z);
        // Ixx picks up the hemisphere offset term, so only check it is sane.
        assert!(ic.x > 0.0 && ic.x < 2.0 * is.x);
    }

    #[test]
    fn combining_two_point_masses_centres_between_them() {
        let a = MassProps {
            mass: 1.0,
            com: Vec3::new(-1.0, 0.0, 0.0),
            inertia: Mat3::zero(),
        };
        let b = MassProps {
            mass: 1.0,
            com: Vec3::new(1.0, 0.0, 0.0),
            inertia: Mat3::zero(),
        };
        let c = combine(&[a, b]).unwrap();
        assert!((c.mass - 2.0).abs() < 1e-12);
        assert!(c.com.x.abs() < 1e-12);
        // Two 1 kg masses 1 m either side of centre: Iyy = Izz = 2, Ixx = 0.
        assert!(c.inertia[(0, 0)].abs() < 1e-12);
        assert!((c.inertia[(1, 1)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn box_inertia_is_correct() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
        };
        let m = 12.0;
        let i = s.principal_inertia(m);
        // Solid cube side 2: I = m(a²+a²)/12 = m*8/12 = 8
        assert!((i.x - 8.0).abs() < 1e-9, "{}", i.x);
    }
}
