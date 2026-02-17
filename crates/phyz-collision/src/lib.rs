//! Collision detection for phyz physics engine.
//!
//! Implements GJK (Gilbert-Johnson-Keerthi) for distance queries,
//! EPA (Expanding Polytope Algorithm) for penetration depth,
//! and broad-phase spatial hashing.

pub mod broad_phase;
pub mod epa;
pub mod geometry;
pub mod gjk;

pub use broad_phase::sweep_and_prune;
pub use epa::{epa_penetration, epa_penetration_rot};
pub use geometry::{AABB, Geometry};
pub use gjk::{gjk_distance, gjk_distance_rot};

use phyz_math::Vec3;

/// Result of a collision query between two geometries.
#[derive(Debug, Clone)]
pub struct Collision {
    /// Index of first body involved.
    pub body_i: usize,
    /// Index of second body involved.
    pub body_j: usize,
    /// Contact point in world frame.
    pub contact_point: Vec3,
    /// Contact normal (direction from i to j).
    pub contact_normal: Vec3,
    /// Penetration depth (positive = penetrating).
    pub penetration_depth: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use geometry::Geometry;
    use phyz_math::Vec3;

    #[test]
    fn test_sphere_sphere_separated() {
        let s1 = Geometry::Sphere { radius: 1.0 };
        let s2 = Geometry::Sphere { radius: 1.0 };
        let p1 = Vec3::zeros();
        let p2 = Vec3::new(3.0, 0.0, 0.0);

        let dist = gjk_distance(&s1, &s2, &p1, &p2);
        assert!(
            (dist - 1.0).abs() < 1e-6,
            "Expected distance ~1.0, got {}",
            dist
        );
    }

    #[test]
    fn test_sphere_sphere_touching() {
        let s1 = Geometry::Sphere { radius: 1.0 };
        let s2 = Geometry::Sphere { radius: 1.0 };
        let p1 = Vec3::zeros();
        let p2 = Vec3::new(2.0, 0.0, 0.0);

        let dist = gjk_distance(&s1, &s2, &p1, &p2);
        assert!(dist.abs() < 1e-6, "Expected distance ~0.0, got {}", dist);
    }

    #[test]
    fn test_sphere_sphere_penetrating() {
        let s1 = Geometry::Sphere { radius: 1.0 };
        let s2 = Geometry::Sphere { radius: 1.0 };
        let p1 = Vec3::zeros();
        let p2 = Vec3::new(1.5, 0.0, 0.0);

        let dist = gjk_distance(&s1, &s2, &p1, &p2);
        // GJK returns -1.0 for penetration (approximate), EPA gives exact depth
        assert!(
            dist <= 0.0,
            "Expected non-positive distance (penetration), got {}",
            dist
        );

        if let Some(pen) = epa_penetration(&s1, &s2, &p1, &p2) {
            // Penetration depth should be approximately 0.5
            // (1.5 distance between centers, 2.0 combined radii => 0.5 overlap)
            assert!(
                (pen.0 - 0.5).abs() < 0.1,
                "Expected penetration ~0.5, got {}",
                pen.0
            );
        }
    }
}
