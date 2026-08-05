//! Collision detection for phyz physics engine.
//!
//! Implements GJK (Gilbert-Johnson-Keerthi) for distance queries,
//! EPA (Expanding Polytope Algorithm) for penetration depth,
//! analytic ray casting, and broad-phase spatial hashing.

#![warn(missing_docs)]

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

pub mod broad_phase;
pub mod epa;
pub mod geometry;
pub mod gjk;
pub mod manifold;
pub mod raycast;

pub use broad_phase::sweep_and_prune;
pub use epa::{epa_from_simplex, epa_penetration, epa_penetration_rot};
pub use geometry::{AABB, Geometry};
pub use gjk::{GjkOutcome, gjk_distance, gjk_distance_rot, gjk_rot};
pub use manifold::{
    MAX_MANIFOLD_POINTS, Manifold, ManifoldPoint, contact_manifold, contact_manifold_within,
};
pub use raycast::{Ray, RayHit, ray_cast, ray_intersect};

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
    /// Contact normal: **the direction `body_i` must move to separate from
    /// `body_j`**, i.e. it points from `j` toward `i`.
    ///
    /// Stated this way because the two producers used to disagree and only one
    /// of them was ever exercised. `find_ground_contacts` emits `+z` for a
    /// body resting on the ground — the separating direction for that body,
    /// and the sense the contact solver is built around, since its
    /// non-penetration row is `J·v ≥ 0` with `J = J_i − J_j`.
    /// `find_contacts` used to pass [`Manifold::normal`] straight through, and
    /// that points from shape *a* toward shape *b* — the exact opposite. The
    /// constraint therefore measured approach instead of separation and the
    /// solver drove overlapping bodies *together*: a box dropped on a box sank
    /// clean through it to full overlap, with four contacts detected the whole
    /// way down. Invisible until bodies other than the feet had collision
    /// geometry, because before that no body-body contact ever occurred.
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
