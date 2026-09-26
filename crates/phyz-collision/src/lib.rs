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
use phyz_model::Attachment;

/// Result of a collision query between two geometries.
#[derive(Debug, Clone)]
pub struct Collision {
    /// Index of first body involved.
    pub body_i: usize,
    /// Index of the second body involved, or [`Collision::WORLD`] for the
    /// static world (ground, terrain, any immovable half-space).
    ///
    /// This is an **in-band sentinel**, kept for the moment because the whole
    /// contact pipeline is written against it. Prefer
    /// [`Collision::attachment_j`] over comparing to `WORLD` by hand: it
    /// returns a [`phyz_model::Attachment`], which makes the world a distinct
    /// variant instead of a reserved index, and is what the Jacobian and
    /// constraint APIs take. The hazard the typed form removes is that
    /// `usize::MAX` looks like an ordinary index, so a stale one is silently
    /// reinterpreted as "the world" and the contact quietly stops pushing back
    /// on one side.
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

impl Collision {
    /// The value of [`Collision::body_j`] that means "the static world".
    ///
    /// Named so the magic number appears once, here, instead of being spelled
    /// `usize::MAX` at every comparison site — where it reads as a bounds
    /// check rather than as a semantic test and is easy to get backwards.
    pub const WORLD: usize = usize::MAX;

    /// [`Collision::body_j`] as a typed [`Attachment`].
    ///
    /// This is the accessor to reach for: it turns the sentinel into a variant
    /// at the one place that knows the convention, so consumers never compare
    /// an index against `usize::MAX` themselves.
    #[inline]
    pub fn attachment_j(&self) -> Attachment {
        if self.body_j == Self::WORLD {
            Attachment::World
        } else {
            Attachment::Body(self.body_j)
        }
    }

    /// Whether the second side of this contact is the static world.
    #[inline]
    pub fn is_world_j(&self) -> bool {
        self.body_j == Self::WORLD
    }
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

#[cfg(test)]
mod attachment_tests {
    use super::*;

    fn collision_with(body_j: usize) -> Collision {
        Collision {
            body_i: 0,
            body_j,
            contact_point: Vec3::zeros(),
            contact_normal: Vec3::z(),
            penetration_depth: 0.0,
        }
    }

    /// The sentinel maps to the typed variant, and an ordinary index does not.
    #[test]
    fn attachment_j_translates_the_sentinel() {
        assert_eq!(
            collision_with(Collision::WORLD).attachment_j(),
            Attachment::World
        );
        assert!(collision_with(Collision::WORLD).is_world_j());

        assert_eq!(collision_with(3).attachment_j(), Attachment::Body(3));
        assert!(!collision_with(3).is_world_j());
    }

    /// `Collision::WORLD` is the value the pipeline has always used. If this
    /// ever changes, every `usize::MAX` still spelled out by hand elsewhere in
    /// the workspace silently stops matching, so pin it.
    #[test]
    fn world_sentinel_is_usize_max() {
        assert_eq!(Collision::WORLD, usize::MAX);
    }
}
