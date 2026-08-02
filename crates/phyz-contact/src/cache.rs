//! Contact persistence across timesteps, for warm starting the solve.
//!
//! The narrow phase hands the solver a fresh `Vec<Collision>` every step, in
//! whatever order the broadphase happened to produce, with no identity
//! attached. So the convex solve started from zero impulses every step even
//! though a standing body's stance contacts are solving *almost the same
//! problem* as the step before. Projected Gauss-Seidel converges linearly:
//! from zero it spends its whole iteration budget re-discovering that the feet
//! carry `m g dt`, and on a big stack it often does not get there at all
//! before `max_iterations` — which means [`crate::gradient`] refuses to
//! differentiate the step, because an unconverged iterate is not a KKT point.
//!
//! This module gives contacts a stable identity so last step's impulses can be
//! carried forward.
//!
//! # What makes a key stable
//!
//! Not the world-space contact point: a foot that is perfectly still in
//! world space is the *easy* case, and a rolling wheel or a walking foot moves
//! its world contact point every step even where the contact is physically the
//! same feature. The key is the contact point expressed in the **local frame
//! of the reference body**, quantized to a grid. That is invariant to the
//! body's own rigid motion, so a box sliding across the floor keeps its four
//! corner contacts identified across steps, and a body rotating under a static
//! contact correctly reports a *changing* feature — which it is.
//!
//! Quantization is what makes the key an equality-comparable hash key at all,
//! and the grid must be coarse relative to per-step contact-point jitter
//! (millimetres) and fine relative to the distance between genuinely distinct
//! features (centimetres). [`ContactCache::DEFAULT_CELL`] sits at 2 mm.
//!
//! A mis-keyed contact costs nothing but a bad initial guess: the problem is
//! strongly convex, so every start converges to the same unique minimizer.
//! Warm starting can only change how many iterations that takes, never the
//! answer. That is the property that makes this safe to do at all.

use phyz_collision::Collision;
use phyz_math::Vec3;
use phyz_model::State;
use std::collections::HashMap;

/// A stable identifier for a contact across timesteps.
///
/// Ordered so that the pair `(i, j)` and `(j, i)` cannot collide, and so the
/// quantized point is read in the frame named by `reference`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContactKey {
    /// The lower-numbered body of the pair (`usize::MAX` = world).
    pub body_i: usize,
    /// The higher-numbered body of the pair (`usize::MAX` = world).
    pub body_j: usize,
    /// Contact point in `body_i`'s local frame, quantized to the cache's grid.
    pub cell: [i64; 3],
}

/// Per-step contact impulses, keyed by feature, for warm starting.
///
/// Usage is two calls per step around the solve: [`ContactCache::warm_start`]
/// before it to build the seed vector, [`ContactCache::store`] after it to
/// record the result. `store` replaces the whole table rather than merging, so
/// contacts that ended are dropped and the cache cannot grow without bound.
#[derive(Debug, Clone)]
pub struct ContactCache {
    cell: f64,
    prev: HashMap<ContactKey, Vec3>,
    next: HashMap<ContactKey, Vec3>,
    hits: usize,
    misses: usize,
}

impl Default for ContactCache {
    fn default() -> Self {
        Self::new(Self::DEFAULT_CELL)
    }
}

impl ContactCache {
    /// Default quantization cell, in metres.
    ///
    /// Coarse enough to absorb the sub-millimetre wander of a narrow-phase
    /// contact point on a resting face, fine enough to keep the corners of a
    /// centimetre-scale box distinct.
    pub const DEFAULT_CELL: f64 = 2e-3;

    /// Create a cache with a given quantization cell size, in metres.
    pub fn new(cell: f64) -> Self {
        Self {
            cell: if cell > 0.0 { cell } else { Self::DEFAULT_CELL },
            prev: HashMap::new(),
            next: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// The feature key for one contact at the current state.
    pub fn key(&self, state: &State, c: &Collision) -> ContactKey {
        // Order the pair so the key does not depend on narrow-phase ordering.
        let (lo, hi) = if c.body_i <= c.body_j {
            (c.body_i, c.body_j)
        } else {
            (c.body_j, c.body_i)
        };
        // Express the point in the reference body's frame: p_local =
        // R^T (p - t). Invariant to that body's own rigid motion, which is
        // what a world-space key would lose.
        let local = match state.body_xform.get(lo) {
            Some(x) => x.rot.transpose() * (c.contact_point - x.pos),
            None => c.contact_point,
        };
        let q = |v: f64| (v / self.cell).round() as i64;
        ContactKey {
            body_i: lo,
            body_j: hi,
            cell: [q(local.x), q(local.y), q(local.z)],
        }
    }

    /// Seed impulses for `contacts`, in the same order, from the previous step.
    ///
    /// Contacts with no cached match seed at zero, which is exactly the cold
    /// start. The returned vector is what [`crate::solve_contacts_warm`] takes.
    pub fn warm_start(&mut self, state: &State, contacts: &[Collision]) -> Vec<Vec3> {
        contacts
            .iter()
            .map(|c| match self.prev.get(&self.key(state, c)) {
                Some(f) => {
                    self.hits += 1;
                    *f
                }
                None => {
                    self.misses += 1;
                    Vec3::zeros()
                }
            })
            .collect()
    }

    /// Record the solved impulses for the next step's warm start.
    pub fn store(&mut self, state: &State, contacts: &[Collision], impulses: &[Vec3]) {
        self.next.clear();
        for (c, f) in contacts.iter().zip(impulses) {
            self.next.insert(self.key(state, c), *f);
        }
        std::mem::swap(&mut self.prev, &mut self.next);
    }

    /// Forget every cached contact. Call after teleporting or resetting a
    /// state, where last step's impulses are not a guess about this step.
    pub fn clear(&mut self) {
        self.prev.clear();
        self.next.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// `(hits, misses)` since construction or the last [`Self::clear`].
    ///
    /// A persistently low hit rate on a body that is visibly resting means the
    /// key is not identifying the feature — the cell size is likely wrong for
    /// the scale of the model.
    pub fn stats(&self) -> (usize, usize) {
        (self.hits, self.misses)
    }

    /// Number of contacts currently remembered.
    pub fn len(&self) -> usize {
        self.prev.len()
    }

    /// Whether anything is remembered.
    pub fn is_empty(&self) -> bool {
        self.prev.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{Mat3, SpatialTransform};
    use phyz_model::ModelBuilder;

    fn state_with(xform: SpatialTransform) -> State {
        let model = ModelBuilder::new()
            .dt(1e-3)
            .add_free_body(
                "b",
                -1,
                SpatialTransform::identity(),
                phyz_math::SpatialInertia::sphere(1.0, 0.1),
            )
            .build();
        let mut s = model.default_state();
        s.body_xform[0] = xform;
        s
    }

    fn ground_contact(p: Vec3) -> Collision {
        Collision {
            body_i: 0,
            body_j: usize::MAX,
            contact_point: p,
            contact_normal: Vec3::z(),
            penetration_depth: 1e-4,
        }
    }

    #[test]
    fn a_resting_contact_keeps_its_key_and_warm_starts() {
        let state = state_with(SpatialTransform::identity());
        let contacts = vec![ground_contact(Vec3::new(0.0, 0.0, -0.1))];
        let mut cache = ContactCache::default();

        assert_eq!(cache.warm_start(&state, &contacts)[0], Vec3::zeros());
        let solved = vec![Vec3::new(9.81e-3, 0.0, 0.0)];
        cache.store(&state, &contacts, &solved);

        // Next step, sub-cell jitter in the contact point.
        let jittered = vec![ground_contact(Vec3::new(1e-5, -2e-5, -0.1))];
        let seed = cache.warm_start(&state, &jittered);
        assert_eq!(seed[0], solved[0], "jitter within a cell must still hit");
    }

    /// The point of keying in the body's local frame: a body that translates
    /// keeps the identity of the feature it is resting on.
    #[test]
    fn a_sliding_body_keeps_its_contact_identity() {
        let mut cache = ContactCache::default();
        let s0 = state_with(SpatialTransform::identity());
        let c0 = vec![ground_contact(Vec3::new(0.0, 0.0, -0.1))];
        cache.store(&s0, &c0, &[Vec3::new(1.0, 0.2, 0.0)]);

        // Body has slid 5 cm in +x; the contact point under it moved with it.
        let s1 = state_with(SpatialTransform::new(
            Mat3::identity(),
            Vec3::new(0.05, 0.0, 0.0),
        ));
        let c1 = vec![ground_contact(Vec3::new(0.05, 0.0, -0.1))];
        let seed = cache.warm_start(&s1, &c1);
        assert_eq!(seed[0], Vec3::new(1.0, 0.2, 0.0));
    }

    #[test]
    fn distinct_features_do_not_collide_and_stale_ones_are_evicted() {
        let state = state_with(SpatialTransform::identity());
        let corners = vec![
            ground_contact(Vec3::new(0.05, 0.05, -0.1)),
            ground_contact(Vec3::new(-0.05, 0.05, -0.1)),
        ];
        let mut cache = ContactCache::default();
        cache.store(
            &state,
            &corners,
            &[Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)],
        );
        assert_eq!(cache.len(), 2);
        let seed = cache.warm_start(&state, &corners);
        assert_eq!(seed[0].x, 1.0);
        assert_eq!(seed[1].x, 2.0);

        // Only one corner survives the next step: the other is forgotten.
        cache.store(&state, &corners[..1], &[Vec3::new(3.0, 0.0, 0.0)]);
        assert_eq!(cache.len(), 1);
        let seed = cache.warm_start(&state, &corners);
        assert_eq!(seed[0].x, 3.0);
        assert_eq!(seed[1], Vec3::zeros());
    }
}
