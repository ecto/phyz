//! Broad-phase collision detection using sweep-and-prune.

use crate::geometry::AABB;

/// Potential collision pair (body indices).
pub type CollisionPair = (usize, usize);

/// Endpoint for sweep-and-prune.
#[derive(Clone, Copy)]
struct Endpoint {
    value: f64,
    body_idx: usize,
    is_min: bool,
}

/// Sweep-and-prune broad phase collision detection.
///
/// Returns pairs of body indices that have overlapping AABBs, **sorted
/// ascending by `(i, j)`** — by identity, not by where the sweep happened to
/// find them.
///
/// # Why the sort is not cosmetic
///
/// The sweep visits endpoints in order of their `x` coordinate, so the order
/// pairs come out in is a function of *continuous positions*. That order is
/// carried all the way to the solver: `find_contacts` emits contacts in pair
/// order, `assemble` builds the Delassus rows in contact order, and projected
/// Gauss-Seidel sweeps those rows in order. Gauss-Seidel is not
/// order-invariant — its iterates differ, and since the solve terminates on a
/// tolerance or an iteration cap rather than at the exact minimizer, so does
/// its answer.
///
/// Which means: without this sort, two bodies whose x-extents cross swap the
/// solver's sweep order mid-rollout, and a perturbation far below the
/// discretization — one ulp — can flip that crossing a step early and change
/// the trajectory discontinuously. That is indistinguishable from a bug, and
/// it is the reason this repository could not reproduce a result across two
/// machines. Sorting by index makes the solve order a function of the model's
/// identity, which does not move under perturbation at all.
///
/// The sort is on `usize` pairs, so it introduces no floating-point comparison
/// of its own, and `O(p log p)` on a pair list that the narrow phase is about
/// to do `O(p)` GJK/EPA solves over is not a measurable cost.
pub fn sweep_and_prune(aabbs: &[AABB]) -> Vec<CollisionPair> {
    if aabbs.len() < 2 {
        return Vec::new();
    }

    let mut pairs = Vec::new();

    // Sweep along X axis. Skip bodies whose AABB has any non-finite component
    // (NaN or infinity); they would either panic in `partial_cmp` below or
    // pollute the active set with garbage pairs. The simulator gets to keep
    // running rather than crashing the whole step.
    let mut endpoints: Vec<Endpoint> = Vec::new();
    for (i, aabb) in aabbs.iter().enumerate() {
        if !aabb_is_finite(aabb) {
            continue;
        }
        endpoints.push(Endpoint {
            value: aabb.min.x,
            body_idx: i,
            is_min: true,
        });
        endpoints.push(Endpoint {
            value: aabb.max.x,
            body_idx: i,
            is_min: false,
        });
    }

    // Sort endpoints by value. We use `total_cmp` so any residual NaN that slips
    // past `aabb_is_finite` (e.g. a denormal mishandled by an underlying SIMD
    // op) is still totally ordered and cannot panic.
    endpoints.sort_by(|a, b| a.value.total_cmp(&b.value));

    // Sweep and collect pairs
    let mut active = Vec::new();
    for ep in endpoints {
        if ep.is_min {
            // Check against all active bodies
            for &other in &active {
                if aabbs[ep.body_idx].overlaps(&aabbs[other]) {
                    let pair = if ep.body_idx < other {
                        (ep.body_idx, other)
                    } else {
                        (other, ep.body_idx)
                    };
                    pairs.push(pair);
                }
            }
            active.push(ep.body_idx);
        } else {
            // Remove from active set
            active.retain(|&x| x != ep.body_idx);
        }
    }

    // Canonical order: by identity, so the narrow phase and the contact solver
    // see the same sweep order for the same model regardless of where the
    // bodies are. `sort_unstable` is fine — the keys are unique, because a
    // pair is emitted at most once (only the `is_min` endpoint emits, and each
    // body has exactly one).
    pairs.sort_unstable();

    pairs
}

/// True iff every component of the AABB's min and max is finite (not NaN,
/// not ±infinity). Used to skip degenerate bodies in the broad phase.
fn aabb_is_finite(aabb: &AABB) -> bool {
    aabb.min.x.is_finite()
        && aabb.min.y.is_finite()
        && aabb.min.z.is_finite()
        && aabb.max.x.is_finite()
        && aabb.max.y.is_finite()
        && aabb.max.z.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::Vec3;

    #[test]
    fn test_sweep_and_prune_no_overlap() {
        let aabbs = vec![
            AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
            AABB::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
        ];
        let pairs = sweep_and_prune(&aabbs);
        assert_eq!(pairs.len(), 0);
    }

    #[test]
    fn test_sweep_and_prune_overlap() {
        let aabbs = vec![
            AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.5, 1.0, 1.0)),
            AABB::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 1.0)),
        ];
        let pairs = sweep_and_prune(&aabbs);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (0, 1));
    }

    /// The pair list must be ordered by identity, not by position along the
    /// sweep axis. Two boxes overlapping a third are laid out so the sweep
    /// *discovers* them in the reverse of index order; the output must not
    /// care. See the function docs for what depends on this.
    #[test]
    fn pairs_come_out_in_canonical_index_order() {
        // Body 0 spans the whole line; bodies 1 and 2 sit inside it, with 2 to
        // the left of 1 so the sweep reaches 2 first.
        let aabbs = vec![
            AABB::new(Vec3::new(-10.0, 0.0, 0.0), Vec3::new(10.0, 1.0, 1.0)),
            AABB::new(Vec3::new(5.0, 0.0, 0.0), Vec3::new(6.0, 1.0, 1.0)),
            AABB::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 1.0)),
        ];
        let pairs = sweep_and_prune(&aabbs);
        assert_eq!(pairs, vec![(0, 1), (0, 2)]);

        // Translating the whole scene, or swapping which of the two inner
        // boxes is further left, must not change the output at all.
        let shifted: Vec<AABB> = aabbs
            .iter()
            .map(|a| {
                AABB::new(
                    a.min + Vec3::new(37.5, 0.0, 0.0),
                    a.max + Vec3::new(37.5, 0.0, 0.0),
                )
            })
            .collect();
        assert_eq!(sweep_and_prune(&shifted), pairs);

        let swapped = vec![
            aabbs[0],
            AABB::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 1.0)),
            AABB::new(Vec3::new(5.0, 0.0, 0.0), Vec3::new(6.0, 1.0, 1.0)),
        ];
        assert_eq!(sweep_and_prune(&swapped), pairs);
    }

    /// The invariant in general form: whatever the geometry, the list is
    /// sorted and each pair has `i < j`.
    #[test]
    fn pair_list_is_sorted_and_lower_triangular() {
        let mut aabbs = Vec::new();
        for k in 0..12 {
            // Deliberately non-monotone in the index so sweep order and index
            // order disagree.
            let x = ((k * 7) % 12) as f64 * 0.4;
            aabbs.push(AABB::new(
                Vec3::new(x, 0.0, 0.0),
                Vec3::new(x + 1.0, 1.0, 1.0),
            ));
        }
        let pairs = sweep_and_prune(&aabbs);
        assert!(!pairs.is_empty());
        for w in pairs.windows(2) {
            assert!(w[0] < w[1], "not sorted: {:?} then {:?}", w[0], w[1]);
        }
        for &(i, j) in &pairs {
            assert!(i < j, "pair ({i}, {j}) is not lower-triangular");
        }
    }

    #[test]
    fn test_sweep_and_prune_nan_aabb_is_ignored() {
        // A poisoned AABB (NaN component) must not crash the sort or produce
        // spurious pairs against the live body.
        let nan = f64::NAN;
        let aabbs = vec![
            AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
            AABB::new(Vec3::new(nan, 0.0, 0.0), Vec3::new(2.0, 1.0, 1.0)),
        ];
        let pairs = sweep_and_prune(&aabbs);
        assert!(pairs.is_empty(), "got unexpected pairs {pairs:?}");
    }

    #[test]
    fn test_sweep_and_prune_infinity_is_ignored() {
        let aabbs = vec![
            AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
            AABB::new(Vec3::new(0.5, 0.0, 0.0), Vec3::new(f64::INFINITY, 1.0, 1.0)),
        ];
        let pairs = sweep_and_prune(&aabbs);
        assert!(pairs.is_empty());
    }
}
