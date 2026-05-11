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
/// Returns pairs of body indices that have overlapping AABBs.
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

    // Sort endpoints by value. We use `total_cmp` so any residual NaN that
    // slips past `aabb_is_finite` is still totally ordered and cannot panic.
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

    #[test]
    fn test_sweep_and_prune_nan_aabb_is_ignored() {
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
            AABB::new(
                Vec3::new(0.5, 0.0, 0.0),
                Vec3::new(f64::INFINITY, 1.0, 1.0),
            ),
        ];
        let pairs = sweep_and_prune(&aabbs);
        assert!(pairs.is_empty());
    }
}
