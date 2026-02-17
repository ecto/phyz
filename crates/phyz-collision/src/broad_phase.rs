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

    // Sweep along X axis
    let mut endpoints: Vec<Endpoint> = Vec::new();
    for (i, aabb) in aabbs.iter().enumerate() {
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

    // Sort endpoints by value
    endpoints.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

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
}
