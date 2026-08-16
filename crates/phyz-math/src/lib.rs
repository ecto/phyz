//! Spatial algebra and math primitives for phyz physics engine.
//!
//! Implements 6D spatial vectors, Plucker transforms, and spatial inertia
//! following Featherstone's conventions.

#![warn(missing_docs)]

// Compile the crate README's Rust blocks as doc-tests so the documented API
// cannot drift from the real one. `cfg(doctest)` keeps it out of rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDocTests;

pub mod fp;
pub mod spatial;

pub use spatial::{SpatialInertia, SpatialMat, SpatialTransform, SpatialTransformExt, SpatialVec};

/// Unit quaternion for 3D rotations (scalar `w`, vector `v`).
pub type Quat = tang::Quat<f64>;

/// 3D vector alias.
pub type Vec3 = tang::Vec3<f64>;
/// 3x3 matrix alias.
pub type Mat3 = tang::Mat3<f64>;
/// 4x4 matrix alias.
pub type Mat4 = tang::Mat4<f64>;
/// Dynamic vector.
pub type DVec = tang_la::DVec<f64>;
/// Dynamic matrix.
pub type DMat = tang_la::DMat<f64>;

pub use tang::{GRAVITY, skew};

/// Exponential map `so(3) → SO(3)`, computed through [`fp`].
///
/// Numerically identical to [`tang::Quat::exp`] except that the `sin`/`cos`
/// come from [`fp::sin_cos`] rather than the platform's libm. Every rotational
/// configuration update in the engine goes through this function — a free
/// joint touches it twice per step — so it is the single highest-traffic
/// transcendental site in a rollout, and the one that most needs to return the
/// same bits on every machine. See `docs/determinism.md`.
pub fn quat_exp(omega: &Vec3) -> Quat {
    let angle = omega.norm();
    // Same small-angle branch as tang: below one epsilon the series is the
    // identity plus half the rotation vector, and dividing by `angle` would be
    // the thing that loses precision rather than the thing that gains it.
    if angle < f64::EPSILON {
        return Quat {
            w: 1.0,
            v: *omega * 0.5,
        };
    }
    let (s, c) = fp::sin_cos(angle * 0.5);
    Quat {
        w: c,
        v: *omega * (s / angle),
    }
}

/// Logarithmic map `SO(3) → so(3)`, computed through [`fp`].
///
/// The inverse of [`quat_exp`], and likewise identical to [`tang::Quat::log`]
/// apart from routing its `atan2` through [`fp::atan2`].
pub fn quat_log(q: &Quat) -> Vec3 {
    let norm_v = q.v.norm();
    if norm_v < f64::EPSILON {
        return q.v * 2.0;
    }
    let angle = 2.0 * fp::atan2(norm_v, q.w);
    q.v * (angle / norm_v)
}

#[cfg(test)]
mod quat_map_tests {
    use super::*;

    #[test]
    fn matches_tang_to_a_few_ulp_and_round_trips() {
        for w in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1e-18, 0.0, 0.0),
            Vec3::new(0.3, -0.7, 1.1),
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(-2.0, 0.5, 0.25),
        ] {
            let ours = quat_exp(&w);
            let theirs = Quat::exp(&w);
            let tol = 8.0 * f64::EPSILON;
            assert!((ours.w - theirs.w).abs() <= tol, "exp w for {w:?}");
            assert!((ours.v - theirs.v).norm() <= tol, "exp v for {w:?}");

            let back = quat_log(&ours);
            assert!((back - w).norm() <= 1e-12, "round trip for {w:?}");
            assert!(
                (quat_log(&theirs) - theirs.log()).norm() <= tol,
                "log {w:?}"
            );
        }
    }
}
