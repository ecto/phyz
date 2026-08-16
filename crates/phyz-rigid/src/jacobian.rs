//! Point Jacobians: how a world point on a body moves with each joint DOF.
//!
//! For a material point `p` fixed in body `b`, the point Jacobian `J_p` is the
//! `3 x nv` matrix with
//!
//! ```text
//! v_p = J_p * qd
//! ```
//!
//! where `v_p` is the point's world-frame linear velocity. This is what turns
//! a contact — a point, a normal, and two tangents — into rows of a constraint
//! system, and it is the missing piece between the narrow phase and the
//! contact solve: the Delassus operator `A = J M^-1 J^T` cannot be assembled
//! without it.
//!
//! Only DOFs on the path from the world to `b` move the point; every other
//! column is zero.

use phyz_math::{DMat, SpatialTransform, SpatialTransformExt, Vec3};
use phyz_model::{JointType, Model};

/// The `3 x nv` world-frame linear-velocity Jacobian of a point fixed in
/// `body`, evaluated at the configuration in `state`.
///
/// `point` is in **world** coordinates. `xforms` must come from
/// [`crate::forward_kinematics`] for the same state.
pub fn point_jacobian(
    model: &Model,
    xforms: &[SpatialTransform],
    body: usize,
    point: Vec3,
) -> DMat {
    let mut j = DMat::zeros(3, model.nv);

    // Walk from `body` up to the root; each ancestor's joint DOFs move it.
    let mut cur = body as i32;
    while cur >= 0 {
        let b = cur as usize;
        let bodyref = &model.bodies[b];
        let joint = &model.joints[bodyref.joint_idx];
        let ndof = joint.ndof();
        if ndof > 0 {
            let v_idx = model.v_offsets[bodyref.joint_idx];
            let xf = &xforms[b];
            let origin = xf.pos;
            let axis_world = xf.body_to_world_dir(joint.axis);

            // Unit body-frame axis for multi-DOF joints, rotated to world.
            let body_axis = |k: usize| {
                xf.body_to_world_dir(match k {
                    0 => Vec3::x(),
                    1 => Vec3::y(),
                    _ => Vec3::z(),
                })
            };
            let arm = point - origin;

            for d in 0..ndof {
                // Angular DOFs contribute `omega x r`; linear DOFs contribute
                // the axis directly.
                let contribution = match joint.joint_type {
                    JointType::Revolute | JointType::Hinge => axis_world.cross(arm),
                    JointType::Prismatic | JointType::Slide => axis_world,
                    JointType::Spherical | JointType::Ball => body_axis(d).cross(arm),
                    JointType::Free if d < 3 => body_axis(d).cross(arm),
                    JointType::Free => body_axis(d - 3),
                    JointType::Fixed => Vec3::zeros(),
                };
                j[(0, v_idx + d)] = contribution.x;
                j[(1, v_idx + d)] = contribution.y;
                j[(2, v_idx + d)] = contribution.z;
            }
        }
        cur = bodyref.parent;
    }

    j
}

/// The `3 x nv` world-frame **angular**-velocity Jacobian of `body`:
///
/// ```text
/// omega_body_in_world = J_w * qd
/// ```
///
/// The rotational counterpart of [`point_jacobian`], and the other half of a
/// full 6-D body Jacobian. Prismatic DOFs contribute nothing; a free joint
/// contributes only through its three rotational DOFs.
///
/// `xforms` must come from [`crate::forward_kinematics`] for the same state.
pub fn body_angular_jacobian(model: &Model, xforms: &[SpatialTransform], body: usize) -> DMat {
    let mut j = DMat::zeros(3, model.nv);

    // Same ancestor walk as `point_jacobian`; only the per-DOF contribution
    // differs, because angular velocity does not depend on the moment arm.
    let mut cur = body as i32;
    while cur >= 0 {
        let b = cur as usize;
        let bodyref = &model.bodies[b];
        let joint = &model.joints[bodyref.joint_idx];
        let ndof = joint.ndof();
        if ndof > 0 {
            let v_idx = model.v_offsets[bodyref.joint_idx];
            let xf = &xforms[b];
            let axis_world = xf.body_to_world_dir(joint.axis);
            let body_axis = |k: usize| {
                xf.body_to_world_dir(match k {
                    0 => Vec3::x(),
                    1 => Vec3::y(),
                    _ => Vec3::z(),
                })
            };

            for d in 0..ndof {
                let contribution = match joint.joint_type {
                    JointType::Revolute | JointType::Hinge => axis_world,
                    JointType::Prismatic | JointType::Slide => Vec3::zeros(),
                    JointType::Spherical | JointType::Ball => body_axis(d),
                    JointType::Free if d < 3 => body_axis(d),
                    JointType::Free => Vec3::zeros(),
                    JointType::Fixed => Vec3::zeros(),
                };
                j[(0, v_idx + d)] = contribution.x;
                j[(1, v_idx + d)] = contribution.y;
                j[(2, v_idx + d)] = contribution.z;
            }
        }
        cur = bodyref.parent;
    }

    j
}

/// The `3 x nv` Jacobian of the *relative* velocity of two coincident points,
/// one on each body: `J_i - J_j`.
///
/// `body_j == usize::MAX` denotes the static world, in which case this is just
/// the Jacobian of the point on `body_i`.
pub fn relative_point_jacobian(
    model: &Model,
    xforms: &[SpatialTransform],
    body_i: usize,
    body_j: usize,
    point: Vec3,
) -> DMat {
    let ji = point_jacobian(model, xforms, body_i, point);
    if body_j == usize::MAX {
        return ji;
    }
    let jj = point_jacobian(model, xforms, body_j, point);
    let mut out = DMat::zeros(3, model.nv);
    for r in 0..3 {
        for c in 0..model.nv {
            out[(r, c)] = ji[(r, c)] - jj[(r, c)];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward_kinematics;
    use phyz_math::{GRAVITY, Mat3, SpatialInertia};
    use phyz_model::ModelBuilder;

    fn pendulum() -> Model {
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(1e-3)
            .add_revolute_body(
                "link",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(
                    1.0,
                    Vec3::new(0.0, -0.5, 0.0),
                    Mat3::from_diagonal(&Vec3::new(0.1, 0.1, 0.1)),
                ),
            )
            .build()
    }

    /// The Jacobian must reproduce the point velocity that forward kinematics
    /// already computes — an independent cross-check of both.
    #[test]
    fn jacobian_reproduces_point_velocity() {
        let model = pendulum();
        let mut state = model.default_state();
        state.q[0] = 0.3;
        state.v[0] = 1.7;
        let (xforms, _) = forward_kinematics(&model, &state);

        // A point 1 m down the link, in world coordinates.
        let xf = &xforms[0];
        let p_world = xf.body_to_world_point(Vec3::new(0.0, -1.0, 0.0));

        let j = point_jacobian(&model, &xforms, 0, p_world);
        let v = Vec3::new(
            j[(0, 0)] * state.v[0],
            j[(1, 0)] * state.v[0],
            j[(2, 0)] * state.v[0],
        );

        // Revolute about +Z through the origin: v = omega x r.
        let expected = (Vec3::z() * state.v[0]).cross(p_world);
        assert!(
            (v - expected).norm() < 1e-9,
            "jacobian velocity {v:?} vs omega x r {expected:?}"
        );
    }

    /// Finite-difference check: `J` is the derivative of the point position
    /// with respect to `q`. This is the pattern used in phyz-regge and
    /// phyz-quantum.
    #[test]
    fn jacobian_matches_finite_difference() {
        let model = pendulum();
        let mut state = model.default_state();
        state.q[0] = 0.37;

        let point_at = |q: f64| {
            let mut s = state.clone();
            s.q[0] = q;
            let (xf, _) = forward_kinematics(&model, &s);
            // Track the *material* point: fixed in the body frame.
            xf[0].body_to_world_point(Vec3::new(0.2, -0.9, 0.05))
        };

        let (xforms, _) = forward_kinematics(&model, &state);
        let p0 = point_at(state.q[0]);
        let j = point_jacobian(&model, &xforms, 0, p0);

        let mut best = f64::INFINITY;
        let mut prev = f64::INFINITY;
        for k in 0..4 {
            let h = 1e-3 / 10f64.powi(k);
            let fd = (point_at(state.q[0] + h) - point_at(state.q[0] - h)) / (2.0 * h);
            let analytic = Vec3::new(j[(0, 0)], j[(1, 0)], j[(2, 0)]);
            let err = (fd - analytic).norm() / analytic.norm().max(1e-12);
            best = best.min(err);
            // Error should shrink as h does, at least initially — the
            // signature of a correct derivative rather than a wrong formula.
            if k < 2 {
                assert!(err <= prev, "error grew with smaller h: {prev} -> {err}");
            }
            prev = err;
        }
        assert!(best < 1e-6, "best relative error {best}");
    }

    /// A revolute joint's angular Jacobian column is its world-frame axis, and
    /// unlike the point Jacobian it does not depend on where the point is.
    #[test]
    fn angular_jacobian_is_the_joint_axis() {
        let model = pendulum();
        let mut state = model.default_state();
        state.q[0] = 0.6;
        let (xforms, _) = forward_kinematics(&model, &state);

        let jw = body_angular_jacobian(&model, &xforms, 0);
        let col = Vec3::new(jw[(0, 0)], jw[(1, 0)], jw[(2, 0)]);
        assert!((col - Vec3::z()).norm() < 1e-12, "column {col:?}");
    }

    /// A free body rotates through its three angular DOFs and not at all
    /// through its three translational ones.
    #[test]
    fn free_body_angular_jacobian_ignores_translation() {
        let model = ModelBuilder::new()
            .dt(1e-3)
            .add_free_body(
                "ball",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::sphere(1.0, 0.1),
            )
            .build();
        let state = model.default_state();
        let (xforms, _) = forward_kinematics(&model, &state);
        let jw = body_angular_jacobian(&model, &xforms, 0);

        for (k, axis) in [Vec3::x(), Vec3::y(), Vec3::z()].iter().enumerate() {
            let col = Vec3::new(jw[(0, k)], jw[(1, k)], jw[(2, k)]);
            assert!((col - *axis).norm() < 1e-12, "angular column {k}: {col:?}");
            for r in 0..3 {
                assert_eq!(jw[(r, 3 + k)], 0.0, "translation moved the orientation");
            }
        }
    }

    /// Only DOFs on the path to the body may be non-zero.
    #[test]
    fn unrelated_dofs_are_zero() {
        let model = ModelBuilder::new()
            .dt(1e-3)
            .add_revolute_body(
                "a",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity()),
            )
            .add_revolute_body(
                "b",
                -1,
                SpatialTransform::new(Mat3::identity(), Vec3::new(2.0, 0.0, 0.0)),
                SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity()),
            )
            .build();
        let state = model.default_state();
        let (xforms, _) = forward_kinematics(&model, &state);

        // A point on body 0 must not respond to body 1's DOF.
        let j = point_jacobian(&model, &xforms, 0, Vec3::new(0.0, 1.0, 0.0));
        for r in 0..3 {
            assert_eq!(j[(r, 1)], 0.0, "body 1's DOF moved a point on body 0");
        }
    }

    /// A free body translates its points one-for-one and rotates them about
    /// its own origin.
    #[test]
    fn free_body_jacobian_has_translation_and_rotation() {
        let model = ModelBuilder::new()
            .dt(1e-3)
            .add_free_body(
                "ball",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::sphere(1.0, 0.1),
            )
            .build();
        let state = model.default_state();
        let (xforms, _) = forward_kinematics(&model, &state);
        let p = Vec3::new(0.0, 0.0, 1.0);
        let j = point_jacobian(&model, &xforms, 0, p);

        // Translational DOFs (3..6) give the identity.
        for (k, axis) in [Vec3::x(), Vec3::y(), Vec3::z()].iter().enumerate() {
            let col = Vec3::new(j[(0, 3 + k)], j[(1, 3 + k)], j[(2, 3 + k)]);
            assert!(
                (col - *axis).norm() < 1e-12,
                "translation column {k}: {col:?}"
            );
        }
        // Rotation about +x moves a point on +z toward -y (omega x r).
        let col = Vec3::new(j[(0, 0)], j[(1, 0)], j[(2, 0)]);
        assert!(
            (col - Vec3::x().cross(p)).norm() < 1e-12,
            "rotation column: {col:?}"
        );
    }
}
