//! Joint-aware semi-implicit Euler integration.
//!
//! This is the canonical integrator. Every other integration site in the
//! workspace (including the GPU `INTEGRATE_SHADER`) must match its semantics
//! or floating-base models will drift apart between backends.
//!
//! A flat `q += dt * v` is **wrong** for anything but 1-DOF joints in this
//! codebase, because `q` and `v` do not use the same parameterisation:
//!
//! | joint  | `q` layout                  | `v` layout                   |
//! |--------|-----------------------------|------------------------------|
//! | hinge  | angle                       | angular rate                 |
//! | slide  | displacement                | linear rate                  |
//! | ball   | exp-coords (3)              | body angular velocity (3)    |
//! | free   | `[exp-coords(3), pos(3)]`   | `[angular(3), linear(3)]`    |
//!
//! The free joint's `q` and `v` now agree *slot for slot*: both are angular
//! first, matching `SpatialVec`'s `[angular; linear]` order. They did not used
//! to — `q` was `[pos(3), exp-coords(3)]`, so a flat `q += dt·v` dropped the
//! vertical acceleration into the yaw coordinate and a falling body yawed
//! instead of falling. Slot agreement is *not* a licence to go back to a flat
//! update: the rotational slots are exponential coordinates (a flat add is only
//! first-order accurate near identity and denormalises past π) and the linear
//! slots hold a *body-frame* velocity that has to be rotated into the parent
//! frame first. [`integrate_configuration`] is the single place that knows the
//! mapping; every integration site in the workspace must go through it.

use phyz_math::{Vec3, quat_exp, quat_log};
use phyz_model::{JointType, Model, State};

/// Advance the configuration `q` by `dt` under the generalized velocity `v`.
///
/// This is the canonical `q ← q ⊕ dt·v` for this codebase's joint
/// parameterisations. Rotational sub-blocks take a proper Lie-group step —
/// `R ← R·exp(ω·dt)` composed on quaternions and re-logged — so a spinning
/// body is exact for constant `ω` rather than first-order correct, and a free
/// joint's translation integrates the body-frame linear velocity rotated into
/// the parent frame.
///
/// `q` must be `Model::nq` long and `v` at least `Model::nv`.
pub fn integrate_configuration(model: &Model, q: &mut [f64], v: &[f64], dt: f64) {
    for (jidx, joint) in model.joints.iter().enumerate() {
        let q_off = model.q_offsets[jidx];
        let v_off = model.v_offsets[jidx];
        match joint.joint_type {
            JointType::Fixed => {}
            JointType::Revolute | JointType::Hinge | JointType::Prismatic | JointType::Slide => {
                q[q_off] += dt * v[v_off];
            }
            JointType::Spherical | JointType::Ball => {
                let omega = Vec3::new(v[v_off], v[v_off + 1], v[v_off + 2]);
                let current = quat_exp(&Vec3::new(q[q_off], q[q_off + 1], q[q_off + 2]));
                let next = current.mul(&quat_exp(&(omega * dt))).normalize();
                let log = quat_log(&next);
                q[q_off] = log.x;
                q[q_off + 1] = log.y;
                q[q_off + 2] = log.z;
            }
            JointType::Free => {
                // q = [exp-coords(3), pos(3)], v = [angular(3), linear(3)].
                let omega = Vec3::new(v[v_off], v[v_off + 1], v[v_off + 2]);
                let lin = Vec3::new(v[v_off + 3], v[v_off + 4], v[v_off + 5]);

                let current = quat_exp(&Vec3::new(q[q_off], q[q_off + 1], q[q_off + 2]));
                let next = current.mul(&quat_exp(&(omega * dt))).normalize();
                // Body-frame linear velocity → parent-frame displacement. The
                // velocity is the step's *updated* one, expressed in the frame
                // it will have at the end of the step (that is what the
                // `−ω × v` term in its update means), so it is rotated by
                // `next`, not `current`. With `current` a body spinning at
                // `ω` travels `cos(|ω| dt)` of its speed.
                let world_lin = next.rotate(lin);
                q[q_off + 3] += dt * world_lin.x;
                q[q_off + 4] += dt * world_lin.y;
                q[q_off + 5] += dt * world_lin.z;
                let log = quat_log(&next);
                q[q_off] = log.x;
                q[q_off + 1] = log.y;
                q[q_off + 2] = log.z;
            }
        }
    }
}

/// Take the frame-turn term out of a free joint's linear acceleration.
///
/// A free joint's linear velocity lives in the body frame, so under a
/// constant world velocity it obeys `v̇ = −ω × v`: not a force, the frame
/// turning. [`aba`](crate::aba) reports that term as part of `qdd`, and an
/// explicit velocity update `v += dt·qdd` then turns *and* stretches the
/// vector — `|v|` grows by `(|ω| dt)² / 2` a step, which for a 27 mm wheel
/// rolling at 2 m/s (74 rad/s, 1 ms steps) is a runaway in half a second.
///
/// This adds `ω × v` back, so `v + dt·qdd` is the velocity a non-rotating
/// observer sees, expressed in the frame the step *started* in — the frame
/// every Jacobian of the step (contacts included) was assembled in. After
/// the velocity update, [`rotate_free_joint_velocities`] turns it into the
/// frame the step ends in, exactly. Together they replace the first-order
/// turn with `R(−ω dt)`, which is exact for constant `ω`; the rotational
/// slots already get that treatment in [`integrate_configuration`].
pub fn strip_free_joint_coriolis(model: &Model, v: &[f64], qdd: &mut [f64]) {
    for (jidx, joint) in model.joints.iter().enumerate() {
        if joint.joint_type != JointType::Free {
            continue;
        }
        let off = model.v_offsets[jidx];
        let omega = Vec3::new(v[off], v[off + 1], v[off + 2]);
        let lin = Vec3::new(v[off + 3], v[off + 4], v[off + 5]);
        let c = omega.cross(lin);
        qdd[off + 3] += c.x;
        qdd[off + 4] += c.y;
        qdd[off + 5] += c.z;
    }
}

/// Express each free joint's linear velocity in the frame the step ends in.
///
/// The second half of [`strip_free_joint_coriolis`]: the updated velocity
/// `v` is in the frame the step started in, and the body has since turned
/// by `ω dt`, so the same world vector reads `R(−ω dt)·v` in the new frame.
/// Call it after every contribution to the step's velocity and before
/// [`integrate_configuration`], which reads the linear slots in that frame.
///
/// `ω` is read from `v_before`, the velocity the step started from — the
/// same `ω` the strip used. The pair has to agree: on a stack of boxes
/// chattering under the contact solver the impulses move `ω` by as much as
/// `ω` itself every step, and a turn taken with the post-solve `ω` against a
/// strip taken with the pre-solve one leaves a bias of `Δω dt × v` a step,
/// which walked a settled five-box stack three centimetres sideways in ten
/// seconds. With one `ω` the scheme is exact for constant `ω` and agrees
/// with the first-order one to `O((ω dt)²)` otherwise.
pub fn rotate_free_joint_velocities(model: &Model, v_before: &[f64], v: &mut [f64], dt: f64) {
    for (jidx, joint) in model.joints.iter().enumerate() {
        if joint.joint_type != JointType::Free {
            continue;
        }
        let off = model.v_offsets[jidx];
        let omega = Vec3::new(v_before[off], v_before[off + 1], v_before[off + 2]);
        let theta = omega.norm() * dt;
        if theta < 1e-12 || !theta.is_finite() {
            continue;
        }
        let lin = Vec3::new(v[off + 3], v[off + 4], v[off + 5]);
        let turned = quat_exp(&(omega * -dt)).rotate(lin);
        v[off + 3] = turned.x;
        v[off + 4] = turned.y;
        v[off + 5] = turned.z;
    }
}

/// Advance `state` by `dt` with semi-implicit Euler: velocity first, then
/// position using the *updated* velocity.
pub fn semi_implicit_euler(model: &Model, state: &mut State, qdd: &[f64], dt: f64) {
    let mut qdd = qdd[..model.nv].to_vec();
    let v_before = state.v.clone();
    strip_free_joint_coriolis(model, v_before.as_slice(), &mut qdd);
    for (i, a) in qdd.iter().enumerate() {
        state.v[i] += dt * a;
    }
    rotate_free_joint_velocities(model, v_before.as_slice(), state.v.as_mut_slice(), dt);

    let v = state.v.clone();
    integrate_configuration(model, state.q.as_mut_slice(), v.as_slice(), dt);

    state.time += dt;
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{SpatialInertia, SpatialTransform};
    use phyz_model::ModelBuilder;

    #[test]
    fn hinge_integrates_scalar() {
        let model = ModelBuilder::new()
            .dt(0.01)
            .add_revolute_body(
                "l",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::point_mass(1.0, Vec3::new(0.0, 0.0, -0.5)),
            )
            .build();
        let mut s = model.default_state();
        semi_implicit_euler(&model, &mut s, &[2.0], 0.1);
        // v = 0 + 0.1*2 = 0.2, q = 0 + 0.1*0.2 = 0.02
        assert!((s.v[0] - 0.2).abs() < 1e-12);
        assert!((s.q[0] - 0.02).abs() < 1e-12);
        assert!((s.time - 0.1).abs() < 1e-12);
    }
}
