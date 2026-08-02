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

use phyz_math::{Quat, Vec3};
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
                let current = Quat::exp(&Vec3::new(q[q_off], q[q_off + 1], q[q_off + 2]));
                let next = current.mul(&Quat::exp(&(omega * dt))).normalize();
                let log = next.log();
                q[q_off] = log.x;
                q[q_off + 1] = log.y;
                q[q_off + 2] = log.z;
            }
            JointType::Free => {
                // q = [exp-coords(3), pos(3)], v = [angular(3), linear(3)].
                let omega = Vec3::new(v[v_off], v[v_off + 1], v[v_off + 2]);
                let lin = Vec3::new(v[v_off + 3], v[v_off + 4], v[v_off + 5]);

                let current = Quat::exp(&Vec3::new(q[q_off], q[q_off + 1], q[q_off + 2]));
                // Body-frame linear velocity → parent-frame displacement.
                let world_lin = current.rotate(lin);
                q[q_off + 3] += dt * world_lin.x;
                q[q_off + 4] += dt * world_lin.y;
                q[q_off + 5] += dt * world_lin.z;

                let next = current.mul(&Quat::exp(&(omega * dt))).normalize();
                let log = next.log();
                q[q_off] = log.x;
                q[q_off + 1] = log.y;
                q[q_off + 2] = log.z;
            }
        }
    }
}

/// Advance `state` by `dt` with semi-implicit Euler: velocity first, then
/// position using the *updated* velocity.
pub fn semi_implicit_euler(model: &Model, state: &mut State, qdd: &[f64], dt: f64) {
    for (i, a) in qdd.iter().enumerate().take(model.nv) {
        state.v[i] += dt * a;
    }

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
