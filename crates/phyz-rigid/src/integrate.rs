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
//! | free   | `[pos(3), exp-coords(3)]`   | `[angular(3), linear(3)]`    |
//!
//! Note the free joint: `q` is position-then-rotation while `v` is
//! rotation-then-position (`kinematics.rs` `joint_velocity` vs
//! `phyz_model::Joint::joint_transform_slice`). Adding them elementwise mixes
//! angular velocity into position. This module is the single place that knows
//! the mapping.

use phyz_math::{Quat, Vec3};
use phyz_model::{JointType, Model, State};

/// Advance `state` by `dt` with semi-implicit Euler: velocity first, then
/// position using the *updated* velocity.
pub fn semi_implicit_euler(model: &Model, state: &mut State, qdd: &[f64], dt: f64) {
    for (i, a) in qdd.iter().enumerate().take(model.nv) {
        state.v[i] += dt * a;
    }

    for (jidx, joint) in model.joints.iter().enumerate() {
        let q_off = model.q_offsets[jidx];
        let v_off = model.v_offsets[jidx];
        match joint.joint_type {
            JointType::Fixed => {}
            JointType::Revolute | JointType::Hinge | JointType::Prismatic | JointType::Slide => {
                state.q[q_off] += dt * state.v[v_off];
            }
            JointType::Spherical | JointType::Ball => {
                let omega = Vec3::new(state.v[v_off], state.v[v_off + 1], state.v[v_off + 2]);
                let current = Quat::exp(&Vec3::new(
                    state.q[q_off],
                    state.q[q_off + 1],
                    state.q[q_off + 2],
                ));
                let next = current.mul(&Quat::exp(&(omega * dt))).normalize();
                let log = next.log();
                state.q[q_off] = log.x;
                state.q[q_off + 1] = log.y;
                state.q[q_off + 2] = log.z;
            }
            JointType::Free => {
                let omega = Vec3::new(state.v[v_off], state.v[v_off + 1], state.v[v_off + 2]);
                let lin = Vec3::new(state.v[v_off + 3], state.v[v_off + 4], state.v[v_off + 5]);

                let current = Quat::exp(&Vec3::new(
                    state.q[q_off + 3],
                    state.q[q_off + 4],
                    state.q[q_off + 5],
                ));
                // Body-frame linear velocity → world displacement.
                let world_lin = current.rotate(lin);
                state.q[q_off] += dt * world_lin.x;
                state.q[q_off + 1] += dt * world_lin.y;
                state.q[q_off + 2] += dt * world_lin.z;

                let next = current.mul(&Quat::exp(&(omega * dt))).normalize();
                let log = next.log();
                state.q[q_off + 3] = log.x;
                state.q[q_off + 4] = log.y;
                state.q[q_off + 5] = log.z;
            }
        }
    }

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
