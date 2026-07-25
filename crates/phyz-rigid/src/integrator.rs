//! Time integrators for the forward dynamics.
//!
//! The state is second-order: `q̈ = ABA(q, q̇, u)`. Both integrators below
//! advance `(q, v)` by `model.dt`, leaving `state.time` and the cached
//! `body_xform` consistent with the new configuration.

use crate::aba::aba;
use crate::kinematics::forward_kinematics;
use phyz_math::DVec;
use phyz_model::{Model, State};

/// Which integration rule a [`Simulator`] applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Integrator {
    /// Semi-implicit (symplectic) Euler: `v' = v + dt·a`, `q' = q + dt·v'`.
    ///
    /// First-order accurate but symplectic, so energy error stays bounded
    /// rather than drifting secularly. This is the rule the differentiable
    /// rollout in `phyz-diff` uses, and the default here so that
    /// `Simulator::default()` matches what gradients are taken through.
    #[default]
    SemiImplicitEuler,
    /// Classical fourth-order Runge–Kutta.
    ///
    /// Fourth-order accurate and much tighter on energy over a short horizon,
    /// at four dynamics evaluations per step. Not symplectic: over very long
    /// horizons its energy error grows secularly where Euler's does not.
    Rk4,
}

/// Advances a [`State`] under a [`Model`]'s dynamics.
#[derive(Debug, Clone, Copy, Default)]
pub struct Simulator {
    /// The integration rule.
    pub integrator: Integrator,
}

impl Simulator {
    /// A simulator using semi-implicit Euler.
    pub fn semi_implicit_euler() -> Self {
        Self {
            integrator: Integrator::SemiImplicitEuler,
        }
    }

    /// A simulator using classical RK4.
    pub fn rk4() -> Self {
        Self {
            integrator: Integrator::Rk4,
        }
    }

    /// Advance `state` by one `model.dt`.
    pub fn step(&self, model: &Model, state: &mut State) {
        match self.integrator {
            Integrator::SemiImplicitEuler => step_semi_implicit_euler(model, state),
            Integrator::Rk4 => step_rk4(model, state),
        }
        state.time += model.dt;
        forward_kinematics(model, state);
    }

    /// Advance `state` by `steps` timesteps.
    pub fn simulate(&self, model: &Model, state: &mut State, steps: usize) {
        for _ in 0..steps {
            self.step(model, state);
        }
    }
}

/// Evaluate `q̈` at an arbitrary `(q, v)`, reusing `state`'s controls.
///
/// `aba` reads `q`, `v`, `ctrl` and `qfrc_external`; the RK4 stages need it at
/// off-state points, so we swap them in on a scratch clone rather than
/// mutating the caller's state.
fn accel_at(model: &Model, state: &State, q: &DVec, v: &DVec) -> DVec {
    let mut probe = state.clone();
    probe.q = q.clone();
    probe.v = v.clone();
    forward_kinematics(model, &mut probe);
    aba(model, &probe)
}

fn step_semi_implicit_euler(model: &Model, state: &mut State) {
    let dt = model.dt;
    let a = aba(model, state);
    state.v += &(&a * dt);
    let dq = &state.v * dt;
    state.q += &dq;
}

fn step_rk4(model: &Model, state: &mut State) {
    let dt = model.dt;
    let (q0, v0) = (state.q.clone(), state.v.clone());
    let half = dt * 0.5;

    // y = (q, v), ẏ = (v, a(q, v)).
    let k1_q = v0.clone();
    let k1_v = accel_at(model, state, &q0, &v0);

    let k2_q = &v0 + &(&k1_v * half);
    let k2_v = accel_at(model, state, &(&q0 + &(&k1_q * half)), &k2_q);

    let k3_q = &v0 + &(&k2_v * half);
    let k3_v = accel_at(model, state, &(&q0 + &(&k2_q * half)), &k3_q);

    let k4_q = &v0 + &(&k3_v * dt);
    let k4_v = accel_at(model, state, &(&q0 + &(&k3_q * dt)), &k4_q);

    // Weighted sum (k1 + 2k2 + 2k3 + k4)·dt/6, accumulated in place.
    let sixth = dt / 6.0;
    let mut dq = k1_q.clone();
    dq.axpy(2.0, &k2_q);
    dq.axpy(2.0, &k3_q);
    dq.axpy(1.0, &k4_q);
    dq.scale(sixth);

    let mut dv = k1_v.clone();
    dv.axpy(2.0, &k2_v);
    dv.axpy(2.0, &k3_v);
    dv.axpy(1.0, &k4_v);
    dv.scale(sixth);

    state.q = &q0 + &dq;
    state.v = &v0 + &dv;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::total_energy;
    use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
    use phyz_model::ModelBuilder;

    /// Compound pendulum: revolute about +Z, gravity along -Y, rod of mass 1kg
    /// and length 1m hanging from the pivot.
    fn pendulum(dt: f64) -> Model {
        let (mass, length) = (1.0, 1.0);
        let i_com = mass * length * length / 12.0;
        ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(dt)
            .add_revolute_body(
                "pendulum",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(
                    mass,
                    Vec3::new(0.0, -length / 2.0, 0.0),
                    Mat3::new(i_com, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_com),
                ),
            )
            .build()
    }

    /// Small-angle period of a compound pendulum:
    /// `T = 2π·sqrt(I_pivot / (m·g·d))`, `I_pivot = mL²/3`, `d = L/2`.
    #[test]
    fn rk4_recovers_small_angle_pendulum_period() {
        let dt = 1e-4;
        let model = pendulum(dt);
        let mut state = model.default_state();
        state.q[0] = 0.05;

        let expected = 2.0 * std::f64::consts::PI * ((1.0 / 3.0) / (GRAVITY * 0.5)).sqrt();

        let sim = Simulator::rk4();
        let mut prev = state.q[0];
        let mut crossings = Vec::new();
        for i in 0..(10.0 / dt) as usize {
            sim.step(&model, &mut state);
            let q = state.q[0];
            // Upward zero crossing, linearly interpolated for sub-step timing.
            if prev < 0.0 && q >= 0.0 {
                let frac = -prev / (q - prev);
                crossings.push((i as f64 + frac) * dt);
            }
            prev = q;
        }

        assert!(crossings.len() >= 3, "expected several periods in 10 s");
        let measured =
            (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64;
        assert!(
            (measured - expected).abs() / expected < 1e-3,
            "period {measured} vs expected {expected}"
        );
    }

    /// RK4 on a conservative system should hold energy far tighter than the
    /// integrator's local error suggests over a short horizon.
    #[test]
    fn rk4_conserves_energy_on_pendulum() {
        let dt = 1e-3;
        let model = pendulum(dt);
        let mut state = model.default_state();
        state.q[0] = 0.5;

        let e0 = total_energy(&model, &state);
        Simulator::rk4().simulate(&model, &mut state, 5000);
        let e1 = total_energy(&model, &state);

        assert!(
            (e1 - e0).abs() / e0.abs().max(1.0) < 1e-6,
            "energy drift {e0} -> {e1}"
        );
    }

    /// Semi-implicit Euler is only first-order, but symplectic: its energy
    /// error must stay bounded rather than growing without limit.
    #[test]
    fn semi_implicit_euler_energy_stays_bounded() {
        let dt = 1e-3;
        let model = pendulum(dt);
        let mut state = model.default_state();
        state.q[0] = 0.5;

        let e0 = total_energy(&model, &state);
        let sim = Simulator::semi_implicit_euler();
        let mut worst: f64 = 0.0;
        for _ in 0..20_000 {
            sim.step(&model, &mut state);
            worst = worst.max((total_energy(&model, &state) - e0).abs());
        }
        assert!(worst / e0.abs().max(1.0) < 1e-2, "energy excursion {worst}");
    }
}
