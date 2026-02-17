//! Multi-rate integration using r-RESPA (Reference System Propagator Algorithm).
//!
//! Allows different forces to be evaluated at different timesteps:
//! - Slow forces (e.g., gravity, soft constraints): large dt_outer
//! - Fast forces (e.g., stiff springs, high-frequency vibrations): small dt_inner

use phyz_math::DVec;
use phyz_model::{Model, State};

/// Force splitting function type.
///
/// Takes (model, state) and returns force vector.
pub type ForceFn = dyn Fn(&Model, &State) -> DVec;

/// r-RESPA integrator for multi-rate time integration.
///
/// Splits forces into "slow" and "fast" components and integrates them
/// at different rates. Fast forces are evaluated dt_outer/dt_inner times
/// per outer step.
pub struct RRespaIntegrator {
    /// Inner timestep for fast forces
    pub dt_inner: f64,
    /// Outer timestep for slow forces
    pub dt_outer: f64,
}

impl RRespaIntegrator {
    /// Create a new r-RESPA integrator.
    ///
    /// # Arguments
    /// - `dt_inner`: Small timestep for fast forces
    /// - `dt_outer`: Large timestep for slow forces (must be integer multiple of dt_inner)
    pub fn new(dt_inner: f64, dt_outer: f64) -> Self {
        assert!(dt_outer >= dt_inner, "dt_outer must be >= dt_inner");
        assert!(
            (dt_outer / dt_inner).fract() < 1e-10,
            "dt_outer must be integer multiple of dt_inner"
        );
        Self { dt_inner, dt_outer }
    }

    /// Perform one r-RESPA step with force splitting.
    ///
    /// # Arguments
    /// - `model`: Physics model
    /// - `state`: Simulation state (modified in place)
    /// - `slow_forces`: Function computing slow (low-frequency) forces
    /// - `fast_forces`: Function computing fast (high-frequency) forces
    pub fn step(
        &self,
        model: &Model,
        state: &mut State,
        slow_forces: &ForceFn,
        fast_forces: &ForceFn,
    ) {
        let n_inner = (self.dt_outer / self.dt_inner).round() as usize;

        // Outer half-step: slow forces
        let f_slow = slow_forces(model, state);
        let acc_slow = self.forces_to_acceleration(model, &f_slow);
        state.v += &acc_slow * (self.dt_outer / 2.0);

        // Inner steps: fast forces
        for _ in 0..n_inner {
            let f_fast = fast_forces(model, state);
            let acc_fast = self.forces_to_acceleration(model, &f_fast);

            state.v += &acc_fast * self.dt_inner;
            state.q += &state.v * self.dt_inner;
        }

        // Outer half-step: slow forces
        let f_slow = slow_forces(model, state);
        let acc_slow = self.forces_to_acceleration(model, &f_slow);
        state.v += &acc_slow * (self.dt_outer / 2.0);

        state.time += self.dt_outer;
    }

    /// Convert generalized forces to accelerations.
    ///
    /// For rigid body dynamics, this requires solving M(q) * qdd = forces,
    /// but for simplicity we approximate with identity mass matrix here.
    /// In practice, you'd use ABA or invert the mass matrix.
    fn forces_to_acceleration(&self, _model: &Model, forces: &DVec) -> DVec {
        // Simplified: assume unit mass
        // In a full implementation, would compute M^{-1} * forces
        forces.clone()
    }
}

/// Split gravitational forces from other forces for r-RESPA.
pub fn split_forces_gravity(model: &Model, state: &State) -> (DVec, DVec) {
    use phyz_rigid::{aba, forward_kinematics};

    // Slow forces: gravity only
    let (xforms, _) = forward_kinematics(model, state);
    let mut gravity_forces = DVec::zeros(model.nv);

    for (i, body) in model.bodies.iter().enumerate() {
        let mass = body.inertia.mass;
        let com_local = body.inertia.com;
        let xf = &xforms[i];
        let _com_world = xf.rot.transpose() * com_local + xf.pos;

        // Gravitational force at CoM
        let f_gravity = model.gravity * mass;

        // This is simplified; proper implementation would compute
        // generalized forces via Jacobian transpose
        // For now, approximate as direct force application
        if i < model.nv {
            gravity_forces[i] = f_gravity.norm();
        }
    }

    // Fast forces: everything else (constraints, applied forces, etc.)
    let total_acc = aba(model, state);
    let total_forces = &total_acc * model.dt; // Approximate

    let fast_forces = &total_forces - &gravity_forces;

    (gravity_forces, fast_forces)
}

/// Example: Split stiff spring forces from soft forces.
///
/// This is a template function showing how to split forces.
pub fn split_forces_stiffness(
    _model: &Model,
    _state: &State,
    _stiffness_threshold: f64,
) -> (Box<ForceFn>, Box<ForceFn>) {
    // Slow forces: soft springs (k < threshold)
    let slow: Box<ForceFn> = Box::new(move |_m, _s| {
        // Compute soft forces here
        DVec::zeros(_m.nv)
    });

    // Fast forces: stiff springs (k >= threshold)
    let fast: Box<ForceFn> = Box::new(move |_m, _s| {
        // Compute stiff forces here
        DVec::zeros(_m.nv)
    });

    (slow, fast)
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
    use phyz_model::ModelBuilder;

    #[test]
    fn test_rrespa_timestep_ratio() {
        let integrator = RRespaIntegrator::new(0.001, 0.01);
        assert_eq!((integrator.dt_outer / integrator.dt_inner).round(), 10.0);
    }

    #[test]
    #[should_panic(expected = "dt_outer must be >= dt_inner")]
    fn test_rrespa_invalid_dt() {
        RRespaIntegrator::new(0.01, 0.001);
    }

    #[test]
    fn test_rrespa_step() {
        let model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, -GRAVITY, 0.0))
            .dt(0.01)
            .add_revolute_body(
                "pendulum",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::new(
                    1.0,
                    Vec3::new(0.0, -0.5, 0.0),
                    Mat3::from_diagonal(&Vec3::new(0.083, 0.0, 0.083)),
                ),
            )
            .build();

        let mut state = model.default_state();
        state.q[0] = 0.1;

        let integrator = RRespaIntegrator::new(0.001, 0.01);

        // Define force splitting (simplified)
        let slow_forces: Box<ForceFn> = Box::new(|m, _s| {
            // Gravity forces
            let mut f = DVec::zeros(m.nv);
            f[0] = -m.gravity.norm() * 0.5; // Simplified
            f
        });

        let fast_forces: Box<ForceFn> = Box::new(|m, _s| {
            // No fast forces in this simple example
            DVec::zeros(m.nv)
        });

        // Step should not crash
        integrator.step(
            &model,
            &mut state,
            slow_forces.as_ref(),
            fast_forces.as_ref(),
        );

        // State should have advanced
        assert!(state.time > 0.0);
    }
}
