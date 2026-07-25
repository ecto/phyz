//! Velocity-Verlet time integration with thermostats and a barostat.
//!
//! Positions are in Å, velocities in Å/fs, forces in eV/Å, masses in amu; the
//! [`super::units::FORCE_TO_ACCEL`] factor keeps `a = f·factor/m` consistent so
//! energy is conserved in NVE.
//!
//! Three ensembles are available:
//!
//! - **NVE** — plain velocity Verlet, no coupling.
//! - **NVT** — [`Berendsen`] velocity rescaling (fast equilibration, but it
//!   does not sample the canonical distribution) or [`NoseHoover`] (an extended
//!   Lagrangian that does).
//! - **NPT** — a [`NoseHoover`] or [`Berendsen`] thermostat plus a
//!   [`Barostat`], which rescales the cell and the coordinates in it.

use super::cell::{Lattice, vec3};
use super::units::{FORCE_TO_ACCEL, KB_EV_PER_K};

/// Berendsen thermostat configuration for constant-temperature (NVT) runs.
///
/// Simple and robust for equilibration. It suppresses temperature fluctuations
/// rather than reproducing them, so it does not sample the canonical ensemble —
/// use [`NoseHoover`] for production NVT averages.
#[derive(Debug, Clone, Copy)]
pub struct Berendsen {
    /// Target temperature in K.
    pub target_k: f64,
    /// Coupling time constant in fs (larger = weaker coupling).
    pub tau_fs: f64,
}

/// Nosé-Hoover thermostat: an extended-system friction variable that does
/// generate the canonical distribution.
///
/// The equations of motion gain `-ζ v` on every atom, with
///
/// ```text
/// dζ/dt = (2K - N_f k_B T₀) / Q,   Q = N_f k_B T₀ τ²
/// ```
///
/// [`NoseHoover::conserved_offset`] returns the extra terms that make the
/// *extended* energy conserved — the physical `KE + PE` is not, by design.
#[derive(Debug, Clone, Copy)]
pub struct NoseHoover {
    /// Target temperature in K.
    pub target_k: f64,
    /// Coupling time constant in fs.
    pub tau_fs: f64,
    /// Friction coefficient ζ (1/fs). Integrated by the stepper.
    pub zeta: f64,
    /// Accumulated thermostat "position" `∫ζ dt`, needed for the conserved
    /// quantity.
    pub eta: f64,
    /// Degrees of freedom. Set to `3N` (or `3N - 3` with COM motion removed).
    pub dof: f64,
}

impl NoseHoover {
    /// A thermostat at `target_k` with coupling time `tau_fs` for `dof`
    /// degrees of freedom.
    pub fn new(target_k: f64, tau_fs: f64, dof: f64) -> Self {
        Self {
            target_k,
            tau_fs,
            zeta: 0.0,
            eta: 0.0,
            dof: dof.max(1.0),
        }
    }

    /// Thermostat mass `Q = N_f k_B T₀ τ²`, in eV·fs².
    #[inline]
    pub fn mass(&self) -> f64 {
        (self.dof * KB_EV_PER_K * self.target_k * self.tau_fs * self.tau_fs).max(1e-30)
    }

    /// The thermostat's contribution to the conserved quantity, in eV:
    /// `½ Q ζ² + N_f k_B T₀ η`. Adding this to `KE + PE` gives a quantity that
    /// is conserved to integrator accuracy.
    #[inline]
    pub fn conserved_offset(&self) -> f64 {
        0.5 * self.mass() * self.zeta * self.zeta
            + self.dof * KB_EV_PER_K * self.target_k * self.eta
    }
}

/// Berendsen barostat for constant-pressure runs.
///
/// Each step the cell and all coordinates are scaled by
/// `μ = [1 − (dt/τ_p) κ (P₀ − P)]^{1/3}`, which relaxes the pressure toward
/// `target` with time constant `τ_p`. Like the Berendsen thermostat it does not
/// sample a rigorous NPT distribution, but it is the standard workhorse for
/// equilibrating a box to a target density.
#[derive(Debug, Clone, Copy)]
pub struct Barostat {
    /// Target pressure in eV/Å³.
    pub target: f64,
    /// Coupling time constant in fs.
    pub tau_fs: f64,
    /// Isothermal compressibility in Å³/eV. Water is ≈ 0.0073 Å³/eV
    /// (4.5e-5 /bar); the value only sets the coupling strength.
    pub compressibility: f64,
    /// Largest fractional length change permitted in one step, as a guard
    /// against a bad pressure estimate collapsing the box.
    pub max_scale_step: f64,
}

impl Barostat {
    /// A barostat at `target` eV/Å³ with coupling time `tau_fs`.
    pub fn new(target: f64, tau_fs: f64, compressibility: f64) -> Self {
        Self {
            target,
            tau_fs,
            compressibility,
            max_scale_step: 0.01,
        }
    }

    /// The isotropic length scale factor for the current pressure.
    pub fn scale_factor(&self, dt: f64, current_pressure: f64) -> f64 {
        let mu3 =
            1.0 - (dt / self.tau_fs) * self.compressibility * (self.target - current_pressure);
        let mu = mu3.max(1e-6).cbrt();
        mu.clamp(1.0 - self.max_scale_step, 1.0 + self.max_scale_step)
    }

    /// Apply the scaling to the cell and to every position in it.
    pub fn apply(&self, mu: f64, positions: &mut [[f64; 3]], cell: &mut Lattice) {
        cell.a = vec3::scale(cell.a, mu);
        cell.b = vec3::scale(cell.b, mu);
        cell.c = vec3::scale(cell.c, mu);
        for p in positions.iter_mut() {
            *p = vec3::scale(*p, mu);
        }
    }
}

/// Kinetic energy in eV for the given velocities (Å/fs) and masses (amu).
pub fn kinetic_energy(velocities: &[[f64; 3]], masses: &[f64]) -> f64 {
    let mut ke = 0.0;
    for (v, &m) in velocities.iter().zip(masses) {
        ke += 0.5 * m * vec3::norm2(*v) / FORCE_TO_ACCEL;
    }
    ke
}

/// Instantaneous temperature in K from the kinetic energy and the number of
/// degrees of freedom.
pub fn temperature(velocities: &[[f64; 3]], masses: &[f64], dof: f64) -> f64 {
    if dof <= 0.0 {
        return 0.0;
    }
    2.0 * kinetic_energy(velocities, masses) / (dof * KB_EV_PER_K)
}

/// Advance one velocity-Verlet step.
///
/// `forces` and `potential` carry the force cache across steps (one force
/// evaluation per step, as velocity-Verlet requires): on entry they must hold
/// the forces/energy at the current positions, and on return they hold the
/// values at the new positions. `eval` computes `(energy, forces)` for a given
/// position array; it may capture and update the caller's own system state.
#[allow(clippy::too_many_arguments)]
pub fn verlet_step<F>(
    dt: f64,
    thermostat: Option<Berendsen>,
    positions: &mut [[f64; 3]],
    velocities: &mut [[f64; 3]],
    masses: &[f64],
    forces: &mut Vec<[f64; 3]>,
    potential: &mut f64,
    mut eval: F,
) where
    F: FnMut(&[[f64; 3]]) -> (f64, Vec<[f64; 3]>),
{
    let n = positions.len();
    // v(t+dt/2) = v(t) + 0.5 dt a(t); x(t+dt) = x(t) + dt v(t+dt/2)
    for i in 0..n {
        let inv_m = FORCE_TO_ACCEL / masses[i];
        let a = vec3::scale(forces[i], inv_m);
        vec3::add_assign(&mut velocities[i], vec3::scale(a, 0.5 * dt));
        let dx = vec3::scale(velocities[i], dt);
        vec3::add_assign(&mut positions[i], dx);
    }
    // Recompute forces at new positions.
    let (pot, new_forces) = eval(positions);
    *forces = new_forces;
    *potential = pot;
    // v(t+dt) = v(t+dt/2) + 0.5 dt a(t+dt)
    for i in 0..n {
        let inv_m = FORCE_TO_ACCEL / masses[i];
        let a = vec3::scale(forces[i], inv_m);
        vec3::add_assign(&mut velocities[i], vec3::scale(a, 0.5 * dt));
    }
    // Berendsen velocity rescale toward target temperature.
    if let Some(t) = thermostat {
        apply_berendsen(dt, t, velocities, masses);
    }
}

/// Berendsen velocity rescaling toward the target temperature.
pub fn apply_berendsen(dt: f64, t: Berendsen, velocities: &mut [[f64; 3]], masses: &[f64]) {
    let n = velocities.len();
    if n == 0 {
        return;
    }
    let dof = 3.0 * n as f64;
    let ke = kinetic_energy(velocities, masses);
    let cur_t = 2.0 * ke / (dof * KB_EV_PER_K);
    if cur_t <= 1e-12 {
        return;
    }
    // lambda = sqrt(1 + dt/tau (T0/T - 1))
    let ratio = t.target_k / cur_t;
    let lambda2 = 1.0 + (dt / t.tau_fs) * (ratio - 1.0);
    let lambda = lambda2.max(0.0).sqrt();
    for v in velocities.iter_mut() {
        *v = vec3::scale(*v, lambda);
    }
}

/// Advance the Nosé-Hoover friction variable by a half step and apply the
/// corresponding velocity scaling.
///
/// Called on both halves of a velocity-Verlet step, this gives the standard
/// second-order-accurate, time-reversible NVT integrator.
pub fn nose_hoover_half_step(
    dt: f64,
    nh: &mut NoseHoover,
    velocities: &mut [[f64; 3]],
    masses: &[f64],
) {
    let target = nh.dof * KB_EV_PER_K * nh.target_k;
    let q = nh.mass();

    // Symmetric (Trotter-factorized) update: quarter-step on ζ, half-step
    // velocity scaling, quarter-step on ζ again with the *new* kinetic energy.
    // The symmetry is what makes the step time-reversible, and reversibility is
    // what keeps the extended energy from drifting.
    let ke = kinetic_energy(velocities, masses);
    nh.zeta += 0.25 * dt * (2.0 * ke - target) / q;

    let scale = (-nh.zeta * 0.5 * dt).exp();
    for v in velocities.iter_mut() {
        *v = vec3::scale(*v, scale);
    }
    nh.eta += 0.5 * dt * nh.zeta;

    let ke = kinetic_energy(velocities, masses);
    nh.zeta += 0.25 * dt * (2.0 * ke - target) / q;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Harmonic oscillator: one particle on a spring toward the origin.
    fn spring_eval(k: f64) -> impl FnMut(&[[f64; 3]]) -> (f64, Vec<[f64; 3]>) {
        move |pos| {
            let x = pos[0];
            let e = 0.5 * k * vec3::norm2(x);
            (e, vec![vec3::scale(x, -k)])
        }
    }

    #[test]
    fn nve_conserves_energy_for_harmonic_oscillator() {
        let k = 1.0;
        let mut eval = spring_eval(k);
        let mut positions = vec![[1.0, 0.0, 0.0]];
        let mut velocities = vec![[0.0; 3]];
        let masses = [10.0];
        let (mut potential, mut forces) = eval(&positions);
        let e0 = potential + kinetic_energy(&velocities, &masses);
        for _ in 0..1000 {
            verlet_step(
                0.1,
                None,
                &mut positions,
                &mut velocities,
                &masses,
                &mut forces,
                &mut potential,
                &mut eval,
            );
        }
        let e1 = potential + kinetic_energy(&velocities, &masses);
        assert!((e1 - e0).abs() < 1e-3 * e0.abs().max(1.0));
    }

    #[test]
    fn nose_hoover_drives_a_hot_system_to_the_target_temperature() {
        // 64 non-interacting particles started far too hot; the thermostat has
        // to pull them down and hold them.
        let n = 64;
        let masses = vec![40.0; n];
        let mut velocities: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let t = i as f64 + 1.0;
                [
                    0.01 * (t * 0.7).sin(),
                    0.01 * (t * 1.1).cos(),
                    0.01 * (t * 1.7).sin(),
                ]
            })
            .collect();
        let dof = 3.0 * n as f64;
        let target = 300.0;
        let mut nh = NoseHoover::new(target, 100.0, dof);

        let t_start = temperature(&velocities, &masses, dof);
        assert!(t_start > target * 2.0, "test needs a hot start: {t_start}");

        // Free particles: the only dynamics is the thermostat coupling.
        let mut avg = 0.0;
        let steps = 40_000;
        for s in 0..steps {
            nose_hoover_half_step(1.0, &mut nh, &mut velocities, &masses);
            nose_hoover_half_step(1.0, &mut nh, &mut velocities, &masses);
            if s > steps / 2 {
                avg += temperature(&velocities, &masses, dof);
            }
        }
        avg /= (steps - steps / 2 - 1) as f64;
        assert!(
            (avg - target).abs() / target < 0.15,
            "time-averaged T = {avg} K, target {target} K"
        );
    }

    #[test]
    fn nose_hoover_conserves_the_extended_energy() {
        // KE + PE alone is not conserved under Nosé-Hoover; KE + PE + the
        // thermostat offset is.
        let k = 1.0;
        let mut eval = spring_eval(k);
        let mut positions = vec![[1.0, 0.0, 0.0]];
        let mut velocities = vec![[0.02, 0.0, 0.0]];
        let masses = [10.0];
        let (mut potential, mut forces) = eval(&positions);
        let mut nh = NoseHoover::new(300.0, 50.0, 3.0);

        let conserved = |pot: f64, v: &[[f64; 3]], nh: &NoseHoover| {
            pot + kinetic_energy(v, &masses) + nh.conserved_offset()
        };
        let e0 = conserved(potential, &velocities, &nh);

        let dt = 0.05;
        for _ in 0..20_000 {
            nose_hoover_half_step(dt, &mut nh, &mut velocities, &masses);
            verlet_step(
                dt,
                None,
                &mut positions,
                &mut velocities,
                &masses,
                &mut forces,
                &mut potential,
                &mut eval,
            );
            nose_hoover_half_step(dt, &mut nh, &mut velocities, &masses);
        }
        let e1 = conserved(potential, &velocities, &nh);
        assert!(
            (e1 - e0).abs() < 1e-3 * e0.abs().max(1e-3),
            "extended energy drifted: {e0} → {e1}"
        );
    }

    #[test]
    fn barostat_scales_toward_the_target_pressure() {
        let b = Barostat::new(0.0, 1000.0, 0.01);
        // Pressure above target → the box should expand (μ > 1).
        assert!(b.scale_factor(1.0, 0.5) > 1.0);
        // Pressure below target → contract.
        assert!(b.scale_factor(1.0, -0.5) < 1.0);
        // At target, nothing moves.
        assert!((b.scale_factor(1.0, 0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn barostat_scaling_preserves_fractional_coordinates() {
        let b = Barostat::new(0.0, 1000.0, 0.01);
        let mut cell = Lattice::cubic(10.0);
        let mut pos = vec![[1.0, 2.0, 3.0], [7.5, 0.5, 9.0]];
        let before: Vec<[f64; 3]> = pos.iter().map(|p| cell.to_fractional(*p)).collect();
        b.apply(1.005, &mut pos, &mut cell);
        assert!((cell.volume() - 1000.0 * 1.005f64.powi(3)).abs() < 1e-9);
        for (p, f0) in pos.iter().zip(&before) {
            let f1 = cell.to_fractional(*p);
            for k in 0..3 {
                assert!((f1[k] - f0[k]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn barostat_step_is_clamped() {
        let b = Barostat::new(0.0, 1.0, 1e6);
        // An absurd pressure must not be able to collapse or explode the box
        // in a single step.
        let mu = b.scale_factor(1.0, 1e6);
        assert!(mu <= 1.0 + b.max_scale_step + 1e-12);
        let mu = b.scale_factor(1.0, -1e6);
        assert!(mu >= 1.0 - b.max_scale_step - 1e-12);
    }
}
