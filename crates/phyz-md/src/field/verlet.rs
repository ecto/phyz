//! Velocity-Verlet time integration with an optional Berendsen thermostat.
//!
//! Positions are in Å, velocities in Å/fs, forces in eV/Å, masses in amu; the
//! [`super::units::FORCE_TO_ACCEL`] factor keeps `a = f·factor/m` consistent so
//! energy is conserved in NVE.

use super::cell::vec3;
use super::units::{FORCE_TO_ACCEL, KB_EV_PER_K};

/// Berendsen thermostat configuration for constant-temperature (NVT) runs.
#[derive(Debug, Clone, Copy)]
pub struct Berendsen {
    /// Target temperature in K.
    pub target_k: f64,
    /// Coupling time constant in fs (larger = weaker coupling).
    pub tau_fs: f64,
}

/// Kinetic energy in eV for the given velocities (Å/fs) and masses (amu).
pub fn kinetic_energy(velocities: &[[f64; 3]], masses: &[f64]) -> f64 {
    let mut ke = 0.0;
    for (v, &m) in velocities.iter().zip(masses) {
        ke += 0.5 * m * vec3::norm2(*v) / FORCE_TO_ACCEL;
    }
    ke
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

fn apply_berendsen(dt: f64, t: Berendsen, velocities: &mut [[f64; 3]], masses: &[f64]) {
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
}
