//! Rigid-body dynamics (`phyz-rigid`) validation against closed-form results.
//!
//! 1. **Pendulum period** — the finite-amplitude period of a rigid rod against
//!    the exact elliptic-integral solution `T = (4/ω₀)K(sin(θ₀/2))`, at
//!    amplitudes where the small-angle approximation is wrong by tens of percent.
//! 2. **Energy conservation** — `phyz_rigid::total_energy` along the trajectory.
//! 3. **Spinning-top precession** — the mean precession rate of a fast symmetric
//!    top against `Ω = mgl/(I₃ω₃)`, including the `ω₃⁻²` decay of the residual.

use crate::report::{Convergence, ErrorKind, Suite, Validation};
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Joint, JointType, Model, ModelBuilder, State};
use phyz_rigid::{aba, total_energy};

const CRATE: &str = "phyz-rigid";
const G: f64 = 9.80665;

/// Complete elliptic integral of the first kind `K(k)` via the AGM.
fn ellip_k(k: f64) -> f64 {
    let mut a = 1.0_f64;
    let mut b = (1.0 - k * k).sqrt();
    for _ in 0..60 {
        if (a - b).abs() <= 1e-16 * a {
            break;
        }
        let a1 = 0.5 * (a + b);
        b = (a * b).sqrt();
        a = a1;
    }
    std::f64::consts::PI / (2.0 * a)
}

/// One classical Runge–Kutta 4 step on `(q, v)` with `v̇ = ABA(q, v)`.
fn rk4_step(model: &Model, state: &mut State, dt: f64) {
    let nq = state.q.len();
    let eval = |q: &[f64], v: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let mut s = state.clone();
        for i in 0..nq {
            s.q[i] = q[i];
            s.v[i] = v[i];
        }
        let a = aba(model, &s);
        (v.to_vec(), (0..nq).map(|i| a[i]).collect())
    };

    let q0: Vec<f64> = (0..nq).map(|i| state.q[i]).collect();
    let v0: Vec<f64> = (0..nq).map(|i| state.v[i]).collect();

    let (k1q, k1v) = eval(&q0, &v0);
    let add = |a: &[f64], b: &[f64], s: f64| -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x + s * y).collect()
    };
    let (k2q, k2v) = eval(&add(&q0, &k1q, dt / 2.0), &add(&v0, &k1v, dt / 2.0));
    let (k3q, k3v) = eval(&add(&q0, &k2q, dt / 2.0), &add(&v0, &k2v, dt / 2.0));
    let (k4q, k4v) = eval(&add(&q0, &k3q, dt), &add(&v0, &k3v, dt));

    for i in 0..nq {
        state.q[i] = q0[i] + dt / 6.0 * (k1q[i] + 2.0 * k2q[i] + 2.0 * k3q[i] + k4q[i]);
        state.v[i] = v0[i] + dt / 6.0 * (k1v[i] + 2.0 * k2v[i] + 2.0 * k3v[i] + k4v[i]);
    }
    state.time += dt;
}

/// Uniform rod of length `l` and mass `m` hinged at one end, swinging in the x–y
/// plane under gravity along `−y`. `q = 0` is hanging straight down.
fn rod_pendulum(m: f64, l: f64) -> Model {
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, -G, 0.0))
        .add_revolute_body(
            "rod",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                m,
                Vec3::new(0.0, -l / 2.0, 0.0),
                Mat3::from_diagonal(&Vec3::new(m * l * l / 12.0, 0.0, m * l * l / 12.0)),
            ),
        )
        .build()
}

/// Root of the cubic Hermite interpolant of `q` on one step, in units of `dt`.
///
/// Uses both the value and the derivative at each endpoint, so the crossing time
/// is fourth-order accurate and does not cap the integrator's own convergence
/// order (linear interpolation would make the measurement O(Δt²)).
fn hermite_root(q0: f64, v0: f64, q1: f64, v1: f64, dt: f64) -> f64 {
    let h = |s: f64| {
        let (s2, s3) = (s * s, s * s * s);
        q0 * (2.0 * s3 - 3.0 * s2 + 1.0)
            + dt * v0 * (s3 - 2.0 * s2 + s)
            + q1 * (-2.0 * s3 + 3.0 * s2)
            + dt * v1 * (s3 - s2)
    };
    let dh = |s: f64| {
        let s2 = s * s;
        q0 * (6.0 * s2 - 6.0 * s)
            + dt * v0 * (3.0 * s2 - 4.0 * s + 1.0)
            + q1 * (-6.0 * s2 + 6.0 * s)
            + dt * v1 * (3.0 * s2 - 2.0 * s)
    };
    let mut s = q0 / (q0 - q1);
    for _ in 0..12 {
        let d = dh(s);
        if d.abs() < 1e-300 {
            break;
        }
        let step = h(s) / d;
        s -= step;
        if step.abs() < 1e-15 {
            break;
        }
    }
    s.clamp(0.0, 1.0)
}

/// Measure the full period by timing zero crossings of `q[0]`, released from
/// rest at `theta0`.
fn pendulum_period(model: &Model, theta0: f64, dt: f64) -> f64 {
    let mut state = model.default_state();
    state.q[0] = theta0;

    let mut crossings = Vec::new();
    let mut prev_q = state.q[0];
    let mut prev_v = state.v[0];
    let mut prev_t = 0.0;
    let max_steps = 2_000_000;
    for _ in 0..max_steps {
        rk4_step(model, &mut state, dt);
        let (q, v) = (state.q[0], state.v[0]);
        if prev_q.signum() != q.signum() && prev_q != 0.0 {
            crossings.push(prev_t + hermite_root(prev_q, prev_v, q, v, dt) * dt);
            if crossings.len() == 3 {
                break;
            }
        }
        prev_q = q;
        prev_v = v;
        prev_t = state.time;
    }
    assert!(crossings.len() >= 3, "pendulum did not complete a period");
    // Crossings 1 and 3 are one full period apart.
    crossings[2] - crossings[0]
}

/// Peak relative drift of `total_energy` over `steps` RK4 steps.
fn pendulum_energy_drift(model: &Model, theta0: f64, dt: f64, steps: usize) -> f64 {
    let mut state = model.default_state();
    state.q[0] = theta0;
    let e0 = total_energy(model, &state);
    // Reference the drift to the swing's kinetic-energy scale, not to a
    // potential-energy zero that is arbitrary.
    let scale = {
        let mut bottom = model.default_state();
        bottom.q[0] = 0.0;
        (e0 - total_energy(model, &bottom)).abs().max(1e-12)
    };
    let mut worst: f64 = 0.0;
    for _ in 0..steps {
        rk4_step(model, &mut state, dt);
        worst = worst.max((total_energy(model, &state) - e0).abs() / scale);
    }
    worst
}

/// Symmetric top on a fixed pivot, modelled as a z–x–z gimbal of three revolute
/// joints so the generalized coordinates are the Euler angles `(ψ, θ, φ)`.
fn spinning_top(m: f64, l: f64, i1_pivot: f64, i3: f64) -> Model {
    let ghost = SpatialInertia::new(
        1e-12,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(1e-12, 1e-12, 1e-12)),
    );
    let hinge = |axis: Vec3| Joint {
        joint_type: JointType::Revolute,
        parent_to_joint: SpatialTransform::identity(),
        axis,
        damping: 0.0,
        limits: None,
    };
    // I₁ is quoted about the pivot; the body frame carries the inertia about
    // its own centre of mass, which sits a distance l up the symmetry axis.
    let i1_com = i1_pivot - m * l * l;
    ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -G))
        .add_body("precess", -1, hinge(Vec3::new(0.0, 0.0, 1.0)), ghost.clone())
        .add_body("nutate", 0, hinge(Vec3::new(1.0, 0.0, 0.0)), ghost)
        .add_body(
            "spin",
            1,
            hinge(Vec3::new(0.0, 0.0, 1.0)),
            SpatialInertia::new(
                m,
                Vec3::new(0.0, 0.0, l),
                Mat3::from_diagonal(&Vec3::new(i1_com, i1_com, i3)),
            ),
        )
        .build()
}

/// Mean precession rate `⟨ψ̇⟩` of a top released from rest at tilt `theta0`
/// with axial spin `omega3`.
fn top_precession(model: &Model, theta0: f64, omega3: f64, dt: f64, t_end: f64) -> f64 {
    let mut state = model.default_state();
    state.q[1] = theta0;
    state.v[2] = omega3;

    let steps = (t_end / dt).round() as usize;
    let (mut sx, mut sy, mut sxx, mut sxy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for s in 0..steps {
        rk4_step(model, &mut state, dt);
        let t = (s + 1) as f64 * dt;
        let psi = state.q[0];
        sx += t;
        sy += psi;
        sxx += t * t;
        sxy += t * psi;
        n += 1.0;
    }
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

/// Run every rigid-body validation.
pub fn run() -> Suite {
    let mut suite = Suite::new("Rigid-body dynamics — Featherstone ABA (`phyz-rigid`)");

    // ---- 1. Finite-amplitude pendulum period -------------------------------
    let (m, l) = (1.0, 1.0);
    let model = rod_pendulum(m, l);
    let omega0 = (3.0 * G / (2.0 * l)).sqrt();
    let t_small = std::f64::consts::TAU / omega0;
    let dt = 1e-5;

    for &deg in &[5.0_f64, 30.0, 90.0, 150.0] {
        let theta0 = deg.to_radians();
        let t_exact = 4.0 / omega0 * ellip_k((theta0 / 2.0).sin());
        let t_meas = pendulum_period(&model, theta0, dt);
        suite.push(
            Validation::new(
                &format!("rigid.pendulum_period.{}deg", deg as i32),
                &format!("Rigid-rod pendulum period at θ₀ = {deg}°"),
                CRATE,
                "Exact solution of θ̈ = −ω₀² sin θ: T = (4/ω₀)K(sin(θ₀/2)), \
                 ω₀ = √(3g/2L) for a uniform rod hinged at one end",
                &format!("period (s), L = 1 m, RK4 with Δt = {dt:.0e}"),
                t_meas,
                t_exact,
                ErrorKind::Relative,
                1e-6,
            )
            .note(format!(
                "small-angle T₀ = {t_small:.6} s would be off by {:.2}% here — the elliptic \
                 correction is the thing being resolved",
                (t_small - t_exact).abs() / t_exact * 100.0
            )),
        );
    }

    // Period error must converge at RK4's fourth order.
    let theta0 = 90.0_f64.to_radians();
    let t_exact = 4.0 / omega0 * ellip_k((theta0 / 2.0).sin());
    let mut samples = Vec::new();
    for &dt in &[8e-3_f64, 4e-3, 2e-3, 1e-3] {
        let t = pendulum_period(&model, theta0, dt);
        samples.push((dt, (t - t_exact).abs() / t_exact));
    }
    let finest = samples.last().unwrap().1;
    suite.push(
        Validation::new(
            "rigid.pendulum_period_order",
            "Pendulum period error converges at fourth order in Δt",
            CRATE,
            "Classical RK4 is fourth-order accurate; the period error inherits that order",
            "relative period error at θ₀ = 90°, Δt = 1e−3 s",
            finest,
            0.0,
            ErrorKind::Absolute,
            1e-8,
        )
        .with_convergence(Convergence::fit("Δt", samples, 4.0, 0.6))
        .note(
            "A fourth-order fit here confirms the ABA acceleration is a smooth, correct function \
             of (q, v) — a wrong-but-consistent force law would still converge, but to the \
             wrong period, which the amplitude sweep above catches.",
        ),
    );

    // ---- 2. Energy conservation --------------------------------------------
    let drift = pendulum_energy_drift(&model, 90.0_f64.to_radians(), 1e-4, 200_000);
    suite.push(
        Validation::new(
            "rigid.pendulum_energy",
            "Total mechanical energy is conserved along the pendulum swing",
            CRATE,
            "Autonomous conservative system: T + V is an exact integral of motion",
            "peak |ΔE| / (swing energy) over 20 s at Δt = 1e−4 s",
            drift,
            0.0,
            ErrorKind::Absolute,
            1e-8,
        )
        .note(
            "Tests `phyz_rigid::kinetic_energy` (via CRBA) and `phyz_rigid::potential_energy` \
             jointly: an inconsistency between the mass matrix and the centre-of-mass height \
             shows up here even when ABA itself is correct.",
        ),
    );

    // ---- 3. Spinning-top precession ----------------------------------------
    let (m, l, i1, i3) = (1.0, 0.1, 0.05, 0.02);
    let top = spinning_top(m, l, i1, i3);
    let theta0 = 30.0_f64.to_radians();

    let omega3 = 200.0;
    let measured = top_precession(&top, theta0, omega3, 1e-4, 10.0);
    let expected = m * G * l / (i3 * omega3);
    suite.push(
        Validation::new(
            "rigid.top_precession",
            "Fast symmetric top: mean precession rate",
            CRATE,
            "Goldstein, *Classical Mechanics* 3e §5.7 — Ω = mgl/(I₃ω₃) in the fast-top limit",
            "⟨ψ̇⟩ (rad/s), ω₃ = 200 rad/s, θ₀ = 30°, averaged over 10 s",
            measured,
            expected,
            ErrorKind::Relative,
            0.01,
        )
        .note(
            "Released from rest in θ and ψ, so the top executes cuspidal nutation; the mean of \
             ψ̇ over many nutation cycles is the fast-top rate. Modelled as a z–x–z gimbal of \
             three revolute joints, which exercises a multi-body chain with a non-trivial \
             inertia tensor.",
        ),
    );

    // The residual must fall off as ω₃⁻², i.e. as (1/ω₃)².
    let mut samples = Vec::new();
    for &w in &[50.0_f64, 100.0, 200.0, 400.0] {
        let meas = top_precession(&top, theta0, w, 2e-5, 4.0);
        let exact = m * G * l / (i3 * w);
        samples.push((1.0 / w, (meas - exact).abs() / exact));
    }
    let finest = samples.last().unwrap().1;
    suite.push(
        Validation::new(
            "rigid.top_precession_order",
            "Top precession approaches the fast-top limit as ω₃⁻²",
            CRATE,
            "The next correction to Ω = mgl/(I₃ω₃) is O(mglI₁/(I₃²ω₃²)) (Goldstein §5.7)",
            "relative deviation from mgl/(I₃ω₃) at ω₃ = 400 rad/s",
            finest,
            0.0,
            ErrorKind::Absolute,
            5e-3,
        )
        .with_convergence(Convergence::fit("1/ω₃", samples, 2.0, 0.4))
        .note(
            "This is a physics convergence test rather than a numerical one: the *analytic* \
             formula is the approximation, and the simulator must reproduce the rate at which \
             it becomes exact.",
        ),
    );

    suite
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elliptic_k_matches_known_values() {
        assert!((ellip_k(0.0) - std::f64::consts::FRAC_PI_2).abs() < 1e-14);
        // K(1/√2) = 1.854074677301372
        assert!((ellip_k(0.5_f64.sqrt()) - 1.854_074_677_301_372).abs() < 1e-12);
    }
}
