//! Lattice-Boltzmann (`phyz-lbm`) validation against analytic and published
//! reference solutions.
//!
//! 1. **Plane Poiseuille flow** — the steady profile of body-force-driven channel
//!    flow against `u(y) = F y(H−y) / (2ρν)`.
//! 2. **Taylor–Green vortex** — the kinetic-energy decay rate against
//!    `E(t) = E₀ exp(−4νk²t)`, which measures the *effective* viscosity the
//!    BGK collision operator actually delivers.
//! 3. **Lid-driven cavity** — centreline velocity profiles at Re = 100 against
//!    Ghia, Ghia & Shin (1982).

use crate::report::{Convergence, ErrorKind, Suite, Validation};
use phyz_lbm::LatticeBoltzmann2D;

const CRATE: &str = "phyz-lbm";

// ---------------------------------------------------------------------------
// Poiseuille flow
// ---------------------------------------------------------------------------

/// Steady body-force-driven channel flow.
///
/// Periodic in x, solid walls at `y = 0` and `y = ny−1`. Returns the measured
/// centre-column velocity profile `u_x(j)` for `j = 0..ny`.
fn poiseuille(ny: usize, nu: f64, u_max: f64, max_steps: usize) -> (Vec<f64>, f64, usize) {
    let nx = 8;
    let h = (ny - 1) as f64;
    let rho = 1.0;
    // u_max = F H² / (8 ρ ν)  ⇒  F = 8 ρ ν u_max / H²
    let force = 8.0 * rho * nu * u_max / (h * h);

    let mut lbm = LatticeBoltzmann2D::new(nx, ny, nu);
    lbm.initialize_uniform(rho, [0.0, 0.0]);
    lbm.set_body_force([force, 0.0]);

    let mut prev = vec![0.0; ny];
    let mut residual = f64::INFINITY;
    let mut steps = 0;
    while steps < max_steps {
        lbm.collide_and_stream();
        for x in 0..nx {
            lbm.set_no_slip_bc(x, 0);
            lbm.set_no_slip_bc(x, ny - 1);
        }
        steps += 1;
        if steps.is_multiple_of(200) {
            let cur: Vec<f64> = (0..ny).map(|j| lbm.velocity(nx / 2, j)[0]).collect();
            residual = cur
                .iter()
                .zip(prev.iter())
                .fold(0.0_f64, |a, (c, p)| a.max((c - p).abs()))
                / u_max;
            prev = cur;
            if residual < 1e-9 {
                break;
            }
        }
    }

    let profile: Vec<f64> = (0..ny).map(|j| lbm.velocity(nx / 2, j)[0]).collect();
    (profile, residual, steps)
}

/// L2 error of a measured profile against the analytic parabola, normalised by `u_max`.
fn poiseuille_error(profile: &[f64], u_max: f64) -> f64 {
    let ny = profile.len();
    let h = (ny - 1) as f64;
    let mut s = 0.0;
    let mut n = 0.0;
    for (j, &u) in profile.iter().enumerate().take(ny - 1).skip(1) {
        let y = j as f64;
        let exact = 4.0 * u_max * y * (h - y) / (h * h);
        s += (u - exact).powi(2);
        n += 1.0;
    }
    (s / n).sqrt() / u_max
}

/// Fit `u = c0 + c1 y + c2 y²` to the interior nodes; return `(u_peak, y_lo, y_hi)`
/// where `y_lo`, `y_hi` are the parabola's roots (the *effective* wall positions).
fn poiseuille_fit(profile: &[f64]) -> (f64, f64, f64) {
    let pts: Vec<(f64, f64)> = profile
        .iter()
        .enumerate()
        .skip(1)
        .take(profile.len() - 2)
        .map(|(j, &u)| (j as f64, u))
        .collect();
    // Normal equations for a quadratic fit.
    let mut a = [[0.0_f64; 3]; 3];
    let mut b = [0.0_f64; 3];
    for &(y, u) in &pts {
        let basis = [1.0, y, y * y];
        for r in 0..3 {
            for c in 0..3 {
                a[r][c] += basis[r] * basis[c];
            }
            b[r] += basis[r] * u;
        }
    }
    // Gaussian elimination (3×3).
    for i in 0..3 {
        let piv = a[i][i];
        for c in 0..3 {
            a[i][c] /= piv;
        }
        b[i] /= piv;
        for r in 0..3 {
            if r != i {
                let f = a[r][i];
                for c in 0..3 {
                    a[r][c] -= f * a[i][c];
                }
                b[r] -= f * b[i];
            }
        }
    }
    let (c0, c1, c2) = (b[0], b[1], b[2]);
    let peak = c0 - c1 * c1 / (4.0 * c2);
    let disc = (c1 * c1 - 4.0 * c2 * c0).max(0.0).sqrt();
    let r1 = (-c1 - disc) / (2.0 * c2);
    let r2 = (-c1 + disc) / (2.0 * c2);
    (peak, r1.min(r2), r1.max(r2))
}

// ---------------------------------------------------------------------------
// Taylor–Green vortex
// ---------------------------------------------------------------------------

/// Decaying Taylor–Green vortex on a doubly periodic `n × n` lattice.
///
/// Returns the effective viscosity recovered from the kinetic-energy decay.
fn taylor_green_nu_eff(n: usize, nu: f64, u0: f64) -> f64 {
    let k = std::f64::consts::TAU / n as f64;
    let mut lbm = LatticeBoltzmann2D::new(n, n, nu);
    for y in 0..n {
        for x in 0..n {
            let (xf, yf) = (x as f64, y as f64);
            let ux = -u0 * (k * xf).cos() * (k * yf).sin();
            let uy = u0 * (k * xf).sin() * (k * yf).cos();
            // Compressible pressure correction for the TGV initial condition.
            let rho = 1.0 - 0.75 * u0 * u0 * ((2.0 * k * xf).cos() + (2.0 * k * yf).cos());
            lbm.initialize_at(x, y, rho, [ux, uy]);
        }
    }

    // Observe over roughly one e-folding of the velocity amplitude.
    let t_end = (1.0 / (2.0 * nu * k * k)).min(20000.0);
    let steps = t_end as usize;
    let sample_every = (steps / 40).max(1);

    let mut samples = Vec::new();
    for s in 0..=steps {
        if s.is_multiple_of(sample_every) {
            let ke = lbm.kinetic_energy();
            if ke > 0.0 {
                samples.push((s as f64, ke.ln()));
            }
        }
        if s < steps {
            lbm.collide_and_stream();
        }
    }

    // Least-squares slope of ln(E) vs t; E ∝ exp(−4νk²t).
    let m = samples.len() as f64;
    let sx: f64 = samples.iter().map(|p| p.0).sum();
    let sy: f64 = samples.iter().map(|p| p.1).sum();
    let sxx: f64 = samples.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = samples.iter().map(|p| p.0 * p.1).sum();
    let slope = (m * sxy - sx * sy) / (m * sxx - sx * sx);
    -slope / (4.0 * k * k)
}

// ---------------------------------------------------------------------------
// Lid-driven cavity — Ghia, Ghia & Shin (1982) reference data
// ---------------------------------------------------------------------------

/// `u`-velocity along the vertical centreline at Re = 100 (Ghia et al. Table I).
const GHIA_U_RE100: [(f64, f64); 17] = [
    (1.0000, 1.00000),
    (0.9766, 0.84123),
    (0.9688, 0.78871),
    (0.9609, 0.73722),
    (0.9531, 0.68717),
    (0.8516, 0.23151),
    (0.7344, 0.00332),
    (0.6172, -0.13641),
    (0.5000, -0.20581),
    (0.4531, -0.21090),
    (0.2813, -0.15662),
    (0.1719, -0.10150),
    (0.1016, -0.06434),
    (0.0703, -0.04775),
    (0.0625, -0.04192),
    (0.0547, -0.03717),
    (0.0000, 0.00000),
];

/// `v`-velocity along the horizontal centreline at Re = 100 (Ghia et al. Table II).
const GHIA_V_RE100: [(f64, f64); 17] = [
    (1.0000, 0.00000),
    (0.9688, -0.05906),
    (0.9609, -0.07391),
    (0.9531, -0.08864),
    (0.9453, -0.10313),
    (0.9063, -0.16914),
    (0.8594, -0.22445),
    (0.8047, -0.24533),
    (0.5000, 0.05454),
    (0.2344, 0.17527),
    (0.2266, 0.17507),
    (0.1563, 0.16077),
    (0.0938, 0.12317),
    (0.0781, 0.10890),
    (0.0703, 0.10091),
    (0.0625, 0.09233),
    (0.0000, 0.00000),
];

/// Steady lid-driven cavity. Returns `(rms_u, rms_v, steps, residual)`, both
/// RMS errors normalised by the lid velocity.
fn cavity(n: usize, re: f64, u_lid: f64, max_steps: usize) -> (f64, f64, usize, f64) {
    let l = (n - 1) as f64;
    let nu = u_lid * l / re;
    let mut lbm = LatticeBoltzmann2D::new(n, n, nu);
    lbm.initialize_uniform(1.0, [0.0, 0.0]);

    let mut prev = vec![0.0; n];
    let mut residual = f64::INFINITY;
    let mut steps = 0;
    while steps < max_steps {
        lbm.collide_and_stream();
        for x in 0..n {
            lbm.set_no_slip_bc(x, 0);
            lbm.set_velocity_bc(x, n - 1, [u_lid, 0.0]);
        }
        for y in 0..n {
            lbm.set_no_slip_bc(0, y);
            lbm.set_no_slip_bc(n - 1, y);
        }
        steps += 1;
        if steps.is_multiple_of(500) {
            let cur: Vec<f64> = (0..n).map(|j| lbm.velocity(n / 2, j)[0]).collect();
            residual = cur
                .iter()
                .zip(prev.iter())
                .fold(0.0_f64, |a, (c, p)| a.max((c - p).abs()))
                / u_lid;
            prev = cur;
            if residual < 1e-8 {
                break;
            }
        }
    }

    let sample = |frac: f64, along_y: bool| -> f64 {
        let pos = frac * l;
        let j0 = pos.floor().clamp(0.0, l - 1.0) as usize;
        let t = pos - j0 as f64;
        let get = |idx: usize| -> f64 {
            if along_y {
                lbm.velocity(n / 2, idx)[0]
            } else {
                lbm.velocity(idx, n / 2)[1]
            }
        };
        (get(j0) * (1.0 - t) + get(j0 + 1) * t) / u_lid
    };

    let rms = |data: &[(f64, f64)], along_y: bool| -> f64 {
        let s: f64 = data
            .iter()
            .map(|&(frac, refv)| (sample(frac, along_y) - refv).powi(2))
            .sum();
        (s / data.len() as f64).sqrt()
    };

    (
        rms(&GHIA_U_RE100, true),
        rms(&GHIA_V_RE100, false),
        steps,
        residual,
    )
}

// ---------------------------------------------------------------------------

/// Run every lattice-Boltzmann validation.
pub fn run() -> Suite {
    let mut suite = Suite::new("Fluids — lattice Boltzmann D2Q9 (`phyz-lbm`)");

    // ---- 1. Poiseuille flow -------------------------------------------------
    let u_max = 0.02;
    let re = 10.0;

    let ny = 33;
    let nu = u_max * (ny - 1) as f64 / re;
    let (profile, residual, steps) = poiseuille(ny, nu, u_max, 400_000);
    let l2 = poiseuille_error(&profile, u_max);
    let (peak, y_lo, y_hi) = poiseuille_fit(&profile);

    suite.push(
        Validation::new(
            "lbm.poiseuille.profile",
            "Plane Poiseuille flow: velocity profile vs u(y) = F y(H−y)/(2ρν)",
            CRATE,
            "Closed form for steady laminar channel flow (Batchelor §4.2)",
            "RMS profile error / u_max, 33-node channel, Re = 10",
            l2,
            0.0,
            ErrorKind::Absolute,
            0.02,
        )
        .note(format!(
            "converged in {steps} steps, residual {residual:.2e}"
        ))
        .note(format!(
            "quadratic fit to the measured profile puts the no-slip planes at y = {y_lo:.3} and \
             y = {y_hi:.3}, i.e. an effective channel height H_eff = {:.3} against the nominal \
             H = {} (node-centred walls at y = 0 and y = {}).",
            y_hi - y_lo,
            ny - 1,
            ny - 1
        ))
        .note(
            "`LatticeBoltzmann2D::set_no_slip_bc` (crates/phyz-lbm/src/d2q9.rs) reflects the \
             *whole* distribution at a wall node after streaming, which places the no-slip \
             plane at the wall node itself rather than half a lattice spacing outside it as \
             halfway bounce-back does. The fitted wall positions above quantify that offset.",
        ),
    );

    suite.push(
        Validation::new(
            "lbm.poiseuille.peak",
            "Plane Poiseuille flow: centreline velocity vs u_max = F H²/(8ρν)",
            CRATE,
            "Closed form for steady laminar channel flow",
            "fitted peak velocity (lattice units), analytic u_max = 0.02",
            peak,
            u_max,
            ErrorKind::Relative,
            0.02,
        )
        .note("The body force is set from the analytic relation, so this is a direct test of the \
               viscosity the BGK operator actually realises together with the wall treatment."),
    );

    // Convergence under grid refinement at fixed Reynolds number and u_max.
    let mut samples = Vec::new();
    for &ny in &[9_usize, 17, 33, 65] {
        let nu = u_max * (ny - 1) as f64 / re;
        let (p, _, _) = poiseuille(ny, nu, u_max, 400_000);
        samples.push((1.0 / (ny - 1) as f64, poiseuille_error(&p, u_max)));
    }
    let finest = samples.last().unwrap().1;
    suite.push(
        Validation::new(
            "lbm.poiseuille.convergence",
            "Poiseuille profile error vanishes as Δx² under refinement",
            CRATE,
            "LBM-BGK with a second-order-accurate wall treatment is second order in Δx \
             (and exact for Poiseuille with halfway bounce-back + Guo forcing)",
            "RMS profile error / u_max at 65 nodes",
            finest,
            0.0,
            ErrorKind::Absolute,
            0.01,
        )
        .with_convergence(Convergence::fit("Δx/H", samples, 2.0, 0.4)),
    );

    // ---- 2. Taylor–Green vortex --------------------------------------------
    let nu = 0.02;
    let u0 = 0.02;
    let n = 64;
    let nu_eff = taylor_green_nu_eff(n, nu, u0);
    suite.push(
        Validation::new(
            "lbm.taylor_green.decay",
            "Taylor–Green vortex: kinetic-energy decay rate vs E(t) = E₀ exp(−4νk²t)",
            CRATE,
            "Taylor & Green (1937); exact unsteady Navier–Stokes solution on a periodic domain",
            "effective viscosity from the decay rate (lattice units), input ν = 0.02, 64² lattice",
            nu_eff,
            nu,
            ErrorKind::Relative,
            0.02,
        )
        .note(
            "This measures the viscosity the collide/stream pair actually delivers, which is \
             what ties τ = 3ν + 1/2 to physical dissipation.",
        ),
    );

    let mut samples = Vec::new();
    for &n in &[16_usize, 32, 64, 128] {
        let e = (taylor_green_nu_eff(n, nu, u0) - nu).abs() / nu;
        samples.push((1.0 / n as f64, e));
    }
    let finest = samples.last().unwrap().1;
    suite.push(
        Validation::new(
            "lbm.taylor_green.convergence",
            "Taylor–Green viscosity error vanishes as Δx² under refinement",
            CRATE,
            "Chapman–Enskog: the leading BGK error in the recovered viscosity is O((kΔx)²)",
            "|ν_eff − ν| / ν at 128²",
            finest,
            0.0,
            ErrorKind::Absolute,
            0.01,
        )
        .with_convergence(Convergence::fit("Δx/L", samples, 2.0, 0.4)),
    );

    // ---- 3. Lid-driven cavity ----------------------------------------------
    let mut cavity_errors = Vec::new();
    for &n in &[33_usize, 65] {
        let (rms_u, rms_v, steps, residual) = cavity(n, 100.0, 0.1, 400_000);
        cavity_errors.push((1.0 / (n - 1) as f64, rms_u));
        suite.push(
            Validation::new(
                &format!("lbm.cavity_re100.u.n{n}"),
                &format!("Lid-driven cavity Re = 100, u along the vertical centreline ({n}²)"),
                CRATE,
                "Ghia, Ghia & Shin, *J. Comput. Phys.* 48 (1982) 387, Table I (Re = 100)",
                "RMS(u/u_lid − Ghia) over the 17 tabulated stations",
                rms_u,
                0.0,
                ErrorKind::Absolute,
                0.02,
            )
            .note(format!("{steps} steps, centreline residual {residual:.2e}")),
        );
        suite.push(Validation::new(
            &format!("lbm.cavity_re100.v.n{n}"),
            &format!("Lid-driven cavity Re = 100, v along the horizontal centreline ({n}²)"),
            CRATE,
            "Ghia, Ghia & Shin, *J. Comput. Phys.* 48 (1982) 387, Table II (Re = 100)",
            "RMS(v/u_lid − Ghia) over the 17 tabulated stations",
            rms_v,
            0.0,
            ErrorKind::Absolute,
            0.02,
        ));
    }

    let finest = cavity_errors.last().unwrap().1;
    suite.push(
        Validation::new(
            "lbm.cavity_re100.convergence",
            "Lid-driven cavity error vanishes as Δx² under refinement",
            CRATE,
            "A second-order wall/lid treatment (halfway bounce-back, Zou–He or \
             regularised velocity boundaries) gives second-order convergence to the \
             Ghia et al. profiles",
            "RMS(u/u_lid − Ghia) at 65²",
            finest,
            0.0,
            ErrorKind::Absolute,
            0.02,
        )
        .with_convergence(Convergence::fit("Δx/L", cavity_errors, 2.0, 0.4))
        .note(
            "Only two resolutions are run (the 65² case already needs ~4·10⁴ steps to reach a \
             10⁻⁸ steady-state residual), so the fitted order is a two-point slope, not a \
             regression. It is still decisive between first and second order.",
        )
        .note(
            "The lid uses `set_velocity_bc`, which overwrites the whole distribution with the \
             equilibrium at the local density. That discards the non-equilibrium part of f and \
             is only first-order accurate, as is the full-node bounce-back on the three solid \
             walls — the same wall treatment the Poiseuille benchmark measures directly.",
        ),
    );

    suite
}
