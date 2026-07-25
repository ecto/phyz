//! Gravity (`phyz-gravity`) validation against closed-form orbital mechanics.
//!
//! 1. **Kepler two-body invariants** — energy, angular momentum and the
//!    Laplace–Runge–Lenz (eccentricity) vector of a Newtonian two-body orbit,
//!    with `O(Δt²)` convergence of the bounded symplectic error.
//! 2. **Mercury perihelion precession** — the post-Newtonian equations of motion
//!    are *integrated* (not merely evaluated in closed form) and the secular
//!    rotation of the eccentricity vector is compared with the general-relativistic
//!    43″/century.

use crate::report::{Convergence, ErrorKind, Suite, Validation};
use phyz_gravity::{C, G, GravityParticle, GravitySolver, PostNewtonianSolver};
use phyz_math::Vec3;

const CRATE: &str = "phyz-gravity";

/// Seconds in a Julian century.
const CENTURY: f64 = 36525.0 * 86400.0;
/// Radians to arcseconds.
const ARCSEC: f64 = 206264.806247096;

/// Conserved quantities of the relative two-body problem.
struct Invariants {
    /// Specific orbital energy `v²/2 − μ/r`.
    energy: f64,
    /// Specific angular momentum vector `r × v`.
    h: Vec3,
    /// Eccentricity (Laplace–Runge–Lenz) vector `(v×h)/μ − r̂`.
    e_vec: Vec3,
}

fn invariants(p: &[GravityParticle]) -> Invariants {
    let mu = G * (p[0].m + p[1].m);
    let r = p[1].x - p[0].x;
    let v = p[1].v - p[0].v;
    let r_mag = r.norm();
    let h = r.cross(&v);
    Invariants {
        energy: 0.5 * v.norm_squared() - mu / r_mag,
        h,
        e_vec: v.cross(&h) / mu - r / r_mag,
    }
}

/// Sun + one planet on an ellipse of semi-major axis `a` and eccentricity `e`,
/// started at apoapsis in the x–y plane.
fn two_body(m_central: f64, m_orbiter: f64, a: f64, e: f64) -> Vec<GravityParticle> {
    let mu = G * (m_central + m_orbiter);
    let r_apo = a * (1.0 + e);
    let v_apo = (mu * (1.0 - e) / (a * (1.0 + e))).sqrt();
    // Place the barycentre at the origin so the system has no net drift.
    let total = m_central + m_orbiter;
    let f_c = m_orbiter / total;
    let f_o = m_central / total;
    vec![
        GravityParticle::new(
            Vec3::new(-r_apo * f_c, 0.0, 0.0),
            Vec3::new(0.0, -v_apo * f_c, 0.0),
            m_central,
        ),
        GravityParticle::new(
            Vec3::new(r_apo * f_o, 0.0, 0.0),
            Vec3::new(0.0, v_apo * f_o, 0.0),
            m_orbiter,
        ),
    ]
}

/// Keplerian orbital period for the given total mass.
fn period(a: f64, m_total: f64) -> f64 {
    2.0 * std::f64::consts::PI * (a * a * a / (G * m_total)).sqrt()
}

/// Result of one integration: peak relative drift of each invariant plus the
/// secular rotation rate of the eccentricity vector (rad/s).
struct Run {
    d_energy: f64,
    d_angular_momentum: f64,
    d_lrl: f64,
    precession: f64,
}

/// Integrate with velocity Verlet using the crate's own particle integrator.
fn integrate(mut solver: PostNewtonianSolver, mut p: Vec<GravityParticle>, dt: f64, steps: usize) -> Run {
    solver.compute_forces(&mut p);
    let inv0 = invariants(&p);
    let ang0 = inv0.e_vec.y.atan2(inv0.e_vec.x);

    let mut d_e: f64 = 0.0;
    let mut d_h: f64 = 0.0;
    let mut d_l: f64 = 0.0;

    // Accumulate the unwrapped eccentricity-vector angle so a full turn is not
    // aliased away, and fit its slope by least squares at the end.
    let mut prev = 0.0_f64;
    let mut turns = 0.0_f64;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut n = 0.0;

    for step in 0..steps {
        for q in p.iter_mut() {
            q.velocity_verlet_step(dt);
        }
        solver.compute_forces(&mut p);
        for q in p.iter_mut() {
            q.velocity_verlet_complete(dt);
        }

        let inv = invariants(&p);
        d_e = d_e.max((inv.energy - inv0.energy).abs() / inv0.energy.abs());
        d_h = d_h.max((inv.h - inv0.h).norm() / inv0.h.norm());
        d_l = d_l.max((inv.e_vec - inv0.e_vec).norm() / inv0.e_vec.norm().max(1e-12));

        let raw = inv.e_vec.y.atan2(inv.e_vec.x) - ang0;
        let mut unwrapped = raw + turns * std::f64::consts::TAU;
        if unwrapped - prev > std::f64::consts::PI {
            turns -= 1.0;
            unwrapped -= std::f64::consts::TAU;
        } else if prev - unwrapped > std::f64::consts::PI {
            turns += 1.0;
            unwrapped += std::f64::consts::TAU;
        }
        prev = unwrapped;

        let t = (step + 1) as f64 * dt;
        sx += t;
        sy += unwrapped;
        sxx += t * t;
        sxy += t * unwrapped;
        n += 1.0;
    }

    let precession = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    Run {
        d_energy: d_e,
        d_angular_momentum: d_h,
        d_lrl: d_l,
        precession,
    }
}

fn newtonian_solver() -> PostNewtonianSolver {
    let mut s = PostNewtonianSolver::new(0.0);
    s.softening = 0.0;
    s
}

fn pn1_solver() -> PostNewtonianSolver {
    let mut s = PostNewtonianSolver::new(1.0);
    s.softening = 0.0;
    s
}

/// Mercury's osculating elements (IAU / JPL DE ephemeris values).
const A_MERCURY: f64 = 5.790905e10;
const E_MERCURY: f64 = 0.20563069;
const M_SUN: f64 = 1.98892e30;
const M_MERCURY: f64 = 3.302e23;

/// GR prediction: Δϖ = 6πGM / (c²a(1−e²)) per orbit.
fn gr_precession_arcsec_per_century() -> f64 {
    let per_orbit = 6.0 * std::f64::consts::PI * G * (M_SUN + M_MERCURY)
        / (C * C * A_MERCURY * (1.0 - E_MERCURY * E_MERCURY));
    let t = period(A_MERCURY, M_SUN + M_MERCURY);
    per_orbit * (CENTURY / t) * ARCSEC
}

/// Run every gravity validation.
pub fn run() -> Suite {
    let mut suite = Suite::new("Gravity — Newtonian and post-Newtonian (`phyz-gravity`)");

    // ---- 1. Kepler two-body invariants --------------------------------------
    let a = 1.4959787e11;
    let ecc = 0.2;
    let m_o = 5.972e24;
    let t_orbit = period(a, M_SUN + m_o);

    let orbits = 20.0;
    let base_spo = 2000.0; // steps per orbit
    let dt = t_orbit / base_spo;
    let run = integrate(
        newtonian_solver(),
        two_body(M_SUN, m_o, a, ecc),
        dt,
        (orbits * base_spo) as usize,
    );

    // Tolerances are derived from theory, not fitted: velocity Verlet's bounded
    // energy error on a Kepler orbit is C(ωΔt)² with C = O(1), so 10(2π/N)² is a
    // generous but principled bound at N steps per orbit.
    let vv_bound = 10.0 * (std::f64::consts::TAU / base_spo).powi(2);
    suite.push(
        Validation::new(
            "gravity.kepler.energy",
            "Kepler two-body: specific orbital energy is conserved",
            CRATE,
            "Closed form: E = −μ/2a is an exact integral of the Newtonian two-body problem",
            "peak |ΔE|/|E| over 20 orbits (e = 0.2, 2000 steps/orbit)",
            run.d_energy,
            0.0,
            ErrorKind::Absolute,
            vv_bound,
        )
        .note("Velocity Verlet is symplectic, so the energy error must stay bounded rather than growing secularly.")
        .note(format!(
            "Tolerance 10(2π/N)² = {vv_bound:.2e} is the theoretical O((ωΔt)²) bound at \
             N = {base_spo:.0} steps/orbit, not a fitted number; the convergence entries below \
             verify the Δt² scaling that justifies it."
        )),
    );
    suite.push(Validation::new(
        "gravity.kepler.angular_momentum",
        "Kepler two-body: specific angular momentum is conserved",
        CRATE,
        "Closed form: h = r × v is exactly conserved for any central force",
        "peak |Δh|/|h| over 20 orbits",
        run.d_angular_momentum,
        0.0,
        ErrorKind::Absolute,
        1e-9,
    ));
    suite.push(
        Validation::new(
            "gravity.kepler.lrl",
            "Kepler two-body: Laplace–Runge–Lenz vector is conserved",
            CRATE,
            "Closed form: e = (v×h)/μ − r̂ is exactly conserved only for a 1/r² force \
             (Bertrand/Runge–Lenz symmetry)",
            "peak |Δe|/|e| over 20 orbits",
            run.d_lrl,
            0.0,
            ErrorKind::Absolute,
            orbits * vv_bound,
        )
        .note("The LRL vector is the sharpest of the three: it detects any spurious \
               non-1/r² component in the force law, which energy and angular momentum do not.")
        .note(format!(
            "Unlike the energy error, the apsidal error of a symplectic integrator accumulates \
             linearly in the number of orbits, so the pre-registered bound is \
             N_orbits · 10(2π/N_steps)² = {:.2e}.",
            orbits * vv_bound
        )),
    );

    // Convergence: bounded symplectic error must scale as Δt².
    let mut e_samples = Vec::new();
    let mut l_samples = Vec::new();
    for &spo in &[500.0_f64, 1000.0, 2000.0, 4000.0] {
        let r = integrate(
            newtonian_solver(),
            two_body(M_SUN, m_o, a, ecc),
            t_orbit / spo,
            (5.0 * spo) as usize,
        );
        e_samples.push((1.0 / spo, r.d_energy));
        l_samples.push((1.0 / spo, r.d_lrl));
    }
    let finest_e = e_samples.last().unwrap().1;
    suite.push(
        Validation::new(
            "gravity.kepler.energy_order",
            "Kepler two-body: energy error scales as Δt² (velocity Verlet)",
            CRATE,
            "Velocity Verlet is a second-order symplectic integrator; the shadow-Hamiltonian \
             energy error is O(Δt²) and bounded",
            "peak |ΔE|/|E| over 5 orbits at 4000 steps/orbit",
            finest_e,
            0.0,
            ErrorKind::Absolute,
            10.0 * (std::f64::consts::TAU / 4000.0).powi(2),
        )
        .with_convergence(Convergence::fit("Δt/T", e_samples, 2.0, 0.25)),
    );
    let finest_l = l_samples.last().unwrap().1;
    suite.push(
        Validation::new(
            "gravity.kepler.lrl_order",
            "Kepler two-body: LRL error scales as Δt²",
            CRATE,
            "Second-order integrator on an exactly-conserved vector",
            "peak |Δe|/|e| over 5 orbits at 4000 steps/orbit",
            finest_l,
            0.0,
            ErrorKind::Absolute,
            5.0 * 10.0 * (std::f64::consts::TAU / 4000.0).powi(2),
        )
        .with_convergence(Convergence::fit("Δt/T", l_samples, 2.0, 0.25)),
    );

    // ---- 2. Mercury perihelion precession, actually integrated --------------
    let t_mercury = period(A_MERCURY, M_SUN + M_MERCURY);
    let gr = gr_precession_arcsec_per_century();
    let orbits = 400.0;
    let spo = 4000.0;
    let dt = t_mercury / spo;
    let steps = (orbits * spo) as usize;

    let newt = integrate(
        newtonian_solver(),
        two_body(M_SUN, M_MERCURY, A_MERCURY, E_MERCURY),
        dt,
        steps,
    );
    let pn = integrate(
        pn1_solver(),
        two_body(M_SUN, M_MERCURY, A_MERCURY, E_MERCURY),
        dt,
        steps,
    );

    // Subtract the Newtonian run so the integrator's own O(Δt²) apsidal drift
    // cancels and only the 1PN contribution remains.
    let measured = (pn.precession - newt.precession) * CENTURY * ARCSEC;
    let baseline = newt.precession * CENTURY * ARCSEC;

    suite.push(
        Validation::new(
            "gravity.pn.mercury_precession",
            "Mercury perihelion precession from integrated 1PN equations of motion",
            CRATE,
            "Einstein (1915); Will, *Living Rev. Rel.* 17 (2014) — Δϖ = 6πGM/(c²a(1−e²)) \
             = 42.98″/century for Mercury",
            "precession of the eccentricity vector (arcsec/century), 400 orbits at 4000 steps/orbit",
            measured,
            gr,
            ErrorKind::Relative,
            0.02,
        )
        .note(format!(
            "Newtonian control run (same integrator, same Δt, 1PN term switched off) drifts \
             {baseline:+.4}″/century; that baseline is subtracted from the 1PN run so the \
             residual is the physical effect and not integrator error."
        ))
        .note(format!(
            "Closed-form check of the textbook formula alone: 6πGM/(c²a(1−e²)) = {gr:.3}″/century \
             — this is what the pre-existing `test_mercury_precession` \
             (crates/phyz-gravity/src/pn.rs:321-334) asserted, without ever calling the solver."
        ))
        .note(
            "The docstring at pn.rs:78-86 states \
             a_1PN = Gm_j/r² [(4G(m_i+m_j)/r − v_i²)n + 4(v_i·v_j)n − (v_i·n)v_j]/c², \
             which is not the standard EIH 1PN acceleration. The EIH form \
             (Will 1993 eq. 6.80; Blanchet 2014 eq. 203) is \
             a_1PN = −(Gm_j/r²c²){ n̂[4Gm_i/r + 5Gm_j/r − v_i² − 2v_j² + 4v_i·v_j + (3/2)(n̂·v_j)²] \
             + (v_i − v_j)(4n̂·v_i − 3n̂·v_j) }, with n̂ pointing from j to i. \
             The code at pn.rs:89-114 differs from EIH in three places: the mass coefficient is \
             4(m_i+m_j) rather than 4m_i + 5m_j; the velocity term multiplies v_j rather than \
             (v_i − v_j); and the overall sign is positive in the code's n = (x_j − x_i)/r \
             convention where EIH requires negative. Any one of those changes the precession.",
        ),
    );

    // Convergence of the measured precession under Δt refinement.
    let mut samples = Vec::new();
    for &spo in &[1000.0_f64, 2000.0, 4000.0] {
        let dt = t_mercury / spo;
        let steps = (60.0 * spo) as usize;
        let n = integrate(
            newtonian_solver(),
            two_body(M_SUN, M_MERCURY, A_MERCURY, E_MERCURY),
            dt,
            steps,
        );
        let p = integrate(
            pn1_solver(),
            two_body(M_SUN, M_MERCURY, A_MERCURY, E_MERCURY),
            dt,
            steps,
        );
        let m = (p.precession - n.precession) * CENTURY * ARCSEC;
        samples.push((1.0 / spo, (m - gr).abs() / gr));
    }
    let finest = samples.last().unwrap().1;
    suite.push(
        Validation::new(
            "gravity.pn.mercury_convergence",
            "Integrated Mercury precession converges to the GR value as Δt → 0",
            CRATE,
            "Δϖ = 6πGM/(c²a(1−e²)); a correct 1PN force law makes the residual a pure \
             integrator error that vanishes as Δt²",
            "|Δϖ_measured − Δϖ_GR| / Δϖ_GR at 4000 steps/orbit (60 orbits)",
            finest,
            0.0,
            ErrorKind::Absolute,
            0.02,
        )
        .with_convergence(Convergence::fit("Δt/T", samples, 2.0, 0.5))
        .note(
            "If the residual does not shrink under refinement, the discrepancy is in the \
             force law, not the integrator.",
        ),
    );

    suite
}
