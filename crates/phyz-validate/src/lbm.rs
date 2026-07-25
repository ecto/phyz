//! Lattice-Boltzmann (`phyz-lbm`) validation against analytic and published
//! reference solutions.
//!
//! 1. **Plane Poiseuille flow** — the steady profile of body-force-driven channel
//!    flow against `u(y) = F y(H−y)/(2ρν)`, and — the sharper test — whether that
//!    error is *independent of viscosity*, which separates a genuine wall
//!    treatment from one that merely happens to be calibrated at one `τ`.
//! 2. **Taylor–Green vortex** — the velocity field and its decay rate against
//!    `u(t) = u₀ exp(−ν(k_x²+k_y²)t)`, with second-order spatial convergence.
//! 3. **Lid-driven cavity** — centreline velocities at Re = 100 against
//!    Ghia, Ghia & Shin (1982).
//!
//! The closed-form solutions and the Ghia tables come from
//! [`phyz_lbm::analytic`] rather than being restated here, so this suite and
//! `phyz-lbm`'s own `tests/validation.rs` are driven by one set of formulas.

use crate::report::{Convergence, ErrorKind, Suite, Validation};
use phyz_lbm::analytic::{self, ghia};
use phyz_lbm::{Boundaries, CollisionModel, LatticeBoltzmann2D, boundary};

const CRATE: &str = "phyz-lbm";

// ---------------------------------------------------------------------------
// Poiseuille flow
// ---------------------------------------------------------------------------

/// Relative L2 error of steady force-driven channel flow against the analytic
/// parabola.
///
/// The body force is chosen to hold the peak velocity fixed as `nu` varies, so
/// the Mach number — and with it the compressibility error — is identical for
/// every viscosity. Anything left over is the boundary treatment.
fn poiseuille_error(collision: CollisionModel, ny: usize, nu: f64, u_peak: f64) -> f64 {
    let h = ny as f64;
    let g = analytic::poiseuille_force_for_peak(h, u_peak, nu);

    let mut lbm = LatticeBoltzmann2D::new(4, ny, nu)
        .with_collision(collision)
        .with_boundaries(boundary::channel_2d())
        .with_force([g, 0.0]);
    lbm.initialize_uniform(1.0, [0.0, 0.0]);
    lbm.run_to_steady_state(1e-12, 400_000, 200);

    let mut num = 0.0;
    let mut den = 0.0;
    for y in 0..ny {
        let want = analytic::poiseuille_force_driven(y as f64, -0.5, h - 0.5, g, nu);
        let got = lbm.velocity(0, y)[0];
        num += (got - want) * (got - want);
        den += want * want;
    }
    (num / den).sqrt()
}

// ---------------------------------------------------------------------------
// Taylor–Green vortex
// ---------------------------------------------------------------------------

/// Initialise a Taylor–Green vortex on an `n × n` periodic lattice.
fn taylor_green(n: usize, u0: f64, nu: f64) -> (LatticeBoltzmann2D, f64) {
    let k = analytic::wavenumber(n, 1);
    let mut lbm = LatticeBoltzmann2D::new(n, n, nu);
    lbm.initialize_with(|x, y| {
        let (fx, fy) = (x as f64, y as f64);
        (
            analytic::taylor_green_density(fx, fy, 0.0, 1.0, u0, k, k, nu),
            analytic::taylor_green_velocity(fx, fy, 0.0, u0, k, k, nu),
        )
    });
    (lbm, k)
}

/// Relative L2 error of the vortex velocity field after `steps` steps.
fn taylor_green_field_error(n: usize, u0: f64, nu: f64, steps: usize) -> f64 {
    let (mut lbm, k) = taylor_green(n, u0, nu);
    lbm.run(steps);
    let t = steps as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for y in 0..n {
        for x in 0..n {
            let want = analytic::taylor_green_velocity(x as f64, y as f64, t, u0, k, k, nu);
            let got = lbm.velocity(x, y);
            num += (got[0] - want[0]).powi(2) + (got[1] - want[1]).powi(2);
            den += want[0] * want[0] + want[1] * want[1];
        }
    }
    (num / den).sqrt()
}

/// Effective viscosity recovered from the kinetic-energy decay rate.
fn taylor_green_nu_eff(n: usize, u0: f64, nu: f64, steps: usize) -> f64 {
    let (mut lbm, k) = taylor_green(n, u0, nu);
    let ke0 = lbm.kinetic_energy();
    lbm.run(steps);
    let ke1 = lbm.kinetic_energy();
    // E(t) ∝ exp(−2ν(k_x²+k_y²)t) with k_x = k_y = k.
    let rate = -((ke1 / ke0).ln()) / steps as f64;
    rate / (2.0 * 2.0 * k * k)
}

// ---------------------------------------------------------------------------
// Lid-driven cavity
// ---------------------------------------------------------------------------

/// Steady lid-driven cavity at the given Reynolds number.
///
/// Returns `(u on the vertical centreline, v on the horizontal centreline,
/// steps, residual)`, both normalised by the lid speed and sampled at the Ghia
/// stations.
fn cavity(n: usize, re: f64, u_lid: f64) -> (Vec<f64>, Vec<f64>, usize, f64) {
    let nu = u_lid * n as f64 / re;
    let mut lbm =
        LatticeBoltzmann2D::new(n, n, nu).with_boundaries(boundary::cavity_2d([u_lid, 0.0]));
    lbm.initialize_uniform(1.0, [0.0, 0.0]);
    let (steps, residual) = lbm.run_to_steady_state(1e-8, 200_000, 500);

    (
        ghia::Y
            .iter()
            .map(|&y| sample(y, n, 0.0, u_lid, |j| lbm.velocity(n / 2, j)[0]) / u_lid)
            .collect(),
        ghia::X
            .iter()
            .map(|&x| sample(x, n, 0.0, 0.0, |i| lbm.velocity(i, n / 2)[1]) / u_lid)
            .collect(),
        steps,
        residual,
    )
}

/// Linear interpolation of a nodal field at normalised position `s ∈ [0, 1]`.
///
/// Node `j` sits at `(j + 0.5)/n`, so in node coordinates `p = s·n − 0.5` the
/// walls lie at `p = −0.5` and `p = n − 0.5`. `wall_lo` and `wall_hi` are the
/// exactly-known wall values — zero on a no-slip wall, the lid speed on the lid.
/// Ghia's outermost stations sit *on* the walls, so interpolating against the
/// interior nodes instead would report a boundary-sampling artefact as solver
/// error. This matches `phyz-lbm`'s own `tests/validation.rs`.
fn sample<F: Fn(usize) -> f64>(s: f64, n: usize, wall_lo: f64, wall_hi: f64, at: F) -> f64 {
    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + (b - a) * t.clamp(0.0, 1.0)
    }
    let p = s * n as f64 - 0.5;
    if p <= 0.0 {
        return lerp(wall_lo, at(0), (p + 0.5) / 0.5);
    }
    let last = (n - 1) as f64;
    if p >= last {
        return lerp(at(n - 1), wall_hi, (p - last) / 0.5);
    }
    let j = p.floor() as usize;
    lerp(at(j), at(j + 1), p - j as f64)
}

/// Worst absolute deviation from a reference profile, in units of the lid speed.
fn worst_deviation(got: &[f64], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .fold(0.0_f64, |a, (g, w)| a.max((g - w).abs()))
}

// ---------------------------------------------------------------------------

/// Run every lattice-Boltzmann validation.
pub fn run() -> Suite {
    let mut suite = Suite::new("Fluids — lattice Boltzmann D2Q9 (`phyz-lbm`)");

    // ---- 1. Poiseuille flow -------------------------------------------------
    let u_peak = 0.02;
    let ny = 21;
    let nu_ref = 0.05;

    let err = poiseuille_error(CollisionModel::default(), ny, nu_ref, u_peak);
    suite.push(
        Validation::new(
            "lbm.poiseuille.profile",
            "Plane Poiseuille flow: velocity profile vs u(y) = F y(H−y)/(2ρν)",
            CRATE,
            "Closed form for steady laminar channel flow (Batchelor §4.2), via \
             `phyz_lbm::analytic::poiseuille_force_driven`",
            "relative L2 profile error, 21-node channel, ν = 0.05, default collision",
            err,
            0.0,
            ErrorKind::Absolute,
            1e-3,
        )
        .note(
            "The body force is set from the analytic relation, so this tests the viscosity the \
             collision operator actually realises together with the wall treatment.",
        ),
    );

    // The viscosity sweep is the decisive test: plain BGK bounce-back places the
    // no-slip plane at a τ-dependent position, so its error moves with ν even
    // though the physical problem does not.
    let viscosities = [0.02_f64, 0.05, 0.2, 1.0];
    for model in [
        ("bgk", CollisionModel::Bgk),
        ("trt", CollisionModel::default()),
        ("mrt", CollisionModel::Mrt),
    ] {
        let errors: Vec<f64> = viscosities
            .iter()
            .map(|&nu| poiseuille_error(model.1, ny, nu, u_peak))
            .collect();
        let spread = errors.iter().fold(0.0_f64, |a, &b| a.max(b))
            - errors.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        suite.push(
            Validation::new(
                &format!("lbm.poiseuille.viscosity_independence.{}", model.0),
                &format!(
                    "Poiseuille error is independent of viscosity ({})",
                    model.0.to_uppercase()
                ),
                CRATE,
                "The analytic profile is exact at every ν once the force is rescaled to hold \
                 u_peak fixed, so a correct wall treatment gives a ν-independent error \
                 (Ginzburg & d'Humières 2003 on the TRT magic parameter)",
                &format!(
                    "spread of the relative L2 error across ν ∈ {{{}}}",
                    viscosities
                        .iter()
                        .map(|v| format!("{v}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                spread,
                0.0,
                ErrorKind::Absolute,
                1e-3,
            )
            .note(format!(
                "errors by ν: {}",
                viscosities
                    .iter()
                    .zip(&errors)
                    .map(|(v, e)| format!("ν={v} → {e:.3e}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        );
    }
    // BGK is a selectable option, not the shipped default, and its τ-dependent
    // wall position is the documented reason the default is TRT. Record the
    // measurement without a pass/fail claim against the default configuration.
    if let Some(v) = suite
        .entries
        .iter_mut()
        .find(|v| v.id == "lbm.poiseuille.viscosity_independence.bgk")
    {
        v.status = crate::report::Status::Report;
        v.notes.push(
            "Reported, not failed: `CollisionModel::Bgk` is not the crate default. Plain BGK \
             bounce-back places the no-slip plane at a τ-dependent position, so its error moves \
             with viscosity even though the physical problem does not — which is precisely why \
             `CollisionModel::default()` is TRT with Λ = 3/16. The TRT and MRT rows above are \
             the pass/fail claims."
                .to_string(),
        );
    }

    // ---- 2. Taylor–Green vortex --------------------------------------------
    let (n, u0, nu) = (48_usize, 0.02, 0.02);
    let nu_eff = taylor_green_nu_eff(n, u0, nu, 2_000);
    suite.push(
        Validation::new(
            "lbm.taylor_green.decay",
            "Taylor–Green vortex: kinetic-energy decay rate vs E(t) = E₀ exp(−2ν(k_x²+k_y²)t)",
            CRATE,
            "Taylor & Green (1937); exact unsteady Navier–Stokes solution on a periodic domain",
            "effective viscosity from the decay rate, input ν = 0.02, 48² lattice",
            nu_eff,
            nu,
            ErrorKind::Relative,
            0.02,
        )
        .note(
            "Measures the viscosity the collide/stream pair actually delivers, which is what \
             ties τ = 3ν + 1/2 to physical dissipation.",
        ),
    );

    let field_err = taylor_green_field_error(n, u0, nu, 1_000);
    suite.push(
        Validation::new(
            "lbm.taylor_green.field",
            "Taylor–Green vortex: velocity field keeps its analytic shape",
            CRATE,
            "Taylor & Green (1937), via `phyz_lbm::analytic::taylor_green_velocity`",
            "relative L2 error of the velocity field after 1000 steps, 48² lattice",
            field_err,
            0.0,
            ErrorKind::Absolute,
            0.02,
        )
        .note("Energy alone can be right while the field is wrong; this pins the shape."),
    );

    // Diffusive scaling: k and u₀ both scale as 1/n at fixed ν, so the physical
    // end state is identical on every grid and only the resolution changes.
    let mut samples = Vec::new();
    for &n in &[16_usize, 32, 64] {
        let u0 = 0.04 * 16.0 / n as f64;
        let steps = 100 * (n / 16) * (n / 16);
        samples.push((1.0 / n as f64, taylor_green_field_error(n, u0, 0.02, steps)));
    }
    let coarsest = samples.first().unwrap().1;
    let finest = samples.last().unwrap().1;
    // Two halvings at second order shrink the error by 16×; allow 1.5× slack.
    // Derived from the coarse-grid measurement, not fitted to the fine one.
    let bound = 1.5 * coarsest / 16.0;
    suite.push(
        Validation::new(
            "lbm.taylor_green.convergence",
            "Taylor–Green field error vanishes as Δx² under refinement",
            CRATE,
            "Chapman–Enskog: LBM recovers Navier–Stokes to second order in Δx",
            "relative L2 field error at 64² under diffusive scaling",
            finest,
            0.0,
            ErrorKind::Absolute,
            bound,
        )
        .with_convergence(Convergence::fit("Δx/L", samples, 2.0, 0.4))
        .note(format!(
            "Tolerance is 1.5 × (error at 16²) / 16 = {bound:.3e}, i.e. what two halvings at \
             second order must deliver from the measured coarse grid. The order fit is the \
             substantive claim; this bound only pins the constant."
        )),
    );

    // ---- 3. Lid-driven cavity ----------------------------------------------
    let n = 65;
    let (u, v, steps, residual) = cavity(n, 100.0, 0.1);
    let worst_u = worst_deviation(&u, &ghia::U_RE100);
    let worst_v = worst_deviation(&v, &ghia::V_RE100);

    suite.push(
        Validation::new(
            "lbm.cavity_re100.u",
            "Lid-driven cavity Re = 100: u on the vertical centreline",
            CRATE,
            "Ghia, Ghia & Shin, *J. Comput. Phys.* 48 (1982) 387, Table I",
            "worst |Δu| / u_lid over the 17 tabulated stations, 65² lattice",
            worst_u,
            0.0,
            ErrorKind::Absolute,
            0.04,
        )
        .note(format!(
            "{steps} steps, steady-state residual {residual:.2e}"
        ))
        .note(
            "Ghia's own data is a 129² multigrid solution; a few percent of the lid speed is the \
             discretisation difference at 65², not solver error. The tolerance is set to that \
             gap, and the vortex-position check below is the shape test that a loose profile \
             tolerance cannot provide.",
        ),
    );
    suite.push(Validation::new(
        "lbm.cavity_re100.v",
        "Lid-driven cavity Re = 100: v on the horizontal centreline",
        CRATE,
        "Ghia, Ghia & Shin, *J. Comput. Phys.* 48 (1982) 387, Table II",
        "worst |Δv| / u_lid over the 17 tabulated stations, 65² lattice",
        worst_v,
        0.0,
        ErrorKind::Absolute,
        0.04,
    ));

    // Where u changes sign on the vertical centreline locates the primary vortex.
    let crossing = ghia::Y
        .windows(2)
        .zip(u.windows(2))
        .find(|(_, uu)| uu[0] <= 0.0 && uu[1] > 0.0)
        .map(|(yy, uu)| yy[0] + (yy[1] - yy[0]) * (-uu[0]) / (uu[1] - uu[0]));
    suite.push(
        Validation::new(
            "lbm.cavity_re100.vortex_position",
            "Lid-driven cavity Re = 100: primary vortex position",
            CRATE,
            "Ghia et al. (1982) Table I — u changes sign at y/L ≈ 0.734 at Re = 100",
            "y/L of the centreline zero crossing",
            crossing.unwrap_or(f64::NAN),
            0.734,
            ErrorKind::Absolute,
            0.03,
        )
        .note(
            "A profile tolerance loose enough to absorb the 65²-vs-129² grid difference cannot \
             detect a misplaced vortex; this can.",
        ),
    );

    // ---- 4. Conservation on a closed domain ---------------------------------
    let mut closed =
        LatticeBoltzmann2D::new(48, 48, 0.05).with_boundaries(boundary::cavity_2d([0.08, 0.0]));
    closed.initialize_uniform(1.0, [0.0, 0.0]);
    let m0 = closed.total_mass();
    closed.run_to_steady_state(1e-7, 120_000, 500);
    let mass_error = (closed.total_mass() - m0).abs() / m0;
    suite.push(
        Validation::new(
            "lbm.cavity.mass_conservation",
            "A closed cavity conserves mass exactly",
            CRATE,
            "Bounce-back and moving-wall boundaries are mass-conserving by construction",
            "|Δm|/m₀ after running to steady state",
            mass_error,
            0.0,
            ErrorKind::Absolute,
            1e-9,
        )
        .note("Guards the whole boundary framework composing correctly on one domain."),
    );

    // A fully periodic domain with a body force must accelerate uniformly, with
    // no spurious momentum in the transverse direction.
    let mut drift = LatticeBoltzmann2D::new(16, 16, 0.05)
        .with_boundaries(Boundaries::periodic())
        .with_force([1e-5, 0.0]);
    drift.initialize_uniform(1.0, [0.0, 0.0]);
    drift.run(500);
    let uy = (0..16)
        .flat_map(|y| (0..16).map(move |x| (x, y)))
        .fold(0.0_f64, |a, (x, y)| a.max(drift.velocity(x, y)[1].abs()));
    suite.push(
        Validation::new(
            "lbm.forcing.transverse_isotropy",
            "Guo forcing injects momentum only along the applied force",
            CRATE,
            "A uniform force on a periodic domain produces uniform acceleration; any \
             transverse velocity is lattice anisotropy in the source term",
            "max |u_y| after 500 steps of a pure +x body force",
            uy,
            0.0,
            ErrorKind::Absolute,
            1e-15,
        )
        .note("Catches a mis-signed or mis-weighted direction in the forcing source term."),
    );

    suite
}
