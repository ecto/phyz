//! Analytic validation benchmarks for the LBM solvers.
//!
//! Every case here compares against a closed-form solution or published
//! reference data — the crate previously checked only that mass was conserved
//! and that kinetic energy went down, neither of which can distinguish a
//! correct solver from a plausible one.
//!
//! The reference solutions live in [`phyz_lbm::analytic`] rather than inline
//! here, so the workspace-wide analytic validation suite can drive the same
//! formulas and reference tables instead of duplicating them.
//!
//! Cases:
//! - `poiseuille_*` — profile against the analytic parabola, and the
//!   viscosity-independence check that is the specific evidence the TRT
//!   upgrade worked.
//! - `taylor_green_*` — decay rate against the analytic exponential.
//! - `lid_driven_cavity_*` — centreline velocities against Ghia et al. (1982).
//! - `spatial_convergence_*` — observed order of accuracy under refinement.

use phyz_lbm::analytic::{self, ghia};
use phyz_lbm::boundary::{self, Boundaries, Boundary};
use phyz_lbm::collision::MAGIC_BOUNCE_BACK;
use phyz_lbm::{CollisionModel, LatticeBoltzmann2D};

/// Run force-driven plane Poiseuille flow to steady state and return the
/// relative L2 error of the velocity profile against the analytic parabola.
///
/// The body force is chosen to hold the peak velocity fixed as viscosity
/// varies, so the Mach number — and hence the compressibility error — is the
/// same for every viscosity. Any remaining spread across viscosities is the
/// boundary-condition defect and nothing else.
fn poiseuille_profile_error(collision: CollisionModel, ny: usize, nu: f64, u_peak: f64) -> f64 {
    let h = ny as f64;
    let g = analytic::poiseuille_force_for_peak(h, u_peak, nu);

    let mut lbm = LatticeBoltzmann2D::new(4, ny, nu)
        .with_collision(collision)
        .with_boundaries(boundary::channel_2d())
        .with_force([g, 0.0]);
    lbm.initialize_uniform(1.0, [0.0, 0.0]);

    let (steps, residual) = lbm.run_to_steady_state(1e-12, 400_000, 200);
    assert!(
        residual < 1e-11,
        "did not reach steady state: {steps} steps, residual {residual}"
    );

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

#[test]
fn poiseuille_profile_matches_the_analytic_parabola() {
    let err = poiseuille_profile_error(
        CollisionModel::Trt {
            magic: MAGIC_BOUNCE_BACK,
        },
        21,
        0.05,
        0.02,
    );
    assert!(err < 1e-9, "relative L2 error {err:e}");
}

/// The headline regression: TRT with `Λ = 3/16` makes the discretisation error
/// independent of viscosity, where BGK's error is a strong function of it.
///
/// With BGK the bounce-back wall sits at the correct half-node position only
/// when `(τ - 1/2)² = 3/16`, i.e. `ν ≈ 0.1443`; away from that the solver
/// silently solves a channel of the wrong width. Refining the *physics* should
/// never move the *geometry*, and after this change it does not.
#[test]
fn poiseuille_error_is_viscosity_independent_under_trt_but_not_bgk() {
    // τ = 0.56 … 3.5 — a 50× sweep in viscosity at fixed peak velocity.
    let viscosities = [0.02, 0.05, 0.2, 1.0];
    let (ny, u_peak) = (21usize, 0.02);

    let trt: Vec<f64> = viscosities
        .iter()
        .map(|&nu| {
            poiseuille_profile_error(
                CollisionModel::Trt {
                    magic: MAGIC_BOUNCE_BACK,
                },
                ny,
                nu,
                u_peak,
            )
        })
        .collect();
    let bgk: Vec<f64> = viscosities
        .iter()
        .map(|&nu| poiseuille_profile_error(CollisionModel::Bgk, ny, nu, u_peak))
        .collect();

    eprintln!("nu  = {viscosities:?}");
    eprintln!("TRT = {trt:?}");
    eprintln!("BGK = {bgk:?}");

    // TRT: exact at every viscosity.
    for (&nu, &e) in viscosities.iter().zip(&trt) {
        assert!(e < 1e-9, "TRT error {e:e} at nu = {nu} is not round-off");
    }

    // BGK: the error swings by orders of magnitude across the same sweep. If
    // this ever stops being true, BGK has silently changed behaviour and the
    // contrast this test is built on no longer holds.
    let bgk_min = bgk.iter().cloned().fold(f64::INFINITY, f64::min);
    let bgk_max = bgk.iter().cloned().fold(0.0, f64::max);
    assert!(
        bgk_max / bgk_min > 10.0,
        "expected BGK error to depend strongly on viscosity, got {bgk:?}"
    );
    assert!(
        bgk_max > 100.0 * trt.iter().cloned().fold(0.0, f64::max),
        "expected TRT to be dramatically more accurate than BGK: {trt:?} vs {bgk:?}"
    );
}

/// MRT inherits the same wall behaviour, because its `s_q` is chosen to
/// reproduce `Λ = 3/16`.
///
/// It is not bit-exact like TRT — the two ghost modes still relax at fixed
/// rates and leave an O(1e-6) residue — but the error is three to four orders
/// below BGK's and no longer swings with viscosity in any way that matters.
#[test]
fn poiseuille_error_is_negligible_under_mrt_at_every_viscosity() {
    let errors: Vec<f64> = [0.02, 0.05, 0.2, 1.0]
        .iter()
        .map(|&nu| poiseuille_profile_error(CollisionModel::Mrt, 21, nu, 0.02))
        .collect();
    eprintln!("MRT = {errors:?}");
    for &e in &errors {
        assert!(e < 5e-5, "MRT error {e:e} is too large: {errors:?}");
    }
}

/// Set up a Taylor–Green vortex on an `n × n` periodic grid.
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

/// Kinetic energy decays as `exp(-2ν k² t)` — twice the velocity decay rate,
/// since energy is quadratic. This is a direct test of the recovered viscosity.
#[test]
fn taylor_green_decays_at_the_analytic_rate() {
    let (n, u0, nu) = (48usize, 0.02, 0.02);
    let (mut lbm, k) = taylor_green(n, u0, nu);

    let ke0 = lbm.kinetic_energy();
    let steps = 2_000;
    lbm.run(steps);
    let ke1 = lbm.kinetic_energy();

    let measured = -((ke1 / ke0).ln()) / steps as f64;
    let expected = 2.0 * nu * (2.0 * k * k);
    eprintln!("decay: measured {measured:e}, analytic {expected:e}");
    assert!(
        (measured - expected).abs() / expected < 0.02,
        "decay rate {measured:e} vs analytic {expected:e}"
    );
}

/// The vortex must also keep its *shape*, not merely its energy budget.
#[test]
fn taylor_green_field_matches_the_analytic_solution() {
    let (n, u0, nu) = (48usize, 0.02, 0.02);
    let (mut lbm, k) = taylor_green(n, u0, nu);
    let steps = 1_000;
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
    let err = (num / den).sqrt();
    eprintln!("Taylor-Green relative L2 error: {err:e}");
    assert!(err < 0.02, "relative L2 error {err:e}");
}

/// Second-order spatial convergence, under diffusive scaling: the wave number
/// and velocity amplitude both scale as `1/n` while viscosity is held fixed, so
/// the Reynolds number and the physical end state are identical on every grid
/// and the only thing changing is the resolution.
#[test]
fn spatial_convergence_is_second_order() {
    let nu = 0.02;
    let cells = [16usize, 32, 64];
    let mut errors = Vec::new();

    for &n in &cells {
        let u0 = 0.04 * 16.0 / n as f64;
        let (mut lbm, k) = taylor_green(n, u0, nu);
        let steps = 100 * (n / 16) * (n / 16);
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
        errors.push((num / den).sqrt());
    }

    let order = analytic::convergence_order(&cells, &errors);
    eprintln!("grids {cells:?} -> errors {errors:?} (order {order:.3})");
    assert!(
        errors.windows(2).all(|w| w[1] < w[0]),
        "error must fall under refinement: {errors:?}"
    );
    assert!(
        order > 1.8,
        "expected second-order convergence, observed {order:.3} from {errors:?}"
    );
}

/// Sample the simulated `u` on the vertical centreline at Ghia's `y` stations.
fn cavity_centreline_u(lbm: &LatticeBoltzmann2D, u_lid: f64) -> Vec<f64> {
    let n = lbm.ny;
    let col = lbm.nx / 2;
    ghia::Y
        .iter()
        .map(|&yn| sample(yn, n, 0.0, u_lid, |j| lbm.velocity(col, j)[0]) / u_lid)
        .collect()
}

/// Sample the simulated `v` on the horizontal centreline at Ghia's `x` stations.
fn cavity_centreline_v(lbm: &LatticeBoltzmann2D, u_lid: f64) -> Vec<f64> {
    let n = lbm.nx;
    let row = lbm.ny / 2;
    ghia::X
        .iter()
        .map(|&xn| sample(xn, n, 0.0, 0.0, |i| lbm.velocity(i, row)[1]) / u_lid)
        .collect()
}

/// Linear interpolation of a nodal field at normalised position `s ∈ [0, 1]`.
///
/// Node `j` sits at `(j + 0.5)/n`, so in node coordinates `p = s·n - 0.5` the
/// walls lie at `p = -0.5` and `p = n - 0.5`. `wall_lo` and `wall_hi` are the
/// exactly-known wall values (zero on a no-slip wall, the lid speed on the lid).
fn sample<F: Fn(usize) -> f64>(s: f64, n: usize, wall_lo: f64, wall_hi: f64, at: F) -> f64 {
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

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Lid-driven cavity at Re = 100 against Ghia, Ghia & Shin (1982).
///
/// The tolerance reflects the grid: on 65² the discretisation error against a
/// 129² reference solution is a few percent of the lid speed. The point of the
/// test is that the profile *shape* — the primary vortex position and the
/// near-wall gradients — matches published data, which nothing in the crate
/// previously checked.
#[test]
fn lid_driven_cavity_matches_ghia_re100() {
    let n = 65usize;
    let u_lid = 0.1;
    let re = 100.0;
    let nu = u_lid * n as f64 / re;

    let mut lbm =
        LatticeBoltzmann2D::new(n, n, nu).with_boundaries(boundary::cavity_2d([u_lid, 0.0]));
    lbm.initialize_uniform(1.0, [0.0, 0.0]);
    let (steps, residual) = lbm.run_to_steady_state(1e-8, 120_000, 500);
    eprintln!("cavity: {steps} steps, residual {residual:e}");
    assert!(residual < 1e-7, "cavity did not converge: {residual:e}");

    let u = cavity_centreline_u(&lbm, u_lid);
    let v = cavity_centreline_v(&lbm, u_lid);

    let mut worst_u = 0.0f64;
    for (i, (&got, &want)) in u.iter().zip(&ghia::U_RE100).enumerate() {
        eprintln!("u  y={:.4}  sim {got:+.5}  ghia {want:+.5}", ghia::Y[i]);
        worst_u = worst_u.max((got - want).abs());
    }
    let mut worst_v = 0.0f64;
    for (i, (&got, &want)) in v.iter().zip(&ghia::V_RE100).enumerate() {
        eprintln!("v  x={:.4}  sim {got:+.5}  ghia {want:+.5}", ghia::X[i]);
        worst_v = worst_v.max((got - want).abs());
    }
    eprintln!("worst |Δu| = {worst_u:.4}, worst |Δv| = {worst_v:.4}");

    assert!(worst_u < 0.04, "u profile off by {worst_u:.4}");
    assert!(worst_v < 0.04, "v profile off by {worst_v:.4}");

    // The primary vortex sits below and left of centre at Re = 100: u on the
    // vertical centreline crosses zero at roughly y = 0.73.
    let crossing = ghia::Y
        .windows(2)
        .zip(u.windows(2))
        .find(|(_, uu)| uu[0] <= 0.0 && uu[1] > 0.0)
        .map(|(yy, uu)| yy[0] + (yy[1] - yy[0]) * (-uu[0]) / (uu[1] - uu[0]))
        .expect("u must change sign on the centreline");
    eprintln!("centreline zero crossing at y = {crossing:.4}");
    assert!(
        (crossing - 0.734).abs() < 0.03,
        "primary vortex misplaced: crossing at y = {crossing:.4}"
    );
}

/// A closed cavity leaks no mass and reaches a genuine steady state — a cheap
/// guard on the whole boundary framework working together.
#[test]
fn cavity_is_closed_and_steady() {
    let mut lbm =
        LatticeBoltzmann2D::new(48, 48, 0.05).with_boundaries(boundary::cavity_2d([0.08, 0.0]));
    lbm.initialize_uniform(1.0, [0.0, 0.0]);
    let m0 = lbm.total_mass();
    let (steps, residual) = lbm.run_to_steady_state(1e-7, 120_000, 500);
    eprintln!("cavity: {steps} steps, residual {residual:e}");
    assert!(residual < 1e-6, "residual {residual:e}");
    assert!((lbm.total_mass() - m0).abs() / m0 < 1e-9);
}

/// A channel driven by an inlet velocity and a pressure outlet develops the
/// same parabolic profile as the force-driven case, confirming that Zou–He and
/// bounce-back compose correctly on the same domain.
///
/// The flow settles within a few thousand steps; thereafter a very slow mode
/// seeded at the four corners — where a Zou–He face meets a bounce-back wall
/// and the corner node is resolved by the wall — grows in the fifth significant
/// figure. That is a known limitation of corner treatment by face precedence,
/// not of the interior scheme, so this test converges the flow and checks the
/// profile rather than demanding a round-off residual.
#[test]
fn inlet_outlet_channel_develops_poiseuille_flow() {
    let (nx, ny) = (96usize, 21usize);
    let nu = 0.08;
    let u_in = 0.02;

    let bc = boundary::inlet_outlet_channel_2d([u_in, 0.0], 1.0);
    let mut lbm = LatticeBoltzmann2D::new(nx, ny, nu).with_boundaries(bc);
    lbm.initialize_uniform(1.0, [u_in, 0.0]);
    let max_steps = 20_000;
    let (steps, residual) = lbm.run_to_steady_state(1e-5, max_steps, 500);
    eprintln!("channel: {steps} steps, residual {residual:e}");
    assert!(
        steps < max_steps,
        "channel never settled: residual {residual:e}"
    );

    // Far downstream the profile is fully developed. Mass conservation fixes
    // the mean, so compare the shape against a parabola of the same flow rate.
    let col = nx - 10;
    let profile: Vec<f64> = (0..ny).map(|y| lbm.velocity(col, y)[0]).collect();
    let mean = profile.iter().sum::<f64>() / ny as f64;
    let h = ny as f64;
    // Mean of the analytic parabola is 2/3 of its peak.
    let peak = 1.5 * mean;
    let g = analytic::poiseuille_force_for_peak(h, peak, nu);

    let mut worst = 0.0f64;
    for (y, &got) in profile.iter().enumerate() {
        let want = analytic::poiseuille_force_driven(y as f64, -0.5, h - 0.5, g, nu);
        worst = worst.max((got - want).abs() / peak);
    }
    eprintln!("developed profile: mean {mean:e}, peak {peak:e}, worst rel {worst:e}");
    assert!(worst < 0.02, "profile deviates by {worst:e} of peak");
}

/// Smagorinsky must not perturb a flow the grid already resolves. A model that
/// changes the answer on a laminar benchmark is adding error, not closure.
#[test]
fn smagorinsky_does_not_disturb_laminar_poiseuille() {
    use phyz_lbm::Turbulence;

    let (ny, nu, u_peak) = (21usize, 0.05, 0.02);
    let h = ny as f64;
    let g = analytic::poiseuille_force_for_peak(h, u_peak, nu);

    let mut lbm = LatticeBoltzmann2D::new(4, ny, nu)
        .with_boundaries(boundary::channel_2d())
        .with_force([g, 0.0])
        .with_turbulence(Turbulence::Smagorinsky { cs: 0.1 });
    lbm.initialize_uniform(1.0, [0.0, 0.0]);
    lbm.run_to_steady_state(1e-12, 400_000, 200);

    let mut worst = 0.0f64;
    for y in 0..ny {
        let want = analytic::poiseuille_force_driven(y as f64, -0.5, h - 0.5, g, nu);
        worst = worst.max((lbm.velocity(0, y)[0] - want).abs() / u_peak);
    }
    eprintln!("LES on laminar Poiseuille: worst relative error {worst:e}");
    assert!(worst < 0.05, "LES perturbed a laminar profile by {worst:e}");
}

/// Boundaries declared once must be enforced on every step without the caller
/// re-applying them — the specific trap the old API had.
#[test]
fn declared_walls_are_enforced_without_manual_reapplication() {
    let bc = Boundaries::periodic().set_axis(1, Boundary::NoSlip);
    let mut lbm = LatticeBoltzmann2D::new(8, 24, 0.05)
        .with_boundaries(bc)
        .with_force([1e-6, 0.0]);
    lbm.initialize_uniform(1.0, [0.0, 0.0]);
    lbm.run(3_000);

    // With walls, the profile is sheared: centre moves, near-wall nodes lag.
    let centre = lbm.velocity(0, 12)[0];
    let near_wall = lbm.velocity(0, 0)[0];
    assert!(centre > 0.0, "flow should be driven: {centre:e}");
    assert!(
        near_wall < 0.35 * centre,
        "wall did not retard the flow: near-wall {near_wall:e} vs centre {centre:e}"
    );

    // Without walls the same force gives plug flow — proof the difference is
    // the boundary condition and not the forcing.
    let mut free = LatticeBoltzmann2D::new(8, 24, 0.05).with_force([1e-6, 0.0]);
    free.initialize_uniform(1.0, [0.0, 0.0]);
    free.run(3_000);
    let a = free.velocity(0, 0)[0];
    let b = free.velocity(0, 12)[0];
    assert!((a - b).abs() / b < 1e-9, "unbounded flow should be uniform");
}
