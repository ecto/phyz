//! Is [`FixedPointSensitivity::apply_transpose`] the transpose of
//! [`FixedPointSensitivity::apply`]?
//!
//! Same oracle as `solver_level_adjoint_transpose.rs`, applied to the IFT
//! path: for every direction pair and every covector,
//!
//! ```text
//! <bar_f, apply(d_stationarity, d_mt_rel)>
//!   == <bar_res, d_stationarity> + <bar_mt, d_mt_rel>
//! ```
//!
//! exactly (to rounding), because the two sides are one linear map applied on
//! opposite sides. This is the pin that lets the reverse-mode smooth adjoint
//! contract a single covector through the IFT contact channel instead of
//! pushing one differential per input lane.
//!
//! The fixture is solved (not hand-classified), so the regimes exercised are
//! whatever the staged solver actually converges to; `mu` is chosen so the
//! set contains both sticking and sliding contacts, checked below rather than
//! assumed.

use phyz_contact::gradient::FixedPointSensitivity;
use phyz_contact::{ContactProblem, ContactRow, ContactSolverConfig, solve_contacts};
use phyz_math::Vec3;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as f64 / u32::MAX as f64) * 2.0 - 1.0
    }
}

/// Diagonally dominant Delassus with tangential free velocities large enough
/// that low-`mu` contacts slide while high-load ones stick.
fn problem(n: usize, mu: f64) -> ContactProblem {
    let dim = 3 * n;
    let mut delassus = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let (fi, fj) = (i as f64, j as f64);
            delassus[i * dim + j] = if i == j {
                2.0 + 0.35 * fi
            } else {
                0.12 * ((fi * 0.7).sin() * (fj * 0.7).sin())
            };
        }
    }
    let free_velocity: Vec<f64> = (0..dim)
        .map(|i| {
            let f = i as f64;
            if i % 3 == 0 {
                -0.4 - 0.05 * f
            } else {
                0.6 * (f * 1.3).cos()
            }
        })
        .collect();
    ContactProblem {
        n,
        delassus,
        free_velocity,
        rows: (0..n)
            .map(|c| ContactRow {
                mu,
                bias: 0.01 * (c as f64 + 1.0),
                ..Default::default()
            })
            .collect(),
        bodies: (0..n).map(|c| (c, usize::MAX)).collect(),
    }
}

#[test]
fn apply_transpose_pairs_with_apply() {
    for (n, mu) in [(1usize, 0.5), (4, 0.3), (6, 0.8)] {
        let p = problem(n, mu);
        let cfg = ContactSolverConfig::gradients();
        let sol = solve_contacts(&p, &cfg);
        assert!(sol.converged, "fixture n={n} mu={mu} must converge");
        let fps =
            FixedPointSensitivity::at(&p, &sol, &cfg).expect("converged fixture must linearize");

        let dim = 3 * n;
        let mut rng = Lcg::new(0x5eed ^ n as u64);
        for probe in 0..6 {
            let d_st: Vec<f64> = (0..dim).map(|_| rng.next()).collect();
            let d_mt: Vec<[f64; 2]> = (0..n).map(|_| [rng.next(), rng.next()]).collect();
            let bar_f: Vec<Vec3> = (0..n)
                .map(|_| Vec3::new(rng.next(), rng.next(), rng.next()))
                .collect();

            let df = fps.apply(&d_st, &d_mt);
            let lhs: f64 = bar_f
                .iter()
                .zip(&df)
                .map(|(b, d)| b.x * d.x + b.y * d.y + b.z * d.z)
                .sum();

            let (bar_res, bar_mt) = fps.apply_transpose(&bar_f);
            let rhs: f64 = bar_res.iter().zip(&d_st).map(|(b, d)| b * d).sum::<f64>()
                + bar_mt
                    .iter()
                    .zip(&d_mt)
                    .map(|(b, d)| b[0] * d[0] + b[1] * d[1])
                    .sum::<f64>();

            let scale: f64 = df
                .iter()
                .map(|d| d.x.abs() + d.y.abs() + d.z.abs())
                .sum::<f64>()
                .max(1e-12);
            assert!(
                (lhs - rhs).abs() <= 1e-12 * scale,
                "pairing broke: n={n} mu={mu} probe={probe}: {lhs} vs {rhs}"
            );
        }
    }
}

/// A non-finite parameter differential must not reroute the primal solve: the
/// differentiated re-execution has to follow the recorded (diff-free) solve
/// branch for branch, with the poison confined to the differential. This pins
/// the `newton_step_diff` repair — before it, a non-finite tangent column
/// returned `None` for the whole Newton step, silently changing the primal
/// iteration path relative to the recorded solve.
#[test]
fn non_finite_differential_does_not_change_the_primal_path() {
    let p = problem(5, 0.4);
    let cfg = ContactSolverConfig::simulation();
    let seed: Vec<Vec3> = vec![Vec3::zeros(); 5];
    let reference = phyz_contact::solve_contacts_warm(&p, &cfg, &seed);

    let dim = 3 * 5;
    let d_apr = vec![f64::NAN; dim * dim];
    let dc = vec![0.0; dim];
    let (replayed, df) =
        phyz_contact::contact_solve_differential(&p, &cfg, &seed, &[], &d_apr, &dc);

    assert_eq!(
        replayed.iterations, reference.iterations,
        "the differentiated replay took a different branch than the recorded solve"
    );
    for (a, b) in replayed.impulses.iter().zip(&reference.impulses) {
        assert_eq!((a.x, a.y, a.z), (b.x, b.y, b.z), "primal impulses moved");
    }
    // The poison stays in the differential, loudly.
    assert!(
        df.iter()
            .any(|d| !d.x.is_finite() || !d.y.is_finite() || !d.z.is_finite()),
        "a NaN parameter differential must surface as a NaN gradient, not vanish"
    );
}
