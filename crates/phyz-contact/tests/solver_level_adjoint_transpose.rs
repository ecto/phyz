//! Is the reverse-mode contact-solve differential the transpose of the forward
//! one?
//!
//! There is no finite difference here and there does not need to be. The
//! forward mode next door ([`contact_solve_differential`]) is already validated
//! against central differences in `solver_level_adjoint.rs`; what remains is
//! whether the reverse mode is the *same linear map, transposed*. That is a
//! statement about two functions and it has an exact oracle:
//!
//! ```text
//! <bar_f, J (d_initial, d_apr, dc)>  ==  <J^T bar_f, (d_initial, d_apr, dc)>
//! ```
//!
//! for every choice of both sides. A finite difference is accurate to about
//! `1e-9`; this identity is accurate to rounding, so it catches errors three
//! orders of magnitude smaller — which matters, because the terms most likely
//! to be dropped in a transpose (the `d det` quotient term, the `t_hat`
//! rotation in the sliding pin rows) are small ones.
//!
//! Both sides are probed with several random covectors and several random
//! directions rather than one of each. A transpose bug that happened to be
//! orthogonal to a single `bar_f` is not a rare accident: the projector in the
//! disc clamp annihilates a whole direction per sliding contact, so one probe
//! genuinely can miss.
//!
//! The fixtures deliberately include a solve that does **not** converge. That
//! is the entire point of a solver-level adjoint — an implicit-function-theorem
//! gradient has no anchor at a truncated iterate, and a transpose that only
//! worked at a fixed point would be no more useful than the one already
//! shipped.

use phyz_contact::{
    ContactProblem, ContactRow, ContactSolverConfig, contact_solve_differential,
    contact_solve_differential_transpose, solve_contacts_warm,
};
use phyz_math::Vec3;

/// A 64-bit linear congruential generator, so the probes are random-looking but
/// byte-reproducible.
///
/// `rand` is not a dev-dependency of this crate and adding one to shake a
/// linear map would be a poor trade. The multiplier is Knuth's MMIX constant;
/// the top 32 bits are the ones used, because an LCG's low bits have short
/// periods and a probe vector whose components cycle is a probe vector that
/// tests fewer directions than it looks like it does.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }

    /// Uniform on `[-1, 1)`.
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as f64 / u32::MAX as f64) * 2.0 - 1.0
    }
}

/// The same deterministic, diagonally dominant Delassus fixture the forward
/// test uses, so a failure here can be compared against a passing case there
/// without wondering whether the problems differ.
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
                0.15 * (f * 1.3).cos()
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

/// A rank-deficient manifold: `k` contacts on one body whose Delassus rows are
/// built from only three independent generators, so `A` is singular and the
/// null space the active-set Newton stage exists for is genuinely there.
///
/// This is the case that separates a transpose that walks the Gauss-Seidel
/// ordering correctly from one that treats the sweep as a per-contact diagonal
/// map. On a redundant manifold the load is passed between contacts through the
/// off-diagonal blocks on every sweep, so the scatter term dominates.
fn redundant_problem(n: usize, mu: f64) -> ContactProblem {
    let dim = 3 * n;
    // Three generators; every row of `g` is a combination of them, so
    // `A = G^T G` has rank at most 3 before regularization.
    let gens: Vec<Vec<f64>> = (0..3)
        .map(|g| {
            (0..dim)
                .map(|i| ((i * 7 + g * 13) as f64 * 0.41).sin())
                .collect()
        })
        .collect();
    let mut delassus = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut acc = 0.0;
            for g in &gens {
                acc += g[i] * g[j];
            }
            // A whisker of diagonal so the normal solve has a positive `a_nn`;
            // far too little to hide the redundancy.
            delassus[i * dim + j] = acc + if i == j { 0.02 } else { 0.0 };
        }
    }
    let free_velocity: Vec<f64> = (0..dim)
        .map(|i| {
            if i % 3 == 0 {
                -0.5
            } else {
                0.2 * ((i * 5) as f64 * 0.9).sin()
            }
        })
        .collect();
    ContactProblem {
        n,
        delassus,
        free_velocity,
        rows: (0..n)
            .map(|_| ContactRow {
                mu,
                bias: 0.005,
                ..Default::default()
            })
            .collect(),
        // One shared body, so `PerBody` coupling would keep every block — the
        // manifold is a single foot, not a set of independent ground contacts.
        bodies: (0..n).map(|_| (0, usize::MAX)).collect(),
    }
}

/// A random symmetric `d(A + R)` and a random `d(b - e_n bias)`.
///
/// Symmetry is not cosmetic: an asymmetric `dA` is not a direction the Delassus
/// operator can move in, so testing along one would validate index arithmetic
/// the physics never exercises — and, worse, would let a transpose that
/// accidentally symmetrized `bar_apr` pass. Because the *directions* are
/// symmetric, the identity only pins down `bar_apr`'s symmetric part, so the
/// asymmetric probe below is kept as a separate test rather than folded in.
fn direction(rng: &mut Lcg, n: usize, symmetric: bool) -> (Vec<f64>, Vec<f64>) {
    let dim = 3 * n;
    let mut d_apr = vec![0.0; dim * dim];
    if symmetric {
        for i in 0..dim {
            for j in 0..=i {
                let v = rng.next();
                d_apr[i * dim + j] = v;
                d_apr[j * dim + i] = v;
            }
        }
    } else {
        for slot in d_apr.iter_mut() {
            *slot = rng.next();
        }
    }
    let dc = (0..dim).map(|_| rng.next()).collect();
    (d_apr, dc)
}

/// The dot-product identity, evaluated once for one `(bar_f, d_initial, d_apr,
/// dc)` quadruple. Returns the relative disagreement.
///
/// Relative to the *magnitude of the terms*, not to the result: the two sides
/// are sums of thousands of products that cancel heavily, so the residual has
/// to be measured against how big the summands got or a well-cancelled pair
/// would look like a failure at `1e-1` while being correct to the last bit.
fn identity_error(
    p: &ContactProblem,
    cfg: &ContactSolverConfig,
    initial: &[Vec3],
    bar_f: &[Vec3],
    d_initial: &[Vec3],
    d_apr: &[f64],
    dc: &[f64],
) -> f64 {
    let (_, df) = contact_solve_differential(p, cfg, initial, d_initial, d_apr, dc);
    let lhs: f64 = bar_f
        .iter()
        .zip(&df)
        .map(|(b, d)| b.x * d.x + b.y * d.y + b.z * d.z)
        .sum();

    let (_, adj) = contact_solve_differential_transpose(p, cfg, initial, bar_f);
    let mut rhs = 0.0;
    let mut scale = 0.0;
    for (a, d) in adj.bar_apr.iter().zip(d_apr) {
        rhs += a * d;
        scale += (a * d).abs();
    }
    for (a, d) in adj.bar_c.iter().zip(dc) {
        rhs += a * d;
        scale += (a * d).abs();
    }
    for (a, d) in adj.bar_initial.iter().zip(d_initial) {
        rhs += a.x * d.x + a.y * d.y + a.z * d.z;
        scale += (a.x * d.x).abs() + (a.y * d.y).abs() + (a.z * d.z).abs();
    }
    (lhs - rhs).abs() / scale.max(1e-12)
}

/// Sweep several probes on both sides and return the worst relative error.
///
/// `d_initial` is non-zero on every probe. That is the channel a warm start
/// actually uses and the one the reverse walk produces last, so a sign error
/// there would survive a cold-start-only test unnoticed.
fn worst_identity_error(
    p: &ContactProblem,
    cfg: &ContactSolverConfig,
    initial: &[Vec3],
    seed: u64,
    symmetric: bool,
) -> f64 {
    let n = p.n;
    let mut rng = Lcg::new(seed);
    let mut worst: f64 = 0.0;
    for _ in 0..4 {
        let bar_f: Vec<Vec3> = (0..n)
            .map(|_| Vec3::new(rng.next(), rng.next(), rng.next()))
            .collect();
        let d_initial: Vec<Vec3> = (0..n)
            .map(|_| Vec3::new(rng.next(), rng.next(), rng.next()) * 0.3)
            .collect();
        for _ in 0..3 {
            let (d_apr, dc) = direction(&mut rng, n, symmetric);
            worst = worst.max(identity_error(
                p, cfg, initial, &bar_f, &d_initial, &d_apr, &dc,
            ));
        }
    }
    worst
}

/// Every fixture also has to prove the taped primal is the shipped primal.
///
/// The whole construction rests on the reverse pass linearizing the solve that
/// actually ran. If taping perturbed the iteration — an extra clone that
/// reordered nothing is fine, a changed branch is not — the transpose would be
/// the exact adjoint of a solver nobody uses. Bitwise equality is the right bar
/// here rather than a tolerance: the taped run performs the *same* float
/// operations in the same order, so anything short of bit-identical is a real
/// divergence.
fn assert_primal_untouched(
    p: &ContactProblem,
    cfg: &ContactSolverConfig,
    initial: &[Vec3],
    label: &str,
) {
    let plain = solve_contacts_warm(p, cfg, initial);
    let bar_f = vec![Vec3::new(1.0, 0.0, 0.0); p.n];
    let (taped, _) = contact_solve_differential_transpose(p, cfg, initial, &bar_f);
    assert_eq!(
        taped.iterations, plain.iterations,
        "{label}: taping changed the iteration count"
    );
    assert_eq!(
        taped.converged, plain.converged,
        "{label}: taping changed convergence"
    );
    assert_eq!(
        taped.residual.to_bits(),
        plain.residual.to_bits(),
        "{label}: taping changed the residual"
    );
    for (a, b) in taped.impulses.iter().zip(&plain.impulses) {
        assert_eq!(a.x.to_bits(), b.x.to_bits(), "{label}: primal moved");
        assert_eq!(a.y.to_bits(), b.y.to_bits(), "{label}: primal moved");
        assert_eq!(a.z.to_bits(), b.z.to_bits(), "{label}: primal moved");
    }
}

fn check(p: ContactProblem, cfg: ContactSolverConfig, seed: u64, tol: f64, label: &str) -> f64 {
    assert_primal_untouched(&p, &cfg, &[], label);
    let worst = worst_identity_error(&p, &cfg, &[], seed, true);
    assert!(
        worst < tol,
        "{label}: <bar_f, J d> and <J^T bar_f, d> disagree by {worst:.3e} relative"
    );
    worst
}

/// **Converged, mixed sticking and sliding.** The baseline: a solve that
/// reaches a KKT point, with both friction regimes represented and the Newton
/// stage on the path.
#[test]
fn the_transpose_matches_forward_mode_at_a_converged_solve() {
    let cfg = ContactSolverConfig {
        tolerance: 1e-14,
        max_iterations: 4000,
        ..ContactSolverConfig::gradients()
    };
    let worst = check(problem(5, 0.7), cfg, 0x5eed_0001, 1e-12, "converged");
    println!("converged, sticking and sliding mixed: worst relative error {worst:.3e}");
}

/// **Everything sliding.** A low `mu` puts every contact on the cone boundary,
/// so the disc clamp's `s (I - t_hat t_hat^T)` projector is on every tangential
/// row. That projector is symmetric, which makes it exactly the term a
/// transpose can get wrong without any asymmetry to give it away — the error
/// shows up only in the `mu t_hat d f_n` coupling that rides alongside it.
#[test]
fn the_transpose_matches_forward_mode_when_every_contact_slides() {
    let cfg = ContactSolverConfig {
        tolerance: 1e-14,
        max_iterations: 4000,
        ..ContactSolverConfig::gradients()
    };
    let worst = check(problem(4, 0.02), cfg, 0x5eed_0002, 1e-12, "all sliding");
    println!("all sliding: worst relative error {worst:.3e}");
}

/// **A separating contact.** `f_n = max(0, .)` pins one contact's whole row.
/// The forward returns zero there; the transpose has to refuse to let anything
/// flow *back* through it, which is a different statement and a different line
/// of code.
#[test]
fn the_transpose_matches_forward_mode_with_a_separating_contact() {
    let cfg = ContactSolverConfig {
        tolerance: 1e-14,
        max_iterations: 4000,
        ..ContactSolverConfig::gradients()
    };
    let mut p = problem(4, 0.6);
    // Drive contact 2 firmly apart: its normal impulse is zero at the solution.
    p.free_velocity[6] = 3.0;
    let sol = solve_contacts_warm(&p, &cfg, &[]);
    assert!(
        sol.impulses[2].x == 0.0,
        "the fixture needs contact 2 separating; it carries {:e}",
        sol.impulses[2].x
    );
    let worst = check(p, cfg, 0x5eed_0003, 1e-12, "separating");
    println!("separating contact: worst relative error {worst:.3e}");
}

/// **Sticking, all of it.** A large `mu` keeps every contact strictly inside
/// the cone, so the clamp never fires and the whole map is the `2x2` tangential
/// solve plus the scatter. Isolating that is worth a fixture of its own: it is
/// the case where a missing `d det` term has nothing else to hide behind.
#[test]
fn the_transpose_matches_forward_mode_when_every_contact_sticks() {
    let cfg = ContactSolverConfig {
        tolerance: 1e-14,
        max_iterations: 4000,
        ..ContactSolverConfig::gradients()
    };
    let worst = check(problem(4, 8.0), cfg, 0x5eed_0004, 1e-12, "all sticking");
    println!("all sticking: worst relative error {worst:.3e}");
}

/// **A redundant, rank-deficient manifold.** Eight contacts sharing three
/// independent Delassus generators — the coplanar-foot case active-set Newton
/// exists for. The load moves between contacts through the off-diagonal blocks
/// on every sweep, so the Gauss-Seidel scatter (the term that has to run
/// contacts in reverse order) carries most of the map here.
#[test]
fn the_transpose_matches_forward_mode_on_a_redundant_manifold() {
    let cfg = ContactSolverConfig {
        tolerance: 1e-13,
        max_iterations: 2000,
        ..ContactSolverConfig::gradients()
    };
    let worst = check(
        redundant_problem(8, 0.5),
        cfg,
        0x5eed_0005,
        1e-11,
        "redundant manifold",
    );
    println!("redundant manifold: worst relative error {worst:.3e}");
}

/// **Two sweeps and nothing else.** The case the whole solver-level
/// construction is for. `tolerance: 0.0` and a cap of two guarantee the solve
/// stops nowhere near a fixed point, and with `newton: false` the map is purely
/// the PGS recursion — so if the reverse walk mishandled the in-place
/// Gauss-Seidel ordering, this is where it shows.
#[test]
fn the_transpose_matches_forward_mode_at_a_deliberately_truncated_solve() {
    let cfg = ContactSolverConfig {
        tolerance: 0.0,
        max_iterations: 2,
        newton: false,
        ..ContactSolverConfig::gradients()
    };
    let p = problem(5, 0.7);
    let sol = solve_contacts_warm(&p, &cfg, &[]);
    assert!(!sol.converged, "the fixture must not converge");
    let worst = check(p, cfg, 0x5eed_0006, 1e-12, "two sweeps, unconverged");
    println!("two sweeps, unconverged: worst relative error {worst:.3e}");
}

/// **Truncated with the Newton stage live.** Forty iterations, no tolerance:
/// enough for several accepted Newton proposals and their line searches, not
/// enough to converge. This is the fixture that exercises `K^T`, the clamp on
/// the raw Newton iterate, and the slip-direction rotation in the pin rows —
/// the three pieces of the transpose that have no counterpart in the PGS path.
#[test]
fn the_transpose_matches_forward_mode_through_the_newton_stage() {
    let cfg = ContactSolverConfig {
        tolerance: 0.0,
        max_iterations: 40,
        ..ContactSolverConfig::gradients()
    };
    let p = problem(5, 0.7);
    let sol = solve_contacts_warm(&p, &cfg, &[]);
    assert!(!sol.converged, "the fixture must not converge");
    let worst = check(p, cfg, 0x5eed_0007, 1e-12, "40 iterations with Newton");
    println!("40 iterations with Newton, unconverged: worst relative error {worst:.3e}");
}

/// **Warm-started.** A non-zero seed puts the `normals_only` warm-up phase on
/// the path with tangential impulses to hold, and makes `bar_initial` a channel
/// with real content rather than a formality. The seed is deliberately outside
/// the friction cone on one contact, which the first sweep projects back — a
/// branch the transpose has to follow like any other.
#[test]
fn the_transpose_matches_forward_mode_from_a_warm_start() {
    let cfg = ContactSolverConfig {
        tolerance: 1e-14,
        max_iterations: 4000,
        ..ContactSolverConfig::gradients()
    };
    let p = problem(4, 0.7);
    let initial = vec![
        Vec3::new(0.2, 0.05, -0.03),
        Vec3::new(0.4, 0.9, 0.7),
        Vec3::new(0.1, 0.0, 0.0),
        Vec3::new(0.3, -0.02, 0.06),
    ];
    assert_primal_untouched(&p, &cfg, &initial, "warm start");
    let worst = worst_identity_error(&p, &cfg, &initial, 0x5eed_0008, true);
    assert!(worst < 1e-12, "warm start: disagreement {worst:.3e}");
    println!("warm start: worst relative error {worst:.3e}");
}

/// **Asymmetric `d(A + R)`.** The other fixtures probe only symmetric
/// directions, which pins down `bar_apr`'s symmetric part and says nothing
/// about the rest. A transpose that wrote `(bar_apr + bar_apr^T)/2` — an easy
/// thing to do by accident when mirroring the `d_apr` reads — would pass every
/// test above and fail this one. The solve is unphysical along such a
/// direction, but the linear map is not, and it is the map under test.
#[test]
fn the_transpose_pins_down_the_asymmetric_part_of_d_apr() {
    let cfg = ContactSolverConfig {
        tolerance: 1e-14,
        max_iterations: 4000,
        ..ContactSolverConfig::gradients()
    };
    let p = problem(4, 0.7);
    let worst = worst_identity_error(&p, &cfg, &[], 0x5eed_0009, false);
    assert!(
        worst < 1e-12,
        "asymmetric direction: disagreement {worst:.3e}"
    );
    println!("asymmetric d_apr: worst relative error {worst:.3e}");
}

/// **Zero contacts.** The empty problem is a real code path — a body in flight
/// produces one every step — and the transpose has to return empty covectors
/// rather than index into nothing.
#[test]
fn the_transpose_handles_an_empty_problem() {
    let cfg = ContactSolverConfig::gradients();
    let p = ContactProblem {
        n: 0,
        delassus: Vec::new(),
        free_velocity: Vec::new(),
        rows: Vec::new(),
        bodies: Vec::new(),
    };
    let (sol, adj) = contact_solve_differential_transpose(&p, &cfg, &[], &[]);
    assert!(sol.converged);
    assert!(adj.bar_apr.is_empty());
    assert!(adj.bar_c.is_empty());
    assert!(adj.bar_initial.is_empty());
}
