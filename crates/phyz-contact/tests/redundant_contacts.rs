//! Redundant coplanar contacts.
//!
//! A flat body resting on many coplanar points is *rank deficient*: eight
//! contacts under two feet supply 24 impulse components to constrain 6 body
//! DoF, so the Delassus operator `A = J M^-1 J^T` is singular and the impulse
//! split is not determined by `A` alone. This is the ordinary case for a
//! humanoid in double support — a Booster K1 standing on two flat feet reports
//! eight coplanar contacts — and it used to make the projected Gauss-Seidel
//! sweep run to its iteration cap without ever reaching tolerance, with a
//! saturated friction impulse (`||f_t|| = mu f_n`) at every contact despite
//! there being no slip anywhere to oppose.
//!
//! What these tests pin down:
//!
//! - the sweep converges on a redundant manifold, not just a determined one;
//! - the normal impulses sum to the weight they carry;
//! - friction stays at zero when nothing slips, at any redundancy.
//!
//! The four-contact case is kept alongside because it converged before this
//! was fixed, and must keep doing so.

use phyz_collision::Collision;
use phyz_contact::{
    ContactAssembly, ContactMaterial, ContactSolverConfig, assemble, solve_contacts,
};
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Model, ModelBuilder, State};

const MASS: f64 = 10.0;
const DT: f64 = 2e-3;
/// Impulse gravity delivers over one step — what the contacts must carry.
const WEIGHT: f64 = MASS * GRAVITY * DT;

fn slab() -> (Model, State) {
    let model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(DT)
        .add_free_body(
            "slab",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                MASS,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(0.3, 0.3, 0.5)),
            ),
        )
        .build();
    let mut state = model.default_state();
    state.body_xform[0] = SpatialTransform::identity();
    (model, state)
}

/// `n` coplanar contacts on a ring under the body, all sharing one normal.
///
/// A ring rather than a grid so the configuration is exactly symmetric for
/// every `n` in the sweep: the correct answer is then "every contact carries
/// `1/n` of the load and no friction at all", with no dependence on `n`.
fn ring(n: usize) -> Vec<Collision> {
    (0..n)
        .map(|k| {
            let t = k as f64 * std::f64::consts::TAU / n as f64;
            Collision {
                body_i: 0,
                body_j: usize::MAX,
                contact_point: Vec3::new(0.1 * t.cos(), 0.1 * t.sin(), -0.05),
                contact_normal: Vec3::z(),
                // Deliberately shallow but non-zero: a resting contact in a
                // stepping loop always carries some penetration, and it is the
                // solref recovery bias on top of the weight that makes the
                // total slightly exceed `WEIGHT`.
                penetration_depth: 5e-5,
            }
        })
        .collect()
}

struct Solved {
    assembly: ContactAssembly,
    impulses: Vec<Vec3>,
    converged: bool,
    iterations: usize,
}

fn solve(n: usize, max_iterations: usize) -> Solved {
    let (model, state) = slab();
    let contacts = ring(n);
    let mut free_qd = DVec::zeros(model.nv);
    free_qd[5] = -GRAVITY * DT;
    let cfg = ContactSolverConfig {
        max_iterations,
        ..ContactSolverConfig::simulation()
    };
    let assembly = assemble(
        &model,
        &state,
        &contacts,
        &[ContactMaterial::default()],
        &free_qd,
        DT,
        &cfg,
    );
    let sol = solve_contacts(&assembly.problem, &cfg);
    Solved {
        assembly,
        impulses: sol.impulses,
        converged: sol.converged,
        iterations: sol.iterations,
    }
}

/// The reported failure, reduced to one rigid body and no robot: eight
/// coplanar contacts must converge and must actually carry the weight.
#[test]
fn eight_coplanar_contacts_converge_and_carry_the_weight() {
    let s = solve(8, 500);
    assert!(
        s.converged,
        "eight coplanar contacts must converge (used {} iterations)",
        s.iterations
    );

    let total: f64 = s.impulses.iter().map(|f| f.x).sum();
    // Soft contact never lands exactly on the weight: the impedance holds a
    // little back and the penetration recovery bias adds a little on top. The
    // band is what "carries the weight" means for a regularized contact — the
    // pre-fix solve was outside it, and on stiffer manifolds not finite at all.
    assert!(
        (total - WEIGHT).abs() / WEIGHT < 0.1,
        "normal impulses sum to {total}, weight is {WEIGHT}"
    );

    // And the body neither sinks nor is launched: the post-solve vertical
    // velocity is separating but under the recovery bias.
    let dv = s.assembly.velocity_delta(&s.impulses);
    let v_after = -GRAVITY * DT + dv[5];
    let bias = s.assembly.problem.rows[0].bias;
    assert!(
        v_after > 0.0 && v_after < bias,
        "post-solve vertical velocity {v_after} outside (0, {bias})"
    );
}

/// Nothing slides, so nothing may carry friction — at any redundancy.
///
/// This is the assertion that fails loudest on the old sweep: it left every
/// contact pinned to the edge of its friction disc, `||f_t|| = mu f_n`, with
/// the tangential free velocity identically zero. Those impulses cancel in the
/// net wrench, which is why the bug hid, but they are what the solve then
/// wandered along instead of terminating.
#[test]
fn a_resting_body_carries_no_friction() {
    for n in [4, 8, 16, 32] {
        let s = solve(n, 5000);
        assert!(s.converged, "n={n} did not converge");
        let max_ft = s
            .impulses
            .iter()
            .map(|f| (f.y * f.y + f.z * f.z).sqrt())
            .fold(0.0f64, f64::max);
        assert!(
            max_ft < 1e-9,
            "n={n}: resting contact carries tangential impulse {max_ft}"
        );
    }
}

/// Redundancy sweep. Convergence must not silently regress as the manifold
/// gets more degenerate, and the load must stay shared evenly — the symmetric
/// configuration has no reason to prefer any contact.
#[test]
fn redundancy_sweep() {
    for n in [4, 8, 16, 32] {
        let s = solve(n, 5000);
        assert!(
            s.converged,
            "n={n} coplanar contacts did not converge in {} iterations",
            s.iterations
        );

        let total: f64 = s.impulses.iter().map(|f| f.x).sum();
        assert!(
            (total - WEIGHT).abs() / WEIGHT < 0.1,
            "n={n}: normal impulses sum to {total}, weight is {WEIGHT}"
        );

        let share = total / n as f64;
        for (i, f) in s.impulses.iter().enumerate() {
            assert!(
                (f.x - share).abs() / share < 1e-6,
                "n={n}: contact {i} carries {} of an even share {share}",
                f.x
            );
            assert!(f.x > 0.0, "n={n}: contact {i} carries no load");
        }
    }
}

/// The four-contact stance converged before the redundant one was fixed, and
/// still has to — well inside the default cap.
#[test]
fn four_contacts_still_converge_promptly() {
    let s = solve(4, 200);
    assert!(s.converged, "four contacts must converge under the cap");
    assert!(
        s.iterations < 200,
        "four contacts took {} iterations",
        s.iterations
    );
}
