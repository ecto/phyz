//! Analytic contact benchmarks (design doc §6.1–6.4).
//!
//! Each of these has a closed-form answer, which is what separates "the solver
//! converged" from "the solver converged to the right physics". They are
//! written against a single rigid body sliding on a plane so the reference
//! solution is exact; the articulated case is covered elsewhere.
//!
//! Every one of these fails against the penalty model this replaces: its
//! friction law `min(mu*Fn, c*|v_t|)` sends friction to zero as the sliding
//! speed does, regardless of normal load, so nothing sticks at any angle.

use phyz_contact::{
    ContactMaterial, ContactProblem, ContactRow, ContactSolverConfig, solve_contacts,
};
use phyz_math::Vec3;

const G: f64 = 9.81;

/// A point mass on a plane, integrated with the convex contact solve in the
/// loop. The contact frame is `(normal, u, w)` with the incline's downhill
/// direction along `u`.
struct Block {
    mass: f64,
    /// Velocity in the contact frame: `.x` along the normal, `.y`/`.z` tangent.
    vel: Vec3,
    /// Tangential position along `u`, i.e. distance travelled downhill.
    slid: f64,
}

impl Block {
    fn new(mass: f64) -> Self {
        Self {
            mass,
            vel: Vec3::zeros(),
            slid: 0.0,
        }
    }

    /// One step on an incline of angle `alpha` with friction `mu`.
    fn step(&mut self, alpha: f64, mu: f64, dt: f64, cfg: &ContactSolverConfig) {
        // Gravity resolved into the contact frame: -g cos(alpha) along the
        // normal (into the surface), +g sin(alpha) downhill along u.
        let g_n = -G * alpha.cos();
        let g_u = G * alpha.sin();

        // Free velocity: where the block would be after gravity, absent
        // contact.
        let free = Vec3::new(self.vel.x + g_n * dt, self.vel.y + g_u * dt, self.vel.z);

        let inv_m = 1.0 / self.mass;
        let mut delassus = vec![0.0; 9];
        for i in 0..3 {
            delassus[i * 3 + i] = inv_m;
        }
        let problem = ContactProblem {
            n: 1,
            delassus,
            free_velocity: vec![free.x, free.y, free.z],
            rows: vec![ContactRow {
                mu,
                restitution: 0.0,
                depth: 0.0,
                ..Default::default()
            }],
        };
        let sol = solve_contacts(&problem, cfg);
        assert!(sol.converged, "contact solve must converge");

        let impulse = sol.impulses[0];
        self.vel = free + impulse * inv_m;
        self.slid += self.vel.y * dt;
    }
}

/// §6.1 Test A — stiction. Below the friction angle the block must not move,
/// at *any* incline angle below it. The old model creeps at every angle.
#[test]
fn block_sticks_below_the_friction_angle() {
    let mu = 0.5_f64;
    let critical = mu.atan();
    let cfg = ContactSolverConfig::simulation();
    let dt = 1e-3;

    for k in 0..=10 {
        let alpha = critical * 0.95 * (k as f64 / 10.0);
        let mut b = Block::new(2.0);
        for _ in 0..5000 {
            b.step(alpha, mu, dt, &cfg);
        }
        assert!(
            b.slid.abs() < 1e-3,
            "alpha={:.3} rad (< {:.3}): slid {:.6} m, should be stuck",
            alpha,
            critical,
            b.slid
        );
    }
}

/// §6.1 Test B — the friction angle itself. Binary-search the transition and
/// check it lands on `atan(mu)`. This catches a cone that is subtly pyramidal
/// or a `mu` that is scaled wrong.
#[test]
fn sliding_transition_is_at_the_friction_angle() {
    let mu = 0.5_f64;
    let expected = mu.atan();
    let cfg = ContactSolverConfig::simulation();
    let dt = 1e-3;

    let slides = |alpha: f64| {
        let mut b = Block::new(1.0);
        for _ in 0..2000 {
            b.step(alpha, mu, dt, &cfg);
        }
        b.slid.abs() > 1e-3
    };

    let (mut lo, mut hi) = (0.0_f64, std::f64::consts::FRAC_PI_2 * 0.99);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if slides(mid) { hi = mid } else { lo = mid }
    }
    let measured = 0.5 * (lo + hi);
    assert!(
        (measured - expected).abs() < 0.5_f64.to_radians(),
        "transition at {:.4} rad, expected atan({mu}) = {:.4}",
        measured,
        expected
    );
}

/// §6.1 Test C — sliding acceleration `a = g(sin a - mu cos a)`.
#[test]
fn sliding_acceleration_matches_theory() {
    let mu = 0.5;
    let alpha = 40.0_f64.to_radians();
    let expected = G * (alpha.sin() - mu * alpha.cos());
    assert!(expected > 0.0, "40 deg is above the friction angle");

    let cfg = ContactSolverConfig::simulation();
    let dt = 1e-4;
    let mut b = Block::new(3.0);
    let steps = 5000;
    for _ in 0..steps {
        b.step(alpha, mu, dt, &cfg);
    }
    let measured = b.vel.y / (steps as f64 * dt);
    assert!(
        (measured - expected).abs() / expected < 0.01,
        "acceleration {measured} vs expected {expected}"
    );
}

/// §6.1 Test D — direction isotropy. Sliding at any heading in the tangent
/// plane must decelerate identically. A pyramidal cone fails this by up to
/// `sqrt(2)`.
#[test]
fn friction_is_isotropic_in_the_tangent_plane() {
    let mu = 0.5;
    let cfg = ContactSolverConfig::simulation();
    let dt = 1e-4;
    let mass = 1.0;

    let decel_at = |heading: f64| {
        // Slide at 1 m/s in direction `heading`, no incline; measure the
        // tangential speed lost in one step.
        let v = Vec3::new(0.0, heading.cos(), heading.sin());
        let free = Vec3::new(v.x - G * dt, v.y, v.z);
        let mut delassus = vec![0.0; 9];
        for i in 0..3 {
            delassus[i * 3 + i] = 1.0 / mass;
        }
        let problem = ContactProblem {
            n: 1,
            delassus,
            free_velocity: vec![free.x, free.y, free.z],
            rows: vec![ContactRow {
                mu,
                restitution: 0.0,
                depth: 0.0,
                ..Default::default()
            }],
        };
        let sol = solve_contacts(&problem, &cfg);
        let after = free + sol.impulses[0] / mass;
        (after.y * after.y + after.z * after.z).sqrt()
    };

    let reference = decel_at(0.0);
    for k in 1..16 {
        let heading = k as f64 * std::f64::consts::TAU / 16.0;
        let s = decel_at(heading);
        assert!(
            (s - reference).abs() < 1e-9,
            "heading {heading}: speed {s} vs {reference} — cone is anisotropic"
        );
    }
}

/// §6.2 — restitution from drop height: `h1/h0 = e^2`.
#[test]
fn restitution_matches_drop_height_ratio() {
    let cfg = ContactSolverConfig::simulation();
    for e in [0.0, 0.3, 0.5, 0.8, 0.95] {
        // Impact speed from h0 = 1 m.
        let h0 = 1.0;
        let v_impact = (2.0 * G * h0).sqrt();

        let mass = 1.0;
        let mut delassus = vec![0.0; 9];
        for i in 0..3 {
            delassus[i * 3 + i] = 1.0 / mass;
        }
        let eff = ContactProblem::effective_restitution(e, v_impact, cfg.restitution_threshold);
        // Approaching at -v_impact along the normal; target is +e*v_impact.
        let problem = ContactProblem {
            n: 1,
            delassus,
            free_velocity: vec![-v_impact * (1.0 + eff), 0.0, 0.0],
            rows: vec![ContactRow {
                mu: 0.0,
                restitution: eff,
                depth: 0.0,
                ..Default::default()
            }],
        };
        let sol = solve_contacts(&problem, &cfg);
        let v_out = -v_impact + sol.impulses[0].x / mass;
        let h1 = v_out * v_out / (2.0 * G);
        let expected = e * e * h0;
        assert!(
            (h1 - expected).abs() < 0.02 * h0.max(0.05),
            "e={e}: bounce height {h1:.4} vs e^2*h0 = {expected:.4}"
        );
    }
}

/// §6.2 settling — a bouncy body must come to rest, not micro-bounce forever.
/// This is what the smooth low-speed restitution ramp exists to guarantee.
#[test]
fn bouncy_contact_settles() {
    let cfg = ContactSolverConfig::simulation();
    let e = 0.8;
    // Well above the threshold: full restitution.
    assert!(
        (ContactProblem::effective_restitution(e, 1.0, cfg.restitution_threshold) - e).abs()
            < 1e-12
    );
    // Well below: none, so the body stops instead of chattering.
    assert_eq!(
        ContactProblem::effective_restitution(e, 0.01, cfg.restitution_threshold),
        0.0
    );
    // And the ramp is continuous and monotone across the transition.
    let mut prev = 0.0;
    for k in 0..=50 {
        let v = cfg.restitution_threshold * (1.0 + k as f64 / 25.0);
        let r = ContactProblem::effective_restitution(e, v, cfg.restitution_threshold);
        assert!(r >= prev - 1e-15, "restitution ramp must be monotone");
        assert!((0.0..=e + 1e-15).contains(&r));
        prev = r;
    }
}

/// §6.3 — a resting multi-point contact carries the load evenly and does not
/// drift. Four coplanar contacts under a symmetric load must share it.
#[test]
fn resting_manifold_shares_load_and_holds() {
    let cfg = ContactSolverConfig::simulation();
    let n = 4;
    let mass = 4.0;
    let weight = mass * G;
    let dt = 1e-3;

    // Each contact sees 1/n of the body's inverse mass along its normal.
    let dim = 3 * n;
    let mut delassus = vec![0.0; dim * dim];
    for i in 0..dim {
        delassus[i * dim + i] = (n as f64) / mass;
    }
    // Approaching the surface at the speed one step of gravity imparts.
    let mut free_velocity = vec![0.0; dim];
    for c in 0..n {
        free_velocity[3 * c] = -(weight / mass) * dt;
    }
    let problem = ContactProblem {
        n,
        delassus,
        free_velocity,
        rows: vec![
            ContactRow {
                mu: 0.6,
                restitution: 0.0,
                depth: 1e-4,
                ..Default::default()
            };
            n
        ],
    };

    let sol = solve_contacts(&problem, &cfg);
    assert!(sol.converged, "resting solve must converge");

    let total: f64 = sol.impulses.iter().map(|f| f.x).sum();
    assert!(
        (total - weight * dt).abs() / (weight * dt) < 1e-6,
        "total normal impulse {total} should support the weight {}",
        weight * dt
    );
    // Symmetric load, symmetric share.
    for f in &sol.impulses {
        assert!(
            (f.x - total / n as f64).abs() / (total / n as f64) < 1e-6,
            "load not shared evenly: {:?}",
            sol.impulses
        );
        assert!(f.y.abs() < 1e-12 && f.z.abs() < 1e-12, "spurious friction");
    }
}

/// Coupled contacts must be solved *together*: the answer differs from
/// treating each contact independently. This is the property a per-contact
/// local force law (what this replaces) structurally cannot have.
#[test]
fn coupled_contacts_differ_from_independent_ones() {
    let cfg = ContactSolverConfig::simulation();
    let n = 2;
    let dim = 3 * n;
    let mut delassus = vec![0.0; dim * dim];
    for i in 0..dim {
        delassus[i * dim + i] = 1.0;
    }
    // Strong off-diagonal coupling between the two normal directions.
    // Row 0 (contact 0 normal) x column 3 (contact 1 normal), and its mirror.
    delassus[3] = 0.6;
    delassus[3 * dim] = 0.6;

    let problem = ContactProblem {
        n,
        delassus,
        free_velocity: vec![-1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        rows: vec![
            ContactRow {
                mu: 0.5,
                restitution: 0.0,
                depth: 0.0,
                ..Default::default()
            };
            n
        ],
    };
    let coupled = solve_contacts(&problem, &cfg);
    assert!(coupled.converged);

    // Independent solve: same diagonal, no coupling.
    let mut indep_delassus = vec![0.0; dim * dim];
    for i in 0..dim {
        indep_delassus[i * dim + i] = 1.0;
    }
    let indep = solve_contacts(
        &ContactProblem {
            delassus: indep_delassus,
            ..problem.clone()
        },
        &cfg,
    );

    let d = (coupled.impulses[0].x - indep.impulses[0].x).abs();
    assert!(
        d > 0.1,
        "coupling must change the answer; got {d} (coupled {:?}, independent {:?})",
        coupled.impulses[0],
        indep.impulses[0]
    );
    // With positive coupling each contact needs less impulse than alone.
    assert!(coupled.impulses[0].x < indep.impulses[0].x);
}

/// The solved impulses must satisfy the KKT conditions of the convex program:
/// feasible (in the cone) and, when strictly inside, driving the contact-space
/// velocity to zero (stiction).
#[test]
fn solution_satisfies_kkt_conditions() {
    let cfg = ContactSolverConfig::simulation();
    let mass = 2.0;
    let mut delassus = vec![0.0; 9];
    for i in 0..3 {
        delassus[i * 3 + i] = 1.0 / mass;
    }
    // Pressing in hard with a modest tangential drift: should stick.
    let problem = ContactProblem {
        n: 1,
        delassus,
        free_velocity: vec![-1.0, 0.05, 0.0],
        rows: vec![ContactRow {
            mu: 0.8,
            restitution: 0.0,
            depth: 0.0,
            ..Default::default()
        }],
    };
    let sol = solve_contacts(&problem, &cfg);
    assert!(sol.converged);
    let f = sol.impulses[0];

    assert!(
        phyz_contact::in_cone(f, 0.8, 1e-9),
        "impulse {f:?} outside the friction cone"
    );
    assert!(
        phyz_contact::in_cone_interior(f, 0.8, 1e-9),
        "expected stiction (cone interior), got {f:?}"
    );

    // Stuck. Contact here is *soft*: the converged KKT condition of the
    // regularized program is `A f + b = -R f`, not `A f + b = 0`, so a stuck
    // contact retains a residual creep of exactly `R * |f|`. That is the
    // penetration/compliance the regularizer buys, and asserting it pins the
    // softness rather than hiding it.
    let after = Vec3::new(-1.0, 0.05, 0.0) + f / mass;
    let predicted = f * -cfg.regularization;
    assert!(
        (after - predicted).norm() < 1e-12,
        "soft-contact KKT residual should be -R*f = {predicted:?}, got {after:?}"
    );
    // And that creep vanishes as the contact is stiffened.
    let stiffer = ContactSolverConfig {
        regularization: cfg.regularization * 1e-3,
        ..cfg
    };
    let sol2 = solve_contacts(&problem, &stiffer);
    let after2 = Vec3::new(-1.0, 0.05, 0.0) + sol2.impulses[0] / mass;
    assert!(
        after2.norm() < after.norm() * 1e-2,
        "stiffer contact must creep less: {} vs {}",
        after2.norm(),
        after.norm()
    );
}

/// Materials round-trip the restitution field that was previously inert.
#[test]
fn bouncy_material_actually_carries_restitution() {
    let m = ContactMaterial::bouncy();
    assert!(m.restitution > 0.0, "bouncy() must have restitution");
    let rigid = ContactMaterial::rigid();
    assert_eq!(rigid.restitution, 0.0);
}
