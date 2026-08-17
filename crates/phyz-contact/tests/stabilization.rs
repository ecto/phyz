//! Position stabilization, pair materials, and warm starting.
//!
//! These cover the failure the convex solve had even after it was solving the
//! right convex program: `ContactRow.depth` was assembled and never read, so
//! nothing ever repaid penetration. A body that sank a micron per step under
//! integration error kept the micron, and an eight-box stack sank forever.
//! With a solref/solimp reference response the penetration reaches a bounded
//! equilibrium and *stays there*, which is what "does not creep" means
//! precisely: not zero penetration (soft contact has some by construction) but
//! zero drift once it settles.

use phyz_contact::{
    ContactMaterial, ContactProblem, ContactRow, ContactSolverConfig, SolImp, SolRef,
    solve_contacts, solve_contacts_warm,
};
use phyz_math::Vec3;

const G: f64 = 9.81;

/// A vertical column of `n` unit-height boxes, stepped with the convex solve
/// in the loop.
///
/// Written against a hand-assembled Delassus operator rather than the full
/// model pipeline on purpose: the point under test is the contact solve's
/// stabilization, and a 1-D column has an exact reference (every box at its
/// rest height) that no narrow-phase or joint convention can muddy.
struct Stack {
    /// Box half-height, metres.
    half: f64,
    mass: f64,
    /// Centre height of each box, bottom-first.
    z: Vec<f64>,
    /// Vertical velocity of each box.
    v: Vec<f64>,
    material: ContactMaterial,
    /// Impulses from the previous step, for warm starting. Contact identity in
    /// a column is positional and never changes, so no cache is needed here.
    warm: Vec<Vec3>,
    /// Iterations the last solve took.
    iterations: usize,
}

impl Stack {
    fn new(n: usize, mass: f64, half: f64, material: ContactMaterial) -> Self {
        Self {
            half,
            mass,
            z: (0..n).map(|i| (2 * i + 1) as f64 * half).collect(),
            v: vec![0.0; n],
            material,
            warm: vec![Vec3::zeros(); n],
            iterations: 0,
        }
    }

    fn n(&self) -> usize {
        self.z.len()
    }

    /// Rest height of box `i` with every contact exactly touching.
    fn rest(&self, i: usize) -> f64 {
        (2 * i + 1) as f64 * self.half
    }

    /// Penetration at contact `c`: `c = 0` is box 0 against the ground, `c > 0`
    /// is box `c` against box `c-1`.
    fn depth(&self, c: usize) -> f64 {
        if c == 0 {
            self.half - self.z[0]
        } else {
            2.0 * self.half - (self.z[c] - self.z[c - 1])
        }
    }

    /// One step: gravity, then the contact solve, then integrate.
    fn step(&mut self, dt: f64, cfg: &ContactSolverConfig, warm_start: bool) {
        let n = self.n();
        let dim = 3 * n;
        let inv_m = 1.0 / self.mass;

        // A = J M^-1 J^T for the column's constraint Jacobian: contact 0 sees
        // box 0 alone, contact c > 0 sees the pair (c-1, c). That makes A
        // tridiagonal with 2/m (1/m for the ground contact) on the diagonal
        // and -1/m off it.
        let mut delassus = vec![0.0; dim * dim];
        let a = |c: usize, k: usize| -> f64 {
            if c == k {
                if c == 0 { inv_m } else { 2.0 * inv_m }
            } else if c.abs_diff(k) == 1 {
                -inv_m
            } else {
                0.0
            }
        };
        for c in 0..n {
            for k in 0..n {
                let v = a(c, k);
                if v == 0.0 {
                    continue;
                }
                for r in 0..3 {
                    delassus[(3 * c + r) * dim + 3 * k + r] = v;
                }
            }
        }

        // Free velocity after gravity, mapped into contact space.
        let free: Vec<f64> = self.v.iter().map(|v| v - G * dt).collect();
        let mut free_velocity = vec![0.0; dim];
        for c in 0..n {
            free_velocity[3 * c] = if c == 0 {
                free[0]
            } else {
                free[c] - free[c - 1]
            };
        }

        let rows: Vec<ContactRow> = (0..n)
            .map(|c| ContactRow::from_material(&self.material, self.depth(c), dt, 0.0))
            .collect();
        let problem = ContactProblem {
            // Synthetic: no body structure, so `PerBody` behaves as `Full`.
            bodies: Vec::new(),
            n,
            delassus,
            free_velocity,
            rows,
        };

        let sol = if warm_start {
            solve_contacts_warm(&problem, cfg, &self.warm)
        } else {
            solve_contacts(&problem, cfg)
        };
        assert!(
            sol.converged,
            "stack solve must converge (residual {})",
            sol.residual
        );
        self.iterations = sol.iterations;
        self.warm.clone_from(&sol.impulses);

        // v' = v_free + M^-1 J^T f, then integrate. Box `i` gets the impulse
        // of the contact below it, less the reaction from the contact above.
        for (i, free_i) in free.iter().enumerate() {
            let mut dv = sol.impulses[i].x * inv_m;
            if i + 1 < n {
                dv -= sol.impulses[i + 1].x * inv_m;
            }
            self.v[i] = free_i + dv;
            self.z[i] += self.v[i] * dt;
        }
    }

    /// How far box `i` has sunk below its rest height.
    fn sink(&self, i: usize) -> f64 {
        self.rest(i) - self.z[i]
    }
}

/// The headline test: eight boxes, `dt = 2 ms`, six seconds. The stack must
/// settle to a bounded penetration and then stop moving.
///
/// Without the solref bias every contact's solve targets *zero* normal
/// velocity, which freezes whatever penetration the previous step's
/// integration error left behind. The stack then sinks monotonically forever,
/// and the bottom box eventually falls through the floor.
#[test]
fn eight_box_stack_does_not_creep_at_2ms() {
    let dt = 2e-3;
    let cfg = ContactSolverConfig::simulation();
    let mut s = Stack::new(8, 1.0, 0.05, ContactMaterial::default());

    // Settle.
    for _ in 0..1500 {
        s.step(dt, &cfg, true);
    }
    let settled: Vec<f64> = (0..s.n()).map(|i| s.z[i]).collect();
    let sink_after_settle = s.sink(7);

    // Another three seconds. This is the interval that matters: a creeping
    // stack keeps sinking here, a stabilized one does not.
    for _ in 0..1500 {
        s.step(dt, &cfg, true);
    }

    for (i, z0) in settled.iter().enumerate() {
        let drift = (s.z[i] - z0).abs();
        assert!(
            drift < 1e-9,
            "box {i} drifted {drift:.3e} m over 3 s after settling — the stack \
             is creeping"
        );
    }

    // The equilibrium penetration is bounded and small. It is *load
    // dependent*, which is soft contact behaving as designed rather than a
    // failure: the bottom contact carries eight boxes and penetrates most, and
    // the total sink is the sum down the column. A few millimetres over an
    // 80 cm stack is the default material's answer at `dt = 2 ms`; a stiffer
    // `solref.timeconst` or a higher `solimp.dmax` buys less.
    assert!(
        sink_after_settle > 0.0,
        "soft contact must settle to *some* penetration, got {sink_after_settle:.3e}"
    );
    assert!(
        sink_after_settle < 1e-2,
        "total sink {sink_after_settle:.3e} m is too large for the default material"
    );

    // Every box must still be above the one below it, and nothing may tunnel.
    for i in 0..s.n() {
        assert!(
            s.z[i] > 0.0,
            "box {i} fell through the floor: z = {}",
            s.z[i]
        );
        if i > 0 {
            assert!(s.z[i] > s.z[i - 1], "stack order inverted at box {i}");
        }
    }

    // No single contact may penetrate anywhere near a box half-height.
    for c in 0..s.n() {
        assert!(
            s.depth(c) < 0.1 * s.half,
            "contact {c} penetrates {:.3e} m, more than a tenth of the box",
            s.depth(c)
        );
    }

    // Load ordering among the internal contacts: they all have the same
    // compliance (two boxes each), so the lower one, carrying more, must sit
    // deeper. The ground contact is excluded — it sees one box instead of two,
    // i.e. half the compliance, so it is not comparable.
    for c in 2..s.n() {
        assert!(
            s.depth(c) < s.depth(c - 1) + 1e-12,
            "contact {c} ({:.3e} m) carries less load than {} ({:.3e} m) yet \
             penetrates more",
            s.depth(c),
            c - 1,
            s.depth(c - 1)
        );
    }
}

/// The same column, one box, seeded with a penetration far below equilibrium.
///
/// This is the sharp statement of what the fix fixes. Without a reference
/// response the solve only ever targets *zero* normal velocity, so a
/// penetration — however it got there, and integration error puts a little in
/// every step — is frozen in place forever. With solref it decays back to the
/// material's equilibrium.
#[test]
fn without_stabilization_penetration_is_never_repaid() {
    let dt = 2e-3;
    let cfg = ContactSolverConfig::simulation();

    // `width = 0` pins the impedance at dmax, and an enormous `timeconst`
    // drives the error reduction to ~0: a contact with no recovery bias.
    let inert = ContactMaterial {
        solref: SolRef {
            timeconst: 1e9,
            dampratio: 1.0,
        },
        solimp: SolImp {
            width: 0.0,
            ..SolImp::default()
        },
        ..ContactMaterial::default()
    };

    let seed = 5e-4;
    let mut stuck = Stack::new(1, 1.0, 0.05, inert);
    let mut fixed = Stack::new(1, 1.0, 0.05, ContactMaterial::default());
    // The equilibrium the stabilized box should return to, measured rather
    // than asserted as a magic number: an identical box dropped in exactly
    // touching.
    let mut reference = Stack::new(1, 1.0, 0.05, ContactMaterial::default());
    for s in [&mut stuck, &mut fixed] {
        s.z[0] -= seed;
    }

    for _ in 0..2500 {
        stuck.step(dt, &cfg, true);
        fixed.step(dt, &cfg, true);
        reference.step(dt, &cfg, true);
    }

    assert!(
        stuck.sink(0) > 0.9 * seed,
        "without a bias the seeded penetration must persist; sank {:.3e}",
        stuck.sink(0)
    );
    let equilibrium = reference.sink(0);
    assert!(
        (fixed.sink(0) - equilibrium).abs() < 0.05 * seed,
        "with solref the seeded penetration must be repaid down to the \
         material's equilibrium {equilibrium:.3e}; still at {:.3e}",
        fixed.sink(0)
    );
    assert!(
        equilibrium < 0.5 * seed,
        "the default material's equilibrium penetration {equilibrium:.3e} is \
         suspiciously deep"
    );
    // Repaid, not overshot into a bounce: the reference response is critically
    // damped and must not launch the box.
    assert!(
        fixed.sink(0) > 0.0,
        "the box must settle *on* the ground, not above it: {:.3e}",
        fixed.sink(0)
    );
}

/// A block on an incline steep enough to slide must slide at the analytic
/// rate, *with* stabilization active. The bias lives in the normal row; if it
/// leaked into the tangential solve the acceleration would come out wrong.
#[test]
fn incline_slide_matches_the_analytic_acceleration_with_stabilization() {
    let cfg = ContactSolverConfig::simulation();
    let dt = 1e-3;
    let mu = 0.2;
    let mass = 1.5;
    let material = ContactMaterial {
        friction: mu,
        ..ContactMaterial::default()
    };

    for alpha_deg in [25.0_f64, 40.0, 55.0] {
        let alpha = alpha_deg.to_radians();
        // Analytic: a = g (sin a - mu cos a) once sliding.
        let expected = G * (alpha.sin() - mu * alpha.cos());

        let inv_m = 1.0 / mass;
        let mut vel = Vec3::zeros();
        // Start at the equilibrium-ish penetration so the bias is live
        // throughout, not just at the first step.
        let mut depth = 2e-4;
        let mut warm = vec![Vec3::zeros()];
        let steps = 500;
        for _ in 0..steps {
            let free = Vec3::new(
                vel.x - G * alpha.cos() * dt,
                vel.y + G * alpha.sin() * dt,
                vel.z,
            );
            let mut delassus = vec![0.0; 9];
            for i in 0..3 {
                delassus[i * 3 + i] = inv_m;
            }
            let problem = ContactProblem {
                // Synthetic: no body structure, so `PerBody` behaves as `Full`.
                bodies: Vec::new(),
                n: 1,
                delassus,
                free_velocity: vec![free.x, free.y, free.z],
                rows: vec![ContactRow::from_material(&material, depth, dt, 0.0)],
            };
            let sol = solve_contacts_warm(&problem, &cfg, &warm);
            assert!(sol.converged);
            warm.clone_from(&sol.impulses);
            vel = free + sol.impulses[0] * inv_m;
            depth -= vel.x * dt;
        }

        let measured = vel.y / (steps as f64 * dt);
        assert!(
            (measured - expected).abs() / expected < 5e-3,
            "alpha={alpha_deg} deg: sliding acceleration {measured:.4} vs \
             analytic {expected:.4}"
        );
        // The normal direction must have gone quiet: stabilization is a
        // transient, not a persistent upward push.
        assert!(
            vel.x.abs() < 1e-3,
            "normal velocity {} should have settled",
            vel.x
        );
        assert!(
            depth > -1e-6,
            "block must not have been pushed off the surface"
        );
    }
}

/// Below the friction angle the block must hold, indefinitely, while the
/// normal direction is simultaneously repaying penetration. The two must not
/// interfere: a stabilization bias that fed into the tangential rows would
/// show up here as slow creep downhill.
#[test]
fn stiction_holds_while_penetration_recovers() {
    let cfg = ContactSolverConfig::simulation();
    let dt = 2e-3;
    let mu = 0.6_f64;
    let alpha = 0.9 * mu.atan();
    let mass = 3.0;
    let inv_m = 1.0 / mass;
    let material = ContactMaterial {
        friction: mu,
        ..ContactMaterial::default()
    };

    let mut vel = Vec3::zeros();
    let mut slid = 0.0;
    let mut depth = 1e-3; // Start a full millimetre deep.
    let mut warm = vec![Vec3::zeros()];
    let mut settled_at = None;

    for k in 0..2500 {
        let free = Vec3::new(
            vel.x - G * alpha.cos() * dt,
            vel.y + G * alpha.sin() * dt,
            vel.z,
        );
        let mut delassus = vec![0.0; 9];
        for i in 0..3 {
            delassus[i * 3 + i] = inv_m;
        }
        let problem = ContactProblem {
            // Synthetic: no body structure, so `PerBody` behaves as `Full`.
            bodies: Vec::new(),
            n: 1,
            delassus,
            free_velocity: vec![free.x, free.y, free.z],
            rows: vec![ContactRow::from_material(&material, depth, dt, 0.0)],
        };
        let sol = solve_contacts_warm(&problem, &cfg, &warm);
        assert!(sol.converged);
        warm.clone_from(&sol.impulses);
        vel = free + sol.impulses[0] * inv_m;
        slid += vel.y * dt;
        depth -= vel.x * dt;
        if settled_at.is_none() && depth < 3e-4 {
            settled_at = Some(k);
        }
    }

    assert!(
        slid.abs() < 1e-6,
        "block below the friction angle slid {slid:.3e} m — stiction leaked"
    );
    assert!(
        settled_at.is_some(),
        "penetration never recovered; still {depth:.3e} m deep"
    );
    assert!(
        depth > 0.0 && depth < 3e-4,
        "penetration should settle small and positive, got {depth:.3e}"
    );
}

/// The block-on-an-incline scenario run in both regularization modes, which is
/// the concrete statement of why exact stiction is the default.
///
/// Below the friction angle Coulomb's law says the block does not move. In the
/// default mode it does not: the tangential rows carry only the config floor,
/// so the stiction constraint is enforced essentially exactly. Under
/// `mujoco_compat` the impedance-derived regularizer lands on the friction rows
/// too, and a converged sticking solve then satisfies `v_t = -R f_t` rather
/// than `v_t = 0`. With `R = (1-d)/d * A_nn`, `A_nn = 1/m` for a single body,
/// and the tangential impulse balancing gravity at `f_t = m g sin(alpha) dt`,
/// the mass cancels and the block slides forever at
///
/// ```text
/// v_creep = g sin(alpha) dt (1 - d) / d
/// ```
///
/// This test asserts both halves: default holds, compat creeps, and the creep
/// matches that closed form. See `phyz_contact::regularization_diag`.
#[test]
fn mujoco_creep_matches_the_analytic_rate() {
    let dt = 2e-3;
    let mu = 0.6_f64;
    let alpha = 0.9 * mu.atan(); // Comfortably inside the friction cone.
    let mass = 3.0;
    let inv_m = 1.0 / mass;
    let material = ContactMaterial {
        friction: mu,
        ..ContactMaterial::default()
    };
    // Hold the penetration fixed so the impedance — and therefore the predicted
    // creep rate — is a single known constant for the whole run. The normal
    // direction is not what is under test here.
    let depth = 2e-4;
    let steps = 1500;

    let run = |cfg: &ContactSolverConfig| {
        let mut vel = Vec3::zeros();
        let mut slid = 0.0;
        let mut warm = vec![Vec3::zeros()];
        for _ in 0..steps {
            let free = Vec3::new(
                vel.x - G * alpha.cos() * dt,
                vel.y + G * alpha.sin() * dt,
                vel.z,
            );
            let mut delassus = vec![0.0; 9];
            for i in 0..3 {
                delassus[i * 3 + i] = inv_m;
            }
            let problem = ContactProblem {
                // Synthetic: no body structure, so `PerBody` behaves as `Full`.
                bodies: Vec::new(),
                n: 1,
                delassus,
                free_velocity: vec![free.x, free.y, free.z],
                rows: vec![ContactRow::from_material(&material, depth, dt, 0.0)],
            };
            let sol = solve_contacts_warm(&problem, cfg, &warm);
            assert!(
                sol.converged,
                "solve must converge (residual {})",
                sol.residual
            );
            warm.clone_from(&sol.impulses);
            vel = free + sol.impulses[0] * inv_m;
            slid += vel.y * dt;
        }
        (vel.y, slid)
    };

    let default_cfg = ContactSolverConfig::simulation();
    let compat_cfg = ContactSolverConfig::mujoco_compat();
    assert!(!default_cfg.mujoco_compat);
    assert!(compat_cfg.mujoco_compat);

    let (v_default, slid_default) = run(&default_cfg);
    let (v_compat, slid_compat) = run(&compat_cfg);

    // The impedance the solve actually saw, read from the row rather than
    // recomputed, so the prediction cannot drift from the material model.
    let d = ContactRow::from_material(&material, depth, dt, 0.0).impedance;
    assert!(
        d > 0.0 && d < 1.0,
        "test needs a genuinely compliant contact, got d = {d}"
    );
    // And the impedance term must dominate the config floor, or the prediction
    // below is measuring `regularization` instead of `solimp`.
    let r_impedance = (1.0 - d) / d * inv_m;
    assert!(
        r_impedance > 100.0 * default_cfg.regularization,
        "impedance regularizer {r_impedance:.3e} is at the config floor; the \
         analytic prediction would not apply"
    );

    let predicted = G * alpha.sin() * dt * (1.0 - d) / d;

    // Default: stiction holds. Not literally zero — the tangential rows still
    // carry the `regularization` floor, which is the same `v_t = -R f_t`
    // relation with `R = config.regularization` instead of the impedance term.
    // That floor is a deliberate, tunable conditioning knob and its residual
    // slip is ~30 nm/s here, five orders of magnitude below the impedance
    // creep. The same floor drives the `mujoco_compat` prediction, so this
    // bound is written against it rather than as a magic constant.
    let floor_slip = default_cfg.regularization * mass * G * alpha.sin() * dt;
    assert!(
        v_default.abs() < 2.0 * floor_slip,
        "default mode must hold to the regularization floor ({floor_slip:.3e} \
         m/s); slip velocity {v_default:.3e} m/s"
    );
    assert!(
        v_default.abs() < 1e-4 * predicted,
        "default slip {v_default:.3e} m/s is not negligible against the compat \
         creep {predicted:.3e} m/s"
    );
    assert!(
        slid_default.abs() < 1e-6,
        "default mode slid {slid_default:.3e} m over {:.1} s",
        steps as f64 * dt
    );

    // Compat: creeps, at the analytic rate.
    assert!(
        (v_compat - predicted).abs() / predicted < 0.02,
        "compat creep {v_compat:.6e} m/s vs analytic prediction \
         {predicted:.6e} m/s (d = {d})"
    );
    // The creep is a steady state, so the distance travelled is the rate times
    // the elapsed time — it accumulates without bound rather than settling.
    let elapsed = steps as f64 * dt;
    assert!(
        (slid_compat - predicted * elapsed).abs() / (predicted * elapsed) < 0.05,
        "compat slid {slid_compat:.6e} m in {elapsed} s, expected \
         {:.6e} m at a steady {predicted:.6e} m/s",
        predicted * elapsed
    );
    // And the headline: the divergence is worth having. Roughly a millimetre
    // per second, which over these three seconds is millimetres of drift for a
    // block that is supposed to be parked.
    assert!(
        slid_compat.abs() > 1e-3,
        "compat mode should creep measurably; only {slid_compat:.3e} m"
    );
}

/// The compat flag must not perturb the default path at all. `simulation` and
/// `mujoco_compat` differ in exactly one field, so anything that changes in
/// default mode when the flag is added is a bug in the plumbing.
#[test]
fn compat_flag_off_is_the_unmodified_solve() {
    let dt = 1e-3;
    let cfg = ContactSolverConfig::simulation();
    let material = ContactMaterial {
        friction: 0.5,
        ..ContactMaterial::default()
    };
    let build = || {
        let mut delassus = vec![0.0; 9];
        for i in 0..3 {
            delassus[i * 3 + i] = 1.0;
        }
        ContactProblem {
            // Synthetic: no body structure, so `PerBody` behaves as `Full`.
            bodies: Vec::new(),
            n: 1,
            delassus,
            free_velocity: vec![-1.0, 0.3, 0.1],
            rows: vec![ContactRow::from_material(&material, 1e-4, dt, 0.0)],
        }
    };
    let p = build();
    // The regularizer itself: tangential rows sit on the config floor, the
    // normal row does not.
    let reg = phyz_contact::regularization_diag(&p, 0, &cfg);
    assert_eq!(reg[1], cfg.regularization);
    assert_eq!(reg[2], cfg.regularization);
    assert!(reg[0] > cfg.regularization);

    let compat = ContactSolverConfig::mujoco_compat();
    let creg = phyz_contact::regularization_diag(&p, 0, &compat);
    assert_eq!(creg, [reg[0], reg[0], reg[0]]);

    // And the two solves genuinely differ, so the test above is not comparing
    // a flag that does nothing.
    let a = solve_contacts(&p, &cfg).impulses[0];
    let b = solve_contacts(&p, &compat).impulses[0];
    assert!((a - b).norm() > 1e-9, "compat flag changed nothing: {a:?}");
}

/// Warm starting must not change the answer — the problem is strongly convex,
/// so it cannot — and must materially cut the iteration count on a stack.
#[test]
fn warm_starting_converges_faster_to_the_same_answer() {
    let dt = 2e-3;
    let cfg = ContactSolverConfig::simulation();

    let run = |warm: bool| {
        let mut s = Stack::new(8, 1.0, 0.05, ContactMaterial::default());
        let mut iters = 0usize;
        for _ in 0..1000 {
            s.step(dt, &cfg, warm);
            iters += s.iterations;
        }
        (s, iters)
    };

    let (cold, cold_iters) = run(false);
    let (hot, hot_iters) = run(true);

    for i in 0..cold.n() {
        assert!(
            (cold.z[i] - hot.z[i]).abs() < 1e-9,
            "box {i}: warm start changed the trajectory ({} vs {})",
            cold.z[i],
            hot.z[i]
        );
    }
    assert!(
        hot_iters * 2 < cold_iters,
        "warm start should more than halve total iterations: {hot_iters} vs {cold_iters}"
    );
}

/// Pair combination has to be commutative all the way through the solve, not
/// merely in `ContactMaterial::combine`: swapping which body the narrow phase
/// lists first must not change the physics.
#[test]
fn pair_combination_is_independent_of_contact_ordering() {
    let dt = 1e-3;
    let cfg = ContactSolverConfig::simulation();
    let rubber = ContactMaterial {
        friction: 1.1,
        ..ContactMaterial::soft()
    };
    let ice = ContactMaterial {
        friction: 0.02,
        ..ContactMaterial::rigid()
    };

    let solve = |m: &ContactMaterial| {
        let mut delassus = vec![0.0; 9];
        for i in 0..3 {
            delassus[i * 3 + i] = 1.0;
        }
        let problem = ContactProblem {
            // Synthetic: no body structure, so `PerBody` behaves as `Full`.
            bodies: Vec::new(),
            n: 1,
            delassus,
            // Pressed in, drifting sideways: friction decides whether it
            // sticks.
            free_velocity: vec![-1.0, 0.3, 0.0],
            rows: vec![ContactRow::from_material(m, 1e-4, dt, 0.0)],
        };
        solve_contacts(&problem, &cfg).impulses[0]
    };

    let a = solve(&ContactMaterial::combine(&rubber, &ice));
    let b = solve(&ContactMaterial::combine(&ice, &rubber));
    assert!(
        (a - b).norm() < 1e-15,
        "ordering changed the impulse: {a:?} vs {b:?}"
    );

    // And the documented rule bites: friction is the max, so the pair sticks
    // rather than sliding on the ice's 0.02.
    let ft = (a.y * a.y + a.z * a.z).sqrt();
    assert!(
        ft < 1.1 * a.x - 1e-12,
        "combined friction should be rubber's, not ice's: |ft| = {ft}, mu*fn = {}",
        1.1 * a.x
    );
    let icy = solve(&ice);
    let icy_ft = (icy.y * icy.y + icy.z * icy.z).sqrt();
    assert!(
        ft > icy_ft,
        "the pair must grip harder than ice alone: {ft} vs {icy_ft}"
    );
}
