//! Validation of `phyz-vbd` against closed-form references.
//!
//! Every test here prints the number it measured, whether it passes or not, so
//! a run of `cargo test -p phyz-vbd -- --nocapture` is a report and not a row
//! of green ticks. Tolerances are stated with the reason they are what they
//! are; where the model and the reference genuinely disagree, the disagreement
//! is reported and explained rather than tuned away.

use phyz_math::Vec3;
use phyz_vbd::{Material, SoftBody, VbdConfig, VbdSolver, mesh};

/// Drive a body to a quasi-static equilibrium.
///
/// Zeroing the velocity after every step turns the dynamic solver into a
/// damped static one: the inertia term `m/h²` degenerates into a proximal
/// regulariser, and at `h = 0.5 s` it is five orders of magnitude below the
/// element stiffness, so the fixed point is the static equilibrium of the
/// elastic energy plus gravity and not something in between.
fn settle(body: &mut SoftBody, solver: &mut VbdSolver, steps: usize) {
    for _ in 0..steps {
        solver.step(body);
        body.velocities.iter_mut().for_each(|v| *v = Vec3::zero());
    }
}

/// Worst-vertex residual of the static equilibrium equations,
/// `‖f_elastic + m g‖`, over the free vertices. This is the honest measure of
/// "did the solver actually converge", in newtons.
fn static_residual(body: &SoftBody, gravity: Vec3) -> f64 {
    let f = body.elastic_forces();
    (0..body.len())
        .filter(|&i| !body.pinned[i])
        .map(|i| (f[i] + gravity * body.masses[i]).norm())
        .fold(0.0f64, f64::max)
}

/// **Static equilibrium of a single tetrahedron under gravity.**
///
/// Three vertices pinned, one free, hanging on a single stable Neo-Hookean
/// element. At equilibrium the element's restoring force on the free vertex
/// must cancel its weight exactly. There is no closed form for the *position*
/// of the sagged vertex — the energy is not quadratic — but there is an exact
/// closed form for the condition it satisfies, and that is what is checked.
#[test]
fn single_tet_static_equilibrium_under_gravity() {
    let rest = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.1, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.1),
        Vec3::new(0.03, 0.1, 0.03),
    ];
    let material = Material {
        youngs_modulus: 5.0e4,
        poisson_ratio: 0.3,
        density: 1000.0,
    };
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let mut body = SoftBody::builder(rest, material)
        .tets(&[[0, 1, 2, 3]])
        .pin(&[0, 1, 2])
        .build()
        .unwrap();

    let mut solver = VbdSolver::new(VbdConfig {
        dt: 0.5,
        iterations: 40,
        gravity,
        ..VbdConfig::default()
    });
    settle(&mut body, &mut solver, 200);

    let weight = body.masses[3] * gravity.norm();
    let residual = static_residual(&body, gravity);
    let sag = body.positions[3].y - body.rest_positions[3].y;
    println!(
        "single tet: sag = {sag:.6e} m, equilibrium residual = {residual:.6e} N \
         ({:.3e} relative to the {weight:.6e} N weight)",
        residual / weight
    );

    // A converged block-Newton fixed point should nail this to round-off; 1e-6
    // of the weight is four orders of magnitude of slack on that.
    assert!(
        residual < 1e-6 * weight,
        "not in equilibrium: {residual:e} N vs weight {weight:e} N"
    );
    assert!(sag < 0.0, "the free vertex did not sag: {sag} m");
}

/// A clamped beam, meshed, with the clamped and tip vertex sets.
fn cantilever(
    nx: usize,
    ny: usize,
    nz: usize,
    length: f64,
    depth: f64,
    breadth: f64,
    material: Material,
) -> (SoftBody, Vec<usize>) {
    let (rest, tets) = mesh::tet_box(nx, ny, nz, Vec3::new(length, depth, breadth));
    let clamped: Vec<usize> = rest
        .iter()
        .enumerate()
        .filter(|(_, p)| p.x <= 0.0)
        .map(|(i, _)| i)
        .collect();
    // Average over the whole tip cross-section rather than one corner: a single
    // corner also picks up the section's rotation, which is not what either
    // beam formula below predicts.
    let tip: Vec<usize> = rest
        .iter()
        .enumerate()
        .filter(|(_, p)| p.x >= length - 1e-12)
        .map(|(i, _)| i)
        .collect();
    let body = SoftBody::builder(rest, material)
        .tets(&tets)
        .pin(&clamped)
        .build()
        .unwrap();
    (body, tip)
}

/// Settings used for both beam solves below.
///
/// This is *dynamic relaxation*, not a proximal iteration: the beam is actually
/// simulated, with Rayleigh damping bleeding off the vibration until it comes
/// to rest at the static equilibrium. It converges far faster here than the
/// velocity-zeroing [`settle`] above, because the low-frequency bending mode —
/// the one Gauss–Seidel sweeps are worst at propagating — is carried by the
/// dynamics instead of by the sweep.
///
/// `damping = 2e-4` at `dt = 2e-3` means `k_d/h = 0.1`: the damping force is a
/// tenth of the elastic one at unit velocity. Raising it does *not* settle the
/// beam faster; past roughly `k_d ≈ h` the solid goes into treacle and the
/// residual stalls at a much larger value. That was measured, not guessed.
fn beam_step(gravity: Vec3) -> VbdConfig {
    VbdConfig {
        dt: 0.002,
        iterations: 40,
        gravity,
        damping: 2.0e-4,
        step_scale: 1.0,
        hessian_floor: 1e-6,
    }
}

fn mean_displacement(body: &SoftBody, group: &[usize], axis: usize) -> f64 {
    group
        .iter()
        .map(|&i| {
            let d = body.positions[i] - body.rest_positions[i];
            match axis {
                0 => d.x,
                1 => d.y,
                _ => d.z,
            }
        })
        .sum::<f64>()
        / group.len() as f64
}

/// **Axial control: a bar hanging under its own weight.**
///
/// Not the headline test — the control for it. A prismatic bar of length `L`
/// clamped at the top and loaded by its own weight extends at the free end by
///
/// ```text
/// δ = ρ g L² / (2 E)
/// ```
///
/// exactly, in linear elasticity. Stretching is the deformation mode
/// constant-strain tetrahedra represent *well*, so agreement here isolates the
/// material model, the mass lumping, the shape gradients and the static solve
/// from the discretisation error that dominates the bending test below. If this
/// one drifts, the bending number means nothing.
#[test]
fn axial_bar_extension_vs_exact_solution() {
    let (length, depth, breadth) = (1.0, 0.1, 0.1);
    let material = Material {
        youngs_modulus: 1.0e8,
        poisson_ratio: 0.0,
        density: 1000.0,
    };
    let gravity = Vec3::new(-9.81, 0.0, 0.0); // along the bar
    let (mut body, tip) = cantilever(8, 1, 1, length, depth, breadth, material);
    let mut solver = VbdSolver::new(beam_step(gravity));
    for _ in 0..3000 {
        solver.step(&mut body);
    }

    let measured = mean_displacement(&body, &tip, 0);
    let expected =
        -material.density * gravity.norm() * length * length / (2.0 * material.youngs_modulus);
    let rel_err = (measured - expected).abs() / expected.abs();
    let residual = static_residual(&body, gravity);
    println!(
        "axial bar 8x1x1: measured extension {measured:.6e} m, exact {expected:.6e} m, \
         relative error {rel_err:.4e}; static residual {residual:.3e} N"
    );

    // 1% covers the O(1/n²) constant-strain discretisation error at n = 8 plus
    // the finite-strain correction to the linear formula. Measured: 2.9e-3.
    assert!(
        rel_err < 1.0e-2,
        "axial extension is {rel_err:e} off the exact solution — the bending \
         test below cannot be interpreted until this is fixed"
    );
}

/// **Cantilever tip deflection against the Euler–Bernoulli closed form.**
///
/// A beam of length `L`, rectangular cross-section `b × d`, clamped at one end
/// and loaded only by its own weight, deflects at the tip by
///
/// ```text
/// δ = w L⁴ / (8 E I),    w = ρ b d g,    I = b d³ / 12
/// ```
///
/// Two modelling choices keep this a fair comparison. `ν = 0`, because
/// Euler–Bernoulli assumes a uniaxial stress state and a Poisson ratio adds
/// transverse coupling the beam formula knows nothing about. And `δ/L ≈ 1.5%`,
/// because the formula is a small-deflection linear result while the FEM energy
/// is finite-strain.
///
/// # The measured disagreement
///
/// **The measured tip deflection is about 5× smaller than Euler–Bernoulli:
/// −3.0e−3 m against −1.47e−2 m, a relative error of ≈ 0.80.** That is
/// reported rather than tuned away, and it is a discretisation error, not a
/// solver error:
///
/// * **Constant-strain tetrahedra lock in bending.** A linear tet has a single
///   constant strain; pure bending has a strain that varies linearly through
///   the thickness. The element cannot represent that, and what it produces
///   instead is spurious shear — parasitic stiffness that grows with the
///   element's length-to-thickness ratio. With one element through the
///   thickness this is the worst case for the worst element type, and factors
///   of several are the documented behaviour, not a surprise.
/// * **It is a discretisation error, and it refines away.** Going from 8×1×1 to
///   8×2×2 raises the measured deflection from 0.20× to 0.26× of the
///   Euler–Bernoulli value — the error moves in the direction refinement
///   predicts. (The finer mesh is not run here; it costs ten times as long,
///   and the 0.26× figure was measured at a somewhat looser residual than the
///   0.20×, so it is a direction, not a converged second data point.)
/// * **It is not the solver.** `axial_bar_extension_vs_exact_solution` above
///   uses the same material, the same mesh resolution and the same solve, in
///   the deformation mode linear tets handle well, and lands within 0.3% of the
///   exact answer.
///
/// The assertion below therefore brackets the *measured* value rather than the
/// reference: it is a regression test on a known discretisation error. If it
/// starts failing, report the new number.
#[test]
fn cantilever_tip_deflection_vs_euler_bernoulli() {
    let (length, breadth, depth) = (1.0, 0.1, 0.1);
    let material = Material {
        youngs_modulus: 1.0e8,
        poisson_ratio: 0.0,
        density: 1000.0,
    };
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let (mut body, tip) = cantilever(8, 1, 1, length, depth, breadth, material);
    let mut solver = VbdSolver::new(beam_step(gravity));
    for _ in 0..3000 {
        solver.step(&mut body);
    }

    let measured = mean_displacement(&body, &tip, 1);
    let second_moment = breadth * depth.powi(3) / 12.0;
    let load = material.density * breadth * depth * gravity.norm();
    let expected = -load * length.powi(4) / (8.0 * material.youngs_modulus * second_moment);
    let rel_err = (measured - expected).abs() / expected.abs();
    let residual = static_residual(&body, gravity);
    let vertex_weight = body.masses[tip[0]] * gravity.norm();

    println!(
        "cantilever 8x1x1: measured tip deflection {measured:.6e} m, \
         Euler-Bernoulli {expected:.6e} m, relative error {rel_err:.4e} \
         (measured/reference = {:.3}); static residual {residual:.3e} N \
         ({:.2e} of a vertex weight)",
        measured / expected,
        residual / vertex_weight
    );

    // The solve has to have converged, or the deflection number is noise.
    // 3000 steps leaves the residual at ~13% of a vertex weight; tripling to
    // 9000 steps takes it to 4.6% and moves the measured ratio from 0.202 to
    // 0.208, so the number reported here is converged to about 3%. 3000 is
    // what the test runs, because 9000 costs three times as long to say the
    // same thing.
    assert!(
        residual < 0.15 * vertex_weight,
        "dynamic relaxation did not converge: residual {residual:e} N against a \
         {vertex_weight:e} N vertex weight"
    );
    assert!(measured < 0.0, "the beam did not bend downward");
    // Brackets the measured 0.202; see the doc comment for why this is a
    // regression bound on a discretisation error and not an accuracy claim.
    let ratio = measured / expected;
    assert!(
        (0.15..0.30).contains(&ratio),
        "tip deflection is {ratio:.3} of Euler-Bernoulli (measured {measured:e} m, \
         reference {expected:e} m, relative error {rel_err:.4e}) — outside the \
         measured range for this mesh; report the new number"
    );
}

/// **Energy behaviour over free vibration.**
///
/// A clamped beam plucked once, with gravity and damping both off, so total
/// mechanical energy is a conserved quantity of the continuous problem. VBD
/// with a finite iteration count is *not* conservative and is not claimed to
/// be: backward Euler is unconditionally dissipative, so the drift must be
/// negative — energy leaving the system, never entering it. Energy *gain*
/// would mean the solver is doing something an implicit integrator cannot do,
/// and that is what this test is really guarding against.
#[test]
fn energy_drift_over_free_vibration() {
    let (rest, tets) = mesh::tet_box(6, 1, 1, Vec3::new(0.3, 0.05, 0.05));
    let clamped: Vec<usize> = rest
        .iter()
        .enumerate()
        .filter(|(_, p)| p.x <= 0.0)
        .map(|(i, _)| i)
        .collect();
    let material = Material {
        youngs_modulus: 1.0e6,
        poisson_ratio: 0.3,
        density: 1000.0,
    };
    let mut body = SoftBody::builder(rest, material)
        .tets(&tets)
        .pin(&clamped)
        .build()
        .unwrap();

    // A pristine copy, to read the rest-state energy offset off later.
    let reference = body.clone();

    // Pluck: transverse velocity growing linearly along the beam.
    for i in 0..body.len() {
        if !body.pinned[i] {
            let s = body.rest_positions[i].x / 0.3;
            body.velocities[i] = Vec3::new(0.0, 0.5 * s, 0.0);
        }
    }

    let gravity = Vec3::zero();
    let mut solver = VbdSolver::new(VbdConfig {
        dt: 1.0e-4,
        iterations: 20,
        gravity,
        damping: 0.0,
        ..VbdConfig::default()
    });

    // Stable Neo-Hookean has a non-zero energy density at F = I (the
    // `−μ/2 log(1+I_C)` term), so the absolute total energy carries a large
    // constant offset. Drift has to be measured against the *excess* over the
    // rest state, or the ratio is dominated — and sign-flipped — by a constant
    // that never changes.
    let rest_offset = reference.elastic_energy();
    let initial = body.total_energy(gravity) - rest_offset;
    let mut peak = initial;
    for _ in 0..2000 {
        solver.step(&mut body);
        peak = peak.max(body.total_energy(gravity) - rest_offset);
    }
    let final_energy = body.total_energy(gravity) - rest_offset;
    let drift = (final_energy - initial) / initial;
    let overshoot = (peak - initial) / initial;

    println!(
        "free vibration, dt = 1e-4 s x 2000 steps, 20 iterations: \
         E0 (excess over rest) = {initial:.6e} J, E = {final_energy:.6e} J, \
         drift = {drift:.4e} (relative), peak overshoot = {overshoot:.4e}"
    );

    assert!(final_energy.is_finite());
    // Dissipative, as backward Euler must be. The small positive allowance
    // absorbs the fact that a partially-converged sweep is not exactly
    // backward Euler, and is far below anything that could grow.
    assert!(
        overshoot < 1e-3,
        "energy grew by {overshoot:.3e} — the step is injecting energy"
    );
    assert!(drift <= 0.0, "net energy gain of {drift:.3e}");
}

/// One symplectic-Euler step, for the explicit-integrator comparison below.
/// This is deliberately the *best* simple explicit scheme (symplectic Euler
/// beats forward Euler on stability), so the comparison is not rigged.
fn explicit_step(body: &mut SoftBody, dt: f64, gravity: Vec3) {
    let forces = body.elastic_forces();
    for (i, force) in forces.iter().enumerate() {
        if body.pinned[i] {
            continue;
        }
        body.velocities[i] += (*force / body.masses[i] + gravity) * dt;
        let v = body.velocities[i];
        body.positions[i] += v * dt;
    }
}

fn beam_for_stability() -> SoftBody {
    let (rest, tets) = mesh::tet_box(8, 1, 1, Vec3::new(0.4, 0.05, 0.05));
    let clamped: Vec<usize> = rest
        .iter()
        .enumerate()
        .filter(|(_, p)| p.x <= 0.0)
        .map(|(i, _)| i)
        .collect();
    SoftBody::builder(
        rest,
        Material {
            youngs_modulus: 1.0e6,
            poisson_ratio: 0.3,
            density: 1000.0,
        },
    )
    .tets(&tets)
    .pin(&clamped)
    .build()
    .unwrap()
}

/// Largest displacement of any vertex from rest, in metres. `f64::INFINITY` if
/// the state has gone non-finite.
///
/// The explicit `is_finite` sweep is not decoration: `f64::max` *ignores* NaN,
/// so folding with it over a diverged state cheerfully reports `0.0` — a
/// blown-up simulation looking like a perfectly still one.
fn excursion(body: &SoftBody) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..body.len() {
        let d = (body.positions[i] - body.rest_positions[i]).norm();
        if !d.is_finite() {
            return f64::INFINITY;
        }
        worst = worst.max(d);
    }
    worst
}

/// **Stability at a timestep where explicit integration explodes.**
///
/// This is VBD's headline claim, so it gets its own test and its own numbers.
///
/// The scene is a 0.4 m clamped beam of `E = 1 MPa`, `ρ = 1000 kg/m³` meshed at
/// `Δx = 0.05 m`. Wave speed is `√(E/ρ) ≈ 31.6 m/s`, so the CFL limit for an
/// explicit scheme is about `Δx/c ≈ 1.6 ms`. The test:
///
/// 1. runs symplectic Euler at `1/60 s` — ten times the CFL limit — and asserts
///    it diverges, so the comparison is against a real limit and not a claim;
/// 2. runs VBD on the same scene at the same timestep and asserts it stays
///    bounded;
/// 3. sweeps the timestep upward on a gravity-free, plucked beam and reports
///    the largest one VBD survives.
///
/// **Measured: no divergence at any timestep tested, up to `h = 256 s` — about
/// 1.6 × 10⁵ times the explicit CFL limit for this scene.** The sweep stops
/// there because it stopped being informative, not because it found a bound.
///
/// "Survives" means finite, and no vertex displaced more than twice what its
/// initial momentum could carry it over the simulated span. That is a
/// stability criterion, **not** an accuracy one: at `h = 1 s` a single step is
/// effectively a static solve and the trajectory is nonsense — bounded
/// nonsense. See the crate docs on what "unconditionally stable" does and does
/// not buy.
#[test]
fn stable_at_a_timestep_where_explicit_explodes() {
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let dt = 1.0 / 60.0;

    let mut explicit = beam_for_stability();
    for _ in 0..60 {
        explicit_step(&mut explicit, dt, gravity);
    }
    let explicit_excursion = excursion(&explicit);
    println!(
        "symplectic Euler at dt = {dt:.4} s (~10x CFL): max excursion after 1 s = \
         {explicit_excursion:.3e} m"
    );
    assert!(
        !explicit_excursion.is_finite() || explicit_excursion > 1.0e3,
        "the explicit reference did not blow up ({explicit_excursion:e} m) — \
         the stability comparison below is meaningless without that"
    );

    let mut implicit = beam_for_stability();
    let mut solver = VbdSolver::new(VbdConfig {
        dt,
        iterations: 20,
        gravity,
        ..VbdConfig::default()
    });
    for _ in 0..60 {
        solver.step(&mut implicit);
    }
    let vbd_excursion = excursion(&implicit);
    println!("VBD at dt = {dt:.4} s: max excursion after 1 s = {vbd_excursion:.6e} m");
    assert!(
        vbd_excursion.is_finite() && vbd_excursion < 1.0,
        "VBD did not stay bounded: {vbd_excursion:e} m"
    );

    // How far up does it go?
    //
    // The sweep runs a *gravity-free, plucked* beam rather than the falling one
    // above, and that matters: with gravity on, a single step at `h = 2 s`
    // moves every vertex `h²g ≈ 39 m` simply by predicting free fall. That is
    // correct ballistic motion, not instability, and any excursion threshold
    // would flag it as divergence. Removing gravity leaves the elastic response
    // as the only thing that can grow, which is what "stable" is a claim about.
    let mut largest = 0.0f64;
    for &dt in &[
        1.0 / 60.0,
        1.0 / 30.0,
        0.1,
        0.25,
        1.0,
        4.0,
        16.0,
        64.0,
        256.0,
    ] {
        let mut body = beam_for_stability();
        for i in 0..body.len() {
            if !body.pinned[i] {
                // A hard transverse pluck: 5 m/s on a 0.4 m beam is a violent
                // initial condition, deliberately.
                body.velocities[i] = Vec3::new(0.0, 5.0, 0.0);
            }
        }
        let mut solver = VbdSolver::new(VbdConfig {
            dt,
            iterations: 20,
            gravity: Vec3::zero(),
            ..VbdConfig::default()
        });
        let steps = (1.0f64 / dt).ceil().max(1.0) as usize;
        for _ in 0..steps {
            solver.step(&mut body);
        }
        let e = excursion(&body);
        // The criterion has to scale with the simulated span, not be a fixed
        // distance: at `h = 64 s` the *first* inertial prediction already
        // carries a 5 m/s vertex 320 m, and that is correct ballistic motion,
        // not instability. The bound is twice what the initial momentum can
        // produce over the simulated time — exceeding it means the elastic
        // response added energy, which is the failure mode being tested for.
        let span = steps as f64 * dt;
        let ballistic = 5.0 * span;
        let ok = e.is_finite() && e < 2.0 * ballistic;
        println!(
            "  dt = {dt:.4} s ({steps} steps, {span:.0} s simulated, plucked at 5 m/s): \
             excursion {e:.4e} m vs {ballistic:.4e} m ballistic — {}",
            if ok { "bounded" } else { "DIVERGED" }
        );
        if ok {
            largest = largest.max(dt);
        }
    }
    println!(
        "largest bounded timestep reached: {largest:.4} s (~{:.1e}x the ~1.6 ms \
         explicit CFL limit for this scene)",
        largest / 1.6e-3
    );
    assert!(
        largest >= 1.0,
        "VBD only reached dt = {largest} s, which does not support the \
         large-timestep claim"
    );
}

/// **The indefinite-Hessian guard, in the one configuration that needs it.**
///
/// A chain of springs compressed well past its rest length has a genuinely
/// indefinite per-vertex Hessian (see `energy::tests`). Without the eigenvalue
/// clamp the local solve takes an uphill step through a near-singular block;
/// with it, the configuration stays bounded and the energy does not grow.
#[test]
fn indefinite_hessian_guard_keeps_a_compressed_chain_bounded() {
    // Four vertices in a line, with springs whose rest length is four times
    // their actual spacing: every spring is deeply compressed, every interior
    // Hessian block has two strongly negative eigenvalues.
    let rest: Vec<Vec3> = (0..4)
        .map(|i| Vec3::new(0.05 * i as f64, 0.0, 0.0))
        .collect();
    let mut body = SoftBody::builder(rest, Material::default())
        .spring(0, 1, 500.0)
        .spring(1, 2, 500.0)
        .spring(2, 3, 500.0)
        .pin(&[0])
        .build()
        .unwrap();
    // Springs take their rest length from the rest configuration, so the chain
    // is compressed by moving the vertices closer than they were built. The
    // tiny z offset breaks the exact collinearity: a perfectly straight
    // compressed chain is an unstable *equilibrium*, and the guard's behaviour
    // is only visible once it is perturbed off it.
    for i in 1..body.len() {
        body.positions[i] = Vec3::new(0.01 * i as f64, 0.0, 1.0e-4 * i as f64);
    }
    for m in body.masses.iter_mut() {
        *m = 0.01;
    }

    let mut solver = VbdSolver::new(VbdConfig {
        dt: 1.0 / 60.0,
        iterations: 20,
        gravity: Vec3::zero(),
        ..VbdConfig::default()
    });
    for _ in 0..600 {
        solver.step(&mut body);
    }

    let e = excursion(&body);
    let energy = body.elastic_energy();
    println!(
        "compressed spring chain after 600 steps: max excursion {e:.4e} m, \
         elastic energy {energy:.6e} J"
    );
    assert!(
        e.is_finite() && e < 1.0,
        "buckling chain went unstable: {e:e} m"
    );
    assert!(energy.is_finite());
}

/// **Determinism.** Same setup in, identical bits out. `HashMap` iteration
/// order is the usual way this breaks; nothing on the solve path uses one, and
/// this test is what keeps it that way.
#[test]
fn stepping_is_bit_identical_across_runs() {
    let run = || {
        let (rest, tets) = mesh::tet_box(4, 2, 2, Vec3::new(0.4, 0.1, 0.1));
        let clamped: Vec<usize> = rest
            .iter()
            .enumerate()
            .filter(|(_, p)| p.x <= 0.0)
            .map(|(i, _)| i)
            .collect();
        let mut body = SoftBody::builder(rest, Material::default())
            .tets(&tets)
            .pin(&clamped)
            .build()
            .unwrap();
        let mut solver = VbdSolver::new(VbdConfig {
            dt: 1.0 / 60.0,
            iterations: 12,
            damping: 0.02,
            ..VbdConfig::default()
        });
        for _ in 0..120 {
            solver.step(&mut body);
        }
        (
            body.positions.clone(),
            body.velocities.clone(),
            body.colors().to_vec(),
        )
    };

    let (pa, va, ca) = run();
    let (pb, vb, cb) = run();
    assert_eq!(ca, cb, "colouring is not reproducible");
    for i in 0..pa.len() {
        assert_eq!(
            pa[i].as_array(),
            pb[i].as_array(),
            "position {i} differs between runs"
        );
        assert_eq!(
            va[i].as_array(),
            vb[i].as_array(),
            "velocity {i} differs between runs"
        );
    }
}
