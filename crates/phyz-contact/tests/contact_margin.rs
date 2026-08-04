//! Contact margin: the normal force must ramp out, not fall off a cliff.
//!
//! `find_ground_contacts` used to filter candidates with a hard `p.z <
//! ground_height`. With *soft* contact that predicate is wrong in a way that a
//! rigid-contact predicate would not be: a candidate at exactly zero depth
//! still gets impedance `solimp.dmin` (0.9 by default), so it carries nearly
//! its full share of the load right up to the instant it stops existing.
//!
//! Forensics on a K1 humanoid stance measured the least-loaded foot corner
//! carrying **22.3 N — 11% of body weight — on the step before it left the
//! contact set**, and 0 N on the next. Total vertical force was conserved (the
//! load redistributed onto the remaining corners), so nothing leaked; the
//! failure is that the contact *set* jumped, underneath a balancing controller,
//! at the moment it could least afford it.
//!
//! These tests pin the three properties the fix has to have simultaneously:
//! the force is continuous through the transition, it never becomes adhesive,
//! and a resting body still carries exactly its own weight.

use phyz_collision::Collision;
use phyz_contact::{
    ContactMaterial, ContactSolverConfig, assemble, find_ground_contacts, solve_contacts,
};
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder, State};

const DT: f64 = 1e-3;
const MASS: f64 = 2.0;
const RADIUS: f64 = 0.1;

/// A single free sphere of radius [`RADIUS`] over a ground plane at `z = 0`.
fn sphere_model() -> Model {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(DT)
        .add_free_body(
            "ball",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::sphere(MASS, RADIUS),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Sphere { radius: RADIUS });
    model
}

fn geometries(model: &Model) -> Vec<Option<Geometry>> {
    model.bodies.iter().map(|b| b.geometry.clone()).collect()
}

/// Place the single free body's centre at height `z` and refresh FK, so
/// `state.body_xform` (which the narrow phase reads) and the Jacobians the
/// assembly builds agree.
fn place(model: &Model, state: &mut State, z: f64) {
    // Free joint q = [wx, wy, wz, x, y, z].
    state.q[5] = z;
    let (xforms, _) = phyz_rigid::forward_kinematics(model, state);
    state.body_xform = xforms;
}

/// Solve one step's contacts for `contacts` and return the total *world* normal
/// force, in newtons.
fn normal_force(
    model: &Model,
    state: &State,
    contacts: &[Collision],
    material: &ContactMaterial,
) -> f64 {
    if contacts.is_empty() {
        return 0.0;
    }
    // Free velocity after one step of gravity alone.
    let mut free_qd = DVec::zeros(model.nv);
    free_qd[5] = -GRAVITY * DT;

    let cfg = ContactSolverConfig::simulation();
    let materials = vec![material.clone(); model.bodies.len()];
    let asm = assemble(model, state, contacts, &materials, &free_qd, DT, &cfg);
    let sol = solve_contacts(&asm.problem, &cfg);
    assert!(
        sol.converged,
        "contact solve must converge to be meaningful"
    );
    // Every ground contact's normal is +z here, so the normal components sum
    // directly. Impulse -> force over the step.
    sol.impulses.iter().map(|f| f.x).sum::<f64>() / DT
}

/// The measurement the bug report is about, run as a sweep: raise the body
/// slowly through the moment of separation and watch the supporting force.
///
/// With zero margin this is a cliff — the force is `dmin * m * g` on one sample
/// and exactly 0 on the next. With a margin it is a ramp, and the assertion is
/// that no single micron-sized rise changes the force by more than a small
/// fraction of the body weight.
#[test]
fn normal_force_ramps_continuously_through_the_margin() {
    let model = sphere_model();
    let geoms = geometries(&model);
    let mut state = model.default_state();

    let material = ContactMaterial::default();
    let margin = material.margin;
    assert!(
        margin > 0.0,
        "the default material must ask for a margin, or nothing below is tested"
    );

    let weight = MASS * GRAVITY;

    // Sweep the centre height from slightly penetrating to past the margin.
    // Contact is made at z = RADIUS (bottom exactly on the plane) and the
    // candidate leaves the set at z = RADIUS + margin.
    let lo = RADIUS - 0.5 * margin;
    let hi = RADIUS + 1.5 * margin;
    let steps = 4000;
    // Rise per sample: 0.5 micron at these numbers, i.e. far finer than any
    // real timestep would move a balancing body.
    let dz = (hi - lo) / steps as f64;

    let mut samples: Vec<(f64, f64)> = Vec::with_capacity(steps + 1);
    for k in 0..=steps {
        let z = lo + dz * k as f64;
        place(&model, &mut state, z);
        let contacts = find_ground_contacts(&state, &geoms, 0.0, margin);
        samples.push((z, normal_force(&model, &state, &contacts, &material)));
    }

    // 1. Continuity. The largest single-sample jump must be a small fraction of
    //    body weight — contrast with 22.3 N of a 203 N body (11%) in one step
    //    on the K1, and with `dmin * m * g` = 90% here at zero margin.
    let mut worst_jump = 0.0f64;
    let mut worst_at = 0.0f64;
    for w in samples.windows(2) {
        let jump = (w[1].1 - w[0].1).abs();
        if jump > worst_jump {
            worst_jump = jump;
            worst_at = w[1].0;
        }
    }
    assert!(
        worst_jump < 0.01 * weight,
        "normal force must not step: worst jump {worst_jump:.6} N \
         ({:.3}% of the {weight:.3} N weight) at z = {worst_at:.6}",
        100.0 * worst_jump / weight
    );

    // 2. Exactly zero at and beyond the margin. The detection predicate is
    //    strict, so the candidate at exactly `margin` is already gone — and by
    //    then its impedance has ramped to zero, which is what makes dropping it
    //    a no-op instead of a step.
    for &(z, f) in &samples {
        if z >= RADIUS + margin {
            assert_eq!(
                f,
                0.0,
                "force at gap {:.9} m (>= margin {margin}) must be exactly zero, got {f}",
                z - RADIUS
            );
        }
    }
    place(&model, &mut state, RADIUS + margin);
    assert!(
        find_ground_contacts(&state, &geoms, 0.0, margin).is_empty(),
        "a candidate exactly at the margin must not be reported"
    );

    // 3. Monotone: rising can only ever unload the contact, never load it.
    for w in samples.windows(2) {
        assert!(
            w[1].1 <= w[0].1 + 1e-9,
            "force must decrease as the body rises: {} N at z={} then {} N at z={}",
            w[0].1,
            w[0].0,
            w[1].1,
            w[1].0
        );
    }

    // 4. The ramp is a ramp, not a shifted cliff: it actually spends the band
    //    handing the load back. Check the force at the band's midpoint sits
    //    strictly between the endpoints by a real margin.
    let at = |z: f64| -> f64 {
        let i = (((z - lo) / dz).round() as usize).min(steps);
        samples[i].1
    };
    let touching = at(RADIUS);
    let mid = at(RADIUS + 0.5 * margin);
    assert!(
        touching > 0.5 * weight,
        "a just-touching contact should still carry most of the weight: {touching} N"
    );
    assert!(
        mid > 0.05 * touching && mid < 0.95 * touching,
        "midband force {mid} N should be partway between {touching} N and 0"
    );

    // 5. And the zero-margin behaviour is the cliff, so the test above is
    //    measuring a real change rather than a property the old code had.
    place(&model, &mut state, RADIUS - 1e-12);
    let just_below = normal_force(
        &model,
        &state,
        &find_ground_contacts(&state, &geoms, 0.0, 0.0),
        &material,
    );
    place(&model, &mut state, RADIUS + 1e-12);
    let just_above = normal_force(
        &model,
        &state,
        &find_ground_contacts(&state, &geoms, 0.0, 0.0),
        &material,
    );
    assert_eq!(just_above, 0.0);
    assert!(
        just_below > 0.5 * weight,
        "zero margin must still be a cliff ({just_below} N -> 0 N across 2 pm); \
         if this ever fails the continuity test above proves nothing"
    );
}

/// A contact inside the margin must never *pull*.
///
/// Impedance going to zero means "infinitely soft", not "attractive", and the
/// normal impulse stays on the `f_n >= 0` half-line of the friction cone. A
/// separated contact also gets no stabilization bias (`from_material` clamps
/// the violation at zero), so there is nothing pushing the surfaces together
/// or apart.
#[test]
fn a_separated_contact_never_pulls_the_body_back_down() {
    let model = sphere_model();
    let geoms = geometries(&model);
    let mut state = model.default_state();
    let material = ContactMaterial::default();
    let margin = material.margin;
    let cfg = ContactSolverConfig::simulation();
    let materials = vec![material.clone(); model.bodies.len()];

    for frac in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
        let z = RADIUS + frac * margin;
        place(&model, &mut state, z);
        let contacts = find_ground_contacts(&state, &geoms, 0.0, margin);
        assert_eq!(
            contacts.len(),
            1,
            "gap {:.6} m is inside the margin and must still be detected",
            frac * margin
        );
        assert!(
            contacts[0].penetration_depth < 0.0,
            "a separated candidate must report a negative depth, got {}",
            contacts[0].penetration_depth
        );

        // The body is rising, gravity notwithstanding: a fast separating
        // velocity that no contact has any business opposing.
        let mut free_qd = DVec::zeros(model.nv);
        free_qd[5] = 0.5;

        let asm = assemble(&model, &state, &contacts, &materials, &free_qd, DT, &cfg);
        let sol = solve_contacts(&asm.problem, &cfg);
        assert!(sol.converged);

        // No impulse at all, and in particular none pointing down.
        for f in &sol.impulses {
            assert!(
                f.x >= 0.0,
                "normal impulse must never be adhesive, got {}",
                f.x
            );
            assert!(
                f.x < 1e-12,
                "a separating body must not be resisted, got {} N·s",
                f.x
            );
        }

        // And the body is not slowed down: no energy is removed on the way up
        // (nor, obviously, injected).
        let dv = asm.velocity_delta(&sol.impulses);
        let v_after = free_qd[5] + dv[5];
        assert!(
            v_after >= free_qd[5] - 1e-12,
            "separating velocity {} was dragged down to {v_after}",
            free_qd[5]
        );
        assert!(
            v_after <= free_qd[5] + 1e-12,
            "separating velocity {} was boosted to {v_after}: energy injected",
            free_qd[5]
        );
    }
}

/// Sitting still and merely *hovering* inside the margin is also not enough to
/// generate a force: with no approach velocity there is nothing to cancel.
#[test]
fn hovering_inside_the_margin_generates_no_force() {
    let model = sphere_model();
    let geoms = geometries(&model);
    let mut state = model.default_state();
    let material = ContactMaterial::default();
    let margin = material.margin;
    let cfg = ContactSolverConfig::simulation();
    let materials = vec![material.clone(); model.bodies.len()];

    place(&model, &mut state, RADIUS + 0.5 * margin);
    let contacts = find_ground_contacts(&state, &geoms, 0.0, margin);
    assert_eq!(contacts.len(), 1);

    // Free velocity is exactly zero: no gravity step, no motion.
    let free_qd = DVec::zeros(model.nv);
    let asm = assemble(&model, &state, &contacts, &materials, &free_qd, DT, &cfg);
    let sol = solve_contacts(&asm.problem, &cfg);
    assert!(sol.converged);
    for f in &sol.impulses {
        assert!(f.norm() < 1e-12, "static hover produced an impulse {f:?}");
    }
    // Explicitly: the row carries no stabilization bias, so it cannot pump.
    assert_eq!(asm.problem.rows[0].bias, 0.0);
}

/// Force balance is untouched: a box settled on the ground carries `m*g`, split
/// across whatever corners the margin now admits.
#[test]
fn a_resting_box_still_carries_exactly_its_own_weight() {
    let half = Vec3::new(0.2, 0.15, 0.1);
    let mass = 3.0;
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(DT)
        .add_free_body(
            "box",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(
                mass,
                Vec3::zeros(),
                Mat3::from_diagonal(&Vec3::new(0.05, 0.05, 0.05)),
            ),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box { half_extents: half });
    let geoms = geometries(&model);

    let material = ContactMaterial::default();
    let materials = vec![material.clone(); model.bodies.len()];
    let cfg = ContactSolverConfig::simulation();

    let mut state = model.default_state();
    place(&model, &mut state, half.z);

    // Step the box until it settles into its soft-contact equilibrium.
    let mut total = 0.0;
    let mut ncontacts = 0;
    for _ in 0..3000 {
        let (xforms, _) = phyz_rigid::forward_kinematics(&model, &state);
        state.body_xform = xforms;
        let contacts = find_ground_contacts(&state, &geoms, 0.0, material.margin);

        let qdd = phyz_rigid::aba(&model, &state);
        let free_qd = &state.v + &(&qdd * DT);

        if contacts.is_empty() {
            state.v = free_qd;
            total = 0.0;
            ncontacts = 0;
        } else {
            let asm = assemble(&model, &state, &contacts, &materials, &free_qd, DT, &cfg);
            let sol = solve_contacts(&asm.problem, &cfg);
            assert!(sol.converged);
            state.v = &free_qd + &asm.velocity_delta(&sol.impulses);
            total = sol.impulses.iter().map(|f| f.x).sum::<f64>() / DT;
            ncontacts = contacts.len();
        }
        let v = state.v.clone();
        phyz_rigid::integrate_configuration(&model, state.q.as_mut_slice(), v.as_slice(), DT);
        state.time += DT;
    }

    assert!(ncontacts >= 4, "a flat box should rest on four corners");
    let weight = mass * GRAVITY;
    assert!(
        (total - weight).abs() / weight < 1e-3,
        "resting contacts must sum to m*g: {total} N vs {weight} N over {ncontacts} contacts"
    );
    // Settled, not still falling.
    assert!(
        state.v[5].abs() < 1e-4,
        "box has not settled: vz = {}",
        state.v[5]
    );
}

/// Detection-level properties of the margin, independent of the solve.
#[test]
fn the_margin_widens_detection_without_changing_penetrating_depths() {
    let model = sphere_model();
    let geoms = geometries(&model);
    let mut state = model.default_state();
    let margin = 1e-3;

    // Penetrating: identical result with or without a margin.
    place(&model, &mut state, RADIUS - 0.05);
    let with = find_ground_contacts(&state, &geoms, 0.0, margin);
    let without = find_ground_contacts(&state, &geoms, 0.0, 0.0);
    assert_eq!(with.len(), 1);
    assert_eq!(without.len(), 1);
    assert!((with[0].penetration_depth - 0.05).abs() < 1e-12);
    assert!((without[0].penetration_depth - 0.05).abs() < 1e-12);

    // Inside the band: kept, with a negative depth equal to minus the gap.
    let gap = 0.4 * margin;
    place(&model, &mut state, RADIUS + gap);
    let with = find_ground_contacts(&state, &geoms, 0.0, margin);
    assert_eq!(with.len(), 1);
    assert!((with[0].penetration_depth + gap).abs() < 1e-12);
    assert!(find_ground_contacts(&state, &geoms, 0.0, 0.0).is_empty());

    // Beyond it: gone.
    place(&model, &mut state, RADIUS + 1.0001 * margin);
    assert!(find_ground_contacts(&state, &geoms, 0.0, margin).is_empty());

    // A negative margin is clamped to zero rather than hiding penetration.
    place(&model, &mut state, RADIUS - 0.05);
    assert_eq!(find_ground_contacts(&state, &geoms, 0.0, -1.0).len(), 1);
}

/// Not an assertion — a printout of the ramp, so the numbers in the commit
/// message can be regenerated. `cargo test -p phyz-contact --test contact_margin
/// -- --ignored --nocapture print_the_ramp`
#[test]
#[ignore]
fn print_the_ramp() {
    let model = sphere_model();
    let geoms = geometries(&model);
    let mut state = model.default_state();
    let material = ContactMaterial::default();
    let margin = material.margin;
    println!("weight = {:.4} N", MASS * GRAVITY);
    for k in -2..=12 {
        let gap = k as f64 * margin / 10.0;
        place(&model, &mut state, RADIUS + gap);
        for m in [0.0, margin] {
            let c = find_ground_contacts(&state, &geoms, 0.0, m);
            let f = normal_force(&model, &state, &c, &material);
            print!("  margin={m:.4} gap={gap:+.5} n={} F={f:8.4} N", c.len());
        }
        println!();
    }

    // Worst single-sample jump over a fine sweep of the whole band.
    let (lo, hi, steps) = (RADIUS - 0.5 * margin, RADIUS + 1.5 * margin, 4000);
    let dz = (hi - lo) / steps as f64;
    let mut prev: Option<f64> = None;
    let mut worst = 0.0f64;
    for k in 0..=steps {
        let z = lo + dz * k as f64;
        place(&model, &mut state, z);
        let f = normal_force(
            &model,
            &state,
            &find_ground_contacts(&state, &geoms, 0.0, margin),
            &material,
        );
        if let Some(p) = prev {
            worst = worst.max((f - p).abs());
        }
        prev = Some(f);
    }
    println!(
        "worst single-sample jump over dz = {dz:.3e} m: {worst:.6} N ({:.4}% of weight)",
        100.0 * worst / (MASS * GRAVITY)
    );
}
