//! The exact per-corner friction step's gradient, against a finite difference,
//! on the audit's own incline box (phyz docs/contact-audit.md section 3).
//!
//! A 0.2 m, 1 kg box on mu 0.5, gravity tilted by the incline angle, stepped
//! exactly as `Simulator::step_with_contacts` steps it until it has settled
//! (26 deg: stuck; 35 deg: sliding with four coupled corners, the case the
//! radial clamp got wrong). At that state the contact problem is assembled
//! and three derivatives of its solve are checked along one parameter
//! direction `(dA, dc)`:
//!
//! 1. forward mode (`contact_solve_differential`) against a central
//!    difference of the solve, relative error < 1e-6;
//! 2. reverse mode (`contact_solve_differential_transpose`, the tape) against
//!    forward mode: `<bar_f, df> = <bar_A, dA> + <bar_c, dc>`;
//! 3. the IFT map (`FixedPointSensitivity`, the max-dissipation fixed point)
//!    against the same central difference.

use phyz::phyz_contact::{
    ContactMaterial, ContactProblem, ContactSolverConfig, assemble,
    contact_solve_differential, contact_solve_differential_transpose, find_contacts,
    find_ground_contacts_model, solve_contacts_warm,
};
use phyz::phyz_contact::gradient::FixedPointSensitivity;
use phyz::phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz::phyz_model::{Geometry, Model, ModelBuilder, State};
use phyz::phyz_rigid::{
    aba, forward_kinematics, integrate_configuration, rotate_free_joint_velocities,
    strip_free_joint_coriolis,
};

const MU: f64 = 0.5;
const HALF: f64 = 0.1;

fn cfg() -> ContactSolverConfig {
    let mut c = ContactSolverConfig::simulation();
    // Tight, and no stall exit: a central difference of a solve that stops
    // after a parameter-dependent number of sweeps is not a derivative.
    c.tolerance = 1e-13;
    c
}

fn incline_box(deg: f64, dt: f64) -> Model {
    let th = deg.to_radians();
    let g = Vec3::new(GRAVITY * th.sin(), 0.0, -GRAVITY * th.cos());
    let i = 1.0 / 3.0 * 2.0 * HALF * HALF;
    let mut model = ModelBuilder::new()
        .gravity(g)
        .dt(dt)
        .add_free_body(
            "box",
            -1,
            SpatialTransform::identity(),
            SpatialInertia::new(1.0, Vec3::zeros(), Mat3::from_diagonal(&Vec3::new(i, i, i))),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Box { half_extents: Vec3::new(HALF, HALF, HALF) });
    model
}

fn material() -> ContactMaterial {
    ContactMaterial { friction: MU, ..ContactMaterial::default() }
}

/// The contact problem of one step from `state`, as the simulator builds it.
fn problem_at(model: &Model, state: &mut State) -> Option<(ContactProblem, Vec<f64>)> {
    let (x, _) = forward_kinematics(model, state);
    state.body_xform = x;
    let mat = material();
    let mut contacts = find_ground_contacts_model(model, state, 0.0, mat.margin);
    contacts.extend(find_contacts(model, state, mat.margin));
    let mut qdd = aba(model, state);
    strip_free_joint_coriolis(model, state.v.clone().as_slice(), qdd.as_mut_slice());
    let free_qd = &state.v + &(&qdd * model.dt);
    if contacts.is_empty() {
        return None;
    }
    let materials = model.contact_materials(&mat);
    let asm = assemble(model, state, &contacts, &materials, &free_qd, model.dt, &cfg());
    Some((asm.problem, free_qd.as_slice().to_vec()))
}

/// Step `secs` of simulation, the `Simulator::step_with_contacts` sequence.
fn settle(model: &Model, secs: f64) -> State {
    let mut state = model.default_state();
    state.q[5] = HALF;
    let dt = model.dt;
    let mat = material();
    for _ in 0..(secs / dt).round() as usize {
        let v_before = state.v.clone();
        let (x, _) = forward_kinematics(model, &state);
        state.body_xform = x;
        let mut contacts = find_ground_contacts_model(model, &state, 0.0, mat.margin);
        contacts.extend(find_contacts(model, &state, mat.margin));
        let mut qdd = aba(model, &state);
        strip_free_joint_coriolis(model, v_before.as_slice(), qdd.as_mut_slice());
        let free_qd = &state.v + &(&qdd * dt);
        if contacts.is_empty() {
            state.v = free_qd;
        } else {
            let materials = model.contact_materials(&mat);
            let asm = assemble(model, &state, &contacts, &materials, &free_qd, dt, &cfg());
            let sol = solve_contacts_warm(&asm.problem, &cfg(), &[]);
            state.v = &free_qd + &asm.velocity_delta(&sol.impulses);
        }
        rotate_free_joint_velocities(model, v_before.as_slice(), state.v.as_mut_slice(), dt);
        let v = state.v.clone();
        integrate_configuration(model, state.q.as_mut_slice(), v.as_slice(), dt);
        state.time += dt;
    }
    state
}

/// A symmetric `dA` and a `dc`, deterministic.
fn direction(dim: usize, scale: f64) -> (Vec<f64>, Vec<f64>) {
    let mut d_apr = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let v = scale * 0.05 * ((i * 31 + j * 17) as f64 * 0.37).sin();
            d_apr[i * dim + j] = v;
            d_apr[j * dim + i] = v;
        }
    }
    let dc = (0..dim).map(|i| scale * 0.08 * ((i * 13) as f64 * 0.29).cos()).collect();
    (d_apr, dc)
}

fn perturbed(base: &ContactProblem, d_apr: &[f64], dc: &[f64], eps: f64) -> ContactProblem {
    let mut p = base.clone();
    for (a, da) in p.delassus.iter_mut().zip(d_apr) {
        *a += eps * da;
    }
    for (b, db) in p.free_velocity.iter_mut().zip(dc) {
        *b += eps * db;
    }
    p
}

fn flat(v: &[Vec3]) -> Vec<f64> {
    v.iter().flat_map(|x| [x.x, x.y, x.z]).collect()
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = a.iter().chain(b).fold(1e-300f64, |m, x| m.max(x.abs()));
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs())) / scale
}

struct Report {
    sliding_corners: usize,
    fd_vs_forward: f64,
    transpose_identity: f64,
    fd_vs_ift: f64,
}

fn check(deg: f64) -> Report {
    let dt = 1e-3;
    let model = incline_box(deg, dt);
    let mut state = settle(&model, 0.5);
    let (p, _) = problem_at(&model, &mut state).expect("the box is on the plane");
    let c = cfg();
    let dim = 3 * p.n;
    // Scale the direction to the problem: the Delassus diagonal is O(10) here.
    let dscale = p.delassus[0].abs().max(1.0) * 1e-2;
    let (d_apr, dc) = direction(dim, dscale);

    let plain = solve_contacts_warm(&p, &c, &[]);
    assert!(plain.converged, "{deg} deg: the base solve must converge (residual {:e})", plain.residual);
    let sliding_corners = plain
        .impulses
        .iter()
        .zip(&p.rows)
        .filter(|(f, r)| f.x > 0.0 && ((f.y * f.y + f.z * f.z).sqrt() - r.mu * f.x).abs() <= 1e-9 * f.x)
        .count();

    let (sol, df) = contact_solve_differential(&p, &c, &[], &[], &d_apr, &dc);
    assert_eq!(sol.iterations, plain.iterations, "{deg} deg: differentiation changed the primal");
    let df = flat(&df);

    // Central difference; best of a small h sweep (O(h^2) truncation vs
    // round-off), as in phyz-contact's solver_level_adjoint.
    let fd_at = |h: f64| {
        let up = solve_contacts_warm(&perturbed(&p, &d_apr, &dc, h), &c, &[]);
        let dn = solve_contacts_warm(&perturbed(&p, &d_apr, &dc, -h), &c, &[]);
        assert!(up.converged && dn.converged, "{deg} deg: a perturbed solve failed to converge");
        flat(&up.impulses)
            .iter()
            .zip(flat(&dn.impulses))
            .map(|(a, b)| (a - b) / (2.0 * h))
            .collect::<Vec<f64>>()
    };
    let fds: Vec<Vec<f64>> = [1e-4, 1e-5, 1e-6, 1e-7].iter().map(|&h| fd_at(h)).collect();
    let fd_vs_forward = fds.iter().map(|fd| rel(&df, fd)).fold(f64::INFINITY, f64::min);

    // Reverse mode: one covector.
    let bar: Vec<Vec3> = (0..p.n)
        .map(|k| Vec3::new(0.3 + 0.1 * k as f64, -0.7 + 0.2 * k as f64, 0.5 - 0.15 * k as f64))
        .collect();
    let (_, adj) = contact_solve_differential_transpose(&p, &c, &[], &bar);
    let lhs: f64 = flat(&bar).iter().zip(&df).map(|(a, b)| a * b).sum();
    let mut rhs = 0.0;
    let mut mag = lhs.abs();
    for (a, d) in adj.bar_apr.iter().zip(&d_apr) {
        rhs += a * d;
        mag += (a * d).abs();
    }
    for (a, d) in adj.bar_c.iter().zip(&dc) {
        rhs += a * d;
        mag += (a * d).abs();
    }
    let transpose_identity = (lhs - rhs).abs() / mag.max(1e-300);

    // IFT at the converged solve: d_stationarity = dA f* + dc, and the
    // max-dissipation fixed point carries no separate own-block channel.
    let s = FixedPointSensitivity::at(&p, &plain, &c).expect("the IFT map exists at a converged solve");
    let f = flat(&plain.impulses);
    let d_stat: Vec<f64> = (0..dim)
        .map(|r| dc[r] + (0..dim).map(|k| d_apr[r * dim + k] * f[k]).sum::<f64>())
        .collect();
    let ift = flat(&s.apply(&d_stat, &vec![[0.0; 2]; p.n]));
    let fd_vs_ift = fds.iter().map(|fd| rel(&ift, fd)).fold(f64::INFINITY, f64::min);

    let r = Report { sliding_corners, fd_vs_forward, transpose_identity, fd_vs_ift };
    eprintln!(
        "incline {deg} deg: n {} sliding corners {} | fd vs forward {:.3e} | transpose identity {:.3e} | fd vs IFT {:.3e}",
        p.n, r.sliding_corners, r.fd_vs_forward, r.transpose_identity, r.fd_vs_ift
    );
    r
}

#[test]
fn the_sliding_box_gradient_matches_a_finite_difference() {
    let r = check(35.0);
    assert!(r.sliding_corners >= 4, "35 deg must slide on every corner, got {}", r.sliding_corners);
    assert!(r.fd_vs_forward < 1e-6, "forward mode vs fd: {:e}", r.fd_vs_forward);
    assert!(r.transpose_identity < 1e-10, "transpose identity: {:e}", r.transpose_identity);
    assert!(r.fd_vs_ift < 1e-6, "IFT vs fd: {:e}", r.fd_vs_ift);
}

#[test]
fn the_stuck_box_gradient_matches_a_finite_difference() {
    let r = check(26.0);
    assert_eq!(r.sliding_corners, 0, "26 deg < atan(0.5) must stick");
    assert!(r.fd_vs_forward < 1e-6, "forward mode vs fd: {:e}", r.fd_vs_forward);
    assert!(r.transpose_identity < 1e-10, "transpose identity: {:e}", r.transpose_identity);
    assert!(r.fd_vs_ift < 1e-6, "IFT vs fd: {:e}", r.fd_vs_ift);
}
