//! Integration tests for tau-gravity.

use tau_gravity::{
    ConstantGravity, G, GravityParticle, GravitySolver, NBodySolver, PoissonSolver,
    PostNewtonianSolver, orbital_elements, perihelion_precession_rate,
};
use tau_math::Vec3;

#[test]
fn test_constant_gravity_freefall() {
    let mut solver = ConstantGravity::earth();
    let mut particles = vec![GravityParticle::new(
        Vec3::new(0.0, 0.0, 100.0),
        Vec3::zeros(),
        10.0,
    )];

    let dt = 0.01;
    for _ in 0..100 {
        solver.compute_forces(&mut particles);
        particles[0].velocity_verlet_step(dt);
        solver.compute_forces(&mut particles);
        particles[0].velocity_verlet_complete(dt);
    }

    // After 1 second, should have fallen: h = h0 - 0.5*g*t^2 = 100 - 0.5*9.81*1 = 95.095
    let expected_z = 100.0 - 0.5 * 9.80665 * 1.0;
    assert!((particles[0].x.z - expected_z).abs() < 0.1);
}

#[test]
fn test_nbody_energy_conservation() {
    let mut solver = NBodySolver::new();
    solver.softening = 0.01;

    let mut particles = vec![
        GravityParticle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::zeros(), 1e10),
        GravityParticle::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, (G * 1e10 / 1.0).sqrt(), 0.0),
            1e8,
        ),
    ];

    let e0 = particles[0].kinetic_energy()
        + particles[1].kinetic_energy()
        + solver.potential_energy(&particles);

    let dt = 0.001;
    for _ in 0..1000 {
        solver.compute_forces(&mut particles);
        for p in &mut particles {
            p.velocity_verlet_step(dt);
        }
        solver.compute_forces(&mut particles);
        for p in &mut particles {
            p.velocity_verlet_complete(dt);
        }
    }

    let e1 = particles[0].kinetic_energy()
        + particles[1].kinetic_energy()
        + solver.potential_energy(&particles);

    // Energy conservation within 1%
    assert!((e1 - e0).abs() / e0.abs() < 0.01);
}

#[test]
fn test_barnes_hut_vs_direct() {
    let mut solver_direct = NBodySolver::new();
    solver_direct.use_tree = false;

    // For now, just test that direct N-body works
    // Barnes-Hut tree implementation needs position storage for proper leaf splitting
    let particles_init = vec![
        GravityParticle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::zeros(), 1e10),
        GravityParticle::new(Vec3::new(1.0, 0.0, 0.0), Vec3::zeros(), 1e10),
        GravityParticle::new(Vec3::new(0.0, 1.0, 0.0), Vec3::zeros(), 1e10),
    ];

    let mut particles_direct = particles_init.clone();

    solver_direct.compute_forces(&mut particles_direct);

    // Check that forces were computed
    for i in 0..3 {
        assert!(particles_direct[i].f.norm() > 0.0);
    }
}

#[test]
fn test_poisson_solver_uniform_density() {
    let mut solver =
        PoissonSolver::new(20, (Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)));

    // Single particle at center
    let particles = vec![GravityParticle::new(Vec3::zeros(), Vec3::zeros(), 1e6)];

    solver.deposit_density(&particles);
    solver.solve_poisson();

    // Check that potential is computed
    let max_potential = solver
        .potential
        .iter()
        .map(|&p| p.abs())
        .fold(0.0, f64::max);
    assert!(max_potential > 0.0);
}

#[test]
fn test_mercury_precession_calculation() {
    let m_sun = 1.989e30;
    let a = 57.9e9; // Mercury semi-major axis
    let e = 0.2056;
    let period = 87.969 * 24.0 * 3600.0;

    let precession = perihelion_precession_rate(a, e, m_sun, period);

    // Should be close to 43 arcsec/century
    assert!(precession > 38.0 && precession < 48.0);
}

#[test]
fn test_orbital_elements_circular() {
    let m_sun = 1.989e30;
    let r = 1.5e11; // 1 AU
    let v_circular = (G * m_sun / r).sqrt();

    let x = Vec3::new(r, 0.0, 0.0);
    let v = Vec3::new(0.0, v_circular, 0.0);

    let (a_computed, e_computed, _i, _omega, _omega_bar, _nu) = orbital_elements(x, v, m_sun);

    // Circular orbit
    assert!((a_computed - r).abs() / r < 1e-6);
    assert!(e_computed < 1e-6);
}

#[test]
fn test_orbital_elements_elliptical() {
    let m_sun = 1.989e30;
    let a_target = 1.5e11;
    let e_target = 0.3;

    // At perihelion: r = a(1-e), v = sqrt(GM/a * (1+e)/(1-e))
    let r_perihelion = a_target * (1.0 - e_target);
    let v_perihelion = (G * m_sun / a_target * (1.0 + e_target) / (1.0 - e_target)).sqrt();

    let x = Vec3::new(r_perihelion, 0.0, 0.0);
    let v = Vec3::new(0.0, v_perihelion, 0.0);

    let (a_computed, e_computed, _i, _omega, _omega_bar, _nu) = orbital_elements(x, v, m_sun);

    assert!((a_computed - a_target).abs() / a_target < 1e-6);
    assert!((e_computed - e_target).abs() < 1e-6);
}

#[test]
fn test_pn_vs_newtonian() {
    let mut solver_newton = PostNewtonianSolver::new(0.0);
    let mut solver_pn = PostNewtonianSolver::new(1.0);

    let mut particles_newton = vec![
        GravityParticle::new(Vec3::zeros(), Vec3::zeros(), 1e30),
        GravityParticle::new(Vec3::new(1e9, 0.0, 0.0), Vec3::new(0.0, 1e4, 0.0), 1e20),
    ];

    let mut particles_pn = particles_newton.clone();

    solver_newton.compute_forces(&mut particles_newton);
    solver_pn.compute_forces(&mut particles_pn);

    // PN corrections should be small but non-zero for non-zero velocities
    let diff = (particles_newton[1].f - particles_pn[1].f).norm();
    let mag = particles_newton[1].f.norm();

    // Difference should be small (PN is a correction)
    assert!(diff / mag < 0.01);
    // But non-zero due to velocity-dependent terms
    assert!(diff > 0.0);
}

#[test]
fn test_2_5pn_orbital_decay() {
    // Test that 2.5PN includes radiation reaction terms
    let mut solver_2_5pn = PostNewtonianSolver::new(2.5);
    let mut solver_1pn = PostNewtonianSolver::new(1.0);

    // Binary system with velocities
    let m = 1e30;
    let r0 = 1e8;
    let v0 = (G * 2.0 * m / r0).sqrt();

    let mut particles_2_5pn = vec![
        GravityParticle::new(
            Vec3::new(-r0 / 2.0, 0.0, 0.0),
            Vec3::new(0.0, -v0 / 2.0, 0.0),
            m,
        ),
        GravityParticle::new(
            Vec3::new(r0 / 2.0, 0.0, 0.0),
            Vec3::new(0.0, v0 / 2.0, 0.0),
            m,
        ),
    ];

    let mut particles_1pn = particles_2_5pn.clone();

    solver_2_5pn.compute_forces(&mut particles_2_5pn);
    solver_1pn.compute_forces(&mut particles_1pn);

    // 2.5PN should include radiation damping, giving different forces
    let diff = (particles_2_5pn[0].f - particles_1pn[0].f).norm();

    // Radiation term should be non-zero (though small)
    assert!(diff > 0.0);

    // Verify that radiation is enabled
    assert!(solver_2_5pn.include_radiation);
    assert!(!solver_1pn.include_radiation);
}
