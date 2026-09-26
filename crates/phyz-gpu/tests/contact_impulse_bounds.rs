//! A contact's dissipative terms are bounded by the mass they act on.
//!
//! A penalty contact has two gains and only one of them was ever bounded. The
//! spring is limited by `omega*dt`, which the API documents; the DAMPERS were
//! not limited at all, and one of them is hidden — regularized Coulomb friction
//! is a tangential damper of slope `mu*f_n/SLIP_EPS`, and that slope grows with
//! the normal load.
//!
//! An explicit damper is stable only while `d*dt <= m_eff`. Above it the damper
//! does not stop the relative motion, it reverses it, further every step.
//!
//! This is not hypothetical. It is what a skateboard wheel does: 0.1 kg of mass
//! and 2.9e-5 kg m^2 of spin inertia at a 27 mm radius, carrying gains sized for
//! the 5.75 kg it holds up. The friction slope came to 5.7e4 N s/m, or 41
//! N m s/rad about the axle — an explicit decay rate of 1.4e6 /s at dt = 1 ms.
//! Four wheels reached 19000 rad/s within twenty steps and the whole state was
//! NaN by 0.02 s, which a `f64::max` peak-finder then silently skipped.
//!
//! These pin the bound: a body far lighter than its gains claim must survive.

use phyz_gpu::{BodyContactGains, GpuBatchSimulator};
use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Joint, ModelBuilder};

const DT: f64 = 1e-3;

fn ball_inertia(mass: f64, radius: f64) -> SpatialInertia {
    let i = 0.4 * mass * radius * radius;
    SpatialInertia::new(
        mass,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(i, i, i)),
    )
}

/// A body whose DAMPING is sized for a mass twenty times its own.
///
/// The spring is left at the body's own mass, so `omega*dt` stays well inside
/// the documented limit and the only thing out of scale is the damper. That
/// isolates the bug: `d*dt/m = 20`, so an unbounded explicit damper multiplies
/// the approach velocity by -19 every step. Bounded, it merely stops it.
#[test]
fn a_damper_sized_for_a_carried_mass_does_not_diverge_on_a_light_body() {
    const RADIUS: f64 = 0.027;
    const MASS: f64 = 0.1;
    const CARRIED: f64 = 10.0;
    const OMEGA: f64 = 100.0;

    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(DT)
        .add_body(
            "ball",
            -1,
            Joint::free(SpatialTransform::identity()),
            ball_inertia(MASS, RADIUS),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Sphere { radius: RADIUS });

    let Ok(mut sim) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut gains = BodyContactGains::uniform_frequency(&model, OMEGA, 1.0);
    gains[0] = BodyContactGains {
        // Spring for what the body IS; damper for what it CARRIES.
        stiffness: MASS * OMEGA * OMEGA,
        damping: 2.0 * CARRIED * OMEGA,
    };
    sim.enable_ground_contact_per_body(0.0, 0.9, &gains)
        .expect("contact");

    let mut s = model.default_state();
    s.q[5] = RADIUS + 0.01; // dropped, so the damper meets a real approach
    sim.load_states(std::slice::from_ref(&s));
    for _ in 0..2000 {
        sim.step();
    }

    let out = &sim.readback_states()[0];
    assert!(
        out.q.iter().chain(out.v.iter()).all(|x| x.is_finite()),
        "contact diverged: q = {:?}, v = {:?}",
        out.q,
        out.v
    );
    // It settles on the ground instead of being thrown off it.
    assert!(
        (out.q[5] - RADIUS).abs() < 0.01,
        "the ball did not settle on the ground (z = {:.4}, radius {RADIUS})",
        out.q[5]
    );
    assert!(
        out.v[5].abs() < 0.1,
        "the ball was still moving vertically (v_z = {:.4})",
        out.v[5]
    );
}

/// The same body, spun about its own axis while pressed to the ground: this is
/// the wheel's failure exactly, since the divergence lived in the spin DOF
/// rather than in any translation.
#[test]
fn a_spinning_light_body_does_not_wind_up_against_the_ground() {
    const RADIUS: f64 = 0.027;
    const MASS: f64 = 0.1;
    const CARRIED: f64 = 5.75;
    const OMEGA: f64 = 100.0;

    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -9.81))
        .dt(DT)
        .add_body(
            "wheel",
            -1,
            Joint::free(SpatialTransform::identity()),
            ball_inertia(MASS, RADIUS),
        )
        .build();
    model.bodies[0].geometry = Some(Geometry::Sphere { radius: RADIUS });

    let Ok(mut sim) = GpuBatchSimulator::new(model.clone(), 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut gains = BodyContactGains::uniform_frequency(&model, OMEGA, 1.0);
    gains[0] = BodyContactGains {
        stiffness: CARRIED * OMEGA * OMEGA,
        damping: 2.0 * CARRIED * OMEGA,
    };
    sim.enable_ground_contact_per_body(0.0, 0.9, &gains)
        .expect("contact");

    let mut s = model.default_state();
    s.q[5] = RADIUS;
    s.v[1] = 10.0; // spin about y: the rolling axis
    sim.load_states(std::slice::from_ref(&s));

    let mut peak_spin = 0.0f64;
    for _ in 0..1000 {
        sim.step();
        let out = &sim.readback_states()[0];
        peak_spin = peak_spin.max(out.v[1].abs());
        assert!(peak_spin.is_finite(), "spin diverged to a non-finite value");
    }
    // Contact may slow the spin or let it roll; it must never drive it far
    // past where it started. Unbounded, this reached thousands of rad/s.
    assert!(
        peak_spin < 20.0,
        "the ground wound the spin up from 10 rad/s to {peak_spin:.1}"
    );
}
