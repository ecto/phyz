//! Tests for the benchmark harness itself.
//!
//! A benchmark suite that silently measures the wrong thing is worse than no
//! benchmark suite, because its output looks authoritative. These tests pin
//! the properties the published numbers depend on.

use phyz_bench::report::SCHEMA_VERSION;
use phyz_bench::scenes::{PhyzSim, Scene};
use phyz_bench::suites::single_sim::settings_for;
use phyz_bench::suites::{self, standard_scenes};

/// Every standard scene must build, step, and stay finite. A scene that
/// diverges gets *faster*, so instability would read as a speedup.
#[test]
fn standard_scenes_are_stable() {
    for scene in standard_scenes() {
        let dt = settings_for(scene).dt;
        let mut sim = PhyzSim::new(scene, dt);
        sim.steps(2_000);
        assert!(
            sim.state_is_finite(),
            "scene {} diverged to a non-finite state within 2000 steps at dt={dt}",
            scene.name()
        );
    }
}

/// `reset` must restore the exact initial state, or repetition 7 of a timed
/// run measures a different trajectory than repetition 1.
#[test]
fn reset_restores_initial_state() {
    for scene in standard_scenes() {
        let dt = settings_for(scene).dt;
        let mut sim = PhyzSim::new(scene, dt);
        let q0 = sim.state.q.as_slice().to_vec();
        let v0 = sim.state.v.as_slice().to_vec();

        sim.steps(100);
        sim.reset();

        assert_eq!(sim.state.q.as_slice(), q0.as_slice(), "{}", scene.name());
        assert_eq!(sim.state.v.as_slice(), v0.as_slice(), "{}", scene.name());
    }
}

/// Stepping must be deterministic: same initial state, same trajectory. If it
/// isn't, none of the repetition statistics mean anything.
#[test]
fn stepping_is_deterministic() {
    let mut a = PhyzSim::new(Scene::DoublePendulum, 1.0e-3);
    let mut b = PhyzSim::new(Scene::DoublePendulum, 1.0e-3);
    a.steps(500);
    b.steps(500);
    assert_eq!(a.state.q.as_slice(), b.state.q.as_slice());
}

/// The pendulum scenes must actually be pendulums.
///
/// phyz's default revolute axis is +Z. With a rod hanging along −Z under −Z
/// gravity, that puts the centre of mass on the rotation axis and the
/// gravitational torque is identically zero — the body never moves and the
/// benchmark measures a stationary object. This test fails if anyone
/// "simplifies" the scene back to the default axis.
#[test]
fn pendulum_scenes_actually_swing() {
    for scene in [Scene::Pendulum, Scene::DoublePendulum] {
        let mut sim = PhyzSim::new(scene, 1.0e-3);
        let q0 = sim.state.q[0];
        sim.steps(500);
        assert!(
            (sim.state.q[0] - q0).abs() > 1.0e-3,
            "{} did not move in 500 steps — the hinge axis is probably degenerate",
            scene.name()
        );
    }
}

/// The box stack must actually generate contacts, or the "contact scene" is
/// measuring free fall.
#[test]
fn box_stack_generates_contact_forces() {
    let scene = Scene::BoxStack(8);
    let mut sim = PhyzSim::new(scene, settings_for(scene).dt);
    let z0 = sim.state.q[2];
    sim.steps(2_000);
    let z = sim.state.q[2];

    assert!(sim.state_is_finite(), "box stack diverged");
    // The bottom box starts resting on the plane and must not sink through it.
    assert!(
        z > -0.05,
        "bottom box fell through the ground (z0={z0}, z={z}) — contact is not engaging"
    );
}

/// A conservative scene must conserve energy to the integrator's order. This
/// is the property the numerical-quality suite reports; if it broke, that
/// suite would publish a wrong conclusion.
#[test]
fn pendulum_conserves_energy_to_order_dt() {
    let peak_err = |dt: f64| {
        let mut sim = PhyzSim::new(Scene::Pendulum, dt);
        let e0 = sim.total_energy();
        let mut peak: f64 = 0.0;
        for _ in 0..(5.0 / dt) as usize {
            sim.step();
            peak = peak.max((sim.total_energy() - e0).abs() / e0.abs());
        }
        peak
    };

    let coarse = peak_err(4.0e-3);
    let fine = peak_err(1.0e-3);

    // Semi-implicit Euler is first order: 4× smaller dt should give roughly 4×
    // smaller error. Bounds are loose enough to survive a different machine.
    let ratio = coarse / fine;
    assert!(
        (2.5..=6.0).contains(&ratio),
        "energy error is not scaling as O(dt): coarse={coarse:.2e}, fine={fine:.2e}, \
         ratio={ratio:.2}"
    );
}

/// Suites must produce serialisable output, and a suite that cannot run must
/// say so rather than returning silently empty results.
#[test]
fn suites_serialise_and_explain_skips() {
    let budget = phyz_bench::Budget::quick();
    let suite = suites::single_sim::run(budget);
    assert!(suite.skipped.is_none());
    assert_eq!(suite.results.len(), standard_scenes().len());

    let json = serde_json::to_string(&suite).expect("suite serialises");
    assert!(
        json.contains("\"settings\""),
        "settings must travel with results"
    );
    assert!(
        json.contains("\"caveats\""),
        "caveats must travel with results"
    );

    // Whatever the build, the GPU suite either ran or explained itself.
    let gpu = suites::gpu_batch::run(budget);
    assert!(
        gpu.skipped.is_some() || !gpu.results.is_empty(),
        "a suite must either produce results or record why it did not"
    );

    assert_eq!(SCHEMA_VERSION, 1);
}

/// Rapier scenes must match the phyz scenes they are compared against.
#[cfg(feature = "rapier")]
#[test]
fn rapier_comparison_covers_the_same_scenes() {
    let suite = suites::rapier::run(phyz_bench::Budget::quick());

    for scene in standard_scenes() {
        let rows: Vec<_> = suite
            .results
            .iter()
            .filter(|r| r.scene == scene.name())
            .collect();
        assert!(!rows.is_empty(), "no row for {}", scene.name());

        // Either both engines are present, or the phyz row explains the gap.
        let has_rapier = rows.iter().any(|r| r.engine.starts_with("rapier"));
        if !has_rapier {
            assert!(
                rows.iter()
                    .any(|r| r.notes.iter().any(|n| n.contains("No Rapier comparison"))),
                "{} has no Rapier row and no explanation for why",
                scene.name()
            );
        }
    }
}
