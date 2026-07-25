//! Dimension 1: single-simulation throughput (steps/sec).

use crate::report::{Metric, Record, Suite};
use crate::scenes::{PhyzSim, Scene};
use crate::settings::{DT_ARTICULATED, DT_CONTACT, Settings};
use crate::timing::{Budget, measure};

/// Steps timed per repetition. Long enough to swamp loop overhead, short
/// enough that a chaotic scene has not yet diverged into a different regime.
pub const STEPS_PER_REP: u64 = 20_000;

/// The timestep and settings a scene is benchmarked at.
pub fn settings_for(scene: Scene) -> Settings {
    if scene.has_contact() {
        Settings::contact(DT_CONTACT)
    } else {
        Settings::articulated(DT_ARTICULATED)
    }
}

/// Measure one scene and produce its record.
pub fn run_scene(scene: Scene, budget: Budget) -> Record {
    let settings = settings_for(scene);
    let mut sim = PhyzSim::new(scene, settings.dt);
    let dof = sim.model.nv;

    let timing = measure(budget, STEPS_PER_REP, || {
        // Reset each repetition: repetition 7 must simulate the same physics
        // as repetition 1, or we are timing a different trajectory.
        sim.reset();
        sim.steps(STEPS_PER_REP as usize);
        sim.state.q[0]
    });

    // A diverged simulation is often the fastest one; refuse to report it as a
    // result.
    let stable = sim.state_is_finite();

    // Report simulated-seconds-per-wall-second too: the number that actually
    // decides whether a controller can run in the loop.
    let realtime_factor = timing.throughput_per_sec * settings.dt;

    let mut notes = vec![
        "One step = forward kinematics + (contact) + ABA + semi-implicit Euler, single \
         threaded, f64."
            .into(),
    ];
    if scene.has_contact() {
        notes.push(
            "Contact scenes include sweep-and-prune broad phase and GJK/EPA narrow phase.".into(),
        );
    }
    if !stable {
        notes.push(format!(
            "UNSTABLE: `{}` diverged to a non-finite state within {STEPS_PER_REP} steps at \
             dt = {}. The throughput figure is not a valid result — a blown-up simulation \
             does less work, not more.",
            scene.name(),
            settings.dt
        ));
    }

    Record {
        engine: "phyz".into(),
        scene: scene.name(),
        description: scene.description(),
        dof: Some(dof),
        batch: Some(1),
        settings,
        timing: Some(timing),
        metrics: vec![
            Metric::new("realtime_factor", realtime_factor, "×"),
            Metric::new("stable", if stable { 1.0 } else { 0.0 }, "bool"),
        ],
        notes,
    }
}

/// Run the single-sim suite over the standard scenes.
pub fn run(budget: Budget) -> Suite {
    let results = super::standard_scenes()
        .into_iter()
        .map(|s| run_scene(s, budget))
        .collect();
    Suite::new(
        "single-sim throughput",
        "Steps per second for one simulation on one CPU thread. `realtime_factor` is \
         simulated seconds per wall-clock second — above 1.0 the scene runs faster than \
         real time at that timestep.",
        results,
    )
}
