//! Dimension 4: numerical quality — energy drift, reported next to speed.
//!
//! A conservative scene (frictionless pendulum, no contact, no damping) must
//! conserve total mechanical energy exactly. A discrete integrator will not,
//! and the size of that error is the honest other half of a throughput
//! number: halving `dt` doubles the cost per simulated second, so the two
//! axes have to be read together.
//!
//! Semi-implicit Euler is symplectic, so its energy error should *oscillate
//! within a bounded band* rather than grow without limit. We report both the
//! net drift at the end of the rollout and the peak excursion along the way,
//! because a bounded oscillation and a slow leak look identical if you only
//! sample the endpoint.

use crate::report::{Metric, Record, Suite};
use crate::scenes::{PhyzSim, Scene};
use crate::settings::Settings;
use crate::timing::{Budget, measure};

/// Timesteps swept, in seconds.
pub const TIMESTEPS: [f64; 4] = [4.0e-3, 2.0e-3, 1.0e-3, 2.5e-4];

/// Simulated durations swept (s).
///
/// Both are needed to tell the two stories apart. At a short horizon the error
/// is the integrator's local truncation error and should fall as O(dt) — that
/// is the number that says whether the integrator is correct. At a long
/// horizon a chaotic scene has diverged onto a neighbouring trajectory and the
/// energy band is set by the horizon rather than by `dt`; shrinking `dt` buys
/// almost nothing there. Reporting only the long horizon would look like an
/// integrator bug; reporting only the short one would oversell how well a
/// 100-second rollout behaves.
pub const HORIZONS_SEC: [f64; 2] = [5.0, 100.0];

/// Measure drift for one scene at one timestep and horizon.
pub fn run_point(scene: Scene, dt: f64, horizon_sec: f64, budget: Budget) -> Record {
    assert!(
        !scene.has_contact(),
        "energy drift is only meaningful for conservative scenes; \
         penalty contact dissipates energy by design"
    );
    let steps = (horizon_sec / dt).round() as usize;

    let mut sim = PhyzSim::new(scene, dt);
    let e0 = sim.total_energy();

    // Accuracy pass: walk the rollout tracking the worst excursion.
    let mut peak_abs_err = 0.0f64;
    for _ in 0..steps {
        sim.step();
        let err = (sim.total_energy() - e0).abs();
        if err > peak_abs_err {
            peak_abs_err = err;
        }
    }
    let stable = sim.state_is_finite();
    let e_final = sim.total_energy();
    let scale = e0.abs().max(1.0e-12);
    let net_drift = (e_final - e0) / scale;
    let peak_drift = peak_abs_err / scale;

    // Speed pass: what this timestep costs per simulated second.
    let timing = measure(budget, steps as u64, || {
        sim.reset();
        sim.steps(steps);
        sim.state.q[0]
    });
    let cost_per_sim_sec = timing.median_sec_per_unit / dt;

    let mut notes = vec![
        "Drift is relative to initial total mechanical energy. Semi-implicit Euler is \
         symplectic: `peak_energy_error` is expected to be a bounded oscillation, not a \
         monotone leak."
            .to_string(),
        "`wall_sec_per_sim_sec` below 1.0 means faster than real time at that timestep — \
         the number to weigh a smaller `dt` against."
            .to_string(),
        "Compare the 5 s and 100 s rows for the same scene and `dt` before drawing a \
         conclusion. Over 5 s the error falls as O(dt) for both scenes, which is the \
         integrator behaving correctly. Over 100 s the double pendulum's error stops \
         responding to `dt` at all: it is chaotic, so by then the rollout is on a \
         neighbouring trajectory and the energy band is set by the horizon. Buying \
         accuracy with a smaller `dt` works on the first timescale and not the second."
            .to_string(),
    ];
    if !stable {
        notes.push(format!(
            "UNSTABLE: the rollout diverged to a non-finite state at dt = {dt}. The drift \
             figures on this row are meaningless; the row is kept because 'this timestep \
             does not survive a {horizon_sec:.0} s rollout' is itself the result."
        ));
    }

    Record {
        engine: "phyz".into(),
        scene: format!("{}@dt={:e}@{:.0}s", scene.name(), dt, horizon_sec),
        description: format!(
            "{} — {horizon_sec:.0} s conservative rollout",
            scene.description()
        ),
        dof: Some(sim.model.nv),
        batch: Some(1),
        settings: Settings::articulated(dt),
        timing: Some(timing),
        metrics: vec![
            Metric::new("net_energy_drift", net_drift, "relative"),
            Metric::new("peak_energy_error", peak_drift, "relative"),
            Metric::new("initial_energy", e0, "J"),
            Metric::new("wall_sec_per_sim_sec", cost_per_sim_sec, "s/s"),
            Metric::new("stable", if stable { 1.0 } else { 0.0 }, "bool"),
        ],
        notes,
    }
}

/// Run the energy-drift sweep.
pub fn run(budget: Budget) -> Suite {
    let mut results = Vec::new();
    for scene in [Scene::Pendulum, Scene::DoublePendulum] {
        for horizon in HORIZONS_SEC {
            for dt in TIMESTEPS {
                results.push(run_point(scene, dt, horizon, budget));
            }
        }
    }
    Suite::new(
        "numerical quality (energy drift)",
        "Energy conservation swept across timestep AND horizon, alongside the cost of each \
         timestep. A fast integrator that bleeds energy is not actually faster, so the two \
         axes are reported together. The horizon sweep separates integrator error (falls \
         as O(dt), visible at 5 s) from chaotic trajectory divergence (insensitive to dt, \
         visible at 100 s on the double pendulum).",
        results,
    )
}
