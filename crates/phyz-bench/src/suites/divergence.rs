//! Dimension 5: reproducibility — how far two rollouts drift apart when you
//! perturb one input by a single representable step.
//!
//! Every other suite here measures how *fast* or how *accurate* the engine is.
//! This one measures something you need before either number means anything:
//! how long a result from a given scene stays comparable to itself.
//!
//! # Why it exists
//!
//! Contact-rich rigid-body dynamics is chaotic, so two rollouts that differ in
//! the last bit of one input separate exponentially. That is physics, and the
//! engine cannot and should not remove it. What it *can* do is remove the
//! second source of difference — the engine computing different numbers from
//! the same inputs — and then tell you how big the first one is, so you can
//! recognise it.
//!
//! Without that calibration the two are indistinguishable, and the cost of
//! confusing them is concrete: in the project this suite was written for, a
//! refactor that changed only the *order* of floating-point operations —
//! verified equal to `2.2e-16` in parameter space — moved a searched
//! candidate's score by hundreds of points, and it took hours to establish
//! that this was amplification rather than a bug. With a table like the one
//! below, that is a two-minute check.
//!
//! # Reading the output
//!
//! For each scene, one row reporting:
//!
//! - `initial_separation` — the size of the perturbation, in state units.
//!   Should be `~1e-16` relative to the coordinate touched. If it is `0`, the
//!   perturbation was quantized away and the row means nothing.
//! - `final_separation` — how far apart the rollouts ended.
//! - `amplification` — the ratio of the two. **This is the headline.**
//! - `lyapunov` — the fitted exponential rate, per second of simulated time,
//!   over the part of the curve that is actually exponential. Negative means
//!   the scene is *contracting*: dissipative contact forgets a perturbation
//!   rather than amplifying it, which is the common case for anything that
//!   settles, and is worth knowing because it is the opposite of the intuition
//!   people bring.
//! - `mantissa_horizon` — `52 * ln(2) / lyapunov`: the simulated time after
//!   which two rollouts from *bit-identical* inputs would have lost every
//!   digit of agreement, if `f64` rounding seeded them a single ulp apart.
//!   Blank for contracting scenes, which have no such horizon.
//!
//! # The rule this table supports
//!
//! Two rollouts of the same scene disagree, and you want to know whether to
//! open a debugger. Divide their disagreement by `1e-16` and compare against
//! `amplification` for that scene at that horizon. Comparable — chaos.
//! Far larger, or present at step 0 — bug.
//!
//! The contact-free scenes are included on purpose: they are the control. If a
//! pendulum's rollout is not bit-identical to itself, no chaos argument
//! applies and something in the engine is wrong.

use phyz::determinism::{Divergence, divergence, hash_rollout, ulp_offset};

use crate::report::{Metric, Record, Suite};
use crate::scenes::{PhyzSim, Scene};
use crate::settings::Settings;
use crate::timing::Budget;

/// Simulated horizon for the sweep, in seconds.
///
/// Long enough for the exponential band to be several decades wide on a
/// chaotic scene, short enough that a settling scene has actually settled — the
/// two behaviours this suite exists to distinguish.
pub const HORIZON_SEC: f64 = 5.0;

/// Timestep every row is measured at.
pub const DT: f64 = 1.0e-3;

/// Which `q` coordinate to perturb, per scene.
///
/// A coordinate that the dynamics actually read on the first step; perturbing
/// a slot the scene never touches would report a flat zero and look like
/// perfect stability.
fn perturbed_coordinate(scene: Scene) -> usize {
    match scene {
        // Hinge angle of the first link.
        Scene::Pendulum | Scene::DoublePendulum => 0,
        // Free base: the height slot. Not the `wx` exp-coordinate, which the
        // ant starts at exactly zero — one ulp from zero is a subnormal, and a
        // subnormal rotation is indistinguishable from none.
        Scene::Ant => 5,
        // Free base of the top box: its height above the stack.
        Scene::BoxStack(_) => 5,
    }
}

/// Measure one scene.
pub fn run_point(scene: Scene) -> Record {
    let steps = (HORIZON_SEC / DT).round() as usize;
    let stride = (steps / 250).max(1);
    let q_index = perturbed_coordinate(scene);

    // Two independent `PhyzSim`s, so nothing is shared between the reference
    // and the perturbed rollout — the whole measurement is void if they are
    // coupled by so much as a scratch buffer.
    let mut sims = [PhyzSim::new(scene, DT), PhyzSim::new(scene, DT)];
    let model_nq = sims[0].model.nq;
    let initial = sims[0].state.clone();

    let d: Divergence = divergence(
        model_nq,
        &initial,
        q_index,
        1,
        steps,
        stride,
        |which, state| {
            // `PhyzSim::step` owns its state, so hand it this rollout's and
            // take it back. Cloning per step would dominate the measurement,
            // which is why the state is moved rather than copied.
            std::mem::swap(&mut sims[which].state, state);
            sims[which].step();
            std::mem::swap(&mut sims[which].state, state);
        },
    );

    // Separately: the *reproducibility* half. Two fresh rollouts with no
    // perturbation at all must fingerprint identically. This is the control
    // that keeps the amplification number honest — an amplification figure
    // measured on an engine that is not self-consistent is measuring the
    // inconsistency.
    let fingerprint = |seed_steps: usize| {
        let mut sim = PhyzSim::new(scene, DT);
        let mut state = sim.state.clone();
        hash_rollout(&mut state, seed_steps, |s| {
            std::mem::swap(&mut sim.state, s);
            sim.step();
            std::mem::swap(&mut sim.state, s);
        })
    };
    let reproducible = fingerprint(steps) == fingerprint(steps);

    let amplification = if d.initial > 0.0 {
        d.final_distance() / d.initial
    } else {
        f64::NAN
    };
    let lyapunov = d.lyapunov.unwrap_or(f64::NAN);
    let mantissa_horizon = d.doubling_time().map(|t| 52.0 * t).unwrap_or(f64::NAN);

    let mut notes = vec![
        format!(
            "Perturbation: q[{q_index}] moved one representable step, from {:e} to {:e}.",
            initial.q[q_index],
            ulp_offset(initial.q[q_index], 1)
        ),
        "`amplification` is the number to compare an unexplained discrepancy against. \
         A disagreement of the same order is chaos; one much larger, or one present at \
         step 0, is a bug."
            .to_string(),
        "`lyapunov` is fitted over the exponential band only — after the separation has \
         grown a factor of 8 out of rounding noise and before it saturates at the scale \
         of the scene. Fitting through either end reports a chaotic scene as a tame one."
            .to_string(),
    ];
    if !reproducible {
        notes.push(
            "REPRODUCIBILITY FAILURE: two rollouts from identical inputs, on \
             independently constructed simulators, did not produce identical bits. Every \
             other number in this row is meaningless until that is fixed — and no chaos \
             argument applies, because the inputs did not differ."
                .to_string(),
        );
    }
    if lyapunov.is_finite() && lyapunov < 0.0 {
        notes.push(
            "Contracting: the fitted rate is negative, so this scene forgets a small \
             perturbation instead of amplifying it. Dissipative contact does that once a \
             body settles. There is no mantissa horizon for such a scene, and a large \
             unexplained discrepancy in it is *more* suspicious, not less."
                .to_string(),
        );
    }
    if d.initial == 0.0 {
        notes.push(format!(
            "q[{q_index}] is zero and moving it one ulp landed on a subnormal, or the \
             coordinate is not read by this scene. The row measures nothing; pick a \
             different coordinate."
        ));
    } else if d.final_distance() == 0.0 {
        notes.push(format!(
            "The two rollouts ended bit-identical: the perturbation was *absorbed*, not \
             amplified. This is the expected outcome when the perturbed coordinate grows \
             far beyond its starting magnitude — q[{q_index}] began at {:e}, where one \
             ulp is {:e}, and by the end of the rollout an ulp of the coordinate itself \
             is larger than that, so the difference rounds away. It is a real property of \
             the scene, not a measurement failure, but it does mean this row supplies no \
             amplification figure to calibrate against.",
            initial.q[q_index],
            ulp_offset(initial.q[q_index], 1) - initial.q[q_index]
        ));
    }

    Record {
        engine: "phyz".into(),
        scene: format!("{}@1ulp", scene.name()),
        description: format!(
            "{} — {HORIZON_SEC:.0} s, one-ulp perturbation of q[{q_index}]",
            scene.description()
        ),
        dof: Some(sims[0].model.nv),
        batch: Some(1),
        settings: if scene.has_contact() {
            Settings::contact(DT)
        } else {
            Settings::articulated(DT)
        },
        timing: None,
        metrics: vec![
            Metric::new("initial_separation", d.initial, "state-units"),
            Metric::new("final_separation", d.final_distance(), "state-units"),
            Metric::new("amplification", amplification, "ratio"),
            Metric::new("lyapunov", lyapunov, "1/s"),
            Metric::new("mantissa_horizon", mantissa_horizon, "s"),
            Metric::new(
                "bitwise_reproducible",
                if reproducible { 1.0 } else { 0.0 },
                "bool",
            ),
        ],
        notes,
    }
}

/// Run the divergence sweep over the standard scenes.
///
/// `budget` is accepted for signature parity with the other suites and
/// deliberately ignored: this suite measures a property of the trajectory, not
/// a timing, so repeating it would produce the same numbers at N times the
/// cost.
pub fn run(_budget: Budget) -> Suite {
    let results = super::standard_scenes()
        .into_iter()
        .map(run_point)
        .collect();
    Suite::new(
        "reproducibility (1-ulp divergence)",
        "How far two rollouts separate when one input is moved by a single representable \
         step, plus a control checking that two *unperturbed* rollouts are bit-identical. \
         Read `amplification` before concluding that an unexplained discrepancy is a bug: \
         a difference of that order is chaos doing what chaos does, one much larger is \
         not. See docs/determinism.md.",
        results,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The control must hold on the scene with no contact and no chaos: a
    /// single pendulum's rollout is bit-identical to itself, and a one-ulp
    /// perturbation stays roughly one ulp.
    #[test]
    fn the_pendulum_control_is_reproducible_and_tame() {
        let r = run_point(Scene::Pendulum);
        let metric = |n: &str| r.metrics.iter().find(|m| m.name == n).unwrap().value;
        assert_eq!(metric("bitwise_reproducible"), 1.0);
        assert!(metric("initial_separation") > 0.0);
        // An integrable scene: the perturbation grows at most polynomially, so
        // over 5 s it stays within a few orders of magnitude of where it began.
        assert!(
            metric("amplification") < 1e6,
            "pendulum amplified by {}",
            metric("amplification")
        );
    }
}
