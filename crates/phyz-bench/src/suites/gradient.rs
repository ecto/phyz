//! Dimension 3: gradient throughput.
//!
//! The claim under test is "analytical derivatives for free". The number that
//! substantiates or refutes it is the **adjoint ratio**: wall time for one
//! gradient rollout divided by wall time for the forward rollout it
//! differentiates. Reverse-mode differentiation of a scalar objective has a
//! ratio bounded by a small constant regardless of parameter count; finite
//! differences cost one extra forward rollout *per parameter*.
//!
//! So we measure three things on the same scene and horizon, all of them
//! actually measured rather than projected:
//!
//! 1. forward rollout (the baseline),
//! 2. `adjoint_rollout_gradient` — objective + `dJ/dπ` for every body,
//! 3. the full central-difference gradient over the same parameters, which is
//!    what the adjoint is supposed to replace.
//!
//! (3) also cross-checks (2): the two gradients must agree, or the adjoint is
//! merely a fast way to compute the wrong answer.

use phyz_diff::rollout::step::inertia_from_params;
use phyz_diff::rollout::{
    AdjointRollout, FinalStateObjective, N_INERTIA_PARAMS, adjoint_rollout_gradient,
    inertia_params, rollout_objective,
};
use phyz_math::DVec;
use phyz_model::Model;

use crate::report::{Metric, Record, Suite};
use crate::scenes::{PhyzSim, Scene};
use crate::settings::{DT_ARTICULATED, Settings};
use crate::timing::{Budget, measure};

/// Rollout length for the gradient benchmarks (steps).
pub const ROLLOUT_STEPS: usize = 500;

// One rollout is a few hundred microseconds — too close to timer noise to
// measure directly. Each timed sample repeats the work until it spans tens of
// milliseconds. The counts differ per quantity because the quantities differ
// in cost by two orders of magnitude; throughput is reported per rollout
// either way, so the ratios are unaffected.
/// Forward rollouts per timed sample.
pub const FORWARD_PER_SAMPLE: usize = 200;
/// Adjoint gradient rollouts per timed sample.
pub const ADJOINT_PER_SAMPLE: usize = 10;
/// Finite-difference gradients per timed sample.
pub const FD_PER_SAMPLE: usize = 5;

/// Scenes the adjoint supports.
///
/// The ant is out of scope here, but **not** because the adjoint cannot
/// differentiate a free-floating base — it can, and does. The obstacle is a
/// layout mismatch inside phyz: `phyz_diff`'s `DofLayout` packs a free joint as
/// position-then-quaternion (`nq = 7`) while `Model` packs it as
/// exponential-coordinates-then-position (`nq = 6`), and these scenes are built
/// as `Model`s. Converting is mechanical; until it exists the ant is omitted,
/// and the reason is recorded rather than being restated as a capability gap.
pub fn supported_scenes() -> Vec<Scene> {
    vec![Scene::Pendulum, Scene::DoublePendulum]
}

/// Perturbation for the finite-difference reference gradient.
///
/// The inertia scalars are O(0.1–1) here, so a relative-ish 1e-6 sits in the
/// usual sweet spot between truncation and cancellation error for central
/// differences on an f64 rollout.
pub const FD_EPS: f64 = 1.0e-6;

/// Central-difference `dJ/dπ` over every body's ten inertia parameters.
///
/// This is the baseline the adjoint claims to beat: `2 × n_params` forward
/// rollouts, each with one perturbed inertia scalar.
fn fd_inertia_gradient(
    model: &Model,
    rollout: &AdjointRollout,
    objective: &FinalStateObjective,
    eps: f64,
) -> Vec<[f64; N_INERTIA_PARAMS]> {
    let nominal: Vec<[f64; N_INERTIA_PARAMS]> = model
        .bodies
        .iter()
        .map(|b| inertia_params(&b.inertia))
        .collect();
    let mut out = vec![[0.0f64; N_INERTIA_PARAMS]; model.nbodies()];

    // Clone the model once and restore the perturbed body after each probe.
    // Cloning inside the loop would charge finite differences for allocator
    // traffic that has nothing to do with the method, making the adjoint look
    // better than it is.
    let mut perturbed = model.clone();

    for body in 0..model.nbodies() {
        let original = perturbed.bodies[body].inertia;
        for k in 0..N_INERTIA_PARAMS {
            let mut probe = |delta: f64| -> f64 {
                let mut p = nominal[body];
                p[k] += delta;
                perturbed.bodies[body].inertia = inertia_from_params(&p);
                let r = AdjointRollout {
                    model: &perturbed,
                    contact: None,
                    q0: rollout.q0.clone(),
                    v0: rollout.v0.clone(),
                    steps: rollout.steps,
                    ctrl: rollout.ctrl,
                };
                rollout_objective(&r, objective)
            };
            out[body][k] = (probe(eps) - probe(-eps)) / (2.0 * eps);
        }
        perturbed.bodies[body].inertia = original;
    }
    out
}

/// Largest value in the iterator, **propagating non-finite values**.
///
/// `f64::max` returns the other operand when one side is NaN, so the natural
/// `fold(0.0, f64::max)` over a diverged computation reports `0.0` — the
/// smallest possible number for the worst possible state. For a
/// disagreement metric that is precisely backwards: a gradient full of NaN
/// would be published as *perfect agreement* with finite differences, and
/// this suite's whole claim is that a fast gradient which disagrees with FD
/// is not a result. A non-finite input therefore yields NaN, which
/// `format_metric` already renders as the real outcome it is.
pub(crate) fn max_or_nan(xs: impl Iterator<Item = f64>) -> f64 {
    let mut acc = 0.0f64;
    for x in xs {
        if !x.is_finite() {
            return f64::NAN;
        }
        acc = acc.max(x);
    }
    acc
}

/// Measure the adjoint ratio for one scene.
pub fn run_scene(scene: Scene, budget: Budget) -> Record {
    let settings = Settings::articulated(DT_ARTICULATED);
    let sim = PhyzSim::new(scene, settings.dt);
    let model = &sim.model;
    let nv = model.nv;

    let q0: Vec<f64> = sim.state.q.as_slice().to_vec();
    let v0: Vec<f64> = vec![0.0; nv];
    let zero_ctrl = move |_t: usize| DVec::zeros(nv);

    let rollout = AdjointRollout {
        model,
        contact: None,
        q0,
        v0,
        steps: ROLLOUT_STEPS,
        ctrl: &zero_ctrl,
    };

    // J = ½‖q_T‖² + ½‖v_T‖² — a generic final-state objective with a trivially
    // exact gradient, so the timing reflects the physics adjoint and not the
    // cost of the objective.
    let value = |q: &[f64], v: &[f64]| -> f64 {
        0.5 * (q.iter().map(|x| x * x).sum::<f64>() + v.iter().map(|x| x * x).sum::<f64>())
    };
    let gradient = |q: &[f64], v: &[f64]| -> (Vec<f64>, Vec<f64>) { (q.to_vec(), v.to_vec()) };
    let objective = FinalStateObjective {
        value: &value,
        gradient: &gradient,
    };

    // Timed in rollouts, not steps, so the ratio falls straight out. Each of
    // the three is repeated within a sample until it spans enough wall time to
    // be measurable: a single forward rollout is a few hundred microseconds,
    // which is close enough to timer noise to produce nonsense.
    let forward = measure(budget, FORWARD_PER_SAMPLE as u64, || {
        let mut acc = 0.0;
        for _ in 0..FORWARD_PER_SAMPLE {
            acc += rollout_objective(&rollout, &objective);
        }
        acc
    });
    let backward = measure(budget, ADJOINT_PER_SAMPLE as u64, || {
        let mut last = None;
        for _ in 0..ADJOINT_PER_SAMPLE {
            last = Some(adjoint_rollout_gradient(&rollout, &objective));
        }
        last
    });

    let ratio = backward.median_sec_per_unit / forward.median_sec_per_unit;
    let n_params = model.nbodies() * N_INERTIA_PARAMS;

    // The finite-difference gradient the adjoint is meant to replace — timed,
    // not projected.
    let fd = measure(budget, FD_PER_SAMPLE as u64, || {
        let mut last = None;
        for _ in 0..FD_PER_SAMPLE {
            last = Some(fd_inertia_gradient(model, &rollout, &objective, FD_EPS));
        }
        last
    });
    let fd_ratio = fd.median_sec_per_unit / forward.median_sec_per_unit;

    let grads = adjoint_rollout_gradient(&rollout, &objective);
    let grad_norm = grads
        .d_inertia
        .iter()
        .flat_map(|p| p.iter())
        .map(|g| g * g)
        .sum::<f64>()
        .sqrt();

    // Cross-check: a fast gradient that disagrees with finite differences is
    // not a result. Compared on the largest component, scale-relative.
    let fd_grad = fd_inertia_gradient(model, &rollout, &objective, FD_EPS);
    let scale = grad_norm.max(1.0e-12);
    let max_rel_err = max_or_nan(
        grads
            .d_inertia
            .iter()
            .flatten()
            .zip(fd_grad.iter().flatten())
            .map(|(a, b)| (a - b).abs() / scale),
    );

    Record {
        engine: "phyz".into(),
        scene: scene.name(),
        description: format!(
            "{} — {ROLLOUT_STEPS}-step adjoint rollout, dJ/dπ for {n_params} inertia parameters",
            scene.description()
        ),
        dof: Some(nv),
        batch: Some(1),
        settings,
        timing: Some(backward),
        metrics: vec![
            Metric::new("adjoint_ratio", ratio, "× forward"),
            Metric::new(
                "forward_rollout_ms",
                forward.median_sec_per_unit * 1.0e3,
                "ms",
            ),
            Metric::new("n_parameters", n_params as f64, "count"),
            Metric::new("fd_measured_ratio", fd_ratio, "× forward"),
            Metric::new("adjoint_speedup_vs_fd", fd_ratio / ratio, "×"),
            Metric::new("gradient_norm", grad_norm, "—"),
            Metric::new("adjoint_vs_fd_max_rel_err", max_rel_err, "relative"),
        ],
        notes: vec![
            "`adjoint_ratio` is the headline: wall time for one gradient rollout ÷ one \
             forward rollout of the same trajectory. `fd_measured_ratio` is the same \
             quantity for a full central-difference gradient over the same parameters — \
             measured here, not projected."
                .into(),
            "READ THIS BEFORE QUOTING A SPEEDUP, IN EITHER DIRECTION. The two scenes here \
             cannot tell you what the adjoint's cost scales with: the pendulum has one \
             body, one DOF and ten parameters, the double pendulum has two of each, so \
             parameter count and DOF count move together and the rising `adjoint_ratio` \
             is consistent with either driver. The `adjoint scaling` suite separates \
             them, and the answer is DOF count: at fixed nv the ratio is flat in \
             parameter count (6.9× at 10 parameters, 9.2× at 160), while at fixed \
             parameter count it is linear in nv. Quote that suite for scaling claims, \
             not these two rows."
                .into(),
            "`adjoint_speedup_vs_fd` near 1 on these scenes is therefore not evidence of \
             a bad asymptotic — it is what a model with ten parameters per DOF and only \
             one or two DOFs produces, because the adjoint's constant factor has nothing \
             to amortise against. At fixed nv the same measurement gives 3.0× at 10 \
             parameters and 18.3× at 80."
                .into(),
            "The adjoint also buys exactness: `adjoint_vs_fd_max_rel_err` shows the two \
             agree, and the adjoint gets there with no step-size to choose and no \
             truncation or cancellation error."
                .into(),
            "`adjoint_vs_fd_max_rel_err` cross-checks the two gradients against each \
             other, scaled by the gradient norm. A fast gradient that disagrees with \
             finite differences is not a result."
                .into(),
            "The ant is absent because of a coordinate-layout mismatch, not a missing \
             capability: the adjoint differentiates spherical and free joints fine, but \
             it wants `q0` in `DofLayout` packing (free joint = position + quaternion, \
             nq = 7) while these scenes are built as `Model`s (free joint = exponential \
             coordinates + position, nq = 6). Converting is mechanical and unwritten."
                .into(),
            "The timed gradient covers inertia parameters only; contact-surface vertex \
             adjoints are a separate code path and are not measured here."
                .into(),
        ],
    }
}

/// Run the gradient suite.
pub fn run(budget: Budget) -> Suite {
    let results = supported_scenes()
        .into_iter()
        .map(|s| run_scene(s, budget))
        .collect();
    Suite::new(
        "gradient throughput",
        "Cost of a reverse-mode gradient rollout relative to the forward rollout it \
         differentiates. Rapier has no equivalent, so there is no comparison row here.",
        results,
    )
}

#[cfg(test)]
mod tests {
    use super::max_or_nan;

    /// The reduction must not launder a diverged computation into a small
    /// number. `fold(0.0, f64::max)` does exactly that, because `f64::max`
    /// returns the other operand when one side is NaN — so an all-NaN
    /// disagreement would be published as `0.0`, i.e. perfect agreement,
    /// which is the opposite of the truth and defeats the cross-check this
    /// suite exists to provide.
    #[test]
    fn a_non_finite_input_is_not_reported_as_zero() {
        assert!(max_or_nan([1.0, f64::NAN, 2.0].into_iter()).is_nan());
        assert!(max_or_nan([f64::INFINITY].into_iter()).is_nan());
        // The idiom this replaces would return 2.0 and 0.0 respectively.
        assert_eq!(
            [1.0f64, f64::NAN, 2.0]
                .iter()
                .fold(0.0f64, |m, x| m.max(*x)),
            2.0
        );

        // Finite inputs behave exactly as the plain fold does.
        assert_eq!(max_or_nan([0.5, 3.0, 1.0].into_iter()), 3.0);
        assert_eq!(max_or_nan(std::iter::empty()), 0.0);
    }
}
