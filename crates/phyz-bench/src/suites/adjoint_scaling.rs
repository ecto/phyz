//! Dimension 6: what the adjoint's cost actually scales with.
//!
//! The gradient suite measures the adjoint ratio on two scenes and finds it
//! rising — 18× the forward pass at 10 parameters, 45× at 20. Read off two
//! points, the obvious conclusion is that the backward pass costs one lane per
//! parameter, which would put it in the same asymptotic class as finite
//! differences and make the whole exercise pointless.
//!
//! Those two scenes cannot distinguish that hypothesis from its rival, because
//! they vary both candidates at once: the pendulum has one body, one DOF and
//! ten parameters; the double pendulum has two of each. Parameter count and
//! DOF count move together, so no amount of staring at those rows says which
//! one is driving the cost.
//!
//! This suite separates them, with a model family designed for exactly that.
//!
//! * **`weld_chain_N`** — one revolute joint carrying `N` welded bodies behind
//!   it. Every welded body has a full ten-parameter spatial inertia and
//!   contributes to the dynamics, but `Fixed` joints add no DOF: `nv` stays at
//!   1 while the parameter count grows to `10·(N+1)`. **Parameter count varies;
//!   DOF count does not.**
//! * **`dof_chain_N`** — an ordinary `N`-link revolute chain, where the two
//!   move together as they do in the gradient suite. The control.
//!
//! If the adjoint really costs a lane per parameter, the `weld_chain` ratios
//! climb with `N`. If the cost is the **state** Jacobian — `nq + nv` dual
//! lanes through ABA per step, which is what
//! [`phyz_diff::rollout::adjoint_rollout_gradient`] actually does — they stay
//! flat, and only `dof_chain` climbs.
//!
//! The `weld_chain` rows also carry a measured finite-difference gradient, so
//! the asymptotic claim is tested where it is supposed to bite: at fixed DOF
//! count, FD costs `2·n_params` rollouts and the adjoint should not.

use phyz_diff::rollout::step::inertia_from_params;
use phyz_diff::rollout::{
    AdjointRollout, FinalStateObjective, N_INERTIA_PARAMS, adjoint_rollout_gradient,
    inertia_params, rollout_objective,
};
use phyz_math::{DVec, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Model, ModelBuilder};

use crate::report::{Metric, Record, Suite};
use crate::settings::{DT_ARTICULATED, Settings};
use crate::timing::{Budget, measure};

/// Rollout length. Shorter than the gradient suite's 500 because this suite
/// runs a dozen models rather than two, and the ratio is what is being
/// measured — it does not depend on the horizon.
pub const ROLLOUT_STEPS: usize = 200;

/// Forward rollouts per timed sample.
const FORWARD_PER_SAMPLE: usize = 50;
/// Adjoint gradients per timed sample.
const ADJOINT_PER_SAMPLE: usize = 5;

/// Body counts swept in both families.
const SWEEP: [usize; 5] = [1, 2, 4, 8, 16];

/// Largest body count that still gets a measured finite-difference gradient.
///
/// FD costs `2·n_params` rollouts, so the 16-body model runs 320 of them per
/// sample. That is minutes of wall clock for a row whose conclusion is already
/// established by the smaller ones — so it is skipped, and
/// `fd_measured` records that it was.
const FD_MAX_BODIES: usize = 8;

/// A uniform link: 1 kg, COM one half-length down, diagonal inertia.
fn link() -> SpatialInertia {
    SpatialInertia::new(
        1.0,
        Vec3::new(0.0, -0.5, 0.0),
        Mat3::from_diagonal(&Vec3::new(0.1, 0.1, 0.1)),
    )
}

/// `n_dof` revolute links followed by `n_welded` rigidly attached bodies.
///
/// `nv == n_dof`; the parameter count is `10·(n_dof + n_welded)`. Welded
/// bodies are the lever this suite pulls: real inertia parameters, no DOF.
fn chain(n_dof: usize, n_welded: usize, dt: f64) -> Model {
    let offset = SpatialTransform::new(Mat3::identity(), Vec3::new(0.0, -1.0, 0.0));
    let mut b = ModelBuilder::new()
        .gravity(Vec3::new(0.0, -phyz_math::GRAVITY, 0.0))
        .dt(dt);
    for i in 0..n_dof {
        let xf = if i == 0 {
            SpatialTransform::identity()
        } else {
            offset
        };
        b = b.add_revolute_body(&format!("link{i}"), i as i32 - 1, xf, link());
    }
    for k in 0..n_welded {
        b = b.add_fixed_body(&format!("weld{k}"), (n_dof + k) as i32 - 1, offset, link());
    }
    b.build()
}

/// `J = ½‖q_T‖² + ½‖v_T‖²` — a generic final-state objective whose gradient is
/// trivially exact, so the timing reflects the physics adjoint and not the
/// objective.
fn objective_value(q: &[f64], v: &[f64]) -> f64 {
    0.5 * (q.iter().map(|x| x * x).sum::<f64>() + v.iter().map(|x| x * x).sum::<f64>())
}

fn objective_gradient(q: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
    (q.to_vec(), v.to_vec())
}

/// Central-difference `dJ/dπ` over every inertia parameter: `2·n_params`
/// forward rollouts, the model cloned once so the baseline is not charged for
/// allocator traffic.
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

/// Measure one model.
fn run_model(name: String, model: Model, family: &str, budget: Budget, with_fd: bool) -> Record {
    let settings = Settings::articulated(model.dt);
    let nv = model.nv;
    let nb = model.nbodies();
    let n_params = nb * N_INERTIA_PARAMS;

    // A configuration that is off-axis in every joint, so no DOF sits at a
    // symmetry point where its contribution to the Jacobian vanishes.
    let q0: Vec<f64> = (0..model.nq).map(|i| 0.3 + 0.01 * i as f64).collect();
    let zero_ctrl = move |_t: usize| DVec::zeros(nv);
    let rollout = AdjointRollout {
        model: &model,
        contact: None,
        q0,
        v0: vec![0.0; nv],
        steps: ROLLOUT_STEPS,
        ctrl: &zero_ctrl,
    };
    let objective = FinalStateObjective {
        value: &objective_value,
        gradient: &objective_gradient,
    };

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

    let mut metrics = vec![
        Metric::new("adjoint_ratio", ratio, "× forward"),
        Metric::new("n_parameters", n_params as f64, "count"),
        Metric::new("nv", nv as f64, "DOF"),
        Metric::new("n_bodies", nb as f64, "count"),
        Metric::new(
            "forward_rollout_ms",
            forward.median_sec_per_unit * 1.0e3,
            "ms",
        ),
        // The quantity the two hypotheses disagree about. Under "a lane per
        // parameter" this is constant across a family; under "a lane per DOF"
        // it is constant down the weld_chain column and rises with N.
        Metric::new("ratio_per_parameter", ratio / n_params as f64, "× / param"),
        Metric::new("ratio_per_dof", ratio / nv as f64, "× / DOF"),
    ];
    let mut notes = Vec::new();

    if with_fd {
        let fd = measure(budget, 1, || {
            fd_inertia_gradient(&model, &rollout, &objective, super::gradient::FD_EPS)
        });
        let fd_ratio = fd.median_sec_per_unit / forward.median_sec_per_unit;
        metrics.push(Metric::new("fd_measured_ratio", fd_ratio, "× forward"));
        metrics.push(Metric::new("adjoint_speedup_vs_fd", fd_ratio / ratio, "×"));
        metrics.push(Metric::new("fd_measured", 1.0, "bool"));

        // A fast gradient that disagrees with finite differences is not a
        // result. Checked on every row that has an FD reference.
        let grads = adjoint_rollout_gradient(&rollout, &objective);
        let fd_grad = fd_inertia_gradient(&model, &rollout, &objective, super::gradient::FD_EPS);
        let scale = grads
            .d_inertia
            .iter()
            .flatten()
            .map(|g| g * g)
            .sum::<f64>()
            .sqrt()
            .max(1.0e-12);
        let max_rel_err = super::gradient::max_or_nan(
            grads
                .d_inertia
                .iter()
                .flatten()
                .zip(fd_grad.iter().flatten())
                .map(|(a, b)| (a - b).abs() / scale),
        );
        metrics.push(Metric::new(
            "adjoint_vs_fd_max_rel_err",
            max_rel_err,
            "relative",
        ));
    } else {
        metrics.push(Metric::new("fd_measured", 0.0, "bool"));
        notes.push(format!(
            "Finite differences not measured for this row: {n_params} parameters is \
             {} forward rollouts per sample, and the smaller rows in the same family \
             already establish the scaling. Nothing is extrapolated into the missing \
             cells — they are simply absent.",
            2 * n_params
        ));
    }

    Record {
        engine: "phyz".into(),
        scene: name,
        description: format!(
            "{family} — {nb} bodies, nv = {nv}, {n_params} inertia parameters, \
             {ROLLOUT_STEPS}-step rollout"
        ),
        dof: Some(nv),
        batch: Some(1),
        settings,
        timing: Some(backward),
        metrics,
        notes,
    }
}

/// Run the adjoint-scaling suite.
pub fn run(budget: Budget) -> Suite {
    let mut results = Vec::new();

    // Family A: parameters vary, DOF held at 1.
    for n in SWEEP {
        let welded = n - 1;
        results.push(run_model(
            format!("weld_chain_{n}"),
            chain(1, welded, DT_ARTICULATED),
            "parameters vary, nv held at 1",
            budget,
            n <= FD_MAX_BODIES,
        ));
    }

    // Family B: the control — both vary together, as in the gradient suite.
    for n in SWEEP {
        results.push(run_model(
            format!("dof_chain_{n}"),
            chain(n, 0, DT_ARTICULATED),
            "parameters and nv vary together",
            budget,
            false,
        ));
    }

    Suite::new(
        "adjoint scaling",
        "What the adjoint's cost scales with. `weld_chain_N` grows the parameter count \
         while holding nv at 1 (welded bodies carry inertia parameters but add no DOF); \
         `dof_chain_N` grows both, as the gradient suite's scenes do. Compare \
         `adjoint_ratio` down each family: flat in the first and rising in the second \
         means the cost is the state Jacobian's dual lanes, not the parameter count.",
        results,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lever this suite depends on: welded bodies must add parameters
    /// without adding DOF. If `Fixed` joints ever start carrying a DOF, every
    /// conclusion here silently inverts, so assert it.
    #[test]
    fn welded_bodies_add_parameters_but_not_dofs() {
        for n in [1usize, 4, 16] {
            let m = chain(1, n - 1, DT_ARTICULATED);
            assert_eq!(m.nv, 1, "weld_chain_{n} should have one DOF");
            assert_eq!(m.nbodies(), n, "weld_chain_{n} should have {n} bodies");
            assert_eq!(m.nbodies() * N_INERTIA_PARAMS, 10 * n);
        }
    }

    /// The control family must actually vary the DOF count, or it is not a
    /// control.
    #[test]
    fn dof_chain_varies_dofs() {
        for n in [1usize, 4, 16] {
            let m = chain(n, 0, DT_ARTICULATED);
            assert_eq!(m.nv, n);
            assert_eq!(m.nbodies(), n);
        }
    }

    /// The welded bodies must be dynamically live — if a welded link did not
    /// affect the trajectory, its parameter gradients would be zero and the
    /// suite would be timing work on a model that ignores most of its own
    /// parameters.
    #[test]
    fn welded_bodies_affect_the_gradient() {
        let model = chain(1, 3, DT_ARTICULATED);
        let zero = DVec::zeros(model.nv);
        let ctrl = |_t: usize| zero.clone();
        let rollout = AdjointRollout {
            model: &model,
            contact: None,
            q0: vec![0.3],
            v0: vec![0.0],
            steps: 50,
            ctrl: &ctrl,
        };
        let objective = FinalStateObjective {
            value: &objective_value,
            gradient: &objective_gradient,
        };
        let g = adjoint_rollout_gradient(&rollout, &objective);

        for (b, params) in g.d_inertia.iter().enumerate() {
            let norm = params.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(norm > 1e-12, "body {b} has a zero parameter gradient");
        }
    }
}
