//! Inverse kinematics: solve for a configuration that puts bodies where you
//! want them.
//!
//! Forward kinematics answers "given `q`, where is the hand?". This module
//! answers the other direction — "what `q` puts the hand *there*?" — for an
//! arbitrary set of position and orientation goals at once, on any model the
//! rest of the crate can simulate.
//!
//! # Method
//!
//! Levenberg–Marquardt on the stacked task Jacobian. Each iteration builds the
//! residual `e(q)` (goal minus current, `3` rows per goal) and its Jacobian
//! `J = de/dqd` from [`point_jacobian`] and [`body_angular_jacobian`], then
//! solves the damped normal equations
//!
//! ```text
//! (JᵀJ + λ²I) δ = Jᵀe
//! ```
//!
//! and applies `δ` through [`integrate_configuration`], so free and ball
//! joints move on the manifold instead of having angles added into quaternion
//! slots. λ adapts: halved on an accepted step, multiplied by ten on a
//! rejected one. That is what makes this behave at a singularity — a plain
//! pseudo-inverse blows up when the arm straightens, and the damping trades a
//! little convergence rate for never doing that.
//!
//! # Redundancy, limits, and what "converged" means
//!
//! * More DOFs than task rows is fine and normal; damping picks the
//!   minimum-norm step, so a redundant arm drifts as little as it can from the
//!   seed you hand it. **The seed matters** — IK is not a function, and a
//!   7-DOF arm has a continuum of answers.
//! * Single-DOF joint limits are enforced by clamping after each accepted
//!   step. That is a projection, not a constraint in the solve: with a goal
//!   outside the reachable set the solver converges to a clamped configuration
//!   and reports `converged == false` with the residual it got stuck at.
//! * [`IkConfig::locked`] freezes DOFs entirely (a fixed torso, a gripper you
//!   do not want IK to open), by dropping their columns from the solve.
//!
//! Solving is deterministic: same `(model, seed, goals, config)` in, same bits
//! out, on every platform the rest of phyz is reproducible on.
//!
//! # Example
//!
//! ```
//! use phyz_math::{Mat3, SpatialInertia, SpatialTransform, Vec3};
//! use phyz_model::ModelBuilder;
//! use phyz_rigid::ik::{IkConfig, IkGoal, solve_ik};
//!
//! let link = |m: f64| SpatialInertia::new(m, Vec3::zeros(), Mat3::identity() * 0.01);
//! let model = ModelBuilder::new()
//!     .dt(1e-3)
//!     .add_revolute_body("shoulder", -1, SpatialTransform::identity(), link(1.0))
//!     .add_revolute_body(
//!         "elbow",
//!         0,
//!         SpatialTransform::new(Mat3::identity(), Vec3::new(1.0, 0.0, 0.0)),
//!         link(1.0),
//!     )
//!     .build();
//!
//! // Put the tip of the second link at (1.0, 1.0, 0.0).
//! let goals = [IkGoal::position(1, Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0))];
//! let seed = model.default_state().q;
//! let sol = solve_ik(&model, seed.as_slice(), &goals, &IkConfig::default());
//! assert!(sol.converged, "residual {}", sol.residual);
//! ```

use crate::{body_angular_jacobian, forward_kinematics, integrate_configuration, point_jacobian};
use phyz_math::{DMat, DVec, Mat3, Quat, SpatialTransformExt, Vec3};
use phyz_model::Model;

/// A single kinematic goal.
///
/// Build with [`IkGoal::position`] or [`IkGoal::orientation`] and adjust
/// `weight` to trade goals off against each other — weights scale the
/// residual rows, so a goal at weight `10.0` is worth a hundred times one at
/// weight `1.0` in the least-squares sense.
#[derive(Debug, Clone, PartialEq)]
pub struct IkGoal {
    /// Body the goal applies to.
    pub body: usize,
    /// What is being asked of it.
    pub kind: IkGoalKind,
    /// Relative importance. Defaults to `1.0`.
    pub weight: f64,
}

/// The two things a goal can ask for.
#[derive(Debug, Clone, PartialEq)]
pub enum IkGoalKind {
    /// Put the body-frame point `local_point` at the world point `target`.
    Position {
        /// Point in **body** coordinates — the tool tip, the foot, the camera.
        local_point: Vec3,
        /// Where it should end up, in **world** coordinates.
        target: Vec3,
    },
    /// Give the body the world→body rotation `target`.
    ///
    /// Same convention as [`phyz_math::SpatialTransform::rot`] and everything
    /// [`forward_kinematics`] produces: `target` maps world directions into
    /// the body frame, so its transpose is the body's orientation in the
    /// world. Residual rows are the rotation vector (axis × angle, radians)
    /// that would carry the current orientation onto the target, in world
    /// axes.
    Orientation {
        /// Desired world→body rotation.
        target: Mat3,
    },
}

impl IkGoal {
    /// A position goal at unit weight: put `local_point` (body frame) on
    /// `target` (world frame).
    pub fn position(body: usize, local_point: Vec3, target: Vec3) -> Self {
        Self {
            body,
            kind: IkGoalKind::Position {
                local_point,
                target,
            },
            weight: 1.0,
        }
    }

    /// An orientation goal at unit weight. `target` is the desired
    /// **world→body** rotation; see [`IkGoalKind::Orientation`].
    pub fn orientation(body: usize, target: Mat3) -> Self {
        Self {
            body,
            kind: IkGoalKind::Orientation { target },
            weight: 1.0,
        }
    }

    /// Set the goal's weight, builder-style.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

/// Solver settings. [`IkConfig::default`] is tuned for robot-arm-sized
/// problems: metres and radians, goals within reach, a seed that is not
/// exactly at a singularity.
#[derive(Debug, Clone, PartialEq)]
pub struct IkConfig {
    /// Maximum Levenberg–Marquardt iterations, counting rejected trials.
    pub max_iterations: usize,
    /// Convergence threshold on the largest single residual component —
    /// metres for position rows, radians for orientation rows.
    pub tolerance: f64,
    /// Initial damping λ. Larger is slower and steadier.
    pub damping: f64,
    /// Floor on λ. Keeps the normal equations conditioned near singularities;
    /// zero is legal but asks for a numerically exact rank decision.
    pub min_damping: f64,
    /// Ceiling on λ. Once damping has to exceed this to make progress, the
    /// solver has stalled and returns.
    pub max_damping: f64,
    /// Fraction of the computed step to take. `1.0` is the LM step; lower it
    /// only when the model has joints whose Jacobian changes violently.
    pub step_scale: f64,
    /// Enforce single-DOF joint limits by clamping after each accepted step.
    pub respect_limits: bool,
    /// Velocity-index DOFs to hold fixed (indices into `0..model.nv`).
    pub locked: Vec<usize>,
}

impl Default for IkConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            damping: 1e-3,
            min_damping: 1e-9,
            max_damping: 1e6,
            step_scale: 1.0,
            respect_limits: true,
            locked: Vec::new(),
        }
    }
}

/// What [`solve_ik`] found.
#[derive(Debug, Clone, PartialEq)]
pub struct IkSolution {
    /// The configuration, in the same layout as [`phyz_model::State::q`].
    pub q: Vec<f64>,
    /// Largest absolute residual component at `q` — metres or radians,
    /// whichever goal is worst. This is the number to report, not the
    /// least-squares cost.
    pub residual: f64,
    /// Whether `residual` fell below [`IkConfig::tolerance`].
    pub converged: bool,
    /// Iterations consumed, rejected trials included.
    pub iterations: usize,
    /// Final damping. A large value on return means the solver was fighting a
    /// singularity or an unreachable goal.
    pub damping: f64,
}

/// Solve for a configuration satisfying `goals`, starting from `seed`.
///
/// `seed` is a `q` vector in [`phyz_model::State::q`] layout; pass
/// `model.default_state().q` for a cold start, or the previous solution when
/// tracking a moving target (much faster, and it keeps a redundant arm from
/// jumping between branches between frames).
///
/// Never panics on an unreachable or over-constrained goal set: it returns the
/// least-squares compromise with `converged == false`. An empty goal list
/// returns the seed.
pub fn solve_ik(model: &Model, seed: &[f64], goals: &[IkGoal], config: &IkConfig) -> IkSolution {
    let mut q = seed.to_vec();
    q.resize(model.nq, 0.0);

    if goals.is_empty() || model.nv == 0 {
        return IkSolution {
            q,
            residual: 0.0,
            converged: true,
            iterations: 0,
            damping: config.damping,
        };
    }

    let rows = 3 * goals.len();
    let mut lambda = config.damping.max(config.min_damping);
    let mut residual = residual_vector(model, &q, goals);
    let mut cost = residual.norm_sq();
    let mut worst = max_abs(&residual);
    let mut iterations = 0;

    while iterations < config.max_iterations {
        if worst <= config.tolerance {
            break;
        }
        iterations += 1;

        let jac = task_jacobian(model, &q, goals, &config.locked);

        // Damped normal equations. `rows` may be smaller or larger than nv;
        // the nv x nv form handles both, and with λ > 0 it is symmetric
        // positive definite, so Cholesky is the right factorization.
        let jt = jac.transpose();
        let mut normal = &jt * &jac;
        for i in 0..model.nv {
            normal[(i, i)] += lambda * lambda;
        }
        // Locked DOFs already have zero columns in `jac`; give them a unit
        // diagonal so the matrix stays non-singular and their step is zero.
        for &d in &config.locked {
            if d < model.nv {
                for k in 0..model.nv {
                    normal[(d, k)] = 0.0;
                    normal[(k, d)] = 0.0;
                }
                normal[(d, d)] = 1.0;
            }
        }
        let rhs = &jt * &residual;

        let Some(delta) = solve_spd(&normal, &rhs) else {
            // Factorization failed outright (λ far too small for a rank-
            // deficient J). Crank the damping and retry rather than giving up.
            lambda = (lambda * 10.0).max(config.min_damping * 10.0);
            if lambda > config.max_damping {
                break;
            }
            continue;
        };

        let mut trial = q.clone();
        let step: Vec<f64> = (0..model.nv)
            .map(|i| delta[i] * config.step_scale)
            .collect();
        integrate_configuration(model, trial.as_mut_slice(), &step, 1.0);
        if config.respect_limits {
            clamp_to_limits(model, &mut trial);
        }

        let trial_residual = residual_vector(model, &trial, goals);
        let trial_cost = trial_residual.norm_sq();

        if trial_cost < cost {
            q = trial;
            residual = trial_residual;
            cost = trial_cost;
            worst = max_abs(&residual);
            lambda = (lambda * 0.5).max(config.min_damping);
        } else {
            // Reject and stiffen. A clamped step that cannot improve the cost
            // at any damping is the unreachable-goal exit.
            lambda *= 10.0;
            if lambda > config.max_damping {
                break;
            }
        }
    }

    debug_assert_eq!(rows, 3 * goals.len());

    IkSolution {
        q,
        residual: worst,
        converged: worst <= config.tolerance,
        iterations,
        damping: lambda,
    }
}

/// Residual `e(q)`: for each goal, three rows of "how far, and which way".
///
/// Exposed because it is the honest way to score a configuration a solver
/// handed back — and the thing to plot when IK is misbehaving.
pub fn residual_vector(model: &Model, q: &[f64], goals: &[IkGoal]) -> DVec {
    let mut state = model.default_state();
    state.q = DVec::from_slice(q);
    let (xforms, _) = forward_kinematics(model, &state);

    let mut e = DVec::zeros(3 * goals.len());
    for (g, goal) in goals.iter().enumerate() {
        let xf = &xforms[goal.body];
        let row = 3 * g;
        let v = match &goal.kind {
            IkGoalKind::Position {
                local_point,
                target,
            } => *target - xf.body_to_world_point(*local_point),
            IkGoalKind::Orientation { target } => {
                // exp(ω) · B_current = B_target with B = Rᵀ (body→world),
                // so exp(ω) = B_target · B_currentᵀ = Rᵀ_target · R_current.
                Quat::from_matrix(&(target.transpose() * xf.rot)).log()
            }
        };
        e[row] = v.x * goal.weight;
        e[row + 1] = v.y * goal.weight;
        e[row + 2] = v.z * goal.weight;
    }
    e
}

/// The stacked `3·goals x nv` task Jacobian at `q`, weighted to match
/// [`residual_vector`], with locked DOF columns zeroed.
fn task_jacobian(model: &Model, q: &[f64], goals: &[IkGoal], locked: &[usize]) -> DMat {
    let mut state = model.default_state();
    state.q = DVec::from_slice(q);
    let (xforms, _) = forward_kinematics(model, &state);

    let mut j = DMat::zeros(3 * goals.len(), model.nv);
    for (g, goal) in goals.iter().enumerate() {
        let xf = &xforms[goal.body];
        let block = match &goal.kind {
            IkGoalKind::Position { local_point, .. } => {
                let world = xf.body_to_world_point(*local_point);
                point_jacobian(model, &xforms, goal.body, world)
            }
            IkGoalKind::Orientation { .. } => body_angular_jacobian(model, &xforms, goal.body),
        };
        for r in 0..3 {
            for c in 0..model.nv {
                j[(3 * g + r, c)] = block[(r, c)] * goal.weight;
            }
        }
    }
    for &d in locked {
        if d < model.nv {
            for r in 0..j.nrows() {
                j[(r, d)] = 0.0;
            }
        }
    }
    j
}

/// Clamp single-DOF joints to their declared limits.
///
/// Multi-DOF joints are skipped: `Joint::limits` is a scalar interval, which
/// says nothing usable about a ball or free joint.
fn clamp_to_limits(model: &Model, q: &mut [f64]) {
    for (j, joint) in model.joints.iter().enumerate() {
        let Some([lo, hi]) = joint.limits else {
            continue;
        };
        if joint.ndof() != 1 {
            continue;
        }
        let Some(&idx) = model.q_offsets.get(j) else {
            continue;
        };
        if idx < q.len() {
            q[idx] = q[idx].clamp(lo, hi);
        }
    }
}

/// Solve `A x = b` for the damped normal matrix, rejecting a non-finite
/// result so the caller can stiffen the damping instead of propagating NaN.
fn solve_spd(a: &DMat, b: &DVec) -> Option<DVec> {
    a.clone()
        .lu()
        .solve(b)
        .filter(|x: &DVec| x.as_slice().iter().all(|v: &f64| v.is_finite()))
}

/// Largest absolute component, **propagating non-finite values** instead of
/// swallowing them.
///
/// `f64::max` returns the other operand when one side is NaN, so the natural
/// `fold(0.0, f64::max)` over a diverged vector reports `0.0` — the smallest
/// possible answer for the worst possible state. Here that would set
/// `residual = 0.0` and `converged = true` on a solution full of NaN, which is
/// the single worst thing this function could do: it is what decides whether
/// the caller trusts the result.
///
/// A non-finite residual is not a small residual, so it reports NaN, and every
/// `residual <= tolerance` comparison downstream is then false.
fn max_abs(v: &DVec) -> f64 {
    if v.as_slice().iter().any(|x| !x.is_finite()) {
        return f64::NAN;
    }
    v.as_slice().iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{Mat3, SpatialInertia, SpatialTransform};
    use phyz_model::ModelBuilder;

    fn link(m: f64) -> SpatialInertia {
        SpatialInertia::new(m, Vec3::zeros(), Mat3::identity() * 0.01)
    }

    /// Planar two-link arm, links along +X, revolute about +Z.
    fn two_link() -> Model {
        ModelBuilder::new()
            .dt(1e-3)
            .add_revolute_body("shoulder", -1, SpatialTransform::identity(), link(1.0))
            .add_revolute_body(
                "elbow",
                0,
                SpatialTransform::new(Mat3::identity(), Vec3::new(1.0, 0.0, 0.0)),
                link(1.0),
            )
            .build()
    }

    fn tip(model: &Model, q: &[f64]) -> Vec3 {
        let mut s = model.default_state();
        s.q = DVec::from_slice(q);
        let (xf, _) = forward_kinematics(model, &s);
        xf[1].body_to_world_point(Vec3::new(1.0, 0.0, 0.0))
    }

    #[test]
    fn reaches_a_point_in_the_workspace() {
        let model = two_link();
        let target = Vec3::new(1.0, 1.0, 0.0);
        let goals = [IkGoal::position(1, Vec3::new(1.0, 0.0, 0.0), target)];
        let sol = solve_ik(&model, &[0.1, 0.1], &goals, &IkConfig::default());

        assert!(
            sol.converged,
            "residual {} after {} it",
            sol.residual, sol.iterations
        );
        assert!(
            (tip(&model, &sol.q) - target).norm() < 1e-6,
            "tip {:?}",
            tip(&model, &sol.q)
        );
    }

    /// The seed is already the answer: zero iterations, zero residual.
    #[test]
    fn seed_on_target_is_a_no_op() {
        let model = two_link();
        let seed = [0.3, -0.4];
        let goals = [IkGoal::position(
            1,
            Vec3::new(1.0, 0.0, 0.0),
            tip(&model, &seed),
        )];
        let sol = solve_ik(&model, &seed, &goals, &IkConfig::default());

        assert_eq!(sol.iterations, 0);
        assert!(sol.converged);
        for (a, b) in sol.q.iter().zip(seed.iter()) {
            assert_eq!(a, b);
        }
    }

    /// A goal outside the reachable set must not converge, must not diverge,
    /// and must land on the closest point the arm can make: fully extended
    /// along the direction of the target.
    #[test]
    fn unreachable_goal_stops_at_full_extension() {
        let model = two_link();
        let goals = [IkGoal::position(
            1,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
        )];
        let sol = solve_ik(&model, &[0.4, 0.4], &goals, &IkConfig::default());

        assert!(!sol.converged);
        let p = tip(&model, &sol.q);
        assert!(
            p.norm() > 2.0 - 1e-4,
            "not extended: {p:?} (|p| = {})",
            p.norm()
        );
        assert!(sol.residual.is_finite());
    }

    /// Singular seed: a straight arm has a rank-deficient Jacobian. Damping is
    /// the whole reason this does not produce NaN or a step to infinity.
    #[test]
    fn straight_arm_singularity_is_survivable() {
        let model = two_link();
        let goals = [IkGoal::position(
            1,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.4, 0.9, 0.0),
        )];
        let sol = solve_ik(&model, &[0.0, 0.0], &goals, &IkConfig::default());

        assert!(sol.q.iter().all(|v| v.is_finite()), "q = {:?}", sol.q);
        assert!(sol.converged, "residual {}", sol.residual);
    }

    #[test]
    fn joint_limits_are_respected() {
        let mut model = two_link();
        model.joints[1].limits = Some([-0.2, 0.2]);
        let goals = [IkGoal::position(
            1,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.5, 0.0),
        )];
        let sol = solve_ik(&model, &[0.2, 0.0], &goals, &IkConfig::default());

        assert!(
            sol.q[1] >= -0.2 - 1e-12 && sol.q[1] <= 0.2 + 1e-12,
            "elbow escaped its limit: {}",
            sol.q[1]
        );
    }

    #[test]
    fn locked_dofs_do_not_move() {
        let model = two_link();
        let config = IkConfig {
            locked: vec![0],
            ..Default::default()
        };
        let goals = [IkGoal::position(
            1,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.2, 0.8, 0.0),
        )];
        let sol = solve_ik(&model, &[0.15, 0.15], &goals, &config);

        assert_eq!(sol.q[0], 0.15, "locked shoulder moved");
        assert_ne!(sol.q[1], 0.15, "elbow should still have moved");
    }

    /// Orientation goals: a free body must be able to rotate onto any target,
    /// which also pins down the world→body convention.
    #[test]
    fn orientation_goal_on_a_free_body() {
        let model = ModelBuilder::new()
            .dt(1e-3)
            .add_free_body(
                "ball",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::sphere(1.0, 0.1),
            )
            .build();

        let target = Quat::from_axis_angle(Vec3::new(0.3, 0.5, 0.8).normalize(), 0.9)
            .to_matrix()
            .transpose(); // world->body
        let goals = [IkGoal::orientation(0, target)];
        let sol = solve_ik(
            &model,
            model.default_state().q.as_slice(),
            &goals,
            &IkConfig::default(),
        );

        assert!(sol.converged, "residual {}", sol.residual);
        let e = residual_vector(&model, &sol.q, &goals);
        assert!(max_abs(&e) < 1e-6, "orientation residual {}", max_abs(&e));
    }

    /// Position and orientation together on the same body, which is the case
    /// that actually matters for a gripper.
    #[test]
    fn position_and_orientation_together() {
        let model = ModelBuilder::new()
            .dt(1e-3)
            .add_free_body(
                "hand",
                -1,
                SpatialTransform::identity(),
                SpatialInertia::sphere(1.0, 0.1),
            )
            .build();

        let target_pos = Vec3::new(0.4, -0.2, 0.7);
        let target_rot = Quat::from_axis_angle(Vec3::z(), 0.6)
            .to_matrix()
            .transpose();
        let goals = [
            IkGoal::position(0, Vec3::zeros(), target_pos),
            IkGoal::orientation(0, target_rot),
        ];
        let sol = solve_ik(
            &model,
            model.default_state().q.as_slice(),
            &goals,
            &IkConfig::default(),
        );

        assert!(sol.converged, "residual {}", sol.residual);
    }

    /// Weights decide who wins when two goals conflict.
    #[test]
    fn weights_break_ties_between_conflicting_goals() {
        let model = two_link();
        let a = Vec3::new(1.0, 1.0, 0.0);
        let b = Vec3::new(1.0, -1.0, 0.0);
        let solve = |wa: f64, wb: f64| {
            let goals = [
                IkGoal::position(1, Vec3::new(1.0, 0.0, 0.0), a).with_weight(wa),
                IkGoal::position(1, Vec3::new(1.0, 0.0, 0.0), b).with_weight(wb),
            ];
            tip(
                &model,
                &solve_ik(&model, &[0.1, 0.1], &goals, &IkConfig::default()).q,
            )
        };

        // Heavily favouring `a` must land closer to `a` than to `b`, and the
        // mirrored weighting must land closer to `b`.
        let toward_a = solve(100.0, 1.0);
        let toward_b = solve(1.0, 100.0);
        assert!((toward_a - a).norm() < (toward_a - b).norm());
        assert!((toward_b - b).norm() < (toward_b - a).norm());
    }

    /// Same inputs, same bits — the property the rest of phyz is built on.
    #[test]
    fn solving_is_deterministic() {
        let model = two_link();
        let goals = [IkGoal::position(
            1,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.7, 1.1, 0.0),
        )];
        let a = solve_ik(&model, &[0.1, 0.2], &goals, &IkConfig::default());
        let b = solve_ik(&model, &[0.1, 0.2], &goals, &IkConfig::default());
        assert_eq!(a, b);
    }

    /// A non-finite seed must not be reported as a converged solution.
    ///
    /// `f64::max` ignores NaN, so a max-abs reduction written the obvious way
    /// returns `0.0` for an all-NaN residual — the smallest possible number
    /// for the worst possible state — and `converged` would be `true` on
    /// garbage. This is the guard against that.
    #[test]
    fn a_non_finite_seed_is_never_reported_as_converged() {
        let model = two_link();
        let goals = [IkGoal::position(
            1,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        )];
        let sol = solve_ik(&model, &[f64::NAN, 0.1], &goals, &IkConfig::default());
        assert!(
            !sol.converged,
            "reported convergence with residual {} on a NaN seed",
            sol.residual
        );
    }

    #[test]
    fn no_goals_returns_the_seed() {
        let model = two_link();
        let sol = solve_ik(&model, &[0.3, 0.4], &[], &IkConfig::default());
        assert_eq!(sol.q, vec![0.3, 0.4]);
        assert!(sol.converged);
        assert_eq!(sol.iterations, 0);
    }
}
