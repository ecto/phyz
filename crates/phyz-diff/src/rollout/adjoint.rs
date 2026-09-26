//! Discrete trajectory adjoint for semi-implicit Euler rollouts.
//!
//! # What this computes
//!
//! For a rollout `x_{t+1} = f(x_t, u_t; π, V)` (states `x = (q, v)`, open-loop
//! controls `u_t`, per-body inertia parameters `π`, collision-mesh vertices
//! `V`) and a final-state objective `J = g(q_T, v_T)`, one backward pass
//! yields **both** parameter sensitivities:
//!
//! - `dJ/dπ` — per body, in the canonical 10-vector packing
//!   `[m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]` (Task: the inertia
//!   parameter adjoint that replaces vcad's finite-difference `∂J/∂p`).
//! - `dJ/dV` — per collision-mesh vertex, `∂J/∂x` in the body frame (the
//!   contact adjoint that plugs into vcad's `surface_gradient`).
//!
//! # How
//!
//! Reverse over the trajectory, tangent within the step. The adjoint state
//! `λ_t = ∂J/∂x_t` is backpropagated through the step Jacobians; those
//! Jacobians are read out **exactly** with [`tang::Dual`] lanes through the
//! scalar-generic step of [`super::step`] — no finite differences anywhere.
//!
//! Semi-implicit Euler has the structure `v' = v + dt·a`, `q' = Φ(q, v')`
//! with `a = ABA(q, v, u; π, V)`. Differentiating and collecting terms, with
//! `w := Φ_v'ᵀ λ_q' + λ_v'` (an `nv`-covector):
//!
//! ```text
//! λ_q = Φ_qᵀ λ_q' + dt·(∂a/∂q)ᵀ w     λ_v = w + dt·(∂a/∂v)ᵀ w
//! dJ/dπ += dt·wᵀ·∂a/∂π               dJ/dV += dt·wᵀ·∂a/∂V
//! ```
//!
//! For a model of single-DOF joints only, `Φ(q, v') = q + dt·v'`, so
//! `Φ_q = I`, `Φ_v' = dt·I`, and this collapses to the familiar
//! `w = dt·λ_q' + λ_v'`, `λ_q = λ_q' + dt·(∂a/∂q)ᵀw` — and the driver takes
//! that closed form directly, so a single-DOF model pays nothing for this.
//! With a spherical or free joint `Φ` is a Lie-group step on quaternion
//! coordinates: `Φ_q` is not the identity and `Φ_v'` is `nq×nv`, not square
//! (see [`super::step`]'s module docs for the layout). Both blocks are then
//! read out with dual lanes through `Φ` alone — `nq + nv` lanes of a few dozen
//! flops each, negligible beside the ABA lanes.
//!
//! The two **state** channels are dual lanes: `2·nv` seeded evaluations of
//! the scalar-generic step per timestep.
//!
//! The **parameter** and **contact** channels are not — a lane per parameter
//! would make the backward pass cost `O(steps · n_params)`, i.e. no better
//! than finite differences. Both instead ride one linear solve. Write
//! `y := M⁻¹w` (read out of a single ABA with zero velocity and zero gravity,
//! where the bias and gravity terms vanish and `qdd = M⁻¹·ctrl`) and let
//! `A^y := J y` be the body-frame motion field that joint rates `y` induce
//! (one `O(nb)` velocity sweep). Then, since `M ∂a/∂π = −∂/∂π ID(q, v, a; π)`
//! at fixed `(q, v, a)` and inverse dynamics is the sum over bodies of
//! `⟨A^y_b, I_b A_b + V_b ×* I_b V_b⟩` (RNEA's backward pass is the adjoint of
//! its acceleration pass):
//!
//! ```text
//! dJ/dπ[b][k] −= dt·⟨A^y_b, ΔI·A_b + V_b ×* (ΔI·V_b)⟩,   ΔI := ∂I_b/∂π_k
//! χ_b          = dt·A^y_b                                (= dt·wᵀ·∂a/∂wrench_b)
//! ```
//!
//! `ΔI` is a constant of the model, priced once outside the time loop; `V_b`
//! and `A_b` come from one `O(nb)` sweep at the nominal `(q, v, qdd)` recorded
//! by the forward rollout. So **all** `10·nb` inertia directions and all
//! `6·nb` wrench directions cost two `O(nb)` sweeps, not `16·nb` dual steps.
//!
//! The vertex channel then contracts `χ_b` against the **local** per-vertex
//! wrench Jacobian (3 tiny dual evaluations of the penalty law per vertex),
//! so vertices never cost a full step either. Cost per timestep is
//! `O(nv)` dual steps + `O(1)` ABAs + `O(nb)` sweeps + `O(3·N)` local
//! evaluations — **independent of the parameter count**.
//!
//! # Contract
//!
//! - Joint domain: all of them — revolute/prismatic/fixed plus spherical and
//!   free. Multi-DOF joints carry quaternion configuration, so `q0` must be
//!   laid out per [`super::step::DofLayout`] (use its `neutral_q`), and
//!   `∂g/∂q` is an `nq`-vector while `∂g/∂v` is an `nv`-vector.
//! - **Open-loop control**: `ctrl(t)` must not read the state. A
//!   state-feedback law would add `∂u/∂x` terms this driver does not model.
//! - Objective reads the **final state** only.
//! - Contact: the per-vertex ground-penalty model of [`super::step`]
//!   (see its module docs for the smoothness contract). The gradient is the
//!   exact derivative of *that* forward model.
//! - Determinism: pure `f64` arithmetic in a fixed order; two runs are
//!   bit-identical.

use super::step::{
    CollisionMesh, GroundContact, N_INERTIA_PARAMS, aba_generic, config_update_generic, fk_generic,
    inertia_from_params, joint_nq, lift_inertia, lift_vec3, nominal_motion, rollout_states,
    step_generic, validate_and_params, vertex_wrench,
};
use phyz_math::{DVec, Vec3};
use phyz_model::Model;
use tang::{Dual, Mat3 as GMat3, SpatialInertia, SpatialMat, SpatialVec, Vec3 as GVec3};

type D = Dual<f64>;

/// `∂I_b/∂π_k` as a 6×6 spatial-inertia matrix — the inertia parameterisation
/// is a fixed algebraic map, so these blocks are constants of the model and
/// are priced once, outside the time loop.
fn inertia_param_derivatives(
    params: &[f64; N_INERTIA_PARAMS],
) -> [SpatialMat<f64>; N_INERTIA_PARAMS] {
    let dual_block = |m: &GMat3<D>| {
        GMat3::new(
            m.get(0, 0).dual,
            m.get(0, 1).dual,
            m.get(0, 2).dual,
            m.get(1, 0).dual,
            m.get(1, 1).dual,
            m.get(1, 2).dual,
            m.get(2, 0).dual,
            m.get(2, 1).dual,
            m.get(2, 2).dual,
        )
    };
    core::array::from_fn(|k| {
        let mut p: [D; N_INERTIA_PARAMS] = core::array::from_fn(|i| D::constant(params[i]));
        p[k] = D::var(params[k]);
        let m = inertia_from_params(&p).to_matrix();
        SpatialMat::new(
            dual_block(&m.upper_left),
            dual_block(&m.upper_right),
            dual_block(&m.lower_left),
            dual_block(&m.lower_right),
        )
    })
}

/// Which block of state coordinates a lane sweep seeds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Seed {
    /// Seed `q`, producing `(∂a/∂q_i)ᵀw` for `i` in `0..nq`.
    Position,
    /// Seed `v`, producing `(∂a/∂v_j)ᵀw` for `j` in `0..nv`.
    Velocity,
}

/// A run of consecutive coordinates, evaluated at lane width `N`.
///
/// Everything except the seeded block is a constant, so a single pass through
/// [`step_generic`] computes the primal once and pushes `N` tangent directions
/// through it. That is the whole point: the kinematic tree walk, the joint
/// transcendentals and the memory traffic are paid once per chunk instead of
/// once per column.
///
/// A group owns *all* the chunks that share a width, so the dual-lifted
/// inertias — `O(nb)` of them, and pure overhead against the lane work on a
/// model with many bodies and few DOFs — are built once per group rather than
/// once per chunk.
struct LaneGroup<'a> {
    model: &'a Model,
    layout: &'a super::step::DofLayout,
    contact: Option<(&'a GroundContact, &'a [CollisionMesh])>,
    q: &'a [f64],
    v: &'a [f64],
    ctrl: &'a [f64],
    /// The covector every lane's `∂qdd` is contracted against.
    w: &'a [f64],
    seed: Seed,
    /// `(start, count)` per chunk. A tail chunk may not fill the width;
    /// unseeded lanes stay exactly zero, so they cost arithmetic and nothing
    /// else.
    chunks: &'a [(usize, usize)],
}

impl crate::multidual::LaneOp for LaneGroup<'_> {
    type Out = Vec<f64>;

    fn call<const N: usize>(self) -> Vec<f64> {
        type M<const N: usize> = crate::multidual::MultiDual<N>;

        let lift = |xs: &[f64]| -> Vec<M<N>> { xs.iter().map(|&x| M::<N>::constant(x)).collect() };
        let q_c = lift(self.q);
        let v_c = lift(self.v);
        let ctrl_d = lift(self.ctrl);
        let inertias: Vec<SpatialInertia<M<N>>> = self
            .model
            .bodies
            .iter()
            .map(|b| lift_inertia(&b.inertia))
            .collect();

        let mut out = Vec::new();
        for &(start, count) in self.chunks {
            let (mut q_d, mut v_d) = (q_c.clone(), v_c.clone());
            for lane in 0..count {
                let idx = start + lane;
                match self.seed {
                    Seed::Position => q_d[idx] = M::<N>::var(self.q[idx], lane),
                    Seed::Velocity => v_d[idx] = M::<N>::var(self.v[idx], lane),
                }
            }

            let (_, _, qdd) = step_generic(
                self.model,
                self.layout,
                &inertias,
                self.contact,
                None,
                &q_d,
                &v_d,
                &ctrl_d,
            );

            // wᵀ·∂qdd/∂(coordinate) for each lane. Summed in coordinate order,
            // the same order the scalar path used, so the result is
            // bit-identical.
            out.extend((0..count).map(|lane| -> f64 {
                qdd.iter()
                    .zip(self.w)
                    .map(|(a, &wi)| a.dual[lane] * wi)
                    .sum()
            }));
        }
        out
    }
}

/// How `n` coordinates are cut into chunks: `(width, start, count)`, widest
/// first, each chunk as wide as the remainder justifies and no wider.
///
/// Lanes past the coordinate count are wasted arithmetic, so the tail narrows
/// rather than padding a full-width chunk: 20 coordinates run as 16 + 4, not
/// 16 + 16.
fn chunk_plan(n: usize) -> Vec<(usize, usize, usize)> {
    let mut plan = Vec::new();
    let mut start = 0;
    while start < n {
        let remaining = n - start;
        let width = crate::multidual::width_for(remaining);
        let count = remaining.min(width);
        plan.push((width, start, count));
        start += count;
    }
    plan
}

/// `(∂a/∂x_k)ᵀw` for every coordinate `k` of the seeded block, in vector mode.
///
/// Chunks that share a width are evaluated together in one monomorphisation,
/// so the per-width setup (dual-lifting every body's inertia) is paid once for
/// the group instead of once per chunk.
///
/// A **single-coordinate block takes the scalar path instead**, using the
/// caller's `inertias_dual`, which is lifted once for the whole rollout. Vector
/// mode at width 1 does the same arithmetic but cannot reuse that lift across
/// timesteps, and on a model with many bodies and one DOF — a welded assembly
/// on one hinge — the lift is comparable to the lane work it accompanies,
/// costing ~10%. Since `MultiDual<1>` is bit-identical to `Dual` this is purely
/// a choice of which code is faster, never which answer comes out.
#[allow(clippy::too_many_arguments)]
fn state_lanes(
    model: &Model,
    layout: &super::step::DofLayout,
    contact: Option<(&GroundContact, &[CollisionMesh])>,
    inertias_dual: &[SpatialInertia<D>],
    q: &[f64],
    v: &[f64],
    ctrl: &[f64],
    w: &[f64],
    seed: Seed,
) -> Vec<f64> {
    let n = match seed {
        Seed::Position => layout.nq,
        Seed::Velocity => layout.nv,
    };

    if n == 1 {
        let lift = |xs: &[f64]| -> Vec<D> { xs.iter().map(|&x| D::constant(x)).collect() };
        let (mut q_d, mut v_d, u_d) = (lift(q), lift(v), lift(ctrl));
        match seed {
            Seed::Position => q_d[0] = D::var(q[0]),
            Seed::Velocity => v_d[0] = D::var(v[0]),
        }
        let (_, _, qdd) = step_generic(
            model,
            layout,
            inertias_dual,
            contact,
            None,
            &q_d,
            &v_d,
            &u_d,
        );
        return vec![qdd.iter().zip(w).map(|(a, &wi)| a.dual * wi).sum()];
    }

    let plan = chunk_plan(n);
    let mut out = Vec::with_capacity(n);

    // Widths appear in descending order and each at most once in practice, but
    // group explicitly rather than relying on that.
    let mut done = vec![false; plan.len()];
    for i in 0..plan.len() {
        if done[i] {
            continue;
        }
        let width = plan[i].0;
        let mut chunks: Vec<(usize, usize)> = Vec::new();
        for (j, c) in plan.iter().enumerate() {
            if c.0 == width && !done[j] {
                done[j] = true;
                chunks.push((c.1, c.2));
            }
        }
        out.extend(crate::multidual::for_lanes(
            width,
            LaneGroup {
                model,
                layout,
                contact,
                q,
                v,
                ctrl,
                w,
                seed,
                chunks: &chunks,
            },
        ));
    }
    out
}

/// Ground contact configuration for an adjoint rollout: the plane and the
/// collision skins that feel it.
pub struct ContactSetup<'a> {
    /// The ground plane and penalty law.
    pub ground: GroundContact,
    /// Collision skins (body-frame vertices), one entry per skinned body.
    pub meshes: &'a [CollisionMesh],
}

/// A differentiable rollout: model, optional contact, initial state, and an
/// open-loop control schedule.
pub struct AdjointRollout<'a> {
    /// Topology, gravity, timestep, joint damping, and **nominal** body
    /// inertias (the π the gradient is taken at).
    pub model: &'a Model,
    /// Optional ground contact.
    pub contact: Option<ContactSetup<'a>>,
    /// Initial joint positions, in [`super::step::DofLayout`] packing (length
    /// `DofLayout::of(model).nq`, which exceeds `Model::nq` when the model has
    /// spherical or free joints).
    pub q0: Vec<f64>,
    /// Initial joint velocities (length `nv`).
    pub v0: Vec<f64>,
    /// Number of semi-implicit Euler steps.
    pub steps: usize,
    /// Open-loop control at step `t` (length `nv`; must not read the state).
    pub ctrl: &'a dyn Fn(usize) -> DVec,
}

/// The analytic gradient of a final-state objective: `(∂g/∂q_T, ∂g/∂v_T)`.
pub type ObjectiveGradientFn<'a> = &'a dyn Fn(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>);

/// A final-state objective `J = g(q_T, v_T)` with its analytic gradient.
pub struct FinalStateObjective<'a> {
    /// `g(q_T, v_T)`.
    pub value: &'a dyn Fn(&[f64], &[f64]) -> f64,
    /// `(∂g/∂q_T, ∂g/∂v_T)`.
    pub gradient: ObjectiveGradientFn<'a>,
}

/// Everything one backward pass produces.
pub struct AdjointGradients {
    /// The objective at the nominal rollout.
    pub objective: f64,
    /// `dJ/dπ` per body, canonical packing `[m, cx, cy, cz, Ixx, Iyy, Izz,
    /// Ixy, Ixz, Iyz]` (COM-frame inertia, body-frame COM).
    pub d_inertia: Vec<[f64; N_INERTIA_PARAMS]>,
    /// `∂J/∂x` per collision-mesh vertex (body frame), parallel to
    /// `contact.meshes`; empty when the rollout has no contact.
    pub d_vertices: Vec<Vec<Vec3>>,
}

/// Run the nominal rollout and return the objective value only (the FD
/// oracle for gates, and a cheap primal probe for callers).
pub fn rollout_objective(rollout: &AdjointRollout, objective: &FinalStateObjective) -> f64 {
    let (layout, _) = validate_and_params(rollout.model);
    let contact = rollout.contact.as_ref().map(|c| (&c.ground, c.meshes));
    let (states, _) = rollout_states(
        rollout.model,
        &layout,
        contact,
        &rollout.q0,
        &rollout.v0,
        rollout.ctrl,
        rollout.steps,
    );
    let (q_t, v_t) = states.last().expect("rollout produced no states");
    (objective.value)(q_t, v_t)
}

/// One nominal rollout forward, one adjoint pass backward: `J`, `dJ/dπ` per
/// body, and (when contact is configured) `∂J/∂x` per collision-mesh vertex.
pub fn adjoint_rollout_gradient(
    rollout: &AdjointRollout,
    objective: &FinalStateObjective,
) -> AdjointGradients {
    let model = rollout.model;
    let (layout, params) = validate_and_params(model);
    let nb = model.nbodies();
    let nq = layout.nq;
    let nv = layout.nv;
    let dt = model.dt;
    let contact = rollout.contact.as_ref().map(|c| (&c.ground, c.meshes));
    assert_eq!(rollout.q0.len(), nq, "q0 length (DofLayout::nq)");
    assert_eq!(rollout.v0.len(), nv, "v0 length (nv)");

    // Forward: store the whole trajectory and its accelerations.
    let (states, accels) = rollout_states(
        model,
        &layout,
        contact,
        &rollout.q0,
        &rollout.v0,
        rollout.ctrl,
        rollout.steps,
    );
    let (q_final, v_final) = states.last().expect("rollout produced no states");
    let j0 = (objective.value)(q_final, v_final);
    let (mut lam_q, mut lam_v) = (objective.gradient)(q_final, v_final);
    assert_eq!(lam_q.len(), nq, "objective ∂g/∂q length");
    assert_eq!(lam_v.len(), nv, "objective ∂g/∂v length");

    let mut d_inertia = vec![[0.0f64; N_INERTIA_PARAMS]; nb];
    let mut d_vertices: Vec<Vec<Vec3>> = rollout
        .contact
        .as_ref()
        .map(|c| {
            c.meshes
                .iter()
                .map(|m| vec![Vec3::zeros(); m.vertices.len()])
                .collect()
        })
        .unwrap_or_default();

    let inertias_f64: Vec<SpatialInertia<f64>> = model.bodies.iter().map(|b| b.inertia).collect();
    // Dual-lifted inertias for the single-coordinate lane path, hoisted out of
    // the time loop the way the all-scalar implementation had them.
    let inertias_dual: Vec<SpatialInertia<D>> = model
        .bodies
        .iter()
        .map(|b| lift_inertia(&b.inertia))
        .collect();

    // Constants of the model, hoisted out of the time loop.
    let d_i_mats: Vec<[SpatialMat<f64>; N_INERTIA_PARAMS]> =
        params.iter().map(inertia_param_derivatives).collect();
    // `y = M⁻¹w` is read out of ABA with zero velocity and zero gravity, so
    // the bias and gravity terms vanish and `qdd = M⁻¹·ctrl`.
    let model_free = {
        let mut m = model.clone();
        m.gravity = Vec3::zeros();
        m
    };
    let zeros_v = vec![0.0f64; nv];
    // With no multi-DOF joint, `Φ(q, v') = q + dt·v'` exactly, so `Φ_q = I` and
    // `Φ_v' = dt·I` and the two Φ lane sweeps below are a known answer. Skip
    // them: this is the common case and it keeps the single-DOF cost at the
    // `2·nv` state lanes.
    let euclidean_q = model
        .joints
        .iter()
        .all(|j| joint_nq(j.joint_type) == j.ndof());

    // Backward over steps.
    for t in (0..rollout.steps).rev() {
        let (q_t, v_t) = &states[t];
        let v_next = &states[t + 1].1;
        let u_t = (rollout.ctrl)(t);
        let u_t = u_t.as_slice();

        let lift = |xs: &[f64]| -> Vec<D> { xs.iter().map(|&x| D::constant(x)).collect() };
        let q_c = lift(q_t);
        let vn_c = lift(v_next);
        let dt_d = D::constant(dt);

        // Configuration-update lanes: Φ contracted against λ_q' along each of
        // its two argument blocks. `phi_q = Φ_qᵀ λ_q'` (nq), `psi = Φ_v'ᵀ λ_q'`
        // (nv). Both are dual lanes through Φ alone — no ABA.
        let phi_dot = |q_d: &[D], vn_d: &[D]| -> f64 {
            let qn = config_update_generic(model, &layout, q_d, vn_d, dt_d);
            qn.iter().zip(&lam_q).map(|(x, &l)| x.dual * l).sum()
        };
        let (phi_q, psi) = if euclidean_q {
            (lam_q.clone(), lam_q.iter().map(|&l| dt * l).collect())
        } else {
            let mut phi_q = vec![0.0f64; nq];
            for i in 0..nq {
                let mut q_d = q_c.clone();
                q_d[i] = D::var(q_t[i]);
                phi_q[i] = phi_dot(&q_d, &vn_c);
            }
            let mut psi = vec![0.0f64; nv];
            for j in 0..nv {
                let mut vn_d = vn_c.clone();
                vn_d[j] = D::var(v_next[j]);
                psi[j] = phi_dot(&q_c, &vn_d);
            }
            (phi_q, psi)
        };

        // w = Φ_v'ᵀλ_q' + λ_v' — the covector every channel contracts against.
        let w: Vec<f64> = psi.iter().zip(&lam_v).map(|(&p, &lv)| p + lv).collect();

        // State lanes: aq[i] = (∂a/∂q_i)ᵀw (nq of them), av[j] = (∂a/∂v_j)ᵀw.
        //
        // These are the backward pass's dominant cost — one seeded evaluation
        // of the step per state coordinate — so they run in vector mode: each
        // pass carries up to 16 tangent directions through a single primal.
        // Lane-for-lane the arithmetic is bit-identical to the scalar `Dual`
        // path (see [`crate::multidual`]), so widening changes the time and
        // not the answer.
        let aq = state_lanes(
            model,
            &layout,
            contact,
            &inertias_dual,
            q_t,
            v_t,
            u_t,
            &w,
            Seed::Position,
        );
        let av = state_lanes(
            model,
            &layout,
            contact,
            &inertias_dual,
            q_t,
            v_t,
            u_t,
            &w,
            Seed::Velocity,
        );

        // Inertia and wrench channels, analytically (see module docs). Two
        // O(nb) sweeps price *all* 10·nb parameter directions and all 6·nb
        // wrench directions at once — no dual lane per parameter.
        //
        //   y   = M⁻¹w                     (one ABA)
        //   A^y = J y                      (one velocity sweep)
        //   dJ/dπ[b][k] -= dt·⟨A^y_b, ΔI·A_b + V_b ×* (ΔI·V_b)⟩
        //   χ_b          = dt·A^y_b
        let (vel_b, acc_b) = nominal_motion(model, &layout, q_t, v_t, &accels[t]);
        let y = aba_generic::<f64>(&model_free, &layout, &inertias_f64, q_t, &zeros_v, &w, None);
        let (_, a_y) = fk_generic::<f64>(model, &layout, q_t, &y);

        for b in 0..nb {
            for k in 0..N_INERTIA_PARAMS {
                let di = &d_i_mats[b][k];
                let f = di.mul_vec(&acc_b[b]) + vel_b[b].cross_force(&di.mul_vec(&vel_b[b]));
                d_inertia[b][k] -= dt * a_y[b].dot(&f);
            }
        }

        // Vertex channel: price the wrench cotangent χ once per contacting
        // body, then contract each vertex's local wrench Jacobian against it.
        if let Some(setup) = rollout.contact.as_ref() {
            // χ[b][c] = dt·wᵀ·∂a/∂(wrench_b component c). A body-frame wrench
            // on body b enters the dynamics as Jᵀ_b, so wᵀM⁻¹Jᵀ_b = (J_b y)ᵀ
            // — the same A^y field the inertia channel already computed.
            let chi: Vec<[f64; 6]> = (0..nb)
                .map(|b| {
                    let m = a_y[b].as_array();
                    core::array::from_fn(|c| dt * m[c])
                })
                .collect();

            // Local per-vertex wrench Jacobian at the nominal kinematics:
            // 3 dual evaluations of the penalty law, holding the state fixed.
            let (xforms, vels) = fk_generic::<f64>(model, &layout, q_t, v_t);
            for (mi, mesh) in setup.meshes.iter().enumerate() {
                let b = mesh.body;
                let xf_d = tang::SpatialTransform::<D>::new(
                    tang::Mat3::from_cols(
                        lift_vec3(xforms[b].rot.col(0)),
                        lift_vec3(xforms[b].rot.col(1)),
                        lift_vec3(xforms[b].rot.col(2)),
                    ),
                    lift_vec3(xforms[b].pos),
                );
                let twist_d =
                    SpatialVec::new(lift_vec3(vels[b].angular), lift_vec3(vels[b].linear));
                for (vi, vtx) in mesh.vertices.iter().enumerate() {
                    for k in 0..3 {
                        let mut x_b: GVec3<D> = lift_vec3(*vtx);
                        match k {
                            0 => x_b.x = D::var(vtx.x),
                            1 => x_b.y = D::var(vtx.y),
                            _ => x_b.z = D::var(vtx.z),
                        }
                        let wr = vertex_wrench(&setup.ground, x_b, &xf_d, &twist_d);
                        let wr = wr.as_array();
                        let mut acc = 0.0;
                        for (c, w_c) in wr.iter().enumerate() {
                            acc += chi[b][c] * w_c.dual;
                        }
                        match k {
                            0 => d_vertices[mi][vi].x += acc,
                            1 => d_vertices[mi][vi].y += acc,
                            _ => d_vertices[mi][vi].z += acc,
                        }
                    }
                }
            }
        }

        // λ update (see module docs for the derivation).
        for i in 0..nq {
            lam_q[i] = phi_q[i] + dt * aq[i];
        }
        for j in 0..nv {
            lam_v[j] = w[j] + dt * av[j];
        }
    }

    AdjointGradients {
        objective: j0,
        d_inertia,
        d_vertices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{Mat3, SpatialInertia as SI64, SpatialTransform};
    use phyz_model::ModelBuilder;

    /// The scalar-`Dual` lane extraction this module used before vector mode:
    /// one seeded pass through the step per coordinate. Kept as the reference
    /// the fast path is checked against.
    fn state_lanes_scalar(
        model: &Model,
        layout: &super::super::step::DofLayout,
        q: &[f64],
        v: &[f64],
        ctrl: &[f64],
        w: &[f64],
        seed: Seed,
    ) -> Vec<f64> {
        let lift = |xs: &[f64]| -> Vec<D> { xs.iter().map(|&x| D::constant(x)).collect() };
        let (q_c, v_c, u_c) = (lift(q), lift(v), lift(ctrl));
        let inertias: Vec<SpatialInertia<D>> = model
            .bodies
            .iter()
            .map(|b| lift_inertia(&b.inertia))
            .collect();

        let n = match seed {
            Seed::Position => layout.nq,
            Seed::Velocity => layout.nv,
        };
        (0..n)
            .map(|k| {
                let (mut q_d, mut v_d) = (q_c.clone(), v_c.clone());
                match seed {
                    Seed::Position => q_d[k] = D::var(q[k]),
                    Seed::Velocity => v_d[k] = D::var(v[k]),
                }
                let (_, _, qdd) =
                    step_generic(model, layout, &inertias, None, None, &q_d, &v_d, &u_c);
                qdd.iter().zip(w).map(|(a, &wi)| a.dual * wi).sum()
            })
            .collect()
    }

    fn link(m: f64) -> SI64 {
        SI64::new(
            m,
            Vec3::new(0.0, -0.5, 0.0),
            Mat3::from_diagonal(&Vec3::new(0.1, 0.13, 0.11)),
        )
    }

    fn chain(n: usize) -> Model {
        let off = SpatialTransform::new(Mat3::identity(), Vec3::new(0.0, -1.0, 0.0));
        let mut b = ModelBuilder::new()
            .gravity(Vec3::new(0.0, -9.81, 0.0))
            .dt(2.0e-3);
        for i in 0..n {
            let xf = if i == 0 {
                SpatialTransform::identity()
            } else {
                off
            };
            b = b.add_revolute_body(
                &format!("l{i}"),
                i as i32 - 1,
                xf,
                link(1.0 + 0.1 * i as f64),
            );
        }
        b.build()
    }

    /// Vector mode must agree with the scalar path **bit for bit**, at every
    /// DOF count — including the ones where the chunk does not fill its width
    /// (3, 5, 17) and the ones that need more than one chunk (17, 20).
    ///
    /// Approximate agreement would not do: `phyz`'s rollouts are bitwise
    /// reproducible, and a gradient that shifted in the last ulp because the
    /// model gained a joint would break that promise silently.
    #[test]
    fn vector_mode_matches_scalar_dual_bitwise() {
        for n in [1usize, 2, 3, 5, 8, 17, 20] {
            let model = chain(n);
            let layout = super::super::step::DofLayout::of(&model);
            let q: Vec<f64> = (0..layout.nq).map(|i| 0.21 + 0.07 * i as f64).collect();
            let v: Vec<f64> = (0..layout.nv).map(|i| -0.13 + 0.05 * i as f64).collect();
            let ctrl: Vec<f64> = (0..layout.nv).map(|i| 0.02 * i as f64).collect();
            let w: Vec<f64> = (0..layout.nv).map(|i| 1.0 - 0.03 * i as f64).collect();

            for seed in [Seed::Position, Seed::Velocity] {
                let inert: Vec<SpatialInertia<D>> = model
                    .bodies
                    .iter()
                    .map(|b| lift_inertia(&b.inertia))
                    .collect();
                let fast = state_lanes(&model, &layout, None, &inert, &q, &v, &ctrl, &w, seed);
                let slow = state_lanes_scalar(&model, &layout, &q, &v, &ctrl, &w, seed);
                assert_eq!(fast.len(), slow.len(), "n = {n}, {seed:?}");
                for (k, (a, b)) in fast.iter().zip(&slow).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "n = {n}, {seed:?}, lane {k}: {a} vs {b}"
                    );
                }
            }
        }
    }

    /// The same, with contact wrenches in the loop — a separate code path
    /// inside the step, and one where the penalty law's `max` could in
    /// principle take a different branch per lane if the comparison ever
    /// consulted a tangent.
    #[test]
    fn vector_mode_matches_scalar_dual_with_contact() {
        let model = chain(3);
        let layout = super::super::step::DofLayout::of(&model);
        let meshes = [CollisionMesh {
            body: 2,
            vertices: vec![
                Vec3::new(0.02, -1.0, 0.0),
                Vec3::new(-0.02, -1.0, 0.01),
                Vec3::new(0.0, -0.98, -0.02),
            ],
        }];
        let ground = GroundContact {
            height: -1.5,
            stiffness: 4.0e3,
            damping: 40.0,
        };
        let contact = Some((&ground, &meshes[..]));

        let q: Vec<f64> = (0..layout.nq).map(|i| 0.4 + 0.05 * i as f64).collect();
        let v = vec![0.1, -0.2, 0.05];
        let ctrl = vec![0.0; layout.nv];
        let w = vec![0.9, 1.1, -0.7];

        for seed in [Seed::Position, Seed::Velocity] {
            let inert: Vec<SpatialInertia<D>> = model
                .bodies
                .iter()
                .map(|b| lift_inertia(&b.inertia))
                .collect();
            let fast = state_lanes(&model, &layout, contact, &inert, &q, &v, &ctrl, &w, seed);
            // The scalar reference takes no contact, so run it against a
            // hand-rolled seeded pass that does.
            let lift = |xs: &[f64]| -> Vec<D> { xs.iter().map(|&x| D::constant(x)).collect() };
            let (q_c, v_c, u_c) = (lift(&q), lift(&v), lift(&ctrl));
            let inertias: Vec<SpatialInertia<D>> = model
                .bodies
                .iter()
                .map(|b| lift_inertia(&b.inertia))
                .collect();
            let n = match seed {
                Seed::Position => layout.nq,
                Seed::Velocity => layout.nv,
            };
            for k in 0..n {
                let (mut q_d, mut v_d) = (q_c.clone(), v_c.clone());
                match seed {
                    Seed::Position => q_d[k] = D::var(q[k]),
                    Seed::Velocity => v_d[k] = D::var(v[k]),
                }
                let (_, _, qdd) =
                    step_generic(&model, &layout, &inertias, contact, None, &q_d, &v_d, &u_c);
                let slow: f64 = qdd.iter().zip(&w).map(|(a, &wi)| a.dual * wi).sum();
                assert_eq!(
                    fast[k].to_bits(),
                    slow.to_bits(),
                    "{seed:?} lane {k}: {} vs {slow}",
                    fast[k]
                );
            }
        }
    }

    /// A non-empty check on the reference itself: a pendulum's `∂a/∂q` is not
    /// zero, so the bitwise comparisons above are comparing real numbers
    /// rather than two identically empty answers.
    #[test]
    fn lanes_are_not_trivially_zero() {
        let model = chain(2);
        let layout = super::super::step::DofLayout::of(&model);
        let q = vec![0.3, 0.5];
        let v = vec![0.2, -0.1];
        let ctrl = vec![0.0, 0.0];
        let w = vec![1.0, 1.0];
        let inert: Vec<SpatialInertia<D>> = model
            .bodies
            .iter()
            .map(|b| lift_inertia(&b.inertia))
            .collect();
        let aq = state_lanes(
            &model,
            &layout,
            None,
            &inert,
            &q,
            &v,
            &ctrl,
            &w,
            Seed::Position,
        );
        assert!(
            aq.iter().any(|x| x.abs() > 1e-9),
            "every position lane vanished: {aq:?}"
        );
    }
}
