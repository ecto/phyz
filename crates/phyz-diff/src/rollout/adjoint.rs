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
//! Semi-implicit Euler has the structure `v' = v + dt·a`, `q' = q + dt·v'`
//! with `a = ABA(q, v, u; π, V)`, so with `w := dt·λ_q' + λ_v'`:
//!
//! ```text
//! λ_q = λ_q' + dt·(∂a/∂q)ᵀ w          λ_v = w + dt·(∂a/∂v)ᵀ w
//! dJ/dπ += dt·wᵀ·∂a/∂π               dJ/dV += dt·wᵀ·∂a/∂V
//! ```
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
//! - Joint domain: single-DOF (revolute/prismatic) + fixed, like the
//!   symbolic tracer. Multi-DOF joints panic.
//! - **Open-loop control**: `ctrl(t)` must not read the state. A
//!   state-feedback law would add `∂u/∂x` terms this driver does not model.
//! - Objective reads the **final state** only.
//! - Contact: the per-vertex ground-penalty model of [`super::step`]
//!   (see its module docs for the smoothness contract). The gradient is the
//!   exact derivative of *that* forward model.
//! - Determinism: pure `f64` arithmetic in a fixed order; two runs are
//!   bit-identical.

use super::step::{
    CollisionMesh, GroundContact, N_INERTIA_PARAMS, aba_generic, fk_generic, inertia_from_params,
    lift_inertia, lift_vec3, nominal_motion, rollout_states, step_generic, validate_and_params,
    vertex_wrench,
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
    /// Initial joint positions (length `nq`).
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
    validate_and_params(rollout.model);
    let contact = rollout.contact.as_ref().map(|c| (&c.ground, c.meshes));
    let (states, _) = rollout_states(
        rollout.model,
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
    let params = validate_and_params(model);
    let nb = model.nbodies();
    let n = model.nv;
    let dt = model.dt;
    let contact = rollout.contact.as_ref().map(|c| (&c.ground, c.meshes));

    // Forward: store the whole trajectory and its accelerations.
    let (states, accels) = rollout_states(
        model,
        contact,
        &rollout.q0,
        &rollout.v0,
        rollout.ctrl,
        rollout.steps,
    );
    let (q_final, v_final) = states.last().expect("rollout produced no states");
    let j0 = (objective.value)(q_final, v_final);
    let (mut lam_q, mut lam_v) = (objective.gradient)(q_final, v_final);
    assert_eq!(lam_q.len(), n, "objective ∂g/∂q length");
    assert_eq!(lam_v.len(), n, "objective ∂g/∂v length");

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

    // Nominal dual-lifted inertias, reused per state lane.
    let inertias_nominal: Vec<SpatialInertia<D>> = model
        .bodies
        .iter()
        .map(|b| lift_inertia(&b.inertia))
        .collect();
    let inertias_f64: Vec<SpatialInertia<f64>> = model.bodies.iter().map(|b| b.inertia).collect();

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
    let zeros_v = vec![0.0f64; n];

    // Backward over steps.
    for t in (0..rollout.steps).rev() {
        let (q_t, v_t) = &states[t];
        let u_t = (rollout.ctrl)(t);
        let u_t = u_t.as_slice();

        // w = dt·λ_q' + λ_v' — the covector every channel contracts against.
        let w: Vec<f64> = lam_q
            .iter()
            .zip(&lam_v)
            .map(|(&lq, &lv)| dt * lq + lv)
            .collect();

        let lift = |xs: &[f64]| -> Vec<D> { xs.iter().map(|&x| D::constant(x)).collect() };
        let (q_c, v_c, u_c) = (lift(q_t), lift(v_t), lift(u_t));

        // One dual lane: returns wᵀ·∂qdd/∂(seeded input).
        let contract = |q_d: &[D],
                        v_d: &[D],
                        inertias: &[SpatialInertia<D>],
                        ext: Option<&[SpatialVec<D>]>|
         -> f64 {
            let (_, _, qdd) = step_generic(model, inertias, contact, ext, q_d, v_d, &u_c);
            qdd.iter().zip(&w).map(|(a, &wi)| a.dual * wi).sum()
        };

        // State lanes: aq[j] = (∂a/∂q)ᵀw, av[j] = (∂a/∂v)ᵀw.
        let mut aq = vec![0.0f64; n];
        let mut av = vec![0.0f64; n];
        for j in 0..n {
            let mut q_d = q_c.clone();
            q_d[j] = D::var(q_t[j]);
            aq[j] = contract(&q_d, &v_c, &inertias_nominal, None);

            let mut v_d = v_c.clone();
            v_d[j] = D::var(v_t[j]);
            av[j] = contract(&q_c, &v_d, &inertias_nominal, None);
        }

        // Inertia and wrench channels, analytically (see module docs). Two
        // O(nb) sweeps price *all* 10·nb parameter directions and all 6·nb
        // wrench directions at once — no dual lane per parameter.
        //
        //   y   = M⁻¹w                     (one ABA)
        //   A^y = J y                      (one velocity sweep)
        //   dJ/dπ[b][k] -= dt·⟨A^y_b, ΔI·A_b + V_b ×* (ΔI·V_b)⟩
        //   χ_b          = dt·A^y_b
        let (vel_b, acc_b) = nominal_motion(model, q_t, v_t, &accels[t]);
        let y = aba_generic::<f64>(&model_free, &inertias_f64, q_t, &zeros_v, &w, None);
        let (_, a_y) = fk_generic::<f64>(model, q_t, &y);

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
            let (xforms, vels) = fk_generic::<f64>(model, q_t, v_t);
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
        for j in 0..n {
            lam_q[j] += dt * aq[j];
            lam_v[j] = w[j] + dt * av[j];
        }
    }

    AdjointGradients {
        objective: j0,
        d_inertia,
        d_vertices,
    }
}
